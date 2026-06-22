from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from finetuneharness.evaluation.comparator import (
    filter_runs_since,
    find_latest_run_pair,
    parse_since_duration,
    compare_runs,
)
from finetuneharness.evaluation.report import format_report, report_to_dict
from finetuneharness.executor.resources import ConcurrencyConfig, NvmlMonitor
from finetuneharness.executor.worker import DegradedRunError, LocalWorker
from finetuneharness.orchestrator.approval import (
    ApprovalError,
    InteractiveApprovalGate,
    resolve_actor,
)
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import EventRecord
from finetuneharness.state.reproducibility import assess_reproducibility, export_manifest
from finetuneharness.state.sqlite import SQLiteStateStore


def _load_handler(spec: str):
    """Import a task handler from a 'module:function' spec, e.g. 'mypkg.h:train'."""
    import importlib

    def _fail(msg: str):
        print(f"Error: {msg}", flush=True)
        raise SystemExit(1)

    if ":" not in spec:
        _fail(f"--handler must be 'module:function', got {spec!r}")
    module_name, _, func_name = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        _fail(f"could not import handler module {module_name!r}: {exc}")
    fn = getattr(module, func_name, None)
    if fn is None or not callable(fn):
        _fail(f"handler {func_name!r} not found or not callable in {module_name!r}")
    return fn


def _is_approved(store, run_id: str) -> bool:
    """True if the run has a recorded approval (a 'run_approved' event)."""
    return any(ev.kind == "run_approved" for ev in store.list_events(run_id))


def _resolve_concurrency(config: dict, args) -> ConcurrencyConfig:
    """Build a ConcurrencyConfig from the stored run config, with CLI overrides.

    Precedence: CLI flag > executor.concurrency in the run config > dataclass default.
    """
    conc = {}
    executor = config.get("executor")
    if isinstance(executor, dict) and isinstance(executor.get("concurrency"), dict):
        conc = dict(executor["concurrency"])
    if args.concurrency_mode is not None:
        conc["mode"] = args.concurrency_mode
    if args.min_free_mb is not None:
        conc["min_free_mb"] = args.min_free_mb
    if args.max_concurrent is not None:
        conc["max_concurrent"] = args.max_concurrent
    defaults = ConcurrencyConfig()
    return ConcurrencyConfig(
        mode=conc.get("mode", defaults.mode),
        admission=conc.get("admission", defaults.admission),
        min_free_mb=conc.get("min_free_mb", defaults.min_free_mb),
        max_util_pct=conc.get("max_util_pct", defaults.max_util_pct),
        max_concurrent=conc.get("max_concurrent", defaults.max_concurrent),
        settle_seconds=conc.get("settle_seconds", defaults.settle_seconds),
        max_oom_retries=conc.get("max_oom_retries", defaults.max_oom_retries),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FineTuneHarness CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    init_run = sub.add_parser("create-run", help="Create a bootstrap run from JSON config")
    init_run.add_argument("--name", required=True)
    init_run.add_argument("--config", required=True, help="Path to JSON config file")
    init_run.add_argument("--tasks", required=True, help="Path to JSON task list file")
    init_run.add_argument("--state-db", default=".finetuneharness/state.db", help="SQLite state DB path")
    init_run.add_argument(
        "--memory", action="store_true",
        help="SMOKE-TEST ONLY: use a process-local in-memory store. The run is NOT "
             "persisted, so the printed run_id cannot be used by any later command.",
    )

    run_cmd = sub.add_parser("run", help="Execute pending tasks for a run with a handler")
    run_cmd.add_argument("--run-id", required=True)
    run_cmd.add_argument("--state-db", default=".finetuneharness/state.db")
    run_cmd.add_argument(
        "--handler", required=True, metavar="MODULE:FUNCTION",
        help="Import path of the task handler, e.g. 'mypkg.handlers:train'",
    )
    run_cmd.add_argument(
        "--concurrency-mode", choices=["sequential", "resource_aware"], default=None,
        help="Override executor.concurrency.mode from the run config. "
             "'resource_aware' admits multiple tasks while the GPU has free memory; "
             "'sequential' runs one at a time.",
    )
    run_cmd.add_argument(
        "--min-free-mb", type=float, default=None,
        help="Resource-aware mode: do not admit another task while free GPU memory "
             "is below this many MB (headroom).",
    )
    run_cmd.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Resource-aware mode: hard ceiling on tasks in flight at once.",
    )
    run_cmd.add_argument(
        "--skip-approval", action="store_true",
        help="Bypass the approval gate. Without this, 'run' refuses a run that was "
             "not approved via 'start-run'.",
    )

    status_cmd = sub.add_parser("status", help="Show run status summary")
    status_cmd.add_argument("--run-id", required=True)
    status_cmd.add_argument("--state-db", default=".finetuneharness/state.db")

    list_tasks_cmd = sub.add_parser("list-tasks", help="List tasks for a run")
    list_tasks_cmd.add_argument("--run-id", required=True)
    list_tasks_cmd.add_argument("--state-db", default=".finetuneharness/state.db")

    list_artifacts_cmd = sub.add_parser("list-artifacts", help="List artifacts for a run")
    list_artifacts_cmd.add_argument("--run-id", required=True)
    list_artifacts_cmd.add_argument("--state-db", default=".finetuneharness/state.db")

    start_cmd = sub.add_parser("start-run", help="Interactively approve a validated run before workers start")
    start_cmd.add_argument("--run-id", required=True)
    start_cmd.add_argument("--state-db", default=".finetuneharness/state.db")
    start_cmd.add_argument(
        "--approver", default=None,
        help="Identity recorded as the run's approver (defaults to the OS user). "
             "Verified against config 'approval.allowed_actors' when present.",
    )

    recover_cmd = sub.add_parser(
        "recover-run",
        help="Requeue tasks stranded RUNNING/LEASED by a hard crash (run ONLY when no worker is active)",
    )
    recover_cmd.add_argument("--run-id", required=True)
    recover_cmd.add_argument("--state-db", default=".finetuneharness/state.db")

    preflight_cmd = sub.add_parser(
        "preflight",
        help="Check whether it is safe to edit runtime code now (no run has in-flight "
             "tasks). Exits 1 if a run is active. Run before editing the harness.",
    )
    preflight_cmd.add_argument("--state-db", default=".finetuneharness/state.db")
    preflight_cmd.add_argument("--format", dest="fmt", choices=["text", "json"], default="text")

    verify_cmd = sub.add_parser(
        "verify-run",
        help="Independently re-check a run's SUCCEEDED results (catches fabricated/"
             "inconsistent results and success states with no execution). Exits 1 on FAIL.",
    )
    verify_cmd.add_argument("--run-id", required=True)
    verify_cmd.add_argument("--state-db", default=".finetuneharness/state.db")
    verify_cmd.add_argument("--format", dest="fmt", choices=["text", "json"], default="text")

    list_runs_cmd = sub.add_parser("list-runs", help="List all runs in the state DB")
    list_runs_cmd.add_argument("--state-db", default=".finetuneharness/state.db")
    list_runs_cmd.add_argument("--format", dest="fmt", choices=["text", "json"], default="text")
    list_runs_cmd.add_argument(
        "--since",
        default=None,
        metavar="DURATION",
        help="Filter runs created within this window, e.g. '7d', '30d', '2h', '1w'",
    )

    compare_cmd = sub.add_parser("compare-runs", help="Compare two or more runs (first is baseline)")
    compare_run_group = compare_cmd.add_mutually_exclusive_group()
    compare_run_group.add_argument(
        "--run-id", dest="run_ids", action="append", metavar="RUN_ID",
        help="Run IDs to compare; first is baseline. Repeat for each run.",
    )
    compare_run_group.add_argument(
        "--latest-previous", action="store_true", default=False,
        help="Auto-select the two most recent runs (previous=baseline, latest=compare)",
    )
    compare_cmd.add_argument("--format", dest="fmt", choices=["text", "json"], default="text")
    compare_cmd.add_argument("--state-db", default=".finetuneharness/state.db")
    compare_cmd.add_argument(
        "--strict", action="store_true", default=False,
        help="Raise an error if runs have incompatible dataset_hash or other error-severity issues",
    )
    compare_cmd.add_argument(
        "--thresholds", default=None, metavar="JSON",
        help="JSON dict of per-metric regression thresholds, e.g. '{\"f1\": 0.03}'",
    )

    repro_cmd = sub.add_parser("validate-reproducibility", help="Assess reproducibility of a run")
    repro_cmd.add_argument("--run-id", required=True)
    repro_cmd.add_argument("--state-db", default=".finetuneharness/state.db")
    repro_cmd.add_argument("--format", dest="fmt", choices=["text", "json"], default="text")

    manifest_cmd = sub.add_parser("export-manifest", help="Export a full reproducibility manifest for a run")
    manifest_cmd.add_argument("--run-id", required=True)
    manifest_cmd.add_argument("--state-db", default=".finetuneharness/state.db")

    args = parser.parse_args()

    if args.command == "create-run":
        with open(args.config) as fh:
            config = json.load(fh)
        with open(args.tasks) as fh:
            tasks = json.load(fh)
        store = InMemoryStateStore() if args.memory else SQLiteStateStore(Path(args.state_db))
        runner = FineTuneRunner(store)
        run_id = runner.create_run(name=args.name, config=config, tasks=tasks)
        print(run_id)
        return

    if args.command == "run":
        store = SQLiteStateStore(Path(args.state_db))
        run = store.get_run(args.run_id)
        if run is None:
            print(f"Error: unknown run_id {args.run_id!r}", flush=True)
            raise SystemExit(1)

        if args.skip_approval:
            # Audit who bypassed the gate — AND say so loudly (a disabled guardrail
            # must never be a silent no-op).
            actor = resolve_actor()
            print(f"WARNING: approval gate BYPASSED by {actor!r} (--skip-approval). "
                  f"This run was not approved.", file=sys.stderr, flush=True)
            store.append_event(EventRecord(
                event_id=uuid.uuid4().hex, run_id=args.run_id, task_id=None,
                kind="approval_skipped", payload={"actor": actor},
            ))
        elif not _is_approved(store, args.run_id):
            print(
                f"Error: run {args.run_id} has not been approved. "
                f"Run 'finetuneharness start-run --run-id {args.run_id}' first, "
                f"or pass --skip-approval to bypass.",
                flush=True,
            )
            raise SystemExit(1)

        handler = _load_handler(args.handler)
        concurrency = _resolve_concurrency(run.config, args)
        # Surface self-sabotaging admission configs loudly before draining.
        from finetuneharness.verification import check_concurrency_config
        for f in check_concurrency_config(concurrency):
            print(f"WARNING: config guardrail [{f.code}]: {f.detail}", file=sys.stderr, flush=True)
        worker = LocalWorker(worker_id="cli", store=store)
        try:
            if concurrency.is_resource_aware:
                succeeded = worker.drain_concurrent(
                    run_id=args.run_id, handler=handler,
                    concurrency=concurrency, monitor=NvmlMonitor(),
                )
            else:
                succeeded = worker.drain(run_id=args.run_id, handler=handler)
            print(f"Run {args.run_id}: {succeeded} task(s) succeeded")
        except DegradedRunError as exc:
            print(str(exc), flush=True)
            raise SystemExit(1)
        return

    if args.command == "recover-run":
        store = SQLiteStateStore(Path(args.state_db))
        run = store.get_run(args.run_id)
        if run is None:
            print(f"Error: unknown run_id {args.run_id!r}", flush=True)
            raise SystemExit(1)
        n = store.recover_orphaned_tasks(run_id=args.run_id)
        # Recompute run status so a run stuck "running" with no worker reflects the
        # requeue (e.g. back to RUNNING/VALIDATED) instead of lying.
        status = FineTuneRunner(store).refresh_run_status(args.run_id)
        print(f"Recovered {n} stranded task(s) to PENDING in run {args.run_id} (status: {status.value}).")
        if n:
            print(f"Re-run with: finetuneharness run --run-id {args.run_id} --handler MODULE:FUNCTION")
        return

    if args.command == "preflight":
        from finetuneharness.verification import preflight
        store = SQLiteStateStore(Path(args.state_db))
        report = preflight(store)
        if args.fmt == "json":
            print(json.dumps({
                "safe_to_edit": report.safe_to_edit,
                "active_runs": [
                    {"run_id": a.run_id, "name": a.name, "in_flight": a.in_flight,
                     "lease_owners": a.lease_owners}
                    for a in report.active
                ],
            }, indent=2))
        else:
            if report.safe_to_edit:
                print("✓ preflight: SAFE to edit — no run has in-flight tasks.")
            else:
                print("✗ preflight: NOT safe to edit — runs are active (in-flight tasks):")
                for a in report.active:
                    print(f"  - {a.run_id[:12]} {a.name}: {a.in_flight} in-flight "
                          f"(owners: {a.lease_owners or '—'})")
                print("  Editing runtime modules now can corrupt a live run. "
                      "Wait, or `recover-run` if a run is crashed (stranded RUNNING).")
        if not report.safe_to_edit:
            raise SystemExit(1)
        return

    if args.command == "verify-run":
        from finetuneharness.verification import verify_run
        store = SQLiteStateStore(Path(args.state_db))
        run = store.get_run(args.run_id)
        if run is None:
            print(f"Error: unknown run_id {args.run_id!r}", flush=True)
            raise SystemExit(1)
        report = verify_run(store.list_tasks(args.run_id), store.list_events(args.run_id))
        if args.fmt == "json":
            print(json.dumps({
                "run_id": args.run_id,
                "verdict": report.verdict.value,
                "checked": report.checked,
                "findings": [
                    {"severity": f.severity, "code": f.code, "task_key": f.task_key, "detail": f.detail}
                    for f in report.findings
                ],
            }, indent=2))
        else:
            icon = {"PASS": "✓", "WARN": "~", "FAIL": "✗"}.get(report.verdict.value, "?")
            print(f"{icon} verify-run {report.verdict.value}  ({report.checked} SUCCEEDED tasks checked, "
                  f"{len(report.fails)} fail / {len(report.warns)} warn)")
            for f in report.findings:
                mark = "✗" if f.severity == "fail" else "~"
                where = f" [{f.task_key}]" if f.task_key else ""
                print(f"  {mark} {f.code}{where}: {f.detail}")
        if report.verdict is report.verdict.FAIL:
            raise SystemExit(1)
        return

    if args.command == "status":
        store = SQLiteStateStore(Path(args.state_db))
        runner = FineTuneRunner(store)
        print(json.dumps(runner.get_run_status(args.run_id), indent=2))
        return

    if args.command == "list-tasks":
        store = SQLiteStateStore(Path(args.state_db))
        tasks = store.list_tasks(args.run_id)
        print(json.dumps([
            {
                "task_id": task.task_id,
                "task_key": task.task_key,
                "status": task.status.value,
                "lease_owner": task.lease_owner,
                "leased_until": task.leased_until.isoformat() if task.leased_until else None,
                "result": task.result,
                "error": task.error,
            }
            for task in tasks
        ], indent=2))
        return

    if args.command == "list-artifacts":
        store = SQLiteStateStore(Path(args.state_db))
        artifacts = store.list_artifacts(args.run_id)
        print(json.dumps([
            {
                "artifact_id": artifact.artifact_id,
                "task_id": artifact.task_id,
                "kind": artifact.kind,
                "path": artifact.path,
                "checksum": artifact.checksum,
            }
            for artifact in artifacts
        ], indent=2))
        return

    if args.command == "list-runs":
        store = SQLiteStateStore(Path(args.state_db))
        runs = store.list_runs()
        if args.since:
            cutoff = parse_since_duration(args.since)
            runs = filter_runs_since(runs, cutoff)
        # Sort by created_at descending (most recent first)
        runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
        if args.fmt == "json":
            print(json.dumps([
                {
                    "run_id": r.run_id,
                    "name": r.name,
                    "status": r.status.value,
                    "created_at": r.created_at.isoformat(),
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                }
                for r in runs
            ], indent=2))
        else:
            for r in runs:
                created = r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "?"
                finished = r.finished_at.strftime("%Y-%m-%d %H:%M") if r.finished_at else "-"
                print(f"{r.run_id}  {r.status.value:<16}  {created}  →  {finished}  {r.name}")
        return

    if args.command == "start-run":
        store = SQLiteStateStore(Path(args.state_db))
        runner = FineTuneRunner(store)
        run = store.get_run(args.run_id)
        if run is None:
            print(f"Error: unknown run_id {args.run_id!r}", flush=True)
            raise SystemExit(1)
        # Optional allowlist of authorized approvers, from run config.
        allowed = None
        approval_cfg = run.config.get("approval") if isinstance(run.config, dict) else None
        if isinstance(approval_cfg, dict) and approval_cfg.get("allowed_actors"):
            allowed = tuple(approval_cfg["allowed_actors"])
        gate = InteractiveApprovalGate(actor=args.approver, allowed_actors=allowed)
        try:
            runner.await_approval(args.run_id, gate)
            print(f"Run {args.run_id} approved. Start workers to begin execution.")
        except ApprovalError as exc:
            print(f"Denied: {exc}")
            raise SystemExit(1)
        return

    if args.command == "validate-reproducibility":
        store = SQLiteStateStore(Path(args.state_db))
        run = store.get_run(args.run_id)
        if run is None:
            print(f"Error: unknown run_id {args.run_id!r}", flush=True)
            raise SystemExit(1)
        assessment = assess_reproducibility(run, store.list_events(args.run_id))
        if args.fmt == "json":
            print(json.dumps({
                "run_id": run.run_id,
                "name": run.name,
                "level": assessment.level,
                "bitwise_reproducible": assessment.bitwise_reproducible,
                "replayable": assessment.replayable,
                "missing_fields": assessment.missing_fields,
                "warnings": assessment.warnings,
            }, indent=2))
        else:
            icon = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}.get(assessment.level, "?")
            print(f"{icon} Reproducibility: {assessment.level}  (run {run.run_id[:8]} — {run.name})")
            if assessment.missing_fields:
                print("  Missing fields:")
                for f in assessment.missing_fields:
                    print(f"    - {f}")
            if assessment.warnings:
                print("  Warnings:")
                for w in assessment.warnings:
                    print(f"    ! {w}")
        return

    if args.command == "export-manifest":
        store = SQLiteStateStore(Path(args.state_db))
        run = store.get_run(args.run_id)
        if run is None:
            print(f"Error: unknown run_id {args.run_id!r}", flush=True)
            raise SystemExit(1)
        tasks = store.list_tasks(args.run_id)
        artifacts = store.list_artifacts(args.run_id)
        manifest = export_manifest(run, tasks, artifacts, store.list_events(args.run_id))
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "compare-runs":
        store = SQLiteStateStore(Path(args.state_db))

        if args.latest_previous:
            try:
                prev_id, latest_id = find_latest_run_pair(store)
            except ValueError as exc:
                parser.error(str(exc))
                return
            run_ids = [prev_id, latest_id]
        else:
            if not args.run_ids or len(args.run_ids) < 2:
                parser.error("compare-runs requires at least two --run-id values or --latest-previous")
            run_ids = args.run_ids

        thresholds = None
        if args.thresholds:
            try:
                thresholds = json.loads(args.thresholds)
            except json.JSONDecodeError as exc:
                parser.error(f"--thresholds is not valid JSON: {exc}")
                return

        report = compare_runs(run_ids, store, thresholds=thresholds, strict=args.strict)
        if args.fmt == "json":
            print(json.dumps(report_to_dict(report), indent=2))
        else:
            print(format_report(report))
        return


if __name__ == "__main__":
    main()
