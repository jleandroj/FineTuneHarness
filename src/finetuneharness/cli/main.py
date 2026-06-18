from __future__ import annotations

import argparse
import json
from pathlib import Path

from finetuneharness.evaluation.comparator import compare_runs
from finetuneharness.evaluation.report import format_report, report_to_dict
from finetuneharness.orchestrator.approval import ApprovalError, InteractiveApprovalGate
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.sqlite import SQLiteStateStore


def main() -> None:
    parser = argparse.ArgumentParser(description="FineTuneHarness CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    init_run = sub.add_parser("create-run", help="Create a bootstrap run from JSON config")
    init_run.add_argument("--name", required=True)
    init_run.add_argument("--config", required=True, help="Path to JSON config file")
    init_run.add_argument("--tasks", required=True, help="Path to JSON task list file")
    init_run.add_argument("--state-db", default=".finetuneharness/state.db", help="SQLite state DB path")
    init_run.add_argument("--memory", action="store_true", help="Use in-memory store instead of SQLite")

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

    list_runs_cmd = sub.add_parser("list-runs", help="List all runs in the state DB")
    list_runs_cmd.add_argument("--state-db", default=".finetuneharness/state.db")
    list_runs_cmd.add_argument("--format", dest="fmt", choices=["text", "json"], default="text")

    compare_cmd = sub.add_parser("compare-runs", help="Compare two or more runs (first is baseline)")
    compare_cmd.add_argument("--run-id", dest="run_ids", action="append", required=True, metavar="RUN_ID",
                             help="Run IDs to compare; first is baseline. Repeat for each run.")
    compare_cmd.add_argument("--format", dest="fmt", choices=["text", "json"], default="text")
    compare_cmd.add_argument("--state-db", default=".finetuneharness/state.db")

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
        if args.fmt == "json":
            print(json.dumps([
                {"run_id": r.run_id, "name": r.name, "status": r.status.value}
                for r in runs
            ], indent=2))
        else:
            for r in runs:
                print(f"{r.run_id}  {r.status.value:<16}  {r.name}")
        return

    if args.command == "start-run":
        store = SQLiteStateStore(Path(args.state_db))
        runner = FineTuneRunner(store)
        gate = InteractiveApprovalGate()
        try:
            runner.await_approval(args.run_id, gate)
            print(f"Run {args.run_id} approved. Start workers to begin execution.")
        except ApprovalError as exc:
            print(f"Denied: {exc}")
            raise SystemExit(1)
        return

    if args.command == "compare-runs":
        if len(args.run_ids) < 2:
            parser.error("compare-runs requires at least two --run-id values")
        store = SQLiteStateStore(Path(args.state_db))
        report = compare_runs(args.run_ids, store)
        if args.fmt == "json":
            print(json.dumps(report_to_dict(report), indent=2))
        else:
            print(format_report(report))
        return
