"""E2E tests: exercise the full CLI from argument parsing to state persistence."""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_BASE_CONFIG: dict[str, Any] = {
    "project": {"name": "e2e-project"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:test",
}

_BASE_TASKS = [
    {"task_key": "task-a", "kind": "train"},
    {"task_key": "task-b", "kind": "eval"},
]


def _write_fixtures(tmp_path: Path, config=None, tasks=None) -> tuple[Path, Path]:
    cfg_file = tmp_path / "config.json"
    tasks_file = tmp_path / "tasks.json"
    cfg_file.write_text(json.dumps(config or _BASE_CONFIG))
    tasks_file.write_text(json.dumps(tasks or _BASE_TASKS))
    return cfg_file, tasks_file


def _invoke(*args: str, stdin_data: str = "", expected_exit: int = 0) -> str:
    """Call the CLI main() with patched argv, return captured stdout."""
    from finetuneharness.cli.main import main

    buf = io.StringIO()
    with (
        redirect_stdout(buf),
        patch.object(sys, "argv", ["finetuneharness", *args]),
        patch("sys.stdin", io.StringIO(stdin_data)),
    ):
        try:
            main()
        except SystemExit as exc:
            if exc.code != expected_exit:
                raise AssertionError(
                    f"Expected exit {expected_exit}, got {exc.code}.\nOutput: {buf.getvalue()}"
                ) from exc
    return buf.getvalue()


# ---------------------------------------------------------------------------
# create-run
# ---------------------------------------------------------------------------

def test_create_run_prints_run_id(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path)
    db = str(tmp_path / "state.db")

    output = _invoke("create-run", "--name", "my-run", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db)

    run_id = output.strip()
    assert len(run_id) == 32, f"Expected 32-char UUID hex, got: {run_id!r}"
    assert run_id.isalnum()


def test_create_run_memory_flag_does_not_crash(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path)
    output = _invoke("create-run", "--name", "mem-run", "--config", str(cfg), "--tasks", str(tasks), "--memory")
    assert output.strip()


def test_create_run_missing_config_file_raises(tmp_path: Path) -> None:
    db = str(tmp_path / "state.db")
    with pytest.raises((SystemExit, FileNotFoundError, OSError)):
        _invoke("create-run", "--name", "bad", "--config", "/nonexistent.json", "--tasks", "/nonexistent.json", "--state-db", db)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_returns_valid_json(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path)
    db = str(tmp_path / "state.db")

    run_id = _invoke("create-run", "--name", "status-test", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db).strip()

    output = _invoke("status", "--run-id", run_id, "--state-db", db)
    data = json.loads(output)

    assert data["run_id"] == run_id
    assert data["name"] == "status-test"
    assert data["task_total"] == 2
    assert "task_counts" in data


def test_status_unknown_run_id_raises(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path, tasks=[])
    db = str(tmp_path / "state.db")
    _invoke("create-run", "--name", "seed", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db)

    with pytest.raises((SystemExit, Exception)):
        _invoke("status", "--run-id", "deadbeef" * 4, "--state-db", db)


# ---------------------------------------------------------------------------
# list-tasks
# ---------------------------------------------------------------------------

def test_list_tasks_returns_task_keys(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path)
    db = str(tmp_path / "state.db")

    run_id = _invoke("create-run", "--name", "lt-test", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db).strip()

    output = _invoke("list-tasks", "--run-id", run_id, "--state-db", db)
    rows = json.loads(output)

    assert len(rows) == 2
    keys = {r["task_key"] for r in rows}
    assert keys == {"task-a", "task-b"}
    assert all(r["status"] == "pending" for r in rows)


# ---------------------------------------------------------------------------
# list-artifacts
# ---------------------------------------------------------------------------

def test_list_artifacts_empty_before_execution(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path)
    db = str(tmp_path / "state.db")

    run_id = _invoke("create-run", "--name", "art-test", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db).strip()

    output = _invoke("list-artifacts", "--run-id", run_id, "--state-db", db)
    assert json.loads(output) == []


# ---------------------------------------------------------------------------
# compare-runs
# ---------------------------------------------------------------------------

def test_compare_runs_requires_two_ids(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path, tasks=[{"task_key": "t"}])
    db = str(tmp_path / "state.db")
    run_id = _invoke("create-run", "--name", "r1", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db).strip()

    with pytest.raises((SystemExit, AssertionError)):
        _invoke("compare-runs", "--run-id", run_id, "--state-db", db)


def test_compare_runs_text_output(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path, tasks=[{"task_key": "t"}])
    db = str(tmp_path / "state.db")

    r1 = _invoke("create-run", "--name", "baseline", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db).strip()
    r2 = _invoke("create-run", "--name", "experiment", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db).strip()

    output = _invoke("compare-runs", "--run-id", r1, "--run-id", r2, "--state-db", db)

    assert "baseline" in output
    assert "experiment" in output
    assert "Run Summaries" in output


def test_compare_runs_json_output(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path, tasks=[{"task_key": "t"}])
    db = str(tmp_path / "state.db")

    r1 = _invoke("create-run", "--name", "r1", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db).strip()
    r2 = _invoke("create-run", "--name", "r2", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db).strip()

    output = _invoke("compare-runs", "--run-id", r1, "--run-id", r2, "--format", "json", "--state-db", db)
    data = json.loads(output)

    assert data["baseline"] == r1
    assert set(data["run_ids"]) == {r1, r2}
    assert isinstance(data["task_comparisons"], list)


# ---------------------------------------------------------------------------
# start-run (approval gate)
# ---------------------------------------------------------------------------

def test_start_run_approved_on_yes(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path)
    db = str(tmp_path / "state.db")
    run_id = _invoke("create-run", "--name", "approve-me", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db).strip()

    output = _invoke("start-run", "--run-id", run_id, "--state-db", db, stdin_data="y\n")

    assert "approved" in output.lower()


def test_start_run_denied_exits_1(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path)
    db = str(tmp_path / "state.db")
    run_id = _invoke("create-run", "--name", "deny-me", "--config", str(cfg), "--tasks", str(tasks), "--state-db", db).strip()

    # expected_exit=1 means _invoke won't raise — just verify the denial message
    output = _invoke("start-run", "--run-id", run_id, "--state-db", db, stdin_data="n\n", expected_exit=1)
    assert "denied" in output.lower() or "Denied" in output


# ---------------------------------------------------------------------------
# run (execute tasks with a handler)
# ---------------------------------------------------------------------------

def test_run_executes_pending_tasks_end_to_end(tmp_path: Path) -> None:
    """`run` must drive a created run to COMPLETED using an imported handler."""
    cfg, tasks = _write_fixtures(tmp_path)
    db = str(tmp_path / "state.db")
    run_id = _invoke(
        "create-run", "--name", "run-exec", "--config", str(cfg),
        "--tasks", str(tasks), "--state-db", db,
    ).strip()

    # Write a handler module and make it importable.
    handler_mod = tmp_path / "cli_handler_mod.py"
    handler_mod.write_text(
        "def handle(task):\n"
        "    return {'accuracy': 0.9, 'f1': 0.88}\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        out = _invoke(
            "run", "--run-id", run_id, "--state-db", db,
            "--handler", "cli_handler_mod:handle", "--skip-approval",
        )
        assert "succeeded" in out

        status = json.loads(_invoke("status", "--run-id", run_id, "--state-db", db))
        assert status["status"].lower() == "completed"
        succeeded = status["task_counts"].get("succeeded") or status["task_counts"].get("SUCCEEDED")
        assert succeeded == len(_BASE_TASKS)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("cli_handler_mod", None)


def test_run_with_bad_handler_spec_exits(tmp_path: Path) -> None:
    cfg, tasks = _write_fixtures(tmp_path)
    db = str(tmp_path / "state.db")
    run_id = _invoke(
        "create-run", "--name", "bad-h", "--config", str(cfg),
        "--tasks", str(tasks), "--state-db", db,
    ).strip()
    # Missing ':' → SystemExit (skip approval so we reach the handler-spec check)
    _invoke("run", "--run-id", run_id, "--state-db", db,
            "--handler", "not_a_valid_spec", "--skip-approval", expected_exit=1)


def test_run_refuses_unapproved_run(tmp_path: Path) -> None:
    """The approval gate is enforced: `run` exits 1 on a run that was not approved."""
    cfg, tasks = _write_fixtures(tmp_path)
    db = str(tmp_path / "state.db")
    run_id = _invoke(
        "create-run", "--name", "needs-approval", "--config", str(cfg),
        "--tasks", str(tasks), "--state-db", db,
    ).strip()

    handler_mod = tmp_path / "approval_handler_mod.py"
    handler_mod.write_text("def handle(task):\n    return {'accuracy': 0.9, 'f1': 0.88}\n")
    sys.path.insert(0, str(tmp_path))
    try:
        # No start-run → blocked.
        out = _invoke(
            "run", "--run-id", run_id, "--state-db", db,
            "--handler", "approval_handler_mod:handle", expected_exit=1,
        )
        assert "not been approved" in out

        # Approve, then run succeeds without --skip-approval.
        _invoke("start-run", "--run-id", run_id, "--state-db", db, stdin_data="y\n")
        out = _invoke(
            "run", "--run-id", run_id, "--state-db", db,
            "--handler", "approval_handler_mod:handle",
        )
        assert "succeeded" in out
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("approval_handler_mod", None)


def test_skip_approval_records_audit_event(tmp_path: Path) -> None:
    """`run --skip-approval` must leave an approval_skipped audit event (who bypassed)."""
    from finetuneharness.state.sqlite import SQLiteStateStore

    cfg, tasks = _write_fixtures(tmp_path)
    db = str(tmp_path / "state.db")
    run_id = _invoke(
        "create-run", "--name", "skip-audit", "--config", str(cfg),
        "--tasks", str(tasks), "--state-db", db,
    ).strip()

    handler_mod = tmp_path / "skip_handler_mod.py"
    handler_mod.write_text("def handle(task):\n    return {'accuracy': 0.9, 'f1': 0.88}\n")
    sys.path.insert(0, str(tmp_path))
    try:
        _invoke("run", "--run-id", run_id, "--state-db", db,
                "--handler", "skip_handler_mod:handle", "--skip-approval")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("skip_handler_mod", None)

    events = SQLiteStateStore(Path(db)).list_events(run_id)
    skipped = [e for e in events if e.kind == "approval_skipped"]
    assert len(skipped) == 1
    assert skipped[0].payload.get("actor")
