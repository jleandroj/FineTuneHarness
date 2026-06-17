"""Integration: approval gate wired into the full execution flow."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.approval import ApprovalError, ApprovalGate, InteractiveApprovalGate
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import RunStatus, TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CONFIG = {
    "project": {"name": "approval-integration"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
}


def _make_store(tmp_path: Path) -> SQLiteStateStore:
    return SQLiteStateStore(tmp_path / "state.db")


def test_approved_run_executes_to_completion(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    run_id = runner.create_run(
        name="approved-run",
        config=_CONFIG,
        tasks=[{"task_key": "t1"}, {"task_key": "t2"}],
    )

    runner.await_approval(run_id, ApprovalGate())  # base gate always approves

    worker = LocalWorker(worker_id="w1", store=store)
    for _ in range(2):
        worker.run_once(run_id=run_id, handler=lambda task: {"done": True})

    run = store.get_run(run_id)
    assert run is not None and run.status == RunStatus.COMPLETED

    events = store.list_events(run_id)
    assert any(e.kind == "run_approved" for e in events)


def test_denied_run_blocks_execution(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    run_id = runner.create_run(
        name="blocked-run",
        config=_CONFIG,
        tasks=[{"task_key": "t1"}],
    )

    gate = InteractiveApprovalGate(stream=io.StringIO("n\n"))

    with pytest.raises(ApprovalError):
        runner.await_approval(run_id, gate)

    # Run was not approved — tasks must remain PENDING
    tasks = store.list_tasks(run_id)
    assert all(t.status == TaskStatus.PENDING for t in tasks)

    events = store.list_events(run_id)
    assert not any(e.kind == "run_approved" for e in events)


def test_approval_recorded_as_event(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    run_id = runner.create_run(
        name="event-check",
        config=_CONFIG,
        tasks=[{"task_key": "t1"}],
    )

    runner.await_approval(run_id, ApprovalGate())

    events = store.list_events(run_id)
    approval_events = [e for e in events if e.kind == "run_approved"]
    assert len(approval_events) == 1
    assert approval_events[0].payload["gate"] == "ApprovalGate"


def test_custom_gate_can_inspect_tasks_before_approving(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    run_id = runner.create_run(
        name="inspect-run",
        config=_CONFIG,
        tasks=[{"task_key": f"t{i}"} for i in range(5)],
    )

    inspected_counts: list[int] = []

    class InspectingGate(ApprovalGate):
        def check(self, run, tasks):
            inspected_counts.append(len(tasks))

    runner.await_approval(run_id, InspectingGate())
    assert inspected_counts == [5]


def test_approval_survives_store_reinit(tmp_path: Path) -> None:
    """Approval event is persisted and visible after store is recreated."""
    db = tmp_path / "state.db"

    store1 = SQLiteStateStore(db)
    runner1 = FineTuneRunner(store1)
    run_id = runner1.create_run(
        name="persist-approval",
        config=_CONFIG,
        tasks=[{"task_key": "t1"}],
    )
    runner1.await_approval(run_id, ApprovalGate())

    # Simulate restart
    store2 = SQLiteStateStore(db)
    events = store2.list_events(run_id)
    assert any(e.kind == "run_approved" for e in events)
