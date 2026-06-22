"""Tests for the pre-edit / contamination guard."""
from __future__ import annotations

from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.verification import find_active_runs, preflight

_CONFIG = {
    "project": {"name": "demo"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:test",
}


def _run(store, tasks):
    return FineTuneRunner(store).create_run(name="r", config=_CONFIG, tasks=tasks)


def test_no_in_flight_is_safe_to_edit() -> None:
    store = InMemoryStateStore()
    _run(store, [{"task_key": "a", "kind": "train"}])  # all PENDING, none in flight
    report = preflight(store)
    assert report.safe_to_edit is True
    assert report.active == []


def test_running_task_blocks_editing() -> None:
    store = InMemoryStateStore()
    run_id = _run(store, [{"task_key": "a", "kind": "train"}, {"task_key": "b", "kind": "train"}])
    t = store.lease_next_pending_task(run_id=run_id, worker_id="w1", lease_seconds=600)
    store.mark_task_running(t.task_id)  # -> RUNNING (in flight)

    report = preflight(store)
    assert report.safe_to_edit is False
    assert len(report.active) == 1
    assert report.active[0].run_id == run_id
    assert report.active[0].in_flight == 1
    assert "w1" in report.active[0].lease_owners


def test_leased_task_also_blocks_editing() -> None:
    store = InMemoryStateStore()
    run_id = _run(store, [{"task_key": "a", "kind": "train"}])
    store.lease_next_pending_task(run_id=run_id, worker_id="w1", lease_seconds=600)  # LEASED
    assert preflight(store).safe_to_edit is False


def test_terminal_only_run_is_safe() -> None:
    store = InMemoryStateStore()
    run_id = _run(store, [{"task_key": "a", "kind": "train"}])
    t = store.lease_next_pending_task(run_id=run_id, worker_id="w1", lease_seconds=600)
    store.mark_task_running(t.task_id)
    store.update_task_status(t.task_id, __import__("finetuneharness.state.models", fromlist=["TaskStatus"]).TaskStatus.SUCCEEDED)
    assert find_active_runs(store) == []
    assert preflight(store).safe_to_edit is True
