"""Tests that the task lifecycle state machine is actually enforced in both stores."""
from __future__ import annotations

from pathlib import Path

import pytest

from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import RunRecord, RunStatus, TaskRecord, TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore
from finetuneharness.state.store import StateStore


def _sqlite(tmp_path: Path) -> SQLiteStateStore:
    return SQLiteStateStore(tmp_path / "sm.db")


def _memory() -> InMemoryStateStore:
    return InMemoryStateStore()


def _seed(store: StateStore, run_id: str = "r1", task_id: str = "t1") -> None:
    store.create_run(
        RunRecord(
            run_id=run_id,
            name="sm-test",
            status=RunStatus.CREATED,
            config={"project": {"name": "x"}, "executor": {"kind": "local"}, "artifacts": {"root": "./"}},
        )
    )
    store.create_task(
        TaskRecord(task_id=task_id, run_id=run_id, task_key="k1", status=TaskStatus.PENDING, payload={})
    )


@pytest.mark.parametrize("store_factory", ["sqlite", "memory"])
def test_pending_to_succeeded_is_rejected(store_factory: str, tmp_path: Path) -> None:
    store: StateStore = _sqlite(tmp_path) if store_factory == "sqlite" else _memory()
    _seed(store)
    with pytest.raises(ValueError, match="invalid task transition"):
        store.update_task_status("t1", TaskStatus.SUCCEEDED)


@pytest.mark.parametrize("store_factory", ["sqlite", "memory"])
def test_succeeded_is_terminal(store_factory: str, tmp_path: Path) -> None:
    store: StateStore = _sqlite(tmp_path) if store_factory == "sqlite" else _memory()
    _seed(store)
    store.lease_next_pending_task(run_id="r1", worker_id="w1", lease_seconds=60)
    store.mark_task_running("t1")
    store.update_task_status("t1", TaskStatus.SUCCEEDED)
    with pytest.raises(ValueError, match="invalid task transition"):
        store.update_task_status("t1", TaskStatus.FAILED)


@pytest.mark.parametrize("store_factory", ["sqlite", "memory"])
def test_failed_is_terminal(store_factory: str, tmp_path: Path) -> None:
    store: StateStore = _sqlite(tmp_path) if store_factory == "sqlite" else _memory()
    _seed(store)
    store.lease_next_pending_task(run_id="r1", worker_id="w1", lease_seconds=60)
    store.mark_task_running("t1")
    store.update_task_status("t1", TaskStatus.FAILED, error="boom")
    with pytest.raises(ValueError, match="invalid task transition"):
        store.update_task_status("t1", TaskStatus.PENDING)


@pytest.mark.parametrize("store_factory", ["sqlite", "memory"])
def test_running_to_pending_is_allowed_for_retry(store_factory: str, tmp_path: Path) -> None:
    store: StateStore = _sqlite(tmp_path) if store_factory == "sqlite" else _memory()
    _seed(store)
    store.lease_next_pending_task(run_id="r1", worker_id="w1", lease_seconds=60)
    store.mark_task_running("t1")
    store.update_task_status("t1", TaskStatus.PENDING, error="transient")
    tasks = store.list_tasks("r1")
    assert tasks[0].status == TaskStatus.PENDING


@pytest.mark.parametrize("store_factory", ["sqlite", "memory"])
def test_full_happy_path(store_factory: str, tmp_path: Path) -> None:
    store: StateStore = _sqlite(tmp_path) if store_factory == "sqlite" else _memory()
    _seed(store)
    store.lease_next_pending_task(run_id="r1", worker_id="w1", lease_seconds=60)
    store.mark_task_running("t1")
    store.update_task_status("t1", TaskStatus.SUCCEEDED, result={"ok": True})
    tasks = store.list_tasks("r1")
    assert tasks[0].status == TaskStatus.SUCCEEDED
    assert tasks[0].result == {"ok": True}
    assert tasks[0].lease_owner is None
    assert tasks[0].leased_until is None
