from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from finetuneharness.state.leases import utc_now
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import EventRecord, RunRecord, RunStatus, TaskRecord, TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore


def test_sqlite_store_roundtrip(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")

    run = RunRecord(
        run_id="run-1",
        name="demo",
        status=RunStatus.CREATED,
        config={"project": {"name": "demo"}, "executor": {"kind": "local"}, "artifacts": {"root": "./artifacts"}},
    )
    store.create_run(run)
    store.update_run_status("run-1", RunStatus.VALIDATED)

    task = TaskRecord(
        task_id="task-1",
        run_id="run-1",
        task_key="cell-1",
        status=TaskStatus.PENDING,
        payload={"kind": "train"},
    )
    store.create_task(task)
    store.lease_next_pending_task(run_id="run-1", worker_id="worker-x", lease_seconds=60)
    store.mark_task_running("task-1")
    store.update_task_status("task-1", TaskStatus.SUCCEEDED, result={"accuracy": 0.9})
    store.append_event(EventRecord(event_id="evt-1", run_id="run-1", task_id="task-1", kind="task_succeeded"))

    loaded = store.get_run("run-1")
    assert loaded is not None
    assert loaded.status == RunStatus.VALIDATED

    tasks = store.list_tasks("run-1")
    assert len(tasks) == 1
    assert tasks[0].status == TaskStatus.SUCCEEDED
    assert tasks[0].result == {"accuracy": 0.9}


def test_sqlite_store_lease_next_pending_task(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    run = RunRecord(
        run_id="run-2",
        name="lease-demo",
        status=RunStatus.CREATED,
        config={"project": {"name": "demo"}, "executor": {"kind": "local"}, "artifacts": {"root": "./artifacts"}},
    )
    store.create_run(run)
    store.create_task(
        TaskRecord(
            task_id="task-a",
            run_id="run-2",
            task_key="a",
            status=TaskStatus.PENDING,
            payload={"kind": "train"},
        )
    )
    leased = store.lease_next_pending_task(run_id="run-2", worker_id="worker-1", lease_seconds=60)
    assert leased is not None
    assert leased.status == TaskStatus.LEASED
    assert leased.lease_owner == "worker-1"


# ── InMemoryStore expired-lease recovery (P2) ─────────────────────────────────

def _make_run_and_task(store, run_id: str, task_id: str) -> None:
    store.create_run(RunRecord(
        run_id=run_id,
        name="lease-test",
        status=RunStatus.CREATED,
        config={"project": {"name": "t"}, "executor": {"kind": "local"}, "artifacts": {"root": "./a"}},
    ))
    store.create_task(TaskRecord(
        task_id=task_id,
        run_id=run_id,
        task_key="cell-1",
        status=TaskStatus.PENDING,
        payload={},
    ))


def test_in_memory_store_reclaims_expired_lease_on_next_lease_call():
    """lease_next_pending_task must reclaim a LEASED task whose leased_until has passed.

    Before the fix, InMemoryStore only considered PENDING tasks — an expired
    LEASED task was invisible to the scheduler, causing it to stall.
    SQLiteStore reclaims expired leases inline via OR (status=LEASED AND leased_until < now).
    """
    store = InMemoryStateStore()
    _make_run_and_task(store, "run-exp", "task-exp")

    # Lease the task with a 1-second TTL
    leased = store.lease_next_pending_task(run_id="run-exp", worker_id="w1", lease_seconds=1)
    assert leased is not None
    assert leased.status == TaskStatus.LEASED

    # Second call while lease is still valid must return nothing
    still_leased = store.lease_next_pending_task(run_id="run-exp", worker_id="w2", lease_seconds=60)
    assert still_leased is None, "task is still leased — a second worker must not steal it"

    # Backdate leased_until to simulate expiry
    from dataclasses import replace
    task = store._tasks["task-exp"]
    expired_at = utc_now() - timedelta(seconds=1)
    store._tasks["task-exp"] = replace(task, leased_until=expired_at)

    # Now a new lease call must reclaim the expired task
    reclaimed = store.lease_next_pending_task(run_id="run-exp", worker_id="w2", lease_seconds=60)
    assert reclaimed is not None, (
        "InMemoryStore did not reclaim the expired lease — "
        "lease_next_pending_task only checks PENDING, missing the LEASED+expired case"
    )
    assert reclaimed.lease_owner == "w2"


def test_in_memory_store_non_expired_lease_not_stolen():
    """A LEASED task whose lease has not expired must not be reclaimed by another worker."""
    store = InMemoryStateStore()
    _make_run_and_task(store, "run-nexp", "task-nexp")
    store.lease_next_pending_task(run_id="run-nexp", worker_id="w1", lease_seconds=3600)
    result = store.lease_next_pending_task(run_id="run-nexp", worker_id="w2", lease_seconds=60)
    assert result is None, "valid (non-expired) lease must not be stolen by another worker"


def test_foreign_key_cascade_deletes_tasks_and_events(tmp_path: Path) -> None:
    """Deleting a run must cascade to its tasks and events.

    Regression guard: _connect() now sets PRAGMA foreign_keys=ON. Without it the
    declared ON DELETE CASCADE constraints are silently unenforced and rows orphan.
    """
    from contextlib import closing

    store = SQLiteStateStore(tmp_path / "state.db")
    _make_run_and_task(store, "run-fk", "task-fk")
    store.append_event(EventRecord(
        event_id="ev-fk", run_id="run-fk", task_id="task-fk", kind="task_created", payload={},
    ))
    assert store.list_tasks("run-fk"), "precondition: task exists"
    assert store.list_events("run-fk"), "precondition: event exists"

    with closing(store._connect()) as conn, conn:
        conn.execute("DELETE FROM runs WHERE run_id = ?", ("run-fk",))

    assert store.list_tasks("run-fk") == [], "tasks not cascade-deleted (FK not enforced)"
    assert store.list_events("run-fk") == [], "events not cascade-deleted (FK not enforced)"


def test_no_connection_leak_under_many_operations(tmp_path: Path) -> None:
    """Repeated store operations must not leak file descriptors (connections close)."""
    import os

    fd_dir = "/proc/self/fd"
    if not os.path.isdir(fd_dir):
        import pytest
        pytest.skip("requires /proc to count file descriptors")

    store = SQLiteStateStore(tmp_path / "state.db")
    _make_run_and_task(store, "run-leak", "task-leak")

    before = len(os.listdir(fd_dir))
    for _ in range(200):
        store.get_run("run-leak")
        store.list_tasks("run-leak")
    after = len(os.listdir(fd_dir))

    assert after - before < 50, (
        f"file descriptors grew by {after - before} over 400 ops — connections leaking"
    )


def test_requeue_expired_lease_emits_event(tmp_path: Path) -> None:
    """requeue_expired_leases must emit a 'lease_expired' audit event per reclaimed task."""
    store = SQLiteStateStore(tmp_path / "state.db")
    _make_run_and_task(store, "run-evt", "task-evt")
    # Lease with a 0s lease so it is immediately expirable.
    store.lease_next_pending_task(run_id="run-evt", worker_id="w1", lease_seconds=0)
    import time
    time.sleep(0.02)

    count = store.requeue_expired_leases(run_id="run-evt")
    assert count == 1

    kinds = [e.kind for e in store.list_events("run-evt")]
    assert "lease_expired" in kinds, "no lease_expired event emitted on reclaim"
