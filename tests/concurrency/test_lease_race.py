"""Lease expiry races: tasks stranded in LEASED must be recoverable."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CONFIG = {
    "project": {"name": "lease-race"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
}


def _make_store(tmp_path: Path) -> SQLiteStateStore:
    return SQLiteStateStore(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# expired lease is reacquired exactly once
# ---------------------------------------------------------------------------

def test_expired_lease_reacquired_by_exactly_one_worker(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="expired-lease",
        config=_CONFIG,
        tasks=[{"task_key": "stranded"}],
    )

    # Lease with 0-second TTL — immediately expired
    leased = store.lease_next_pending_task(run_id=run_id, worker_id="dead-worker", lease_seconds=0)
    assert leased is not None
    assert leased.status == TaskStatus.LEASED

    # 10 workers race to re-acquire the expired lease
    acquired: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(10)

    def try_acquire(worker_id: str) -> None:
        barrier.wait()
        task = store.lease_next_pending_task(run_id=run_id, worker_id=worker_id, lease_seconds=60)
        if task is not None:
            with lock:
                acquired.append(worker_id)

    threads = [threading.Thread(target=try_acquire, args=(f"w{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(acquired) == 1, f"Expected 1 winner, got {len(acquired)}: {acquired}"


# ---------------------------------------------------------------------------
# task stranded in LEASED (worker crash simulation) is re-executed
# ---------------------------------------------------------------------------

def test_stranded_leased_task_is_completed_by_second_worker(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="stranded-task",
        config=_CONFIG,
        tasks=[{"task_key": "fragile"}],
    )

    # Worker 1 leases with 0-TTL and "crashes" without completing
    store.lease_next_pending_task(run_id=run_id, worker_id="crashed-worker", lease_seconds=0)
    task_state = store.list_tasks(run_id)[0]
    assert task_state.status == TaskStatus.LEASED

    # Worker 2 uses scheduler which calls requeue_expired_leases first
    worker2 = LocalWorker(worker_id="w2", store=store)
    completed = worker2.run_once(run_id=run_id, handler=lambda t: {"recovered": True})

    assert completed is not None
    assert completed.task_key == "fragile"

    final = store.list_tasks(run_id)[0]
    assert final.status == TaskStatus.SUCCEEDED
    assert final.lease_owner is None


# ---------------------------------------------------------------------------
# many workers competing after lease expires — no task executed twice
# ---------------------------------------------------------------------------

def test_many_workers_after_lease_expiry_no_double_execution(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="many-after-expiry",
        config=_CONFIG,
        tasks=[{"task_key": f"t{i}"} for i in range(5)],
    )

    # Strand all tasks with expired leases
    for _ in range(5):
        store.lease_next_pending_task(run_id=run_id, worker_id="dead", lease_seconds=0)

    executed: list[str] = []
    lock = threading.Lock()
    errors: list[Exception] = []

    def run_worker(wid: str) -> None:
        worker = LocalWorker(worker_id=wid, store=store)
        while True:
            try:
                task = worker.run_once(run_id=run_id, handler=lambda t: {"done": True})
            except Exception as exc:
                with lock:
                    errors.append(exc)
                break
            if task is None:
                break
            with lock:
                executed.append(task.task_key)

    threads = [threading.Thread(target=run_worker, args=(f"w{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    from collections import Counter
    counts = Counter(executed)
    assert all(v == 1 for v in counts.values()), f"Duplicate executions: {counts}"
    assert len(executed) == 5


# ---------------------------------------------------------------------------
# lease TTL respected: active lease blocks second worker
# ---------------------------------------------------------------------------

def test_active_lease_blocks_second_worker(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="active-lease",
        config=_CONFIG,
        tasks=[{"task_key": "held"}],
    )

    # Lease with a long TTL — still active
    store.lease_next_pending_task(run_id=run_id, worker_id="holder", lease_seconds=3600)

    # Another worker must not be able to take it
    second = store.lease_next_pending_task(run_id=run_id, worker_id="interloper", lease_seconds=60)
    assert second is None, "Active lease should block second worker"
