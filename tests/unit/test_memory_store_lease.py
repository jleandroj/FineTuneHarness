"""InMemoryStateStore lease-expiry parity with SQLiteStore.

The in-memory store must reclaim LEASED tasks whose lease has expired, exactly
like SQLiteStore — otherwise tests that use the fast in-memory store would not
exercise the recovery path that production (SQLite) relies on.
"""
from __future__ import annotations

import time

from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import TaskStatus


_CONFIG = {
    "project": {"name": "lease-test"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 7,
    "dataset_hash": "sha256:abc123",
}


def _make_run_with_one_task():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = runner.create_run(name="r", config=_CONFIG, tasks=[{"task_key": "a"}])
    return store, run_id


def test_expired_lease_is_reclaimed_by_other_worker():
    store, run_id = _make_run_with_one_task()

    # Worker 1 takes a zero-second lease — it expires immediately.
    leased = store.lease_next_pending_task(run_id=run_id, worker_id="w1", lease_seconds=0)
    assert leased is not None
    assert leased.lease_owner == "w1"
    assert leased.status is TaskStatus.LEASED

    # Let the clock advance past leased_until.
    time.sleep(0.02)

    # Worker 2 must be able to reclaim the expired lease.
    reclaimed = store.lease_next_pending_task(run_id=run_id, worker_id="w2", lease_seconds=60)
    assert reclaimed is not None, "expired lease was not reclaimed (parity with SQLiteStore broken)"
    assert reclaimed.lease_owner == "w2"
    assert reclaimed.task_id == leased.task_id


def test_valid_lease_is_not_reclaimed():
    store, run_id = _make_run_with_one_task()

    # Worker 1 holds a long, still-valid lease.
    leased = store.lease_next_pending_task(run_id=run_id, worker_id="w1", lease_seconds=60)
    assert leased is not None

    # Worker 2 must NOT steal a lease that has not expired.
    stolen = store.lease_next_pending_task(run_id=run_id, worker_id="w2", lease_seconds=60)
    assert stolen is None, "a non-expired lease was stolen — double-execution risk"
