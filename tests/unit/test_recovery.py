from pathlib import Path

from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CONFIG = {
    "project": {"name": "demo"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:test",
}


def test_expired_lease_can_be_reacquired(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="recovery-demo",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": "./artifacts"},
            "seed": 42,
            "dataset_hash": "sha256:test",
        },
        tasks=[{"task_key": "cell-1", "kind": "train"}],
    )

    first = store.lease_next_pending_task(run_id=run_id, worker_id="worker-a", lease_seconds=0)
    assert first is not None
    assert first.status == TaskStatus.LEASED

    second = store.lease_next_pending_task(run_id=run_id, worker_id="worker-b", lease_seconds=60)
    assert second is not None
    assert second.task_id == first.task_id
    assert second.lease_owner == "worker-b"


def _make_crashed_run(store) -> str:
    """Create a run and strand two tasks (1 RUNNING, 1 LEASED) as a hard crash would,
    with un-expired leases so no automatic path reclaims them."""
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="hard-crash", config=_CONFIG,
        tasks=[{"task_key": f"cell-{i}", "kind": "train"} for i in range(3)],
    )
    t1 = store.lease_next_pending_task(run_id=run_id, worker_id="dead-worker", lease_seconds=600)
    assert t1 is not None
    store.mark_task_running(t1.task_id)  # -> RUNNING (no automatic reclaim covers this)
    t2 = store.lease_next_pending_task(run_id=run_id, worker_id="dead-worker", lease_seconds=600)
    assert t2 is not None  # -> LEASED, lease not expired
    return run_id


def test_recover_orphaned_tasks_requeues_running_and_leased(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    run_id = _make_crashed_run(store)

    pre = {t.status for t in store.list_tasks(run_id)}
    assert TaskStatus.RUNNING in pre and TaskStatus.LEASED in pre

    n = store.recover_orphaned_tasks(run_id=run_id)
    assert n == 2

    after = store.list_tasks(run_id)
    assert all(t.status is TaskStatus.PENDING for t in after)
    assert all(t.lease_owner is None and t.leased_until is None for t in after)

    kinds = [e.kind for e in store.list_events(run_id)]
    assert kinds.count("task_recovered") == 2

    # Idempotent: nothing left to recover; recovered tasks are leaseable again.
    assert store.recover_orphaned_tasks(run_id=run_id) == 0
    assert store.lease_next_pending_task(run_id=run_id, worker_id="fresh", lease_seconds=60) is not None


def test_recover_orphaned_tasks_memory_store_parity() -> None:
    store = InMemoryStateStore()
    run_id = _make_crashed_run(store)
    assert store.recover_orphaned_tasks(run_id=run_id) == 2
    assert all(t.status is TaskStatus.PENDING for t in store.list_tasks(run_id))
    assert [e.kind for e in store.list_events(run_id)].count("task_recovered") == 2
