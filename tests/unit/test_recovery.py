from pathlib import Path

from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore


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
