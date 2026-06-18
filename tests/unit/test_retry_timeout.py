from pathlib import Path
import time

from finetuneharness.artifacts.store import ArtifactStore
from finetuneharness.executor.policy import RetryPolicy, TimeoutPolicy
from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import RunStatus, TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore


def test_worker_retries_then_succeeds(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    artifacts = ArtifactStore(root=tmp_path / "artifacts", state_store=store)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="retry-demo",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": str(tmp_path / 'artifacts')},
            "seed": 42,
            "dataset_hash": "sha256:test",
        },
        tasks=[{"task_key": "cell-1", "kind": "train", "max_attempts": 2}],
    )

    attempts = {"count": 0}

    def flaky(task):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("first fail")
        return {"ok": True}

    worker = LocalWorker(
        worker_id="worker-a",
        store=store,
        artifact_store=artifacts,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    try:
        worker.run_once(run_id=run_id, handler=flaky)
    except RuntimeError:
        pass
    worker.run_once(run_id=run_id, handler=flaky)

    task = store.list_tasks(run_id)[0]
    run = store.get_run(run_id)
    assert task.status == TaskStatus.SUCCEEDED
    assert task.attempt_count == 2
    assert run is not None and run.status == RunStatus.COMPLETED


def test_worker_marks_timeout(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="timeout-demo",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": str(tmp_path / 'artifacts')},
            "seed": 42,
            "dataset_hash": "sha256:test",
        },
        tasks=[{"task_key": "cell-1", "kind": "train"}],
    )

    worker = LocalWorker(
        worker_id="worker-a",
        store=store,
        timeout_policy=TimeoutPolicy(timeout_seconds=0),
    )

    def slow(task):
        time.sleep(0.01)
        return {"ok": True}

    try:
        worker.run_once(run_id=run_id, handler=slow)
    except TimeoutError:
        pass

    task = store.list_tasks(run_id)[0]
    assert task.status == TaskStatus.TIMED_OUT
    assert task.attempt_count == 1
