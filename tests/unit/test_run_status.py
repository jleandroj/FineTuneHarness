from pathlib import Path

from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import RunStatus
from finetuneharness.state.sqlite import SQLiteStateStore


def test_run_becomes_completed_when_all_tasks_succeed(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="complete-demo",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": "./artifacts"},
            "seed": 42,
            "dataset_hash": "sha256:test",
        },
        tasks=[
            {"task_key": "cell-1", "kind": "train"},
            {"task_key": "cell-2", "kind": "eval"},
        ],
    )
    worker = LocalWorker(worker_id="worker-a", store=store)
    worker.run_once(run_id=run_id, handler=lambda task: {"done": task.task_key})
    worker.run_once(run_id=run_id, handler=lambda task: {"done": task.task_key})

    run = store.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED


def test_run_becomes_partial_failed_when_some_tasks_fail(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="partial-fail-demo",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": "./artifacts"},
            "seed": 42,
            "dataset_hash": "sha256:test",
        },
        tasks=[
            {"task_key": "cell-1", "kind": "train"},
            {"task_key": "cell-2", "kind": "eval"},
        ],
    )
    worker = LocalWorker(worker_id="worker-a", store=store)
    try:
        worker.run_once(run_id=run_id, handler=lambda task: (_ for _ in ()).throw(RuntimeError("boom")))
    except RuntimeError:
        pass
    worker.run_once(run_id=run_id, handler=lambda task: {"done": task.task_key})

    run = store.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.PARTIAL_FAILED
