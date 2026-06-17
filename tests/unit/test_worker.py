from pathlib import Path

from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.sqlite import SQLiteStateStore


def test_worker_leases_and_completes_one_task(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="worker-demo",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": "./artifacts"},
        },
        tasks=[{"task_key": "cell-1", "kind": "train"}],
    )

    worker = LocalWorker(worker_id="worker-a", store=store)
    task = worker.run_once(run_id=run_id, handler=lambda task: {"ok": True, "task_key": task.task_key})

    assert task is not None
    tasks = store.list_tasks(run_id)
    assert len(tasks) == 1
    assert tasks[0].result == {"ok": True, "task_key": "cell-1"}
