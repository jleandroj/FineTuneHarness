from pathlib import Path

from finetuneharness.artifacts.store import ArtifactStore
from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.sqlite import SQLiteStateStore


def test_worker_persists_result_artifact(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    artifacts = ArtifactStore(root=tmp_path / "artifacts", state_store=store)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="worker-artifact-demo",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": str(tmp_path / 'artifacts')},
            "seed": 42,
            "dataset_hash": "sha256:test",
        },
        tasks=[{"task_key": "cell-1", "kind": "train"}],
    )

    worker = LocalWorker(worker_id="worker-a", store=store, artifact_store=artifacts)
    worker.run_once(run_id=run_id, handler=lambda task: {"ok": True})

    task = store.list_tasks(run_id)[0]
    assert task.result is not None
    assert task.result["ok"] is True
    assert "wall_seconds" in task.result
