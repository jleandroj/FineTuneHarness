from pathlib import Path

from finetuneharness.artifacts.store import ArtifactStore
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.sqlite import SQLiteStateStore


def test_write_json_artifact_registers_checksum(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="artifact-demo",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": str(tmp_path / 'artifacts')},
        },
        tasks=[{"task_key": "cell-1", "kind": "train"}],
    )
    task_id = store.list_tasks(run_id)[0].task_id
    artifacts = ArtifactStore(root=tmp_path / "artifacts", state_store=store)
    record = artifacts.write_json_artifact(
        run_id=run_id,
        task_id=task_id,
        kind="result",
        payload={"accuracy": 0.9},
    )

    assert Path(record.path).exists()
    assert len(record.checksum) == 64
    assert artifacts.checksum_file(record.path) == record.checksum
