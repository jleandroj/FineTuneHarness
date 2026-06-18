"""End-to-end: the reference handler drives a run to COMPLETED with a real model.

The CPU test runs everywhere (validates the handler + harness contract: persisted
result + artifact + COMPLETED). The GPU test (@pytest.mark.gpu) asserts the same on
a real CUDA device — the system exercised against its actual purpose.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from finetuneharness.artifacts.store import ArtifactStore
from finetuneharness.examples.reference_handler import train
from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import RunStatus, TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CONFIG = {
    "project": {"name": "reference"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 7,
    "dataset_hash": "sha256:synthetic",
}


def _run_reference(tmp_path: Path) -> tuple[SQLiteStateStore, str]:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="ref", config=_CONFIG,
        tasks=[{"task_key": "cell-0", "steps": 20, "hidden": 8}],
    )
    artifacts = ArtifactStore(root=tmp_path / "artifacts", state_store=store)
    worker = LocalWorker(worker_id="w", store=store, artifact_store=artifacts)
    succeeded = worker.drain(run_id=run_id, handler=train)
    assert succeeded == 1
    return store, run_id


def test_reference_handler_end_to_end_cpu(tmp_path: Path) -> None:
    store, run_id = _run_reference(tmp_path)

    assert FineTuneRunner(store).refresh_run_status(run_id) is RunStatus.COMPLETED
    task = store.list_tasks(run_id)[0]
    assert task.status is TaskStatus.SUCCEEDED
    assert task.result is not None
    assert 0.0 <= task.result["accuracy"] <= 1.0
    assert task.result["loss"] >= 0.0
    assert task.result["device"] == "cpu"
    # Result was persisted as an artifact with a checksum.
    artifacts = store.list_artifacts(run_id)
    assert any(a.kind == "task_result" and a.checksum for a in artifacts)


@pytest.mark.gpu
def test_reference_handler_end_to_end_gpu(tmp_path: Path) -> None:
    store, run_id = _run_reference(tmp_path)

    assert FineTuneRunner(store).refresh_run_status(run_id) is RunStatus.COMPLETED
    task = store.list_tasks(run_id)[0]
    assert task.status is TaskStatus.SUCCEEDED
    assert task.result["device"] == "cuda", "training did not run on the GPU"
    assert 0.0 <= task.result["accuracy"] <= 1.0
    assert any(a.kind == "task_result" for a in store.list_artifacts(run_id))
