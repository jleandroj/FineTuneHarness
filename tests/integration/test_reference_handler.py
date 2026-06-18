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


def _run_reference(tmp_path: Path, device: str) -> tuple[SQLiteStateStore, str]:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="ref", config=_CONFIG,
        tasks=[{"task_key": "cell-0", "steps": 20, "hidden": 8, "device": device}],
    )
    artifacts = ArtifactStore(root=tmp_path / "artifacts", state_store=store)
    worker = LocalWorker(worker_id="w", store=store, artifact_store=artifacts)
    succeeded = worker.drain(run_id=run_id, handler=train)
    assert succeeded == 1
    return store, run_id


def test_reference_handler_end_to_end_cpu(tmp_path: Path) -> None:
    store, run_id = _run_reference(tmp_path, device="cpu")

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


def _train_once(tmp_path: Path, tag: str, seed: int, device: str) -> dict:
    store = SQLiteStateStore(tmp_path / f"{tag}.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name=tag, config={**_CONFIG, "seed": seed},
        tasks=[{"task_key": "c", "steps": 30, "hidden": 8, "device": device}],
    )
    LocalWorker(worker_id="w", store=store).drain(run_id=run_id, handler=train)
    result = store.list_tasks(run_id)[0].result
    assert result is not None
    return result


def test_training_is_deterministic_cpu(tmp_path: Path) -> None:
    """Two runs with the same seed produce bit-identical training metrics (CPU)."""
    r1 = _train_once(tmp_path, "a", seed=123, device="cpu")
    r2 = _train_once(tmp_path, "b", seed=123, device="cpu")
    assert r1["accuracy"] == r2["accuracy"]
    assert r1["loss"] == r2["loss"]


@pytest.mark.gpu
def test_training_is_deterministic_gpu(tmp_path: Path) -> None:
    """Same-seed determinism on a real GPU (apply_seed forces deterministic algos)."""
    r1 = _train_once(tmp_path, "a", seed=123, device="cuda")
    r2 = _train_once(tmp_path, "b", seed=123, device="cuda")
    assert r1["accuracy"] == r2["accuracy"]
    assert r1["loss"] == r2["loss"]


@pytest.mark.gpu
def test_reference_handler_end_to_end_gpu(tmp_path: Path) -> None:
    # Pin the device explicitly so the run cannot silently fall back to CPU.
    store, run_id = _run_reference(tmp_path, device="cuda")

    assert FineTuneRunner(store).refresh_run_status(run_id) is RunStatus.COMPLETED
    task = store.list_tasks(run_id)[0]
    assert task.status is TaskStatus.SUCCEEDED
    assert task.result["device"] == "cuda", "training did not run on the pinned GPU"
    assert 0.0 <= task.result["accuracy"] <= 1.0
    assert any(a.kind == "task_result" for a in store.list_artifacts(run_id))
