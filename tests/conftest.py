"""Root conftest — fixtures available to all test subdirectories."""
from __future__ import annotations

from pathlib import Path

import pytest

from finetuneharness.artifacts.store import ArtifactStore
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.sqlite import SQLiteStateStore

BASE_CONFIG = {
    "project": {"name": "test-project"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: requires a real CUDA GPU (and torch); skipped automatically otherwise.",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def base_config() -> dict:
    return dict(BASE_CONFIG)


@pytest.fixture
def memory_store() -> InMemoryStateStore:
    return InMemoryStateStore()


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SQLiteStateStore:
    return SQLiteStateStore(tmp_path / "state.db")


@pytest.fixture
def runner(memory_store: InMemoryStateStore) -> FineTuneRunner:
    return FineTuneRunner(memory_store)


@pytest.fixture
def runner_sqlite(sqlite_store: SQLiteStateStore) -> FineTuneRunner:
    return FineTuneRunner(sqlite_store)


@pytest.fixture
def artifact_store(tmp_path: Path, sqlite_store: SQLiteStateStore) -> ArtifactStore:
    return ArtifactStore(root=tmp_path / "artifacts", state_store=sqlite_store)
