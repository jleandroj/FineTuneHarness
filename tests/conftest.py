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
