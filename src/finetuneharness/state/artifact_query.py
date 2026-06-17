from __future__ import annotations

from finetuneharness.state.models import ArtifactRecord
from finetuneharness.state.store import StateStore


def list_run_artifacts(store: StateStore, run_id: str) -> list[ArtifactRecord]:
    return store.list_artifacts(run_id)
