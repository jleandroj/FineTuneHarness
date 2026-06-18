"""Tests for seed application — Q1 reproducibility enforcement.

The harness applies seeds before every handler call so the handler sees a
deterministic random state without having to seed manually.
"""
from __future__ import annotations

import os
import random

import pytest

from finetuneharness.executor.seeding import apply_seed
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore


_BASE_CONFIG = {
    "project": {"name": "seed-test"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:abc123",
}


# ── apply_seed unit tests ─────────────────────────────────────────────────────

def test_apply_seed_deterministic_random():
    apply_seed(99)
    val1 = random.random()
    apply_seed(99)
    val2 = random.random()
    assert val1 == val2


def test_apply_seed_different_seeds_differ():
    apply_seed(1)
    val1 = random.random()
    apply_seed(2)
    val2 = random.random()
    assert val1 != val2


def test_apply_seed_sets_cublas_env_if_absent(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    apply_seed(42)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_apply_seed_does_not_overwrite_existing_cublas_env(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    apply_seed(42)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_apply_seed_rejects_non_int():
    with pytest.raises(TypeError, match="seed must be int"):
        apply_seed("42")  # type: ignore[arg-type]


# ── Worker applies seed before handler ───────────────────────────────────────

def test_worker_applies_seed_before_handler():
    """Two tasks in two separate runs with the same seed should see identical
    random state at the start of the handler."""
    from finetuneharness.executor.worker import LocalWorker

    store = InMemoryStateStore()
    runner = FineTuneRunner(store)

    captured: list[float] = []

    def recording_handler(task):
        captured.append(random.random())
        return {}

    run1 = runner.create_run(name="r1", config=_BASE_CONFIG, tasks=[{"task_key": "a"}])
    run2 = runner.create_run(name="r2", config=_BASE_CONFIG, tasks=[{"task_key": "b"}])

    worker = LocalWorker(worker_id="w", store=store, runner=runner)
    worker.run_once(run_id=run1, handler=recording_handler)
    worker.run_once(run_id=run2, handler=recording_handler)

    assert len(captured) == 2
    assert captured[0] == captured[1], (
        f"Same seed should produce same random value; got {captured}"
    )


def test_worker_different_seeds_produce_different_random_state():
    """Different seeds in different runs should produce different random state."""
    from finetuneharness.executor.worker import LocalWorker

    store = InMemoryStateStore()
    runner = FineTuneRunner(store)

    captured: list[float] = []

    def recording_handler(task):
        captured.append(random.random())
        return {}

    config_seed1 = {**_BASE_CONFIG, "seed": 1}
    config_seed2 = {**_BASE_CONFIG, "seed": 2}

    run1 = runner.create_run(name="r1", config=config_seed1, tasks=[{"task_key": "a"}])
    run2 = runner.create_run(name="r2", config=config_seed2, tasks=[{"task_key": "b"}])

    worker = LocalWorker(worker_id="w", store=store, runner=runner)
    worker.run_once(run_id=run1, handler=recording_handler)
    worker.run_once(run_id=run2, handler=recording_handler)

    assert len(captured) == 2
    assert captured[0] != captured[1], (
        "Different seeds should produce different random states"
    )


# ── Q5 bug fix: comparability uses dataset_hashes field ──────────────────────

def test_comparability_detects_mismatch_via_datasets_dict_form():
    """Runs created with the 'datasets' dict form (not 'dataset_hash') should
    still trigger a dataset_hashes ERROR if the data differs."""
    from finetuneharness.evaluation.comparator import check_comparability

    store = InMemoryStateStore()
    runner = FineTuneRunner(store)

    config_a = {
        "project": {"name": "p"}, "executor": {"kind": "local"}, "artifacts": {"root": "."},
        "seed": 42,
        "datasets": {"train": "sha256:aaa", "test": "sha256:bbb"},
    }
    config_b = {
        "project": {"name": "p"}, "executor": {"kind": "local"}, "artifacts": {"root": "."},
        "seed": 42,
        "datasets": {"train": "sha256:ccc", "test": "sha256:ddd"},  # different data
    }
    run_id_a = runner.create_run(name="a", config=config_a, tasks=[])
    run_id_b = runner.create_run(name="b", config=config_b, tasks=[])

    run_a = store.get_run(run_id_a)
    run_b = store.get_run(run_id_b)

    issues = check_comparability(run_a, run_b)

    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 1
    assert errors[0].field == "dataset_hashes"
    assert "dataset_hash" in errors[0].message


def test_comparability_no_error_when_dataset_hashes_match():
    """Identical dataset_hashes (same data, same splits) should not produce an error."""
    from finetuneharness.evaluation.comparator import check_comparability

    store = InMemoryStateStore()
    runner = FineTuneRunner(store)

    same_config = {
        "project": {"name": "p"}, "executor": {"kind": "local"}, "artifacts": {"root": "."},
        "seed": 42,
        "dataset_hash": "sha256:abc123",
    }
    run_id_a = runner.create_run(name="a", config=same_config, tasks=[])
    run_id_b = runner.create_run(name="b", config=same_config, tasks=[])

    run_a = store.get_run(run_id_a)
    run_b = store.get_run(run_id_b)

    issues = check_comparability(run_a, run_b)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == []
