"""Reproducibility enforcement tests — the 12 cases the user named explicitly.

Each test answers one specific question about whether the harness can
guarantee that a run can be repeated within 3 months.
"""
from __future__ import annotations

import pytest

from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import RunRecord, RunStatus
from finetuneharness.state.reproducibility import (
    ReproducibilityAssessment,
    assess_reproducibility,
    canonical_json_hash,
    export_manifest,
)
from finetuneharness.validation.configs import validate_run_config


_VALID_CONFIG = {
    "project": {"name": "repro-test"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:abc123",
}


def _run_record(**kwargs) -> RunRecord:
    defaults = dict(
        run_id="run-test",
        name="test",
        status=RunStatus.COMPLETED,
        config=_VALID_CONFIG.copy(),
        seed=42,
        dataset_hashes={"default": "sha256:abc123"},
        config_hash="deadbeef",
        env_snapshot={"git_commit": "abcdef1234567890"},
    )
    defaults.update(kwargs)
    return RunRecord(**defaults)


# ── 1. create_run requires seed ───────────────────────────────────────────────

def test_create_run_requires_seed():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    config_no_seed = {k: v for k, v in _VALID_CONFIG.items() if k != "seed"}
    with pytest.raises(ValueError, match="seed"):
        runner.create_run(name="r", config=config_no_seed, tasks=[])


# ── 2. create_run requires dataset_hash ──────────────────────────────────────

def test_create_run_requires_dataset_hash():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    config_no_hash = {k: v for k, v in _VALID_CONFIG.items() if k not in ("dataset_hash", "datasets")}
    with pytest.raises(ValueError, match="dataset_hash"):
        runner.create_run(name="r", config=config_no_hash, tasks=[])


# ── 3. config_hash is stable for same config ─────────────────────────────────

def test_config_hash_is_stable_for_same_config():
    config_a = {"z": 1, "a": 2, "m": [3, 4]}
    config_b = {"a": 2, "m": [3, 4], "z": 1}  # different key order
    assert canonical_json_hash(config_a) == canonical_json_hash(config_b)


# ── 4. config_hash changes when config changes ───────────────────────────────

def test_config_hash_changes_when_config_changes():
    config_a = {"seed": 42, "lr": 1e-4, "epochs": 10}
    config_b = {"seed": 42, "lr": 2e-4, "epochs": 10}  # lr changed
    assert canonical_json_hash(config_a) != canonical_json_hash(config_b)


# ── 5. RunRecord stores seed as first-level field ────────────────────────────

def test_run_record_stores_seed_top_level():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = runner.create_run(name="r", config=_VALID_CONFIG, tasks=[])
    run = store.get_run(run_id)
    assert run is not None
    assert run.seed == 42
    # seed must be accessible directly, not only via run.config["seed"]
    assert isinstance(run.seed, int)


# ── 6. RunRecord stores dataset_hash as first-level field ────────────────────

def test_run_record_stores_dataset_hash_top_level():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = runner.create_run(name="r", config=_VALID_CONFIG, tasks=[])
    run = store.get_run(run_id)
    assert run is not None
    assert run.dataset_hashes == {"default": "sha256:abc123"}
    assert isinstance(run.dataset_hashes, dict)


# ── 7. reproducibility assessment FAIL without seed ──────────────────────────

def test_reproducibility_assessment_fails_without_seed():
    run = _run_record(seed=None)
    result = assess_reproducibility(run)
    assert result.level == "FAIL"
    assert "seed" in result.missing_fields
    assert result.replayable is False


# ── 8. reproducibility assessment FAIL without dataset_hash ──────────────────

def test_reproducibility_assessment_fails_without_dataset_hash():
    run = _run_record(dataset_hashes={})
    result = assess_reproducibility(run)
    assert result.level == "FAIL"
    assert "dataset_hashes" in result.missing_fields
    assert result.replayable is False


# ── 9. reproducibility assessment PARTIAL without container digest ────────────

def test_reproducibility_assessment_partial_without_container_digest():
    run = _run_record(env_snapshot={
        "git_commit": "abcdef12",
        "container": {"engine": "none"},  # no digest
    })
    result = assess_reproducibility(run)
    assert result.level == "PARTIAL"
    assert result.replayable is True
    assert any("container" in w for w in result.warnings)


# ── 10. reproducibility assessment PASS with container digest ─────────────────

def test_reproducibility_assessment_pass_with_container_digest_and_required_hashes():
    run = _run_record(env_snapshot={
        "git_commit": "abcdef12",
        "determinism_env": {"harness_enforces_determinism": True},
        "container": {
            "engine": "docker",
            "image": "my-image:v1.0",
            "digest": "sha256:deadbeef1234",
        },
    })
    result = assess_reproducibility(run)
    assert result.level == "PASS"
    assert result.replayable is True
    assert not result.missing_fields


def test_assessment_degrades_to_partial_without_forced_determinism():
    """Full metadata + container digest, but determinism not forced -> not PASS."""
    run = _run_record(env_snapshot={
        "git_commit": "abcdef12",
        "container": {"engine": "docker", "digest": "sha256:img"},
        # no determinism_env.harness_enforces_determinism
    })
    result = assess_reproducibility(run)
    assert result.level == "PARTIAL"
    assert any("deterministic algorithms were not forced" in w for w in result.warnings)


# ── 11. dirty git state adds warning ──────────────────────────────────────────

def test_dirty_git_state_adds_warning():
    run = _run_record(env_snapshot={
        "git_commit": "abcdef12",
        "git_dirty": True,
        "git_diff_hash": "sha256:dirtydiff",
        "container": {"engine": "docker", "digest": "sha256:img"},
    })
    result = assess_reproducibility(run)
    assert any("dirty" in w for w in result.warnings)


# ── 12. manifest contains required reproducibility fields ─────────────────────

def test_manifest_export_contains_required_reproducibility_fields():
    run = _run_record()
    manifest = export_manifest(run, tasks=[], artifacts=[])

    required = ["run_id", "seed", "dataset_hashes", "config_hash", "config",
                "git", "env_snapshot", "container", "hardware",
                "reproducibility", "tasks", "output_hashes"]
    for field in required:
        assert field in manifest, f"manifest missing required field: {field!r}"

    repro = manifest["reproducibility"]
    assert "level" in repro
    assert "missing_fields" in repro
    assert "warnings" in repro

    git = manifest["git"]
    assert "commit" in git
    assert "dirty" in git


# ── Bonus: datasets dict is accepted instead of dataset_hash ──────────────────

def test_create_run_accepts_datasets_dict():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    config = {**_VALID_CONFIG}
    del config["dataset_hash"]
    config["datasets"] = {
        "train": "sha256:aaa",
        "validation": "sha256:bbb",
        "test": "sha256:ccc",
    }
    run_id = runner.create_run(name="r", config=config, tasks=[])
    run = store.get_run(run_id)
    assert run.dataset_hashes == {"train": "sha256:aaa", "validation": "sha256:bbb", "test": "sha256:ccc"}


def test_create_run_config_hash_is_set():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = runner.create_run(name="r", config=_VALID_CONFIG, tasks=[])
    run = store.get_run(run_id)
    assert run.config_hash is not None
    assert len(run.config_hash) == 64  # SHA-256 hex = 64 chars


# ── harness_version in env_snapshot (P2) ─────────────────────────────────────

def test_env_snapshot_contains_harness_version():
    """harness_version must appear at the top level of the env snapshot.

    A snapshot without this field makes it impossible to know which version of
    the harness produced a historical run, breaking reproducibility audits.
    """
    from finetuneharness.state.env_snapshot import capture_env_snapshot

    snap = capture_env_snapshot()
    assert "harness_version" in snap, (
        "env snapshot is missing 'harness_version' — "
        "historical runs cannot be traced back to a harness release"
    )
    # Value must be a non-empty string: a semver ("1.2.3") or "dev" for editable installs
    v = snap["harness_version"]
    assert isinstance(v, str) and v, f"harness_version must be a non-empty string, got {v!r}"


def test_env_snapshot_harness_version_survives_package_not_found(monkeypatch):
    """When finetuneharness is not installed as a package, harness_version must be 'dev'."""
    import importlib.metadata
    from finetuneharness.state.env_snapshot import capture_env_snapshot

    original_version = importlib.metadata.version

    def _raise_for_harness(name):
        if name == "finetuneharness":
            raise importlib.metadata.PackageNotFoundError(name)
        return original_version(name)

    monkeypatch.setattr(importlib.metadata, "version", _raise_for_harness)
    snap = capture_env_snapshot()

    assert snap["harness_version"] == "dev", (
        "harness_version must fall back to 'dev' when the package is not installed"
    )
