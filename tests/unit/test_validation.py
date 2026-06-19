"""Tests for validate_run_config covering all validation rules."""
from __future__ import annotations

import pytest

from finetuneharness.validation.configs import validate_run_config


_VALID = {
    "project": {"name": "my-project"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:abc123",
}


def test_valid_config_passes() -> None:
    validate_run_config(_VALID)


def test_missing_top_level_key_raises() -> None:
    cfg = {k: v for k, v in _VALID.items() if k != "artifacts"}
    with pytest.raises(ValueError, match="missing keys"):
        validate_run_config(cfg)


def test_non_dict_raises() -> None:
    with pytest.raises(ValueError):
        validate_run_config("not a dict")  # type: ignore[arg-type]


def test_project_not_dict_raises() -> None:
    cfg = {**_VALID, "project": "bad"}
    with pytest.raises(ValueError, match="project.*dict"):
        validate_run_config(cfg)


def test_project_name_missing_raises() -> None:
    cfg = {**_VALID, "project": {"description": "no name"}}
    with pytest.raises(ValueError, match="project.name"):
        validate_run_config(cfg)


def test_project_name_empty_raises() -> None:
    cfg = {**_VALID, "project": {"name": ""}}
    with pytest.raises(ValueError, match="project.name"):
        validate_run_config(cfg)


def test_executor_not_dict_raises() -> None:
    cfg = {**_VALID, "executor": None}
    with pytest.raises(ValueError, match="executor.*dict"):
        validate_run_config(cfg)


def test_artifacts_not_dict_raises() -> None:
    cfg = {**_VALID, "artifacts": 42}
    with pytest.raises(ValueError, match="artifacts.*dict"):
        validate_run_config(cfg)


def test_artifacts_root_missing_raises() -> None:
    cfg = {**_VALID, "artifacts": {"backend": "s3"}}
    with pytest.raises(ValueError, match="artifacts.root"):
        validate_run_config(cfg)


def test_artifacts_root_empty_raises() -> None:
    cfg = {**_VALID, "artifacts": {"root": ""}}
    with pytest.raises(ValueError, match="artifacts.root"):
        validate_run_config(cfg)


def test_executor_max_workers_is_deprecated_not_fatal() -> None:
    """Legacy max_workers no longer sizes anything: warn, never raise.

    Old configs on disk must keep loading; the value is simply ignored in favor
    of executor.concurrency.
    """
    cfg = {**_VALID, "executor": {"kind": "local", "max_workers": 8}}
    with pytest.warns(DeprecationWarning, match="max_workers"):
        validate_run_config(cfg)

    # Even nonsensical legacy values must not raise — the field is dead.
    for bad in (0, -2, "8"):
        cfg = {**_VALID, "executor": {"kind": "local", "max_workers": bad}}
        with pytest.warns(DeprecationWarning):
            validate_run_config(cfg)


def test_seed_bool_rejected() -> None:
    """bool is a subclass of int but must not pose as a seed."""
    for bad in (True, False):
        cfg = {**_VALID, "seed": bad}
        with pytest.raises(ValueError, match="seed"):
            validate_run_config(cfg)


def test_executor_concurrency_valid_passes() -> None:
    cfg = {**_VALID, "executor": {"kind": "local", "concurrency": {
        "mode": "resource_aware", "min_free_mb": 2000, "max_concurrent": 4,
        "settle_seconds": 5, "max_oom_retries": 3,
    }}}
    validate_run_config(cfg)  # must not raise


def test_executor_concurrency_absent_passes() -> None:
    cfg = {**_VALID, "executor": {"kind": "local"}}
    validate_run_config(cfg)


def test_executor_concurrency_bad_mode_raises() -> None:
    cfg = {**_VALID, "executor": {"kind": "local", "concurrency": {"mode": "turbo"}}}
    with pytest.raises(ValueError, match="concurrency.mode"):
        validate_run_config(cfg)


def test_executor_concurrency_zero_max_concurrent_raises() -> None:
    cfg = {**_VALID, "executor": {"kind": "local", "concurrency": {"max_concurrent": 0}}}
    with pytest.raises(ValueError, match="max_concurrent"):
        validate_run_config(cfg)


def test_executor_concurrency_negative_min_free_raises() -> None:
    cfg = {**_VALID, "executor": {"kind": "local", "concurrency": {"min_free_mb": -1}}}
    with pytest.raises(ValueError, match="min_free_mb"):
        validate_run_config(cfg)


def test_executor_concurrency_utilization_admission_passes() -> None:
    cfg = {**_VALID, "executor": {"kind": "local", "concurrency": {
        "mode": "resource_aware", "admission": "utilization", "max_util_pct": 95,
        "max_concurrent": 32,
    }}}
    validate_run_config(cfg)  # must not raise


def test_executor_concurrency_bad_admission_raises() -> None:
    cfg = {**_VALID, "executor": {"kind": "local", "concurrency": {"admission": "psychic"}}}
    with pytest.raises(ValueError, match="concurrency.admission"):
        validate_run_config(cfg)


def test_executor_concurrency_max_util_pct_out_of_range_raises() -> None:
    for bad in (0, -5, 101, 150):
        cfg = {**_VALID, "executor": {"kind": "local", "concurrency": {"max_util_pct": bad}}}
        with pytest.raises(ValueError, match="max_util_pct"):
            validate_run_config(cfg)


# ── Regression: actual config files must pass validation ─────────────────────
# The seed was previously nested under experiment.seed (bug). These tests load
# the real JSON files so a future structural change is caught immediately.

import json
from pathlib import Path

_CONFIGS_ROOT = Path(__file__).parents[2] / "configs"


def test_prod_config_passes_validation() -> None:
    """configs/production/prod_config.json must always pass validate_run_config.

    Regression guard: seed was nested under experiment.seed before the fix.
    If anyone moves it back, this test fails before the bug reaches a real run.
    """
    cfg = json.loads((_CONFIGS_ROOT / "production" / "prod_config.json").read_text())
    validate_run_config(cfg)


def test_dev_config_passes_validation() -> None:
    cfg = json.loads((_CONFIGS_ROOT / "local" / "dev_config.json").read_text())
    validate_run_config(cfg)


def test_ablation_config_passes_validation() -> None:
    cfg = json.loads((_CONFIGS_ROOT / "experiments" / "ablation_config.json").read_text())
    validate_run_config(cfg)
