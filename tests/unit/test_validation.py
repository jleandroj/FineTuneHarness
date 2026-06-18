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
