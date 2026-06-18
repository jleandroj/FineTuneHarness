from __future__ import annotations

import warnings
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = frozenset({"project", "executor", "artifacts"})

_VALID_CONCURRENCY_MODES = ("sequential", "resource_aware")


def validate_run_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("run config must be a dict")
    missing = REQUIRED_TOP_LEVEL_KEYS.difference(config)
    if missing:
        raise ValueError(f"invalid run config: missing keys: {', '.join(sorted(missing))}")

    project = config.get("project")
    if not isinstance(project, dict):
        raise ValueError("run config 'project' must be a dict")
    if not isinstance(project.get("name"), str) or not project.get("name"):
        raise ValueError("run config 'project.name' must be a non-empty string")

    executor = config.get("executor")
    if not isinstance(executor, dict):
        raise ValueError("run config 'executor' must be a dict")
    # 'max_workers' is dead: it never sized any pool. Concurrency is governed at
    # runtime by 'executor.concurrency'. Accept the legacy field but warn so old
    # configs on disk keep loading instead of hard-failing.
    if "max_workers" in executor:
        warnings.warn(
            "run config 'executor.max_workers' is deprecated and ignored. "
            "Use 'executor.concurrency' (mode/min_free_mb/max_concurrent) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if "concurrency" in executor:
        _validate_concurrency(executor["concurrency"])

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("run config 'artifacts' must be a dict")
    if not isinstance(artifacts.get("root"), str) or not artifacts.get("root"):
        raise ValueError("run config 'artifacts.root' must be a non-empty string")

    # ── Reproducibility fields (required) ────────────────────────────────────
    _validate_seed(config)
    _validate_dataset_hash(config)


def _validate_concurrency(conc: Any) -> None:
    if not isinstance(conc, dict):
        raise ValueError("run config 'executor.concurrency' must be a dict")
    mode = conc.get("mode", "sequential")
    if mode not in _VALID_CONCURRENCY_MODES:
        raise ValueError(
            f"run config 'executor.concurrency.mode' must be one of "
            f"{_VALID_CONCURRENCY_MODES}, got {mode!r}"
        )
    for key in ("min_free_mb", "settle_seconds"):
        if key in conc:
            val = conc[key]
            if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
                raise ValueError(
                    f"run config 'executor.concurrency.{key}' must be a non-negative number, "
                    f"got {val!r}"
                )
    if "max_concurrent" in conc:
        val = conc["max_concurrent"]
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise ValueError(
                f"run config 'executor.concurrency.max_concurrent' must be a positive int, "
                f"got {val!r}"
            )
    if "max_oom_retries" in conc:
        val = conc["max_oom_retries"]
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise ValueError(
                f"run config 'executor.concurrency.max_oom_retries' must be a non-negative int, "
                f"got {val!r}"
            )


def _validate_seed(config: dict[str, Any]) -> None:
    seed = config.get("seed")
    if seed is None:
        raise ValueError(
            "run config missing required field 'seed' (int). "
            "Set a deterministic seed to enable reproducibility. "
            "Example: {\"seed\": 42}"
        )
    # bool is a subclass of int — reject it explicitly so True/False can't pose as
    # a seed (it would seed every RNG with 0/1 and read as a deterministic seed).
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"run config 'seed' must be an int, got {type(seed).__name__!r}. "
            "Example: {\"seed\": 42}"
        )


def _validate_dataset_hash(config: dict[str, Any]) -> None:
    dataset_hash = config.get("dataset_hash")
    datasets = config.get("datasets")

    if dataset_hash is None and not datasets:
        raise ValueError(
            "run config missing required reproducibility field. "
            "Provide either:\n"
            "  'dataset_hash': 'sha256:<hex>' (single dataset), or\n"
            "  'datasets': {'train': 'sha256:<hex>', 'validation': 'sha256:<hex>', ...}"
        )

    if dataset_hash is not None:
        if not isinstance(dataset_hash, str) or not dataset_hash.strip():
            raise ValueError(
                "run config 'dataset_hash' must be a non-empty string "
                "(e.g. 'sha256:abc123...' or a stable identifier for your data)"
            )

    if datasets is not None:
        if not isinstance(datasets, dict):
            raise ValueError(
                "run config 'datasets' must be a dict mapping split names to hash strings"
            )
        if not datasets:
            raise ValueError("run config 'datasets' must not be empty")
        for k, v in datasets.items():
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f"run config 'datasets.{k}' must be a non-empty string hash"
                )
