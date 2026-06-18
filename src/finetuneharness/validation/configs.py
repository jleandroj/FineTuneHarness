from __future__ import annotations

from typing import Any


REQUIRED_TOP_LEVEL_KEYS = frozenset({"project", "executor", "artifacts"})


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
    if "max_workers" in executor:
        mw = executor["max_workers"]
        if not isinstance(mw, int) or mw < 1:
            raise ValueError(
                f"run config 'executor.max_workers' must be a positive int, got {mw!r}"
            )

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("run config 'artifacts' must be a dict")
    if not isinstance(artifacts.get("root"), str) or not artifacts.get("root"):
        raise ValueError("run config 'artifacts.root' must be a non-empty string")

    # ── Reproducibility fields (required) ────────────────────────────────────
    _validate_seed(config)
    _validate_dataset_hash(config)


def _validate_seed(config: dict[str, Any]) -> None:
    seed = config.get("seed")
    if seed is None:
        raise ValueError(
            "run config missing required field 'seed' (int). "
            "Set a deterministic seed to enable reproducibility. "
            "Example: {\"seed\": 42}"
        )
    if not isinstance(seed, int):
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
