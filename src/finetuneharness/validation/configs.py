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

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("run config 'artifacts' must be a dict")
    if not isinstance(artifacts.get("root"), str) or not artifacts.get("root"):
        raise ValueError("run config 'artifacts.root' must be a non-empty string")
