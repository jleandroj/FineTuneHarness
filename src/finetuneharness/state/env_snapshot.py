"""Best-effort environment snapshot captured at run creation time.

Never raises: missing tools (git, torch, CUDA) are silently omitted.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def capture_env_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    snap["packages"] = _installed_packages([
        "torch", "transformers", "peft", "datasets", "tokenizers",
        "accelerate", "bitsandbytes", "evaluate", "scikit-learn",
        "finetuneharness",
    ])

    snap.update(_git_info())

    cuda = _cuda_info()
    if cuda:
        snap["cuda"] = cuda

    return snap


def _installed_packages(names: list[str]) -> dict[str, str]:
    try:
        from importlib.metadata import packages_distributions, version, PackageNotFoundError
    except ImportError:
        return {}
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except Exception:
            pass
    return result


def _git_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["git_commit"] = commit
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) != 0
        info["git_dirty"] = dirty
    except Exception:
        pass
    return info


def _cuda_info() -> dict[str, Any] | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return {
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        }
    except Exception:
        return None
