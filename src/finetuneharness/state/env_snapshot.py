"""Best-effort environment snapshot captured at run creation time.

Never raises: missing tools (git, torch, nvidia-smi) are silently omitted.
Fields that cannot be determined are absent from the returned dict.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def capture_env_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    # Narrow except: only PackageNotFoundError (editable/uninstalled package) maps
    # to "dev". Any other error from importlib.metadata is unexpected and should
    # surface rather than be silently masked as a version of "dev".
    from importlib.metadata import version as _meta_version, PackageNotFoundError
    try:
        snap["harness_version"] = _meta_version("finetuneharness")
    except PackageNotFoundError:
        snap["harness_version"] = "dev"

    snap["packages"] = _installed_packages([
        # Core ML
        "torch", "torchvision", "torchaudio",
        # Transformers ecosystem
        "transformers", "tokenizers", "accelerate", "peft", "trl",
        "datasets", "evaluate",
        # Quantization / efficiency
        "bitsandbytes", "flash-attn", "einops",
        # Numerics
        "numpy", "scipy",
        # Serialization
        "safetensors", "sentencepiece", "protobuf",
        # Tracking
        "wandb", "mlflow",
        # Infra
        "scikit-learn", "finetuneharness",
    ])

    snap.update(_git_info())

    container = _container_info()
    if container:
        snap["container"] = container

    hardware = _hardware_info()
    if hardware:
        snap["hardware"] = hardware

    cuda = _cuda_info()
    if cuda:
        snap["cuda"] = cuda

    snap["determinism_env"] = _determinism_env()

    return snap


def _installed_packages(names: list[str]) -> dict[str, str]:
    try:
        from importlib.metadata import version
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

        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            info["git_branch"] = branch
        except Exception:
            pass

        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) != 0
        info["git_dirty"] = dirty

        if dirty:
            try:
                diff_bytes = subprocess.check_output(
                    ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL
                )
                info["git_diff_hash"] = hashlib.sha256(diff_bytes).hexdigest()
            except Exception:
                pass

    except Exception:
        pass
    return info


def _container_info() -> dict[str, Any] | None:
    """Best-effort container detection. Returns None if not in a container."""
    info: dict[str, Any] = {}

    # Docker: /.dockerenv file exists inside containers
    if Path("/.dockerenv").exists():
        info["engine"] = "docker"
        # Try to read image digest from labels (set at image build time)
        digest = os.environ.get("IMAGE_DIGEST") or os.environ.get("DOCKER_IMAGE_DIGEST")
        if digest:
            info["digest"] = digest
        image = os.environ.get("IMAGE_NAME") or os.environ.get("DOCKER_IMAGE")
        if image:
            info["image"] = image
        return info

    # Apptainer / Singularity
    for var in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        val = os.environ.get(var)
        if val:
            info["engine"] = "apptainer"
            info["image"] = val
            digest = os.environ.get("APPTAINER_IMAGE_DIGEST") or os.environ.get("SINGULARITY_IMAGE_DIGEST")
            if digest:
                info["digest"] = digest
            return info

    # Not in a recognized container — return sentinel so assess_reproducibility
    # can flag the missing container digest
    return {"engine": "none"}


def _hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        info["hostname"] = platform.node()
        cpu = platform.processor() or platform.machine()
        if cpu:
            info["cpu"] = cpu
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            info["memory_gb"] = round(page * pages / (1024 ** 3), 1)
        except Exception:
            pass
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            # nvidia-smi driver version
            try:
                driver = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL, text=True,
                ).strip().splitlines()[0].strip()
                info["gpu_driver_version"] = driver
            except Exception:
                pass
    except Exception:
        pass

    return info


def _determinism_env() -> dict[str, str | None]:
    """Capture env vars that control GPU/CPU determinism at run time."""
    vars_ = ("CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING", "PYTHONHASHSEED")
    return {v: os.environ.get(v) for v in vars_}


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
