"""Resource probing for the resource-aware concurrency scheduler.

The harness has no fixed worker count: a single ``finetuneharness run`` can drain
several tasks concurrently, admitting a new one only while the GPU has memory to
spare. This module provides:

  * ``ConcurrencyConfig`` — the knobs that govern admission.
  * ``ResourceMonitor`` — the protocol the scheduler queries for free GPU memory.
  * ``NvmlMonitor`` — the production monitor (pynvml, falling back to nvidia-smi).
  * ``is_oom_error`` — classifies a handler exception as a GPU out-of-memory event
    so the scheduler can requeue + back off instead of failing the task.

"measure-and-estimate" model: there is no per-task memory declaration. The monitor
reports *currently free* GPU memory; the scheduler admits tasks while that stays
above ``min_free_mb``. This cannot know a task's peak footprint before it runs, so
an OOM is still possible — it is handled by requeue + concurrency backoff, not
prevented.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from finetuneharness.observability.logging import get_logger

# Substrings that mark a GPU out-of-memory condition across torch / CUDA / cuBLAS.
# "outofmemoryerror" covers the FirejailSandbox case where the original exception
# class name is flattened into a RuntimeError message string.
_OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "outofmemoryerror",
)


def is_oom_error(exc: BaseException) -> bool:
    """True if *exc* looks like a GPU/CUDA out-of-memory error.

    Detects ``torch.cuda.OutOfMemoryError`` by class name (without importing
    torch) and the common CUDA OOM messages carried on a ``RuntimeError`` —
    including the FirejailSandbox case where the original type is flattened into
    a ``RuntimeError`` string ("handler raised in sandbox: OutOfMemoryError: ...").
    """
    if type(exc).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _OOM_MARKERS)


@dataclass(frozen=True)
class ConcurrencyConfig:
    """Admission knobs for resource-aware draining.

    mode:           "sequential" (one task at a time) or "resource_aware".
    min_free_mb:    do not admit a *further* task while free GPU memory is below
                    this headroom. The first task is always admitted (otherwise an
                    already-busy GPU would deadlock the run).
    max_concurrent: hard ceiling on in-flight tasks regardless of free memory.
    settle_seconds: pause after admitting a task before re-reading memory, so the
                    new task's allocation is reflected (mitigates the cold-start
                    race where an empty GPU admits everything at once).
    max_oom_retries: how many times a single task may be requeued after OOM before
                    it is marked FAILED for good.
    """

    mode: str = "sequential"
    min_free_mb: float = 2000.0
    max_concurrent: int = 8
    settle_seconds: float = 5.0
    max_oom_retries: int = 5

    VALID_MODES = ("sequential", "resource_aware")

    @property
    def is_resource_aware(self) -> bool:
        return self.mode == "resource_aware"


@runtime_checkable
class ResourceMonitor(Protocol):
    def free_gpu_memory_mb(self) -> float | None:
        """Free GPU memory in MB, or None when no GPU is visible/detectable.

        A None return tells the scheduler to fall back to sequential draining —
        there is no resource signal to gate concurrency on (e.g. CPU-only CI).
        """
        ...


class NvmlMonitor:
    """Free-GPU-memory probe via NVML (pynvml), falling back to the nvidia-smi CLI.

    Reports the MINIMUM free memory across visible GPUs — the binding constraint
    for admitting another task. Returns None when no GPU is present, so callers
    degrade to sequential rather than crash.

    Note: both NVML and nvidia-smi report *physical* devices and do not honor
    ``CUDA_VISIBLE_DEVICES``. On a single-GPU host (the common case) this is exact;
    on multi-GPU hosts with pinned processes the minimum-across-all reading is a
    conservative lower bound, which is the safe direction for admission control.
    """

    def __init__(self) -> None:
        self._log = get_logger("finetuneharness.resources")
        self._pynvml = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
        except Exception:  # pynvml missing, no driver, or init failed
            self._pynvml = None

    def free_gpu_memory_mb(self) -> float | None:
        if self._pynvml is not None:
            try:
                p = self._pynvml
                count = p.nvmlDeviceGetCount()
                if count == 0:
                    return None
                frees = []
                for i in range(count):
                    handle = p.nvmlDeviceGetHandleByIndex(i)
                    mem = p.nvmlDeviceGetMemoryInfo(handle)
                    frees.append(mem.free / (1024 * 1024))
                return min(frees) if frees else None
            except Exception:
                self._log.warning("nvml_query_failed_falling_back_to_smi")
        return self._nvidia_smi_free_mb()

    def _nvidia_smi_free_mb(self) -> float | None:
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return None
        if out.returncode != 0:
            return None
        vals = []
        for line in out.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                continue
        return min(vals) if vals else None
