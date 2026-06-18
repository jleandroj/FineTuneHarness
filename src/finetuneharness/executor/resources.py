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

# Substrings that mark specifically a GPU/CUDA out-of-memory condition. NOTE: the
# bare phrase "out of memory" is deliberately excluded — a *system RAM* OOM also
# contains it, and lowering GPU concurrency does not help a host-RAM OOM. Each
# marker here is CUDA/cuBLAS/cuDNN-specific. "outofmemoryerror" covers the
# FirejailSandbox case where torch's class name is flattened into a string.
_OOM_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "outofmemoryerror",
)


def is_oom_error(exc: BaseException) -> bool:
    """True if *exc* looks like a GPU/CUDA out-of-memory error.

    Priority is the exception *type*. If torch is importable we anchor on the real
    ``torch.cuda.OutOfMemoryError`` via ``isinstance`` (robust to subclasses); we
    also match by class name so detection works without importing torch and across
    the FirejailSandbox flattening ("...: OutOfMemoryError: ..."). Only then fall
    back to CUDA-specific message markers — deliberately NOT the bare "out of
    memory", which a system RAM OOM also carries (GPU backoff would not help that).
    """
    try:
        import torch  # noqa: PLC0415 — optional, only when present

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
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
    settle_seconds: float = 1.0
    max_oom_retries: int = 5

    VALID_MODES = ("sequential", "resource_aware")

    @property
    def is_resource_aware(self) -> bool:
        return self.mode == "resource_aware"


@runtime_checkable
class ResourceMonitor(Protocol):
    def free_memory_per_device(self) -> list[float] | None:
        """Free MB for each physical GPU (index-aligned), or None when no GPU is
        visible/detectable. A None return tells the scheduler to fall back to
        sequential draining — there is no resource signal to gate on (e.g. CPU CI).
        """
        ...

    def free_gpu_memory_mb(self) -> float | None:
        """Free MB on the most-constrained device (min across devices), or None."""
        ...


class NvmlMonitor:
    """Free-GPU-memory probe via NVML (pynvml), falling back to the nvidia-smi CLI.

    Reports the MINIMUM free memory across visible GPUs. Returns None when no GPU
    is present, so callers degrade to sequential rather than crash.

    SCOPE — single-GPU only. Both NVML and nvidia-smi report *physical* devices and
    do not honor ``CUDA_VISIBLE_DEVICES``, and drain_concurrent does NOT pin child
    processes to specific GPUs (every child uses the default device, cuda:0). On a
    multi-GPU host this means the gate may measure a different GPU than the tasks
    use — under-admitting (if another GPU is full) or over-admitting into cuda:0
    (if other GPUs are free), and tasks are never distributed across GPUs. The
    monitor logs a warning once if it sees >1 device. Proper multi-GPU support
    (per-child pinning + per-device measurement) is not implemented.
    """

    def __init__(self) -> None:
        self._log = get_logger("finetuneharness.resources")
        self._warned_multigpu = False
        self._pynvml = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
        except Exception:  # pynvml missing, no driver, or init failed
            self._pynvml = None

    def _warn_if_multigpu(self, count: int) -> None:
        if count > 1 and not self._warned_multigpu:
            self._warned_multigpu = True
            self._log.warning(
                "multi_gpu_not_supported",
                extra={
                    "gpu_count": count,
                    "detail": "resource_aware concurrency assumes a single GPU; it "
                              "gates on the minimum free across all devices and does "
                              "not pin children per GPU. Use sequential mode or one "
                              "run process per GPU (CUDA_VISIBLE_DEVICES).",
                },
            )

    def free_memory_per_device(self) -> list[float] | None:
        if self._pynvml is not None:
            try:
                p = self._pynvml
                count = p.nvmlDeviceGetCount()
                if count == 0:
                    return None
                self._warn_if_multigpu(count)
                frees = []
                for i in range(count):
                    handle = p.nvmlDeviceGetHandleByIndex(i)
                    mem = p.nvmlDeviceGetMemoryInfo(handle)
                    frees.append(mem.free / (1024 * 1024))
                return frees or None
            except Exception:
                self._log.warning("nvml_query_failed_falling_back_to_smi")
        return self._nvidia_smi_per_device()

    def free_gpu_memory_mb(self) -> float | None:
        per = self.free_memory_per_device()
        return min(per) if per else None

    def _nvidia_smi_per_device(self) -> list[float] | None:
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
        self._warn_if_multigpu(len(vals))
        return vals or None
