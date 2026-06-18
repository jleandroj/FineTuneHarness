"""Built-in hooks for fine-tuning workloads.

These hooks provide common ML experiment needs:
- GPU memory monitoring and OOM prevention
- Automatic checkpointing
- Metrics collection and logging
- Early stopping
- Resource cleanup between tasks
"""

from __future__ import annotations

import fcntl
import gc
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetuneharness.observability.logging import get_logger
from finetuneharness.orchestrator.hooks import HookRegistry
from finetuneharness.state.models import RunStatus, TaskRecord, TaskStatus

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_log = get_logger("finetuneharness.hooks")


__all__ = [
    "GPUMemoryHook",
    "CheckpointHook",
    "MetricsHook",
    "EarlyStoppingHook",
    "CleanupHook",
    "ProgressHook",
    "register_default_hooks",
]


@dataclass
class GPUMemoryHook:
    """Monitor GPU memory and warn/cleanup on high usage."""

    threshold_mb: int = 8000  # Warn when allocated > 8GB
    cleanup_on_oom: bool = True

    def before_task(self, task: TaskRecord) -> None:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        gc.collect()
        allocated = torch.cuda.memory_allocated() / 1024**2
        if allocated > self.threshold_mb:
            _log.warning("gpu_memory_high_before_task", extra={
                "task_key": task.task_key, "allocated_mb": round(allocated, 1),
                "threshold_mb": self.threshold_mb,
            })

    def after_task_success(self, task: TaskRecord, result: dict[str, Any]) -> None:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        result["gpu_allocated_mb"] = round(allocated, 1)
        result["gpu_reserved_mb"] = round(reserved, 1)
        if allocated > self.threshold_mb:
            _log.warning("gpu_memory_high_after_task", extra={
                "task_key": task.task_key, "allocated_mb": round(allocated, 1),
                "reserved_mb": round(reserved, 1), "threshold_mb": self.threshold_mb,
            })
        if self.cleanup_on_oom:
            torch.cuda.empty_cache()

    def after_task_failure(self, task: TaskRecord, error: Exception) -> None:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        if "out of memory" in str(error).lower() or "oom" in str(error).lower():
            _log.warning("gpu_oom_detected", extra={"task_key": task.task_key, "error": str(error)})
            torch.cuda.empty_cache()
            gc.collect()


@dataclass
class CheckpointHook:
    """Save checkpoints at regular intervals during long-running tasks.

    Note: This hook requires the handler to support checkpointing via
    a 'checkpoint_dir' in the task payload and return checkpoint paths in results.
    """

    checkpoint_dir: str = ".finetuneharness/checkpoints"
    save_every_n_tasks: int = 1

    def __post_init__(self):
        self._task_count = 0
        self._lock = threading.Lock()
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def before_task(self, task: TaskRecord) -> None:
        with self._lock:
            self._task_count += 1
        if "checkpoint_dir" in task.payload:
            requested = Path(task.payload["checkpoint_dir"]).resolve()
            allowed = Path(self.checkpoint_dir).resolve()
            if not requested.is_relative_to(allowed):
                raise ValueError(
                    f"checkpoint_dir {task.payload['checkpoint_dir']!r} is outside "
                    f"the allowed root {self.checkpoint_dir!r}"
                )
        else:
            task.payload["checkpoint_dir"] = self.checkpoint_dir

    def after_task_success(self, task: TaskRecord, result: dict) -> None:
        # Log checkpoint info if handler returned it
        if "checkpoint_path" in result:
            _log.info("checkpoint_saved", extra={
                "task_key": task.task_key, "checkpoint_path": result["checkpoint_path"],
            })


@dataclass
class MetricsHook:
    """Collect and aggregate metrics across tasks.

    Writes are guarded by both an in-process ``threading.Lock`` and an inter-process
    ``fcntl`` advisory lock, so concurrent workers (threads or separate processes)
    cannot interleave JSONL lines. The file lock is acquired with ``LOCK_NB`` and
    retried until ``lock_timeout_seconds`` so a dead process holding the lock on a
    networked filesystem (NFS) fails loudly with TimeoutError instead of blocking
    the worker forever.
    """

    output_file: str = ".finetuneharness/metrics.jsonl"
    lock_timeout_seconds: float = 10.0

    def __post_init__(self):
        self._metrics: list[dict] = []
        self._lock = threading.Lock()
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

    def _acquire_file_lock(self, f) -> None:
        """Acquire an exclusive advisory lock, retrying until the timeout.

        Raises TimeoutError rather than blocking indefinitely (the LOCK_EX-only
        path could hang forever if a crashed process on NFS still holds the lock).
        """
        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"MetricsHook could not acquire file lock on {self.output_file!r} "
                        f"within {self.lock_timeout_seconds}s — another process may hold it"
                    )
                time.sleep(0.02)

    def after_task_success(self, task: TaskRecord, result: dict) -> None:
        metric_entry = {
            "task_key": task.task_key,
            "task_id": task.task_id,
            "timestamp": time.time(),
            **{k: v for k, v in result.items() if isinstance(v, (int, float, str))},
        }
        with self._lock:
            self._metrics.append(metric_entry)
            with open(self.output_file, "a") as f:
                self._acquire_file_lock(f)
                try:
                    f.write(json.dumps(metric_entry) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def get_summary(self) -> dict[str, Any]:
        """Aggregate from the JSONL file, the cross-process source of truth.

        Reading the file (not the in-memory ``self._metrics``) means the summary is
        correct even when metrics were written by several worker processes — each
        appends to the same JSONL — or after a restart.
        """
        metrics = self._read_metrics_file()
        if not metrics:
            return {"count": 0}
        import statistics

        numeric: dict[str, list[float]] = {}
        for m in metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "timestamp":
                    numeric.setdefault(k, []).append(v)
        summary: dict[str, Any] = {"count": len(metrics)}
        for k, vals in numeric.items():
            summary[k] = {
                "mean": statistics.mean(vals),
                "min": min(vals),
                "max": max(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0,
            }
        return summary

    def _read_metrics_file(self) -> list[dict]:
        path = Path(self.output_file)
        if not path.exists():
            return []
        metrics: list[dict] = []
        with open(path) as f:
            self._acquire_file_lock(f)
            try:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            metrics.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return metrics


@dataclass
class EarlyStoppingHook:
    """Stop run if metric doesn't improve for N consecutive tasks."""

    metric: str = "accuracy"
    patience: int = 5
    min_delta: float = 0.001
    mode: str = "max"  # "max" or "min"

    def __post_init__(self):
        self._best: float | None = None
        self._counter = 0
        self._should_stop = False
        self._lock = threading.Lock()

    def after_task_success(self, task: TaskRecord, result: dict) -> None:
        with self._lock:
            if self.metric not in result:
                return
            value = result[self.metric]
            if self._best is None:
                self._best = value
                self._counter = 0
                return

            improved = (
                (value - self._best) > self.min_delta
                if self.mode == "max"
                else (self._best - value) > self.min_delta
            )

            if improved:
                self._best = value
                self._counter = 0
            else:
                self._counter += 1
                if self._counter >= self.patience:
                    self._should_stop = True
                    _log.warning("early_stopping_triggered", extra={
                        "metric": self.metric, "patience": self.patience,
                        "best": self._best, "task_key": task.task_key,
                    })

    def should_stop(self) -> bool:
        with self._lock:
            return self._should_stop

    def reset(self) -> None:
        with self._lock:
            self._best = None
            self._counter = 0
            self._should_stop = False


@dataclass
class CleanupHook:
    """Release Python and GPU memory between tasks.

    Calls gc.collect() and torch.cuda.empty_cache() (when CUDA is available).
    Gradient zeroing is intentionally NOT performed here: the harness does not
    own model state. Call model.zero_grad() inside your handler instead.
    """

    clear_cuda_cache: bool = True
    gc_collect: bool = True

    def after_task_success(self, task: TaskRecord, result: dict) -> None:
        self._cleanup()

    def after_task_failure(self, task: TaskRecord, error: Exception) -> None:
        self._cleanup()

    def after_task_timeout(self, task: TaskRecord) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self.gc_collect:
            gc.collect()
        if TORCH_AVAILABLE and self.clear_cuda_cache:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@dataclass
class ProgressHook:
    """Log progress and ETA for long-running grids.

    Set ``total_tasks`` to enable percent-complete and ETA in the progress log.
    Without it, only completed-count and throughput are reported (the hook has no
    other way to know the grid size).
    """

    log_every: int = 1
    total_tasks: int | None = None

    def __post_init__(self):
        self._completed = 0
        self._start_time = time.time()
        self._lock = threading.Lock()

    def on_run_status_changed(self, run_id: str, status: RunStatus) -> None:
        if status == RunStatus.COMPLETED:
            elapsed = time.time() - self._start_time
            _log.info("run_completed", extra={
                "run_id": run_id, "elapsed_seconds": round(elapsed, 1),
                "tasks_completed": self._completed,
            })

    def after_task_success(self, task: TaskRecord, result: dict) -> None:
        with self._lock:
            self._completed += 1
            if self._completed % self.log_every == 0:
                elapsed = time.time() - self._start_time
                rate = self._completed / elapsed if elapsed > 0 else 0
                extra: dict[str, Any] = {
                    "tasks_completed": self._completed,
                    "rate_per_second": round(rate, 3),
                    "elapsed_seconds": round(elapsed, 1),
                }
                if self.total_tasks:
                    remaining = max(0, self.total_tasks - self._completed)
                    extra["total_tasks"] = self.total_tasks
                    extra["percent_complete"] = round(
                        100 * self._completed / self.total_tasks, 1
                    )
                    if rate > 0:
                        extra["eta_seconds"] = round(remaining / rate, 1)
                _log.info("progress", extra=extra)


def register_default_hooks(
    registry: HookRegistry,
    *,
    gpu_monitor: bool = True,
    checkpoint: bool = True,
    metrics: bool = True,
    early_stopping: bool = False,
    cleanup: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Register a sensible default set of hooks for fine-tuning.

    Returns the hook instances so they can be queried (e.g., early_stopping.should_stop()).
    """
    hooks: dict[str, Any] = {}

    if gpu_monitor:
        gpu_hook = GPUMemoryHook()
        registry.register("before_task", gpu_hook.before_task)
        registry.register("after_task_success", gpu_hook.after_task_success)
        registry.register("after_task_failure", gpu_hook.after_task_failure)
        hooks["gpu"] = gpu_hook

    if checkpoint:
        ckpt_hook = CheckpointHook()
        registry.register("before_task", ckpt_hook.before_task)
        registry.register("after_task_success", ckpt_hook.after_task_success)
        hooks["checkpoint"] = ckpt_hook

    if metrics:
        metrics_hook = MetricsHook()
        registry.register("after_task_success", metrics_hook.after_task_success)
        hooks["metrics"] = metrics_hook

    if early_stopping:
        es_hook = EarlyStoppingHook()
        registry.register("after_task_success", es_hook.after_task_success)
        hooks["early_stopping"] = es_hook

    if cleanup:
        cleanup_hook = CleanupHook()
        registry.register("after_task_success", cleanup_hook.after_task_success)
        registry.register("after_task_failure", cleanup_hook.after_task_failure)
        registry.register("after_task_timeout", cleanup_hook.after_task_timeout)
        hooks["cleanup"] = cleanup_hook

    if progress:
        progress_hook = ProgressHook()
        registry.register("after_task_success", progress_hook.after_task_success)
        registry.register("on_run_status_changed", progress_hook.on_run_status_changed)
        hooks["progress"] = progress_hook

    return hooks