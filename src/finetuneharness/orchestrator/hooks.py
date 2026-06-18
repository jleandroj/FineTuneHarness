from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from finetuneharness.observability.logging import get_logger

HookFn = Callable[..., None]

_log = get_logger(__name__)

VALID_POINTS = frozenset({
    "before_run_start",
    "before_task",
    "after_task_success",
    "after_task_failure",
    "after_task_timeout",
    "on_run_status_changed",
})


class HookRegistry:
    """Registry for lifecycle hooks fired around task and run events.

    Hooks are fire-and-forget: a hook that raises does not propagate the error
    to the worker, so hooks cannot crash the harness.

    Valid hook points and their kwargs:
      before_task(task: TaskRecord)
      after_task_success(task: TaskRecord, result: dict)
      after_task_failure(task: TaskRecord, error: Exception)
      after_task_timeout(task: TaskRecord)
      on_run_status_changed(run_id: str, status: RunStatus)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hooks: dict[str, list[HookFn]] = {}

    def register(self, point: str, fn: HookFn) -> None:
        if point not in VALID_POINTS:
            raise ValueError(f"unknown hook point: {point!r}. valid: {sorted(VALID_POINTS)}")
        with self._lock:
            self._hooks.setdefault(point, []).append(fn)

    def hook_names(self, point: str) -> list[str]:
        """Return qualnames of hooks registered at *point* (for event auditing)."""
        with self._lock:
            return [getattr(fn, "__qualname__", repr(fn)) for fn in self._hooks.get(point, [])]

    def fire(self, point: str, **kwargs: Any) -> None:
        with self._lock:
            fns = list(self._hooks.get(point, []))
        for fn in fns:
            try:
                fn(**kwargs)
            except Exception as exc:
                _log.warning(
                    "hook_error",
                    extra={
                        "point": point,
                        "hook": getattr(fn, "__qualname__", repr(fn)),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
