from __future__ import annotations

import threading
from typing import Any, Callable


HookFn = Callable[..., None]

VALID_POINTS = frozenset({
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

    def fire(self, point: str, **kwargs: Any) -> None:
        with self._lock:
            fns = list(self._hooks.get(point, []))
        for fn in fns:
            try:
                fn(**kwargs)
            except Exception:
                pass
