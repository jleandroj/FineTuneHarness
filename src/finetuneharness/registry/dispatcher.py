"""TaskDispatcher: routes tasks to handlers by payload["kind"].

The harness core (state/, executor/, orchestrator/) never changes when
a new task kind is added. Only the dispatcher and its tests change.

Usage:
    registry = TaskDispatcher()
    registry.register("distill", run_distillation)
    registry.register("prune",   run_pruning)

    worker.run_once(run_id=run_id, handler=registry.dispatch)
"""
from __future__ import annotations

from typing import Any, Callable

from finetuneharness.state.models import TaskRecord

TaskFn = Callable[[TaskRecord], dict[str, Any]]


def validate_task_payload(task: TaskRecord) -> None:
    """Validate that task.payload contains a non-empty string 'kind'.

    Raises ValueError with a clear message on any violation.
    Called by TaskDispatcher.dispatch() before routing.
    """
    if "kind" not in task.payload:
        raise ValueError(
            f"task {task.task_id!r} payload is missing required field 'kind'. "
            f"Got keys: {sorted(task.payload.keys())}"
        )
    kind = task.payload["kind"]
    if not isinstance(kind, str):
        raise ValueError(
            f"task {task.task_id!r} payload['kind'] must be a str, "
            f"got {type(kind).__name__}"
        )
    if not kind.strip():
        raise ValueError(
            f"task {task.task_id!r} payload['kind'] must be non-empty"
        )


class TaskDispatcher:
    """Maps task.payload['kind'] → handler function.

    Adding a new task type only requires:
      1. Write run_new_task(task) -> dict
      2. dispatcher.register("new_kind", run_new_task)
      3. Add tests

    Nothing in state/, executor/, or orchestrator/ changes.
    """

    def __init__(self) -> None:
        self._registry: dict[str, TaskFn] = {}

    def register(self, kind: str, fn: TaskFn) -> None:
        """Register a handler for *kind*. Raises if kind already registered."""
        if not kind or not isinstance(kind, str):
            raise ValueError(f"kind must be a non-empty str, got {kind!r}")
        if kind in self._registry:
            raise ValueError(f"kind {kind!r} is already registered")
        self._registry[kind] = fn

    def kinds(self) -> list[str]:
        return sorted(self._registry)

    def dispatch(self, task: TaskRecord) -> dict[str, Any]:
        """Validate payload and route to the registered handler.

        This is the callable passed to worker.run_once(handler=dispatcher.dispatch).
        """
        validate_task_payload(task)
        kind = task.payload["kind"]
        fn = self._registry.get(kind)
        if fn is None:
            raise ValueError(
                f"No handler registered for task kind {kind!r}. "
                f"Registered kinds: {self.kinds()}"
            )
        return fn(task)
