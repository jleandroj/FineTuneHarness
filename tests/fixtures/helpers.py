"""Shared test helpers — imported by all test tiers."""
from __future__ import annotations

from typing import Any

from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.store import StateStore

BASE_CONFIG: dict[str, Any] = {
    "project": {"name": "test-project"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
}


def make_run(
    runner: FineTuneRunner,
    name: str = "test-run",
    task_keys: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    return runner.create_run(
        name=name,
        config=config or BASE_CONFIG,
        tasks=[{"task_key": k} for k in (task_keys or ["a"])],
    )


def succeed_task(
    store: StateStore,
    run_id: str,
    task_key: str,
    result: dict[str, Any] | None = None,
) -> None:
    task = next(t for t in store.list_tasks(run_id) if t.task_key == task_key)
    store.update_task_status(task.task_id, TaskStatus.LEASED)
    store.update_task_status(task.task_id, TaskStatus.RUNNING)
    store.update_task_status(task.task_id, TaskStatus.SUCCEEDED, result=result or {})


def fail_task(
    store: StateStore,
    run_id: str,
    task_key: str,
    error: str = "test failure",
) -> None:
    task = next(t for t in store.list_tasks(run_id) if t.task_key == task_key)
    store.update_task_status(task.task_id, TaskStatus.LEASED)
    store.update_task_status(task.task_id, TaskStatus.RUNNING)
    store.update_task_status(task.task_id, TaskStatus.FAILED, error=error)


def make_run_with_results(
    runner: FineTuneRunner,
    store: StateStore,
    name: str,
    outcomes: dict[str, dict[str, Any] | None],
) -> str:
    """Create a run and drive all tasks to SUCCEEDED or FAILED.

    outcomes maps task_key -> result dict (or None to mark as failed).
    """
    run_id = make_run(runner, name=name, task_keys=list(outcomes.keys()))
    for task_key, result in outcomes.items():
        if result is not None:
            succeed_task(store, run_id, task_key, result=result)
        else:
            fail_task(store, run_id, task_key)
    return run_id


def run_all_tasks(
    store: StateStore,
    run_id: str,
    handler: Any = None,
) -> None:
    """Drive every pending task in a run to SUCCEEDED using the given handler."""
    from finetuneharness.executor.worker import LocalWorker

    worker = LocalWorker(worker_id="test-worker", store=store)
    fn = handler or (lambda task: {"done": task.task_key})
    tasks = store.list_tasks(run_id)
    for _ in tasks:
        result = worker.run_once(run_id=run_id, handler=fn)
        if result is None:
            break
