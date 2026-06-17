from __future__ import annotations

from finetuneharness.observability.logging import get_logger
from finetuneharness.state.models import TaskRecord
from finetuneharness.state.store import StateStore


class TaskScheduler:
    """Thin bootstrap scheduler for leasing tasks to workers."""

    def __init__(self, store: StateStore) -> None:
        self._store = store
        self._log = get_logger("finetuneharness.scheduler")

    def lease_next_task(self, *, run_id: str, worker_id: str, lease_seconds: int = 300) -> TaskRecord | None:
        self._store.requeue_expired_leases(run_id=run_id)
        task = self._store.lease_next_pending_task(run_id=run_id, worker_id=worker_id, lease_seconds=lease_seconds)
        if task is not None:
            self._log.info("task_leased", extra={"run_id": run_id, "task_id": task.task_id})
        return task
