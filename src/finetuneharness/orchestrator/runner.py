from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import asdict
from typing import Any

from finetuneharness.observability.logging import get_logger
from finetuneharness.orchestrator.lifecycle import ensure_run_transition
from finetuneharness.state.models import EventRecord, RunRecord, RunStatus, TaskRecord, TaskStatus
from finetuneharness.state.store import StateStore
from finetuneharness.validation.configs import validate_run_config


class FineTuneRunner:
    """Bootstrap runner with explicit state transitions and status aggregation."""

    def __init__(self, store: StateStore) -> None:
        self._store = store
        self._log = get_logger("finetuneharness.runner")

    def create_run(self, *, name: str, config: dict[str, Any], tasks: list[dict[str, Any]]) -> str:
        validate_run_config(config)
        run_id = uuid.uuid4().hex
        run = RunRecord(run_id=run_id, name=name, status=RunStatus.CREATED, config=config)
        self._store.create_run(run)
        self._store.append_event(EventRecord(event_id=uuid.uuid4().hex, run_id=run_id, task_id=None, kind="run_created", payload={"name": name}))

        ensure_run_transition(RunStatus.CREATED, RunStatus.VALIDATED)
        self._store.update_run_status(run_id, RunStatus.VALIDATED)
        self._store.append_event(EventRecord(event_id=uuid.uuid4().hex, run_id=run_id, task_id=None, kind="run_validated", payload={}))

        for index, payload in enumerate(tasks):
            task_id = uuid.uuid4().hex
            task = TaskRecord(
                task_id=task_id,
                run_id=run_id,
                task_key=payload.get("task_key", f"task-{index}"),
                status=TaskStatus.PENDING,
                payload=payload,
            )
            self._store.create_task(task)
            self._store.append_event(EventRecord(event_id=uuid.uuid4().hex, run_id=run_id, task_id=task_id, kind="task_created", payload=payload))

        self._log.info("run_created", extra={"run_id": run_id, "task_count": len(tasks), "config": asdict(run)})
        return run_id

    def refresh_run_status(self, run_id: str) -> RunStatus:
        run = self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run_id: {run_id}")
        tasks = self._store.list_tasks(run_id)
        if not tasks:
            target = RunStatus.VALIDATED
        else:
            statuses = {task.status for task in tasks}
            attempted = any(task.attempt_count > 0 for task in tasks)
            if all(status is TaskStatus.SUCCEEDED for status in statuses):
                target = RunStatus.COMPLETED
            elif TaskStatus.FAILED in statuses and (TaskStatus.SUCCEEDED in statuses or TaskStatus.PENDING in statuses or TaskStatus.LEASED in statuses or TaskStatus.RUNNING in statuses):
                target = RunStatus.PARTIAL_FAILED
            elif TaskStatus.TIMED_OUT in statuses and (TaskStatus.SUCCEEDED in statuses or TaskStatus.PENDING in statuses or TaskStatus.LEASED in statuses or TaskStatus.RUNNING in statuses):
                target = RunStatus.PARTIAL_FAILED
            elif statuses.issubset({TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT}):
                target = RunStatus.FAILED
            elif TaskStatus.RUNNING in statuses or TaskStatus.LEASED in statuses or TaskStatus.SUCCEEDED in statuses or attempted:
                target = RunStatus.RUNNING
            else:
                target = RunStatus.VALIDATED

        if target != run.status:
            ensure_run_transition(run.status, target)
            self._store.update_run_status(run_id, target)
            self._store.append_event(
                EventRecord(
                    event_id=uuid.uuid4().hex,
                    run_id=run_id,
                    task_id=None,
                    kind="run_status_changed",
                    payload={"from": run.status.value, "to": target.value},
                )
            )
        return target

    def get_run_status(self, run_id: str) -> dict[str, object]:
        status = self.refresh_run_status(run_id)
        run = self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run_id: {run_id}")
        tasks = self._store.list_tasks(run_id)
        counts = Counter(task.status.value for task in tasks)
        return {
            "run_id": run.run_id,
            "name": run.name,
            "status": status.value,
            "task_counts": dict(sorted(counts.items())),
            "task_total": len(tasks),
        }
