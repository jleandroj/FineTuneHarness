from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime

from finetuneharness.orchestrator.lifecycle import ensure_task_transition
from finetuneharness.state.leases import Lease, utc_now
from finetuneharness.state.models import ArtifactRecord, EventRecord, RunRecord, RunStatus, TaskRecord, TaskStatus
from finetuneharness.state.store import StateStore


class InMemoryStateStore(StateStore):
    """Thread-safe in-memory state store for tests and local development."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, RunRecord] = {}
        self._tasks: dict[str, TaskRecord] = {}
        self._events: list[EventRecord] = []
        self._artifacts: list[ArtifactRecord] = []

    def create_run(self, run: RunRecord) -> None:
        with self._lock:
            if run.run_id in self._runs:
                raise ValueError(f"run already exists: {run.run_id}")
            self._runs[run.run_id] = run

    def update_run_status(self, run_id: str, status: RunStatus) -> None:
        with self._lock:
            run = self._runs[run_id]
            self._runs[run_id] = replace(run, status=status)

    def update_run_finished_at(self, run_id: str, finished_at: datetime) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"unknown run_id: {run_id}")
            self._runs[run_id] = replace(run, finished_at=finished_at)

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            return list(self._runs.values())

    def create_task(self, task: TaskRecord) -> None:
        with self._lock:
            if task.task_id in self._tasks:
                raise ValueError(f"task already exists: {task.task_id}")
            self._tasks[task.task_id] = task

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"unknown task_id: {task_id}")
            ensure_task_transition(task.status, status)
            self._tasks[task_id] = replace(
                task,
                status=status,
                result=result,
                error=error,
                lease_owner=None,
                leased_until=None,
            )

    def list_tasks(self, run_id: str) -> list[TaskRecord]:
        with self._lock:
            return sorted(
                [task for task in self._tasks.values() if task.run_id == run_id],
                key=lambda t: t.task_key,
            )

    def lease_next_pending_task(self, *, run_id: str, worker_id: str, lease_seconds: int) -> TaskRecord | None:
        now = utc_now()
        with self._lock:
            for task in sorted(
                [t for t in self._tasks.values() if t.run_id == run_id],
                key=lambda t: t.task_key,
            ):
                # Claim PENDING tasks, or LEASED tasks whose lease has expired —
                # matching SQLiteStore's SELECT … OR (status=LEASED AND leased_until < now).
                if task.status is TaskStatus.PENDING or (
                    task.status is TaskStatus.LEASED
                    and task.leased_until is not None
                    and task.leased_until <= now
                ):
                    lease = Lease.from_seconds(worker_id, lease_seconds)
                    leased = replace(task, status=TaskStatus.LEASED, lease_owner=worker_id, leased_until=lease.leased_until)
                    self._tasks[task.task_id] = leased
                    return leased
        return None

    def mark_task_running(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status is not TaskStatus.LEASED:
                raise KeyError(f"task is not leaseable/running-ready: {task_id}")
            self._tasks[task_id] = replace(task, status=TaskStatus.RUNNING)

    def increment_task_attempts(self, task_id: str) -> int:
        with self._lock:
            task = self._tasks[task_id]
            updated = replace(task, attempt_count=task.attempt_count + 1)
            self._tasks[task_id] = updated
            return updated.attempt_count

    def requeue_expired_leases(self, *, run_id: str) -> int:
        now = utc_now()
        with self._lock:
            count = 0
            for task in list(self._tasks.values()):
                if (
                    task.run_id == run_id
                    and task.status is TaskStatus.LEASED
                    and task.leased_until is not None
                    and task.leased_until <= now
                ):
                    self._tasks[task.task_id] = replace(
                        task,
                        status=TaskStatus.PENDING,
                        lease_owner=None,
                        leased_until=None,
                    )
                    count += 1
        return count

    def append_event(self, event: EventRecord) -> None:
        with self._lock:
            self._events.append(event)

    def create_artifact(self, artifact: ArtifactRecord) -> None:
        with self._lock:
            self._artifacts.append(artifact)

    def list_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        with self._lock:
            return [artifact for artifact in self._artifacts if artifact.run_id == run_id]

    def list_events(self, run_id: str) -> list[EventRecord]:
        with self._lock:
            return [event for event in self._events if event.run_id == run_id]
