from __future__ import annotations

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from finetuneharness.state.models import ArtifactRecord, EventRecord, RunRecord, RunStatus, TaskRecord, TaskStatus


class StateStore(ABC):
    """Source-of-truth contract for FineTuneHarness state."""

    @abstractmethod
    def create_run(self, run: RunRecord) -> None: ...

    @abstractmethod
    def update_run_status(self, run_id: str, status: RunStatus) -> None: ...

    @abstractmethod
    def update_run_finished_at(self, run_id: str, finished_at: "datetime") -> None: ...

    @abstractmethod
    def get_run(self, run_id: str) -> RunRecord | None: ...

    @abstractmethod
    def list_runs(self) -> list[RunRecord]: ...

    @abstractmethod
    def create_task(self, task: TaskRecord) -> None: ...

    @abstractmethod
    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None: ...

    @abstractmethod
    def list_tasks(self, run_id: str) -> list[TaskRecord]: ...

    @abstractmethod
    def lease_next_pending_task(self, *, run_id: str, worker_id: str, lease_seconds: int) -> TaskRecord | None: ...

    @abstractmethod
    def mark_task_running(self, task_id: str) -> None: ...

    @abstractmethod
    def increment_task_attempts(self, task_id: str) -> int: ...

    @abstractmethod
    def requeue_expired_leases(self, *, run_id: str) -> int: ...

    @abstractmethod
    def append_event(self, event: EventRecord) -> None: ...

    @abstractmethod
    def create_artifact(self, artifact: ArtifactRecord) -> None: ...

    @abstractmethod
    def list_artifacts(self, run_id: str) -> list[ArtifactRecord]: ...

    @abstractmethod
    def list_events(self, run_id: str) -> list[EventRecord]: ...
