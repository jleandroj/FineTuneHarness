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
    def reclaim_dead_worker(self, *, run_id: str, worker_id: str, max_reclaims: int) -> str:
        """Recover the task a dead worker left LEASED or RUNNING.

        A worker whose process died (e.g. OS OOM-killer after mark_task_running)
        leaves its task non-terminal and otherwise unrecoverable — lease reclaim
        only covers LEASED, never RUNNING. Given the *worker_id* of a process known
        to be dead, find its in-flight task and:
          * requeue it to PENDING (returns "requeued") if it has been reclaimed
            fewer than *max_reclaims* times, so it can be retried; or
          * mark it FAILED (returns "failed") once the reclaim budget is exhausted,
            so a task that keeps killing its host terminates instead of looping.
        Returns "none" if the worker holds no in-flight task.
        """
        ...

    @abstractmethod
    def recover_orphaned_tasks(self, *, run_id: str) -> int:
        """Requeue every RUNNING or LEASED task of a run back to PENDING.

        Operator-invoked post-crash recovery. When the orchestrator process is
        hard-killed (SIGKILL, power loss) its drain loop never runs its cleanup, so
        in-flight tasks are stranded RUNNING/LEASED and no automatic path reclaims a
        RUNNING task (reclaim_dead_worker needs a known worker_id; expired-lease
        requeue covers only LEASED). This clears the lease and returns them to
        PENDING so a fresh worker can pick them up. Emits one 'task_recovered' event
        per task. MUST only be called when no worker is active for the run — it does
        not check liveness, so requeuing a task another worker is still running would
        double-execute it. Returns the number of tasks requeued.
        """
        ...

    @abstractmethod
    def append_event(self, event: EventRecord) -> None: ...

    @abstractmethod
    def create_artifact(self, artifact: ArtifactRecord) -> None: ...

    @abstractmethod
    def list_artifacts(self, run_id: str) -> list[ArtifactRecord]: ...

    @abstractmethod
    def list_events(self, run_id: str) -> list[EventRecord]: ...
