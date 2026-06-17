from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Callable

from finetuneharness.artifacts.store import ArtifactStore
from finetuneharness.executor.policy import RetryPolicy, TimeoutPolicy
from finetuneharness.observability.logging import get_logger
from finetuneharness.orchestrator.hooks import HookRegistry
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.orchestrator.scheduler import TaskScheduler
from finetuneharness.state.models import EventRecord, TaskRecord, TaskStatus
from finetuneharness.state.store import StateStore


TaskHandler = Callable[[TaskRecord], dict[str, object]]


class LocalWorker:
    """Local worker with preemptive timeout, hooks, and injectable dependencies.

    Timeout is enforced via concurrent.futures: the worker unblocks after
    timeout_seconds even if the handler is still running. The handler thread
    continues in the background until it completes naturally.
    """

    def __init__(
        self,
        *,
        worker_id: str,
        store: StateStore,
        artifact_store: ArtifactStore | None = None,
        retry_policy: RetryPolicy | None = None,
        timeout_policy: TimeoutPolicy | None = None,
        runner: FineTuneRunner | None = None,
        scheduler: TaskScheduler | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._store = store
        self._artifact_store = artifact_store
        self._retry_policy = retry_policy or RetryPolicy()
        self._timeout_policy = timeout_policy or TimeoutPolicy()
        self._scheduler = scheduler or TaskScheduler(store)
        self._runner = runner or FineTuneRunner(store)
        self._hooks = hooks or HookRegistry()
        self._log = get_logger("finetuneharness.worker")

    def run_once(self, *, run_id: str, handler: TaskHandler) -> TaskRecord | None:
        task = self._scheduler.lease_next_task(run_id=run_id, worker_id=self.worker_id)
        if task is None:
            return None

        attempt_count = self._store.increment_task_attempts(task.task_id)
        self._store.mark_task_running(task.task_id)
        self._store.append_event(
            EventRecord(
                event_id=uuid.uuid4().hex,
                run_id=run_id,
                task_id=task.task_id,
                kind="task_running",
                payload={"worker_id": self.worker_id, "attempt": attempt_count},
            )
        )
        self._runner.refresh_run_status(run_id)
        self._log.info("task_running", extra={"run_id": run_id, "task_id": task.task_id})

        self._hooks.fire("before_task", task=task)

        timeout_seconds = self._resolve_timeout(task)
        try:
            result = self._execute(handler, task, timeout_seconds)
        except TimeoutError as exc:
            self._store.update_task_status(task.task_id, TaskStatus.TIMED_OUT, error=str(exc))
            self._store.append_event(
                EventRecord(
                    event_id=uuid.uuid4().hex,
                    run_id=run_id,
                    task_id=task.task_id,
                    kind="task_timed_out",
                    payload={"worker_id": self.worker_id, "attempt": attempt_count, "error": str(exc)},
                )
            )
            run_status = self._runner.refresh_run_status(run_id)
            self._log.info("task_timed_out", extra={"run_id": run_id, "task_id": task.task_id})
            self._hooks.fire("after_task_timeout", task=task)
            self._hooks.fire("on_run_status_changed", run_id=run_id, status=run_status)
            raise
        except Exception as exc:
            max_attempts = self._resolve_max_attempts(task)
            if attempt_count < max_attempts:
                self._store.update_task_status(task.task_id, TaskStatus.PENDING, error=str(exc))
                self._store.append_event(
                    EventRecord(
                        event_id=uuid.uuid4().hex,
                        run_id=run_id,
                        task_id=task.task_id,
                        kind="task_retry_scheduled",
                        payload={
                            "worker_id": self.worker_id,
                            "attempt": attempt_count,
                            "max_attempts": max_attempts,
                            "error": str(exc),
                        },
                    )
                )
            else:
                self._store.update_task_status(task.task_id, TaskStatus.FAILED, error=str(exc))
                self._store.append_event(
                    EventRecord(
                        event_id=uuid.uuid4().hex,
                        run_id=run_id,
                        task_id=task.task_id,
                        kind="task_failed",
                        payload={"worker_id": self.worker_id, "attempt": attempt_count, "error": str(exc)},
                    )
                )
            run_status = self._runner.refresh_run_status(run_id)
            self._log.info("task_failed", extra={"run_id": run_id, "task_id": task.task_id})
            self._hooks.fire("after_task_failure", task=task, error=exc)
            self._hooks.fire("on_run_status_changed", run_id=run_id, status=run_status)
            raise

        self._store.update_task_status(task.task_id, TaskStatus.SUCCEEDED, result=result)
        if self._artifact_store is not None:
            self._artifact_store.write_json_artifact(
                run_id=run_id,
                task_id=task.task_id,
                kind="task_result",
                payload=result,
                filename=f"{task.task_key}-result.json",
            )
        self._store.append_event(
            EventRecord(
                event_id=uuid.uuid4().hex,
                run_id=run_id,
                task_id=task.task_id,
                kind="task_succeeded",
                payload={"worker_id": self.worker_id, "attempt": attempt_count},
            )
        )
        run_status = self._runner.refresh_run_status(run_id)
        self._log.info("task_succeeded", extra={"run_id": run_id, "task_id": task.task_id})
        self._hooks.fire("after_task_success", task=task, result=result)
        self._hooks.fire("on_run_status_changed", run_id=run_id, status=run_status)
        return task

    def _execute(
        self,
        handler: TaskHandler,
        task: TaskRecord,
        timeout_seconds: int | None,
    ) -> dict[str, object]:
        if timeout_seconds is None:
            return handler(task)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(handler, task)
        # shutdown(wait=False) detaches executor so the worker is not blocked when
        # future.result() times out — the underlying thread continues until handler returns.
        executor.shutdown(wait=False)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            raise TimeoutError(f"task exceeded timeout of {timeout_seconds}s")

    def _resolve_max_attempts(self, task: TaskRecord) -> int:
        raw = task.payload.get("max_attempts", self._retry_policy.max_attempts)
        return max(1, int(raw))

    def _resolve_timeout(self, task: TaskRecord) -> int | None:
        raw = task.payload.get("timeout_seconds", self._timeout_policy.timeout_seconds)
        return None if raw is None else int(raw)
