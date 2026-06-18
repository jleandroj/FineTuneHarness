from __future__ import annotations

import dataclasses
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Callable

from finetuneharness.artifacts.store import ArtifactStore
from finetuneharness.evaluation.validator import ResultStatus, validate_result
from finetuneharness.executor.policy import NoSandbox, RetryPolicy, SandboxPolicy, TimeoutPolicy
from finetuneharness.executor.seeding import apply_seed
from finetuneharness.observability.logging import get_logger
from finetuneharness.orchestrator.hooks import HookRegistry
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.orchestrator.scheduler import TaskScheduler
from finetuneharness.state.models import EventRecord, TaskRecord, TaskStatus
from finetuneharness.state.store import StateStore


TaskHandler = Callable[[TaskRecord], dict[str, object]]


class DegradedRunError(Exception):
    """Raised by drain() when one or more tasks failed or timed out.

    All tasks were attempted; this error summarises the failures so the caller
    can decide whether to abort or accept partial results.
    """

    def __init__(
        self,
        run_id: str,
        failed_tasks: list[TaskRecord],
        succeeded: int,
    ) -> None:
        self.run_id = run_id
        self.failed_tasks = failed_tasks
        self.succeeded = succeeded
        keys = [t.task_key for t in failed_tasks]
        total = succeeded + len(failed_tasks)
        super().__init__(
            f"{len(failed_tasks)}/{total} tasks failed in run {run_id}: {keys}"
        )


class LocalWorker:
    """Local worker with best-effort timeout, hooks, and injectable dependencies.

    Timeout semantics (read carefully — this is NOT preemptive for in-process
    handlers):

      * The timeout makes the worker *stop waiting* for the handler and move on.
        It does NOT kill the handler's computation — Python offers no safe way to
        kill a running thread, so a hung in-process handler keeps consuming
        CPU/GPU until it returns naturally. Its result is discarded.
      * Each timed task runs in its OWN single-use ThreadPoolExecutor. A hung
        handler therefore leaks only its own thread and can NEVER starve later
        tasks (a previous shared, bounded pool would exhaust after N hangs and
        falsely mark healthy tasks TIMED_OUT). See _execute.
      * For TRUE preemption that frees the GPU, run handlers under FirejailSandbox
        (subprocess) and set a subprocess timeout — a killed process releases its
        resources. The in-process NoSandbox path cannot do this.

    ``max_workers`` is retained for config compatibility but no longer sizes a
    shared pool; drain() is sequential, so only one handler runs at a time.
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
        sandbox: SandboxPolicy | None = None,
        max_workers: int = 4,
    ) -> None:
        self.worker_id = worker_id
        self._store = store
        self._artifact_store = artifact_store
        self._retry_policy = retry_policy or RetryPolicy()
        self._timeout_policy = timeout_policy or TimeoutPolicy()
        self._scheduler = scheduler or TaskScheduler(store)
        self._runner = runner or FineTuneRunner(store)
        self._hooks = hooks or HookRegistry()
        self._sandbox: SandboxPolicy = sandbox or NoSandbox()
        self._log = get_logger("finetuneharness.worker")
        # No shared pool: each timed task gets a fresh single-use executor in
        # _execute so a hung handler cannot starve later tasks. Kept for config
        # compatibility / introspection only.
        self._max_workers = max_workers
        self._started_runs: set[str] = set()
        self._run_seeds: dict[str, int | None] = {}

    def run_once(self, *, run_id: str, handler: TaskHandler) -> TaskRecord | None:
        if run_id not in self._started_runs:
            self._started_runs.add(run_id)
            _run = self._store.get_run(run_id)
            self._run_seeds[run_id] = _run.seed if _run is not None else None
            self._hooks.fire("before_run_start", run_id=run_id)

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

        # Apply seed before hooks and handler so every participant sees a
        # reproducible RNG state from the start of the task.
        # Seed is cached at run-start (one get_run per run, not per task).
        _seed = self._run_seeds.get(run_id)
        if _seed is not None:
            apply_seed(_seed)

        # Give hooks and handler a shallow payload copy so before_task mutations
        # (e.g. CheckpointHook injecting checkpoint_dir) never touch the
        # canonical TaskRecord that the store owns.
        working_task = dataclasses.replace(task, payload=dict(task.payload))

        self._hooks.fire("before_task", task=working_task)
        self._store.append_event(EventRecord(
            event_id=uuid.uuid4().hex, run_id=run_id, task_id=task.task_id,
            kind="hook_fired",
            payload={"point": "before_task", "hooks": self._hooks.hook_names("before_task")},
        ))

        timeout_seconds = self._resolve_timeout(working_task)

        started = time.monotonic()
        try:
            result = self._execute(handler, working_task, timeout_seconds)
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
            self._hooks.fire("after_task_timeout", task=working_task)
            self._store.append_event(EventRecord(
                event_id=uuid.uuid4().hex, run_id=run_id, task_id=task.task_id,
                kind="hook_fired",
                payload={"point": "after_task_timeout", "hooks": self._hooks.hook_names("after_task_timeout")},
            ))
            self._hooks.fire("on_run_status_changed", run_id=run_id, status=run_status)
            raise
        except Exception as exc:
            max_attempts = self._resolve_max_attempts(working_task)
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
            self._hooks.fire("after_task_failure", task=working_task, error=exc)
            self._store.append_event(EventRecord(
                event_id=uuid.uuid4().hex, run_id=run_id, task_id=task.task_id,
                kind="hook_fired",
                payload={"point": "after_task_failure", "hooks": self._hooks.hook_names("after_task_failure")},
            ))
            self._hooks.fire("on_run_status_changed", run_id=run_id, status=run_status)
            if attempt_count < self._resolve_max_attempts(working_task):
                delay = self._retry_policy.delay_for_attempt(attempt_count)
                if delay > 0:
                    self._log.info(
                        "task_retry_backoff",
                        extra={"run_id": run_id, "task_id": task.task_id,
                               "attempt": attempt_count, "delay_seconds": delay},
                    )
                    time.sleep(delay)
            raise

        wall_seconds = round(time.monotonic() - started, 3)
        if "wall_seconds" not in result:
            result = {**result, "wall_seconds": wall_seconds}

        outcome = validate_result(result, method=working_task.payload.get("method"))
        result = {**result, "_validation_status": outcome.status.value}
        if outcome.status in (ResultStatus.DEGENERATE_RESULT, ResultStatus.FAILED_VALIDATION):
            result["_validation_errors"] = outcome.validation_errors

        # Fire after_task_success hooks BEFORE persisting so hook mutations
        # (e.g. gpu_allocated_mb from GPUMemoryHook) are included in the stored
        # result. HookRegistry.fire() is fire-and-forget: a crashing hook is
        # logged and skipped, so the store write always completes.
        self._hooks.fire("after_task_success", task=working_task, result=result)
        self._store.append_event(EventRecord(
            event_id=uuid.uuid4().hex, run_id=run_id, task_id=task.task_id,
            kind="hook_fired",
            payload={"point": "after_task_success", "hooks": self._hooks.hook_names("after_task_success")},
        ))
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
        self._hooks.fire("on_run_status_changed", run_id=run_id, status=run_status)
        return task

    def drain(
        self,
        *,
        run_id: str,
        handler: TaskHandler,
        stop_fn: Callable[[], bool] | None = None,
    ) -> int:
        """Run all pending tasks in a run, continuing past individual failures.

        Returns the number of tasks that succeeded. Raises DegradedRunError at
        the end (not mid-loop) if any tasks ended in FAILED or TIMED_OUT, so
        the caller always gets a complete grid run before learning about errors.

        stop_fn: optional callable checked after each successful task. When it
        returns True, drain() stops leasing new tasks immediately. Intended for
        EarlyStoppingHook.should_stop() — without this, the hook sets its flag
        but drain() keeps running tasks regardless.
        """
        succeeded = 0
        while True:
            try:
                task = self.run_once(run_id=run_id, handler=handler)
                if task is None:
                    break
                succeeded += 1
                if stop_fn is not None and stop_fn():
                    break
            except Exception:
                pass  # status + event already recorded by run_once; keep going

        failed_tasks = [
            t for t in self._store.list_tasks(run_id)
            if t.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT)
        ]
        if failed_tasks:
            raise DegradedRunError(
                run_id=run_id, failed_tasks=failed_tasks, succeeded=succeeded
            )
        return succeeded

    def _execute(
        self,
        handler: TaskHandler,
        task: TaskRecord,
        timeout_seconds: int | None,
    ) -> dict[str, object]:
        if timeout_seconds is None:
            return self._sandbox.run(handler, task)
        # Per-task executor: isolates each timed handler so a hung one leaks only
        # its own thread and never occupies a slot needed by a later task.
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"worker-{self.worker_id}-task"
        )
        future = executor.submit(self._sandbox.run, handler, task)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            raise TimeoutError(f"task exceeded timeout of {timeout_seconds}s")
        finally:
            # wait=False: a timed-out handler thread cannot be killed; abandon it
            # so the worker proceeds immediately to the next task.
            executor.shutdown(wait=False)

    def _resolve_max_attempts(self, task: TaskRecord) -> int:
        raw = task.payload.get("max_attempts", self._retry_policy.max_attempts)
        return max(1, int(raw))

    def _resolve_timeout(self, task: TaskRecord) -> int | None:
        raw = task.payload.get("timeout_seconds", self._timeout_policy.timeout_seconds)
        return None if raw is None else int(raw)
