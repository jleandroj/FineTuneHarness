from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
import queue as _queue
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Callable, NoReturn

from finetuneharness.artifacts.store import ArtifactStore
from finetuneharness.evaluation.validator import ResultStatus, ValidationOutcome, validate_result
from finetuneharness.executor.policy import NoSandbox, RetryPolicy, SandboxPolicy, TimeoutPolicy
from finetuneharness.executor.resources import ConcurrencyConfig, ResourceMonitor, is_oom_error
from finetuneharness.executor.seeding import apply_seed
from finetuneharness.observability.logging import get_logger
from finetuneharness.orchestrator.hooks import HookRegistry
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.orchestrator.scheduler import TaskScheduler
from finetuneharness.state.leases import utc_now
from finetuneharness.state.models import EventRecord, TaskRecord, TaskStatus
from finetuneharness.state.store import StateStore


TaskHandler = Callable[[TaskRecord], dict[str, object]]


class DegradedRunError(Exception):
    """Raised by drain() when one or more tasks failed, timed out, or were degenerate.

    All tasks were attempted; this error summarises the non-successes so the caller
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
            f"{len(failed_tasks)}/{total} tasks did not succeed in run {run_id}: {keys}"
        )


class DegenerateResultError(Exception):
    """Raised by run_once() when a handler returned but its result failed validation.

    The handler did not raise — it produced a structurally untrustworthy result
    (degenerate experiment, or out-of-range/NaN metrics). The task is marked
    TaskStatus.DEGENERATE (terminal, NOT retried) before this is raised, so
    drain() treats it like any other non-success and keeps going.
    """

    def __init__(self, task_key: str, status: str, errors: list[str]) -> None:
        self.task_key = task_key
        self.status = status
        self.errors = errors
        super().__init__(
            f"task {task_key!r} result failed validation ({status}): {errors}"
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

    Concurrency: ``drain`` runs tasks one at a time. ``drain_concurrent`` runs
    several at once under a resource-aware admission policy (see executor.resources)
    — it admits a new task only while the GPU has free memory above a headroom, and
    on a GPU out-of-memory error it requeues the task and lowers the concurrency
    ceiling. There is no static ``max_workers`` knob; concurrency is governed by the
    live resource signal, not a fixed thread count.

    Reproducibility under concurrency: each concurrent task runs in its OWN process
    (fork), so the per-task seed (apply_seed mutates the *process-global* numpy/
    torch/random RNG) is isolated per task. Thread-based concurrency would corrupt
    reproducibility — sibling threads share one global RNG and interleave draws
    non-deterministically. drain_concurrent therefore requires a persistent (SQLite)
    store the child processes can reopen, and handlers run in forked children. The
    OOM retry budget is tracked via persisted events (not worker memory) because
    each attempt is a fresh process.
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
        # _execute so a hung handler cannot starve later tasks.
        self._started_runs: set[str] = set()
        self._run_seeds: dict[str, int | None] = {}
        # 0 disables OOM-specific requeue (sequential drain): OOM then follows the
        # normal retry/fail path. drain_concurrent raises this to the configured
        # cap so a transient memory contention requeues instead of burning a retry.
        # The per-task budget is counted from persisted 'task_oom_requeued' events
        # (NOT worker memory) because under drain_concurrent each attempt is a fresh
        # forked process with its own worker instance.
        self._max_oom_retries = 0

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
            # GPU OOM is treated as transient resource contention, not a
            # deterministic failure: requeue the task (without consuming a normal
            # retry) so drain_concurrent can re-run it at a lower concurrency.
            if self._max_oom_retries > 0 and is_oom_error(exc):
                self._handle_oom(
                    run_id=run_id,
                    task=task,
                    working_task=working_task,
                    exc=exc,
                    attempt_count=attempt_count,
                )
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
            return self._record_degenerate(
                run_id=run_id,
                task=task,
                working_task=working_task,
                result=result,
                outcome=outcome,
                attempt_count=attempt_count,
            )

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

    def _record_degenerate(
        self,
        *,
        run_id: str,
        task: TaskRecord,
        working_task: TaskRecord,
        result: dict[str, object],
        outcome: ValidationOutcome,
        attempt_count: int,
    ) -> NoReturn:
        """Persist a handler result that the validator rejected.

        The handler returned normally, so this is NOT the exception path: the task
        is marked TaskStatus.DEGENERATE (terminal, never retried — a structurally
        invalid result is deterministic, re-running the same config reproduces it).
        The result is still persisted and written as an artifact so the scientist
        can inspect *why* it was rejected. Raises DegenerateResultError so drain()
        accounts for it as a non-success.
        """
        errors = list(outcome.validation_errors)
        error_summary = f"{outcome.status.value}: {errors}"
        self._store.update_task_status(
            task.task_id, TaskStatus.DEGENERATE, result=result, error=error_summary
        )
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
                kind="task_degenerate",
                payload={
                    "worker_id": self.worker_id,
                    "attempt": attempt_count,
                    "validation_status": outcome.status.value,
                    "validation_errors": errors,
                },
            )
        )
        run_status = self._runner.refresh_run_status(run_id)
        self._log.warning(
            "task_degenerate",
            extra={
                "run_id": run_id,
                "task_id": task.task_id,
                "validation_status": outcome.status.value,
                "validation_errors": errors,
            },
        )
        exc = DegenerateResultError(task.task_key, outcome.status.value, errors)
        # A degenerate result did not succeed: fire the failure hook, not success.
        self._hooks.fire("after_task_failure", task=working_task, error=exc)
        self._store.append_event(EventRecord(
            event_id=uuid.uuid4().hex, run_id=run_id, task_id=task.task_id,
            kind="hook_fired",
            payload={"point": "after_task_failure", "hooks": self._hooks.hook_names("after_task_failure")},
        ))
        self._hooks.fire("on_run_status_changed", run_id=run_id, status=run_status)
        raise exc

    def _handle_oom(
        self,
        *,
        run_id: str,
        task: TaskRecord,
        working_task: TaskRecord,
        exc: Exception,
        attempt_count: int,
    ) -> None:
        """React to a GPU out-of-memory error from a handler.

        While under the per-task OOM budget, requeue the task to PENDING (so
        drain_concurrent re-runs it once memory frees) and re-raise *exc* so the
        scheduler lowers its concurrency ceiling. Once the budget is exhausted,
        return normally — the caller then follows the ordinary failure path and
        marks the task FAILED, so an unschedulable task can never loop forever.

        The budget is counted from persisted ``task_oom_requeued`` events, not
        worker memory: under drain_concurrent each attempt is a separate forked
        process, so an in-memory counter would always read 1 and never exhaust.
        """
        prior = sum(
            1 for e in self._store.list_events(run_id)
            if e.task_id == task.task_id and e.kind == "task_oom_requeued"
        )
        if prior >= self._max_oom_retries:
            self._log.warning(
                "task_oom_exhausted",
                extra={"run_id": run_id, "task_id": task.task_id, "oom_attempts": prior + 1},
            )
            return  # fall through to the normal failure path -> FAILED
        self._store.update_task_status(task.task_id, TaskStatus.PENDING, error=str(exc))
        self._store.append_event(
            EventRecord(
                event_id=uuid.uuid4().hex,
                run_id=run_id,
                task_id=task.task_id,
                kind="task_oom_requeued",
                payload={
                    "worker_id": self.worker_id,
                    "attempt": attempt_count,
                    "oom_attempt": prior + 1,
                    "max_oom_retries": self._max_oom_retries,
                    "error": str(exc),
                },
            )
        )
        run_status = self._runner.refresh_run_status(run_id)
        self._log.warning(
            "task_oom_requeued",
            extra={"run_id": run_id, "task_id": task.task_id, "oom_attempt": prior + 1},
        )
        self._hooks.fire("on_run_status_changed", run_id=run_id, status=run_status)
        raise exc

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
        self._record_drain_started(run_id, ConcurrencyConfig(mode="sequential"))
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
            if t.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.DEGENERATE)
        ]
        if failed_tasks:
            raise DegradedRunError(
                run_id=run_id, failed_tasks=failed_tasks, succeeded=succeeded
            )
        return succeeded

    def drain_concurrent(
        self,
        *,
        run_id: str,
        handler: TaskHandler,
        concurrency: ConcurrencyConfig,
        monitor: ResourceMonitor,
        stop_fn: Callable[[], bool] | None = None,
    ) -> int:
        """Drain a run with resource-aware, PROCESS-ISOLATED concurrency.

        Each admitted task runs ``run_once`` in its own forked process, so the
        per-task seed (which mutates process-global numpy/torch/random RNG) is
        isolated — reproducibility is preserved. (Thread-based concurrency would
        share one global RNG across tasks and corrupt it.) A new task is admitted
        only while the live in-flight count is below the dynamic ceiling AND free
        GPU memory is above ``min_free_mb``; the first task is always admitted so a
        busy GPU cannot deadlock the run. On a GPU OOM the task is requeued by
        ``_handle_oom`` (budget tracked via persisted events) and the ceiling is
        lowered, converging toward sequential under memory pressure.

        Like ``drain``, returns the count of succeeded tasks and raises
        ``DegradedRunError`` if any task ended FAILED/TIMED_OUT/DEGENERATE.

        Requires a persistent (SQLite) store — child processes reopen it to share
        state. If the monitor reports no GPU (``free_gpu_memory_mb() is None``),
        there is no signal to gate on, so this degrades to sequential ``drain``.
        """
        if monitor.free_gpu_memory_mb() is None:
            self._log.info("no_gpu_detected_draining_sequentially", extra={"run_id": run_id})
            return self.drain(run_id=run_id, handler=handler, stop_fn=stop_fn)

        from finetuneharness.state.sqlite import SQLiteStateStore
        if not isinstance(self._store, SQLiteStateStore):
            raise TypeError(
                "drain_concurrent requires a persistent SQLiteStateStore: each task "
                "runs in its own process and shares state through the database. "
                "Use SQLiteStateStore, or call drain() for in-memory/sequential runs."
            )

        self._max_oom_retries = concurrency.max_oom_retries

        # Run-start bookkeeping ONCE in the parent, before forking. Children inherit
        # _started_runs + _run_seeds via fork, so run_once still applies the seed in
        # each child but does NOT re-fire before_run_start per process.
        if run_id not in self._started_runs:
            self._started_runs.add(run_id)
            _run = self._store.get_run(run_id)
            self._run_seeds[run_id] = _run.seed if _run is not None else None
            self._hooks.fire("before_run_start", run_id=run_id)
        self._record_drain_started(run_id, concurrency)

        ctx = mp.get_context("fork")
        result_q: mp.Queue = ctx.Queue()

        def _child() -> None:
            # Runs in a forked child: its own RNG globals, own SQLite connections.
            try:
                task = self.run_once(run_id=run_id, handler=handler)
                result_q.put((os.getpid(), "succeeded" if task is not None else "empty"))
            except Exception as exc:  # noqa: BLE001 — outcome relayed to the parent
                result_q.put((os.getpid(), "oom" if is_oom_error(exc) else "error"))

        succeeded = 0
        stopped = False
        ceiling = max(1, concurrency.max_concurrent)
        min_free = concurrency.min_free_mb
        in_flight: dict[int, mp.Process] = {}
        try:
            while True:
                # ── Admission ──────────────────────────────────────────────
                while (
                    not stopped
                    and len(in_flight) < ceiling
                    and self._count_leaseable(run_id) > 0
                ):
                    free = monitor.free_gpu_memory_mb()
                    # Gate on memory only once at least one task is already running —
                    # the first admission must always proceed.
                    if free is not None and in_flight and free < min_free:
                        break
                    proc = ctx.Process(target=_child)
                    proc.start()
                    in_flight[proc.pid] = proc
                    # Let the just-admitted task allocate before re-reading memory.
                    if concurrency.settle_seconds > 0:
                        time.sleep(concurrency.settle_seconds)

                if not in_flight:
                    if stopped or self._count_leaseable(run_id) == 0:
                        break
                    continue

                try:
                    pid, outcome = result_q.get(timeout=0.5)
                except _queue.Empty:
                    # Reap any child that died without reporting (e.g. OS OOM-killer
                    # or segfault). Its task stays RUNNING until its lease expires and
                    # is reclaimed; back off so we do not immediately respawn into the
                    # same memory wall.
                    for dead_pid, dead in list(in_flight.items()):
                        if not dead.is_alive():
                            dead.join()
                            del in_flight[dead_pid]
                            ceiling = max(1, ceiling - 1)
                            self._log.warning(
                                "worker_child_died_unreported",
                                extra={"run_id": run_id, "pid": dead_pid,
                                       "exitcode": dead.exitcode},
                            )
                    continue

                proc = in_flight.pop(pid, None)
                if proc is not None:
                    proc.join()
                if outcome == "succeeded":
                    succeeded += 1
                    if stop_fn is not None and stop_fn():
                        stopped = True
                elif outcome == "oom":
                    ceiling = max(1, ceiling - 1)
                    min_free *= 1.5
                    self._log.info(
                        "concurrency_backoff_after_oom",
                        extra={"run_id": run_id, "ceiling": ceiling,
                               "min_free_mb": round(min_free, 1)},
                    )
                # "empty" (lost lease race) / "error" (recorded by run_once): keep going
        finally:
            for proc in in_flight.values():
                proc.join()

        failed_tasks = [
            t for t in self._store.list_tasks(run_id)
            if t.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.DEGENERATE)
        ]
        if failed_tasks:
            raise DegradedRunError(
                run_id=run_id, failed_tasks=failed_tasks, succeeded=succeeded
            )
        return succeeded

    def _record_drain_started(self, run_id: str, concurrency: ConcurrencyConfig) -> None:
        """Record the actual execution mode so reproducibility tooling can see it."""
        self._store.append_event(
            EventRecord(
                event_id=uuid.uuid4().hex,
                run_id=run_id,
                task_id=None,
                kind="drain_started",
                payload={
                    "worker_id": self.worker_id,
                    "mode": concurrency.mode,
                    "max_concurrent": concurrency.max_concurrent,
                    "min_free_mb": concurrency.min_free_mb,
                },
            )
        )

    def _count_leaseable(self, run_id: str) -> int:
        """Tasks that could be leased right now: PENDING or expired LEASED.

        Used by drain_concurrent to stop admitting once the queue is empty,
        without submitting wasted no-op run_once calls.
        """
        now = utc_now()
        n = 0
        for t in self._store.list_tasks(run_id):
            if t.status is TaskStatus.PENDING:
                n += 1
            elif (
                t.status is TaskStatus.LEASED
                and t.leased_until is not None
                and t.leased_until < now
            ):
                n += 1
        return n

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
