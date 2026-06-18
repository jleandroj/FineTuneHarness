from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
import queue as _queue
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
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

# Hooks that accumulate state across tasks in instance memory. Under drain_concurrent
# each task runs in its own process with a fresh hook instance, so their state never
# aggregates (EarlyStopping would never trip; Metrics.get_summary/Progress would see
# one task). drain_concurrent rejects them rather than give silently-wrong results.
_CUMULATIVE_HOOK_CLASSES = (
    "EarlyStoppingHook",
    "MetricsHook",
    "ProgressHook",
    "CheckpointHook",
)


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
    (multiprocessing 'spawn'), so the per-task seed (apply_seed mutates the
    *process-global* numpy/torch/random RNG) is isolated per task. Thread-based
    concurrency would corrupt reproducibility — sibling threads share one global RNG
    and interleave draws non-deterministically. 'spawn' (not 'fork') also avoids the
    CUDA-after-fork crash: a fresh interpreter never inherits a CUDA context the
    parent may have created. Consequences: drain_concurrent requires a persistent
    (SQLite) store the children reopen; handlers must be importable (module:function,
    like FirejailSandbox) since spawn pickles them; custom hooks/sandbox do not
    propagate to children; the OOM retry budget is tracked via persisted events (not
    worker memory) because each attempt is a fresh process.
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
        hooks_factory: str | None = None,
        sandbox_factory: str | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._store = store
        self._artifact_store = artifact_store
        self._retry_policy = retry_policy or RetryPolicy()
        self._timeout_policy = timeout_policy or TimeoutPolicy()
        self._scheduler = scheduler or TaskScheduler(store)
        self._runner = runner or FineTuneRunner(store)
        # hooks/sandbox factories are importable 'module:function' specs that BUILD a
        # HookRegistry / SandboxPolicy. drain_concurrent needs them: spawned children
        # rebuild the worker, and a factory is the only thing picklable enough to
        # recreate hooks/sandbox in the child (concrete objects with loggers/closures
        # are not). When a factory is given it also builds the parent's instance, so
        # parent and children are guaranteed identical.
        self._hooks_factory = hooks_factory
        self._sandbox_factory = sandbox_factory
        self._hooks = _call_factory(hooks_factory) if hooks_factory else (hooks or HookRegistry())
        self._sandbox: SandboxPolicy = (
            _call_factory(sandbox_factory) if sandbox_factory else (sandbox or NoSandbox())
        )
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
        # spawned process with its own worker instance.
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
        worker memory: under drain_concurrent each attempt is a separate spawned
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

        Each admitted task runs ``run_once`` in its own spawned process, so the
        per-task seed (which mutates process-global numpy/torch/random RNG) is
        isolated — reproducibility is preserved. A task is admitted to a GPU only
        while that device has free memory above ``min_free_mb`` (the first task on an
        idle device is always admitted), up to the dynamic ceiling. On a GPU OOM the
        task is requeued by ``_handle_oom`` and the ceiling is lowered, converging
        toward sequential under memory pressure.

        Multi-GPU: admission is per-device — each child is pinned to a chosen GPU via
        CUDA_VISIBLE_DEVICES and gated on THAT device's free memory, distributing
        tasks across devices.

        Like ``drain``, returns the count of succeeded tasks and raises
        ``DegradedRunError`` if any task ended FAILED/TIMED_OUT/DEGENERATE.

        Requires a persistent (SQLite) store — children reopen it to share state.
        Custom hooks/sandbox must be supplied as importable factories
        (``hooks_factory``/``sandbox_factory``) so children can recreate them;
        otherwise this raises (concrete hooks/sandbox cannot cross to a spawned
        child). If the monitor reports no GPU, this degrades to sequential ``drain``.
        """
        per_device = monitor.free_memory_per_device()
        if per_device is None:
            self._log.info("no_gpu_detected_draining_sequentially", extra={"run_id": run_id})
            return self.drain(run_id=run_id, handler=handler, stop_fn=stop_fn)

        # stop_fn (e.g. EarlyStopping.should_stop) decides from aggregate state that no
        # single task process can see — unsound in concurrent mode. Reject it here; the
        # no-GPU path above still honors it because that runs sequentially.
        if stop_fn is not None:
            raise RuntimeError(
                "drain_concurrent: stop_fn is not supported in concurrent mode — its "
                "decision relies on aggregate state that each task process cannot see. "
                "Use sequential drain() for early stopping."
            )

        from finetuneharness.state.sqlite import SQLiteStateStore
        if not isinstance(self._store, SQLiteStateStore):
            raise TypeError(
                "drain_concurrent requires a persistent SQLiteStateStore: each task "
                "runs in its own process and shares state through the database. "
                "Use SQLiteStateStore, or call drain() for in-memory/sequential runs."
            )
        # Consistency guard: spawned children rebuild the worker, so concrete custom
        # hooks/sandbox would silently NOT run in concurrent mode. Require factories.
        if self._hooks_factory is None and self._hooks.total() > 0:
            raise RuntimeError(
                "drain_concurrent: this worker has custom hooks that cannot reach "
                "spawned children. Pass hooks_factory='module:function' (a factory that "
                "rebuilds the HookRegistry) so each child recreates them, or use drain()."
            )
        if self._sandbox_factory is None and not isinstance(self._sandbox, NoSandbox):
            raise RuntimeError(
                "drain_concurrent: this worker has a non-default sandbox that cannot "
                "reach spawned children. Pass sandbox_factory='module:function', "
                "or use drain()."
            )
        # Cumulative hooks keep per-instance state that cannot aggregate across the
        # separate processes each task runs in — reject rather than mislead.
        offenders = sorted({
            name.split(".", 1)[0]
            for name in self._hooks.all_hook_names()
            if name.split(".", 1)[0] in _CUMULATIVE_HOOK_CLASSES
        })
        if offenders:
            raise RuntimeError(
                f"drain_concurrent: cumulative hooks {offenders} keep per-instance state "
                "that does not aggregate across the separate task processes (EarlyStopping "
                "would never trip; Metrics.get_summary/Progress would see one task). Use "
                "sequential drain() for these hooks."
            )

        self._max_oom_retries = concurrency.max_oom_retries
        db_path = str(self._store._path)
        num_devices = len(per_device)

        # Run-start bookkeeping ONCE in the parent, before spawning any child. The
        # cached seed is passed to each child so its run_once applies the seed but
        # does NOT re-fire before_run_start per process.
        if run_id not in self._started_runs:
            self._started_runs.add(run_id)
            _run = self._store.get_run(run_id)
            self._run_seeds[run_id] = _run.seed if _run is not None else None
            self._hooks.fire("before_run_start", run_id=run_id)
        self._record_drain_started(run_id, concurrency)
        seed = self._run_seeds.get(run_id)

        # 'spawn' (NOT 'fork'): a fresh interpreter per task never inherits a CUDA
        # context the parent may have created, avoiding the CUDA-after-fork crash.
        ctx = mp.get_context("spawn")
        result_q: mp.Queue = ctx.Queue()

        def _free_for(device: int, live: list[float]) -> float:
            return live[device] if device < len(live) else 0.0

        succeeded = 0
        stopped = False
        child_seq = 0
        ceiling = max(1, concurrency.max_concurrent)
        min_free = concurrency.min_free_mb
        # pid -> (Process, worker_id, device_index)
        in_flight: dict[int, tuple[mp.Process, str, int]] = {}
        try:
            while True:
                # ── Admission (per-device) ─────────────────────────────────
                while (
                    not stopped
                    and len(in_flight) < ceiling
                    and self._count_leaseable(run_id) > 0
                ):
                    live = monitor.free_memory_per_device() or per_device
                    inflight_by_dev: Counter[int] = Counter(d for _, _, d in in_flight.values())
                    order = sorted(range(num_devices), key=lambda d: _free_for(d, live), reverse=True)
                    # Prefer an idle device (first task there always proceeds); else a
                    # busy device that still clears the memory headroom.
                    device = next((d for d in order if inflight_by_dev[d] == 0), None)
                    if device is None:
                        device = next(
                            (d for d in order if _free_for(d, live) >= min_free), None
                        )
                    if device is None:
                        break  # all devices busy and none has headroom; wait

                    child_seq += 1
                    child_wid = f"{self.worker_id}-c{child_seq}"
                    proc = ctx.Process(
                        target=_run_once_subprocess,
                        args=(db_path, run_id, child_wid, handler, seed,
                              self._max_oom_retries, self._hooks_factory,
                              self._sandbox_factory,
                              device if num_devices > 1 else None, result_q),
                    )
                    proc.start()
                    in_flight[proc.pid] = (proc, child_wid, device)
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
                    # A child that died WITHOUT reporting (e.g. OS OOM-killer or
                    # segfault) left its task LEASED/RUNNING — which no lease reclaim
                    # ever recovers. Recover it explicitly via the child's worker_id:
                    # requeue (retry) or, once the reclaim budget is spent, FAIL it so
                    # the run terminates instead of hanging on a lost task.
                    for dead_pid, (dead, dead_wid, _dev) in list(in_flight.items()):
                        if not dead.is_alive():
                            dead.join()
                            del in_flight[dead_pid]
                            reclaim = self._store.reclaim_dead_worker(
                                run_id=run_id, worker_id=dead_wid,
                                max_reclaims=self._max_oom_retries,
                            )
                            ceiling = max(1, ceiling - 1)
                            self._log.warning(
                                "worker_child_died_unreported",
                                extra={"run_id": run_id, "pid": dead_pid,
                                       "worker_id": dead_wid, "exitcode": dead.exitcode,
                                       "reclaim": reclaim},
                            )
                    continue

                entry = in_flight.pop(pid, None)
                if entry is not None:
                    entry[0].join()
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
            # Teardown safety net: join any survivors and recover a task they may
            # have left non-terminal (early stop, or death during join).
            for proc, wid, _dev in in_flight.values():
                proc.join()
                self._store.reclaim_dead_worker(
                    run_id=run_id, worker_id=wid, max_reclaims=self._max_oom_retries,
                )

        failed_tasks = [
            t for t in self._store.list_tasks(run_id)
            if t.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.DEGENERATE)
        ]
        if failed_tasks:
            raise DegradedRunError(
                run_id=run_id, failed_tasks=failed_tasks, succeeded=succeeded
            )
        return succeeded

    def _prime_for_child(self, *, run_id: str, seed: int | None, max_oom_retries: int) -> None:
        """Prepare a freshly-built worker inside a spawned child.

        Sets the OOM budget and pre-marks the run as started with its seed cached, so
        run_once applies the seed but does NOT re-fire before_run_start (the parent
        fired it once before spawning). One explicit entry point instead of poking
        several private attributes from the subprocess module function.
        """
        self._max_oom_retries = max_oom_retries
        self._started_runs.add(run_id)
        self._run_seeds[run_id] = seed

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


def _call_factory(spec: str):
    """Import a 'module:function' factory and call it with no args, returning the result.

    Used to (re)build hooks/sandbox identically in the parent and in spawned children.
    """
    import importlib

    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise ValueError(f"factory spec must be 'module:function', got {spec!r}")
    fn = getattr(importlib.import_module(module_name), func_name)
    return fn()


def _run_once_subprocess(
    db_path: str,
    run_id: str,
    worker_id: str,
    handler: TaskHandler,
    seed: int | None,
    max_oom_retries: int,
    hooks_factory: str | None,
    sandbox_factory: str | None,
    gpu_device: int | None,
    result_q: "mp.Queue",
) -> None:
    """Module-level entrypoint for a spawned child: run exactly one task.

    Must be importable (top-level) because ``spawn`` pickles the target by
    reference. A fresh interpreter means an isolated process-global RNG, which is
    why drain_concurrent is reproducible. The child rebuilds a worker from *db_path*
    (SQLite is process-safe — fresh connection per call), recreating hooks/sandbox
    from the factory specs so behavior matches sequential mode, and runs one
    ``run_once``; the outcome is relayed to the parent via *result_q*.

    *gpu_device* (when set) is pinned via CUDA_VISIBLE_DEVICES BEFORE the handler can
    import torch, so the task uses exactly that physical GPU as cuda:0.

    The seed is pre-seeded and the run pre-marked started so run_once applies the
    seed without re-firing before_run_start (the parent fired it once).
    """
    if gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

    from finetuneharness.state.sqlite import SQLiteStateStore

    store = SQLiteStateStore(Path(db_path))
    worker = LocalWorker(
        worker_id=worker_id, store=store,
        hooks_factory=hooks_factory, sandbox_factory=sandbox_factory,
    )
    worker._prime_for_child(run_id=run_id, seed=seed, max_oom_retries=max_oom_retries)
    try:
        task = worker.run_once(run_id=run_id, handler=handler)
        result_q.put((os.getpid(), "succeeded" if task is not None else "empty"))
    except Exception as exc:  # noqa: BLE001 — outcome relayed to the parent
        result_q.put((os.getpid(), "oom" if is_oom_error(exc) else "error"))
