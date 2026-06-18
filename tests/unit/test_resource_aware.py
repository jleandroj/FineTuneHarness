"""Resource-aware, process-isolated concurrent draining (drain_concurrent).

drain_concurrent runs each task in its own forked process so per-task seeding
stays reproducible. These tests observe across processes via shared memory
(mp.Value) and marker files, since a child's in-memory state is a fork copy the
parent never sees.
"""
from __future__ import annotations

import multiprocessing as mp
import random
from pathlib import Path

import pytest

from finetuneharness.executor.resources import ConcurrencyConfig, is_oom_error
from finetuneharness.executor.worker import DegradedRunError, LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CTX = mp.get_context("fork")

_CONFIG = {
    "project": {"name": "resaware"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:test",
}


class _FakeMonitor:
    """Returns a fixed free-memory figure; None means 'no GPU'."""

    def __init__(self, free_mb: float | None) -> None:
        self._free = free_mb

    def free_gpu_memory_mb(self) -> float | None:
        return self._free


def _make_run(tmp_path: Path, keys: list[str]) -> tuple[SQLiteStateStore, str]:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="r", config=_CONFIG, tasks=[{"task_key": k} for k in keys]
    )
    return store, run_id


def _resource_aware(**kw) -> ConcurrencyConfig:
    kw.setdefault("mode", "resource_aware")
    kw.setdefault("min_free_mb", 1000)
    kw.setdefault("settle_seconds", 0.0)
    return ConcurrencyConfig(**kw)


# ── is_oom_error classification ──────────────────────────────────────────────

def test_is_oom_error_detects_cuda_messages() -> None:
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate ..."))
    assert is_oom_error(RuntimeError("handler raised in sandbox: OutOfMemoryError: ..."))

    class OutOfMemoryError(Exception):
        pass

    assert is_oom_error(OutOfMemoryError("boom"))
    assert not is_oom_error(ValueError("unrelated failure"))


# ── store guard ──────────────────────────────────────────────────────────────

def test_drain_concurrent_requires_persistent_store() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = runner.create_run(name="m", config=_CONFIG, tasks=[{"task_key": "t"}])
    worker = LocalWorker(worker_id="w", store=store)
    with pytest.raises(TypeError, match="persistent"):
        worker.drain_concurrent(
            run_id=run_id, handler=lambda t: {"accuracy": 0.9, "f1": 0.88},
            concurrency=_resource_aware(), monitor=_FakeMonitor(50000),
        )


# ── concurrency ──────────────────────────────────────────────────────────────

def test_drain_concurrent_runs_tasks_in_parallel(tmp_path: Path) -> None:
    """With plentiful memory, several child processes run at once; each task once."""
    keys = [f"t{i}" for i in range(6)]
    store, run_id = _make_run(tmp_path, keys)
    worker = LocalWorker(worker_id="w", store=store)

    live = _CTX.Value("i", 0)
    max_live = _CTX.Value("i", 0)
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    def handler(task):
        import os
        import time as _t
        with live.get_lock():
            live.value += 1
            if live.value > max_live.value:
                max_live.value = live.value
        (marker_dir / f"{task.task_key}__{os.getpid()}").write_text("x")
        _t.sleep(0.15)
        with live.get_lock():
            live.value -= 1
        return {"accuracy": 0.9, "f1": 0.88}

    succeeded = worker.drain_concurrent(
        run_id=run_id, handler=handler,
        concurrency=_resource_aware(max_concurrent=4),
        monitor=_FakeMonitor(50000),
    )

    assert succeeded == 6
    assert max_live.value >= 2, "expected real concurrency, ran effectively sequentially"
    # Each task executed exactly once.
    per_task = [f.name.split("__")[0] for f in marker_dir.iterdir()]
    assert sorted(per_task) == sorted(keys)
    assert all(t.status is TaskStatus.SUCCEEDED for t in store.list_tasks(run_id))


def test_drain_concurrent_no_gpu_falls_back_to_sequential(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path, ["a", "b", "c"])
    worker = LocalWorker(worker_id="w", store=store)

    succeeded = worker.drain_concurrent(
        run_id=run_id, handler=lambda t: {"accuracy": 0.9, "f1": 0.88},
        concurrency=_resource_aware(), monitor=_FakeMonitor(None),
    )
    assert succeeded == 3
    assert all(t.status is TaskStatus.SUCCEEDED for t in store.list_tasks(run_id))


def test_low_memory_serializes_admission(tmp_path: Path) -> None:
    """Free memory below the headroom forces one task at a time."""
    keys = [f"t{i}" for i in range(4)]
    store, run_id = _make_run(tmp_path, keys)
    worker = LocalWorker(worker_id="w", store=store)

    live = _CTX.Value("i", 0)
    max_live = _CTX.Value("i", 0)

    def handler(task):
        import time as _t
        with live.get_lock():
            live.value += 1
            if live.value > max_live.value:
                max_live.value = live.value
        _t.sleep(0.05)
        with live.get_lock():
            live.value -= 1
        return {"accuracy": 0.9, "f1": 0.88}

    succeeded = worker.drain_concurrent(
        run_id=run_id, handler=handler,
        concurrency=_resource_aware(max_concurrent=4, min_free_mb=1000),
        monitor=_FakeMonitor(500),  # below headroom
    )
    assert succeeded == 4
    assert max_live.value == 1, "low memory must force one-at-a-time admission"


# ── reproducibility (item 3: determinism under concurrency) ──────────────────

def test_seeding_is_deterministic_under_concurrency(tmp_path: Path) -> None:
    """Per-task RNG is isolated per process: concurrent runs are reproducible.

    Every task is seeded with the run seed before its handler runs, so each task's
    global-RNG draw is identical and stable across executions. If concurrency
    corrupted the shared RNG (the thread-based bug), these draws would vary.
    """
    def _execute_once(subdir: str) -> dict[str, str]:
        out = tmp_path / subdir
        out.mkdir()
        store = SQLiteStateStore(tmp_path / f"{subdir}.db")
        runner = FineTuneRunner(store)
        run_id = runner.create_run(
            name=subdir, config=_CONFIG, tasks=[{"task_key": f"t{i}"} for i in range(5)]
        )
        worker = LocalWorker(worker_id="w", store=store)

        def handler(task):
            # run_once applies the run seed before calling us, so this global-RNG
            # draw is deterministic and identical for every task.
            (out / task.task_key).write_text(repr(random.random()))
            return {"accuracy": 0.9, "f1": 0.88}

        worker.drain_concurrent(
            run_id=run_id, handler=handler,
            concurrency=_resource_aware(max_concurrent=4),
            monitor=_FakeMonitor(50000),
        )
        return {f.name: f.read_text() for f in out.iterdir()}

    run_a = _execute_once("a")
    run_b = _execute_once("b")

    assert len(run_a) == 5 and len(run_b) == 5
    # All tasks share one seed -> one identical draw, stable across both runs.
    assert len(set(run_a.values())) == 1, f"per-task RNG not isolated: {run_a}"
    assert set(run_a.values()) == set(run_b.values()), "not reproducible across runs"


# ── OOM handling ─────────────────────────────────────────────────────────────

def test_oom_task_is_requeued_then_succeeds(tmp_path: Path) -> None:
    """A task that OOMs once is requeued (not failed) and succeeds on retry.

    The OOM budget is persisted (events), so it survives across the separate
    processes each attempt runs in.
    """
    store, run_id = _make_run(tmp_path, ["flaky"])
    worker = LocalWorker(worker_id="w", store=store)
    attempt_dir = tmp_path / "attempts"
    attempt_dir.mkdir()

    def handler(task):
        import os
        prior = len(list(attempt_dir.glob(f"{task.task_key}__*")))
        (attempt_dir / f"{task.task_key}__{os.getpid()}").write_text("x")
        if prior == 0:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return {"accuracy": 0.9, "f1": 0.88}

    succeeded = worker.drain_concurrent(
        run_id=run_id, handler=handler,
        concurrency=_resource_aware(max_concurrent=2, max_oom_retries=3),
        monitor=_FakeMonitor(50000),
    )
    assert succeeded == 1
    assert store.list_tasks(run_id)[0].status is TaskStatus.SUCCEEDED
    kinds = [e.kind for e in store.list_events(run_id)]
    assert "task_oom_requeued" in kinds


def test_oom_task_fails_after_exhausting_retries(tmp_path: Path) -> None:
    """A task that always OOMs is FAILED after max_oom_retries, surfacing as degraded."""
    store, run_id = _make_run(tmp_path, ["doomed"])
    worker = LocalWorker(worker_id="w", store=store)

    def handler(task):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(DegradedRunError):
        worker.drain_concurrent(
            run_id=run_id, handler=handler,
            concurrency=_resource_aware(max_concurrent=2, max_oom_retries=2),
            monitor=_FakeMonitor(50000),
        )
    assert store.list_tasks(run_id)[0].status is TaskStatus.FAILED
    requeues = [e for e in store.list_events(run_id) if e.kind == "task_oom_requeued"]
    assert len(requeues) == 2, "should requeue exactly max_oom_retries times before failing"
