"""Concurrency stress tests — barriers synchronize worker START, not handler entry."""
from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import RunStatus, TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CONFIG = {
    "project": {"name": "concurrency-stress"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:test",
}


def _make_store(tmp_path: Path) -> SQLiteStateStore:
    return SQLiteStateStore(tmp_path / "state.db")


def _run_worker(
    store: SQLiteStateStore,
    run_id: str,
    worker_id: str,
    executed: list[str],
    errors: list[Exception],
    lock: threading.Lock,
    start_barrier: threading.Barrier | None = None,
    handler_delay: float = 0.0,
) -> None:
    """Worker loop that drains a run.

    start_barrier synchronizes all workers to begin simultaneously — race window
    is at lease acquisition, not inside the handler (avoids barrier deadlock when
    fewer tasks than workers).
    """
    if start_barrier is not None:
        start_barrier.wait()

    worker = LocalWorker(worker_id=worker_id, store=store)

    def handler(task):
        if handler_delay > 0:
            time.sleep(handler_delay)
        return {"done": task.task_key}

    try:
        while True:
            task = worker.run_once(run_id=run_id, handler=handler)
            if task is None:
                break
            with lock:
                executed.append(task.task_key)
    except Exception as exc:
        with lock:
            errors.append(exc)


# ---------------------------------------------------------------------------
# 20 workers race for 1 task — exactly one wins
# ---------------------------------------------------------------------------

def test_20_workers_race_for_single_task(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="single-task-race",
        config=_CONFIG,
        tasks=[{"task_key": "only-task"}],
    )

    n_workers = 20
    barrier = threading.Barrier(n_workers)  # all workers start run_once simultaneously
    executed: list[str] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    threads = [
        threading.Thread(
            target=_run_worker,
            args=(store, run_id, f"w{i}", executed, errors, lock, barrier, 0.005),
        )
        for i in range(n_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not any(t.is_alive() for t in threads), "Workers did not finish — possible deadlock"
    assert not errors, f"Worker errors: {errors}"
    assert executed == ["only-task"], f"Task executed {len(executed)} times: {executed}"


# ---------------------------------------------------------------------------
# 8 workers drain 20 tasks — each task executed exactly once
# ---------------------------------------------------------------------------

def test_8_workers_drain_20_tasks_no_duplicates(tmp_path: Path) -> None:
    n_tasks = 20
    n_workers = 8
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="multi-task-drain",
        config=_CONFIG,
        tasks=[{"task_key": f"t{i:02d}"} for i in range(n_tasks)],
    )

    barrier = threading.Barrier(n_workers)
    executed: list[str] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    threads = [
        threading.Thread(
            target=_run_worker,
            args=(store, run_id, f"w{i}", executed, errors, lock, barrier, 0.002),
        )
        for i in range(n_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not any(t.is_alive() for t in threads), "Workers did not finish"
    assert not errors, f"Worker errors: {errors}"

    counts = Counter(executed)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    assert not duplicates, f"Duplicate executions: {duplicates}"
    assert len(executed) == n_tasks

    run = store.get_run(run_id)
    assert run is not None and run.status == RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# concurrent run_status refresh alongside execution — no corruption
# ---------------------------------------------------------------------------

def test_concurrent_status_refresh_is_safe(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="status-refresh",
        config=_CONFIG,
        tasks=[{"task_key": f"t{i}"} for i in range(10)],
    )

    errors: list[Exception] = []
    lock = threading.Lock()

    def refresh_loop() -> None:
        for _ in range(30):
            try:
                runner.refresh_run_status(run_id)
                time.sleep(0.002)
            except Exception as exc:
                with lock:
                    errors.append(exc)

    def worker_loop() -> None:
        worker = LocalWorker(worker_id=f"w-{threading.get_ident()}", store=store)
        while True:
            try:
                task = worker.run_once(run_id=run_id, handler=lambda t: {"done": True})
            except Exception as exc:
                with lock:
                    errors.append(exc)
                break
            if task is None:
                break

    threads = (
        [threading.Thread(target=refresh_loop) for _ in range(3)] +
        [threading.Thread(target=worker_loop) for _ in range(3)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not any(t.is_alive() for t in threads), "Threads did not finish"
    assert not errors, f"Errors during concurrent refresh+execution: {errors}"

    run = store.get_run(run_id)
    assert run is not None and run.status == RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# repeated runs under concurrent load — no deadlock
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_runs", [3])
def test_no_deadlock_under_repeated_concurrent_runs(tmp_path: Path, n_runs: int) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    for run_idx in range(n_runs):
        run_id = runner.create_run(
            name=f"deadlock-check-{run_idx}",
            config=_CONFIG,
            tasks=[{"task_key": f"t{i}"} for i in range(8)],
        )

        executed: list[str] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        threads = [
            threading.Thread(
                target=_run_worker,
                args=(store, run_id, f"w{i}", executed, errors, lock),
            )
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not any(t.is_alive() for t in threads), f"Deadlock in run {run_idx}"
        assert not errors
        assert len(executed) == 8
