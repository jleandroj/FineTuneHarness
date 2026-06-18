"""Integration: multiple workers drain a run completely with no double-execution."""
from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

import pytest

from finetuneharness.executor.policy import RetryPolicy, TimeoutPolicy
from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import RunStatus, TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CONFIG = {
    "project": {"name": "drain-test"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:test",
}


def _make_store(tmp_path: Path) -> SQLiteStateStore:
    return SQLiteStateStore(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# worker loop until complete
# ---------------------------------------------------------------------------

def _drain(store: SQLiteStateStore, run_id: str, worker_id: str, results: list, lock: threading.Lock) -> None:
    worker = LocalWorker(worker_id=worker_id, store=store)
    while True:
        task = worker.run_once(
            run_id=run_id,
            handler=lambda t: {"done": t.task_key},
        )
        if task is None:
            break
        with lock:
            results.append(task.task_key)


def test_single_worker_drains_all_tasks(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="drain-single",
        config=_CONFIG,
        tasks=[{"task_key": f"t{i}"} for i in range(6)],
    )

    results: list[str] = []
    lock = threading.Lock()
    _drain(store, run_id, "w1", results, lock)

    assert sorted(results) == [f"t{i}" for i in range(6)]
    run = store.get_run(run_id)
    assert run is not None and run.status == RunStatus.COMPLETED


def test_two_workers_drain_all_tasks_no_duplicates(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="drain-two",
        config=_CONFIG,
        tasks=[{"task_key": f"t{i}"} for i in range(10)],
    )

    results: list[str] = []
    lock = threading.Lock()

    threads = [
        threading.Thread(target=_drain, args=(store, run_id, f"w{i}", results, lock))
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    counts = Counter(results)
    assert all(v == 1 for v in counts.values()), f"Duplicates: {counts}"
    assert sorted(results) == [f"t{i}" for i in range(10)]

    run = store.get_run(run_id)
    assert run is not None and run.status == RunStatus.COMPLETED


def test_four_workers_drain_large_run(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="drain-four",
        config=_CONFIG,
        tasks=[{"task_key": f"t{i:03d}"} for i in range(40)],
    )

    results: list[str] = []
    lock = threading.Lock()

    threads = [
        threading.Thread(target=_drain, args=(store, run_id, f"w{i}", results, lock))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    counts = Counter(results)
    assert all(v == 1 for v in counts.values()), f"Duplicates: {counts}"
    assert len(results) == 40

    run = store.get_run(run_id)
    assert run is not None and run.status == RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# retry within drain
# ---------------------------------------------------------------------------

def test_drain_with_flaky_tasks(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="drain-flaky",
        config=_CONFIG,
        tasks=[{"task_key": f"t{i}", "max_attempts": 2} for i in range(4)],
    )

    attempt_counts: dict[str, int] = {}
    lock = threading.Lock()

    def flaky_handler(task):
        with lock:
            attempt_counts[task.task_key] = attempt_counts.get(task.task_key, 0) + 1
            if attempt_counts[task.task_key] == 1 and task.task_key == "t0":
                raise RuntimeError("first attempt fails for t0")
        return {"done": task.task_key}

    worker = LocalWorker(
        worker_id="w1",
        store=store,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    for _ in range(6):  # 4 tasks + 1 retry for t0
        try:
            result = worker.run_once(run_id=run_id, handler=flaky_handler)
        except RuntimeError:
            pass

    tasks = store.list_tasks(run_id)
    assert all(t.status == TaskStatus.SUCCEEDED for t in tasks)
    assert attempt_counts.get("t0", 0) == 2


# ---------------------------------------------------------------------------
# run stays FAILED when all tasks fail
# ---------------------------------------------------------------------------

def test_all_tasks_fail_run_is_failed(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="all-fail",
        config=_CONFIG,
        tasks=[{"task_key": f"t{i}"} for i in range(3)],
    )

    worker = LocalWorker(worker_id="w1", store=store)
    for _ in range(3):
        try:
            worker.run_once(run_id=run_id, handler=lambda t: (_ for _ in ()).throw(RuntimeError("fail")))
        except RuntimeError:
            pass

    run = store.get_run(run_id)
    assert run is not None and run.status == RunStatus.FAILED
