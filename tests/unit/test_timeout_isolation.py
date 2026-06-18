"""Timeout isolation: a hung handler must not degrade the worker (P1 regression).

These tests pin the two correctness properties that the shared-pool design
violated:
  1. A hung handler never starves later tasks (no false TIMED_OUT by starvation).
  2. A timed-out task stays TIMED_OUT and registers no result, even after its
     abandoned handler thread eventually finishes.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from finetuneharness.executor.policy import TimeoutPolicy
from finetuneharness.executor.worker import DegradedRunError, LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore


_CONFIG = {
    "project": {"name": "demo"}, "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"}, "seed": 42, "dataset_hash": "sha256:test",
}


def test_hung_handlers_do_not_starve_a_later_task(tmp_path: Path) -> None:
    """Two handlers hang and time out; a third healthy task must still SUCCEED.

    Under the old shared bounded pool, accumulated hung threads would occupy all
    worker slots and the healthy task would be marked TIMED_OUT without ever
    running. With per-task executors it runs normally.
    """
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="starvation-test",
        config=_CONFIG,
        # task_key order = lease order: the two hangs are leased before the good one
        tasks=[
            {"task_key": "a-hang-1"},
            {"task_key": "b-hang-2"},
            {"task_key": "c-good"},
        ],
    )

    def handler(task):
        if "hang" in task.task_key:
            time.sleep(10)  # far exceeds the 1s timeout; thread is abandoned
            return {}
        return {"ok": True}

    worker = LocalWorker(
        worker_id="w1", store=store,
        timeout_policy=TimeoutPolicy(timeout_seconds=1),
    )

    try:
        worker.drain(run_id=run_id, handler=handler)
    except DegradedRunError:
        pass  # expected: the two hang tasks ended TIMED_OUT

    by_key = {t.task_key: t for t in store.list_tasks(run_id)}
    assert by_key["a-hang-1"].status is TaskStatus.TIMED_OUT
    assert by_key["b-hang-2"].status is TaskStatus.TIMED_OUT
    assert by_key["c-good"].status is TaskStatus.SUCCEEDED, (
        "healthy task was starved by hung handlers — cascade bug regressed"
    )


def test_timed_out_task_registers_no_result_after_thread_finishes(tmp_path: Path) -> None:
    """The abandoned handler thread completing later must not flip state or add a result."""
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="post-timeout-write",
        config=_CONFIG,
        tasks=[{"task_key": "cell-1"}],
    )

    finished = threading.Event()

    def slow_handler(task):
        time.sleep(2)            # exceeds the 1s timeout
        finished.set()           # the abandoned thread reaches here AFTER timeout
        return {"sneaky": "late-result"}

    worker = LocalWorker(
        worker_id="w1", store=store,
        timeout_policy=TimeoutPolicy(timeout_seconds=1),
    )

    try:
        worker.run_once(run_id=run_id, handler=slow_handler)
    except TimeoutError:
        pass

    task = store.list_tasks(run_id)[0]
    assert task.status is TaskStatus.TIMED_OUT
    assert task.result is None

    # Wait for the abandoned thread to run to completion, then re-check.
    assert finished.wait(timeout=5), "handler thread never finished"
    time.sleep(0.05)
    task = store.list_tasks(run_id)[0]
    assert task.status is TaskStatus.TIMED_OUT, "state flipped after timeout"
    assert task.result is None, "abandoned thread leaked a result into the store"
    assert store.list_artifacts(run_id) == [], "timed-out task registered an artifact"
