"""Tests that timeout is preemptive: the worker unblocks even when the handler is still running."""
from __future__ import annotations

import time
from pathlib import Path

from finetuneharness.executor.policy import TimeoutPolicy
from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore


def test_timeout_unblocks_worker_before_handler_finishes(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="hang-test",
        config={"project": {"name": "demo"}, "executor": {"kind": "local"}, "artifacts": {"root": "./artifacts"}, "seed": 42, "dataset_hash": "sha256:test"},
        tasks=[{"task_key": "cell-1", "kind": "train"}],
    )
    worker = LocalWorker(
        worker_id="w1",
        store=store,
        timeout_policy=TimeoutPolicy(timeout_seconds=1),
    )

    started = time.monotonic()
    try:
        # Handler sleeps 3s; timeout is 1s — worker must unblock in < 3s
        worker.run_once(run_id=run_id, handler=lambda task: time.sleep(3) or {})
    except TimeoutError:
        pass
    elapsed = time.monotonic() - started

    assert elapsed < 2.5, f"worker took {elapsed:.1f}s — timeout did not preempt the handler"
    task = store.list_tasks(run_id)[0]
    assert task.status == TaskStatus.TIMED_OUT


def test_task_without_timeout_runs_to_completion(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="no-timeout-test",
        config={"project": {"name": "demo"}, "executor": {"kind": "local"}, "artifacts": {"root": "./artifacts"}, "seed": 42, "dataset_hash": "sha256:test"},
        tasks=[{"task_key": "cell-1", "kind": "train"}],
    )
    worker = LocalWorker(worker_id="w1", store=store)
    task = worker.run_once(run_id=run_id, handler=lambda task: {"ok": True})

    assert task is not None
    assert store.list_tasks(run_id)[0].status == TaskStatus.SUCCEEDED
