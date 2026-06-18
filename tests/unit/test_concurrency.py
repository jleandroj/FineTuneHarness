"""Tests that two concurrent workers don't double-execute any task (SQLite BEGIN IMMEDIATE)."""
from __future__ import annotations

import threading
from pathlib import Path

from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore


def test_two_workers_each_execute_exactly_one_unique_task(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="concurrency-test",
        config={"project": {"name": "demo"}, "executor": {"kind": "local"}, "artifacts": {"root": "./artifacts"}, "seed": 42, "dataset_hash": "sha256:test"},
        tasks=[
            {"task_key": "cell-1"},
            {"task_key": "cell-2"},
        ],
    )

    executed: list[str] = []
    lock = threading.Lock()

    def handler(task):
        with lock:
            executed.append(task.task_key)
        return {"done": task.task_key}

    errors: list[Exception] = []

    def run_worker(worker_id: str) -> None:
        worker = LocalWorker(worker_id=worker_id, store=store)
        try:
            worker.run_once(run_id=run_id, handler=handler)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=run_worker, args=("worker-a",))
    t2 = threading.Thread(target=run_worker, args=("worker-b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"worker errors: {errors}"
    assert sorted(executed) == ["cell-1", "cell-2"], f"unexpected executions: {executed}"

    tasks = store.list_tasks(run_id)
    assert all(t.status == TaskStatus.SUCCEEDED for t in tasks)


def test_no_task_executed_twice_under_race(tmp_path: Path) -> None:
    """Many workers racing for a single task — exactly one should win."""
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="race-test",
        config={"project": {"name": "demo"}, "executor": {"kind": "local"}, "artifacts": {"root": "./artifacts"}, "seed": 42, "dataset_hash": "sha256:test"},
        tasks=[{"task_key": "only-task"}],
    )

    executed: list[str] = []
    lock = threading.Lock()

    def handler(task):
        with lock:
            executed.append(task.task_key)
        return {"done": True}

    threads = [
        threading.Thread(
            target=lambda wid=f"worker-{i}": LocalWorker(worker_id=wid, store=store).run_once(
                run_id=run_id, handler=handler
            ),
        )
        for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert executed == ["only-task"], f"task executed {len(executed)} times: {executed}"
