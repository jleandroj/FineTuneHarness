"""Real multi-PROCESS execution: the only true parallelism mode.

The harness has no in-process worker pool; scale comes from running several
`finetuneharness run` processes against the same SQLite store. These tests spawn
actual OS processes (not threads) and assert the lease (BEGIN IMMEDIATE +
SELECT/UPDATE under WAL) makes every task run exactly once with no double
execution, even under contention.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from finetuneharness.executor.worker import DegradedRunError, LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CONFIG = {
    "project": {"name": "multiproc"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:test",
}


def _drain_in_process(db_path: str, run_id: str, marker_dir: str, worker_id: str, barrier) -> None:
    """Child-process entrypoint: drain the run, dropping one marker file per
    handler invocation so the parent can detect any double execution.

    Both workers wait on *barrier* before draining so they are provably running
    concurrently (no startup skew where one drains the whole queue first)."""
    store = SQLiteStateStore(Path(db_path))
    worker = LocalWorker(worker_id=worker_id, store=store)

    def handler(task):
        # A unique file per *invocation*; the parent counts files per task_key,
        # so a task executed twice would leave two files.
        marker = Path(marker_dir) / f"{task.task_key}__{os.getpid()}__{uuid.uuid4().hex}"
        marker.write_text("x")
        time.sleep(0.02)  # widen the window for lease races
        return {"accuracy": 0.9, "f1": 0.88}

    try:
        barrier.wait(timeout=30)
    except Exception:
        pass
    try:
        worker.drain(run_id=run_id, handler=handler)
    except DegradedRunError:
        pass  # other workers may have drained the queue first; not an error here


def test_two_processes_execute_each_task_exactly_once(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    store = SQLiteStateStore(db)
    runner = FineTuneRunner(store)
    keys = [f"t{i}" for i in range(24)]
    run_id = runner.create_run(
        name="mp", config=_CONFIG, tasks=[{"task_key": k} for k in keys]
    )

    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(2)
    procs = [
        ctx.Process(target=_drain_in_process, args=(str(db), run_id, str(marker_dir), f"w{i}", barrier))
        for i in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"worker process exited with {p.exitcode}"

    # Every task reached SUCCEEDED.
    tasks = store.list_tasks(run_id)
    assert {t.task_key for t in tasks} == set(keys)
    assert all(t.status is TaskStatus.SUCCEEDED for t in tasks), {
        t.task_key: t.status.value for t in tasks
    }

    # No task was executed twice: exactly one marker file per task_key.
    files = list(marker_dir.iterdir())
    per_task = Counter(f.name.split("__")[0] for f in files)
    assert sum(per_task.values()) == len(keys), f"expected {len(keys)} executions, got {dict(per_task)}"
    dupes = {k: v for k, v in per_task.items() if v != 1}
    assert not dupes, f"tasks executed more than once: {dupes}"

    # Both processes actually did work — confirms real parallelism, not one
    # process starving the other.
    pids = {f.name.split("__")[1] for f in files}
    assert len(pids) == 2, f"expected both workers to execute tasks, saw pids {pids}"
