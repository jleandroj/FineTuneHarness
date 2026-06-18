import pytest
from pathlib import Path

from finetuneharness.executor.worker import DegradedRunError, LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore


def test_worker_leases_and_completes_one_task(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="worker-demo",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": "./artifacts"},
            "seed": 42,
            "dataset_hash": "sha256:test",
        },
        tasks=[{"task_key": "cell-1", "kind": "train"}],
    )

    worker = LocalWorker(worker_id="worker-a", store=store)
    task = worker.run_once(run_id=run_id, handler=lambda task: {"ok": True, "task_key": task.task_key})

    assert task is not None
    tasks = store.list_tasks(run_id)
    assert len(tasks) == 1
    assert tasks[0].result is not None
    assert tasks[0].result["ok"] is True
    assert tasks[0].result["task_key"] == "cell-1"
    assert "wall_seconds" in tasks[0].result


def _make_run(tmp_path: Path, task_keys: list[str]) -> tuple[SQLiteStateStore, str]:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="drain-test",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": "./artifacts"},
            "seed": 42,
            "dataset_hash": "sha256:test",
        },
        tasks=[{"task_key": k, "kind": "train"} for k in task_keys],
    )
    return store, run_id


def test_drain_all_succeed(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path, ["cell-1", "cell-2", "cell-3"])
    worker = LocalWorker(worker_id="w", store=store)

    succeeded = worker.drain(run_id=run_id, handler=lambda t: {"ok": True})

    assert succeeded == 3
    statuses = {t.task_key: t.status for t in store.list_tasks(run_id)}
    assert all(s == TaskStatus.SUCCEEDED for s in statuses.values())


def test_drain_continues_past_failure(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path, ["cell-1", "cell-2", "cell-3"])
    worker = LocalWorker(worker_id="w", store=store)

    def handler(task):
        if task.task_key == "cell-2":
            raise RuntimeError("OOM")
        return {"ok": True}

    with pytest.raises(DegradedRunError) as exc_info:
        worker.drain(run_id=run_id, handler=handler)

    err = exc_info.value
    assert err.succeeded == 2
    assert len(err.failed_tasks) == 1
    assert err.failed_tasks[0].task_key == "cell-2"
    assert "1/3" in str(err)

    statuses = {t.task_key: t.status for t in store.list_tasks(run_id)}
    assert statuses["cell-1"] == TaskStatus.SUCCEEDED
    assert statuses["cell-2"] == TaskStatus.FAILED
    assert statuses["cell-3"] == TaskStatus.SUCCEEDED


def test_drain_multiple_failures(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path, ["a", "b", "c", "d"])
    worker = LocalWorker(worker_id="w", store=store)

    def handler(task):
        if task.task_key in ("b", "d"):
            raise ValueError("bad config")
        return {"ok": True}

    with pytest.raises(DegradedRunError) as exc_info:
        worker.drain(run_id=run_id, handler=handler)

    err = exc_info.value
    assert err.succeeded == 2
    assert len(err.failed_tasks) == 2
    failed_keys = {t.task_key for t in err.failed_tasks}
    assert failed_keys == {"b", "d"}


def test_drain_returns_count_when_all_succeed(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path, ["x", "y"])
    worker = LocalWorker(worker_id="w", store=store)

    result = worker.drain(run_id=run_id, handler=lambda t: {"v": 1})

    assert result == 2


def test_worker_stamps_validation_status_on_result(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path, ["cell-1"])
    worker = LocalWorker(worker_id="w", store=store)

    worker.run_once(
        run_id=run_id,
        handler=lambda t: {"accuracy": 0.91, "loss": 0.3},
    )

    result = store.list_tasks(run_id)[0].result
    assert result is not None
    assert "_validation_status" in result
    assert result["_validation_status"] == "SUCCEEDED_VALIDATED"


def test_worker_stamps_degenerate_on_adapter_not_loaded(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path, ["adalora-cell"])
    worker = LocalWorker(worker_id="w", store=store)

    worker.run_once(
        run_id=run_id,
        handler=lambda t: {
            "method": "adalora",
            "accuracy": 0.5326,
            "adapter_loaded": False,
        },
    )

    task = store.list_tasks(run_id)[0]
    assert task.status == TaskStatus.SUCCEEDED
    assert task.result["_validation_status"] == "DEGENERATE_RESULT"
    assert any("adapter_loaded" in e for e in task.result["_validation_errors"])


def test_worker_max_workers_passed_to_thread_pool(tmp_path: Path) -> None:
    """max_workers must be wired through to the ThreadPoolExecutor, not hardcoded."""
    store, run_id = _make_run(tmp_path, ["t1"])
    worker = LocalWorker(worker_id="w", store=store, max_workers=2)
    assert worker._executor._max_workers == 2


def test_worker_default_max_workers_is_4(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path, ["t1"])
    worker = LocalWorker(worker_id="w", store=store)
    assert worker._executor._max_workers == 4


def test_drain_stop_fn_halts_after_n_tasks(tmp_path: Path) -> None:
    """stop_fn returning True must stop drain() — remaining tasks stay PENDING."""
    keys = [f"t{i}" for i in range(6)]
    store, run_id = _make_run(tmp_path, keys)
    worker = LocalWorker(worker_id="w", store=store)

    completed: list[str] = []
    stop_after = 3

    def handler(task):
        completed.append(task.task_key)
        return {"ok": True}

    def stop_fn():
        return len(completed) >= stop_after

    succeeded = worker.drain(run_id=run_id, handler=handler, stop_fn=stop_fn)

    assert succeeded == stop_after
    assert len(completed) == stop_after
    tasks = store.list_tasks(run_id)
    pending = [t for t in tasks if t.status == TaskStatus.PENDING]
    assert len(pending) == len(keys) - stop_after


def test_drain_without_stop_fn_runs_all(tmp_path: Path) -> None:
    """Without stop_fn, drain() runs every task (default behavior unchanged)."""
    keys = [f"t{i}" for i in range(5)]
    store, run_id = _make_run(tmp_path, keys)
    worker = LocalWorker(worker_id="w", store=store)

    succeeded = worker.drain(run_id=run_id, handler=lambda t: {"ok": True})
    assert succeeded == len(keys)


def test_drain_stop_fn_false_does_not_halt(tmp_path: Path) -> None:
    """stop_fn that always returns False must not stop drain()."""
    keys = [f"t{i}" for i in range(4)]
    store, run_id = _make_run(tmp_path, keys)
    worker = LocalWorker(worker_id="w", store=store)

    succeeded = worker.drain(
        run_id=run_id,
        handler=lambda t: {"ok": True},
        stop_fn=lambda: False,
    )
    assert succeeded == len(keys)


def test_drain_early_stopping_hook_integration(tmp_path: Path) -> None:
    """EarlyStoppingHook.should_stop wired to stop_fn must halt the grid."""
    from finetuneharness.orchestrator.hooks import HookRegistry
    from finetuneharness.registry.hooks import EarlyStoppingHook

    keys = [f"t{i}" for i in range(10)]
    store, run_id = _make_run(tmp_path, keys)

    es = EarlyStoppingHook(metric="accuracy", patience=2, min_delta=0.01, mode="max")
    registry = HookRegistry()
    registry.register("after_task_success", es.after_task_success)

    worker = LocalWorker(worker_id="w", store=store, hooks=registry)

    call_count = 0

    def handler(task):
        nonlocal call_count
        call_count += 1
        # Flat accuracy — no improvement, so EarlyStopping triggers after patience=2
        return {"accuracy": 0.7}

    worker.drain(run_id=run_id, handler=handler, stop_fn=es.should_stop)

    # patience=2 means: first task sets best=0.7, second and third don't improve →
    # counter reaches 2 after the 3rd task → should_stop=True → drain stops at 3.
    # (first task: best=0.7, counter=0; second: no improvement, counter=1;
    #  third: no improvement, counter=2 → should_stop=True → stop after 3rd)
    assert call_count == 3, f"Expected 3 tasks before early stop, got {call_count}"
