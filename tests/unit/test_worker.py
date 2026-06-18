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
