"""E2E: create run → execute all tasks → verify COMPLETED state → compare two runs."""
from __future__ import annotations

from pathlib import Path

import pytest

from finetuneharness.artifacts.store import ArtifactStore
from finetuneharness.evaluation.comparator import compare_runs
from finetuneharness.evaluation.report import format_report
from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import RunStatus, TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CONFIG = {
    "project": {"name": "e2e-pipeline"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:test",
}


def _make_store(tmp_path: Path) -> SQLiteStateStore:
    return SQLiteStateStore(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# full happy path
# ---------------------------------------------------------------------------

def test_create_execute_verify_completed(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    run_id = runner.create_run(
        name="full-run",
        config=_CONFIG,
        tasks=[
            {"task_key": f"cell-{i}", "kind": "train"}
            for i in range(5)
        ],
    )

    worker = LocalWorker(worker_id="w1", store=store)
    handler = lambda task: {"accuracy": 0.9, "loss": 0.1}

    for _ in range(5):
        task = worker.run_once(run_id=run_id, handler=handler)
        assert task is not None

    run = store.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED

    tasks = store.list_tasks(run_id)
    assert all(t.status == TaskStatus.SUCCEEDED for t in tasks)
    assert all(t.result is not None for t in tasks)
    assert all("wall_seconds" in t.result for t in tasks)  # type: ignore[index]


def test_partial_failure_run_status(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    run_id = runner.create_run(
        name="partial-run",
        config=_CONFIG,
        tasks=[{"task_key": "ok"}, {"task_key": "bad"}],
    )

    worker = LocalWorker(worker_id="w1", store=store)

    worker.run_once(run_id=run_id, handler=lambda task: {"done": True})
    try:
        worker.run_once(run_id=run_id, handler=lambda task: (_ for _ in ()).throw(RuntimeError("fail")))
    except RuntimeError:
        pass

    run = store.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.PARTIAL_FAILED


# ---------------------------------------------------------------------------
# compare two real runs end-to-end
# ---------------------------------------------------------------------------

def test_compare_two_completed_runs(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    def _run(name: str, accuracy: float) -> str:
        run_id = runner.create_run(
            name=name,
            config=_CONFIG,
            tasks=[{"task_key": "cell-train"}, {"task_key": "cell-eval"}],
        )
        worker = LocalWorker(worker_id="w1", store=store)
        for _ in range(2):
            worker.run_once(run_id=run_id, handler=lambda task: {"accuracy": accuracy, "loss": 1 - accuracy})
        return run_id

    baseline_id = _run("baseline", accuracy=0.80)
    experiment_id = _run("experiment", accuracy=0.88)

    report = compare_runs([baseline_id, experiment_id], store)

    assert report.regressions == []
    assert report.improvements == []
    assert report.snapshots[baseline_id].success_rate == 1.0
    assert report.snapshots[experiment_id].success_rate == 1.0

    for tc in report.task_comparisons:
        delta = tc.metric_deltas.get("accuracy", {}).get(experiment_id)
        assert delta is not None
        assert abs(delta - 0.08) < 1e-9

    text = format_report(report)
    assert "baseline" in text
    assert "experiment" in text


# ---------------------------------------------------------------------------
# artifacts end-to-end
# ---------------------------------------------------------------------------

def test_artifacts_written_and_registered(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    artifact_store = ArtifactStore(root=tmp_path / "artifacts", state_store=store)
    runner = FineTuneRunner(store)

    run_id = runner.create_run(
        name="artifact-run",
        config=_CONFIG,
        tasks=[{"task_key": "train"}],
    )

    worker = LocalWorker(worker_id="w1", store=store, artifact_store=artifact_store)
    worker.run_once(run_id=run_id, handler=lambda task: {"accuracy": 0.95})

    artifacts = store.list_artifacts(run_id)
    assert len(artifacts) == 1
    assert artifacts[0].kind == "task_result"
    assert artifacts[0].checksum  # SHA-256 must be present

    artifact_path = Path(artifacts[0].path)
    assert artifact_path.exists()
    assert artifact_path.name == "train-result.json"


# ---------------------------------------------------------------------------
# run survives process-restart simulation (state persisted in SQLite)
# ---------------------------------------------------------------------------

def test_run_state_survives_store_reinit(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"

    store1 = SQLiteStateStore(db_path)
    runner1 = FineTuneRunner(store1)
    run_id = runner1.create_run(
        name="persistent-run",
        config=_CONFIG,
        tasks=[{"task_key": "cell-1"}, {"task_key": "cell-2"}],
    )

    # Complete one task, then simulate restart by creating a new store instance
    worker1 = LocalWorker(worker_id="w1", store=store1)
    worker1.run_once(run_id=run_id, handler=lambda task: {"done": True})

    # New store instance — simulates process restart
    store2 = SQLiteStateStore(db_path)
    tasks = store2.list_tasks(run_id)
    succeeded = [t for t in tasks if t.status == TaskStatus.SUCCEEDED]
    pending = [t for t in tasks if t.status == TaskStatus.PENDING]

    assert len(succeeded) == 1
    assert len(pending) == 1

    # Resume with new store
    worker2 = LocalWorker(worker_id="w2", store=store2)
    worker2.run_once(run_id=run_id, handler=lambda task: {"done": True})

    run = store2.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
