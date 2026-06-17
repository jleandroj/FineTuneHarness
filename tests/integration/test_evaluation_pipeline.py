"""Integration: create runs with real workers → compare with evaluator."""
from __future__ import annotations

from pathlib import Path

import pytest

from finetuneharness.evaluation.comparator import compare_runs
from finetuneharness.evaluation.report import format_report, report_to_dict
from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import RunStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CONFIG = {
    "project": {"name": "eval-integration"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
}


def _make_store(tmp_path: Path) -> SQLiteStateStore:
    return SQLiteStateStore(tmp_path / "state.db")


def _run_experiment(store, runner, name: str, metrics: dict[str, float], fail_keys: set[str] | None = None) -> str:
    task_keys = list(metrics.keys())
    run_id = runner.create_run(
        name=name,
        config=_CONFIG,
        tasks=[{"task_key": k} for k in task_keys],
    )
    worker = LocalWorker(worker_id="w1", store=store)
    for _ in range(len(task_keys)):
        try:
            worker.run_once(
                run_id=run_id,
                handler=lambda task, _m=metrics, _f=fail_keys: (
                    (_ for _ in ()).throw(RuntimeError("injected failure"))
                    if _f and task.task_key in _f
                    else {"accuracy": _m[task.task_key]}
                ),
            )
        except RuntimeError:
            pass
    return run_id


def test_regression_detected_after_real_execution(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    baseline_id = _run_experiment(store, runner, "v1", {"cell-a": 0.90, "cell-b": 0.85})
    new_id = _run_experiment(store, runner, "v2", {"cell-a": 0.90}, fail_keys={"cell-b"})

    # cell-b: succeeded in baseline, failed in v2 → regression
    report = compare_runs([baseline_id, new_id], store)
    assert "cell-b" in report.regressions


def test_improvement_detected_after_real_execution(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    baseline_id = _run_experiment(store, runner, "v1", {"cell-a": 0.80}, fail_keys={"cell-a"})
    new_id = _run_experiment(store, runner, "v2", {"cell-a": 0.88})

    report = compare_runs([baseline_id, new_id], store)
    assert "cell-a" in report.improvements


def test_metric_deltas_from_real_worker_results(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    baseline_id = _run_experiment(store, runner, "v1", {"train": 0.80})
    new_id = _run_experiment(store, runner, "v2", {"train": 0.90})

    report = compare_runs([baseline_id, new_id], store)
    tc = next(c for c in report.task_comparisons if c.task_key == "train")
    delta = tc.metric_deltas.get("accuracy", {}).get(new_id)

    assert delta is not None
    assert abs(delta - 0.10) < 1e-9


def test_wall_seconds_present_after_real_execution(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    baseline_id = _run_experiment(store, runner, "v1", {"cell": 0.80})
    new_id = _run_experiment(store, runner, "v2", {"cell": 0.88})

    report = compare_runs([baseline_id, new_id], store)

    # wall_seconds is injected by LocalWorker automatically
    assert report.snapshots[baseline_id].total_wall_seconds is not None
    assert report.snapshots[new_id].total_wall_seconds is not None


def test_format_report_renders_real_data(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    baseline_id = _run_experiment(store, runner, "baseline", {"a": 0.80, "b": 0.75})
    new_id = _run_experiment(store, runner, "experiment", {"a": 0.85}, fail_keys={"b"})

    report = compare_runs([baseline_id, new_id], store)
    text = format_report(report)

    assert "baseline" in text
    assert "experiment" in text
    assert "Regressions" in text
    assert "b" in text


def test_report_to_dict_is_json_serializable(tmp_path: Path) -> None:
    import json

    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    baseline_id = _run_experiment(store, runner, "baseline", {"cell": 0.80})
    new_id = _run_experiment(store, runner, "experiment", {"cell": 0.90})

    report = compare_runs([baseline_id, new_id], store)
    d = report_to_dict(report)

    serialized = json.dumps(d)  # must not raise
    restored = json.loads(serialized)

    assert restored["baseline"] == baseline_id
    assert len(restored["task_comparisons"]) == 1


def test_three_run_comparison_integration(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    runner = FineTuneRunner(store)

    r1 = _run_experiment(store, runner, "v1", {"a": 0.70, "b": 0.80})
    r2 = _run_experiment(store, runner, "v2", {"a": 0.75, "b": 0.80})
    r3 = _run_experiment(store, runner, "v3", {"a": 0.82, "b": 0.83})

    report = compare_runs([r1, r2, r3], store)

    assert len(report.run_ids) == 3
    assert report.regressions == []
    assert report.improvements == []

    # Metric delta for "a" vs baseline (v1)
    tc_a = next(c for c in report.task_comparisons if c.task_key == "a")
    assert tc_a.metric_deltas["accuracy"][r3] == pytest.approx(0.12)
