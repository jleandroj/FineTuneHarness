from __future__ import annotations

import pytest

from finetuneharness.evaluation.comparator import compare_runs
from finetuneharness.evaluation.report import format_report, report_to_dict
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import TaskStatus

_BASE_CONFIG = {
    "project": {"name": "eval-test"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
}


def _make_run(runner: FineTuneRunner, name: str, task_keys: list[str]) -> str:
    return runner.create_run(
        name=name,
        config=_BASE_CONFIG,
        tasks=[{"task_key": k} for k in task_keys],
    )


def _succeed(store: InMemoryStateStore, run_id: str, task_key: str, result: dict | None = None) -> None:
    task = next(t for t in store.list_tasks(run_id) if t.task_key == task_key)
    store.update_task_status(task.task_id, TaskStatus.LEASED)
    store.update_task_status(task.task_id, TaskStatus.RUNNING)
    store.update_task_status(task.task_id, TaskStatus.SUCCEEDED, result=result or {})


def _fail(store: InMemoryStateStore, run_id: str, task_key: str) -> None:
    task = next(t for t in store.list_tasks(run_id) if t.task_key == task_key)
    store.update_task_status(task.task_id, TaskStatus.LEASED)
    store.update_task_status(task.task_id, TaskStatus.RUNNING)
    store.update_task_status(task.task_id, TaskStatus.FAILED, error="boom")


# --- basic comparison ---

def test_compare_requires_two_runs() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "r1", ["a"])
    with pytest.raises(ValueError, match="at least two"):
        compare_runs([run1], store)


def test_compare_unknown_run_raises() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "r1", ["a"])
    with pytest.raises(KeyError):
        compare_runs([run1, "does-not-exist"], store)


def test_no_regressions_no_improvements_when_identical() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a", "b"])
    run2 = _make_run(runner, "run2", ["a", "b"])
    for run_id in (run1, run2):
        _succeed(store, run_id, "a")
        _succeed(store, run_id, "b")

    report = compare_runs([run1, run2], store)

    assert report.regressions == []
    assert report.improvements == []
    assert report.snapshots[run1].success_rate == 1.0
    assert report.snapshots[run2].success_rate == 1.0


# --- regressions ---

def test_detects_regression() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a", "b"])
    run2 = _make_run(runner, "run2", ["a", "b"])

    _succeed(store, run1, "a")
    _succeed(store, run1, "b")
    _succeed(store, run2, "a")
    _fail(store, run2, "b")

    report = compare_runs([run1, run2], store)

    assert report.regressions == ["b"]
    assert report.improvements == []


# --- improvements ---

def test_detects_improvement() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a", "b"])
    run2 = _make_run(runner, "run2", ["a", "b"])

    _fail(store, run1, "a")
    _succeed(store, run1, "b")
    _succeed(store, run2, "a")
    _succeed(store, run2, "b")

    report = compare_runs([run1, run2], store)

    assert report.improvements == ["a"]
    assert report.regressions == []


# --- metric deltas ---

def test_metric_deltas_computed() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["cell"])
    run2 = _make_run(runner, "run2", ["cell"])

    _succeed(store, run1, "cell", result={"accuracy": 0.80, "loss": 0.40})
    _succeed(store, run2, "cell", result={"accuracy": 0.85, "loss": 0.35})

    report = compare_runs([run1, run2], store)

    tc = report.task_comparisons[0]
    assert tc.task_key == "cell"
    assert abs(tc.metric_deltas["accuracy"][run2] - 0.05) < 1e-9
    assert abs(tc.metric_deltas["loss"][run2] - (-0.05)) < 1e-9


def test_metric_deltas_only_for_shared_keys() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["cell"])
    run2 = _make_run(runner, "run2", ["cell"])

    _succeed(store, run1, "cell", result={"accuracy": 0.80, "only_in_baseline": 1.0})
    _succeed(store, run2, "cell", result={"accuracy": 0.85, "only_in_run2": 2.0})

    report = compare_runs([run1, run2], store)
    tc = report.task_comparisons[0]

    assert "accuracy" in tc.metric_deltas
    assert "only_in_baseline" not in tc.metric_deltas
    assert "only_in_run2" not in tc.metric_deltas


# --- missing task ---

def test_missing_task_in_new_run_is_regression() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a", "b"])
    run2 = _make_run(runner, "run2", ["a"])  # "b" was dropped

    _succeed(store, run1, "a")
    _succeed(store, run1, "b")
    _succeed(store, run2, "a")

    report = compare_runs([run1, run2], store)

    assert "b" in report.regressions
    tc_b = next(c for c in report.task_comparisons if c.task_key == "b")
    assert tc_b.status_by_run[run2] == "missing"


# --- three-run comparison ---

def test_three_run_comparison() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "v1", ["a", "b"])
    run2 = _make_run(runner, "v2", ["a", "b"])
    run3 = _make_run(runner, "v3", ["a", "b"])

    _succeed(store, run1, "a")
    _succeed(store, run1, "b")
    _fail(store, run2, "a")
    _succeed(store, run2, "b")
    _succeed(store, run3, "a")
    _succeed(store, run3, "b")

    report = compare_runs([run1, run2, run3], store)

    assert "a" in report.regressions  # succeeded in v1, failed in v2
    assert len(report.run_ids) == 3


# --- success_rate edge case ---

def test_success_rate_nan_when_no_tasks() -> None:
    import math
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "empty1", [])
    run2 = _make_run(runner, "empty2", [])

    report = compare_runs([run1, run2], store)

    assert math.isnan(report.snapshots[run1].success_rate)


# --- timing / wall_seconds ---

def test_total_wall_seconds_aggregated() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a", "b"])
    run2 = _make_run(runner, "run2", ["a", "b"])

    _succeed(store, run1, "a", result={"wall_seconds": 1.5})
    _succeed(store, run1, "b", result={"wall_seconds": 2.5})
    _succeed(store, run2, "a", result={"wall_seconds": 1.0})
    _succeed(store, run2, "b", result={"wall_seconds": 1.0})

    report = compare_runs([run1, run2], store)

    assert report.snapshots[run1].total_wall_seconds == pytest.approx(4.0)
    assert report.snapshots[run2].total_wall_seconds == pytest.approx(2.0)


def test_wall_seconds_none_when_not_reported() -> None:
    import math
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a"])
    run2 = _make_run(runner, "run2", ["a"])

    _succeed(store, run1, "a", result={"accuracy": 0.9})  # no wall_seconds
    _succeed(store, run2, "a", result={"accuracy": 0.95})

    report = compare_runs([run1, run2], store)

    assert report.snapshots[run1].total_wall_seconds is None


def test_wall_seconds_per_task_in_comparison() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["cell"])
    run2 = _make_run(runner, "run2", ["cell"])

    _succeed(store, run1, "cell", result={"wall_seconds": 3.0})
    _succeed(store, run2, "cell", result={"wall_seconds": 1.5})

    report = compare_runs([run1, run2], store)
    tc = report.task_comparisons[0]

    assert tc.wall_seconds_by_run[run1] == pytest.approx(3.0)
    assert tc.wall_seconds_by_run[run2] == pytest.approx(1.5)


def test_wall_seconds_not_in_metric_deltas() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["cell"])
    run2 = _make_run(runner, "run2", ["cell"])

    _succeed(store, run1, "cell", result={"accuracy": 0.8, "wall_seconds": 5.0})
    _succeed(store, run2, "cell", result={"accuracy": 0.9, "wall_seconds": 3.0})

    report = compare_runs([run1, run2], store)
    tc = report.task_comparisons[0]

    assert "wall_seconds" not in tc.metric_deltas
    assert "accuracy" in tc.metric_deltas


def test_format_report_shows_timing() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a"])
    run2 = _make_run(runner, "run2", ["a"])
    _succeed(store, run1, "a", result={"wall_seconds": 2.5})
    _succeed(store, run2, "a", result={"wall_seconds": 1.2})

    report = compare_runs([run1, run2], store)
    text = format_report(report)

    assert "2.5s" in text
    assert "Task Timing" in text


# --- report rendering ---

def test_format_report_smoke() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a"])
    run2 = _make_run(runner, "run2", ["a"])
    _succeed(store, run1, "a", result={"accuracy": 0.9})
    _fail(store, run2, "a")

    report = compare_runs([run1, run2], store)
    text = format_report(report)

    assert "Regressions" in text
    assert "baseline" in text
    assert "run2" in text


def test_report_to_dict_smoke() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a"])
    run2 = _make_run(runner, "run2", ["a"])
    _succeed(store, run1, "a")
    _succeed(store, run2, "a")

    report = compare_runs([run1, run2], store)
    d = report_to_dict(report)

    assert d["baseline"] == run1
    assert len(d["run_ids"]) == 2
    assert isinstance(d["task_comparisons"], list)
