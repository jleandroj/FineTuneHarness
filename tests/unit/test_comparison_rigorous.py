"""Rigorous comparison tests: the 10 cases the user named explicitly.

These tests answer: "¿Puedo comparar el run de hoy con el de la semana pasada
de forma rigurosa?" — and verify each condition that makes the answer YES.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finetuneharness.evaluation.comparator import (
    DEFAULT_THRESHOLDS,
    ComparabilityError,
    check_comparability,
    compare_runs,
    filter_runs_since,
    find_latest_run_pair,
)
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import RunRecord, RunStatus, TaskStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_run(
    runner: FineTuneRunner,
    name: str,
    task_keys: list[str],
    config: dict | None = None,
    env_snapshot: dict | None = None,
) -> str:
    base = {"project": {"name": "test"}, "executor": {"kind": "local"}, "artifacts": {"root": "."}}
    if config:
        base.update(config)
    return runner.create_run(
        name=name,
        config=base,
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


# ── 1. RunRecord has created_at ───────────────────────────────────────────────

def test_run_record_has_created_at():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner, "r1", ["a"])
    run = store.get_run(run_id)
    assert run is not None
    assert run.created_at is not None
    assert isinstance(run.created_at, datetime)
    # created_at should be recent (within 5 seconds)
    delta = datetime.now(timezone.utc) - run.created_at
    assert delta.total_seconds() < 5


def test_run_record_finished_at_set_on_completion():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner, "r1", ["a"])
    run = store.get_run(run_id)
    assert run.finished_at is None  # not set yet

    _succeed(store, run_id, "a")
    runner.refresh_run_status(run_id)

    run = store.get_run(run_id)
    assert run.finished_at is not None
    assert isinstance(run.finished_at, datetime)


# ── 2. filter_runs_since ──────────────────────────────────────────────────────

def test_list_runs_since_date():
    now = datetime.now(timezone.utc)
    old_run = RunRecord(
        run_id="old",
        name="old-run",
        status=RunStatus.COMPLETED,
        config={},
        created_at=now - timedelta(days=10),
    )
    recent_run = RunRecord(
        run_id="new",
        name="new-run",
        status=RunStatus.COMPLETED,
        config={},
        created_at=now - timedelta(days=2),
    )
    all_runs = [old_run, recent_run]
    cutoff = now - timedelta(days=7)

    filtered = filter_runs_since(all_runs, cutoff)

    assert len(filtered) == 1
    assert filtered[0].run_id == "new"


def test_list_runs_since_returns_all_when_no_filter():
    now = datetime.now(timezone.utc)
    runs = [
        RunRecord(run_id=f"r{i}", name=f"r{i}", status=RunStatus.COMPLETED,
                  config={}, created_at=now - timedelta(days=i))
        for i in range(5)
    ]
    filtered = filter_runs_since(runs, now - timedelta(days=10))
    assert len(filtered) == 5


# ── 3. compare_runs rejects different dataset_hash ────────────────────────────

def test_compare_runs_rejects_different_dataset_hash():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a"], config={"dataset_hash": "abc123"})
    run2 = _make_run(runner, "compare", ["a"], config={"dataset_hash": "def456"})
    _succeed(store, run1, "a", {"accuracy": 0.8})
    _succeed(store, run2, "a", {"accuracy": 0.85})

    report = compare_runs([run1, run2], store)

    errors = [i for i in report.comparability_issues if i.severity == "error"]
    assert any(i.field == "dataset_hash" for i in errors)
    assert any("dataset_hash" in i.message for i in errors)


def test_compare_runs_strict_raises_on_dataset_hash_mismatch():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a"], config={"dataset_hash": "abc123"})
    run2 = _make_run(runner, "compare", ["a"], config={"dataset_hash": "def456"})

    with pytest.raises(ComparabilityError, match="dataset_hash"):
        compare_runs([run1, run2], store, strict=True)


# ── 4. compare_runs warns on different config ─────────────────────────────────

def test_compare_runs_rejects_different_config_hash():
    """Different model_name produces a comparability WARNING."""
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a"], config={"model_name": "bert-base"})
    run2 = _make_run(runner, "compare", ["a"], config={"model_name": "bert-large"})
    _succeed(store, run1, "a", {"accuracy": 0.8})
    _succeed(store, run2, "a", {"accuracy": 0.85})

    report = compare_runs([run1, run2], store)

    warnings = [i for i in report.comparability_issues if i.severity == "warning"]
    assert any("model_name" in i.field for i in warnings)


# ── 5. Metric drop is a regression ───────────────────────────────────────────

def test_metric_drop_is_regression():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["cell"])
    run2 = _make_run(runner, "compare", ["cell"])
    _succeed(store, run1, "cell", {"f1": 0.85, "accuracy": 0.88})
    _succeed(store, run2, "cell", {"f1": 0.75, "accuracy": 0.78})  # -0.10 drop — exceeds threshold

    report = compare_runs([run1, run2], store)

    tc = report.task_comparisons[0]
    assert "f1" in tc.metric_regressions
    assert tc.metric_regressions["f1"][run2] == pytest.approx(-0.10, abs=1e-9)
    assert "cell" in report.metric_regressions


# ── 6. Small metric delta is not a regression ─────────────────────────────────

def test_small_metric_delta_is_not_regression():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["cell"])
    run2 = _make_run(runner, "compare", ["cell"])
    _succeed(store, run1, "cell", {"f1": 0.850, "accuracy": 0.880})
    _succeed(store, run2, "cell", {"f1": 0.845, "accuracy": 0.875})  # -0.005 delta — below threshold

    report = compare_runs([run1, run2], store)

    tc = report.task_comparisons[0]
    assert "f1" not in tc.metric_regressions
    assert "cell" not in report.metric_regressions


# ── 7. SUCCEEDED → FAILED is a status regression ─────────────────────────────

def test_succeeded_to_failed_is_regression():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a", "b"])
    run2 = _make_run(runner, "compare", ["a", "b"])
    _succeed(store, run1, "a")
    _succeed(store, run1, "b")
    _succeed(store, run2, "a")
    _fail(store, run2, "b")

    report = compare_runs([run1, run2], store)

    assert "b" in report.regressions
    tc = next(c for c in report.task_comparisons if c.task_key == "b")
    assert tc.is_regression is True


# ── 8. Degenerate results are excluded from comparison ───────────────────────

def test_degenerate_results_are_excluded_from_comparison():
    """A task with adapter_loaded=False (DEGENERATE_RESULT) must not contribute
    to metric deltas, but should still appear in the report."""
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["cell"])
    run2 = _make_run(runner, "compare", ["cell"])

    _succeed(store, run1, "cell", {"method": "adalora", "accuracy": 0.85, "f1": 0.83,
                                    "adapter_loaded": True, "trainable_params": 1_200_000})
    _succeed(store, run2, "cell", {"method": "adalora", "accuracy": 0.40, "f1": 0.30,
                                    "adapter_loaded": False, "trainable_params": 1_200_000})

    report = compare_runs([run1, run2], store)

    tc = report.task_comparisons[0]
    # run2's result is degenerate — it must be excluded from metric deltas
    assert run2 in tc.excluded_from_metrics
    assert "f1" not in tc.metric_deltas
    # It should show up in excluded_tasks report
    assert "cell" in report.excluded_tasks.get(run2, [])


# ── 9. find_latest_run_pair ───────────────────────────────────────────────────

def test_compare_latest_previous():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "first", ["a"])
    run2 = _make_run(runner, "second", ["a"])
    run3 = _make_run(runner, "third", ["a"])

    prev_id, latest_id = find_latest_run_pair(store)

    # The two most recent should be run2 (previous) and run3 (latest)
    assert latest_id == run3
    assert prev_id == run2


def test_find_latest_run_pair_requires_two_runs():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    _make_run(runner, "only", ["a"])
    with pytest.raises(ValueError, match="at least 2"):
        find_latest_run_pair(store)


# ── 10. Missing tasks are reported ───────────────────────────────────────────

def test_compare_runs_reports_missing_tasks():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["a", "b", "c"])
    run2 = _make_run(runner, "compare", ["a", "b"])  # "c" dropped
    for key in ("a", "b", "c"):
        _succeed(store, run1, key, {"accuracy": 0.8})
    for key in ("a", "b"):
        _succeed(store, run2, key, {"accuracy": 0.8})

    report = compare_runs([run1, run2], store)

    tc_c = next(c for c in report.task_comparisons if c.task_key == "c")
    assert tc_c.status_by_run[run2] == "missing"
    assert "c" in report.regressions


# ── Thresholds override ───────────────────────────────────────────────────────

def test_custom_thresholds_override_defaults():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["cell"])
    run2 = _make_run(runner, "compare", ["cell"])
    _succeed(store, run1, "cell", {"f1": 0.850})
    _succeed(store, run2, "cell", {"f1": 0.845})  # -0.005 drop

    # With default threshold (0.02), this is NOT a regression
    report_default = compare_runs([run1, run2], store)
    assert "cell" not in report_default.metric_regressions

    # With strict threshold (0.001), this IS a regression
    report_strict = compare_runs([run1, run2], store, thresholds={"f1": 0.001})
    assert "cell" in report_strict.metric_regressions


def test_empty_thresholds_disables_metric_regression():
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run1 = _make_run(runner, "baseline", ["cell"])
    run2 = _make_run(runner, "compare", ["cell"])
    _succeed(store, run1, "cell", {"f1": 0.90})
    _succeed(store, run2, "cell", {"f1": 0.50})  # catastrophic drop

    report = compare_runs([run1, run2], store, thresholds={})
    assert "cell" not in report.metric_regressions  # thresholds={} disables detection
