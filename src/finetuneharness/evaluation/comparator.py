from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from finetuneharness.state.models import TaskStatus
from finetuneharness.state.store import StateStore

_FAIL_STATUSES = frozenset({
    TaskStatus.FAILED.value,
    TaskStatus.TIMED_OUT.value,
    TaskStatus.CANCELLED.value,
    "missing",
})
_OK_STATUSES = frozenset({TaskStatus.SUCCEEDED.value})


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    name: str
    status: str
    total_tasks: int
    succeeded: int
    failed: int
    success_rate: float  # NaN when total_tasks == 0
    total_wall_seconds: float | None  # None when no tasks reported timing


@dataclass
class TaskComparison:
    task_key: str
    status_by_run: dict[str, str]
    result_by_run: dict[str, dict[str, Any] | None]
    # metric_key -> {run_id: delta from baseline}
    metric_deltas: dict[str, dict[str, float]]
    wall_seconds_by_run: dict[str, float | None]
    is_regression: bool
    is_improvement: bool


@dataclass
class ComparisonReport:
    run_ids: list[str]  # first entry is the baseline
    snapshots: dict[str, RunSnapshot]
    task_comparisons: list[TaskComparison]
    regressions: list[str]
    improvements: list[str]


def _numeric_metrics(result: dict[str, Any] | None) -> dict[str, float]:
    """Extract finite numeric metrics from a result dict.

    NaN and Inf are excluded: they would corrupt deltas and comparisons silently.
    """
    if result is None:
        return {}
    out: dict[str, float] = {}
    for k, v in result.items():
        if isinstance(v, (int, float)):
            fv = float(v)
            if math.isfinite(fv):
                out[k] = fv
    return out


def compare_runs(run_ids: list[str], store: StateStore) -> ComparisonReport:
    if len(run_ids) < 2:
        raise ValueError("compare_runs requires at least two run_ids")

    snapshots: dict[str, RunSnapshot] = {}
    tasks_by_run: dict[str, dict[str, Any]] = {}

    for run_id in run_ids:
        run = store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run_id: {run_id!r}")
        tasks = store.list_tasks(run_id)
        tasks_by_run[run_id] = {t.task_key: t for t in tasks}

        succeeded = sum(1 for t in tasks if t.status is TaskStatus.SUCCEEDED)
        failed = sum(
            1 for t in tasks
            if t.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED)
        )
        total = len(tasks)
        timed = [
            float(t.result["wall_seconds"])
            for t in tasks
            if t.result is not None and isinstance(t.result.get("wall_seconds"), (int, float))
        ]
        snapshots[run_id] = RunSnapshot(
            run_id=run_id,
            name=run.name,
            status=run.status.value,
            total_tasks=total,
            succeeded=succeeded,
            failed=failed,
            success_rate=succeeded / total if total > 0 else float("nan"),
            total_wall_seconds=round(sum(timed), 3) if timed else None,
        )

    baseline_id = run_ids[0]
    all_keys: set[str] = set()
    for task_map in tasks_by_run.values():
        all_keys.update(task_map.keys())

    task_comparisons: list[TaskComparison] = []
    regressions: list[str] = []
    improvements: list[str] = []

    for task_key in sorted(all_keys):
        status_by_run = {
            run_id: (
                tasks_by_run[run_id][task_key].status.value
                if task_key in tasks_by_run[run_id]
                else "missing"
            )
            for run_id in run_ids
        }
        result_by_run: dict[str, dict[str, Any] | None] = {
            run_id: (
                tasks_by_run[run_id][task_key].result
                if task_key in tasks_by_run[run_id]
                else None
            )
            for run_id in run_ids
        }

        baseline_metrics = _numeric_metrics(result_by_run.get(baseline_id))
        metric_deltas: dict[str, dict[str, float]] = {}
        for run_id in run_ids[1:]:
            run_metrics = _numeric_metrics(result_by_run.get(run_id))
            for metric_key in set(baseline_metrics) & set(run_metrics):
                if metric_key == "wall_seconds":
                    continue  # timing is surfaced separately
                metric_deltas.setdefault(metric_key, {})[run_id] = (
                    run_metrics[metric_key] - baseline_metrics[metric_key]
                )

        wall_seconds_by_run: dict[str, float | None] = {}
        for run_id in run_ids:
            res = result_by_run.get(run_id)
            raw = res.get("wall_seconds") if res is not None else None
            wall_seconds_by_run[run_id] = float(raw) if isinstance(raw, (int, float)) else None

        baseline_status = status_by_run[baseline_id]
        later_statuses = [status_by_run[r] for r in run_ids[1:]]

        is_regression = baseline_status in _OK_STATUSES and any(
            s in _FAIL_STATUSES for s in later_statuses
        )
        is_improvement = baseline_status in _FAIL_STATUSES and any(
            s in _OK_STATUSES for s in later_statuses
        )

        tc = TaskComparison(
            task_key=task_key,
            status_by_run=status_by_run,
            result_by_run=result_by_run,
            metric_deltas=metric_deltas,
            wall_seconds_by_run=wall_seconds_by_run,
            is_regression=is_regression,
            is_improvement=is_improvement,
        )
        task_comparisons.append(tc)
        if is_regression:
            regressions.append(task_key)
        if is_improvement:
            improvements.append(task_key)

    return ComparisonReport(
        run_ids=run_ids,
        snapshots=snapshots,
        task_comparisons=task_comparisons,
        regressions=regressions,
        improvements=improvements,
    )
