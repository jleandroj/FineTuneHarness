from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from finetuneharness.state.models import RunRecord, TaskStatus
from finetuneharness.state.store import StateStore

# ── Regression severity types ─────────────────────────────────────────────────

_FAIL_STATUSES = frozenset({
    TaskStatus.FAILED.value,
    TaskStatus.TIMED_OUT.value,
    TaskStatus.CANCELLED.value,
    "missing",
})
_OK_STATUSES = frozenset({TaskStatus.SUCCEEDED.value})

# Metrics where a positive delta (new > baseline) is a regression
_LOWER_IS_BETTER = frozenset({"loss", "val_loss", "loss_start", "loss_end"})

# Config keys that, when present in both runs, must match for comparison to be rigorous
_COMPARABLE_CONFIG_KEYS = frozenset({
    "model_name", "tokenizer_type", "max_length", "num_labels",
    "model_type", "vocab_size",
})

MetricThresholds = dict[str, float]

DEFAULT_THRESHOLDS: MetricThresholds = {
    "accuracy": 0.02,
    "f1": 0.02,
    "f1_macro": 0.02,
    "f1_micro": 0.02,
    "f1_weighted": 0.02,
    "precision": 0.03,
    "recall": 0.03,
    "auc": 0.02,
    "roc_auc": 0.02,
    "balanced_accuracy": 0.02,
    "loss": 0.05,
    "val_loss": 0.05,
}


# ── Comparability ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComparabilityIssue:
    severity: Literal["error", "warning"]
    field: str
    baseline_value: Any
    compare_value: Any
    message: str


class ComparabilityError(ValueError):
    """Raised by compare_runs when strict=True and error-severity issues exist."""

    def __init__(self, issues: list[ComparabilityIssue]) -> None:
        self.issues = issues
        msgs = "; ".join(i.message for i in issues if i.severity == "error")
        super().__init__(f"Runs are not comparable: {msgs}")


def check_comparability(run_a: RunRecord, run_b: RunRecord) -> list[ComparabilityIssue]:
    """Check whether two runs are comparable.

    Returns a list of ComparabilityIssue objects.
    severity='error'   → comparison will produce meaningless results (e.g. different dataset)
    severity='warning' → comparison is possible but results should be interpreted carefully
    """
    issues: list[ComparabilityIssue] = []

    # dataset_hashes — ERROR if both populated and different.
    # Use the first-class RunRecord.dataset_hashes field (always populated from
    # either dataset_hash or datasets at create_run time) rather than
    # config.get("dataset_hash"), which is absent for runs that used the
    # datasets-dict form and would silently skip the check.
    hashes_a = run_a.dataset_hashes
    hashes_b = run_b.dataset_hashes
    if hashes_a and hashes_b and hashes_a != hashes_b:
        issues.append(ComparabilityIssue(
            severity="error",
            field="dataset_hashes",
            baseline_value=hashes_a,
            compare_value=hashes_b,
            message=f"dataset_hashes differ — results measure different data: {hashes_a} vs {hashes_b}",
        ))

    # seed — WARNING if both present and different
    seed_a = run_a.config.get("seed")
    seed_b = run_b.config.get("seed")
    if seed_a is not None and seed_b is not None and seed_a != seed_b:
        issues.append(ComparabilityIssue(
            severity="warning",
            field="seed",
            baseline_value=seed_a,
            compare_value=seed_b,
            message=f"seed differs: {seed_a} vs {seed_b} — variance from different seeds may exceed metric deltas",
        ))

    # git_commit — WARNING if both present and different
    commit_a = run_a.env_snapshot.get("git_commit")
    commit_b = run_b.env_snapshot.get("git_commit")
    if commit_a is not None and commit_b is not None and commit_a != commit_b:
        short_a = commit_a[:8] if isinstance(commit_a, str) else str(commit_a)
        short_b = commit_b[:8] if isinstance(commit_b, str) else str(commit_b)
        issues.append(ComparabilityIssue(
            severity="warning",
            field="git_commit",
            baseline_value=commit_a,
            compare_value=commit_b,
            message=f"git_commit differs: {short_a} vs {short_b} — code may have changed between runs",
        ))

    # Semantically significant config keys
    for key in sorted(_COMPARABLE_CONFIG_KEYS):
        val_a = run_a.config.get(key)
        val_b = run_b.config.get(key)
        if val_a is not None and val_b is not None and val_a != val_b:
            issues.append(ComparabilityIssue(
                severity="warning",
                field=f"config.{key}",
                baseline_value=val_a,
                compare_value=val_b,
                message=f"config.{key} differs: {val_a!r} vs {val_b!r}",
            ))

    return issues


# ── Report structures ─────────────────────────────────────────────────────────

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
    created_at: datetime | None = None
    finished_at: datetime | None = None


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
    # metric_key -> {run_id: delta} where the delta exceeds the threshold
    metric_regressions: dict[str, dict[str, float]] = field(default_factory=dict)
    # run_id -> reason string when result was excluded from metric comparison
    excluded_from_metrics: dict[str, str] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    run_ids: list[str]  # first entry is the baseline
    snapshots: dict[str, RunSnapshot]
    task_comparisons: list[TaskComparison]
    regressions: list[str]      # task keys with status-based regressions
    improvements: list[str]     # task keys with status-based improvements
    metric_regressions: list[str] = field(default_factory=list)  # task keys with metric regressions
    comparability_issues: list[ComparabilityIssue] = field(default_factory=list)
    excluded_tasks: dict[str, list[str]] = field(default_factory=dict)  # run_id → task_keys excluded


# ── Internal helpers ──────────────────────────────────────────────────────────

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


def _is_metric_regression(
    metric_key: str,
    delta: float,
    thresholds: MetricThresholds,
) -> bool:
    """Return True if the metric delta constitutes a regression given thresholds."""
    threshold = thresholds.get(metric_key)
    if threshold is None:
        return False
    if metric_key in _LOWER_IS_BETTER:
        return delta > threshold    # loss went up = bad
    else:
        return delta < -threshold   # accuracy went down = bad


def _result_is_valid_for_comparison(result: dict[str, Any] | None) -> tuple[bool, str]:
    """Return (is_valid, reason). Uses validate_result when available."""
    if result is None:
        return False, "result is None"
    try:
        from finetuneharness.evaluation.validator import ResultStatus, validate_result
        outcome = validate_result(result)
        if outcome.status in (ResultStatus.DEGENERATE_RESULT, ResultStatus.FAILED_VALIDATION):
            reasons = "; ".join(outcome.validation_errors[:3])
            return False, f"{outcome.status}: {reasons}"
    except ImportError:
        pass
    return True, ""


# ── Public API ────────────────────────────────────────────────────────────────

def filter_runs_since(runs: list[RunRecord], cutoff: datetime) -> list[RunRecord]:
    """Return runs whose created_at is >= cutoff (UTC-aware comparison)."""
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    result = []
    for r in runs:
        ts = r.created_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            result.append(r)
    return result


def parse_since_duration(s: str) -> datetime:
    """Parse '7d', '30d', '2h', '1w' into a UTC cutoff datetime."""
    import re
    m = re.match(r"^(\d+)([dhw])$", s.strip().lower())
    if not m:
        raise ValueError(
            f"Invalid since format {s!r}. Use e.g. '7d', '30d', '2h', '1w'."
        )
    n, unit = int(m.group(1)), m.group(2)
    delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "w": timedelta(weeks=n)}[unit]
    return datetime.now(timezone.utc) - delta


def find_latest_run_pair(store: StateStore) -> tuple[str, str]:
    """Return (previous_run_id, latest_run_id) ordered by created_at.

    Raises ValueError if fewer than 2 runs exist.
    """
    runs = store.list_runs()
    if len(runs) < 2:
        raise ValueError(
            f"find_latest_run_pair requires at least 2 runs, found {len(runs)}"
        )
    sorted_runs = sorted(runs, key=lambda r: r.created_at)
    return sorted_runs[-2].run_id, sorted_runs[-1].run_id


def compare_runs(
    run_ids: list[str],
    store: StateStore,
    *,
    thresholds: MetricThresholds | None = None,
    strict: bool = False,
) -> ComparisonReport:
    """Compare multiple runs, first being the baseline.

    Args:
        run_ids:    At least two run IDs. First is the baseline.
        store:      StateStore to load runs and tasks from.
        thresholds: Per-metric regression thresholds. Defaults to DEFAULT_THRESHOLDS.
                    Pass {} to disable metric regression detection.
        strict:     If True, raises ComparabilityError when error-severity issues exist.
    """
    if len(run_ids) < 2:
        raise ValueError("compare_runs requires at least two run_ids")

    effective_thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds

    # ── Load all runs and collect comparability issues ────────────────────────
    runs: dict[str, RunRecord] = {}
    for run_id in run_ids:
        run = store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run_id: {run_id!r}")
        runs[run_id] = run

    baseline_run = runs[run_ids[0]]
    all_comparability_issues: list[ComparabilityIssue] = []
    for run_id in run_ids[1:]:
        issues = check_comparability(baseline_run, runs[run_id])
        all_comparability_issues.extend(issues)

    if strict and any(i.severity == "error" for i in all_comparability_issues):
        raise ComparabilityError([i for i in all_comparability_issues if i.severity == "error"])

    # ── Load tasks ────────────────────────────────────────────────────────────
    tasks_by_run: dict[str, dict[str, Any]] = {}
    excluded_tasks: dict[str, list[str]] = {run_id: [] for run_id in run_ids}
    snapshots: dict[str, RunSnapshot] = {}

    for run_id in run_ids:
        run = runs[run_id]
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
            created_at=run.created_at,
            finished_at=run.finished_at,
        )

    # ── Per-task comparison ───────────────────────────────────────────────────
    baseline_id = run_ids[0]
    all_keys: set[str] = set()
    for task_map in tasks_by_run.values():
        all_keys.update(task_map.keys())

    task_comparisons: list[TaskComparison] = []
    status_regressions: list[str] = []
    status_improvements: list[str] = []
    metric_regression_keys: list[str] = []

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

        # ── Exclusion check ───────────────────────────────────────────────────
        task_excluded: dict[str, str] = {}
        for run_id in run_ids:
            res = result_by_run.get(run_id)
            valid, reason = _result_is_valid_for_comparison(res)
            if not valid and res is not None:
                task_excluded[run_id] = reason
                excluded_tasks[run_id].append(task_key)

        # ── Metric deltas (only for non-excluded results) ─────────────────────
        baseline_metrics = (
            _numeric_metrics(result_by_run.get(baseline_id))
            if baseline_id not in task_excluded
            else {}
        )
        metric_deltas: dict[str, dict[str, float]] = {}
        task_metric_regressions: dict[str, dict[str, float]] = {}

        for run_id in run_ids[1:]:
            if run_id in task_excluded or baseline_id in task_excluded:
                continue
            run_metrics = _numeric_metrics(result_by_run.get(run_id))
            for metric_key in set(baseline_metrics) & set(run_metrics):
                if metric_key == "wall_seconds":
                    continue
                delta = run_metrics[metric_key] - baseline_metrics[metric_key]
                metric_deltas.setdefault(metric_key, {})[run_id] = delta
                if _is_metric_regression(metric_key, delta, effective_thresholds):
                    task_metric_regressions.setdefault(metric_key, {})[run_id] = delta

        # ── Wall seconds ──────────────────────────────────────────────────────
        wall_seconds_by_run: dict[str, float | None] = {}
        for run_id in run_ids:
            res = result_by_run.get(run_id)
            raw = res.get("wall_seconds") if res is not None else None
            wall_seconds_by_run[run_id] = float(raw) if isinstance(raw, (int, float)) else None

        # ── Status-based regression/improvement ───────────────────────────────
        baseline_status = status_by_run[baseline_id]
        later_statuses = [status_by_run[r] for r in run_ids[1:]]

        is_regression = baseline_status in _OK_STATUSES and any(
            s in _FAIL_STATUSES for s in later_statuses
        )
        is_improvement = baseline_status in _FAIL_STATUSES and any(
            s in _OK_STATUSES for s in later_statuses
        )

        has_metric_regression = bool(task_metric_regressions)

        tc = TaskComparison(
            task_key=task_key,
            status_by_run=status_by_run,
            result_by_run=result_by_run,
            metric_deltas=metric_deltas,
            wall_seconds_by_run=wall_seconds_by_run,
            is_regression=is_regression or has_metric_regression,
            is_improvement=is_improvement,
            metric_regressions=task_metric_regressions,
            excluded_from_metrics=task_excluded,
        )
        task_comparisons.append(tc)
        if is_regression:
            status_regressions.append(task_key)
        if is_improvement:
            status_improvements.append(task_key)
        if has_metric_regression and task_key not in status_regressions:
            metric_regression_keys.append(task_key)

    return ComparisonReport(
        run_ids=run_ids,
        snapshots=snapshots,
        task_comparisons=task_comparisons,
        regressions=status_regressions,
        improvements=status_improvements,
        metric_regressions=metric_regression_keys,
        comparability_issues=all_comparability_issues,
        excluded_tasks={k: v for k, v in excluded_tasks.items() if v},
    )
