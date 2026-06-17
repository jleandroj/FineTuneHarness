from __future__ import annotations

import math
from typing import Any

from finetuneharness.evaluation.comparator import ComparisonReport


def format_report(report: ComparisonReport) -> str:
    lines: list[str] = []
    baseline_id = report.run_ids[0]
    baseline_snap = report.snapshots[baseline_id]

    lines.append("=== Run Comparison Report ===")
    lines.append(f"Baseline : {baseline_snap.name} ({baseline_id[:8]})")
    for run_id in report.run_ids[1:]:
        snap = report.snapshots[run_id]
        lines.append(f"Compare  : {snap.name} ({run_id[:8]})")
    lines.append("")

    lines.append("Run Summaries")
    lines.append("-" * 62)
    for run_id in report.run_ids:
        snap = report.snapshots[run_id]
        rate = f"{snap.success_rate:.1%}" if not math.isnan(snap.success_rate) else "n/a"
        timing = f"  {snap.total_wall_seconds:.1f}s" if snap.total_wall_seconds is not None else ""
        tag = "  [baseline]" if run_id == baseline_id else ""
        lines.append(
            f"  {snap.name[:28]:<28}  {snap.status:<15}  "
            f"{snap.succeeded}/{snap.total_tasks} tasks  {rate}{timing}{tag}"
        )
    lines.append("")

    if report.regressions:
        lines.append(f"Regressions ({len(report.regressions)})")
        lines.append("-" * 62)
        for key in report.regressions:
            tc = next(c for c in report.task_comparisons if c.task_key == key)
            statuses = "  ".join(
                f"{r[:8]}:{tc.status_by_run[r]}" for r in report.run_ids
            )
            lines.append(f"  {key}: {statuses}")
        lines.append("")

    if report.improvements:
        lines.append(f"Improvements ({len(report.improvements)})")
        lines.append("-" * 62)
        for key in report.improvements:
            tc = next(c for c in report.task_comparisons if c.task_key == key)
            statuses = "  ".join(
                f"{r[:8]}:{tc.status_by_run[r]}" for r in report.run_ids
            )
            lines.append(f"  {key}: {statuses}")
        lines.append("")

    has_metrics = any(tc.metric_deltas for tc in report.task_comparisons)
    if has_metrics:
        lines.append("Metric Deltas (vs baseline)")
        lines.append("-" * 62)
        for tc in report.task_comparisons:
            if not tc.metric_deltas:
                continue
            lines.append(f"  {tc.task_key}:")
            for metric_key, deltas in sorted(tc.metric_deltas.items()):
                for run_id, delta in sorted(deltas.items()):
                    sign = "+" if delta >= 0 else ""
                    lines.append(
                        f"    {metric_key}: {sign}{delta:.4f}  ({run_id[:8]})"
                    )
        lines.append("")

    has_timing = any(
        any(v is not None for v in tc.wall_seconds_by_run.values())
        for tc in report.task_comparisons
    )
    if has_timing:
        lines.append("Task Timing (wall seconds)")
        lines.append("-" * 62)
        for tc in report.task_comparisons:
            vals = "  ".join(
                f"{r[:8]}:{tc.wall_seconds_by_run[r]:.2f}s"
                if tc.wall_seconds_by_run.get(r) is not None
                else f"{r[:8]}:n/a"
                for r in report.run_ids
            )
            lines.append(f"  {tc.task_key}: {vals}")
        lines.append("")

    if not report.regressions and not report.improvements:
        lines.append("No regressions or improvements detected.")

    return "\n".join(lines)


def report_to_dict(report: ComparisonReport) -> dict[str, Any]:
    return {
        "run_ids": report.run_ids,
        "baseline": report.run_ids[0],
        "snapshots": {
            run_id: {
                "run_id": s.run_id,
                "name": s.name,
                "status": s.status,
                "total_tasks": s.total_tasks,
                "succeeded": s.succeeded,
                "failed": s.failed,
                "success_rate": None if math.isnan(s.success_rate) else s.success_rate,
            }
            for run_id, s in report.snapshots.items()
        },
        "regressions": report.regressions,
        "improvements": report.improvements,
        "task_comparisons": [
            {
                "task_key": tc.task_key,
                "status_by_run": tc.status_by_run,
                "result_by_run": tc.result_by_run,
                "metric_deltas": tc.metric_deltas,
                "is_regression": tc.is_regression,
                "is_improvement": tc.is_improvement,
            }
            for tc in report.task_comparisons
        ],
    }
