"""verify_run: independent re-checking of a run's self-reported results.

Catches lies the generic result validator cannot: fabricated or internally
inconsistent numbers, success states with no execution behind them, impossible
timings. It re-derives invariants from the result itself and cross-checks the
event log — it never trusts a single reported field in isolation.

Checks (codes map to docs/LIE_RESISTANCE_AUDIT.md probes):
  RAN_EVIDENCE (P4)      — a SUCCEEDED task must have a 'task_running' event.
  SUCCEEDED_EVENT (P4)   — a SUCCEEDED task must have a 'task_succeeded' event.
  VALIDATION_COHERENT(P2)— stored _validation_status must be a SUCCEEDED_* status.
  METRIC_TWIN (P3)       — accuracy == test_accuracy (and f1 == test_f1) if both present.
  IMPROVEMENT (P3)       — improvement == test_accuracy - baseline_accuracy.
  PARAMS (P3)            — 0 < trainable_params <= total_params.
  TIMING (P9)            — train_seconds > 0 and not greater than wall_seconds.
  PROVENANCE (P1)        — minimal evidence present (timing + params). WARN if thin.

This module is intentionally NOT imported by the executor: the verifier must be a
different code path from the producer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from finetuneharness.state.models import EventRecord, TaskRecord, TaskStatus

_SUCCEEDED_VALIDATION = {"SUCCEEDED_VALIDATED", "SUCCEEDED_WITH_WARNINGS"}
_ABS_TOL = 1e-6
_IMPROVEMENT_TOL = 1e-3


class Verdict(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Finding:
    severity: str          # "fail" | "warn"
    code: str
    task_key: str | None
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    verdict: Verdict
    checked: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def fails(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "fail"]

    @property
    def warns(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]


def _num(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def verify_run(tasks: list[TaskRecord], events: list[EventRecord]) -> VerificationReport:
    """Independently verify the SUCCEEDED tasks of a run. Pure; reads no I/O."""
    running_ids = {e.task_id for e in events if e.kind == "task_running"}
    succeeded_evt_ids = {e.task_id for e in events if e.kind == "task_succeeded"}

    findings: list[Finding] = []
    checked = 0

    def fail(tk: str | None, code: str, detail: str) -> None:
        findings.append(Finding("fail", code, tk, detail))

    def warn(tk: str | None, code: str, detail: str) -> None:
        findings.append(Finding("warn", code, tk, detail))

    for t in tasks:
        if t.status is not TaskStatus.SUCCEEDED:
            continue
        checked += 1
        tk = t.task_key
        r = t.result or {}

        # ── Execution evidence (a green with no run behind it) ────────────────
        if t.task_id not in running_ids:
            fail(tk, "RAN_EVIDENCE", "SUCCEEDED but no 'task_running' event was recorded")
        if t.task_id not in succeeded_evt_ids:
            fail(tk, "SUCCEEDED_EVENT", "SUCCEEDED state with no 'task_succeeded' event")

        # ── Validation coherence ──────────────────────────────────────────────
        vs = r.get("_validation_status")
        if vs is not None and vs not in _SUCCEEDED_VALIDATION:
            fail(tk, "VALIDATION_COHERENT", f"_validation_status={vs!r} is not a SUCCEEDED_* status")

        # ── Internal consistency (fabrication catchers) ───────────────────────
        acc, tacc = _num(r.get("accuracy")), _num(r.get("test_accuracy"))
        if acc is not None and tacc is not None and abs(acc - tacc) > _ABS_TOL:
            fail(tk, "METRIC_TWIN", f"accuracy={acc} != test_accuracy={tacc}")
        f1, tf1 = _num(r.get("f1")), _num(r.get("test_f1"))
        if f1 is not None and tf1 is not None and abs(f1 - tf1) > _ABS_TOL:
            fail(tk, "METRIC_TWIN", f"f1={f1} != test_f1={tf1}")

        imp, base = _num(r.get("improvement")), _num(r.get("baseline_accuracy"))
        if imp is not None and tacc is not None and base is not None:
            if abs(imp - (tacc - base)) > _IMPROVEMENT_TOL:
                fail(tk, "IMPROVEMENT",
                     f"improvement={imp} != test_accuracy-baseline={tacc - base:.4f}")

        tp, tot = _num(r.get("trainable_params")), _num(r.get("total_params"))
        if tp is not None and tot is not None:
            if tp <= 0 or tot <= 0:
                fail(tk, "PARAMS", f"non-positive params (trainable={tp}, total={tot})")
            elif tp > tot:
                fail(tk, "PARAMS", f"trainable_params={tp} > total_params={tot}")

        # ── Timing plausibility ───────────────────────────────────────────────
        train_s, wall_s = _num(r.get("train_seconds")), _num(r.get("wall_seconds"))
        if train_s is not None:
            if train_s <= 0:
                fail(tk, "TIMING", f"train_seconds={train_s} (a real training run takes time)")
            elif wall_s is not None and train_s > wall_s + _ABS_TOL:
                fail(tk, "TIMING", f"train_seconds={train_s} > wall_seconds={wall_s} (impossible)")

        # ── Minimal provenance ────────────────────────────────────────────────
        if train_s is None and wall_s is None:
            warn(tk, "PROVENANCE", "no timing evidence (train_seconds/wall_seconds) in result")
        if tp is None and tot is None:
            warn(tk, "PROVENANCE", "no parameter-count evidence in result")

    verdict = (
        Verdict.FAIL if any(f.severity == "fail" for f in findings)
        else Verdict.WARN if findings
        else Verdict.PASS
    )
    return VerificationReport(verdict=verdict, checked=checked, findings=findings)
