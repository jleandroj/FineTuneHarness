"""Result validator: distinguishes bad-but-valid results from degenerate/invalid ones.

SUCCEEDED_VALIDATED  — handler ran correctly, metrics are finite and in range
SUCCEEDED_WITH_WARNINGS — valid but some optional checks flagged (e.g. constant preds)
FAILED_VALIDATION    — metrics out of range, NaN, Inf, or required fields missing
DEGENERATE_RESULT    — experiment was structurally invalid (adapter not loaded,
                       trainable_params == 0, caught exception, empty predictions)
FAILED_RUNTIME       — task status was not SUCCEEDED (caller convenience)

Key design rule:
    A numerically bad result (accuracy=0.09) is SUCCEEDED_VALIDATED as long as the
    experiment ran correctly. Low performance is not an error; it is a finding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

_ADAPTER_METHODS = frozenset({
    "lora", "adalora", "ia3", "prefix", "prompt", "adapter"
})
_BOUNDED_METRICS = frozenset({
    "accuracy", "f1", "f1_macro", "f1_micro", "f1_weighted",
    "precision", "recall", "auc", "roc_auc", "balanced_accuracy",
})
_NON_NEGATIVE_METRICS = frozenset({"loss", "loss_start", "loss_end", "val_loss"})


class ResultStatus(StrEnum):
    SUCCEEDED_VALIDATED = "SUCCEEDED_VALIDATED"
    SUCCEEDED_WITH_WARNINGS = "SUCCEEDED_WITH_WARNINGS"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    DEGENERATE_RESULT = "DEGENERATE_RESULT"
    FAILED_RUNTIME = "FAILED_RUNTIME"


@dataclass(frozen=True)
class ValidationOutcome:
    status: ResultStatus
    poor_performance: bool | None
    metric_validated: bool
    degeneracy_flag: bool
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)


def validate_result(result: Any, *, method: str | None = None) -> ValidationOutcome:
    """Validate a task result dict and return a structured outcome.

    Args:
        result:  The dict returned by the handler (task.result).
        method:  The fine-tuning technique name (e.g. "adalora"). If not given,
                 falls back to result.get("method") or result.get("technique").

    Returns:
        ValidationOutcome describing whether the result is trustworthy.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── Structural check ──────────────────────────────────────────────────────
    if not isinstance(result, dict):
        return ValidationOutcome(
            status=ResultStatus.FAILED_VALIDATION,
            poor_performance=None,
            metric_validated=False,
            degeneracy_flag=False,
            validation_errors=[f"result must be a dict, got {type(result).__name__}"],
        )

    # ── Detect explicit caught exceptions ────────────────────────────────────
    # Handlers that swallow exceptions and return a result dict MUST set this.
    if result.get("caught_exception"):
        errors.append(
            f"caught_exception is set: {result['caught_exception']!r} — "
            "a silently-caught error cannot be exported as a valid result"
        )

    # ── Resolve method name ──────────────────────────────────────────────────
    effective_method = (
        method
        or result.get("method")
        or result.get("technique")
    )

    # ── Numeric metrics validation ───────────────────────────────────────────
    for key, raw in result.items():
        if not isinstance(raw, (int, float)):
            continue
        v = float(raw)
        if math.isnan(v):
            errors.append(f"metric {key!r} is NaN")
        elif math.isinf(v):
            errors.append(f"metric {key!r} is Inf")
        elif key in _BOUNDED_METRICS and not (0.0 <= v <= 1.0):
            errors.append(
                f"metric {key!r} = {v} is outside the expected range [0, 1]"
            )
        elif key in _NON_NEGATIVE_METRICS and v < 0:
            errors.append(f"metric {key!r} = {v} is negative — loss must be >= 0")

    # ── Eval examples ────────────────────────────────────────────────────────
    if "eval_examples" in result:
        n = result["eval_examples"]
        if isinstance(n, (int, float)) and n <= 0:
            errors.append(f"eval_examples = {n} — evaluation over zero examples")

    # ── Degeneracy checks ────────────────────────────────────────────────────
    degeneracy_reasons: list[str] = []

    # Predictions empty
    if result.get("predictions_non_empty") is False:
        degeneracy_reasons.append("predictions_non_empty is False")

    # Adapter methods: adapter_loaded and trainable_params
    is_adapter_method = (
        effective_method is not None and effective_method.lower() in _ADAPTER_METHODS
    )
    if "adapter_loaded" in result and result["adapter_loaded"] is False:
        degeneracy_reasons.append("adapter_loaded is False")
    if is_adapter_method and "trainable_params" in result:
        tp = result["trainable_params"]
        if isinstance(tp, (int, float)) and tp == 0:
            degeneracy_reasons.append(
                f"trainable_params == 0 for adapter method {effective_method!r}"
            )

    # trainable_params > total_params is nonsensical
    if "trainable_params" in result and "total_params" in result:
        tp = result.get("trainable_params")
        tot = result.get("total_params")
        if (
            isinstance(tp, (int, float))
            and isinstance(tot, (int, float))
            and tp > tot
        ):
            degeneracy_reasons.append(
                f"trainable_params ({tp}) > total_params ({tot})"
            )

    # Constant predictions (only warn unless also empty)
    if result.get("predictions_constant") is True:
        warnings.append(
            "predictions_constant is True — all predictions are the same class; "
            "may indicate model collapse"
        )

    # All bounded metrics are zero without explanation
    bounded_vals = {
        k: float(result[k])
        for k in _BOUNDED_METRICS
        if k in result and isinstance(result[k], (int, float))
    }
    if bounded_vals and all(v == 0.0 for v in bounded_vals.values()):
        if not result.get("all_zeros_expected"):
            degeneracy_reasons.append(
                "all bounded metrics (accuracy, f1, …) are exactly 0.0 "
                "without all_zeros_expected flag"
            )

    # ── Poor-performance flag (informational only, never an error) ───────────
    poor_performance: bool | None = None
    if any(k in result for k in ("accuracy", "f1", "auc", "roc_auc")):
        top_metric = max(
            (float(result[k]) for k in ("accuracy", "f1", "auc", "roc_auc") if k in result
             and isinstance(result[k], (int, float))),
            default=None,
        )
        if top_metric is not None and not math.isnan(top_metric) and not math.isinf(top_metric):
            poor_performance = top_metric < 0.55

    # ── Decide final status ──────────────────────────────────────────────────
    if errors or result.get("caught_exception"):
        status = ResultStatus.FAILED_VALIDATION
    elif degeneracy_reasons:
        errors.extend(degeneracy_reasons)
        status = ResultStatus.DEGENERATE_RESULT
    elif warnings:
        status = ResultStatus.SUCCEEDED_WITH_WARNINGS
    else:
        status = ResultStatus.SUCCEEDED_VALIDATED

    return ValidationOutcome(
        status=status,
        poor_performance=poor_performance,
        metric_validated=status in (
            ResultStatus.SUCCEEDED_VALIDATED, ResultStatus.SUCCEEDED_WITH_WARNINGS
        ),
        degeneracy_flag=status is ResultStatus.DEGENERATE_RESULT,
        validation_errors=errors,
        validation_warnings=warnings,
    )
