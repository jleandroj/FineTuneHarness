"""Tests for evaluation/validator.py — semantic result validation.

Covers exactly the 10 tests the user requested plus a few structural ones.
"""
from __future__ import annotations

import math

from finetuneharness.evaluation.validator import ResultStatus, validate_result


# ── 1. Low accuracy can be a valid result ─────────────────────────────────────

def test_low_accuracy_can_be_valid_result():
    """accuracy=0.09 with a real adapter run must be SUCCEEDED_VALIDATED."""
    result = {
        "method": "adalora",
        "accuracy": 0.09,
        "f1": 0.03,
        "adapter_loaded": True,
        "trainable_params": 1_200_000,
        "total_params": 7_000_000,
        "eval_examples": 1000,
        "predictions_non_empty": True,
    }
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.SUCCEEDED_VALIDATED
    assert outcome.degeneracy_flag is False
    assert outcome.metric_validated is True
    assert outcome.poor_performance is True  # 0.09 < 0.55 threshold


# ── 2. AdaLoRA with zero trainable params is degenerate ──────────────────────

def test_adalora_zero_trainable_params_is_degenerate():
    result = {
        "method": "adalora",
        "accuracy": 0.53,
        "f1": 0.41,
        "adapter_loaded": True,
        "trainable_params": 0,
        "total_params": 7_000_000,
        "eval_examples": 1000,
    }
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.DEGENERATE_RESULT
    assert outcome.degeneracy_flag is True
    assert outcome.metric_validated is False
    assert any("trainable_params" in e for e in outcome.validation_errors)


# ── 3. Adapter not loaded is degenerate ──────────────────────────────────────

def test_adapter_not_loaded_is_degenerate():
    result = {
        "method": "lora",
        "accuracy": 0.53,
        "f1": 0.41,
        "adapter_loaded": False,
        "trainable_params": 1_000_000,
        "eval_examples": 1000,
    }
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.DEGENERATE_RESULT
    assert outcome.degeneracy_flag is True
    assert any("adapter_loaded" in e for e in outcome.validation_errors)


# ── 4. NaN metric fails validation ────────────────────────────────────────────

def test_nan_metric_fails_validation():
    result = {"method": "sft", "accuracy": float("nan"), "f1": 0.7}
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.FAILED_VALIDATION
    assert any("NaN" in e for e in outcome.validation_errors)
    assert outcome.metric_validated is False


# ── 5. Inf metric fails validation ────────────────────────────────────────────

def test_inf_metric_fails_validation():
    result = {"method": "sft", "accuracy": 0.8, "loss": float("inf")}
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.FAILED_VALIDATION
    assert any("Inf" in e for e in outcome.validation_errors)


# ── 6. Accuracy out of range fails validation ─────────────────────────────────

def test_accuracy_out_of_range_fails_validation():
    result = {"method": "sft", "accuracy": 1.5, "f1": 0.8}
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.FAILED_VALIDATION
    assert any("[0, 1]" in e for e in outcome.validation_errors)


def test_accuracy_negative_fails_validation():
    result = {"method": "sft", "accuracy": -0.1, "f1": 0.8}
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.FAILED_VALIDATION


# ── 7. Empty predictions are degenerate ───────────────────────────────────────

def test_empty_predictions_are_degenerate():
    result = {
        "method": "lora",
        "accuracy": 0.5,
        "f1": 0.4,
        "predictions_non_empty": False,
    }
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.DEGENERATE_RESULT
    assert outcome.degeneracy_flag is True


# ── 8. Constant predictions produce a warning ─────────────────────────────────

def test_constant_predictions_warn_or_degenerate():
    result = {
        "method": "bitfit",
        "accuracy": 0.53,
        "f1": 0.0,
        "predictions_constant": True,
        "predictions_non_empty": True,
    }
    outcome = validate_result(result)
    # Constant predictions are a WARNING, not an error — the result is still
    # structurally valid but suspect.
    assert outcome.status in (
        ResultStatus.SUCCEEDED_WITH_WARNINGS,
        ResultStatus.DEGENERATE_RESULT,
    )
    assert outcome.validation_warnings or outcome.validation_errors


# ── 9. Handler exception cannot be exported as success ───────────────────────

def test_handler_exception_cannot_be_exported_as_success():
    result = {
        "method": "prefix",
        "accuracy": 0.53,
        "caught_exception": "RuntimeError: CUDA OOM",
    }
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.FAILED_VALIDATION
    assert outcome.metric_validated is False
    assert any("caught_exception" in e for e in outcome.validation_errors)


# ── 10. Comparator ignores invalid/degenerate results ─────────────────────────

def test_comparator_ignores_nan_metrics():
    """_numeric_metrics must exclude NaN so it never reaches delta computation."""
    from finetuneharness.evaluation.comparator import _numeric_metrics
    result = {"accuracy": 0.8, "loss": float("nan"), "f1": float("inf"), "wall_seconds": 1.0}
    metrics = _numeric_metrics(result)
    assert "accuracy" in metrics
    assert "loss" not in metrics
    assert "f1" not in metrics
    assert math.isfinite(metrics["accuracy"])


# ── Additional structural tests ───────────────────────────────────────────────

def test_non_dict_result_fails_validation():
    outcome = validate_result("not a dict")
    assert outcome.status == ResultStatus.FAILED_VALIDATION
    assert "dict" in outcome.validation_errors[0]


def test_negative_loss_fails_validation():
    result = {"method": "sft", "accuracy": 0.8, "loss": -0.5}
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.FAILED_VALIDATION
    assert any("negative" in e for e in outcome.validation_errors)


def test_zero_eval_examples_fails():
    result = {"method": "sft", "accuracy": 0.8, "eval_examples": 0}
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.FAILED_VALIDATION


def test_method_passed_explicitly_overrides_result_field():
    """method kwarg should take precedence for adapter checks."""
    result = {
        "technique": "bitfit",  # won't match adapter check
        "accuracy": 0.8,
        "trainable_params": 0,
    }
    # When method="adalora" is passed explicitly, trainable_params=0 is degenerate
    outcome = validate_result(result, method="adalora")
    assert outcome.status == ResultStatus.DEGENERATE_RESULT


def test_valid_high_accuracy_result():
    result = {
        "method": "sft",
        "accuracy": 0.92,
        "f1": 0.91,
        "precision": 0.90,
        "recall": 0.92,
        "loss": 0.18,
        "eval_examples": 5000,
    }
    outcome = validate_result(result)
    assert outcome.status == ResultStatus.SUCCEEDED_VALIDATED
    assert outcome.poor_performance is False
    assert outcome.degeneracy_flag is False


# --- Adapter (LoRA/PEFT) degeneration ---

def test_lora_adapter_not_loaded_is_degenerate() -> None:
    from finetuneharness.evaluation.validator import ResultStatus, validate_result

    outcome = validate_result({"accuracy": 0.92, "method": "lora", "adapter_loaded": False})
    assert outcome.status is ResultStatus.DEGENERATE_RESULT
    assert outcome.degeneracy_flag is True
    assert any("adapter_loaded" in e for e in outcome.validation_errors)


def test_lora_zero_trainable_params_is_degenerate() -> None:
    from finetuneharness.evaluation.validator import ResultStatus, validate_result

    outcome = validate_result({"accuracy": 0.92, "method": "lora", "trainable_params": 0})
    assert outcome.status is ResultStatus.DEGENERATE_RESULT
    assert any("trainable_params" in e for e in outcome.validation_errors)


def test_healthy_lora_adapter_is_valid() -> None:
    from finetuneharness.evaluation.validator import ResultStatus, validate_result

    outcome = validate_result({
        "accuracy": 0.92, "method": "lora",
        "adapter_loaded": True, "trainable_params": 4096, "total_params": 1_000_000,
    })
    assert outcome.status is ResultStatus.SUCCEEDED_VALIDATED
