from __future__ import annotations

import pytest

from finetuneharness.evaluation.metrics import ClassificationMetrics, best_metric, from_result


def test_from_result_happy_path() -> None:
    result = {"accuracy": 0.90, "f1": 0.88, "precision": 0.87, "recall": 0.89}
    m = from_result(result)
    assert m.accuracy == pytest.approx(0.90)
    assert m.f1 == pytest.approx(0.88)
    assert m.precision == pytest.approx(0.87)
    assert m.recall == pytest.approx(0.89)
    assert m.auc is None
    assert m.n_params is None


def test_from_result_with_optional_fields() -> None:
    result = {
        "accuracy": 0.91,
        "f1": 0.89,
        "precision": 0.88,
        "recall": 0.90,
        "auc": 0.95,
        "n_params": 1_000_000,
    }
    m = from_result(result)
    assert m.auc == pytest.approx(0.95)
    assert m.n_params == 1_000_000


def test_from_result_missing_required_key_raises() -> None:
    result = {"accuracy": 0.90, "f1": 0.88, "precision": 0.87}  # missing recall
    with pytest.raises(ValueError, match="recall"):
        from_result(result)


def test_from_result_missing_all_raises() -> None:
    with pytest.raises(ValueError):
        from_result({})


def test_to_dict_minimal() -> None:
    m = ClassificationMetrics(accuracy=0.9, f1=0.85, precision=0.84, recall=0.86)
    d = m.to_dict()
    assert d == {"accuracy": 0.9, "f1": 0.85, "precision": 0.84, "recall": 0.86}
    assert "auc" not in d
    assert "n_params" not in d


def test_to_dict_includes_optional_when_set() -> None:
    m = ClassificationMetrics(
        accuracy=0.9, f1=0.85, precision=0.84, recall=0.86, auc=0.93, n_params=50_000
    )
    d = m.to_dict()
    assert d["auc"] == pytest.approx(0.93)
    assert d["n_params"] == 50_000


def test_best_metric_known_key() -> None:
    m = ClassificationMetrics(accuracy=0.9, f1=0.85, precision=0.84, recall=0.86)
    assert best_metric(m, "accuracy") == pytest.approx(0.9)
    assert best_metric(m, "f1") == pytest.approx(0.85)


def test_best_metric_auc_none_raises() -> None:
    m = ClassificationMetrics(accuracy=0.9, f1=0.85, precision=0.84, recall=0.86)
    with pytest.raises(ValueError, match="auc"):
        best_metric(m, "auc")


def test_best_metric_unknown_key_raises() -> None:
    m = ClassificationMetrics(accuracy=0.9, f1=0.85, precision=0.84, recall=0.86)
    with pytest.raises(ValueError):
        best_metric(m, "rmse")


def test_from_result_roundtrip() -> None:
    result = {"accuracy": 0.92, "f1": 0.91, "precision": 0.90, "recall": 0.92, "auc": 0.97}
    m = from_result(result)
    d = m.to_dict()
    assert d["accuracy"] == pytest.approx(0.92)
    assert d["auc"] == pytest.approx(0.97)
