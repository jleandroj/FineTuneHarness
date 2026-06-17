from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    auc: float | None = None
    n_params: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }
        if self.auc is not None:
            d["auc"] = self.auc
        if self.n_params is not None:
            d["n_params"] = self.n_params
        return d


def from_result(result: dict[str, Any]) -> ClassificationMetrics:
    """Extract ClassificationMetrics from a task result dict.

    Expects keys: accuracy, f1, precision, recall. auc and n_params are optional.
    Raises ValueError if required keys are missing.
    """
    missing = [k for k in ("accuracy", "f1", "precision", "recall") if k not in result]
    if missing:
        raise ValueError(f"result dict missing required metric keys: {missing}")
    return ClassificationMetrics(
        accuracy=float(result["accuracy"]),
        f1=float(result["f1"]),
        precision=float(result["precision"]),
        recall=float(result["recall"]),
        auc=float(result["auc"]) if "auc" in result else None,
        n_params=int(result["n_params"]) if "n_params" in result else None,
    )


def best_metric(metrics: ClassificationMetrics, key: str) -> float:
    value = getattr(metrics, key, None)
    if value is None:
        raise ValueError(f"metric {key!r} is not available in ClassificationMetrics")
    return float(value)
