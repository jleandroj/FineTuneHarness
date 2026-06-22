"""Tests for the config guardrails (self-sabotaging admission configs)."""
from __future__ import annotations

from finetuneharness.executor.resources import ConcurrencyConfig
from finetuneharness.verification import check_concurrency_config


def _codes(cfg, **kw):
    return {f.code for f in check_concurrency_config(cfg, **kw)}


def test_sane_utilization_config_passes() -> None:
    cfg = ConcurrencyConfig(mode="resource_aware", admission="utilization",
                            max_util_pct=70, max_concurrent=4)
    assert check_concurrency_config(cfg, gpu_count=1) == []


def test_sequential_never_warns() -> None:
    cfg = ConcurrencyConfig(mode="sequential", min_free_mb=0, max_concurrent=999)
    assert check_concurrency_config(cfg) == []


def test_disabled_memory_gate_warns() -> None:
    cfg = ConcurrencyConfig(mode="resource_aware", admission="memory", min_free_mb=0)
    assert "GATE_DISABLED" in _codes(cfg)


def test_high_util_threshold_with_high_cap_warns() -> None:
    # The GB10 footgun: 32 / 95.
    cfg = ConcurrencyConfig(mode="resource_aware", admission="utilization",
                            max_util_pct=95, max_concurrent=32)
    assert "OVER_ADMISSION_RISK" in _codes(cfg)


def test_high_cap_relative_to_gpu_count_warns() -> None:
    cfg = ConcurrencyConfig(mode="resource_aware", admission="utilization",
                            max_util_pct=70, max_concurrent=32)
    assert "HIGH_CAP" in _codes(cfg, gpu_count=1)
    # Plenty of GPUs → the same cap is fine.
    assert "HIGH_CAP" not in _codes(cfg, gpu_count=8)
