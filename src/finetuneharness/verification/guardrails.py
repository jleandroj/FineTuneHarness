"""Config guardrails — catch self-sabotaging admission configs BEFORE launch.

A run can silently disable its own safety: a memory gate set to 0, or a high
concurrency cap on a noisy utilization signal (the GB10 case where 32/95 caused a
53% timeout cascade). These checks surface such footguns loudly at launch instead of
letting the run discover them by failing. Advisory by default (warnings); the caller
decides whether to block. See docs/LIE_RESISTANCE_AUDIT.md (dim 10).
"""
from __future__ import annotations

from finetuneharness.executor.resources import ConcurrencyConfig
from finetuneharness.verification.verifier import Finding

# Above this many in-flight tasks per GPU, contention/timeout risk is high for
# training workloads on a single device (calibration learned on the GB10).
_PER_GPU_SOFT_CAP = 8


def check_concurrency_config(
    concurrency: ConcurrencyConfig, *, gpu_count: int | None = None
) -> list[Finding]:
    """Return advisory findings for a risky admission config (empty == looks sane)."""
    findings: list[Finding] = []

    def warn(code: str, detail: str) -> None:
        findings.append(Finding("warn", code, None, detail))

    if concurrency.mode == "sequential":
        return findings  # one at a time — nothing to over-admit

    if concurrency.admission == "memory" and concurrency.min_free_mb <= 0:
        warn("GATE_DISABLED",
             "memory admission with min_free_mb<=0 disables the memory gate — "
             "concurrency is then bounded only by max_concurrent")

    if concurrency.admission == "utilization" and concurrency.max_util_pct >= 90 \
            and concurrency.max_concurrent > _PER_GPU_SOFT_CAP:
        warn("OVER_ADMISSION_RISK",
             f"max_util_pct={concurrency.max_util_pct:.0f} with max_concurrent="
             f"{concurrency.max_concurrent}: the utilization signal is noisy (dips during "
             "data-load/eval), so a high cap can ramp up and over-subscribe the GPU. On a "
             "single GB10, 32/95 produced ~53% timeouts; 4/70 produced 0. Calibrate.")

    if gpu_count is not None and gpu_count >= 1 \
            and concurrency.max_concurrent > _PER_GPU_SOFT_CAP * gpu_count:
        warn("HIGH_CAP",
             f"max_concurrent={concurrency.max_concurrent} is high for {gpu_count} GPU(s) "
             f"(> {_PER_GPU_SOFT_CAP}/GPU); training tasks usually contend well before this")

    return findings
