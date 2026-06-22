"""Independent verification of run results — the 'no silent lies' layer.

Separate from the code that *produces* results: it re-checks claims a handler made,
catching fabricated or internally-inconsistent results that pass the generic
result validator (which only checks ranges/NaN/declared degeneracy). See
docs/LIE_RESISTANCE_AUDIT.md.
"""
from __future__ import annotations

from finetuneharness.verification.guardrails import check_concurrency_config
from finetuneharness.verification.integrity import (
    ActiveRun,
    PreflightReport,
    find_active_runs,
    preflight,
)
from finetuneharness.verification.verifier import (
    Finding,
    VerificationReport,
    Verdict,
    verify_run,
)

__all__ = [
    "Finding", "VerificationReport", "Verdict", "verify_run",
    "ActiveRun", "PreflightReport", "find_active_runs", "preflight",
    "check_concurrency_config",
]
