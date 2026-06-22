"""Independent verification of run results — the 'no silent lies' layer.

Separate from the code that *produces* results: it re-checks claims a handler made,
catching fabricated or internally-inconsistent results that pass the generic
result validator (which only checks ranges/NaN/declared degeneracy). See
docs/LIE_RESISTANCE_AUDIT.md.
"""
from __future__ import annotations

from finetuneharness.verification.verifier import (
    Finding,
    VerificationReport,
    Verdict,
    verify_run,
)

__all__ = ["Finding", "VerificationReport", "Verdict", "verify_run"]
