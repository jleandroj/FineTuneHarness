"""Domain-specific validators for biology/genomics fine-tuning experiments.

These extend the generic COMMON_INPUT_SCHEMA with fields that only make sense
in a biological sequence context (k-mer size, per-species sample caps, etc.).
They are intentionally outside the harness core — see CLAUDE.md.
"""
from __future__ import annotations

from typing import Any


def validate_bio_input(payload: dict[str, Any]) -> None:
    """Validate biology-domain fields on top of the generic input contract.

    Fields validated (all optional; only checked when present):
      k             — k-mer size, must be int in [1, 9]
      max_per_species — max training samples per species, must be positive int

    The [1, 9] range matches the architecture-ablation grid (k=1..9). An earlier
    [1, 6] cap was stale and would have wrongly rejected a third of that grid if
    this validator were wired into a run.
    """
    if "k" in payload:
        if not (isinstance(payload["k"], int) and 1 <= payload["k"] <= 9):
            raise ValueError(f"k (k-mer size) must be an int in [1, 9], got {payload['k']}")
    if "max_per_species" in payload:
        if not (isinstance(payload["max_per_species"], int) and payload["max_per_species"] > 0):
            raise ValueError(
                f"max_per_species must be a positive int, got {payload['max_per_species']}"
            )


def validate_bio_output(result: dict[str, Any]) -> None:
    """Validate biology-domain output fields (all optional; checked when present).

    Moved here from the generic core validator: ``k`` (k-mer size) is meaningless
    for non-genomic skills (e.g. an NLP run may emit ``k`` as a top-k value), so
    the generic harness must not impose the [1, 9] range on it.

    Fields validated:
      k — k-mer size used to produce the result, must be int in [1, 9]
    """
    if "k" in result:
        if not (isinstance(result["k"], int) and 1 <= result["k"] <= 9):
            raise ValueError(f"k (k-mer size) must be an int in [1, 9], got {result['k']}")
