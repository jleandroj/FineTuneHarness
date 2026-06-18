"""Deterministic seeding: applied by the worker before every handler call.

The harness owns seed application so handlers do not need to seed manually.
torch and numpy are optional — degrades gracefully when not installed.
"""
from __future__ import annotations

import os
import random


def apply_seed(seed: int) -> None:
    """Seed all known random sources with *seed*.

    Applied sources (in order):
      1. Python built-in ``random``
      2. ``numpy.random`` (if numpy is importable)
      3. ``torch`` CPU + CUDA (if torch is importable)
      4. ``CUBLAS_WORKSPACE_CONFIG`` env var — set to ``:4096:8`` if not already
         present, so cuBLAS uses a deterministic workspace algorithm the next
         time a cuBLAS workspace is created.

    Note on CUBLAS_WORKSPACE_CONFIG: the variable is read when the cuBLAS
    workspace is first allocated per-stream, not at CUDA init. Setting it here,
    before the handler runs any CUDA ops, is sufficient in the common case
    (fresh CUDA context per task). If a task reuses a pre-warmed CUDA context
    across calls the setting may have no effect; use a fresh process per run
    for bit-exact GPU reproducibility.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except RuntimeError:
            pass
    except ImportError:
        pass

    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
