"""Deterministic seeding: applied by the worker before every handler call.

The harness owns seed application so handlers do not need to seed manually.
torch and numpy are optional — degrades gracefully when not installed.

Per-cell seeding policy: every task (grid cell) in a run is seeded with the SAME
run-level seed — the worker calls ``apply_seed(run.seed)`` before each task, in
both sequential and concurrent (per-process) modes. This makes each cell's RNG
stream reproducible run-to-run, but it does NOT vary the seed per cell: two cells
that differ only in a non-seeded knob start from identical randomness. A handler
that needs per-cell-distinct randomness (e.g. different data shuffles per cell)
must derive its own seed, e.g. ``hash((run_seed, task.task_key))`` or a
``np.random.default_rng(...)`` keyed on the task.
"""
from __future__ import annotations

import os
import random


def apply_seed(seed: int) -> None:
    """Seed all known random sources with *seed*.

    Applied sources (in order):
      1. Python built-in ``random``
      2. ``numpy.random`` legacy global RNG (if numpy is importable)
      3. ``torch`` CPU + CUDA (if torch is importable)
      4. ``CUBLAS_WORKSPACE_CONFIG`` env var — set to ``:4096:8`` if not already
         present, so cuBLAS uses a deterministic workspace algorithm the next
         time a cuBLAS workspace is created.

    **Known limitations (handlers must seed these explicitly):**

    *numpy.random.default_rng()*: ``np.random.seed()`` only seeds the *legacy*
    numpy global RNG. Handlers that call ``np.random.default_rng()`` without an
    explicit seed receive an independent, unseeded Generator.
    Fix: ``rng = np.random.default_rng(task.payload["seed"])``

    *JAX*: JAX uses explicit functional PRNG keys — there is no global state to
    seed. ``apply_seed`` does not import or affect JAX.
    Fix: ``key = jax.random.PRNGKey(task.payload["seed"])`` inside the handler.

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

    # CUBLAS_WORKSPACE_CONFIG must be set BEFORE torch.use_deterministic_algorithms
    # for deterministic CUDA matmuls; set it first.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        import torch
        torch.manual_seed(seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except RuntimeError:
            pass
        # Force deterministic algorithms. warn_only=True so ops without a
        # deterministic implementation warn instead of raising (the run still
        # proceeds). cuDNN is pinned to deterministic + no autotuning.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except ImportError:
        pass
