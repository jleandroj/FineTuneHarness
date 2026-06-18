"""Reproducibility assessment and manifest export for FineTuneHarness runs.

Three output levels:
  PASS    — seed + dataset_hashes + config_hash + git_commit + container_digest
  PARTIAL — same except container digest (environment snapshot only)
  FAIL    — missing seed, dataset_hashes, config_hash, or git_commit
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from finetuneharness.state.models import ArtifactRecord, EventRecord, RunRecord, TaskRecord


def _drain_modes(events: list[EventRecord] | None) -> list[str]:
    """Distinct execution modes recorded by 'drain_started' events, in first-seen order."""
    if not events:
        return []
    seen: list[str] = []
    for e in events:
        if e.kind == "drain_started":
            mode = e.payload.get("mode")
            if isinstance(mode, str) and mode not in seen:
                seen.append(mode)
    return seen


def canonical_json_hash(d: dict[str, Any]) -> str:
    """Stable SHA-256 of a dict: sorted keys, compact separators, UTF-8.

    Produces identical output regardless of the original key insertion order.
    Excludes nothing — callers are responsible for removing non-deterministic
    fields before hashing if needed.
    """
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReproducibilityAssessment:
    level: Literal["PASS", "PARTIAL", "FAIL"]
    bitwise_reproducible: bool
    replayable: bool
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def assess_reproducibility(
    run: RunRecord, events: list[EventRecord] | None = None
) -> ReproducibilityAssessment:
    """Assess how reproducible a run is based on what metadata was captured.

    PASS    → seed + dataset_hashes + config_hash + git_commit + container digest.
              These together are sufficient to reproduce the exact environment and data.
    PARTIAL → all of the above except container digest.
              Can reconstruct environment from package versions, but not bit-for-bit.
    FAIL    → missing seed, dataset_hashes, config_hash, or git_commit.
              Not enough information to attempt reproduction.

    When *events* are supplied, the actual execution mode is surfaced: a run drained
    under ``resource_aware`` concurrency is reproducible because each task runs in an
    isolated process (per-process RNG), but that fact is recorded as a warning so the
    distinction from gold-standard sequential execution is explicit.
    """
    missing: list[str] = []
    warnings: list[str] = []

    modes = _drain_modes(events)
    if "resource_aware" in modes:
        warnings.append(
            "run executed under resource_aware concurrency; reproducibility relies on "
            "per-process RNG isolation (each task runs in its own process). This is "
            "sound, but sequential execution remains the gold standard for bit-for-bit "
            "comparison."
        )

    # ── Hard requirements (FAIL if absent) ───────────────────────────────────
    if run.seed is None:
        missing.append("seed")

    if not run.dataset_hashes:
        missing.append("dataset_hashes")

    if not run.config_hash:
        missing.append("config_hash")

    git_commit = run.env_snapshot.get("git_commit")
    if not git_commit:
        missing.append("env_snapshot.git_commit")

    det = run.env_snapshot.get("determinism_env", {})
    # Determinism must be FORCED for results to be reproducible, not merely
    # metadata-complete. A run that did not force it cannot be PASS (capped below).
    determinism_forced = bool(det.get("harness_enforces_determinism"))
    if not determinism_forced:
        warnings.append(
            "deterministic algorithms were not forced for this run "
            "(torch.use_deterministic_algorithms / cudnn.deterministic) — results may "
            "not be bit-for-bit reproducible even with full metadata."
        )
    if det.get("CUBLAS_WORKSPACE_CONFIG") is None:
        warnings.append(
            "CUBLAS_WORKSPACE_CONFIG was not set — cuBLAS may use non-deterministic algorithms. "
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching for reproducible GPU results."
        )
    if det.get("CUDA_LAUNCH_BLOCKING") != "1":
        warnings.append(
            "CUDA_LAUNCH_BLOCKING was not set to 1 — async GPU execution may produce "
            "non-deterministic results under some workloads."
        )

    if run.env_snapshot.get("git_dirty"):
        warnings.append(
            "git working tree was dirty at run time — "
            "uncommitted changes may not be recoverable from git_commit alone"
        )
        if run.env_snapshot.get("git_diff_hash"):
            warnings[-1] += f" (diff_hash={run.env_snapshot['git_diff_hash'][:12]}...)"

    if missing:
        return ReproducibilityAssessment(
            level="FAIL",
            bitwise_reproducible=False,
            replayable=False,
            missing_fields=missing,
            warnings=warnings,
        )

    # ── Container check (PARTIAL if absent) ──────────────────────────────────
    container = run.env_snapshot.get("container", {})
    has_container_digest = bool(container.get("digest"))

    if not has_container_digest:
        has_image_name = bool(container.get("image"))
        base = (
            "container image name recorded but no digest — image tags are mutable "
            "and may point to different layers in the future. "
            if has_image_name
            else "no container digest recorded — "
        )
        warnings.append(
            base
            + "reproducibility_level=environment_snapshot_only. "
            "Package versions are captured, but system libraries (glibc, CUDA driver, firmware) "
            "are not pinned. Use Docker/Apptainer with a digest for full reproducibility."
        )
        return ReproducibilityAssessment(
            level="PARTIAL",
            bitwise_reproducible=False,
            replayable=True,
            missing_fields=[],
            warnings=warnings,
        )

    # ── PASS (capped to PARTIAL if determinism was not forced) ───────────────
    # Full metadata + a container digest, but if deterministic algorithms were not
    # forced the results are not bit-for-bit replayable, so this is not a true PASS.
    if not determinism_forced:
        return ReproducibilityAssessment(
            level="PARTIAL",
            bitwise_reproducible=False,
            replayable=True,
            missing_fields=[],
            warnings=warnings,
        )
    return ReproducibilityAssessment(
        level="PASS",
        bitwise_reproducible=False,  # bit-exact requires deterministic ops too
        replayable=True,
        missing_fields=[],
        warnings=warnings,
    )


def export_manifest(
    run: RunRecord,
    tasks: list[TaskRecord],
    artifacts: list[ArtifactRecord],
    events: list[EventRecord] | None = None,
) -> dict[str, Any]:
    """Export a self-contained reproducibility manifest for a run.

    The manifest contains everything needed to attempt replay:
    seeds, data hashes, config hash, git state, env snapshot,
    container info, hardware info, task results, and output hashes.

    When *events* are supplied, an ``execution`` section records the actual
    drain mode(s) the run ran under, so the manifest does not silently imply
    sequential execution.
    """
    assessment = assess_reproducibility(run, events)
    drain_modes = _drain_modes(events)

    env = run.env_snapshot
    git_info = {
        "commit": env.get("git_commit"),
        "branch": env.get("git_branch"),
        "dirty": env.get("git_dirty", False),
        "diff_hash": env.get("git_diff_hash"),
    }

    output_hashes = {
        a.artifact_id: {
            "kind": a.kind,
            "path": a.path,
            "sha256": a.checksum,
            "task_id": a.task_id,
        }
        for a in artifacts
    }

    task_summaries = [
        {
            "task_id": t.task_id,
            "task_key": t.task_key,
            "status": t.status.value,
            "result": t.result,
            "error": t.error,
            "attempt_count": t.attempt_count,
        }
        for t in tasks
    ]

    return {
        "manifest_version": "1",
        "run_id": run.run_id,
        "name": run.name,
        "created_at": run.created_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        # ── Reproducibility core ──────────────────────────────────────────────
        "seed": run.seed,
        "dataset_hashes": run.dataset_hashes,
        "config_hash": run.config_hash,
        "config": run.config,
        # ── Code provenance ───────────────────────────────────────────────────
        "git": git_info,
        # ── Environment ───────────────────────────────────────────────────────
        "env_snapshot": env,
        "container": env.get("container", {"engine": "none"}),
        "hardware": env.get("hardware", {}),
        # ── Execution ─────────────────────────────────────────────────────────
        "execution": {
            # Empty when no events were supplied (mode unknown to the manifest).
            "drain_modes": drain_modes,
        },
        # ── Assessment ────────────────────────────────────────────────────────
        "reproducibility": {
            "level": assessment.level,
            "bitwise_reproducible": assessment.bitwise_reproducible,
            "replayable": assessment.replayable,
            "missing_fields": assessment.missing_fields,
            "warnings": assessment.warnings,
        },
        # ── Tasks and outputs ─────────────────────────────────────────────────
        "tasks": task_summaries,
        "output_hashes": output_hashes,
    }
