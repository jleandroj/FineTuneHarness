"""Pre-edit / contamination guard — 'is it safe to touch the code right now?'

Editing a runtime module while a run is in flight is unsafe: each task is a fresh
`spawn` that re-imports the harness from disk, so a mid-run edit splits the parent
(old code, already imported) from new children (edited code) and silently corrupts
the experiment. This module lets an agent or pre-commit hook *detect* that condition
instead of relying on discipline.

`find_active_runs` returns runs with in-flight (LEASED/RUNNING) tasks; `preflight`
turns that into a go/no-go for editing. See docs/LIE_RESISTANCE_AUDIT.md (dim 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from finetuneharness.state.models import TaskStatus
from finetuneharness.state.store import StateStore

_IN_FLIGHT = (TaskStatus.LEASED, TaskStatus.RUNNING)


@dataclass(frozen=True)
class ActiveRun:
    run_id: str
    name: str
    in_flight: int          # LEASED + RUNNING tasks
    lease_owners: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreflightReport:
    safe_to_edit: bool
    active: list[ActiveRun] = field(default_factory=list)


def find_active_runs(store: StateStore) -> list[ActiveRun]:
    """Runs that have tasks currently LEASED or RUNNING (potentially executing).

    Note: a hard-crashed run can leave tasks stranded RUNNING with no live worker —
    those still show here (correctly: their lease is unresolved). Resolve with
    `recover-run` before treating the code as safe to edit.
    """
    active: list[ActiveRun] = []
    for run in store.list_runs():
        tasks = store.list_tasks(run.run_id)
        in_flight = [t for t in tasks if t.status in _IN_FLIGHT]
        if in_flight:
            owners = sorted({t.lease_owner for t in in_flight if t.lease_owner})
            active.append(ActiveRun(
                run_id=run.run_id, name=run.name,
                in_flight=len(in_flight), lease_owners=owners,
            ))
    return active


def preflight(store: StateStore) -> PreflightReport:
    """Go/no-go for editing runtime code: NOT safe while any run has in-flight tasks."""
    active = find_active_runs(store)
    return PreflightReport(safe_to_edit=not active, active=active)
