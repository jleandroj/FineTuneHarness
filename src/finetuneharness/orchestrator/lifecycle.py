from __future__ import annotations

from finetuneharness.state.models import RunStatus, TaskStatus


ALLOWED_RUN_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.CREATED: {RunStatus.VALIDATED, RunStatus.CANCELLED},
    RunStatus.VALIDATED: {RunStatus.RUNNING, RunStatus.CANCELLED},
    RunStatus.RUNNING: {RunStatus.PARTIAL_FAILED, RunStatus.FAILED, RunStatus.COMPLETED, RunStatus.CANCELLED},
    RunStatus.PARTIAL_FAILED: {RunStatus.FAILED, RunStatus.COMPLETED, RunStatus.CANCELLED},
    RunStatus.FAILED: set(),
    RunStatus.COMPLETED: set(),
    RunStatus.CANCELLED: set(),
}

ALLOWED_TASK_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.LEASED, TaskStatus.CANCELLED},
    TaskStatus.LEASED: {TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT},
    # PENDING is allowed from RUNNING to support the retry path (worker re-queues on failure)
    TaskStatus.RUNNING: {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED, TaskStatus.PENDING},
    TaskStatus.SUCCEEDED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.TIMED_OUT: set(),
    TaskStatus.CANCELLED: set(),
}


def ensure_run_transition(current: RunStatus, target: RunStatus) -> None:
    if target not in ALLOWED_RUN_TRANSITIONS[current]:
        raise ValueError(f"invalid run transition: {current} -> {target}")



def ensure_task_transition(current: TaskStatus, target: TaskStatus) -> None:
    if target not in ALLOWED_TASK_TRANSITIONS[current]:
        raise ValueError(f"invalid task transition: {current} -> {target}")
