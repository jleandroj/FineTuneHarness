from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import asdict
from typing import Any

from datetime import datetime, timezone

from finetuneharness.observability.logging import get_logger
from finetuneharness.orchestrator.approval import ApprovalGate
from finetuneharness.orchestrator.lifecycle import ensure_run_transition
from finetuneharness.state.env_snapshot import capture_env_snapshot
from finetuneharness.state.models import EventRecord, RunRecord, RunStatus, TaskRecord, TaskStatus
from finetuneharness.state.reproducibility import canonical_json_hash
from finetuneharness.state.store import StateStore
from finetuneharness.validation.configs import validate_run_config

_TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})


class FineTuneRunner:
    """Bootstrap runner with explicit state transitions and status aggregation."""

    def __init__(self, store: StateStore) -> None:
        self._store = store
        self._log = get_logger("finetuneharness.runner")

    def create_run(self, *, name: str, config: dict[str, Any], tasks: list[dict[str, Any]]) -> str:
        validate_run_config(config)
        run_id = uuid.uuid4().hex

        # Extract reproducibility fields from config at creation time so they
        # are first-class fields on RunRecord, not buried in config_json.
        seed: int = config["seed"]
        dataset_hashes: dict[str, str]
        if "dataset_hash" in config:
            dataset_hashes = {"default": config["dataset_hash"]}
        else:
            dataset_hashes = dict(config["datasets"])
        config_hash = canonical_json_hash(config)

        run = RunRecord(
            run_id=run_id,
            name=name,
            status=RunStatus.CREATED,
            config=config,
            created_at=datetime.now(timezone.utc),
            env_snapshot=capture_env_snapshot(),
            seed=seed,
            dataset_hashes=dataset_hashes,
            config_hash=config_hash,
        )
        self._store.create_run(run)
        self._store.append_event(EventRecord(event_id=uuid.uuid4().hex, run_id=run_id, task_id=None, kind="run_created", payload={"name": name}))

        ensure_run_transition(RunStatus.CREATED, RunStatus.VALIDATED)
        self._store.update_run_status(run_id, RunStatus.VALIDATED)
        self._store.append_event(EventRecord(event_id=uuid.uuid4().hex, run_id=run_id, task_id=None, kind="run_validated", payload={}))

        for index, payload in enumerate(tasks):
            task_id = uuid.uuid4().hex
            task = TaskRecord(
                task_id=task_id,
                run_id=run_id,
                task_key=payload.get("task_key", f"task-{index}"),
                status=TaskStatus.PENDING,
                payload=payload,
            )
            self._store.create_task(task)
            self._store.append_event(EventRecord(event_id=uuid.uuid4().hex, run_id=run_id, task_id=task_id, kind="task_created", payload=payload))

        self._log.info("run_created", extra={"run_id": run_id, "task_count": len(tasks), "config": asdict(run)})
        return run_id

    def refresh_run_status(self, run_id: str) -> RunStatus:
        run = self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run_id: {run_id}")
        tasks = self._store.list_tasks(run_id)
        if not tasks:
            target = RunStatus.VALIDATED
        else:
            statuses = {task.status for task in tasks}
            attempted = any(task.attempt_count > 0 for task in tasks)
            # DEGENERATE is a non-success terminal: the handler returned but the
            # result failed validation. It must never count toward COMPLETED.
            unsuccessful = {TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.DEGENERATE}
            in_progress = {TaskStatus.PENDING, TaskStatus.LEASED, TaskStatus.RUNNING}
            if all(status is TaskStatus.SUCCEEDED for status in statuses):
                target = RunStatus.COMPLETED
            elif (statuses & unsuccessful) and (TaskStatus.SUCCEEDED in statuses or (statuses & in_progress)):
                target = RunStatus.PARTIAL_FAILED
            elif statuses.issubset(unsuccessful | {TaskStatus.CANCELLED}):
                target = RunStatus.FAILED
            elif TaskStatus.RUNNING in statuses or TaskStatus.LEASED in statuses or TaskStatus.SUCCEEDED in statuses or attempted:
                target = RunStatus.RUNNING
            else:
                target = RunStatus.VALIDATED

        if target != run.status:
            # If the direct transition is invalid, step through RUNNING first.
            # This happens when tasks are completed without ever reaching the RUNNING
            # run-level state (e.g. in tests or if the worker was bypassed).
            from finetuneharness.orchestrator.lifecycle import ALLOWED_RUN_TRANSITIONS
            if target not in ALLOWED_RUN_TRANSITIONS.get(run.status, set()):
                if RunStatus.RUNNING in ALLOWED_RUN_TRANSITIONS.get(run.status, set()) and \
                        target in ALLOWED_RUN_TRANSITIONS.get(RunStatus.RUNNING, set()):
                    ensure_run_transition(run.status, RunStatus.RUNNING)
                    self._store.update_run_status(run_id, RunStatus.RUNNING)
                    self._store.append_event(EventRecord(
                        event_id=uuid.uuid4().hex, run_id=run_id, task_id=None,
                        kind="run_status_changed",
                        payload={"from": run.status.value, "to": RunStatus.RUNNING.value},
                    ))
                    run = self._store.get_run(run_id)
            ensure_run_transition(run.status, target)
            self._store.update_run_status(run_id, target)
            if target in _TERMINAL_STATUSES and run.finished_at is None:
                self._store.update_run_finished_at(run_id, datetime.now(timezone.utc))
            self._store.append_event(
                EventRecord(
                    event_id=uuid.uuid4().hex,
                    run_id=run_id,
                    task_id=None,
                    kind="run_status_changed",
                    payload={"from": run.status.value, "to": target.value},
                )
            )
        return target

    def await_approval(self, run_id: str, gate: ApprovalGate) -> None:
        """Run the gate check for a validated run. Raises ApprovalError if denied."""
        run = self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run_id: {run_id}")
        tasks = self._store.list_tasks(run_id)
        actor = gate.check(run, tasks)
        self._store.append_event(
            EventRecord(
                event_id=uuid.uuid4().hex,
                run_id=run_id,
                task_id=None,
                kind="run_approved",
                payload={"gate": type(gate).__name__, "actor": actor},
            )
        )
        self._log.info("run_approved", extra={"run_id": run_id, "actor": actor})

    def get_run_status(self, run_id: str) -> dict[str, object]:
        status = self.refresh_run_status(run_id)
        run = self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run_id: {run_id}")
        tasks = self._store.list_tasks(run_id)
        counts = Counter(task.status.value for task in tasks)
        return {
            "run_id": run.run_id,
            "name": run.name,
            "status": status.value,
            "task_counts": dict(sorted(counts.items())),
            "task_total": len(tasks),
        }
