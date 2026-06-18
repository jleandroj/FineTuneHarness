from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    CREATED = "created"
    VALIDATED = "validated"
    RUNNING = "running"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskStatus(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    status: RunStatus
    config: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    env_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    run_id: str
    task_key: str
    status: TaskStatus
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    lease_owner: str | None = None
    leased_until: datetime | None = None
    attempt_count: int = 0


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    run_id: str
    task_id: str | None
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    run_id: str
    task_id: str | None
    kind: str
    path: str
    checksum: str
