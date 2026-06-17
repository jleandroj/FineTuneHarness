from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from finetuneharness.orchestrator.lifecycle import ensure_task_transition
from finetuneharness.state.leases import Lease, utc_now
from finetuneharness.state.models import ArtifactRecord, EventRecord, RunRecord, RunStatus, TaskRecord, TaskStatus
from finetuneharness.state.store import StateStore

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


class SQLiteStateStore(StateStore):
    """SQLite-backed persistent state store for local serious use."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    lease_owner TEXT,
                    leased_until TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    UNIQUE(run_id, task_key)
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                """
            )
            task_cols = {row['name'] for row in conn.execute("PRAGMA table_info(tasks)")}
            if 'attempt_count' not in task_cols:
                conn.execute("ALTER TABLE tasks ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0")
            event_cols = {row['name'] for row in conn.execute("PRAGMA table_info(events)")}
            if 'created_at' not in event_cols:
                conn.execute("ALTER TABLE events ADD COLUMN created_at TEXT")

    def create_run(self, run: RunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, name, status, config_json) VALUES (?, ?, ?, ?)",
                (run.run_id, run.name, run.status.value, json.dumps(run.config, sort_keys=True)),
            )

    def update_run_status(self, run_id: str, status: RunStatus) -> None:
        with self._connect() as conn:
            cur = conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status.value, run_id))
            if cur.rowcount != 1:
                raise KeyError(f"unknown run_id: {run_id}")

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id, name, status, config_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row["run_id"],
            name=row["name"],
            status=RunStatus(row["status"]),
            config=json.loads(row["config_json"]),
        )

    def create_task(self, task: TaskRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (task_id, run_id, task_key, status, payload_json, result_json, error, lease_owner, leased_until, attempt_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.run_id,
                    task.task_key,
                    task.status.value,
                    json.dumps(task.payload, sort_keys=True),
                    json.dumps(task.result, sort_keys=True) if task.result is not None else None,
                    task.error,
                    task.lease_owner,
                    task.leased_until.isoformat() if task.leased_until is not None else None,
                    task.attempt_count,
                ),
            )

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown task_id: {task_id}")
            ensure_task_transition(TaskStatus(row["status"]), status)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, result_json = ?, error = ?, lease_owner = NULL, leased_until = NULL WHERE task_id = ?",
                (
                    status.value,
                    json.dumps(result, sort_keys=True) if result is not None else None,
                    error,
                    task_id,
                ),
            )
            if cur.rowcount != 1:
                raise KeyError(f"unknown task_id: {task_id}")

    def list_tasks(self, run_id: str) -> list[TaskRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, run_id, task_key, status, payload_json, result_json, error, lease_owner, leased_until, attempt_count
                FROM tasks WHERE run_id = ? ORDER BY task_key
                """,
                (run_id,),
            ).fetchall()
        return [
            TaskRecord(
                task_id=row["task_id"],
                run_id=row["run_id"],
                task_key=row["task_key"],
                status=TaskStatus(row["status"]),
                payload=json.loads(row["payload_json"]),
                result=json.loads(row["result_json"]) if row["result_json"] else None,
                error=row["error"],
                lease_owner=row["lease_owner"],
                leased_until=datetime.fromisoformat(row["leased_until"]) if row["leased_until"] else None,
                attempt_count=int(row["attempt_count"] or 0),
            )
            for row in rows
        ]

    def lease_next_pending_task(self, *, run_id: str, worker_id: str, lease_seconds: int) -> TaskRecord | None:
        now = utc_now()
        lease = Lease.from_seconds(worker_id, lease_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT task_id, run_id, task_key, status, payload_json, result_json, error, lease_owner, leased_until, attempt_count
                FROM tasks
                WHERE run_id = ?
                  AND (
                    status = ?
                    OR (status = ? AND leased_until IS NOT NULL AND leased_until < ?)
                  )
                ORDER BY task_key
                LIMIT 1
                """,
                (run_id, TaskStatus.PENDING.value, TaskStatus.LEASED.value, now.isoformat()),
            ).fetchone()
            if row is None:
                return None

            cur = conn.execute(
                "UPDATE tasks SET status = ?, lease_owner = ?, leased_until = ? WHERE task_id = ?",
                (TaskStatus.LEASED.value, worker_id, lease.leased_until.isoformat(), row["task_id"]),
            )
            if cur.rowcount != 1:
                return None

        return TaskRecord(
            task_id=row["task_id"],
            run_id=row["run_id"],
            task_key=row["task_key"],
            status=TaskStatus.LEASED,
            payload=json.loads(row["payload_json"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
            lease_owner=worker_id,
            leased_until=lease.leased_until,
            attempt_count=int(row["attempt_count"] or 0),
        )

    def mark_task_running(self, task_id: str) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status = ? WHERE task_id = ? AND status = ?",
                (TaskStatus.RUNNING.value, task_id, TaskStatus.LEASED.value),
            )
            if cur.rowcount != 1:
                raise KeyError(f"task is not leaseable/running-ready: {task_id}")

    def increment_task_attempts(self, task_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE tasks SET attempt_count = attempt_count + 1 WHERE task_id = ?",
                (task_id,),
            )
            if cur.rowcount != 1:
                raise KeyError(f"unknown task_id: {task_id}")
            row = conn.execute("SELECT attempt_count FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown task_id: {task_id}")
            return int(row["attempt_count"])

    def requeue_expired_leases(self, *, run_id: str) -> int:
        now = utc_now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE tasks
                SET status = ?, lease_owner = NULL, leased_until = NULL
                WHERE run_id = ? AND status = ? AND leased_until IS NOT NULL AND leased_until < ?
                """,
                (TaskStatus.PENDING.value, run_id, TaskStatus.LEASED.value, now),
            )
            return int(cur.rowcount)

    def append_event(self, event: EventRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (event_id, run_id, task_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.run_id,
                    event.task_id,
                    event.kind,
                    json.dumps(event.payload, sort_keys=True),
                    event.created_at.isoformat(),
                ),
            )

    def create_artifact(self, artifact: ArtifactRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts (artifact_id, run_id, task_id, kind, path, checksum) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.task_id,
                    artifact.kind,
                    artifact.path,
                    artifact.checksum,
                ),
            )

    def list_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT artifact_id, run_id, task_id, kind, path, checksum FROM artifacts WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        return [
            ArtifactRecord(
                artifact_id=row["artifact_id"],
                run_id=row["run_id"],
                task_id=row["task_id"],
                kind=row["kind"],
                path=row["path"],
                checksum=row["checksum"],
            )
            for row in rows
        ]

    def list_events(self, run_id: str) -> list[EventRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id, run_id, task_id, kind, payload_json, created_at FROM events WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        return [
            EventRecord(
                event_id=row["event_id"],
                run_id=row["run_id"],
                task_id=row["task_id"],
                kind=row["kind"],
                payload=json.loads(row["payload_json"]),
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else _EPOCH,
            )
            for row in rows
        ]
