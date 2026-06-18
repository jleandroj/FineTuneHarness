from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from finetuneharness.orchestrator.lifecycle import ensure_task_transition
from finetuneharness.state.leases import Lease, utc_now
from finetuneharness.state.models import ArtifactRecord, EventRecord, RunRecord, RunStatus, TaskRecord, TaskStatus
from finetuneharness.state.store import StateStore

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def _row_to_run(row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        name=row["name"],
        status=RunStatus(row["status"]),
        config=json.loads(row["config_json"]),
        created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(timezone.utc),
        env_snapshot=json.loads(row["env_snapshot_json"]) if row["env_snapshot_json"] else {},
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        seed=int(row["seed"]) if row["seed"] is not None else None,
        dataset_hashes=json.loads(row["dataset_hashes_json"]) if row["dataset_hashes_json"] else {},
        config_hash=row["config_hash"],
    )


class SQLiteStateStore(StateStore):
    """SQLite-backed persistent state store for local serious use."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        # foreign_keys is a per-connection PRAGMA (defaults OFF). Without this,
        # the ON DELETE CASCADE constraints declared in the schema are NOT
        # enforced on any operational connection.
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT,
                    env_snapshot_json TEXT,
                    finished_at TEXT,
                    seed INTEGER,
                    dataset_hashes_json TEXT,
                    config_hash TEXT
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
            run_cols = {row['name'] for row in conn.execute("PRAGMA table_info(runs)")}
            if 'created_at' not in run_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN created_at TEXT")
            if 'env_snapshot_json' not in run_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN env_snapshot_json TEXT")
            if 'finished_at' not in run_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN finished_at TEXT")
            if 'seed' not in run_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN seed INTEGER")
            if 'dataset_hashes_json' not in run_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN dataset_hashes_json TEXT")
            if 'config_hash' not in run_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN config_hash TEXT")

    def create_run(self, run: RunRecord) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO runs"
                " (run_id, name, status, config_json, created_at, env_snapshot_json, finished_at,"
                "  seed, dataset_hashes_json, config_hash)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.name,
                    run.status.value,
                    json.dumps(run.config, sort_keys=True),
                    run.created_at.isoformat(),
                    json.dumps(run.env_snapshot, sort_keys=True) if run.env_snapshot else None,
                    run.finished_at.isoformat() if run.finished_at is not None else None,
                    run.seed,
                    json.dumps(run.dataset_hashes, sort_keys=True) if run.dataset_hashes else None,
                    run.config_hash,
                ),
            )

    def update_run_finished_at(self, run_id: str, finished_at: datetime) -> None:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE runs SET finished_at = ? WHERE run_id = ?",
                (finished_at.isoformat(), run_id),
            )
            if cur.rowcount != 1:
                raise KeyError(f"unknown run_id: {run_id}")

    def update_run_status(self, run_id: str, status: RunStatus) -> None:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status.value, run_id))
            if cur.rowcount != 1:
                raise KeyError(f"unknown run_id: {run_id}")

    def get_run(self, run_id: str) -> RunRecord | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT run_id, name, status, config_json, created_at, env_snapshot_json, finished_at,"
                "       seed, dataset_hashes_json, config_hash"
                " FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_run(row)

    def list_runs(self) -> list[RunRecord]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT run_id, name, status, config_json, created_at, env_snapshot_json, finished_at,"
                "       seed, dataset_hashes_json, config_hash"
                " FROM runs ORDER BY rowid"
            ).fetchall()
        return [_row_to_run(row) for row in rows]

    def create_task(self, task: TaskRecord) -> None:
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE tasks SET status = ? WHERE task_id = ? AND status = ?",
                (TaskStatus.RUNNING.value, task_id, TaskStatus.LEASED.value),
            )
            if cur.rowcount != 1:
                raise KeyError(f"task is not leaseable/running-ready: {task_id}")

    def increment_task_attempts(self, task_id: str) -> int:
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            expired = conn.execute(
                """
                SELECT task_id, lease_owner FROM tasks
                WHERE run_id = ? AND status = ? AND leased_until IS NOT NULL AND leased_until < ?
                """,
                (run_id, TaskStatus.LEASED.value, now),
            ).fetchall()
            if not expired:
                return 0
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, lease_owner = NULL, leased_until = NULL
                WHERE run_id = ? AND status = ? AND leased_until IS NOT NULL AND leased_until < ?
                """,
                (TaskStatus.PENDING.value, run_id, TaskStatus.LEASED.value, now),
            )
            # Emit an audit event per reclaimed lease so silent recovery is visible.
            for row in expired:
                conn.execute(
                    "INSERT INTO events (event_id, run_id, task_id, kind, payload_json, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        uuid.uuid4().hex, run_id, row["task_id"], "lease_expired",
                        json.dumps({"previous_owner": row["lease_owner"]}, sort_keys=True),
                        utc_now().isoformat(),
                    ),
                )
            return len(expired)

    def append_event(self, event: EventRecord) -> None:
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
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
