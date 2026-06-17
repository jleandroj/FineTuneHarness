from pathlib import Path

from finetuneharness.state.models import EventRecord, RunRecord, RunStatus, TaskRecord, TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore


def test_sqlite_store_roundtrip(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")

    run = RunRecord(
        run_id="run-1",
        name="demo",
        status=RunStatus.CREATED,
        config={"project": {"name": "demo"}, "executor": {"kind": "local"}, "artifacts": {"root": "./artifacts"}},
    )
    store.create_run(run)
    store.update_run_status("run-1", RunStatus.VALIDATED)

    task = TaskRecord(
        task_id="task-1",
        run_id="run-1",
        task_key="cell-1",
        status=TaskStatus.PENDING,
        payload={"kind": "train"},
    )
    store.create_task(task)
    store.lease_next_pending_task(run_id="run-1", worker_id="worker-x", lease_seconds=60)
    store.mark_task_running("task-1")
    store.update_task_status("task-1", TaskStatus.SUCCEEDED, result={"accuracy": 0.9})
    store.append_event(EventRecord(event_id="evt-1", run_id="run-1", task_id="task-1", kind="task_succeeded"))

    loaded = store.get_run("run-1")
    assert loaded is not None
    assert loaded.status == RunStatus.VALIDATED

    tasks = store.list_tasks("run-1")
    assert len(tasks) == 1
    assert tasks[0].status == TaskStatus.SUCCEEDED
    assert tasks[0].result == {"accuracy": 0.9}


def test_sqlite_store_lease_next_pending_task(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    run = RunRecord(
        run_id="run-2",
        name="lease-demo",
        status=RunStatus.CREATED,
        config={"project": {"name": "demo"}, "executor": {"kind": "local"}, "artifacts": {"root": "./artifacts"}},
    )
    store.create_run(run)
    store.create_task(
        TaskRecord(
            task_id="task-a",
            run_id="run-2",
            task_key="a",
            status=TaskStatus.PENDING,
            payload={"kind": "train"},
        )
    )
    leased = store.lease_next_pending_task(run_id="run-2", worker_id="worker-1", lease_seconds=60)
    assert leased is not None
    assert leased.status == TaskStatus.LEASED
    assert leased.lease_owner == "worker-1"
