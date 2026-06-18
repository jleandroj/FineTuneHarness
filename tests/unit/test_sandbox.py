from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from finetuneharness.executor.policy import FirejailSandbox, NoSandbox
from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore


# Module-level functions required by FirejailSandbox (lambdas are not picklable)

def _ok_handler(task):
    return {"ok": True, "task_key": task.task_key}


def _failing_handler(task):
    raise RuntimeError("boom from handler")


def _make_run(store, tmp_path, *, tasks=None):
    runner = FineTuneRunner(store)
    return runner.create_run(
        name="sandbox-test",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": str(tmp_path / "artifacts")},
        },
        tasks=tasks or [{"task_key": "cell-1", "kind": "train"}],
    )


class TestNoSandbox:
    def test_passthrough_succeeds(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        run_id = _make_run(store, tmp_path)
        worker = LocalWorker(worker_id="w1", store=store, sandbox=NoSandbox())
        task = worker.run_once(run_id=run_id, handler=_ok_handler)
        assert task is not None
        assert store.list_tasks(run_id)[0].status == TaskStatus.SUCCEEDED

    def test_passthrough_propagates_exception(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        run_id = _make_run(store, tmp_path)
        worker = LocalWorker(worker_id="w1", store=store, sandbox=NoSandbox())
        with pytest.raises(RuntimeError, match="boom"):
            worker.run_once(run_id=run_id, handler=_failing_handler)
        assert store.list_tasks(run_id)[0].status == TaskStatus.FAILED

    def test_is_default_when_sandbox_omitted(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        run_id = _make_run(store, tmp_path)
        worker = LocalWorker(worker_id="w1", store=store)
        task = worker.run_once(run_id=run_id, handler=_ok_handler)
        assert task is not None


class TestFirejailSandbox:
    def test_raises_when_firejail_missing(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="firejail not found"):
                FirejailSandbox()

    @pytest.mark.skipif(
        shutil.which("firejail") is None,
        reason="firejail not installed",
    )
    def test_runs_handler_in_isolation(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        run_id = _make_run(store, tmp_path)
        worker = LocalWorker(worker_id="w1", store=store, sandbox=FirejailSandbox())
        task = worker.run_once(run_id=run_id, handler=_ok_handler)
        assert task is not None
        result = store.list_tasks(run_id)[0]
        assert result.status == TaskStatus.SUCCEEDED
        assert result.result["ok"] is True

    @pytest.mark.skipif(
        shutil.which("firejail") is None,
        reason="firejail not installed",
    )
    def test_propagates_handler_exception(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        run_id = _make_run(store, tmp_path)
        worker = LocalWorker(worker_id="w1", store=store, sandbox=FirejailSandbox())
        with pytest.raises(RuntimeError, match="boom from handler"):
            worker.run_once(run_id=run_id, handler=_failing_handler)
        assert store.list_tasks(run_id)[0].status == TaskStatus.FAILED

    @pytest.mark.skipif(
        shutil.which("firejail") is None,
        reason="firejail not installed",
    )
    def test_extra_args_forwarded(self, tmp_path: Path) -> None:
        store = SQLiteStateStore(tmp_path / "state.db")
        run_id = _make_run(store, tmp_path)
        # --caps.drop=all is a valid firejail flag that should not break execution
        worker = LocalWorker(
            worker_id="w1",
            store=store,
            sandbox=FirejailSandbox(extra_args=("--caps.drop=all",)),
        )
        task = worker.run_once(run_id=run_id, handler=_ok_handler)
        assert task is not None
