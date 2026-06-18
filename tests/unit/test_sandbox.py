from __future__ import annotations

import json
import pickle
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finetuneharness.executor.policy import FirejailSandbox, NoSandbox
from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.models import TaskRecord, TaskStatus
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
            "seed": 42,
            "dataset_hash": "sha256:test",
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


class TestFirejailSandboxJsonTransport:
    """Unit tests for the JSON output transport — no firejail binary needed.

    All tests patch both pickle.dumps (input serialization) and subprocess.run
    to avoid needing a picklable handler or a real firejail binary.
    """

    def _make_proc(self, stdout: bytes, returncode: int = 0) -> MagicMock:
        proc = MagicMock()
        proc.stdout = stdout
        proc.stderr = b""
        proc.returncode = returncode
        return proc

    def _make_task(self) -> TaskRecord:
        return TaskRecord(
            task_id="t1", run_id="r1", task_key="k1",
            status=TaskStatus.PENDING, payload={},
        )

    def _run_with_stdout(self, stdout: bytes) -> dict:
        """Helper: run sandbox.run() with a mocked subprocess returning stdout."""
        sandbox = FirejailSandbox.__new__(FirejailSandbox)
        sandbox._extra_args = ()
        with patch("finetuneharness.executor.policy.pickle") as mock_pickle:
            mock_pickle.dumps.return_value = b"payload"
            with patch("subprocess.run", return_value=self._make_proc(stdout)):
                return sandbox.run(lambda t: {}, self._make_task())

    def test_successful_result_decoded_from_json(self) -> None:
        """Child writes JSON {ok: true, result: {...}} — parent decodes without pickle."""
        expected = {"accuracy": 0.9, "task_key": "k1"}
        stdout = json.dumps({"ok": True, "result": expected}).encode()
        result = self._run_with_stdout(stdout)
        assert result == expected

    def test_handler_exception_raises_runtime_error_with_message(self) -> None:
        """Child writes JSON {ok: false, error: 'ValueError: bad'} — parent raises RuntimeError."""
        stdout = json.dumps({"ok": False, "error": "ValueError: bad input"}).encode()
        sandbox = FirejailSandbox.__new__(FirejailSandbox)
        sandbox._extra_args = ()
        with patch("finetuneharness.executor.policy.pickle") as mock_pickle:
            mock_pickle.dumps.return_value = b"payload"
            with patch("subprocess.run", return_value=self._make_proc(stdout)):
                with pytest.raises(RuntimeError, match="bad input"):
                    sandbox.run(lambda t: {}, self._make_task())

    def test_no_pickle_loads_on_stdout(self) -> None:
        """Verify subprocess output is decoded with json.loads, pickle.loads never called."""
        stdout = json.dumps({"ok": True, "result": {"x": 1}}).encode()
        sandbox = FirejailSandbox.__new__(FirejailSandbox)
        sandbox._extra_args = ()
        with patch("finetuneharness.executor.policy.pickle") as mock_pickle:
            mock_pickle.dumps.return_value = b"payload"
            mock_pickle.loads.side_effect = AssertionError(
                "pickle.loads called on subprocess output — security regression"
            )
            with patch("subprocess.run", return_value=self._make_proc(stdout)):
                sandbox.run(lambda t: {}, self._make_task())
        # Reaching here means pickle.loads was never called on proc.stdout

    def test_malformed_subprocess_output_raises_runtime_error(self) -> None:
        """Non-JSON stdout (e.g. firejail crash) raises RuntimeError, not pickle error."""
        sandbox = FirejailSandbox.__new__(FirejailSandbox)
        sandbox._extra_args = ()
        with patch("finetuneharness.executor.policy.pickle") as mock_pickle:
            mock_pickle.dumps.return_value = b"payload"
            with patch("subprocess.run", return_value=self._make_proc(b"not json", returncode=1)):
                with pytest.raises(RuntimeError, match="firejail subprocess failed"):
                    sandbox.run(lambda t: {}, self._make_task())


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
        # Exception crosses the process boundary as RuntimeError wrapping the
        # original class name and message (JSON transport — no pickle.loads).
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
