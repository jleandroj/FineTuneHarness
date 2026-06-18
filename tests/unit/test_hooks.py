"""Tests for the HookRegistry and hook integration in LocalWorker."""
from __future__ import annotations

from pathlib import Path

import pytest

from finetuneharness.executor.worker import LocalWorker
from finetuneharness.orchestrator.hooks import HookRegistry
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.sqlite import SQLiteStateStore


def _make_run(tmp_path: Path, name: str = "hook-test", tasks: list | None = None) -> tuple:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name=name,
        config={"project": {"name": "demo"}, "executor": {"kind": "local"}, "artifacts": {"root": "./artifacts"}, "seed": 42, "dataset_hash": "sha256:test"},
        tasks=tasks or [{"task_key": "cell-1", "kind": "train"}],
    )
    return store, run_id


def test_before_and_after_success_hooks_fire(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path)
    fired: list[tuple] = []
    hooks = HookRegistry()
    hooks.register("before_task", lambda task: fired.append(("before_task", task.task_key)))
    hooks.register("after_task_success", lambda task, result: fired.append(("after_task_success", task.task_key)))

    worker = LocalWorker(worker_id="w1", store=store, hooks=hooks)
    worker.run_once(run_id=run_id, handler=lambda task: {"ok": True})

    assert ("before_task", "cell-1") in fired
    assert ("after_task_success", "cell-1") in fired


def test_after_failure_hook_fires(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path)
    fired: list[tuple] = []
    hooks = HookRegistry()
    hooks.register("after_task_failure", lambda task, error: fired.append(("failure", task.task_key, type(error).__name__)))

    worker = LocalWorker(worker_id="w1", store=store, hooks=hooks)
    try:
        worker.run_once(run_id=run_id, handler=lambda task: (_ for _ in ()).throw(RuntimeError("boom")))
    except RuntimeError:
        pass

    assert ("failure", "cell-1", "RuntimeError") in fired


def test_hook_crash_does_not_propagate(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path)
    hooks = HookRegistry()
    hooks.register("before_task", lambda task: (_ for _ in ()).throw(RuntimeError("hook error")))

    worker = LocalWorker(worker_id="w1", store=store, hooks=hooks)
    task = worker.run_once(run_id=run_id, handler=lambda task: {"ok": True})
    assert task is not None


def test_register_unknown_hook_point_raises() -> None:
    hooks = HookRegistry()
    with pytest.raises(ValueError, match="unknown hook point"):
        hooks.register("nonexistent_point", lambda: None)


def test_hook_crash_is_logged(tmp_path: Path, caplog) -> None:
    import logging
    store, run_id = _make_run(tmp_path)
    hooks = HookRegistry()
    hooks.register("before_task", lambda task: (_ for _ in ()).throw(ValueError("hook kaboom")))

    worker = LocalWorker(worker_id="w1", store=store, hooks=hooks)
    with caplog.at_level(logging.WARNING):
        worker.run_once(run_id=run_id, handler=lambda task: {"ok": True})

    assert any("hook_error" in r.message or "hook kaboom" in r.message for r in caplog.records)


def test_on_run_status_changed_fires_on_completion(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path)
    statuses: list = []
    hooks = HookRegistry()
    hooks.register("on_run_status_changed", lambda run_id, status: statuses.append(status.value))

    worker = LocalWorker(worker_id="w1", store=store, hooks=hooks)
    worker.run_once(run_id=run_id, handler=lambda task: {"ok": True})

    assert "completed" in statuses
