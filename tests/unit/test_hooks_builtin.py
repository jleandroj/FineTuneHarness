"""Tests for built-in fine-tuning hooks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from finetuneharness.registry.hooks import (
    GPUMemoryHook,
    CheckpointHook,
    MetricsHook,
    EarlyStoppingHook,
    CleanupHook,
    ProgressHook,
    register_default_hooks,
)
from finetuneharness.orchestrator.hooks import HookRegistry
from finetuneharness.state.models import RunStatus, TaskRecord, TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.executor.worker import LocalWorker

_BASE_CONFIG = {
    "project": {"name": "hook-test"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:test",
}


class TestGPUMemoryHook:
    def test_hooks_register_without_torch(self):
        """Hook should not crash when torch not available or CUDA not available."""
        hook = GPUMemoryHook()
        registry = HookRegistry()
        registry.register("before_task", hook.before_task)
        registry.register("after_task_success", hook.after_task_success)
        registry.register("after_task_failure", hook.after_task_failure)

        task = TaskRecord(
            task_id="test-1",
            run_id="run-1",
            task_key="test",
            status=TaskStatus.PENDING,
            payload={},
        )

        # Should not raise
        registry.fire("before_task", task=task)
        registry.fire("after_task_success", task=task, result={})
        registry.fire("after_task_failure", task=task, error=RuntimeError("test"))

    def test_after_task_failure_oom(self):
        hook = GPUMemoryHook()
        task = TaskRecord(
            task_id="test-1",
            run_id="run-1",
            task_key="test",
            status=TaskStatus.PENDING,
            payload={},
        )
        # Should not raise even with OOM error
        hook.after_task_failure(task, RuntimeError("CUDA out of memory"))


class TestCheckpointHook:
    def test_creates_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            hook = CheckpointHook(checkpoint_dir=str(ckpt_dir))
            assert ckpt_dir.exists()

    def test_before_task_adds_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            hook = CheckpointHook(checkpoint_dir=str(ckpt_dir))

            task = TaskRecord(
                task_id="test-1",
                run_id="run-1",
                task_key="test",
                status=TaskStatus.PENDING,
                payload={},
            )
            hook.before_task(task)
            assert task.payload.get("checkpoint_dir") == str(ckpt_dir)


class TestMetricsHook:
    def test_collects_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.jsonl"
            hook = MetricsHook(output_file=str(metrics_file))

            task = TaskRecord(
                task_id="test-1",
                run_id="run-1",
                task_key="k3-lora",
                status=TaskStatus.PENDING,
                payload={"k": 3, "technique": "lora"},
            )

            hook.after_task_success(task, {"accuracy": 0.9, "f1": 0.89, "epochs": 10})

            assert metrics_file.exists()
            content = metrics_file.read_text().strip()
            assert "k3-lora" in content
            assert "0.9" in content

    def test_get_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.jsonl"
            hook = MetricsHook(output_file=str(metrics_file))

            for i in range(5):
                task = TaskRecord(
                    task_id=f"test-{i}",
                    run_id="run-1",
                    task_key=f"task-{i}",
                    status=TaskStatus.PENDING,
                    payload={},
                )
                hook.after_task_success(task, {"accuracy": 0.8 + i * 0.02, "f1": 0.75 + i * 0.02})

            summary = hook.get_summary()
            assert summary["count"] == 5
            assert "accuracy" in summary
            assert "mean" in summary["accuracy"]
            assert "min" in summary["accuracy"]
            assert "max" in summary["accuracy"]
            assert "stdev" in summary["accuracy"]


class TestEarlyStoppingHook:
    def test_no_improvement_triggers_stop(self):
        hook = EarlyStoppingHook(metric="accuracy", patience=3, min_delta=0.01)

        task = TaskRecord(task_id="1", run_id="run-1", task_key="t1", status=TaskStatus.PENDING, payload={})

        # First task sets baseline
        hook.after_task_success(task, {"accuracy": 0.90})
        assert not hook.should_stop()

        # Next 2 tasks don't improve enough (counter = 1, 2)
        for acc in [0.901, 0.902]:
            hook.after_task_success(task, {"accuracy": acc})
            assert not hook.should_stop()

        # 3rd non-improvement should trigger (counter = 3 >= patience)
        hook.after_task_success(task, {"accuracy": 0.903})
        assert hook.should_stop()

    def test_improvement_resets_counter(self):
        hook = EarlyStoppingHook(metric="accuracy", patience=3, min_delta=0.01)

        task = TaskRecord(task_id="1", run_id="run-1", task_key="t1", status=TaskStatus.PENDING, payload={})

        hook.after_task_success(task, {"accuracy": 0.90})
        hook.after_task_success(task, {"accuracy": 0.901})  # no improvement
        hook.after_task_success(task, {"accuracy": 0.915})  # improvement!
        hook.after_task_success(task, {"accuracy": 0.916})  # no improvement

        assert not hook.should_stop()  # counter reset after improvement


class TestCleanupHook:
    def test_cleans_up_on_success(self):
        hook = CleanupHook()
        task = TaskRecord(task_id="1", run_id="run-1", task_key="t1", status=TaskStatus.PENDING, payload={})
        hook.after_task_success(task, {})  # Should not raise

    def test_cleans_up_on_failure(self):
        hook = CleanupHook()
        task = TaskRecord(task_id="1", run_id="run-1", task_key="t1", status=TaskStatus.PENDING, payload={})
        hook.after_task_failure(task, RuntimeError("test"))  # Should not raise

    def test_cleans_up_on_timeout(self):
        hook = CleanupHook()
        task = TaskRecord(task_id="1", run_id="run-1", task_key="t1", status=TaskStatus.PENDING, payload={})
        hook.after_task_timeout(task)  # Should not raise


class TestProgressHook:
    def test_logs_progress(self):
        hook = ProgressHook(log_every=2)

        task = TaskRecord(task_id="1", run_id="run-1", task_key="t1", status=TaskStatus.PENDING, payload={})

        hook.after_task_success(task, {})  # 1st - not logged (log_every=2)
        hook.after_task_success(task, {})  # 2nd - logged

    def test_on_run_status_changed_completed(self):
        hook = ProgressHook()
        hook.on_run_status_changed("run-1", RunStatus.COMPLETED)  # Should not raise


class TestRegisterDefaultHooks:
    def test_registers_all_hooks(self):
        registry = HookRegistry()
        hooks = register_default_hooks(
            registry,
            gpu_monitor=True,
            checkpoint=True,
            metrics=True,
            early_stopping=True,
            cleanup=True,
            progress=True,
        )

        assert "gpu" in hooks
        assert "checkpoint" in hooks
        assert "metrics" in hooks
        assert "early_stopping" in hooks
        assert "cleanup" in hooks
        assert "progress" in hooks

        # Verify hooks are registered
        # Note: we can't easily test the internal _hooks dict, but we can fire them
        task = TaskRecord(task_id="1", run_id="run-1", task_key="t1", status=TaskStatus.PENDING, payload={})
        registry.fire("before_task", task=task)
        registry.fire("after_task_success", task=task, result={"accuracy": 0.9})
        registry.fire("after_task_failure", task=task, error=RuntimeError("test"))
        registry.fire("after_task_timeout", task=task)
        registry.fire("on_run_status_changed", run_id="run-1", status=RunStatus.COMPLETED)

    def test_can_disable_hooks(self):
        registry = HookRegistry()
        hooks = register_default_hooks(
            registry,
            gpu_monitor=False,
            checkpoint=False,
            metrics=False,
            early_stopping=False,
            cleanup=False,
            progress=False,
        )
        assert hooks == {}


# ── Regression: hook mutations persisted to store (Bug: GPUMemoryHook) ────────

class TestHookMutationPersistence:
    """after_task_success hook mutations to the result dict must be persisted.

    Before the fix, hooks fired AFTER update_task_status, so SQLiteStateStore
    had already JSON-serialized the result and mutations were silently lost.
    InMemoryStateStore stored by reference so the bug was invisible in tests.
    """

    def test_hook_enrichment_persisted_to_sqlite(self, tmp_path):
        """Hook-added fields must appear in the result stored in SQLite."""
        store = SQLiteStateStore(tmp_path / "state.db")
        runner = FineTuneRunner(store)
        registry = HookRegistry()

        def gpu_sim_hook(task: TaskRecord, result: dict) -> None:
            result["gpu_peak_mb"] = 2048.0  # simulates GPUMemoryHook

        registry.register("after_task_success", gpu_sim_hook)

        run_id = runner.create_run(name="r", config=_BASE_CONFIG, tasks=[{"task_key": "a"}])
        worker = LocalWorker(worker_id="w", store=store, runner=runner, hooks=registry)
        worker.run_once(run_id=run_id, handler=lambda t: {"accuracy": 0.9})

        # Reload from SQLite — the hook-added field must survive the round-trip
        task = store.list_tasks(run_id)[0]
        assert task.result is not None, "result should be stored"
        assert task.result.get("gpu_peak_mb") == 2048.0, (
            "hook-added field was not persisted — after_task_success fired after store write"
        )

    def test_hook_enrichment_persisted_to_memory_store(self):
        """Baseline: in-memory store should also see hook mutations."""
        from finetuneharness.state.memory_store import InMemoryStateStore

        store = InMemoryStateStore()
        runner = FineTuneRunner(store)
        registry = HookRegistry()

        def hook(task: TaskRecord, result: dict) -> None:
            result["hook_flag"] = True

        registry.register("after_task_success", hook)

        run_id = runner.create_run(name="r", config=_BASE_CONFIG, tasks=[{"task_key": "a"}])
        worker = LocalWorker(worker_id="w", store=store, runner=runner, hooks=registry)
        worker.run_once(run_id=run_id, handler=lambda t: {})

        task = store.list_tasks(run_id)[0]
        assert task.result is not None
        assert task.result.get("hook_flag") is True

    def test_crashing_hook_does_not_lose_result(self, tmp_path):
        """A hook that raises must not prevent the result from being persisted."""
        store = SQLiteStateStore(tmp_path / "state.db")
        runner = FineTuneRunner(store)
        registry = HookRegistry()

        def broken_hook(task: TaskRecord, result: dict) -> None:
            raise RuntimeError("hook exploded")

        registry.register("after_task_success", broken_hook)

        run_id = runner.create_run(name="r", config=_BASE_CONFIG, tasks=[{"task_key": "a"}])
        worker = LocalWorker(worker_id="w", store=store, runner=runner, hooks=registry)
        worker.run_once(run_id=run_id, handler=lambda t: {"accuracy": 0.9})

        task = store.list_tasks(run_id)[0]
        assert task.result is not None
        assert task.result.get("accuracy") == 0.9


# ── Regression: CleanupHook no longer allocates dummy model ──────────────────

class TestCleanupHookNoDummyModel:
    """CleanupHook must not allocate any torch model during cleanup.

    Before the fix, _cleanup created nn.Linear(1,1) to 'clear gradients' —
    but the dummy model had no gradients (never backpropped), so it was a no-op
    that additionally allocated tensors on every cleanup call.
    """

    def test_cleanup_hook_has_no_clear_torch_grad_parameter(self):
        import inspect
        sig = inspect.signature(CleanupHook.__init__)
        assert "clear_torch_grad" not in sig.parameters, (
            "clear_torch_grad was removed — it silently did nothing (dummy model, no backprop)"
        )

    def test_cleanup_does_not_allocate_nn_linear(self):
        """_cleanup must not instantiate any torch.nn.Module."""
        try:
            import torch.nn as nn
            from unittest.mock import patch

            hook = CleanupHook()
            task = TaskRecord(task_id="1", run_id="r", task_key="t", status=TaskStatus.PENDING, payload={})

            with patch.object(nn, "Linear", side_effect=AssertionError("should not create nn.Linear")) as mock_linear:
                hook.after_task_success(task, {})
                hook.after_task_failure(task, RuntimeError("err"))
                hook.after_task_timeout(task)
                # If nn.Linear was called, the patch would have raised
                assert mock_linear.call_count == 0
        except ImportError:
            pytest.skip("torch not installed")