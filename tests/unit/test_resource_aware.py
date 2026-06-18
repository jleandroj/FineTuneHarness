"""Resource-aware, process-isolated concurrent draining (drain_concurrent).

drain_concurrent runs each task in its own SPAWNED process so per-task seeding
stays reproducible and no CUDA context is inherited across a fork. Because spawn
pickles the handler by reference, handlers here are module-level functions and
observation is cross-process (task-payload directories + interval files), not
shared in-memory state.
"""
from __future__ import annotations

import os
import random
import time
import uuid
from pathlib import Path

import pytest

from finetuneharness.executor.resources import ConcurrencyConfig, is_oom_error
from finetuneharness.executor.worker import DegradedRunError, LocalWorker
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import TaskStatus
from finetuneharness.state.sqlite import SQLiteStateStore

_CONFIG = {
    "project": {"name": "resaware"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 42,
    "dataset_hash": "sha256:test",
}


# ── Module-level handlers (picklable for spawn) ──────────────────────────────

def _h_record(task):
    """Write one [start end] interval file per invocation. payload: obs_dir, sleep."""
    d = Path(task.payload["obs_dir"])
    start = time.time()
    time.sleep(float(task.payload.get("sleep", 0.0)))
    end = time.time()
    (d / f"{task.task_key}__{os.getpid()}__{uuid.uuid4().hex}").write_text(f"{start} {end}")
    return {"accuracy": 0.9, "f1": 0.88}


def _h_seed_draw(task):
    """Write the seeded global-RNG draw (run_once applies the seed before us)."""
    d = Path(task.payload["obs_dir"])
    (d / task.task_key).write_text(repr(random.random()))
    return {"accuracy": 0.9, "f1": 0.88}


def _h_oom_once(task):
    """OOM on the first attempt (counted via marker files), succeed afterwards."""
    d = Path(task.payload["obs_dir"])
    prior = len(list(d.glob(f"{task.task_key}__*")))
    (d / f"{task.task_key}__{os.getpid()}").write_text("x")
    if prior == 0:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    return {"accuracy": 0.9, "f1": 0.88}


def _h_always_oom(task):
    raise RuntimeError("CUDA out of memory")


def _h_ok(task):
    return {"accuracy": 0.9, "f1": 0.88}


def _h_hard_exit(task):
    """Die without reporting (simulates an OS OOM-kill) after the task is RUNNING."""
    os._exit(137)


def _h_record_device(task):
    """Record which GPU the child was pinned to (CUDA_VISIBLE_DEVICES)."""
    d = Path(task.payload["obs_dir"])
    (d / task.task_key).write_text(os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    time.sleep(float(task.payload.get("sleep", 0.0)))
    return {"accuracy": 0.9, "f1": 0.88}


def _hook_factory():
    """Importable factory: a HookRegistry whose after_task_success hook appends to
    the file named by env FTH_HOOK_LOG (set by the test, inherited by spawn children)."""
    from finetuneharness.orchestrator.hooks import HookRegistry

    reg = HookRegistry()
    path = os.environ["FTH_HOOK_LOG"]

    def _record(**kwargs):
        with open(path, "a") as fh:
            fh.write("after_task_success\n")

    reg.register("after_task_success", _record)
    return reg


class _DummySandbox:
    """A non-NoSandbox sandbox (passthrough) used to exercise the guard."""

    def run(self, handler, task):
        return handler(task)


class EarlyStoppingHook:  # name intentionally matches the cumulative-hook guard list
    """Stand-in whose method qualname trips the cumulative-hook guard."""

    def after_task_success(self, *args, **kwargs):
        pass


def _cumulative_hook_factory():
    """Factory producing a registry with a cumulative (EarlyStopping) hook."""
    from finetuneharness.orchestrator.hooks import HookRegistry

    reg = HookRegistry()
    reg.register("after_task_success", EarlyStoppingHook().after_task_success)
    return reg


# ── helpers ──────────────────────────────────────────────────────────────────

class _FakeMonitor:
    """Fixed free-memory monitor. Accepts a scalar (single GPU), a list (multi-GPU),
    or None (no GPU). Parent-side only — never crosses to children."""

    def __init__(self, free_mb) -> None:
        if free_mb is None:
            self._per = None
        elif isinstance(free_mb, (list, tuple)):
            self._per = [float(x) for x in free_mb]
        else:
            self._per = [float(free_mb)]

    def free_memory_per_device(self) -> list[float] | None:
        return list(self._per) if self._per is not None else None

    def free_gpu_memory_mb(self) -> float | None:
        return min(self._per) if self._per else None


def _make_run(tmp_path: Path, tasks: list[dict]) -> tuple[SQLiteStateStore, str]:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(name="r", config=_CONFIG, tasks=tasks)
    return store, run_id


def _resource_aware(**kw) -> ConcurrencyConfig:
    kw.setdefault("mode", "resource_aware")
    kw.setdefault("min_free_mb", 1000)
    kw.setdefault("settle_seconds", 0.0)
    return ConcurrencyConfig(**kw)


def _max_overlap(obs_dir: Path) -> int:
    """Max number of simultaneously-open [start, end] intervals across all files."""
    deltas: list[tuple[float, int]] = []
    for f in obs_dir.iterdir():
        s, e = f.read_text().split()
        deltas.append((float(s), +1))
        deltas.append((float(e), -1))
    # At a tie, apply ends (-1) before starts (+1) so merely-touching intervals
    # are not counted as overlapping.
    deltas.sort(key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, d in deltas:
        cur += d
        mx = max(mx, cur)
    return mx


# ── is_oom_error classification ──────────────────────────────────────────────

def test_is_oom_error_detects_cuda_messages() -> None:
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate ..."))
    assert is_oom_error(RuntimeError("handler raised in sandbox: OutOfMemoryError: ..."))

    class OutOfMemoryError(Exception):
        pass

    assert is_oom_error(OutOfMemoryError("boom"))
    assert not is_oom_error(ValueError("unrelated failure"))


# ── store guard ──────────────────────────────────────────────────────────────

def test_drain_concurrent_requires_persistent_store() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = runner.create_run(name="m", config=_CONFIG, tasks=[{"task_key": "t"}])
    worker = LocalWorker(worker_id="w", store=store)
    with pytest.raises(TypeError, match="persistent"):
        worker.drain_concurrent(
            run_id=run_id, handler=_h_always_oom,
            concurrency=_resource_aware(), monitor=_FakeMonitor(50000),
        )


# ── concurrency ──────────────────────────────────────────────────────────────

def test_drain_concurrent_uses_spawn_not_fork(tmp_path: Path, monkeypatch) -> None:
    """Anti-regression: concurrency must use 'spawn', never 'fork'.

    'fork' would inherit any CUDA context the parent created (e.g. by importing the
    handler module) and hand children a corrupted context. A fresh-interpreter
    'spawn' is the only GPU-safe choice; this pins it so nobody silently reverts.
    """
    import multiprocessing as _mp

    from finetuneharness.executor import worker as worker_mod

    seen: list[str | None] = []
    real_get_context = _mp.get_context

    def _spy(method=None):
        seen.append(method)
        return real_get_context(method)

    monkeypatch.setattr(worker_mod.mp, "get_context", _spy)

    obs = tmp_path / "obs"
    obs.mkdir()
    store, run_id = _make_run(tmp_path, [{"task_key": "t", "obs_dir": str(obs)}])
    worker = LocalWorker(worker_id="w", store=store)
    worker.drain_concurrent(
        run_id=run_id, handler=_h_record,
        concurrency=_resource_aware(), monitor=_FakeMonitor(50000),
    )

    assert "spawn" in seen, f"drain_concurrent must request 'spawn'; saw {seen}"
    assert "fork" not in seen, "drain_concurrent must NOT use 'fork' (CUDA-after-fork hazard)"


def test_drain_concurrent_runs_tasks_in_parallel(tmp_path: Path) -> None:
    """With plentiful memory, several spawned processes run at once; each task once."""
    obs = tmp_path / "obs"
    obs.mkdir()
    keys = [f"t{i}" for i in range(6)]
    store, run_id = _make_run(
        tmp_path, [{"task_key": k, "obs_dir": str(obs), "sleep": 0.4} for k in keys]
    )
    worker = LocalWorker(worker_id="w", store=store)

    succeeded = worker.drain_concurrent(
        run_id=run_id, handler=_h_record,
        concurrency=_resource_aware(max_concurrent=4),
        monitor=_FakeMonitor(50000),
    )

    assert succeeded == 6
    assert _max_overlap(obs) >= 2, "expected real concurrency, ran effectively sequentially"
    # Each task executed exactly once.
    per_task = sorted(f.name.split("__")[0] for f in obs.iterdir())
    assert per_task == sorted(keys)
    assert all(t.status is TaskStatus.SUCCEEDED for t in store.list_tasks(run_id))


def test_drain_concurrent_no_gpu_falls_back_to_sequential(tmp_path: Path) -> None:
    obs = tmp_path / "obs"
    obs.mkdir()
    store, run_id = _make_run(
        tmp_path, [{"task_key": k, "obs_dir": str(obs)} for k in ("a", "b", "c")]
    )
    worker = LocalWorker(worker_id="w", store=store)

    succeeded = worker.drain_concurrent(
        run_id=run_id, handler=_h_record,
        concurrency=_resource_aware(), monitor=_FakeMonitor(None),
    )
    assert succeeded == 3
    assert all(t.status is TaskStatus.SUCCEEDED for t in store.list_tasks(run_id))


def test_low_memory_serializes_admission(tmp_path: Path) -> None:
    """Free memory below the headroom forces one task at a time."""
    obs = tmp_path / "obs"
    obs.mkdir()
    keys = [f"t{i}" for i in range(4)]
    store, run_id = _make_run(
        tmp_path, [{"task_key": k, "obs_dir": str(obs), "sleep": 0.1} for k in keys]
    )
    worker = LocalWorker(worker_id="w", store=store)

    succeeded = worker.drain_concurrent(
        run_id=run_id, handler=_h_record,
        concurrency=_resource_aware(max_concurrent=4, min_free_mb=1000),
        monitor=_FakeMonitor(500),  # below headroom
    )
    assert succeeded == 4
    assert _max_overlap(obs) == 1, "low memory must force one-at-a-time admission"


# ── reproducibility (determinism under concurrency) ──────────────────────────

def test_seeding_is_deterministic_under_concurrency(tmp_path: Path) -> None:
    """Per-task RNG is isolated per process: concurrent runs are reproducible.

    Every task is seeded with the run seed before its handler runs, so each task's
    global-RNG draw is identical and stable across executions. If concurrency
    corrupted a shared RNG (the thread-based bug), these draws would vary.
    """
    def _execute_once(tag: str) -> set[str]:
        out = tmp_path / tag
        out.mkdir()
        store = SQLiteStateStore(tmp_path / f"{tag}.db")
        runner = FineTuneRunner(store)
        run_id = runner.create_run(
            name=tag, config=_CONFIG,
            tasks=[{"task_key": f"t{i}", "obs_dir": str(out)} for i in range(5)],
        )
        worker = LocalWorker(worker_id="w", store=store)
        worker.drain_concurrent(
            run_id=run_id, handler=_h_seed_draw,
            concurrency=_resource_aware(max_concurrent=4),
            monitor=_FakeMonitor(50000),
        )
        return {f.read_text() for f in out.iterdir()}

    run_a = _execute_once("a")
    run_b = _execute_once("b")

    # All tasks share one seed -> one identical draw, stable across both runs.
    assert len(run_a) == 1, f"per-task RNG not isolated: {run_a}"
    assert run_a == run_b, "not reproducible across runs"


# ── OOM handling ─────────────────────────────────────────────────────────────

def test_oom_task_is_requeued_then_succeeds(tmp_path: Path) -> None:
    """A task that OOMs once is requeued (not failed) and succeeds on retry.

    The OOM budget is persisted (events), so it survives across the separate
    processes each attempt runs in.
    """
    obs = tmp_path / "obs"
    obs.mkdir()
    store, run_id = _make_run(tmp_path, [{"task_key": "flaky", "obs_dir": str(obs)}])
    worker = LocalWorker(worker_id="w", store=store)

    succeeded = worker.drain_concurrent(
        run_id=run_id, handler=_h_oom_once,
        concurrency=_resource_aware(max_concurrent=2, max_oom_retries=3),
        monitor=_FakeMonitor(50000),
    )
    assert succeeded == 1
    assert store.list_tasks(run_id)[0].status is TaskStatus.SUCCEEDED
    assert "task_oom_requeued" in [e.kind for e in store.list_events(run_id)]


def test_os_killed_child_is_recovered_not_lost(tmp_path: Path) -> None:
    """A child that dies WITHOUT reporting (OS OOM-kill) must not silently strand its
    task in RUNNING. drain_concurrent reclaims it; once the budget is spent it FAILs,
    so the run terminates as degraded instead of hanging / returning false success.
    """
    store, run_id = _make_run(tmp_path, [{"task_key": "killed"}])
    worker = LocalWorker(worker_id="w", store=store)

    with pytest.raises(DegradedRunError):
        worker.drain_concurrent(
            run_id=run_id, handler=_h_hard_exit,
            concurrency=_resource_aware(max_concurrent=1, max_oom_retries=2),
            monitor=_FakeMonitor(50000),
        )
    task = store.list_tasks(run_id)[0]
    assert task.status is TaskStatus.FAILED, "OS-killed task must end FAILED, not stuck RUNNING"
    kinds = [e.kind for e in store.list_events(run_id)]
    assert kinds.count("lease_reclaimed") == 2
    assert "task_abandoned" in kinds


def test_multi_gpu_distributes_and_pins_children(tmp_path: Path) -> None:
    """With 2 GPUs, admission spreads tasks across devices and pins each child
    (CUDA_VISIBLE_DEVICES) to the device it was admitted to."""
    obs = tmp_path / "obs"
    obs.mkdir()
    keys = [f"t{i}" for i in range(4)]
    store, run_id = _make_run(
        tmp_path, [{"task_key": k, "obs_dir": str(obs), "sleep": 0.3} for k in keys]
    )
    worker = LocalWorker(worker_id="w", store=store)

    succeeded = worker.drain_concurrent(
        run_id=run_id, handler=_h_record_device,
        concurrency=_resource_aware(max_concurrent=4, min_free_mb=1000),
        monitor=_FakeMonitor([50000, 50000]),  # two GPUs
    )
    assert succeeded == 4
    devices = {f.read_text() for f in obs.iterdir()}
    assert devices == {"0", "1"}, f"tasks not distributed/pinned across both GPUs: {devices}"


# ── P1: hooks/sandbox parity & guard ─────────────────────────────────────────

def test_drain_concurrent_rejects_custom_hooks_without_factory(tmp_path: Path) -> None:
    from finetuneharness.orchestrator.hooks import HookRegistry

    reg = HookRegistry()
    reg.register("before_task", lambda **kw: None)
    store, run_id = _make_run(tmp_path, [{"task_key": "t"}])
    worker = LocalWorker(worker_id="w", store=store, hooks=reg)
    with pytest.raises(RuntimeError, match="custom hooks"):
        worker.drain_concurrent(
            run_id=run_id, handler=_h_always_oom,
            concurrency=_resource_aware(), monitor=_FakeMonitor(50000),
        )


def test_drain_concurrent_rejects_custom_sandbox_without_factory(tmp_path: Path) -> None:
    store, run_id = _make_run(tmp_path, [{"task_key": "t"}])
    worker = LocalWorker(worker_id="w", store=store, sandbox=_DummySandbox())
    with pytest.raises(RuntimeError, match="sandbox"):
        worker.drain_concurrent(
            run_id=run_id, handler=_h_always_oom,
            concurrency=_resource_aware(), monitor=_FakeMonitor(50000),
        )


def test_drain_concurrent_rejects_cumulative_hooks(tmp_path: Path) -> None:
    """Cumulative hooks (EarlyStopping/Metrics/Progress/Checkpoint) keep per-instance
    state that cannot aggregate across the separate task processes — reject, not mislead."""
    store, run_id = _make_run(tmp_path, [{"task_key": "t"}])
    worker = LocalWorker(
        worker_id="w", store=store, hooks_factory=f"{__name__}:_cumulative_hook_factory"
    )
    with pytest.raises(RuntimeError, match="cumulative hooks"):
        worker.drain_concurrent(
            run_id=run_id, handler=_h_ok,
            concurrency=_resource_aware(), monitor=_FakeMonitor(50000),
        )


def test_drain_concurrent_rejects_stop_fn(tmp_path: Path) -> None:
    """stop_fn relies on aggregate state no single task process sees — rejected."""
    store, run_id = _make_run(tmp_path, [{"task_key": "t"}])
    worker = LocalWorker(worker_id="w", store=store)
    with pytest.raises(RuntimeError, match="stop_fn"):
        worker.drain_concurrent(
            run_id=run_id, handler=_h_ok,
            concurrency=_resource_aware(), monitor=_FakeMonitor(50000),
            stop_fn=lambda: True,
        )


def test_hooks_fire_with_parity_across_modes(tmp_path: Path, monkeypatch) -> None:
    """A hook supplied via factory fires the same number of times in sequential and
    concurrent modes — proving children recreate hooks rather than dropping them."""
    spec = f"{__name__}:_hook_factory"

    def _run(mode_dir: str, concurrent: bool) -> int:
        log = tmp_path / f"{mode_dir}.log"
        monkeypatch.setenv("FTH_HOOK_LOG", str(log))
        store = SQLiteStateStore(tmp_path / f"{mode_dir}.db")
        runner = FineTuneRunner(store)
        run_id = runner.create_run(
            name=mode_dir, config=_CONFIG,
            tasks=[{"task_key": f"t{i}"} for i in range(3)],
        )
        worker = LocalWorker(worker_id="w", store=store, hooks_factory=spec)
        if concurrent:
            worker.drain_concurrent(
                run_id=run_id, handler=_h_ok,
                concurrency=_resource_aware(max_concurrent=2),
                monitor=_FakeMonitor(50000),
            )
        else:
            worker.drain(run_id=run_id, handler=_h_ok)
        return log.read_text().count("after_task_success") if log.exists() else 0

    seq = _run("seq", concurrent=False)
    con = _run("con", concurrent=True)
    assert seq == 3, f"sequential hook fired {seq} times, expected 3"
    assert con == 3, f"concurrent hook fired {con} times, expected 3"
    assert seq == con, "hook firing must be at parity across modes"


def test_oom_task_fails_after_exhausting_retries(tmp_path: Path) -> None:
    """A task that always OOMs is FAILED after max_oom_retries, surfacing as degraded."""
    store, run_id = _make_run(tmp_path, [{"task_key": "doomed"}])
    worker = LocalWorker(worker_id="w", store=store)

    with pytest.raises(DegradedRunError):
        worker.drain_concurrent(
            run_id=run_id, handler=_h_always_oom,
            concurrency=_resource_aware(max_concurrent=2, max_oom_retries=2),
            monitor=_FakeMonitor(50000),
        )
    assert store.list_tasks(run_id)[0].status is TaskStatus.FAILED
    requeues = [e for e in store.list_events(run_id) if e.kind == "task_oom_requeued"]
    assert len(requeues) == 2, "should requeue exactly max_oom_retries times before failing"
