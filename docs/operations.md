# Operations runbook

How `drain_concurrent` admits, times out, backs off, and recovers — and how an
operator drives recovery. Companion to [architecture.md](architecture.md), which
covers the data model.

## Admission control

A run drains several tasks at once under a resource-aware admission policy. There
is no fixed worker count; concurrency is governed by a live signal plus a hard cap.

`ConcurrencyConfig` knobs:

| knob | meaning |
|---|---|
| `mode` | `sequential` (one at a time) or `resource_aware` |
| `admission` | `memory` (default) or `utilization` — which live signal gates |
| `min_free_mb` | [memory] admit while free GPU memory ≥ this |
| `max_util_pct` | [utilization] admit while GPU utilization ≤ this |
| `max_concurrent` | hard ceiling on in-flight tasks, regardless of the signal |
| `settle_seconds` | pause after each admission before re-reading the signal |
| `max_oom_retries` | OOM requeues allowed per task before it is FAILED |

The first task on an idle GPU is always admitted (no deadlock). Each task runs in
its own **spawned** process, so per-task RNG seeding stays isolated and no CUDA
context is inherited across a fork.

### Which `admission` to use

- **`memory`** (default) — normal discrete GPUs where NVML reports free memory.
  Admits while `free ≥ min_free_mb`. Memory is the bottleneck.
- **`utilization`** — unified-memory GPUs (Grace-Blackwell **GB10**, GH200) where
  NVML does **not** report free memory (`memory.free` = `[N/A]`,
  `nvmlDeviceGetMemoryInfo` → `NVMLError_NotSupported`) but **does** report
  utilization. Compute is the bottleneck; admits while `utilization ≤ max_util_pct`.
  Use `UtilizationMonitor` (or `NvmlMonitor`, which also implements the signal).

### Calibration is not optional

The utilization signal is **instantaneous and noisy** — it reads low during a
task's data-load/eval phases, so a high `max_concurrent` lets the run *ramp up*
during those windows and then all tasks hit compute at once. On a single GB10:

| config | outcome |
|---|---|
| `max_concurrent=50` (faked capacity) | ~21% timeouts |
| `max_concurrent=32, max_util_pct=95` | ~53% timeouts (ramped to 32 in low-util windows) |
| `max_concurrent=4, max_util_pct=70, settle=5s` | 0 timeouts on the hardest LSTM set |

**Lesson: on a single GPU the low hard cap is the real throttle.** Set
`max_concurrent` to what the GPU can actually run concurrently (small: ~2–4 for
heavy models), and use the utilization gate to pack light tasks. A future EMA on
the utilization signal would let the cap be raised safely (see ROADMAP).

## Timeout is preemptive (frees the GPU)

Each task carries `timeout_seconds`. The handler runs in a **daemon** thread; when
it overruns, the worker stops waiting *and* — because the thread is daemon — the
task's spawned process exits promptly, tearing down its CUDA context and freeing
the GPU. The task is marked `TIMED_OUT` (terminal, not retried). The parent reaps
each finished child with a bounded `join → terminate → kill` so it never blocks on
a process wedged in CUDA teardown. A timeout also lowers the concurrency ceiling
(backoff), so a contended run converges toward sequential instead of timing out
task after task.

> In-process `drain` (sequential, `NoSandbox`) cannot free the GPU on timeout — only
> the per-process `drain_concurrent` path does. For hard preemption of in-process
> handlers, run under `FirejailSandbox` (subprocess) with a subprocess timeout.

## OOM handling

A GPU OOM is treated as transient contention, not a deterministic failure: the task
is requeued (without burning a normal retry) and the ceiling is lowered, converging
toward sequential under memory pressure. After `max_oom_retries` requeues the task
is FAILED so it can't loop forever. System-RAM OOM is deliberately *not* matched
(GPU backoff would not help it).

## Recovery

| situation | what happens | operator action |
|---|---|---|
| Task fails (transient) | requeued up to `max_attempts` (exponential backoff + jitter), else FAILED | none |
| Task exceeds timeout | preemptively killed, marked TIMED_OUT (terminal) | re-run as a new run over the subset if desired |
| Lease expires (worker slow/gone) | next `lease_next_pending_task` reclaims it (LEASED→PENDING) + `lease_expired` event | none |
| **Graceful stop** (`SIGINT`/`SIGTERM` to the drain process) | the `finally` reaps children (terminate→kill) and requeues their in-flight tasks | re-run to resume |
| **Hard crash** (`SIGKILL`, power loss) | tasks stranded `RUNNING`/`LEASED`; no automatic path reclaims a RUNNING task | **`recover-run`** (below), then re-run |

### `recover-run`

```bash
finetuneharness recover-run --run-id <RUN_ID> --state-db <DB>
```

Requeues every `RUNNING`/`LEASED` task of the run to `PENDING` (emits a
`task_recovered` event each) and refreshes the run status. **Only run it when no
worker is active for that run** — it does not check liveness, so requeuing a task a
live worker is still running would double-execute it.

## Task status semantics

`PENDING → LEASED → RUNNING → {SUCCEEDED | FAILED | TIMED_OUT | DEGENERATE}`
(`RUNNING → PENDING` on retry). `DEGENERATE` = the handler returned but the result
failed validation (NaN/Inf/out-of-range metric, `trainable_params==0` for an
adapter method, all-zero metrics, …); it is terminal and never retried because the
same config reproduces it. The run reaches `COMPLETED` only when **all** tasks are
SUCCEEDED; any non-success leaves it `PARTIAL_FAILED` or `FAILED`.

## Auditing a run

Every state transition emits an event (`drain_started` records the admission mode
and caps; `task_*`, `lease_*`, `task_recovered`, `concurrency_backoff_*`). Reconstruct
a run with `list-tasks`, `list-events` (via the store), `export-manifest`, and
`validate-reproducibility`.
