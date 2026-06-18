# FineTuneHarness

Execution harness for ML fine-tuning experiments.
Handles state, retries, timeout, hooks, and artifact tracking for ablation grids.

## Environment setup

```bash
pip install -e .
```

## Key commands

```bash
# Run tests
python -m pytest tests/ -v

# CLI — create a run
finetuneharness create-run --name "my-experiment" --config configs/example.json --tasks configs/tasks.json

# CLI — execute pending tasks with your handler (module:function)
finetuneharness run --run-id <id> --handler mypkg.handlers:train --max-workers 4
```

## Project layout

```
src/finetuneharness/
  state/         — models, SQLite store, InMemory store, leases
  orchestrator/  — runner, scheduler, lifecycle, hooks
  executor/      — worker (preemptive timeout), policy
  artifacts/     — filesystem store with SHA-256 checksums
  observability/ — structured JSON logging
  validation/    — run config validation
  registry/      — (in progress) skill registry
  evaluation/    — (in progress) metrics and comparison
skills/          — SKILL.md context files for fine-tuning techniques
tests/unit/      — 42 tests
```

## Skills available

| File | Covers |
|------|--------|
| `skills/peft-fine-tuning.md` | LoRA, QLoRA, AdaLoRA, IA³, Prefix, Prompt, Adapter, BitFit |
| `skills/knowledge-distillation.md` | Teacher-student distillation, MiniLLM, soft targets |
| `skills/model-merging.md` | SLERP, TIES, DARE, Task Arithmetic |
| `skills/axolotl.md` | Full fine-tuning orchestration |
| `skills/lm-evaluation-harness.md` | Model evaluation, benchmarks |
| `skills/weights-and-biases.md` | Experiment tracking |
| `skills/unsloth.md` | Fast fine-tuning, 2x speed, less memory |
| `skills/llama-factory.md` | Full fine-tuning framework, curriculum learning |
| `skills/trl-fine-tuning.md` | RLHF, DPO, PPO, SFTTrainer |
| `skills/grpo-rl-training.md` | RL with group rewards |
| `skills/deepspeed.md` | Multi-GPU distributed training |
| `skills/model-pruning.md` | Structured and unstructured pruning |
| `skills/mlflow.md` | Experiment tracking and model registry |
| `skills/tensorboard.md` | Training visualization and metrics |
| `skills/machine-learning-engineer.md` | Production serving, ONNX/TensorRT conversion, monitoring |
| `skills/scientific-literature-researcher.md` | Literature search, published baselines, bioinformatics papers |
| `skills/nlp-engineer.md` | BERT fine-tuning, tokenization strategies, evaluation pipelines |

## Architecture

- Tasks have explicit state: PENDING → LEASED → RUNNING → SUCCEEDED/FAILED/TIMED_OUT/DEGENERATE
- State machine is enforced — invalid transitions raise ValueError
- DEGENERATE is terminal: the handler returned normally but the result validator
  (evaluation/validator.py) judged it DEGENERATE_RESULT or FAILED_VALIDATION. It is
  NOT counted as a success and NOT retried (a structurally invalid result is
  deterministic). The result is still persisted + written as an artifact for
  inspection. run_once raises DegenerateResultError; drain() surfaces it via
  DegradedRunError. A run with any degenerate task can never be COMPLETED.
- SQLite with WAL + BEGIN IMMEDIATE prevents double-execution under concurrency
- Timeout is best-effort, NOT preemptive for in-process handlers: it makes the
  worker stop waiting and move on, but Python cannot kill the handler thread, so a
  hung in-process handler keeps using CPU/GPU until it returns (its result is
  discarded). Each timed task runs in its own single-use executor, so a hung
  handler never starves later tasks. For TRUE preemption that frees the GPU, run
  under FirejailSandbox (subprocess) with a subprocess timeout.
- Hooks fire at: before_task, after_task_success, after_task_failure, after_task_timeout, on_run_status_changed
- Concurrency: there is NO static `max_workers` knob (the old one was dead — it
  never sized any pool). `worker.drain()` runs tasks one at a time;
  `worker.drain_concurrent()` runs several under a resource-aware admission policy
  (`executor/resources.py`): it admits a new task only while free GPU memory stays
  above `min_free_mb` (the first task is always admitted), up to `max_concurrent`.
  This is "measure-and-estimate" — there is no per-task memory declaration, so an
  OOM is still possible; a GPU OOM is treated as transient (`is_oom_error`),
  requeues the task up to `max_oom_retries` times, and lowers the concurrency
  ceiling (converging toward sequential under pressure).
- **drain_concurrent is PROCESS-ISOLATED via `spawn` (reproducibility- AND
  GPU-critical).** Each task runs in its own `multiprocessing.spawn` process, for two
  reasons: (a) `apply_seed` mutates the *process-global* numpy/torch/random RNG, so
  seeded tasks as threads would interleave one shared RNG non-deterministically and
  silently destroy reproducibility; (b) `spawn` (not `fork`) gives a fresh interpreter
  that never inherits a CUDA context the parent may have created (importing the
  handler module can do this) — `fork` there is the classic CUDA-after-fork crash.
  Do NOT switch to `fork` or threads (a test asserts `spawn`). Consequences: (1) it
  REQUIRES a persistent `SQLiteStateStore` (children reopen it; InMemory raises
  `TypeError`); (2) handlers must be importable `module:function` (spawn pickles
  them, like FirejailSandbox); (3) custom hooks/sandbox must be supplied as importable
  factories — `LocalWorker(hooks_factory="module:fn", sandbox_factory="module:fn")` —
  so children recreate them; passing a *concrete* non-default `hooks`/`sandbox` to
  drain_concurrent RAISES (they cannot cross to a spawned child). A parity test
  asserts hooks fire equally in both modes; (4) the OOM retry budget is counted from
  persisted `task_oom_requeued` events, not worker memory, since each attempt is a
  fresh process; (5) `before_run_start` fires once in the parent — children inherit
  the cached seed and apply it without re-firing. With no GPU detectable
  (`free_memory_per_device() is None`, e.g. CPU CI) it degrades to sequential `drain`.
- **Dead-worker recovery.** A child OS-killed (e.g. OOM-killer) after
  `mark_task_running` dies without reporting and leaves its task RUNNING — which NO
  lease reclaim recovers (reclaim only covers LEASED). drain_concurrent detects the
  dead child (queue timeout + not alive) and calls `store.reclaim_dead_worker(run_id,
  worker_id, max_reclaims)`: requeue to PENDING for retry, or FAIL once the budget is
  spent (`task_abandoned` event) so the run terminates as degraded instead of hanging
  on a lost task or returning false success. Each child gets a unique
  `worker_id` (`<worker>-c<n>`) precisely so its stranded task is identifiable.
- Execution mode is recorded: both `drain`/`drain_concurrent` emit a `drain_started`
  event with the mode. `assess_reproducibility(run, events)` and
  `export_manifest(run, tasks, artifacts, events)` accept events and surface the
  actual mode (manifest `execution.drain_modes`); resource_aware adds a warning that
  reproducibility holds via per-process isolation. Example configs default to
  `mode: "sequential"` (the gold standard); resource_aware is opt-in.
- Configure concurrency via `executor.concurrency` (mode/min_free_mb/max_concurrent/
  settle_seconds/max_oom_retries) or CLI flags `--concurrency-mode/--min-free-mb/
  --max-concurrent`. NVML reading needs `pip install -e '.[gpu]'` (pynvml); without
  it, nvidia-smi is the fallback. **Multi-GPU:** admission is per-device — the monitor
  reports `free_memory_per_device()`, drain_concurrent picks a device (idle device
  first, else the one with most free memory that clears `min_free_mb`) and pins each
  child to it via `CUDA_VISIBLE_DEVICES`, distributing tasks across GPUs. Note: one
  GPU ⇒ effectively sequential for jobs that fill it; small jobs that fit can overlap.
  For multi-host scale, run several `finetuneharness run` processes against the same
  store — the lease guarantees each task runs exactly once (see
  `tests/concurrency/test_multiprocess.py`).
- Approval gate is ENFORCED: `finetuneharness run` refuses a run with no recorded
  approval (a `run_approved` event from `start-run`) unless `--skip-approval` is
  passed. Enforcement lives in the CLI only; `worker.drain*` itself is unguarded so
  tests and library callers are not forced through the gate.

### Two validation routes (they are NOT equivalent)

Input/output validation happens in two distinct places with different contracts:

1. **Schema validation** — `SkillRegistry.validate_input/validate_output`. Enforces
   *presence and type* of every key in the skill's `input_schema`/`output_schema`
   (e.g. `model_name` is required because it is in `COMMON_INPUT_SCHEMA`). Only runs
   when a task goes through `registry.execute(...)`.
2. **Custom validation** — the `validate_input`/`validate_output` callables on a
   `SkillSpec` (e.g. `validate_common_input`). Enforces *ranges and semantics*
   (`epochs > 0`, `accuracy in [0,1]`). The common validators do NOT re-require
   `model_name`; they assume schema validation already checked presence.

Consequence: a handler invoked directly through the worker (`worker.run_once`) does
NOT pass through schema validation, so `model_name` presence is only guaranteed on
the `registry.execute` path. Domain-specific checks (k-mer `k`, `max_per_species`)
live in `skills/biology/validators.py`, never in the generic core.

## Hard rules

- Do NOT skip the state machine — always go through the worker, never call update_task_status directly from PENDING
- Do NOT add features to the harness core for a specific experiment — the harness is generic, the skill is specific
- registry/ is a stub — do not import from it until implemented
