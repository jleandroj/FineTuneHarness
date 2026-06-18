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

- Tasks have explicit state: PENDING → LEASED → RUNNING → SUCCEEDED/FAILED/TIMED_OUT
- State machine is enforced — invalid transitions raise ValueError
- SQLite with WAL + BEGIN IMMEDIATE prevents double-execution under concurrency
- Timeout is best-effort, NOT preemptive for in-process handlers: it makes the
  worker stop waiting and move on, but Python cannot kill the handler thread, so a
  hung in-process handler keeps using CPU/GPU until it returns (its result is
  discarded). Each timed task runs in its own single-use executor, so a hung
  handler never starves later tasks. For TRUE preemption that frees the GPU, run
  under FirejailSandbox (subprocess) with a subprocess timeout.
- Hooks fire at: before_task, after_task_success, after_task_failure, after_task_timeout, on_run_status_changed

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
