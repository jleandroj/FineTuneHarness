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
- Timeout is preemptive via ThreadPoolExecutor
- Hooks fire at: before_task, after_task_success, after_task_failure, after_task_timeout, on_run_status_changed

## Hard rules

- Do NOT skip the state machine — always go through the worker, never call update_task_status directly from PENDING
- Do NOT add features to the harness core for a specific experiment — the harness is generic, the skill is specific
- registry/ is a stub — do not import from it until implemented
