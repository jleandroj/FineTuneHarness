# FineTuneHarness

FineTuneHarness is a production-minded execution harness for long-running experiments, evaluations, and agent-style workloads.

It exists to fix the exact class of problems found in fragile research runners:
- silent partial failures
- duplicate execution
- non-transactional checkpointing
- weak reproducibility
- poor observability
- shell-script orchestration drift

## Installation

```bash
# Base (CPU): the harness itself has no heavy dependencies.
pip install -e .

# Dev (tests, lint, types):
pip install -e '.[dev]'

# GPU resource monitoring (NVML, for resource_aware concurrency):
pip install -e '.[gpu]'   # pulls nvidia-ml-py
```

### Running on a GPU

The harness core is dependency-light; your **handler** brings the ML stack
(torch, etc.). Install a CUDA-enabled torch that matches your driver and GPU — a
plain `pip install torch` often yields a CPU-only build (`2.x.y+cpu`,
`torch.version.cuda == None`), and then `drain_concurrent` will run, but training
falls back to CPU.

Pick the wheel for your CUDA version (`nvidia-smi` top-right shows the max CUDA the
driver supports). Example for an **NVIDIA GB10 / DGX Spark** (aarch64, compute
capability `sm_121`, driver supporting CUDA 13) on Python 3.13 — verified working:

```bash
pip uninstall -y torch pynvml          # drop CPU torch + deprecated pynvml
pip install --index-url https://download.pytorch.org/whl/cu130 torch
# -> torch 2.12.1+cu130. If a real op fails with "no kernel image is available",
#    try the cu130 nightly: pip install --pre --index-url \
#    https://download.pytorch.org/whl/nightly/cu130 torch
```

Verify CUDA is actually usable (not just `is_available`):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); \
print('available', torch.cuda.is_available()); \
print('cap', torch.cuda.get_device_capability()); \
print('op', (torch.ones(3, device='cuda')*2).sum().item())"
```

GPU-only tests are marked `@pytest.mark.gpu` and auto-skip without CUDA:

```bash
pytest -m gpu -v        # real model trains to COMPLETED; real CUDA OOM is classified
```

Note: on unified-memory GPUs (Grace-Blackwell GB10), per-device free memory is not
reported by NVML/nvidia-smi, so `resource_aware` concurrency degrades to sequential
(the correct, safe behavior — and with one GPU, sequential is the right model).

## Design goals

1. **State is explicit**
2. **Failure is never silent**
3. **Artifacts are traceable**
4. **Runs are reproducible**
5. **Concurrency is controlled**
6. **Evaluation is separate from execution**

## Planned core modules

- `orchestrator` — run/task lifecycle and recovery
- `state` — source of truth for runs, tasks, events, artifacts
- `executor` — isolated workload execution
- `validation` — config, dataset, checkpoint, output validation
- `observability` — structured logs and audit events
- `evaluation` — metrics, comparisons, reports
- `registry` — techniques, models, datasets, evaluators
- `artifacts` — immutable artifact naming and storage contracts

## Non-negotiable rules

- CSV is not the source of truth
- task state transitions must be auditable
- completed/failed/partial runs must be distinguishable
- every artifact must be attributable to a run and task
- configuration must be frozen per run

## Current status

This is the initial scaffold created from a technical audit of a fragile experiment runner.
It is not yet feature-complete, but the structure is oriented toward a world-class rebuild rather than incremental patching.

Current bootstrap capabilities:
- explicit run/task state models
- lifecycle transition validation
- in-memory state store for tests
- SQLite persistent state store for local serious use
- bootstrap CLI for persistent run creation
- task leasing and lease requeue
- automatic run status aggregation
- basic event persistence for task/run transitions
- filesystem artifact store with SHA-256 checksums
- retry and timeout handling with persisted attempt counts
