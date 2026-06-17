# FineTuneHarness

FineTuneHarness is a production-minded execution harness for long-running experiments, evaluations, and agent-style workloads.

It exists to fix the exact class of problems found in fragile research runners:
- silent partial failures
- duplicate execution
- non-transactional checkpointing
- weak reproducibility
- poor observability
- shell-script orchestration drift

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
