# FineTuneHarness Architecture

## Purpose

FineTuneHarness is designed to turn fragile experiment execution into auditable, recoverable, reproducible execution.

## Core model

### Run
A top-level execution request.

States:
- `created`
- `validated`
- `running`
- `partial_failed`
- `failed`
- `completed`
- `cancelled`

### Task
A unit of execution within a run.

States:
- `pending`
- `leased`
- `running`
- `succeeded`
- `failed`
- `timed_out`
- `cancelled`

### Event
Immutable audit record for transitions and important side effects.

### Artifact
Named output with checksum and provenance.

## Architectural principles

1. The state store is the source of truth.
2. The orchestrator owns lifecycle transitions.
3. The executor never mutates global state directly.
4. Validation happens before and after execution.
5. Logs are structured and machine-readable.
6. Recovery is a first-class feature.

## Immediate build order

1. State model
2. Run/task lifecycle
3. Structured logging
4. CLI commands
5. Validation layer
6. Persistent store backend
7. Recovery and leases
8. Evaluation and reporting
