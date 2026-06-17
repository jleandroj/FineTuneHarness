from __future__ import annotations

import io

import pytest

from finetuneharness.orchestrator.approval import ApprovalError, ApprovalGate, InteractiveApprovalGate
from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore
from finetuneharness.state.models import EventRecord

_BASE_CONFIG = {
    "project": {"name": "approval-test"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
}


def _make_run(runner: FineTuneRunner, name: str = "test-run") -> str:
    return runner.create_run(
        name=name,
        config=_BASE_CONFIG,
        tasks=[{"task_key": "a"}, {"task_key": "b"}],
    )


# --- ApprovalGate base class ---

def test_base_gate_always_approves() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner)

    runner.await_approval(run_id, ApprovalGate())  # must not raise


def test_await_approval_records_event() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner)

    runner.await_approval(run_id, ApprovalGate())

    events = store.list_events(run_id)
    kinds = [e.kind for e in events]
    assert "run_approved" in kinds


def test_await_approval_unknown_run_raises() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)

    with pytest.raises(KeyError):
        runner.await_approval("does-not-exist", ApprovalGate())


# --- InteractiveApprovalGate ---

def test_interactive_gate_approves_on_yes() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner)

    gate = InteractiveApprovalGate(stream=io.StringIO("y\n"))
    runner.await_approval(run_id, gate)  # must not raise


def test_interactive_gate_approves_on_yes_uppercase() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner)

    gate = InteractiveApprovalGate(stream=io.StringIO("Y\n"))
    runner.await_approval(run_id, gate)


def test_interactive_gate_denies_on_no() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner)

    gate = InteractiveApprovalGate(stream=io.StringIO("n\n"))
    with pytest.raises(ApprovalError):
        runner.await_approval(run_id, gate)


def test_interactive_gate_denies_on_empty_input() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner)

    gate = InteractiveApprovalGate(stream=io.StringIO("\n"))
    with pytest.raises(ApprovalError):
        runner.await_approval(run_id, gate)


def test_interactive_gate_denies_on_eof() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner)

    gate = InteractiveApprovalGate(stream=io.StringIO(""))
    with pytest.raises(ApprovalError):
        runner.await_approval(run_id, gate)


# --- Custom gate subclass ---

def test_custom_gate_can_block() -> None:
    class AlwaysDeny(ApprovalGate):
        def check(self, run, tasks):  # type: ignore[override]
            raise ApprovalError("blocked by policy")

    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner)

    with pytest.raises(ApprovalError, match="blocked by policy"):
        runner.await_approval(run_id, AlwaysDeny())


def test_denied_run_does_not_record_approved_event() -> None:
    store = InMemoryStateStore()
    runner = FineTuneRunner(store)
    run_id = _make_run(runner)

    gate = InteractiveApprovalGate(stream=io.StringIO("no\n"))
    with pytest.raises(ApprovalError):
        runner.await_approval(run_id, gate)

    events = store.list_events(run_id)
    assert "run_approved" not in [e.kind for e in events]
