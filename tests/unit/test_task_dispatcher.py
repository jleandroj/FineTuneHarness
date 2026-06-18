"""Tests for TaskDispatcher and validate_task_payload.

Covers the 5 cases the user asked for:
  1. dispatch routes to the correct handler by kind
  2. unknown kind raises ValueError with the registry listed
  3. missing 'kind' in payload raises ValueError
  4. 'kind' is not a string raises ValueError
  5. duplicate register raises ValueError
"""
from __future__ import annotations

import pytest

from finetuneharness.registry.dispatcher import TaskDispatcher, validate_task_payload
from finetuneharness.state.models import TaskRecord, TaskStatus


def _make_task(payload: dict) -> TaskRecord:
    return TaskRecord(
        task_id="task-abc",
        run_id="run-xyz",
        task_key="t0",
        status=TaskStatus.RUNNING,
        payload=payload,
    )


# ── 1. dispatch routes to the right handler ──────────────────────────────────

def test_dispatch_routes_by_kind():
    called_with = {}

    def run_distill(task):
        called_with["kind"] = task.payload["kind"]
        return {"accuracy": 0.9}

    dispatcher = TaskDispatcher()
    dispatcher.register("distill", run_distill)

    task = _make_task({"kind": "distill", "epochs": 5})
    result = dispatcher.dispatch(task)

    assert result == {"accuracy": 0.9}
    assert called_with["kind"] == "distill"


def test_dispatch_routes_to_correct_handler_among_multiple():
    results = []

    def handler_a(task):
        results.append("a")
        return {"ok": True}

    def handler_b(task):
        results.append("b")
        return {"ok": True}

    dispatcher = TaskDispatcher()
    dispatcher.register("prune", handler_a)
    dispatcher.register("distill", handler_b)

    dispatcher.dispatch(_make_task({"kind": "distill"}))
    dispatcher.dispatch(_make_task({"kind": "prune"}))

    assert results == ["b", "a"]


# ── 2. unknown kind raises ValueError ────────────────────────────────────────

def test_dispatch_unknown_kind_raises_with_registry_list():
    dispatcher = TaskDispatcher()
    dispatcher.register("distill", lambda t: {})

    with pytest.raises(ValueError, match="distill"):
        dispatcher.dispatch(_make_task({"kind": "unknown_task"}))


def test_dispatch_empty_registry_raises():
    dispatcher = TaskDispatcher()
    with pytest.raises(ValueError, match="No handler registered"):
        dispatcher.dispatch(_make_task({"kind": "anything"}))


# ── 3. missing 'kind' raises ValueError ──────────────────────────────────────

def test_dispatch_missing_kind_raises():
    dispatcher = TaskDispatcher()
    dispatcher.register("distill", lambda t: {})

    with pytest.raises(ValueError, match="missing required field 'kind'"):
        dispatcher.dispatch(_make_task({"epochs": 5}))


def test_validate_task_payload_missing_kind():
    task = _make_task({"param": 42})
    with pytest.raises(ValueError, match="missing required field 'kind'"):
        validate_task_payload(task)


# ── 4. kind is not a string raises ValueError ─────────────────────────────────

def test_dispatch_kind_not_a_string_raises():
    dispatcher = TaskDispatcher()
    with pytest.raises(ValueError, match="must be a str"):
        dispatcher.dispatch(_make_task({"kind": 99}))


def test_validate_task_payload_kind_empty_string_raises():
    task = _make_task({"kind": "   "})
    with pytest.raises(ValueError, match="non-empty"):
        validate_task_payload(task)


# ── 5. duplicate register raises ValueError ───────────────────────────────────

def test_register_duplicate_kind_raises():
    dispatcher = TaskDispatcher()
    dispatcher.register("distill", lambda t: {})
    with pytest.raises(ValueError, match="already registered"):
        dispatcher.register("distill", lambda t: {})


# ── kinds() lists registered names sorted ────────────────────────────────────

def test_kinds_returns_sorted_list():
    dispatcher = TaskDispatcher()
    dispatcher.register("prune", lambda t: {})
    dispatcher.register("distill", lambda t: {})
    dispatcher.register("lora", lambda t: {})

    assert dispatcher.kinds() == ["distill", "lora", "prune"]
