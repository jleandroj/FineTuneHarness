from __future__ import annotations

import io as _io
import sys
from typing import TextIO

from finetuneharness.state.models import RunRecord, TaskRecord


class ApprovalError(Exception):
    """Raised when a run is denied by an approval gate."""


class ApprovalGate:
    """Base approval gate — always approves. Subclass to add logic."""

    def check(self, run: RunRecord, tasks: list[TaskRecord]) -> None:
        """Raise ApprovalError to block execution; return normally to approve."""


class InteractiveApprovalGate(ApprovalGate):
    """Prints a run summary and prompts the operator for explicit confirmation.

    Accepts a stream for testing so stdin is never required in automated contexts.
    """

    def __init__(self, *, stream: TextIO | None = None) -> None:
        self._stream = stream

    def check(self, run: RunRecord, tasks: list[TaskRecord]) -> None:
        pending = sum(1 for t in tasks if t.status.value == "pending")
        total = len(tasks)
        print(f"\n--- Run Approval Request ---")
        print(f"  Run ID : {run.run_id}")
        print(f"  Name   : {run.name}")
        print(f"  Status : {run.status.value}")
        print(f"  Tasks  : {pending} pending / {total} total")
        print()

        src = self._stream or sys.stdin
        try:
            sys.stdout.write("Approve this run? [y/N] ")
            sys.stdout.flush()
            answer = src.readline().strip().lower()
        except EOFError:
            answer = ""

        if answer not in ("y", "yes"):
            raise ApprovalError(f"run {run.run_id!r} denied by operator")
