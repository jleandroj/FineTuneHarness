from __future__ import annotations

import getpass
import sys
from typing import TextIO

from finetuneharness.state.models import RunRecord, TaskRecord


def resolve_actor(actor: str | None = None) -> str:
    """Resolve the acting operator's identity: explicit value, else the OS user."""
    if actor:
        return actor
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


class ApprovalError(Exception):
    """Raised when a run is denied by an approval gate."""


class ApprovalGate:
    """Base approval gate — always approves. Subclass to add logic.

    ``check`` returns the verified *actor* (who approved), recorded in the audit
    event. The base gate attributes approval to the resolved OS user.
    """

    def check(self, run: RunRecord, tasks: list[TaskRecord]) -> str | None:
        """Raise ApprovalError to block; return the approving actor to allow."""
        return resolve_actor()


class InteractiveApprovalGate(ApprovalGate):
    """Prints a run summary and prompts the operator for explicit confirmation.

    Records the approving *actor* (``--approver`` / explicit, else the OS user) and,
    when ``allowed_actors`` is set, verifies the actor is authorized — a lightweight
    multi-tenant gate (attribution + allowlist, NOT cryptographic identity/SSO).
    Accepts a stream for testing so stdin is never required in automated contexts.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        actor: str | None = None,
        allowed_actors: tuple[str, ...] | None = None,
    ) -> None:
        self._stream = stream
        self._actor = resolve_actor(actor)
        self._allowed_actors = allowed_actors

    def check(self, run: RunRecord, tasks: list[TaskRecord]) -> str | None:
        if self._allowed_actors is not None and self._actor not in self._allowed_actors:
            raise ApprovalError(
                f"actor {self._actor!r} is not authorized to approve run "
                f"{run.run_id!r} (allowed: {sorted(self._allowed_actors)})"
            )

        pending = sum(1 for t in tasks if t.status.value == "pending")
        total = len(tasks)
        print("\n--- Run Approval Request ---")
        print(f"  Run ID  : {run.run_id}")
        print(f"  Name    : {run.name}")
        print(f"  Status  : {run.status.value}")
        print(f"  Tasks   : {pending} pending / {total} total")
        print(f"  Approver: {self._actor}")
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
        return self._actor
