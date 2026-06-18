from __future__ import annotations

import math
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from finetuneharness.state.models import TaskRecord


@runtime_checkable
class SandboxPolicy(Protocol):
    def run(
        self,
        handler: "Callable[[TaskRecord], dict[str, object]]",
        task: "TaskRecord",
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    delay after attempt n = min(cap, base * 2^n) * U(0, 1)
    where n is 0-indexed (first retry = n=0).
    """

    max_attempts: int = 1
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    jitter: bool = True

    def delay_for_attempt(self, attempt: int) -> float:
        """Return seconds to wait before attempt number *attempt* (0-indexed).

        Attempt 0 is the first try — no delay.  Attempt 1 is the first retry.
        """
        if attempt == 0:
            return 0.0
        import random

        exp = min(self.max_delay_seconds, self.base_delay_seconds * math.pow(2, attempt - 1))
        return random.uniform(0, exp) if self.jitter else exp


@dataclass(frozen=True)
class TimeoutPolicy:
    timeout_seconds: int | None = None


class NoSandbox:
    """Passthrough: runs the handler directly in-process (default behavior)."""

    def run(self, handler, task) -> dict[str, object]:
        return handler(task)


class FirejailSandbox:
    """Runs each handler in an isolated firejail subprocess.

    The handler and task are serialized with pickle, piped to a child Python
    process launched under firejail, and the result is deserialized on return.
    Handlers must be picklable (module-level functions — not lambdas).

    Network is disabled (--net=none) and /tmp is private by default.
    Pass extra_args to add further firejail restrictions (e.g. --read-only=/).
    """

    _RUNNER = "\n".join([
        "import pickle, sys",
        "fn, t = pickle.loads(sys.stdin.buffer.read())",
        "try:",
        "    out = (True, fn(t))",
        "except Exception as exc:",
        "    out = (False, exc)",
        "sys.stdout.buffer.write(pickle.dumps(out))",
    ])

    def __init__(self, *, extra_args: tuple[str, ...] = ()) -> None:
        if shutil.which("firejail") is None:
            raise RuntimeError("firejail not found on PATH — install it first")
        self._extra_args = extra_args

    def run(self, handler, task) -> dict[str, object]:
        payload = pickle.dumps((handler, task))
        cmd = [
            "firejail", "--quiet", "--net=none", "--private-tmp",
            *self._extra_args,
            "python3", "-c", self._RUNNER,
        ]
        proc = subprocess.run(cmd, input=payload, capture_output=True, check=False)
        # stdout comes from our own child process — safe to unpickle
        try:
            ok, value = pickle.loads(proc.stdout)  # noqa: S301
        except Exception:
            stderr = proc.stderr.decode(errors="replace")
            raise RuntimeError(
                f"firejail subprocess failed (exit {proc.returncode}): {stderr}"
            ) from None
        if not ok:
            raise value
        return value
