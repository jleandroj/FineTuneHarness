from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import subprocess
import sys
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

    Input (parent→child): handler + task are serialized with pickle — unavoidable
    because handlers are callables. Handlers must be module-level functions
    (not lambdas or closures) to be picklable.

    Output (child→parent): result is serialized as JSON, not pickle.
    This eliminates pickle.loads on subprocess output, which can execute
    arbitrary code if the child writes a crafted payload or raises a custom
    exception whose class is not importable in the parent.

    Tradeoff: the original exception type does not cross the process boundary.
    Handler exceptions are re-raised as RuntimeError with the class name and
    message embedded in the string.

    Network is disabled (--net=none) and /tmp is private by default.
    Pass extra_args to add further firejail restrictions (e.g. --read-only=/).
    """

    _RUNNER = "\n".join([
        "import json, pickle, sys",
        "fn, t = pickle.loads(sys.stdin.buffer.read())",
        "try:",
        "    result = fn(t)",
        "    sys.stdout.buffer.write(json.dumps({'ok': True, 'result': result}).encode())",
        "except Exception as exc:",
        "    msg = type(exc).__name__ + ': ' + str(exc)",
        "    sys.stdout.buffer.write(json.dumps({'ok': False, 'error': msg}).encode())",
    ])

    def __init__(self, *, extra_args: tuple[str, ...] = ()) -> None:
        if shutil.which("firejail") is None:
            raise RuntimeError("firejail not found on PATH — install it first")
        self._extra_args = extra_args

    def run(self, handler, task) -> dict[str, object]:
        try:
            payload = pickle.dumps((handler, task))
        except (AttributeError, pickle.PicklingError) as exc:
            raise RuntimeError(
                f"FirejailSandbox requires a module-level function, not a lambda or closure. "
                f"Define your handler as a top-level def in an importable module. "
                f"pickle error: {exc}"
            ) from exc
        # Use sys.executable so the subprocess runs in the same venv, and pass
        # the parent's sys.path as PYTHONPATH so handlers defined in installed
        # packages (and test modules) are importable in the child process.
        pythonpath = os.pathsep.join(p for p in sys.path if p)
        cmd = [
            "firejail", "--quiet", "--net=none", "--private-tmp",
            f"--env=PYTHONPATH={pythonpath}",
            *self._extra_args,
            sys.executable, "-c", self._RUNNER,
        ]
        proc = subprocess.run(cmd, input=payload, capture_output=True, check=False)
        try:
            data = json.loads(proc.stdout)
        except Exception:
            stderr = proc.stderr.decode(errors="replace")
            raise RuntimeError(
                f"firejail subprocess failed (exit {proc.returncode}): {stderr}"
            ) from None
        if not data["ok"]:
            raise RuntimeError(f"handler raised in sandbox: {data['error']}")
        return data["result"]
