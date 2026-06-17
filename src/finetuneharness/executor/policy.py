from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1


@dataclass(frozen=True)
class TimeoutPolicy:
    timeout_seconds: int | None = None
