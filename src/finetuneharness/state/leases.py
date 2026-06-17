from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Lease:
    worker_id: str
    leased_until: datetime

    @classmethod
    def from_seconds(cls, worker_id: str, seconds: int) -> "Lease":
        return cls(worker_id=worker_id, leased_until=utc_now() + timedelta(seconds=seconds))

    def is_expired(self, *, now: datetime | None = None) -> bool:
        ref = now or utc_now()
        return self.leased_until <= ref
