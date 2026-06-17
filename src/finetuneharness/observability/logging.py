from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key in ("run_id", "task_id", "event_kind"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


_configure_lock = threading.Lock()
_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        with _configure_lock:
            if not _configured:
                handler = logging.StreamHandler()
                handler.setFormatter(JsonFormatter())
                root = logging.getLogger("finetuneharness")
                root.setLevel(logging.INFO)
                root.handlers.clear()
                root.addHandler(handler)
                _configured = True
    return logging.getLogger(name)
