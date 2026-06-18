from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any


# Standard attributes present on every LogRecord — exclude from extra= scan.
# Includes both documented attributes and internal ones set by the logging machinery.
_BUILTIN_ATTRS: frozenset[str] = frozenset({
    "args", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "message", "module", "msecs",
    "msg", "name", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName", "taskName",
})


def _json_default(obj: Any) -> Any:
    """Fallback serializer for types json.dumps cannot handle natively.

    Converts to str rather than crashing — a lossy but safe representation
    that lets the log line appear even when complex objects are in extra=.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        # Handles enums (.value), dataclasses (asdict already done upstream), etc.
        return str(obj)
    except Exception:
        return repr(obj)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            # record.created is the epoch float set when Logger.log() was called —
            # earlier than format() is called, so timestamps reflect the actual event.
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Include every field injected via extra={} — skip built-ins and privates.
        # This replaces the old 3-key whitelist: any caller-supplied context survives.
        for key, value in record.__dict__.items():
            if key in _BUILTIN_ATTRS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=_json_default, ensure_ascii=False)


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
