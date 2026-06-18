from __future__ import annotations

import json
import logging
import re
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

# Keys whose values must never appear in logs in plaintext.
#
# Two-tier matching (all case-insensitive):
#   _SENSITIVE_WORDS  — each word segment (split on _ - .) is checked against
#                       these; catches "token", "hf_token", "db_password", "auth"
#                       without false-positives like "author" (segment "author" ≠ "auth").
#   _SENSITIVE_FULL_KEYS — exact full-key matches for compound names that cannot
#                          be caught by single-word segments (e.g. "api_key").
#
# Note on "pass": we deliberately do NOT add the bare segment "pass" to
# _SENSITIVE_WORDS — it would false-positive on legitimate ML metric names like
# "pass_rate" or "pass@k". Instead we add the unambiguous words "passwd"/
# "passphrase" and enumerate common password abbreviations as full keys below.
_SENSITIVE_WORDS: frozenset[str] = frozenset({
    "token", "secret", "password", "passwd", "passphrase",
    "credential", "credentials", "auth",
})
_SENSITIVE_FULL_KEYS: frozenset[str] = frozenset({
    "api_key", "apikey", "private_key", "access_key",
    "auth_key", "auth_token", "bearer_token", "oauth_token",
    "root_pass", "db_pass", "user_pass", "admin_pass",
})

_REDACTED = "[REDACTED]"
_SPLIT_RE = re.compile(r"[_\-\.]")


def _is_sensitive(key: str) -> bool:
    k = key.lower()
    if k in _SENSITIVE_FULL_KEYS:
        return True
    return bool(set(_SPLIT_RE.split(k)) & _SENSITIVE_WORDS)


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
        # Sensitive keys (password, token, api_key, …) are redacted in place.
        for key, value in record.__dict__.items():
            if key in _BUILTIN_ATTRS or key.startswith("_"):
                continue
            payload[key] = _REDACTED if _is_sensitive(key) else value
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
