"""Tests for JsonFormatter — completeness of structured log output."""
from __future__ import annotations

import json
import logging
import time

import pytest

from finetuneharness.observability.logging import JsonFormatter


def _make_record(
    msg: str = "hello",
    level: int = logging.INFO,
    *,
    exc_info=None,
    stack_info: str | None = None,
    extra: dict | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    record.stack_info = stack_info
    for k, v in (extra or {}).items():
        setattr(record, k, v)
    return record


class TestJsonFormatterTimestamp:
    def test_ts_matches_record_created_not_format_time(self):
        """ts must reflect when the log was emitted, not when format() runs."""
        formatter = JsonFormatter()
        before = time.time()
        record = _make_record()
        after = time.time()
        # Simulate a delay between record creation and format (e.g. async buffering)
        time.sleep(0.05)
        payload = json.loads(formatter.format(record))
        # record.created is between before and after — ts must be in that window
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(payload["ts"]).timestamp()
        assert before <= ts <= after, (
            f"ts={payload['ts']} is not within the record creation window "
            f"[{before}, {after}] — it reflects format() time, not event time"
        )

    def test_ts_is_utc_iso8601(self):
        formatter = JsonFormatter()
        payload = json.loads(formatter.format(_make_record()))
        assert payload["ts"].endswith("+00:00") or payload["ts"].endswith("Z")


class TestJsonFormatterStructure:
    def test_required_fields_present(self):
        formatter = JsonFormatter()
        payload = json.loads(formatter.format(_make_record()))
        assert "ts" in payload
        assert "logger" in payload
        assert "level" in payload
        assert "message" in payload

    def test_optional_context_fields_included_when_set(self):
        formatter = JsonFormatter()
        record = _make_record(extra={"run_id": "abc", "task_id": "xyz"})
        payload = json.loads(formatter.format(record))
        assert payload["run_id"] == "abc"
        assert payload["task_id"] == "xyz"

    def test_optional_context_fields_absent_when_not_set(self):
        formatter = JsonFormatter()
        payload = json.loads(formatter.format(_make_record()))
        assert "run_id" not in payload
        assert "task_id" not in payload
        assert "event_kind" not in payload


class TestJsonFormatterExceptions:
    def test_exc_field_included_when_exc_info_set(self):
        """Exceptions must appear in the JSON — not silently dropped."""
        formatter = JsonFormatter()
        try:
            raise ValueError("something went wrong")
        except ValueError:
            import sys
            record = _make_record(level=logging.ERROR, exc_info=sys.exc_info())

        payload = json.loads(formatter.format(record))
        assert "exc" in payload, (
            "exc_info is set but 'exc' field is missing from JSON — "
            "tracebacks are silently dropped, making 2am debugging impossible"
        )
        assert "ValueError" in payload["exc"]
        assert "something went wrong" in payload["exc"]

    def test_no_exc_field_when_no_exception(self):
        formatter = JsonFormatter()
        payload = json.loads(formatter.format(_make_record()))
        assert "exc" not in payload

    def test_stack_field_included_when_stack_info_set(self):
        formatter = JsonFormatter()
        record = _make_record(stack_info="Stack:\n  File test.py line 1")
        payload = json.loads(formatter.format(record))
        assert "stack" in payload
        assert "test.py" in payload["stack"]

    def test_output_is_valid_json(self):
        formatter = JsonFormatter()
        try:
            raise RuntimeError("boom\nwith newline")
        except RuntimeError:
            import sys
            record = _make_record(level=logging.ERROR, exc_info=sys.exc_info())
        # Must not raise
        raw = formatter.format(record)
        parsed = json.loads(raw)
        assert parsed["level"] == "ERROR"


class TestJsonFormatterExtraFields:
    """extra= fields must all appear in JSON — not just the 3-key whitelist."""

    def test_arbitrary_extra_field_included(self):
        """Any field in extra={} must survive into the JSON payload."""
        formatter = JsonFormatter()
        record = _make_record(extra={"task_count": 72, "gpu_peak_mb": 4200.5})
        payload = json.loads(formatter.format(record))
        assert payload["task_count"] == 72, (
            "task_count was passed via extra= but is absent from JSON — "
            "only the 3-key whitelist was extracted (regression)"
        )
        assert payload["gpu_peak_mb"] == 4200.5

    def test_hook_error_context_fields_included(self):
        """hooks.py logs 'hook_error' with point/hook/error/traceback — all must appear."""
        formatter = JsonFormatter()
        record = _make_record(
            msg="hook_error",
            level=logging.WARNING,
            extra={
                "point": "after_task_success",
                "hook": "GPUMemoryHook.after_task_success",
                "error": "CUDA out of memory",
                "traceback": "Traceback (most recent call last):\n  ...",
            },
        )
        payload = json.loads(formatter.format(record))
        assert payload["point"] == "after_task_success"
        assert payload["hook"] == "GPUMemoryHook.after_task_success"
        assert payload["error"] == "CUDA out of memory"
        assert "Traceback" in payload["traceback"]

    def test_run_created_context_fields_included(self):
        """runner.py logs run_created with run_id + task_count — task_count must appear."""
        formatter = JsonFormatter()
        record = _make_record(extra={"run_id": "abc123", "task_count": 12})
        payload = json.loads(formatter.format(record))
        assert payload["run_id"] == "abc123"
        assert payload["task_count"] == 12

    def test_builtin_logrecord_attrs_not_duplicated(self):
        """Standard LogRecord attrs (levelno, lineno, etc.) must not leak into payload."""
        formatter = JsonFormatter()
        payload = json.loads(formatter.format(_make_record()))
        # These are internal LogRecord attrs — they should not appear in JSON
        assert "levelno" not in payload
        assert "lineno" not in payload
        assert "pathname" not in payload
        assert "process" not in payload
        assert "thread" not in payload

    def test_non_serializable_extra_field_does_not_crash(self):
        """Non-JSON-serializable types (datetime, enum, custom class) must not raise."""
        from datetime import datetime, timezone
        from enum import Enum

        class Color(Enum):
            RED = 1

        formatter = JsonFormatter()
        record = _make_record(extra={
            "started_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "color": Color.RED,
        })
        # Must not raise — uses _json_default fallback
        raw = formatter.format(record)
        payload = json.loads(raw)
        assert "started_at" in payload
        assert "color" in payload

    def test_private_underscore_fields_excluded(self):
        """Fields starting with _ must not appear — they are internal to logging."""
        formatter = JsonFormatter()
        record = _make_record()
        # logging sets some internal _ attrs; none should appear in JSON
        payload = json.loads(formatter.format(record))
        for key in payload:
            assert not key.startswith("_"), f"Private field {key!r} leaked into JSON"


class TestJsonFormatterSensitiveRedaction:
    """Sensitive keys must be redacted — never logged in plaintext."""

    def test_password_redacted(self):
        formatter = JsonFormatter()
        record = _make_record(extra={"password": "hunter2"})
        payload = json.loads(formatter.format(record))
        assert payload["password"] == "[REDACTED]"
        assert "hunter2" not in json.dumps(payload)

    def test_api_key_redacted(self):
        formatter = JsonFormatter()
        record = _make_record(extra={"api_key": "sk-abc123"})
        payload = json.loads(formatter.format(record))
        assert payload["api_key"] == "[REDACTED]"

    def test_token_variants_redacted(self):
        """token, hf_token, auth_token, bearer_token — all must be redacted."""
        formatter = JsonFormatter()
        for key in ("token", "hf_token", "auth_token", "bearer_token", "HF_TOKEN"):
            record = _make_record(extra={key: "secret-value"})
            payload = json.loads(formatter.format(record))
            assert payload[key] == "[REDACTED]", f"key {key!r} not redacted"

    def test_secret_redacted(self):
        formatter = JsonFormatter()
        record = _make_record(extra={"db_secret": "password123", "secret_key": "xyz"})
        payload = json.loads(formatter.format(record))
        assert payload["db_secret"] == "[REDACTED]"
        assert payload["secret_key"] == "[REDACTED]"

    def test_non_sensitive_fields_pass_through(self):
        """Ordinary fields must not be redacted."""
        formatter = JsonFormatter()
        record = _make_record(extra={
            "run_id": "abc", "task_count": 5, "accuracy": 0.9, "author": "alice",
        })
        payload = json.loads(formatter.format(record))
        assert payload["run_id"] == "abc"
        assert payload["task_count"] == 5
        assert payload["accuracy"] == 0.9
        assert payload["author"] == "alice"

    def test_redacted_value_is_literal_string(self):
        """Redaction marker must be the literal string [REDACTED], not None or empty."""
        formatter = JsonFormatter()
        record = _make_record(extra={"password": "x"})
        payload = json.loads(formatter.format(record))
        assert payload["password"] == "[REDACTED]"

    def test_sensitive_key_present_in_output_as_redacted_not_absent(self):
        """Key must be present with [REDACTED] value — absence would hide that a sensitive
        field was passed, making auditing harder."""
        formatter = JsonFormatter()
        record = _make_record(extra={"api_key": "real-key"})
        payload = json.loads(formatter.format(record))
        assert "api_key" in payload, "sensitive key must appear in output (as [REDACTED])"
