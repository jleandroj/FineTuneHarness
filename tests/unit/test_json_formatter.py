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
