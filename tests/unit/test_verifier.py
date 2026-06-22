"""Tests for the independent result verifier (no-silent-lies layer)."""
from __future__ import annotations

from finetuneharness.state.models import EventRecord, TaskRecord, TaskStatus
from finetuneharness.verification import Verdict, verify_run


def _task(task_key: str, result: dict, *, status: TaskStatus = TaskStatus.SUCCEEDED) -> TaskRecord:
    return TaskRecord(
        task_id=f"id-{task_key}", run_id="r", task_key=task_key, status=status,
        payload={}, result=result,
    )


def _events(task_key: str, *, ran: bool = True, succeeded: bool = True) -> list[EventRecord]:
    tid = f"id-{task_key}"
    evs = []
    if ran:
        evs.append(EventRecord(event_id=f"e1-{task_key}", run_id="r", task_id=tid, kind="task_running"))
    if succeeded:
        evs.append(EventRecord(event_id=f"e2-{task_key}", run_id="r", task_id=tid, kind="task_succeeded"))
    return evs


_GOOD = {
    "_validation_status": "SUCCEEDED_VALIDATED",
    "accuracy": 0.55, "test_accuracy": 0.55, "f1": 0.54, "test_f1": 0.54,
    "baseline_accuracy": 0.53, "improvement": 0.02,
    "trainable_params": 100, "total_params": 100,
    "train_seconds": 12.0, "wall_seconds": 13.0,
}


def test_clean_result_passes() -> None:
    report = verify_run([_task("ok", dict(_GOOD))], _events("ok"))
    assert report.verdict is Verdict.PASS
    assert report.checked == 1
    assert report.findings == []


def test_fabricated_improvement_is_caught() -> None:
    bad = {**_GOOD, "improvement": 0.40}  # != test_accuracy - baseline (0.02)
    report = verify_run([_task("liar", bad)], _events("liar"))
    assert report.verdict is Verdict.FAIL
    assert any(f.code == "IMPROVEMENT" for f in report.fails)


def test_metric_twin_mismatch_is_caught() -> None:
    bad = {**_GOOD, "accuracy": 0.99}  # != test_accuracy 0.55
    report = verify_run([_task("twin", bad)], _events("twin"))
    assert report.verdict is Verdict.FAIL
    assert any(f.code == "METRIC_TWIN" for f in report.fails)


def test_succeeded_without_running_event_is_caught() -> None:
    report = verify_run([_task("phantom", dict(_GOOD))], _events("phantom", ran=False))
    assert report.verdict is Verdict.FAIL
    assert any(f.code == "RAN_EVIDENCE" for f in report.fails)


def test_zero_train_seconds_is_caught() -> None:
    bad = {**_GOOD, "train_seconds": 0.0}
    report = verify_run([_task("instant", bad)], _events("instant"))
    assert report.verdict is Verdict.FAIL
    assert any(f.code == "TIMING" for f in report.fails)


def test_train_longer_than_wall_is_caught() -> None:
    bad = {**_GOOD, "train_seconds": 99.0, "wall_seconds": 10.0}
    report = verify_run([_task("warp", bad)], _events("warp"))
    assert report.verdict is Verdict.FAIL
    assert any(f.code == "TIMING" for f in report.fails)


def test_trainable_exceeds_total_is_caught() -> None:
    bad = {**_GOOD, "trainable_params": 200, "total_params": 100}
    report = verify_run([_task("params", bad)], _events("params"))
    assert report.verdict is Verdict.FAIL
    assert any(f.code == "PARAMS" for f in report.fails)


def test_validation_status_incoherent_is_caught() -> None:
    bad = {**_GOOD, "_validation_status": "DEGENERATE_RESULT"}
    report = verify_run([_task("incoherent", bad)], _events("incoherent"))
    assert report.verdict is Verdict.FAIL
    assert any(f.code == "VALIDATION_COHERENT" for f in report.fails)


def test_thin_provenance_warns_not_fails() -> None:
    thin = {"_validation_status": "SUCCEEDED_VALIDATED", "accuracy": 0.55, "test_accuracy": 0.55}
    report = verify_run([_task("thin", thin)], _events("thin"))
    assert report.verdict is Verdict.WARN
    assert any(f.code == "PROVENANCE" for f in report.warns)
    assert report.fails == []


def test_only_succeeded_tasks_are_checked() -> None:
    failed = _task("f", {"error": "x"}, status=TaskStatus.FAILED)
    report = verify_run([failed], [])
    assert report.checked == 0
    assert report.verdict is Verdict.PASS
