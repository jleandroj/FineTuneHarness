"""Reference LoRA handler: a healthy adapter run, and the degeneracy contract.

Skipped entirely unless the `lora` extra (peft) is installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("peft", reason="requires the 'lora' extra (peft)")

from finetuneharness.evaluation.validator import ResultStatus, validate_result  # noqa: E402
from finetuneharness.examples.lora_handler import train  # noqa: E402
from finetuneharness.executor.worker import LocalWorker  # noqa: E402
from finetuneharness.orchestrator.runner import FineTuneRunner  # noqa: E402
from finetuneharness.state.models import RunStatus, TaskStatus  # noqa: E402
from finetuneharness.state.sqlite import SQLiteStateStore  # noqa: E402

_CONFIG = {
    "project": {"name": "lora"},
    "executor": {"kind": "local"},
    "artifacts": {"root": "./artifacts"},
    "seed": 11,
    "dataset_hash": "sha256:synthetic",
}


def test_lora_handler_end_to_end_cpu(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "state.db")
    runner = FineTuneRunner(store)
    run_id = runner.create_run(
        name="lora", config=_CONFIG,
        tasks=[{"task_key": "cell", "hidden": 16, "steps": 10, "device": "cpu"}],
    )
    assert LocalWorker(worker_id="w", store=store).drain(run_id=run_id, handler=train) == 1

    assert FineTuneRunner(store).refresh_run_status(run_id) is RunStatus.COMPLETED
    task = store.list_tasks(run_id)[0]
    assert task.status is TaskStatus.SUCCEEDED
    res = task.result
    assert res["adapter_loaded"] is True
    assert res["trainable_params"] > 0
    assert res["trainable_params"] < res["total_params"]
    # And the healthy adapter result passes the result validator.
    assert validate_result(res).status is ResultStatus.SUCCEEDED_VALIDATED
