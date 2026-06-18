from finetuneharness.orchestrator.runner import FineTuneRunner
from finetuneharness.state.memory_store import InMemoryStateStore


def test_create_run_bootstrap() -> None:
    runner = FineTuneRunner(InMemoryStateStore())
    run_id = runner.create_run(
        name="bootstrap",
        config={
            "project": {"name": "demo"},
            "executor": {"kind": "local"},
            "artifacts": {"root": "./artifacts"},
            "seed": 42,
            "dataset_hash": "sha256:test",
        },
        tasks=[{"task_key": "cell-1", "kind": "train"}],
    )
    assert run_id
