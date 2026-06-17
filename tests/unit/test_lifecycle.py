from finetuneharness.orchestrator.lifecycle import ensure_run_transition
from finetuneharness.state.models import RunStatus


def test_valid_run_transition() -> None:
    ensure_run_transition(RunStatus.CREATED, RunStatus.VALIDATED)


def test_invalid_terminal_transition() -> None:
    try:
        ensure_run_transition(RunStatus.COMPLETED, RunStatus.RUNNING)
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid terminal transition")
