from __future__ import annotations

import pytest

from finetuneharness.executor.policy import RetryPolicy


def test_first_attempt_no_delay() -> None:
    policy = RetryPolicy(max_attempts=3, base_delay_seconds=1.0, jitter=False)
    assert policy.delay_for_attempt(0) == 0.0


def test_no_jitter_exponential_growth() -> None:
    policy = RetryPolicy(max_attempts=5, base_delay_seconds=1.0, max_delay_seconds=60.0, jitter=False)
    assert policy.delay_for_attempt(1) == pytest.approx(1.0)   # 1 * 2^0
    assert policy.delay_for_attempt(2) == pytest.approx(2.0)   # 1 * 2^1
    assert policy.delay_for_attempt(3) == pytest.approx(4.0)   # 1 * 2^2
    assert policy.delay_for_attempt(4) == pytest.approx(8.0)   # 1 * 2^3


def test_max_delay_cap() -> None:
    policy = RetryPolicy(max_attempts=10, base_delay_seconds=1.0, max_delay_seconds=5.0, jitter=False)
    assert policy.delay_for_attempt(5) == pytest.approx(5.0)
    assert policy.delay_for_attempt(9) == pytest.approx(5.0)


def test_jitter_within_bounds() -> None:
    policy = RetryPolicy(max_attempts=5, base_delay_seconds=2.0, max_delay_seconds=30.0, jitter=True)
    for _ in range(50):
        delay = policy.delay_for_attempt(1)
        assert 0.0 <= delay <= 2.0


def test_default_policy_no_retries() -> None:
    policy = RetryPolicy()
    assert policy.max_attempts == 1
    assert policy.delay_for_attempt(0) == 0.0
