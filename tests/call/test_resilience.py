"""RetryPolicy and FailureBudget: the bounds that turn a long run's "skip and go on" into "stop"."""

from __future__ import annotations

import pytest

from llm_ports.call.resilience import FailureBudget, RetryPolicy


def test_a_run_of_failures_in_a_row_stops_the_run() -> None:
    b = FailureBudget(max_consecutive=3)
    for _ in range(2):
        b.record(False)
    assert not b.exhausted
    b.record(False)
    assert b.exhausted and "3 failures in a row" in (b.reason or "")


def test_a_success_resets_the_row_but_not_the_rate() -> None:
    b = FailureBudget(max_consecutive=2, min_attempts=100)
    for ok in (False, True, False, True, False):
        b.record(ok)
    assert not b.exhausted and b.failures == 3 and b.consecutive == 1 and b.rate == pytest.approx(0.6)


def test_the_rate_is_judged_only_after_min_attempts() -> None:
    b = FailureBudget(max_rate=0.1, min_attempts=10, max_consecutive=99)
    b.record(False)
    b.record(True)
    assert not b.exhausted  # 50% of 2: too few tried to judge
    for _ in range(8):
        b.record(True)
    assert not b.exhausted  # 1 of 10 = 10%, not above 10%
    b.record(False)
    assert b.exhausted and "failure rate" in (b.reason or "")


def test_isolated_failures_under_the_rate_never_stop_the_run() -> None:
    b = FailureBudget()
    for i in range(1000):
        b.record(i % 50 != 0)  # 2%, never two in a row
    assert not b.exhausted and b.failures == 20


@pytest.mark.parametrize("kw", [{"max_rate": 1.5}, {"max_rate": -0.1}, {"min_attempts": 0}, {"max_consecutive": 0}])
def test_budget_refuses_bounds_that_cannot_hold(kw: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        FailureBudget(**kw)  # type: ignore[arg-type]


@pytest.mark.parametrize("kw", [{"attempts": 0}, {"wait_initial_s": -1.0}, {"wait_initial_s": 10.0, "wait_max_s": 5.0}])
def test_policy_refuses_bounds_that_cannot_hold(kw: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kw)  # type: ignore[arg-type]


# -- LP4: the wire policy, the deadline and the in-process limiter as values ---------------------


def test_a_limiter_from_model_facts_takes_the_smallest_count_per_group() -> None:
    from llm_ports.call.resilience import LocalLimiter

    lim = LocalLimiter.for_models({"chat": ("gpu", 2), "web": ("gpu", 1), "solo": (None, 3), "free": ("gpu", None)})
    assert (lim.size("gpu"), lim.size("solo"), lim.size("cloud")) == (1, 3, None)


def test_a_deadline_bounds_one_request_by_attempt_s_and_never_past_the_end() -> None:
    from llm_ports.ports import Deadline

    idle = Deadline(total_s=10.0, attempt_s=3.0)
    assert idle.remaining(99.0) == 10.0 and idle.attempt_bound(99.0) == 3.0
    clocks = idle.start(100.0)
    assert clocks.at == 110.0 and clocks.attempt_bound(101.0) == 3.0 and clocks.attempt_bound(108.5) == 1.5
    assert Deadline(total_s=5.0).start(0.0).attempt_bound(4.0) == 1.0 and clocks.remaining(200.0) == 0.0
    with pytest.raises(ValueError, match="start"):
        _ = idle.at
