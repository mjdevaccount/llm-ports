"""adapters.litellm_proxy: what a LiteLLM proxy says about a call, read from headers and bodies.

Failures acceptance test 6 (a gateway budget 429 is `quota`, a parallel-limit 429 stays
`rate_limited` with its wait), both ways, plus the header readers the result contract uses.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from llm_ports.adapters import litellm_proxy as lp
from llm_ports.adapters._classify import WireFacts, classify_core
from llm_ports.ports import GatewayReader


def test_the_module_is_a_gateway_reader() -> None:
    assert isinstance(lp, GatewayReader)


def test_cost_reads_the_header_and_none_when_absent() -> None:
    assert lp.cost_from_headers({"x-litellm-response-cost": "0.0123"}) == Decimal("0.0123")
    assert lp.cost_from_headers({"X-LiteLLM-Response-Cost": " 0 "}) == Decimal(0), "a reported 0 stays 0"
    assert lp.cost_from_headers({}) is None
    assert lp.cost_from_headers({"x-litellm-response-cost": ""}) is None


@pytest.mark.parametrize("raw", ["abc", "nan", "inf", "-0.5", "1e9999999", "1e7", "1e-9999999", "9e-13"])
def test_a_cost_header_that_is_not_a_plausible_dollar_amount_raises(raw: str) -> None:
    with pytest.raises(ValueError, match="x-litellm-response-cost"):
        lp.cost_from_headers({"x-litellm-response-cost": raw})


@pytest.mark.parametrize(
    ("raw", "usd"),
    [("-0", Decimal(0)), ("0E-50", Decimal(0)), ("1e-12", Decimal("1e-12")), ("9999999.99", Decimal("9999999.99"))],
)
def test_a_cost_header_at_the_edges_of_the_plausible_band_is_kept(raw: str, usd: Decimal) -> None:
    got = lp.cost_from_headers({"x-litellm-response-cost": raw})
    assert got == usd and got is not None and not got.is_signed(), "a zero is never a signed zero"
    assert (lp.COST_MIN_USD, lp.COST_MAX_USD) == (Decimal("1e-12"), Decimal("1e7"))


def test_answered_and_attempts_read_their_headers() -> None:
    h = {
        "x-litellm-model-id": "dep-7",
        "x-litellm-model-api-base": "https://upstream.example/v1",
        "x-litellm-attempted-retries": "2",
        "x-litellm-attempted-fallbacks": "x",
    }
    assert lp.answered_from_headers(h) == ("dep-7", "https://upstream.example/v1")
    assert lp.attempts_from_headers(h) == (2, None), "an unreadable count is unknown"
    assert lp.answered_from_headers({}) == (None, None)
    assert lp.attempts_from_headers({}) == (None, None)


def test_retry_hint_reads_hours_minutes_and_seconds() -> None:
    assert lp.retry_hint_s("Please retry in 8h26m23.4s.") == pytest.approx(8 * 3600 + 26 * 60 + 23.4)
    assert lp.retry_hint_s("retry in 12s") == 12.0
    assert lp.retry_hint_s("retry in 3m") == 180.0
    assert lp.retry_hint_s("try again later") is None


# Recorded 429 bodies (trimmed).
BUDGET = WireFacts(
    status=429,
    error_type="budget_exceeded",
    error_message="Budget has been exceeded! Current cost: 5.02, Max budget: 5.0",
)
PER_DAY = WireFacts(
    status=429, error_message="Quota exceeded for metric: generate_content_free_tier_requests, limit per day"
)
LONG_HINT = WireFacts(status=429, error_message="You exceeded your current quota. Please retry in 8h26m23s.")
PARALLEL = WireFacts(
    status=429,
    headers={"retry-after": "2"},
    error_message="Max parallel request limit reached. Hit limit for api_key: sk-...",
)
SHORT_HINT = WireFacts(status=429, error_message="Rate limit reached. Please retry in 20s.")


@pytest.mark.parametrize("facts", [BUDGET, PER_DAY, LONG_HINT], ids=["budget", "per_day", "long_hint"])
def test_a_spent_budget_or_quota_is_quota_and_not_retryable(facts: WireFacts) -> None:
    got = classify_core(None, facts, extra=(lp.classify_gateway,))
    assert got is not None and (got.kind, got.outcome, got.retryable, got.retry_after_s) == (
        "quota",
        "unavailable",
        False,
        None,
    )


@pytest.mark.parametrize("facts", [PARALLEL, SHORT_HINT], ids=["parallel_limit", "short_hint"])
def test_a_parallel_limit_429_stays_rate_limited_with_its_wait(facts: WireFacts) -> None:
    assert lp.classify_gateway(None, facts) is None
    got = classify_core(None, facts, extra=(lp.classify_gateway,))
    assert got is not None and (got.kind, got.retryable) == ("rate_limited", True)


def test_the_parallel_limit_keeps_the_server_wait() -> None:
    got = classify_core(None, PARALLEL, extra=(lp.classify_gateway,))
    assert got is not None and got.retry_after_s == 2.0


def test_without_the_gateway_classifier_a_budget_429_is_just_rate_limited() -> None:
    # MUST FIRE: the gateway's knowledge is what turns a spent budget into `quota`.
    got = classify_core(None, BUDGET)
    assert got is not None and got.kind == "rate_limited"


def test_a_budget_named_only_in_the_exception_body_is_quota() -> None:
    exc = ModelHTTPError(429, "m", body={"error": {"message": "Budget exceeded for team"}})
    got = classify_core(exc, WireFacts(), extra=(lp.classify_gateway,))
    assert got is not None and got.kind == "quota"
    text_body = ModelHTTPError(429, "m", body="ExceededBudget: key over budget")
    got = classify_core(text_body, WireFacts(), extra=(lp.classify_gateway,))
    assert got is not None and got.kind == "quota"


def test_a_non_429_is_left_to_the_table() -> None:
    assert lp.classify_gateway(None, WireFacts(status=400, error_message="budget field invalid")) is None
    assert lp.classify_gateway(None, WireFacts()) is None
