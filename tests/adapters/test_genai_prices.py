"""adapters.genai_prices: the Pricer over genai-prices' table, with caller overrides.

Result-contract acceptance test 4 at the pricer (unknown model both ways) and invariant 5 (a
self-hosted model costs what the caller says, never free by default).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from llm_ports.adapters.genai_prices import GenaiPricesPricer
from llm_ports.ports import Answered, Pricer, Usage

USAGE = Usage(input=1000, output=100, cache_read=200, requests=1)


def test_it_is_a_pricer() -> None:
    assert isinstance(GenaiPricesPricer(), Pricer)


def test_a_known_model_is_priced_from_the_table_with_cached_input_cheaper() -> None:
    got = GenaiPricesPricer().price(USAGE, Answered(requested="mini", model="gpt-4o-mini"))
    assert got.source == "price_table" and got.reason is None and got.priced_model == "gpt-4o-mini"
    assert isinstance(got.usd, Decimal) and got.usd > 0
    uncached = GenaiPricesPricer().price(
        Usage(input=1000, output=100, requests=1), Answered(requested="mini", model="gpt-4o-mini")
    )
    assert uncached.usd is not None and got.usd < uncached.usd, "input is inclusive; the cached part is cheaper"


def test_an_unknown_model_is_unknown_never_free() -> None:
    got = GenaiPricesPricer().price(USAGE, Answered(requested="local-8b"))
    assert (got.usd, got.source, got.reason) == (None, "unknown", "unknown_model")


def test_unmeasured_usage_is_not_priced() -> None:
    partial = USAGE + Usage(requests=1, unmeasured=1)
    got = GenaiPricesPricer().price(partial, Answered(requested="gpt-4o-mini"))
    assert (got.usd, got.reason) == (None, "no_usage")


def test_an_override_wins_per_request_under_requested_then_answered() -> None:
    p = GenaiPricesPricer(overrides={"local-8b": Decimal(0), "gpt-4o-mini": Decimal("0.01")})
    local = p.price(Usage(requests=3, unmeasured=3), Answered(requested="local-8b"))
    assert (local.usd, local.source, local.priced_model) == (Decimal(0), "price_table", "local-8b")
    by_answer = p.price(Usage(input=1, output=1, requests=2), Answered(requested="alias", model="gpt-4o-mini"))
    assert by_answer.usd == Decimal("0.02")


def test_the_provider_hook_narrows_the_lookup() -> None:
    seen: list[Answered] = []

    def provider_for(a: Answered) -> str | None:
        seen.append(a)
        return "openai"

    got = GenaiPricesPricer(provider_for=provider_for).price(USAGE, Answered(requested="gpt-4o-mini"))
    assert got.usd is not None and len(seen) == 1


def test_a_failing_provider_hook_is_a_pricer_error_not_a_raise() -> None:
    def broken(_a: Answered) -> str | None:
        raise RuntimeError("bug")

    got = GenaiPricesPricer(provider_for=broken).price(USAGE, Answered(requested="gpt-4o-mini"))
    assert (got.usd, got.reason) == (None, "pricer_error")


@pytest.mark.parametrize("provider", ["no-such-provider"])
def test_an_unknown_provider_is_unknown_model(provider: str) -> None:
    got = GenaiPricesPricer(provider_for=lambda _a: provider).price(USAGE, Answered(requested="gpt-4o-mini"))
    assert got.usd is None and got.reason == "unknown_model"
