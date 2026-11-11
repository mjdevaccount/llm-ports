"""A `Pricer` over genai-prices: what a call cost, from the model that answered and its usage.

genai-prices ships a dated price table for the hosted providers and computes a call's price as
a `Decimal` from its token counts (input inclusive of cached tokens, as `Usage` is). This module
adapts it to `ports.Pricer` and adds the two things the table cannot know:

- a self-hosted model costs what the caller says: `overrides` maps a model id to a per-request
  price (often `Decimal(0)` for a local box, stated, never assumed);
- a model the table does not know is `Cost(None, "unknown", reason="unknown_model")`, never free.

Invariants:
    - `price` never raises: a failing lookup is `Cost(None, "unknown", reason="pricer_error")`.
    - Unknown is None: a call with unmeasured usage is `reason="no_usage"`, never priced from a
      partial count.
    - No I/O: the bundled table is read; genai-prices' own `UpdatePrices` refresh is the
      caller's choice.

Example:
    gen = PydanticAIGenerator(model_for=..., pricer=GenaiPricesPricer(overrides={"local-8b": Decimal(0)}))

Rationale: ADR-0019.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal

from llm_ports._extras import extra
from llm_ports.ports import Answered, Cost, Usage

with extra("prices"):
    import genai_prices


class GenaiPricesPricer:
    """Price a call from genai-prices' table, with caller overrides for models it cannot know.

    Args:
        provider_for: The genai-prices provider id for a call (e.g. `"openai"`), or None to let
            genai-prices match the model id across providers.
        overrides: A per-request price in US dollars by model id, looked up under the requested
            id first and then the answering model's; it wins over the table.

    Rationale: ADR-0019.
    """

    def __init__(
        self,
        *,
        provider_for: Callable[[Answered], str | None] | None = None,
        overrides: Mapping[str, Decimal] | None = None,
    ) -> None:
        self._provider_for = provider_for
        self._overrides = dict(overrides or {})

    def price(self, usage: Usage, answered: Answered) -> Cost:
        """Return what the call cost; `usd` is None when it cannot be known. Never raises."""
        model = answered.model or answered.requested
        for key in (answered.requested, answered.model):
            if key is not None and key in self._overrides:
                return Cost(usd=self._overrides[key] * usage.requests, source="price_table", priced_model=key)
        if not usage.complete or usage.input is None or usage.output is None:
            return Cost(usd=None, source="unknown", priced_model=model, reason="no_usage")
        try:
            provider = self._provider_for(answered) if self._provider_for is not None else None
            got = genai_prices.calc_price(
                genai_prices.Usage(
                    input_tokens=usage.input,
                    output_tokens=usage.output,
                    cache_read_tokens=usage.cache_read,
                    cache_write_tokens=usage.cache_write,
                ),
                model,
                provider_id=provider,
            )
        except LookupError:
            return Cost(usd=None, source="unknown", priced_model=model, reason="unknown_model")
        except Exception:  # a table or provider-hook bug must not fail the call it prices
            return Cost(usd=None, source="unknown", priced_model=model, reason="pricer_error")
        return Cost(usd=Decimal(got.total_price), source="price_table", priced_model=got.model.id)


__all__ = ["GenaiPricesPricer"]
