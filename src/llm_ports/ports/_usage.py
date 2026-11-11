"""What a result says it measured, what it cost, who answered, how it ended and what was dropped.

These are the value types a `Generated`, `Embedded` or `Reranked` carries beside its value. Each
is frozen and holds no behaviour beyond arithmetic over its own fields.

Invariants:
    - Pure: no I/O.
    - Unknown is None at every level: a count the provider did not report is None, a reported 0
      stays 0, and an unknown cost is `Cost(usd=None)`, never 0.
    - Token buckets are inclusive: `input` includes the cached tokens, `output` the reasoning
      tokens. An adapter normalises a provider that reports them apart.

Rationale: ADR-0019.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from llm_ports.ports._failure import Failure

#: How a reply ended, in one closed vocabulary across providers.
UnifiedFinish = Literal["stop", "length", "content_filter", "tool_call", "error", "interrupted", "other"]
#: Where a cost came from: a gateway's own report, a price table, the provider's response, or nowhere.
CostSource = Literal["gateway", "price_table", "provider", "unknown"]
#: What happened to a setting on its way to the model.
WarningKind = Literal["unsupported", "clamped", "deprecated", "other"]


def _plus(a: int | None, b: int | None) -> int | None:
    """Add two counts where None is unknown: None + n = n, None + None = None."""
    if a is None:
        return b
    if b is None:
        return a
    return a + b


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts summed over every wire request of one call.

    Attributes:
        input: All input tokens, cached included; None when no request reported it.
        output: All output tokens, reasoning included; None when no request reported it.
        cache_read: The part of `input` served from a provider cache.
        cache_write: The part of `input` written to a provider cache.
        reasoning: The part of `output` spent reasoning.
        requests: Wire requests folded in (output re-asks, tool steps, retries that returned a body).
        unmeasured: Of those, how many reported no usage at all.
        details: Provider extras under their own keys, summed key by key.

    Invariants:
        - `cache_read <= input` and `reasoning <= output` whenever both sides are known.

    Rationale: ADR-0019.
    """

    input: int | None = None
    output: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None
    reasoning: int | None = None
    requests: int = 0
    unmeasured: int = 0
    details: Mapping[str, int] = field(default_factory=dict)

    def __add__(self, other: Usage) -> Usage:
        """Return the sum of two usages; a side unknown on both stays None."""
        details = dict(self.details)
        for k, v in other.details.items():
            details[k] = details.get(k, 0) + v
        return Usage(
            input=_plus(self.input, other.input),
            output=_plus(self.output, other.output),
            cache_read=_plus(self.cache_read, other.cache_read),
            cache_write=_plus(self.cache_write, other.cache_write),
            reasoning=_plus(self.reasoning, other.reasoning),
            requests=self.requests + other.requests,
            unmeasured=self.unmeasured + other.unmeasured,
            details=details,
        )

    @property
    def complete(self) -> bool:
        """True when at least one request was made, every one reported usage, and both totals are known."""
        return self.requests > 0 and self.unmeasured == 0 and self.input is not None and self.output is not None

    def as_mapping(self) -> Mapping[str, int | None]:
        """Return the 1.x `{"input", "output"}` shape of `Generated.usage`."""
        return {"input": self.input, "output": self.output}


@dataclass(frozen=True, slots=True)
class Cost:
    """What a call cost, in US dollars, and where that number came from.

    Attributes:
        usd: The cost; None when unknown, never 0 for unknown.
        source: Where `usd` came from.
        priced_model: The model reference the price was looked up under.
        reason: Why the cost is unknown: `no_usage`, `unknown_model`, `no_pricer` or `pricer_error`.

    Rationale: ADR-0019.
    """

    usd: Decimal | None
    source: CostSource
    priced_model: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class Finish:
    """How the last reply of a call ended.

    Attributes:
        unified: The closed-vocabulary reading of `raw`; `other` for a value no mapping knows.
        raw: The provider's own string, unmodified.

    Rationale: ADR-0019.
    """

    unified: UnifiedFinish
    raw: str | None


@dataclass(frozen=True, slots=True)
class Answered:
    """Who produced a result.

    Attributes:
        requested: The model id the request carried (a route or an alias is fine).
        model: The model name the response reported; None when it reported none. Never an echo
            of `requested`.
        provider: The provider that served it, when the caller's provider table names it.
        tier: The tier that produced this result, when the caller's routing names it.
        deployment: The gateway deployment id, when a gateway reported one.
        response_id: The response's own id, when it had one.

    Rationale: ADR-0019.
    """

    requested: str
    model: str | None = None
    provider: str | None = None
    tier: str | None = None
    deployment: str | None = None
    response_id: str | None = None


@dataclass(frozen=True, slots=True)
class Attempt:
    """One wire request or fallback step that did not answer, and the wait that followed it.

    Attributes:
        n: The attempt's number, from 1.
        tier: The tier it ran on, when known.
        model: The model it asked.
        failure: How it failed.
        waited_s: The backoff slept before the next attempt, in seconds.
        usage: What it consumed, when it reported anything.

    Rationale: ADR-0019.
    """

    n: int
    tier: str | None
    model: str
    failure: Failure
    waited_s: float
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class CallWarning:
    """A setting that was dropped or altered on its way to the model.

    Attributes:
        kind: What happened to it.
        setting: The setting's name.
        detail: One line saying what happened.

    Rationale: ADR-0019.
    """

    kind: WarningKind
    setting: str
    detail: str


__all__ = [
    "Answered",
    "Attempt",
    "CallWarning",
    "Cost",
    "CostSource",
    "Finish",
    "UnifiedFinish",
    "Usage",
    "WarningKind",
]
