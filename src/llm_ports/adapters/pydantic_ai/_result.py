"""The result contract for one pydantic-ai call: usage, who answered, how it ended, cost, warnings.

Each function reads two sources and prefers the wire: the capture's `WireReply` list (what the
OpenAI wire showed, exact about unknowns) when the call ran on `openai_compatible`, else the
`ModelResponse` messages pydantic-ai kept (a fixed `model=`, a replay, a caller's own transport).

Invariants:
    - Unknown is None: nothing here turns an unreported count or an unknown price into 0.
    - `price` never raises; a failing `Pricer` becomes `Cost(None, "unknown", reason="pricer_error")`.

Rationale: ADR-0019.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation, Overflow, Underflow, localcontext
from typing import Any

from llm_ports._extras import extra
from llm_ports.adapters._classify import WireFacts
from llm_ports.adapters.pydantic_ai._capture import WireReply
from llm_ports.adapters.pydantic_ai._classify import openai_finish
from llm_ports.ports import Answered, CallWarning, Cost, Finish, GatewayReader, Pricer, Usage

with extra("pydantic-ai"):
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models import Model
    from pydantic_ai.models.wrapper import WrapperModel

#: Event hook as `observability.EventHook`: `(logging level, event, fields)`.
Emit = Callable[[int, str, Mapping[str, Any]], None]


def responses(messages: Sequence[Any]) -> list[ModelResponse]:
    """The `ModelResponse` messages of a run, in order."""
    return [m for m in messages if isinstance(m, ModelResponse)]


def _framework_usage(r: ModelResponse) -> Usage:
    """One response's usage as pydantic-ai recorded it.

    pydantic-ai flags a response that carried no usage (`unmeasured_requests`), which becomes one
    unmeasured request here. Its sub-buckets are plain counts: a provider that reports no cached
    or reasoning tokens reads as 0 on this path; the wire path keeps those as None.
    """
    u = r.usage
    if getattr(u, "unmeasured_requests", 0):
        return Usage(requests=1, unmeasured=1)
    return Usage(
        input=u.input_tokens,
        output=u.output_tokens,
        cache_read=u.cache_read_tokens,
        cache_write=u.cache_write_tokens,
        reasoning=u.output_reasoning_tokens,
        requests=1,
        details=dict(u.details),
    )


def usage_of(replies: Sequence[WireReply], resps: Sequence[ModelResponse]) -> Usage | None:
    """Sum the call's usage over every request: the wire's replies, else pydantic-ai's responses.

    None when no request answered at all (the wire failed before any reply).
    """
    total: Usage | None = None
    for part in [r.usage for r in replies] if replies else [_framework_usage(r) for r in resps]:
        total = part if total is None else total + part
    return total


def finish_of(facts: WireFacts | None, resps: Sequence[ModelResponse]) -> Finish | None:
    """How the last reply ended: the wire's raw string, else pydantic-ai's provider details or reading."""
    raw = facts.finish_raw if facts is not None else None
    if raw is None and resps:
        last = resps[-1]
        detail = (last.provider_details or {}).get("finish_reason")
        raw = detail if isinstance(detail, str) else last.finish_reason
    unified = openai_finish(raw)
    return None if raw is None or unified is None else Finish(unified=unified, raw=raw)


def refused_in(resps: Sequence[ModelResponse]) -> bool:
    """True when pydantic-ai recorded an explicit refusal on the last response."""
    return bool(resps) and bool((resps[-1].provider_details or {}).get("refusal"))


def answered_of(
    requested: str,
    replies: Sequence[WireReply],
    resps: Sequence[ModelResponse],
    gateway: GatewayReader | None,
) -> Answered:
    """Who answered: the model and id the last response reported, and the gateway's deployment."""
    model = replies[-1].model if replies else (resps[-1].model_name if resps else None)
    rid = replies[-1].response_id if replies else (resps[-1].provider_response_id if resps else None)
    deployment = None
    if gateway is not None and replies:
        deployment, _ = gateway.answered_from_headers(replies[-1].headers)
    return Answered(requested=requested, model=model, deployment=deployment, response_id=rid)


def usage_1x(usage: Usage | None, captured: Mapping[str, int | None] | None) -> dict[str, int | None] | None:
    """The 1.x `Generated.usage` of a returned run, as 1.22.0 computed it.

    The summed counts when either is non-zero; else whatever the capture recorded (a 1.x custom
    capture that saw no usage block recorded nothing, which 1.x reported as None).
    """
    if usage is not None and (usage.input or usage.output):
        return dict(usage.as_mapping())
    return dict(captured or {}) or None


def gateway_cost(gateway: GatewayReader | None, replies: Sequence[WireReply], emit: Emit) -> Decimal | None:
    """The gateway's cost for the call: the sum over every reply, when every reply carried one.

    Never raises: a cost header the reader refuses, a reader that fails, or a sum that overflows
    leaves the cost unknown (None) and is reported as `llm_gateway_header_invalid`; the call it
    prices is unaffected.
    """
    if gateway is None or not replies:
        return None
    total = Decimal(0)
    for r in replies:
        try:
            got = gateway.cost_from_headers(r.headers)
            if got is None:
                return None
            total = _add(total, got)
        except Exception as exc:  # a cost is a report about the call, never a reason to fail it
            level = 30 if isinstance(exc, ValueError | ArithmeticError) else 40
            emit(level, "llm_gateway_header_invalid", {"detail": f"{type(exc).__name__}: {exc}"[:200]})
            return None
    return total


def _add(a: Decimal, b: Decimal) -> Decimal:
    """`a + b`, raising `ArithmeticError` on overflow, underflow or an invalid operand."""
    with localcontext() as ctx:
        ctx.traps[Overflow] = ctx.traps[Underflow] = ctx.traps[InvalidOperation] = True
        return a + b


def price(
    usage: Usage | None,
    answered: Answered,
    *,
    gateway_usd: Decimal | None,
    pricer: Pricer | None,
    emit: Emit,
) -> Cost | None:
    """The call's cost: the gateway's report, else the `Pricer`'s, else None when neither exists."""
    if gateway_usd is not None:
        return Cost(usd=gateway_usd, source="gateway", priced_model=answered.deployment or answered.model)
    if pricer is None:
        return None
    try:
        return pricer.price(usage or Usage(), answered)
    except Exception as exc:  # a custom pricer's bug must not fail the call it prices
        emit(40, "llm_pricer_failed", {"model": answered.model or answered.requested, "error": repr(exc)[:200]})
        return Cost(usd=None, source="unknown", priced_model=answered.model, reason="pricer_error")


def _unwrap(model: Model) -> Model:
    seen = 0
    while isinstance(model, WrapperModel) and seen < 8:
        model = model.wrapped
        seen += 1
    return model


def unsupported(model: Model, settings: Mapping[str, Any]) -> tuple[CallWarning, ...]:
    """One `CallWarning(kind="unsupported")` per set setting the model class ignores.

    Read from pydantic-ai's own declaration (`Model.unsupported_settings`, plus what the model's
    profile adds), before the call, so concurrent calls never share a warnings filter. Unlike
    pydantic-ai's `UnsupportedSettingWarning`, a tuning knob (`temperature`, `seed`) counts too:
    a role's sampling setting a model ignores is a different experiment.
    """
    inner = _unwrap(model)
    declared = getattr(inner, "_unsupported_settings", None)
    ignored = declared() if callable(declared) else inner.unsupported_settings
    name = getattr(inner, "model_name", "") or type(inner).__name__
    return tuple(
        CallWarning(kind="unsupported", setting=s, detail=f"model {name!r} ignores the {s!r} setting")
        for s in sorted(k for k in settings if k in ignored)
    )


__all__: list[str] = []
