"""What the pydantic-ai `Generator` and `Chat` share: client, capture, classification, retry, usage, cost.

`AdapterCore` is the base of `PydanticAIGenerator` and `PydanticAIChat`: the model source, the
wire capture, the classification table, the wire-retry wrapper, the limiter slot, and the result
contract (usage summed over every request, who answered, finish, cost, warnings) are built here
once. `build_agent` makes the call's pydantic-ai `Agent` with the caller's validator as an output
validator. Neither adapter writes any of this twice (ADR-0023).

Invariants:
    - A model, wire or gate failure never raises out of this module's helpers; it is classified.
    - Unknown is None: an unreported count, an unpriced model and a missing finish are None.
    - `asyncio.CancelledError` is never caught.

Rationale: ADR-0019, ADR-0020, ADR-0021, ADR-0023.
"""

from __future__ import annotations

import contextlib
import logging
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, cast

from llm_ports._extras import extra
from llm_ports.adapters._classify import Classifier, WireFacts
from llm_ports.adapters._retry import WireRetry
from llm_ports.adapters.pydantic_ai._capture import CaptureLike, WireReply, capture
from llm_ports.adapters.pydantic_ai._result import (
    answered_of,
    finish_of,
    gateway_cost,
    gateway_deployment,
    price,
    refused_in,
    responses,
    usage_of,
)
from llm_ports.adapters.pydantic_ai._schema import framework_output
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import (
    Answered,
    CallWarning,
    Cost,
    Deadline,
    Failure,
    Finish,
    GatewayReader,
    Generated,
    Limiter,
    Pricer,
    RetryMessage,
    Usage,
    Validator,
    WirePolicy,
    failure,
)

with extra("pydantic-ai"):
    from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import Tool as AITool

    from llm_ports.adapters.pydantic_ai._resilient import ResilientModel, judge_for

#: Models already reported as `llm_cost_unknown` in this process.
_COST_UNKNOWN_SEEN: set[str] = set()
#: The longest traceback an `llm_call_internal_error` event carries, in characters.
TRACEBACK_MAX = 4000
#: The running call's wire-retry state, so `run` (which a subclass may call) shares it.
WIRE: ContextVar[WireRetry | None] = ContextVar("llm_ports_wire", default=None)


class Facts:
    """What one run showed: the wire's replies, the latest facts, the finish, the usage, the responses."""

    __slots__ = ("finish", "replies", "resps", "usage", "wire")

    def __init__(
        self,
        replies: tuple[WireReply, ...],
        wire: WireFacts,
        finish: Finish | None,
        usage: Usage | None,
        resps: Sequence[Any],
    ) -> None:
        self.replies = replies
        self.wire = wire
        self.finish = finish
        self.usage = usage
        self.resps = resps


class AdapterCore[R]:
    """The planes a pydantic-ai port adapter shares; each adapter's docstring states its contract.

    `R` is the request type a `model_for` factory takes (`GenRequest` or `ChatRequest`).

    Args:
        model: A fixed pydantic-ai `Model`.
        model_for: A per-request model factory; the source `stats` and `aclose` reach.
        capture: The capture factory; the default is filled by `openai_compatible`'s response hook.
        gated: Exception types that classify as `gated`.
        wrap: Applied to each model a `model_for` yields, e.g. a `replay.Recorder`.
        pricer: Prices a call the gateway did not price; None leaves `cost` None unless a gateway
            reported one.
        gateway: Reads a gateway's response headers: its cost (before `pricer`) and deployment.
        extra: Classifiers consulted before the shared table, in order.
        on_event: Receives the adapter's events; without one they go to the stdlib logger
            `llm_ports.adapters.pydantic_ai`.
        policy: How a failed model request is retried when the request carries no policy of its
            own; None is `WirePolicy.default()`.
        limiter: Caps concurrent calls per concurrency group (else per model id); None runs every
            call at once.

    Raises:
        ValueError: When neither or both of `model` and `model_for` are given.
    """

    def __init__(
        self,
        model: Model | None = None,
        *,
        model_for: Callable[[R], AbstractAsyncContextManager[Model]] | None = None,
        capture: Callable[[], AbstractContextManager[CaptureLike]] = capture,
        gated: tuple[type[BaseException], ...] = (),
        wrap: Callable[[Model], Model] | None = None,
        pricer: Pricer | None = None,
        gateway: GatewayReader | None = None,
        extra: Sequence[Classifier] = (),
        on_event: EventHook | None = None,
        policy: WirePolicy | None = None,
        limiter: Limiter | None = None,
    ) -> None:
        if (model is None) == (model_for is None):
            raise ValueError(f"{type(self).__name__} needs exactly one of model= / model_for=")
        self._model = model
        self._model_for = model_for
        self._capture = capture
        self._gated = gated
        self._wrap = wrap
        self._pricer = pricer
        self._gateway = gateway
        self._extra = tuple(extra)
        self._emit: EventHook = on_event or log_events("llm_ports.adapters.pydantic_ai")
        self._policy = policy if policy is not None else WirePolicy.default()
        self._limiter = limiter
        self._in_flight = 0
        self._source: object | None = model_for

    def stats(self) -> Mapping[str, int]:
        """`calls_in_flight`, and the client's counts when the model source reports them (`openai_compatible`)."""
        inner = getattr(self._source, "stats", None)
        return {"calls_in_flight": self._in_flight, **(inner() if callable(inner) else {})}

    async def aclose(self) -> None:
        """Close the client the model source holds, when it holds one. Idempotent."""
        close = getattr(self._source, "aclose", None)
        if callable(close):
            await close()

    def _slot(self, group: str) -> AbstractAsyncContextManager[float]:
        if self._limiter is None:
            return contextlib.nullcontext(0.0)
        return self._limiter.slot(group)

    def _wire(self, policy: WirePolicy | None, model: str, deadline_at: float | None, label: str) -> WireRetry:
        return WireRetry(policy or self._policy, model=model, deadline_at=deadline_at, emit=self._emit, label=label)

    def _resilient(
        self, model: Model, policy: WirePolicy | None, name: str, clocks: Deadline, cap: CaptureLike
    ) -> Model:
        """`model` behind the call's wire retry: the running call's, else a fresh one for this run."""
        wire = WIRE.get() or self._wire(policy, name, None, "")

        def bound() -> float | None:
            # The call's own bound is enforced around the whole call; only a tighter one is set here.
            return clocks.attempt_s if clocks.attempt_s is not None and clocks.attempt_s < clocks.total_s else None

        def reset() -> None:
            if hasattr(cap, "last"):
                cap.last = None

        return ResilientModel(
            model, wire, judge_for(cap, extra=self._extra, gated=self._gated), bound=bound, reset=reset
        )

    def _facts(self, cap: CaptureLike, messages: Sequence[Any]) -> Facts:
        """Read what the call showed, preferring the wire's replies over pydantic-ai's messages."""
        replies: tuple[WireReply, ...] = tuple(getattr(cap, "replies", None) or ())
        last: WireFacts | None = getattr(cap, "last", None)
        resps = responses(messages)
        if last is None and cap.finish_reason is not None:  # a 1.x capture: finish only
            last = WireFacts(finish_raw=cap.finish_reason)
        finish = finish_of(last, resps)
        wire = replace(
            last or WireFacts(),
            finish_raw=finish.raw if finish is not None else None,
            refusal=(last is not None and last.refusal) or refused_in(resps),
        )
        usage = usage_of(replies, resps)
        return Facts(replies, wire, finish, usage, resps)

    def _result(
        self,
        requested: str,
        value: Any,
        got: Failure | None,
        retries: int | None,
        facts: Facts,
        cap: CaptureLike,
        warns: tuple[CallWarning, ...],
        problems: tuple[Any, ...],
        *,
        usage_1x: Mapping[str, int | None] | None = None,
    ) -> Generated[Any]:
        """Assemble the result contract for one finished run, and emit its usage and cost events."""
        deployment = gateway_deployment(self._gateway, facts.replies, self._emit)
        answered = answered_of(requested, facts.replies, facts.resps, deployment)
        gw = gateway_cost(self._gateway, facts.replies, self._emit)
        cost = price(facts.usage, answered, gateway_usd=gw, pricer=self._pricer, emit=self._emit)
        self._report_usage(answered, facts.usage, cost)
        res = Generated(
            value=value,
            outcome=got.outcome if got is not None else "ok",
            model=requested,
            retries=retries,
            problems=problems,
            sources=cap.sources,
            usage=usage_1x,
            usage_v2=facts.usage,
            cost=cost,
            finish=facts.finish,
            answered=answered,
            warnings=warns,
            failure=got,
        )
        if usage_1x is None and res.usage is not None:
            # 1.x reported None here (a capture that recorded no usage); `usage_v2` keeps the counts.
            object.__setattr__(res, "usage", None)
        return res

    def _report_usage(self, answered: Answered, usage: Usage | None, cost: Cost | None) -> None:
        model = answered.model or answered.requested
        if usage is not None and usage.unmeasured:
            self._emit(
                logging.INFO,
                "llm_usage_unmeasured",
                {"model": model, "requests": usage.requests, "unmeasured": usage.unmeasured},
            )
        if cost is not None and cost.reason == "unknown_model" and model not in _COST_UNKNOWN_SEEN:
            _COST_UNKNOWN_SEEN.add(model)
            self._emit(logging.WARNING, "llm_cost_unknown", {"model": model, "priced_model": cost.priced_model})

    def _report_internal(self, label: str, model: str, exc: BaseException, got: Failure) -> None:
        if got.kind != "internal":
            return
        tb = "".join(traceback.format_exception(exc))[-TRACEBACK_MAX:]
        self._emit(
            logging.ERROR,
            "llm_call_internal_error",
            {"label": label, "model": model, "reason": got.reason, "traceback": tb},
        )


def build_agent[T](
    model: Model,
    output: type[T],
    *,
    settings: Mapping[str, Any],
    output_retries: int,
    validate: Validator[T] | None,
    retry_message: RetryMessage | None,
    system: str = "",
    instructions: str | None = None,
    tools: Sequence[AITool[None]] = (),
) -> tuple[Agent[None, Any], dict[str, tuple[Any, ...]]]:
    """Build one call's agent, and the slot its validator leaves the last problems in.

    `str` asks for prose; any other output type is sent as a strict json_schema (`NativeOutput`).
    A non-empty problem list from `validate` raises `ModelRetry` while re-asks remain; after the
    last re-ask the value is kept with its problems.
    """
    out_spec: Any = str if output is str else NativeOutput(framework_output(output))
    agent: Agent[None, Any] = Agent(
        model,
        output_type=out_spec,
        system_prompt=system or (),
        instructions=instructions,
        retries={"output": output_retries, "tools": output_retries},
        model_settings=cast(ModelSettings, dict(settings)),
        tools=tuple(tools),
    )
    kept: dict[str, tuple[Any, ...]] = {"problems": ()}
    if validate is None:
        return agent, kept

    @agent.output_validator
    def _check(ctx: RunContext[None], out: Any) -> Any:
        probs = list(validate(out))
        if probs and ctx.retry < output_retries:
            raise ModelRetry(retry_message(probs) if retry_message else "; ".join(map(str, probs)))
        kept["problems"] = tuple(probs)
        return out

    return agent, kept


def call_settings(settings: Mapping[str, Any], clocks: Deadline, *, streamed: bool, cache: bool) -> dict[str, Any]:
    """The run's model settings: the request's, the provider client's per-attempt timeout, the stream bound, the cache hint.

    The deadline itself is enforced by the call; `timeout` here is the provider client's own.
    `cache` asks pydantic-ai to cache the stable prefix only (tool definitions and instructions).
    """
    out = dict(settings)
    out.setdefault("timeout", clocks.attempt_s or clocks.total_s)
    if streamed and clocks.first_event_s is not None:
        out.setdefault("first_event_timeout", clocks.first_event_s)
    if cache:
        out.setdefault("cache", {"messages": False})
    return out


def settled[T](res: Generated[T], wire: WireRetry, total_s: float, waited: float | None) -> Generated[T]:
    """Stamp the call's wire facts on its result; a deadline that ended the call makes it `timeout`."""
    attempts = (*wire.attempts, *res.attempts)
    if wire.expired and res.outcome != "ok":
        last = res.failure.reason if res.failure is not None else "no reply"
        got = failure(
            "timeout",
            phase="total",
            status=res.failure.status if res.failure is not None else None,
            reason=f"deadline: the call's {total_s:g} s ran out ({last})",
        )
        res = replace(res, outcome="unavailable", value=None, failure=got, reason=got.reason)
    out = replace(res, attempts=attempts, stamp={**res.stamp, "limiter_wait_s": waited, "wire_requests": wire.sent})
    if res.usage is None and out.usage is not None:
        object.__setattr__(out, "usage", None)  # 1.x left `usage` None here; `replace` re-derives it
    return out


__all__: list[str] = []
