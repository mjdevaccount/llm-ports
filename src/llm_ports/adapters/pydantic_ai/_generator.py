"""The `Generator` port implemented on pydantic-ai.

One `Agent` runs per call. `NativeOutput(T)` sends T's schema as a strict json_schema (the grammar
a llama.cpp server compiles; the structured-output mode OpenAI honours) and validates the reply
into T; `str` asks for prose. The caller's `validate` runs as an `output_validator`: a non-empty
problem list raises `ModelRetry(retry_message(problems))` while re-asks remain; after the last
re-ask the value is kept with its problems, and the caller prunes or drops it. Retries, schema
validation and re-asks are pydantic-ai's; this module does not re-implement them.

The model source is the caller's choice:

- `model=`: a fixed pydantic-ai `Model` (tests pass a `FunctionModel`).
- `model_for=`: `(GenRequest) -> async context manager yielding a Model`, called per request, so
  a caller can put the model on its own transport (admission control, leases, a work queue).
- `openai_compatible(base_url, api_key)` builds a `model_for` for any OpenAI-wire endpoint (a
  LiteLLM gateway, vLLM, llama.cpp's server, OpenAI itself), on one reused client per event loop.

Resilience (ADR-0021): every model request goes through the call's `WirePolicy` (`policy=`, or the
request's own), keyed on `Failure.kind`; this loop is the only retry owner. The whole call (slot
wait, every request, re-ask and backoff) runs under the request's `Deadline`: its expiry is a
`timeout` result with `phase="total"` and the attempts so far, never an exception. A `limiter=`
slot is taken inside the deadline and outside the retry loop, and held across retries.

`capture=` is a context-manager factory whose value (a `CaptureLike`) carries what the wire
showed: every reply's usage, finish reason, model, id and headers, and the latest response's
status and error code (pydantic-ai drops non-standard keys and keeps no headers). The default
capture is filled by the response hook on the httpx2 client `openai_compatible` builds; a caller
on its own transport passes its own factory, and a capture without per-reply detail falls back to
pydantic-ai's own messages.

Every result carries the result contract (ADR-0019): `usage_v2` summed over every request,
`answered` (the model and id the response reported, the gateway's deployment), `finish`, `cost`
(the gateway's report, else the `pricer=`, else None), and `warnings` (settings the model class
ignores). A failure carries a `Failure` from the shared classification table (ADR-0020).

Invariants:
    - A model, wire or gate failure never raises; it returns a `Generated` with a named outcome.
    - `asyncio.CancelledError` is never caught: a cancelled call propagates, and its limiter slot
      is released on the way out.
    - A length-cut reply is `truncated` even when it parsed; its value is kept on the result.
      `GenRequest.accept_truncated` returns it as `ok` with a `CallWarning`.
    - An exception the classification table does not know (a bug in the caller's `validate`, a
      hook, this adapter) is `internal` under `request_fault`, and is reported as
      `llm_call_internal_error` with its traceback; it never reads as a model outage.
    - Exception types passed as `gated=` (a refusal raised by the caller's transport) map to
      `gated`.
    - Unknown is None: an unreported count, an unpriced model and a missing finish reason are
      None, never 0 or `stop`.

Rationale: ADR-0001, ADR-0019, ADR-0020, ADR-0021; the httpx2 client, ADR-0009.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, cast

from llm_ports._extras import extra
from llm_ports._headers import HEADER_VALUE_RULE, header_value_ok
from llm_ports.adapters._classify import Classifier, WireFacts
from llm_ports.adapters._retry import WireRetry
from llm_ports.adapters.pydantic_ai._capture import CaptureLike, WireReply, capture, record_response
from llm_ports.adapters.pydantic_ai._classify import classify, classify_failure
from llm_ports.adapters.pydantic_ai._result import (
    answered_of,
    finish_of,
    gateway_cost,
    gateway_deployment,
    price,
    refused_in,
    responses,
    unsupported,
    usage_1x,
    usage_of,
)
from llm_ports.adapters.pydantic_ai._schema import framework_output
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import (
    Answered,
    CallWarning,
    Cost,
    Failure,
    Finish,
    GatewayReader,
    Generated,
    GenRequest,
    Limiter,
    Pricer,
    SlotWaitExceeded,
    Usage,
    WirePolicy,
    failure,
)

with extra("pydantic-ai"):
    import httpx
    import httpx2
    from pydantic_ai import Agent, BinaryContent, ModelRetry, NativeOutput, RunContext, capture_run_messages
    from pydantic_ai.models import Model
    from pydantic_ai.profiles import ModelProfileSpec
    from pydantic_ai.settings import ModelSettings

    from llm_ports.adapters.pydantic_ai._client import OpenAICompatible
    from llm_ports.adapters.pydantic_ai._resilient import ResilientModel, judge_for

ModelFor = Callable[[GenRequest], AbstractAsyncContextManager[Model]]

#: The 1.x names of the capture and its hook, kept for callers that reached for them.
_capture = capture
_record_response = record_response
#: Models already reported as `llm_cost_unknown` in this process.
_COST_UNKNOWN_SEEN: set[str] = set()
#: The longest traceback an `llm_call_internal_error` event carries, in characters.
TRACEBACK_MAX = 4000
#: The running call's wire-retry state, so `run` (which a subclass may call) shares it.
_WIRE: ContextVar[WireRetry | None] = ContextVar("llm_ports_wire", default=None)


def openai_compatible(
    base_url: str,
    api_key: str | None = None,
    *,
    max_retries: int = 0,
    transport: httpx2.AsyncBaseTransport | httpx.AsyncBaseTransport | None = None,
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
    client: httpx2.AsyncClient | None = None,
) -> OpenAICompatible:
    """Build a `model_for` over any OpenAI-wire chat-completions endpoint.

    One `httpx2.AsyncClient` (the client pydantic-ai's OpenAI provider takes) is built per event
    loop, with explicit pool limits, a 5 s connect bound and the capture hook, and reused by every
    call; the per-request bound is passed per call. `aclose()` closes it.

    Args:
        base_url: The URL up to where `/chat/completions` starts, e.g. `http://localhost:4000/v1`.
        api_key: The bearer key; None sends a placeholder, for servers that need no key.
        max_retries: The OpenAI SDK's own retry count. Defaults to 0, so retries are not stacked
            on those of a gateway in front.
        transport: Replaces httpx2's own transport (a caller's admitting or counting transport, or
            a `MockTransport` in a test). An httpx2 transport is used as is; a legacy httpx one is
            carried across a bridge, so the provider still gets an httpx2 client.
        profile: A pydantic-ai `ModelProfileSpec`: a partial `ModelProfile` merged over the one
            the provider resolves for the model, or a function from that resolved profile to the
            profile to use. E.g. it moves an agent whose output type the caller does not construct
            onto native structured output. None keeps the provider's.
        settings: Model-level `ModelSettings`, applied under each run's own settings (pydantic-ai
            merges run settings over model settings, key by key at the top level). None sets none.
        client: A caller's own `httpx2.AsyncClient`, used as given and never closed here (add
            `record_response` to its response hooks to keep the wire capture).

    Returns:
        A `ModelFor` (`OpenAICompatible`) that yields an `OpenAIChatModel` for `req.model`, carrying
        `profile` and `settings`, with `stats()` and `aclose()`.

    Raises:
        ValueError: `api_key` holds whitespace, a control or a non-ASCII character (the HTTP stack
            would quote it, value and all, when it refused the header). The value is not shown.

    Rationale: ADR-0009.
    """
    if api_key is not None and not header_value_ok(api_key):
        raise ValueError(f"api_key {HEADER_VALUE_RULE}")
    return OpenAICompatible(
        base_url,
        api_key,
        max_retries=max_retries,
        transport=transport,
        profile=profile,
        settings=settings,
        client=client,
    )


def _prompt(req: GenRequest) -> Any:
    """Return the user prompt: the text alone, or the text followed by the request's images."""
    if not req.images:
        return req.prompt
    return [req.prompt, *(BinaryContent(data=b, media_type=m) for b, m in req.images)]


class PydanticAIGenerator:
    """The `Generator` port on pydantic-ai; the module docstring states the contract.

    Args:
        model: A fixed pydantic-ai `Model`.
        model_for: A per-request model factory.
        capture: The capture factory; the default is filled by `openai_compatible`'s response hook.
        gated: Exception types that classify as `gated`.
        wrap: Applied to each model that `model_for` yields (not to a fixed `model`), e.g. a
            `replay.Recorder`.
        pricer: Prices a call the gateway did not price (e.g. `adapters.genai_prices.GenaiPricesPricer`);
            None leaves `cost` None unless a gateway reported one.
        gateway: Reads a gateway's response headers (e.g. the module `adapters.litellm_proxy`):
            its cost, which takes precedence over `pricer`, and its deployment id.
        extra: Classifiers consulted before the shared table, in order (e.g.
            `adapters.litellm_proxy.classify_gateway`).
        on_event: Receives `llm_call_internal_error`, `llm_cost_unknown`, `llm_usage_unmeasured`,
            `llm_pricer_failed`, `llm_gateway_header_invalid`, `llm_wire_retry` and
            `llm_deadline_expired`; without one they go to the stdlib logger
            `llm_ports.adapters.pydantic_ai`.
        policy: How a failed model request is retried when the request carries no policy of its
            own; None is `WirePolicy.default()`. `WirePolicy.none()` sends every request once.
        limiter: Caps concurrent calls per `GenRequest.concurrency_group` (else per model id);
            None runs every call at once.

    Raises:
        ValueError: When neither or both of `model` and `model_for` are given.
    """

    def __init__(
        self,
        model: Model | None = None,
        *,
        model_for: ModelFor | None = None,
        capture: Callable[[], AbstractContextManager[CaptureLike]] = _capture,
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
            raise ValueError("PydanticAIGenerator needs exactly one of model= / model_for=")
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

    def stats(self) -> Mapping[str, int]:
        """`calls_in_flight`, and the client's counts when `model_for` reports them (`openai_compatible`)."""
        inner = getattr(self._model_for, "stats", None)
        return {"calls_in_flight": self._in_flight, **(inner() if callable(inner) else {})}

    async def aclose(self) -> None:
        """Close the client `model_for` holds, when it holds one. Idempotent."""
        close = getattr(self._model_for, "aclose", None)
        if callable(close):
            await close()

    async def generate[T](
        self,
        req: GenRequest,
        output: type[T],
        *,
        validate: Callable[[T], Sequence[Any]] | None = None,
        retry_message: Callable[[Sequence[Any]], str] | None = None,
    ) -> Generated[T]:
        """Run one generation; a model, wire or gate failure returns a named outcome, never raises.

        Args:
            req: The call's prompt, model and limits.
            output: `str` for prose, a pydantic model, or a `JsonOutput` subclass.
            validate: Returns the problems in a parsed value. A non-empty list triggers a re-ask
                while `req.output_retries` remain; after that the value is kept with its problems.
                An exception it raises is `internal`, never a model outage.
            retry_message: Renders the problems into the re-ask message; by default they are
                joined with "; ".

        Returns:
            A `Generated` whose `outcome` is `ok` or a named failure. Its `attempts` hold every
            retried request; `stamp["limiter_wait_s"]` the seconds waited for a slot (None without
            a limiter); `stamp["wire_requests"]` the requests sent.

        Raises:
            asyncio.CancelledError: The caller cancelled; the slot is released first.
        """
        loop = asyncio.get_running_loop()
        clocks = req.clocks.start(loop.time())
        req = replace(req, deadline=clocks)
        wire = WireRetry(
            req.policy or self._policy, model=req.model, deadline_at=clocks.at, emit=self._emit, label=req.label
        )
        token = _WIRE.set(wire)
        self._in_flight += 1
        waited: float | None = None
        try:
            async with asyncio.timeout_at(clocks.at):
                async with self._slot(req) as waited_s:
                    waited = waited_s
                    res = await self._generate(req, output, validate, retry_message)
        except TimeoutError:
            wire.expire("slot" if waited is None and self._limiter is not None else "request")
            res = Generated(value=None, outcome="unavailable", model=req.model, answered=Answered(requested=req.model))
        except SlotWaitExceeded as exc:
            got = failure("rate_limited", provider_code="local_limiter", reason=str(exc))
            res = Generated(
                value=None, outcome=got.outcome, model=req.model, answered=Answered(requested=req.model), failure=got
            )
        finally:
            _WIRE.reset(token)
            self._in_flight -= 1
        return _settled(res, wire, clocks.total_s, waited if self._limiter is not None else None)

    def _slot(self, req: GenRequest) -> AbstractAsyncContextManager[float]:
        if self._limiter is None:
            return contextlib.nullcontext(0.0)
        return self._limiter.slot(req.concurrency_group or req.model)

    async def _generate[T](
        self,
        req: GenRequest,
        output: type[T],
        validate: Callable[[T], Sequence[Any]] | None,
        retry_message: Callable[[Sequence[Any]], str] | None,
    ) -> Generated[T]:
        if self._model is not None:
            return await self.run(self._model, req, output, validate, retry_message)
        assert self._model_for is not None
        try:
            async with self._model_for(req) as model:
                if self._wrap is not None:
                    model = self._wrap(model)
                return await self.run(model, req, output, validate, retry_message)
        except Exception as exc:  # building the wire failed (bad URL, a refusing transport)
            got = classify_failure(exc, WireFacts(), extra=self._extra, gated=self._gated)
            assert got is not None  # an exception always classifies
            self._report_internal(req, exc, got)
            return Generated(
                value=None,
                outcome=got.outcome,
                model=req.model,
                retries=None,
                answered=Answered(requested=req.model),
                failure=got,
            )

    async def run[T](
        self,
        model: Model,
        req: GenRequest,
        output: type[T],
        validate: Callable[[T], Sequence[Any]] | None,
        retry_message: Callable[[Sequence[Any]], str] | None,
    ) -> Generated[T]:
        """Run one agent on `model` and classify the result; the arguments are as for `generate`.

        A subclass that supplies `model` on its own transport calls this to share the run, the
        validator and the classification.
        """
        agent, kept = self._agent(model, req, output, validate, retry_message)
        warns = unsupported(model, {**(model.settings or {}), **req.settings})
        with self._capture() as cap, capture_run_messages() as messages:
            try:
                res = await agent.run(_prompt(req), model=self._resilient(model, req, cap), **_streamed(req))
            except Exception as exc:
                facts = self._facts(cap, messages)
                got = classify_failure(exc, facts.wire, extra=self._extra, gated=self._gated)
                assert got is not None  # an exception always classifies
                self._report_internal(req, exc, got)
                # 1.x `usage` on a failed run was the last reply's; it stays so (usage_v2 sums them all).
                last_1x = dict(cap.usage) or None
                return self._result(req, None, got, None, facts, cap, warns, (), usage_1x=last_1x)
        facts = self._facts(cap, messages)
        retries = max(0, len(responses(res.all_messages())) - 1)
        got = classify_failure(None, facts.wire, extra=self._extra, gated=self._gated)
        value: Any = res.output
        if got is not None and got.kind == "truncated" and req.accept_truncated:
            got = None
            warns = (
                *warns,
                CallWarning("other", "accept_truncated", "finish_reason=length: the cut reply is kept as ok"),
            )
        elif got is not None and got.kind != "truncated":
            value = None
        u1 = usage_1x(facts.usage, cap.usage)
        return self._result(req, value, got, retries, facts, cap, warns, kept["problems"], usage_1x=u1)

    def _resilient(self, model: Model, req: GenRequest, cap: CaptureLike) -> Model:
        """`model` behind the call's wire retry: the running call's, else a fresh one for this run."""
        wire = _WIRE.get() or WireRetry(req.policy or self._policy, model=req.model, deadline_at=None, emit=self._emit)
        clocks = req.clocks

        def bound() -> float | None:
            # The call's own bound is enforced around the whole call; only a tighter one is set here.
            return clocks.attempt_s if clocks.attempt_s is not None and clocks.attempt_s < clocks.total_s else None

        def reset() -> None:
            if hasattr(cap, "last"):
                cap.last = None

        return ResilientModel(
            model, wire, judge_for(cap, extra=self._extra, gated=self._gated), bound=bound, reset=reset
        )

    def _facts(self, cap: CaptureLike, messages: Sequence[Any]) -> _Facts:
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
        return _Facts(replies, wire, finish, usage, resps)

    def _result(
        self,
        req: GenRequest,
        value: Any,
        got: Failure | None,
        retries: int | None,
        facts: _Facts,
        cap: CaptureLike,
        warns: tuple[CallWarning, ...],
        problems: tuple[Any, ...],
        *,
        usage_1x: Mapping[str, int | None] | None = None,
    ) -> Generated[Any]:
        """Assemble the result contract for one finished run, and emit its usage and cost events."""
        deployment = gateway_deployment(self._gateway, facts.replies, self._emit)
        answered = answered_of(req.model, facts.replies, facts.resps, deployment)
        gw = gateway_cost(self._gateway, facts.replies, self._emit)
        cost = price(facts.usage, answered, gateway_usd=gw, pricer=self._pricer, emit=self._emit)
        self._report_usage(answered, facts.usage, cost)
        res = Generated(
            value=value,
            outcome=got.outcome if got is not None else "ok",
            model=req.model,
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

    def _report_internal(self, req: GenRequest, exc: BaseException, got: Failure) -> None:
        if got.kind != "internal":
            return
        tb = "".join(traceback.format_exception(exc))[-TRACEBACK_MAX:]
        self._emit(
            logging.ERROR,
            "llm_call_internal_error",
            {"label": req.label, "model": req.model, "reason": got.reason, "traceback": tb},
        )

    @staticmethod
    def _agent[T](
        model: Model,
        req: GenRequest,
        output: type[T],
        validate: Callable[[T], Sequence[Any]] | None,
        retry_message: Callable[[Sequence[Any]], str] | None,
    ) -> tuple[Agent[None, Any], dict[str, tuple[Any, ...]]]:
        """Build the call's agent, and the slot its validator leaves the last problems in."""
        out_spec: Any = str if output is str else NativeOutput(framework_output(output))
        settings = dict(req.settings)
        clocks = req.clocks
        # The provider client's own per-attempt timeout; the deadline itself is enforced by the call.
        settings.setdefault("timeout", clocks.attempt_s or clocks.total_s)
        if req.stream_internally and clocks.first_event_s is not None:
            settings.setdefault("first_event_timeout", clocks.first_event_s)
        agent: Agent[None, Any] = Agent(
            model,
            output_type=out_spec,
            system_prompt=req.system or (),
            retries={"output": req.output_retries},
            model_settings=cast(ModelSettings, settings),
        )
        kept: dict[str, tuple[Any, ...]] = {"problems": ()}
        if validate is None:
            return agent, kept

        @agent.output_validator
        def _check(ctx: RunContext[None], out: Any) -> Any:
            probs = list(validate(out))
            if probs and ctx.retry < req.output_retries:
                raise ModelRetry(retry_message(probs) if retry_message else "; ".join(map(str, probs)))
            kept["problems"] = tuple(probs)
            return out

        return agent, kept


async def _drain(_ctx: RunContext[None], events: Any) -> None:
    """The event handler that makes a run stream internally; the events are not needed."""
    async for _ in events:
        pass


def _streamed(req: GenRequest) -> dict[str, Any]:
    """`agent.run` keywords: an event handler when the request streams internally, else none."""
    return {"event_stream_handler": _drain} if req.stream_internally else {}


def _settled[T](res: Generated[T], wire: WireRetry, total_s: float, waited: float | None) -> Generated[T]:
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


class _Facts:
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


__all__ = [
    "TRACEBACK_MAX",
    "CaptureLike",
    "ModelFor",
    "PydanticAIGenerator",
    "classify",
    "openai_compatible",
]
