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
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import replace
from typing import Any

from llm_ports._extras import extra
from llm_ports._headers import HEADER_VALUE_RULE, header_value_ok
from llm_ports.adapters._classify import Classifier, WireFacts
from llm_ports.adapters.pydantic_ai._capture import CaptureLike, capture, record_response
from llm_ports.adapters.pydantic_ai._classify import classify, classify_failure
from llm_ports.adapters.pydantic_ai._core import (
    TRACEBACK_MAX,
    WIRE,
    AdapterCore,
    build_agent,
    call_settings,
    settled,
)
from llm_ports.adapters.pydantic_ai._result import responses, unsupported, usage_1x
from llm_ports.observability import EventHook
from llm_ports.ports import (
    Answered,
    CallWarning,
    GatewayReader,
    Generated,
    GenRequest,
    Limiter,
    Pricer,
    SlotWaitExceeded,
    WirePolicy,
    failure,
)

with extra("pydantic-ai"):
    import httpx
    import httpx2
    from pydantic_ai import BinaryContent, RunContext, capture_run_messages
    from pydantic_ai.models import Model
    from pydantic_ai.profiles import ModelProfileSpec
    from pydantic_ai.settings import ModelSettings

    from llm_ports.adapters.pydantic_ai._client import OpenAICompatible

ModelFor = Callable[[GenRequest], AbstractAsyncContextManager[Model]]

#: The 1.x names of the capture and its hook, kept for callers that reached for them.
_capture = capture
_record_response = record_response


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


class PydanticAIGenerator(AdapterCore):
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
        super().__init__(
            capture=capture,
            gated=gated,
            wrap=wrap,
            pricer=pricer,
            gateway=gateway,
            extra=extra,
            on_event=on_event,
            policy=policy,
            limiter=limiter,
        )
        self._model = model
        self._model_for = model_for
        self._source = model_for

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
        wire = self._wire(req.policy, req.model, clocks.at, req.label)
        token = WIRE.set(wire)
        self._in_flight += 1
        waited: float | None = None
        try:
            async with asyncio.timeout_at(clocks.at):
                async with self._slot(req.concurrency_group or req.model) as waited_s:
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
            WIRE.reset(token)
            self._in_flight -= 1
        return settled(res, wire, clocks.total_s, waited if self._limiter is not None else None)

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
            self._report_internal(req.label, req.model, exc, got)
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
        settings = call_settings(req.settings, req.clocks, streamed=req.stream_internally, cache=req.cache_prefix)
        agent, kept = build_agent(
            model,
            output,
            settings=settings,
            output_retries=req.output_retries,
            validate=validate,
            retry_message=retry_message,
            system=req.system,
        )
        warns = unsupported(model, {**(model.settings or {}), **req.settings})
        with self._capture() as cap, capture_run_messages() as messages:
            try:
                resilient = self._resilient(model, req.policy, req.model, req.clocks, cap)
                res = await agent.run(_prompt(req), model=resilient, **_streamed(req))
            except Exception as exc:
                facts = self._facts(cap, messages)
                got = classify_failure(exc, facts.wire, extra=self._extra, gated=self._gated)
                assert got is not None  # an exception always classifies
                self._report_internal(req.label, req.model, exc, got)
                # 1.x `usage` on a failed run was the last reply's; it stays so (usage_v2 sums them all).
                last_1x = dict(cap.usage) or None
                return self._result(req.model, None, got, None, facts, cap, warns, (), usage_1x=last_1x)
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
        return self._result(req.model, value, got, retries, facts, cap, warns, kept["problems"], usage_1x=u1)


async def _drain(_ctx: RunContext[None], events: Any) -> None:
    """The event handler that makes a run stream internally; the events are not needed."""
    async for _ in events:
        pass


def _streamed(req: GenRequest) -> dict[str, Any]:
    """`agent.run` keywords: an event handler when the request streams internally, else none."""
    return {"event_stream_handler": _drain} if req.stream_internally else {}


__all__ = [
    "TRACEBACK_MAX",
    "CaptureLike",
    "ModelFor",
    "PydanticAIGenerator",
    "classify",
    "openai_compatible",
]
