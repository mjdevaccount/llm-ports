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
  LiteLLM gateway, vLLM, llama.cpp's server, OpenAI itself).

`capture=` is a context-manager factory whose value carries the last reply's `finish_reason`,
`usage` and `sources` as they appeared on the wire (pydantic-ai drops non-standard keys);
`openai_compatible` installs an httpx response hook that fills `capture()`.

Invariants:
    - A model, wire or gate failure never raises; it returns a `Generated` with a named outcome.
    - `classify` names the outcome from the exception chain, whose root cause is at the bottom
      (pydantic-ai -> openai -> httpx -> the transport).
    - Exception types passed as `gated=` (a refusal raised by the caller's transport) map to
      `gated`.

Rationale: ADR-0001.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from pydantic_ai import Agent, BinaryContent, ModelRetry, NativeOutput, RunContext, StructuredDict
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model

from llm_ports.failure import exception_chain, failure_reason
from llm_ports.ports import Generated, GenRequest, JsonOutput, Outcome

#: HTTP statuses that blame the request (the input) rather than the model server.
REQUEST_FAULT = frozenset({400, 413, 422})

ModelFor = Callable[[GenRequest], AbstractAsyncContextManager[Model]]


class CaptureLike(Protocol):
    """The fields `PydanticAIGenerator` reads from a capture after a call."""

    finish_reason: str | None
    usage: Mapping[str, int | None]
    sources: tuple[str, ...] | None


@dataclass
class Capture:
    """What the wire showed on the last chat-completions reply of one call.

    Attributes:
        finish_reason: The reply's `finish_reason`, or None when the reply had none.
        usage: `input` and `output` token counts; a count the server did not report is None.
        sources: The reply's non-standard top-level `sources` strings, or None when it had none.
    """

    finish_reason: str | None = None
    usage: dict[str, int | None] = field(default_factory=dict)
    sources: tuple[str, ...] | None = None


_capture: ContextVar[Capture | None] = ContextVar("llm_ports_capture", default=None)


@contextmanager
def capture() -> Iterator[Capture]:
    """Open a capture for one call; `record_response` fills it from inside the httpx client."""
    cap = Capture()
    token = _capture.set(cap)
    try:
        yield cap
    finally:
        try:
            _capture.reset(token)
        except ValueError:
            _capture.set(None)


async def record_response(response: httpx.Response) -> None:
    """Record a chat-completions reply into the active capture.

    Installed as an httpx `event_hooks["response"]` hook. The active capture takes the reply's
    `finish_reason`, its usage, and a non-standard top-level `sources` list.

    Invariants:
        - Never raises; an unreadable body is ignored.
        - With no capture open, does nothing.
    """
    cap = _capture.get()
    if cap is None:
        return
    try:
        await response.aread()
        body = json.loads(response.content or b"{}")
        if not isinstance(body, dict):
            return
        choices = body.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        cap.finish_reason = choice.get("finish_reason")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        cap.usage = {"input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens")}
        src = body.get("sources")
        cap.sources = tuple(u for u in src if isinstance(u, str)) if isinstance(src, list) else None
    except Exception:  # telemetry must not break the call it describes
        return


def openai_compatible(
    base_url: str,
    api_key: str | None = None,
    *,
    max_retries: int = 0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ModelFor:
    """Build a `model_for` over any OpenAI-wire chat-completions endpoint.

    Each call opens one httpx client with the request's timeout and the capture hook.

    Args:
        base_url: The URL up to where `/chat/completions` starts, e.g. `http://localhost:4000/v1`.
        api_key: The bearer key; None sends a placeholder, for servers that need no key.
        max_retries: The OpenAI SDK's own retry count. Defaults to 0, so retries are not stacked
            on those of a gateway in front.
        transport: Replaces httpx's own transport (a caller's admitting or counting transport, or
            `httpx.MockTransport` in a test).

    Returns:
        A `ModelFor` that yields an `OpenAIChatModel` for `req.model`.
    """
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    @asynccontextmanager
    async def model_for(req: GenRequest) -> AsyncIterator[Model]:
        async with httpx.AsyncClient(
            timeout=req.timeout_s, transport=transport, event_hooks={"response": [record_response]}
        ) as client:
            provider = OpenAIProvider(base_url=base_url, api_key=api_key or "no-key", http_client=client)
            provider = OpenAIProvider(openai_client=provider.client.with_options(max_retries=max_retries))
            yield OpenAIChatModel(req.model, provider=provider)

    return model_for


def classify(
    exc: BaseException, finish_reason: str | None = None, *, gated: tuple[type[BaseException], ...] = ()
) -> tuple[Outcome, str]:
    """Name the outcome a failed run maps to, with a one-line reason.

    The checks run in order: an exception of a `gated` type anywhere in the chain is `gated`; a
    `length` finish is `truncated`, even though it surfaces as a parse error; an HTTP status in
    `REQUEST_FAULT` is `request_fault` and any other status is `unavailable`; a parse failure is
    `invalid`. Anything else is `unavailable`. Pure.

    Args:
        exc: The exception the run raised.
        finish_reason: The last reply's `finish_reason`, when the wire reported one.
        gated: Exception types that mean the caller's gate refused the call.

    Returns:
        The outcome and its reason.

    Rationale: ADR-0001.
    """
    chain = list(exception_chain(exc))
    if gated:
        for e in chain:
            if isinstance(e, gated):
                return "gated", str(e)[:200]
    if finish_reason == "length":
        return "truncated", "finish_reason=length: the token ceiling cut the reply"
    for e in chain:
        if isinstance(e, ModelHTTPError):
            return ("request_fault" if e.status_code in REQUEST_FAULT else "unavailable"), failure_reason(e)
        if isinstance(e, httpx.HTTPStatusError):
            code = e.response.status_code
            return ("request_fault" if code in REQUEST_FAULT else "unavailable"), failure_reason(e)
    if isinstance(exc, UnexpectedModelBehavior):
        return "invalid", failure_reason(exc)
    return "unavailable", failure_reason(exc)


def structured_dict(schema: Mapping[str, Any], name: str) -> type[dict[str, Any]]:
    """Wrap a hand-written JSON schema as an output type that returns a plain dict.

    The schema is compiled into the grammar unchanged, so `maxLength`, `enum`, `maxItems` and
    `required` all survive. `name` is the schema's title and the name a `FunctionModel` stub
    dispatches on. No shape validation runs beyond what the grammar enforces.
    """
    return StructuredDict(dict(schema), name=name)


def framework_output(output: Any) -> Any:
    """Map the port's `JsonOutput` to pydantic-ai's `StructuredDict`; any other type passes unchanged."""
    if isinstance(output, type) and issubclass(output, JsonOutput):
        return structured_dict(output.json_schema, output.schema_name)
    return output


def _usage(res: Any, fallback: Mapping[str, int | None]) -> dict[str, int | None] | None:
    """Return the run's token usage, else the capture's.

    pydantic-ai's run usage sums every request, re-asks included; the capture holds only the last
    reply's. A count nobody reported stays None, never 0.

    Rationale: ADR-0007.
    """
    try:
        u = res.usage()
        got = {"input": u.input_tokens, "output": u.output_tokens}
        if any(v for v in got.values()):
            return got
    except Exception:
        pass
    return dict(fallback) or None


def _last_finish(res: Any) -> str | None:
    for m in reversed(res.all_messages()):
        if isinstance(m, ModelResponse):
            return getattr(m, "finish_reason", None)
    return None


class PydanticAIGenerator:
    """The `Generator` port on pydantic-ai; the module docstring states the contract.

    Args:
        model: A fixed pydantic-ai `Model`.
        model_for: A per-request model factory.
        capture: The capture factory; the default is filled by `record_response`.
        gated: Exception types that classify as `gated`.
        wrap: Applied to each model that `model_for` yields (not to a fixed `model`), e.g. a
            `pydantic_ai_replay.Recorder`.

    Raises:
        ValueError: When neither or both of `model` and `model_for` are given.
    """

    def __init__(
        self,
        model: Model | None = None,
        *,
        model_for: ModelFor | None = None,
        capture: Callable[[], AbstractContextManager[CaptureLike]] = capture,
        gated: tuple[type[BaseException], ...] = (),
        wrap: Callable[[Model], Model] | None = None,
    ) -> None:
        if (model is None) == (model_for is None):
            raise ValueError("PydanticAIGenerator needs exactly one of model= / model_for=")
        self._model = model
        self._model_for = model_for
        self._capture = capture
        self._gated = gated
        self._wrap = wrap

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
            retry_message: Renders the problems into the re-ask message; by default they are
                joined with "; ".

        Returns:
            A `Generated` whose `outcome` is `ok` or a named failure.
        """
        if self._model is not None:
            return await self.run(self._model, req, output, validate, retry_message)
        assert self._model_for is not None
        try:
            async with self._model_for(req) as model:
                if self._wrap is not None:
                    model = self._wrap(model)
                return await self.run(model, req, output, validate, retry_message)
        except Exception as exc:  # building the wire failed (bad URL, a refusing transport)
            outcome, reason = classify(exc, gated=self._gated)
            return Generated(value=None, outcome=outcome, model=req.model, reason=reason, retries=None)

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
        out_spec: Any = str if output is str else NativeOutput(framework_output(output))
        agent: Agent[None, Any] = Agent(
            model,
            output_type=out_spec,
            system_prompt=req.system or (),
            retries={"output": req.output_retries},  # type: ignore[arg-type]
            model_settings=dict(req.settings),
        )  # type: ignore[arg-type]
        kept: dict[str, tuple[Any, ...]] = {"problems": ()}

        if validate is not None:

            @agent.output_validator
            def _check(ctx: RunContext[None], out: Any) -> Any:
                probs = list(validate(out))
                if probs and ctx.retry < req.output_retries:
                    raise ModelRetry(retry_message(probs) if retry_message else "; ".join(map(str, probs)))
                kept["problems"] = tuple(probs)
                return out

        prompt: Any = req.prompt
        if req.images:
            prompt = [req.prompt, *(BinaryContent(data=b, media_type=m) for b, m in req.images)]
        with self._capture() as cap:
            try:
                res = await agent.run(prompt)
            except Exception as exc:
                outcome, reason = classify(exc, cap.finish_reason, gated=self._gated)
                return Generated(
                    value=None,
                    outcome=outcome,
                    model=req.model,
                    reason=reason,
                    retries=None,
                    usage=cap.usage or None,
                    sources=cap.sources,
                    finish_reason=cap.finish_reason,
                )
        retries = max(0, sum(isinstance(m, ModelResponse) for m in res.all_messages()) - 1)
        return Generated(
            value=res.output,
            outcome="ok",
            model=req.model,
            retries=retries,
            problems=kept["problems"],
            usage=_usage(res, cap.usage),
            sources=cap.sources,
            finish_reason=cap.finish_reason or _last_finish(res),
        )


__all__ = [
    "REQUEST_FAULT",
    "Capture",
    "CaptureLike",
    "ModelFor",
    "PydanticAIGenerator",
    "capture",
    "classify",
    "framework_output",
    "openai_compatible",
    "record_response",
    "structured_dict",
]
