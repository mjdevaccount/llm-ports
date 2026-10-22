"""PydanticAIGenerator -- the `Generator` port on pydantic-ai (OUR seam, THEIR impl).

One `Agent` per call: `NativeOutput(T)` sends T's schema as a strict json_schema (the grammar a
llama.cpp server compiles; the structured-output mode OpenAI honours) and validates the reply into
T; `str` asks for prose. The caller's `validate` runs as an `output_validator`: a non-empty problem
list raises `ModelRetry(retry_message(problems))` while re-asks remain, after which the value is
KEPT with its problems (the caller prunes or drops). Retries, schema validation and re-asks are
pydantic-ai's; nothing here re-implements them.

WHERE THE MODEL COMES FROM is the consumer's:
  * `model=`      a fixed pydantic-ai `Model` (tests pass a `FunctionModel`);
  * `model_for=`  `(GenRequest) -> async context manager yielding a Model`, per call -- how a
                  consumer puts the model on its own transport (admission, leases, a work row);
  * `openai_compatible(base_url, api_key)` builds a `model_for` for any OpenAI-wire endpoint
                  (a LiteLLM gateway, vLLM, llama.cpp's server, OpenAI itself).

`capture=` is a context-manager factory whose value carries `finish_reason`, `usage` and
`sources` of the last reply as the WIRE saw them (pydantic-ai drops non-standard keys);
`openai_compatible` installs an httpx response hook that fills `capture()` for you.

NEVER RAISES for a model, wire or gate failure. `classify` names the outcome from the exception
chain (the cause is at the bottom: pydantic-ai -> openai -> httpx -> the transport); exception
types in `gated=` (a consumer's switchboard refusal raised from its transport) map to `gated`.
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

#: Statuses that blame the REQUEST (the input), not the model server.
REQUEST_FAULT = frozenset({400, 413, 422})

ModelFor = Callable[[GenRequest], AbstractAsyncContextManager[Model]]


class CaptureLike(Protocol):
    finish_reason: str | None
    usage: Mapping[str, int | None]
    sources: tuple[str, ...] | None


@dataclass
class Capture:
    """What the wire saw on the LAST chat-completions reply of one call."""

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
    """httpx `event_hooks["response"]` hook: the active capture takes the reply's finish_reason,
    usage and a non-standard top-level `sources` list. Never raises; no capture open = no-op."""
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
    except Exception:          # telemetry must not break the call it describes
        return


def openai_compatible(base_url: str, api_key: str | None = None, *, max_retries: int = 0,
                      transport: httpx.AsyncBaseTransport | None = None) -> ModelFor:
    """A `model_for` over any OpenAI-wire endpoint (`base_url` ends where `/chat/completions`
    starts, e.g. `http://litellm:4000/v1`). One httpx client per call with the request's timeout
    and the capture hook. The SDK's own retries default OFF (`max_retries=0`): a gateway in front
    retries, and two stacked policies multiply. `transport` replaces httpx's own (a consumer's
    admitting/counting transport, or `httpx.MockTransport` in a test)."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    @asynccontextmanager
    async def model_for(req: GenRequest) -> AsyncIterator[Model]:
        async with httpx.AsyncClient(timeout=req.timeout_s, transport=transport,
                                     event_hooks={"response": [record_response]}) as client:
            provider = OpenAIProvider(base_url=base_url, api_key=api_key or "no-key", http_client=client)
            provider = OpenAIProvider(openai_client=provider.client.with_options(max_retries=max_retries))
            yield OpenAIChatModel(req.model, provider=provider)

    return model_for


def classify(exc: BaseException, finish_reason: str | None = None, *,
             gated: tuple[type[BaseException], ...] = ()) -> tuple[Outcome, str]:
    """Pure. The outcome a failed run maps to, and its one-line reason. Order matters: a gate
    refusal anywhere in the chain wins; a `length` finish is a BUDGET failure even though it
    surfaces as a parse error; then status codes; then the parse failure itself."""
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
    """A hand-written JSON schema as an output type: compiled into the grammar unchanged (maxLength
    / enum / maxItems / required all survive) and returned as a plain dict. `name` is the schema's
    title and the name a `FunctionModel` stub dispatches on. NO shape validation past the grammar."""
    return StructuredDict(dict(schema), name=name)


def framework_output(output: Any) -> Any:
    """The port's `JsonOutput` becomes pydantic-ai's `StructuredDict`; a pydantic model passes."""
    if isinstance(output, type) and issubclass(output, JsonOutput):
        return structured_dict(output.json_schema, output.schema_name)
    return output


def _usage(res: Any, fallback: Mapping[str, int | None]) -> dict[str, int | None] | None:
    """pydantic-ai's run usage SUMS every request (re-asks included); the capture holds only the
    last reply's. None-safe: an unreported count stays None, never 0."""
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
    """See the module docstring. Exactly one of `model` / `model_for` is required."""

    def __init__(self, model: Model | None = None, *, model_for: ModelFor | None = None,
                 capture: Callable[[], AbstractContextManager[CaptureLike]] = capture,
                 gated: tuple[type[BaseException], ...] = (),
                 wrap: Callable[[Model], Model] | None = None) -> None:
        if (model is None) == (model_for is None):
            raise ValueError("PydanticAIGenerator needs exactly one of model= / model_for=")
        self._model = model
        self._model_for = model_for
        self._capture = capture
        self._gated = gated
        self._wrap = wrap

    async def generate[T](self, req: GenRequest, output: type[T], *,
                          validate: Callable[[T], Sequence[Any]] | None = None,
                          retry_message: Callable[[Sequence[Any]], str] | None = None) -> Generated[T]:
        if self._model is not None:
            return await self.run(self._model, req, output, validate, retry_message)
        assert self._model_for is not None
        try:
            async with self._model_for(req) as model:
                if self._wrap is not None:
                    model = self._wrap(model)
                return await self.run(model, req, output, validate, retry_message)
        except Exception as exc:        # building the wire failed (bad URL, a refusing transport)
            outcome, reason = classify(exc, gated=self._gated)
            return Generated(value=None, outcome=outcome, model=req.model, reason=reason, retries=None)

    async def run[T](self, model: Model, req: GenRequest, output: type[T],
                     validate: Callable[[T], Sequence[Any]] | None,
                     retry_message: Callable[[Sequence[Any]], str] | None) -> Generated[T]:
        """One agent run on `model`. Public so a consumer's subclass can put `model` on its own
        transport and still share the run, the validator and the classification."""
        out_spec: Any = str if output is str else NativeOutput(framework_output(output))
        agent: Agent[None, Any] = Agent(
            model, output_type=out_spec, system_prompt=req.system or (),
            retries={"output": req.output_retries},  # type: ignore[arg-type]
            model_settings=dict(req.settings))  # type: ignore[arg-type]
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
                return Generated(value=None, outcome=outcome, model=req.model, reason=reason,
                                 retries=None, usage=cap.usage or None, sources=cap.sources,
                                 finish_reason=cap.finish_reason)
        retries = max(0, sum(isinstance(m, ModelResponse) for m in res.all_messages()) - 1)
        return Generated(value=res.output, outcome="ok", model=req.model, retries=retries,
                         problems=kept["problems"], usage=_usage(res, cap.usage),
                         sources=cap.sources, finish_reason=cap.finish_reason or _last_finish(res))


__all__ = ["REQUEST_FAULT", "Capture", "CaptureLike", "ModelFor", "PydanticAIGenerator", "capture",
           "classify", "framework_output", "openai_compatible", "record_response", "structured_dict"]
