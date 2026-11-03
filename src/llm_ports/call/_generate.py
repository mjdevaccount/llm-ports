"""The single generation path: role, request, gate, trace, `Generator`, graders.

Example:
    res = await call.generate(CATEGORIZE, prompt=text, output=Reply, generator=gen, hooks=HOOKS)
    if res.outcome != "ok": ...  # gated | unavailable | request_fault | truncated | invalid

Per call, the role resolves to one `GenRequest` (`request_for`), the category is entered with
`observability.llm_feature`, and the `admit` hook runs. A refusal returns `gated` before any trace opens:
a switched-off category is not an error. Otherwise one trace observation (the `trace` hook) wraps
the `Generator` call. Every `ok` reply is scored by the graders its role names; each score is
written on the observation and carried under `stamp["scores"]`, then handed to `on_scores` (a
gauge) and `on_ok` (a sampler). Feature code carries no model literal, no timeout and no framework
import: the role carries them, and the `Generator` is passed in (an adapter, a test stub, a
recording replay).

The hooks are where a consumer's own planes plug in: a category switchboard, a tracing backend, a
metrics sink, an LLM-judge sampler. Every hook is optional. With none, the path is role,
`Generator`, graders, and the outcome is still named and a grader failure still recorded as a score.

Invariants:
    - `generate` never raises for a model, wire or gate failure; the outcome names it.
    - A non-`ok`, non-`gated` outcome ends the observation in error and emits `llm_call_failed`.
      `_NotOk` is raised inside the trace block only and never escapes.
    - A caller whose ledger is an `except` uses `require_ok`.

Rationale: ADR-0001.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from typing import Any

from llm_ports import ports
from llm_ports.call.graders import CallGrader, CallScore, graders_for_inputs, score_all
from llm_ports.call.roles import Role
from llm_ports.observability import EventHook, NoopTracer, Observation, llm_feature, log_events, observe
from llm_ports.ports import Generated, Generator, GenRequest, Tracer
from llm_ports.ports.outcomes import REASON_MAX, GenerationFailed

#: The longest output text written on a trace observation, in characters.
OUTPUT_TRACE_MAX = 4000
#: Where events go when `Hooks.on_event` is None.
_log_event = log_events("llm_ports.call")


#: Trace hook: `(role, request, prompt, metadata)` to an async context manager yielding an
#: `Observation`. On an exception inside the block, the hook ends the observation in error and lets
#: the exception propagate.
TraceHook = Callable[[Role, GenRequest, str, Mapping[str, Any]], AbstractAsyncContextManager[Observation]]
#: Admission hook: `(role)` to None to admit, or to the refusal reason (the outcome becomes `gated`).
AdmitHook = Callable[[Role], str | None]


@dataclass(frozen=True)
class Hooks:
    """Optional consumer hooks for `generate`.

    Attributes:
        admit: Runs before any trace opens; a returned reason makes the outcome `gated`.
        trace: Opens the call's observation; without one, a sink-less tracer is used.
        graders: The grader table by name; None means `call.graders.GRADERS`.
        on_scores: Awaited with the role and its scores whenever a reply was graded.
        on_ok: Called with the role, prompt and result after an `ok` reply.
        on_event: Receives `llm_call_failed` and `llm_call_graded`; without one, they go to the
            stdlib logger `llm_ports.call`.
    """

    admit: AdmitHook | None = None
    trace: TraceHook | None = None
    graders: Mapping[str, CallGrader] | None = None
    on_scores: Callable[[Role, tuple[CallScore, ...]], Awaitable[Any]] | None = None
    on_ok: Callable[[Role, str, Generated[Any]], Any] | None = None
    on_event: EventHook | None = None


def tracer_trace(tracer: Tracer, *, provider: str = "openai") -> TraceHook:
    """Return a `trace` hook over any `Tracer` port adapter (Langfuse, OTel, a test recorder).

    The hook opens one generation per call (`observability.observe`), named by the role's category,
    and ends it in error on a non-`ok` outcome.
    """

    def trace(
        role: Role, req: GenRequest, prompt: str, meta: Mapping[str, Any]
    ) -> AbstractAsyncContextManager[Observation]:
        return observe(
            tracer,
            name=role.category,
            model=req.model,
            provider=provider,
            input=prompt,
            metadata=meta,
            prompt_name=role.pack,
        )

    return trace


#: The trace used when no hook is given: the port path over the sink-less tracer, so the package
#: has one null sink (`observability.NoopGeneration`) at either layer.
_null_trace = tracer_trace(NoopTracer())


class _NotOk(Exception):
    """Ends the observation in error; raised inside the trace block only and never escapes."""


def request_for(
    role: Role,
    *,
    prompt: str,
    system: str = "",
    images: Sequence[tuple[bytes, str]] = (),
    settings: Mapping[str, Any] | None = None,
) -> GenRequest:
    """Resolve the role into one wire request.

    Pure. `settings` override the role's model settings for this call (a seed for a vote, a tighter
    `max_tokens`); the role stays the default.
    """
    tier = role.resolved_tier
    return GenRequest(
        model=role.resolved_model,
        prompt=prompt,
        system=system,
        images=tuple(images),
        settings={**role.model_settings(), **(settings or {})},
        self_hosted=tier.self_hosted,
        timeout_s=role.resolved_timeout_s,
        output_retries=role.output_retries,
        label=role.name,
    )


def json_output(schema: Mapping[str, Any], name: str) -> type[dict[str, Any]]:
    """Return a hand-written JSON schema as an `output=` type.

    The schema goes to the model unchanged as the grammar, and the value is a plain dict. `name` is
    what a test stub dispatches on. The result is a port value type (`ports.JsonOutput`); the
    `Generator` adapter translates it.
    """
    return ports.json_output(schema, name)


def require_ok[T](res: Generated[T]) -> T:
    """Return the result's value, or raise its failure.

    For a caller whose `except Exception` is its ledger: under the never-raise contract a dead model
    would otherwise become an unrecorded None.

    Raises:
        GenerationFailed: `(outcome, reason)` for any non-`ok` outcome, `gated` included, and for an
            `ok` result without a value.
    """
    if res.outcome != "ok" or res.value is None:
        raise GenerationFailed(res.outcome, res.reason)
    return res.value


def _trace_output(value: Any) -> str:
    """Return the observation's output text, capped at `OUTPUT_TRACE_MAX` characters.

    A pydantic model or a mapping is rendered as JSON; None is the empty string.
    """
    if value is None:
        return ""
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        return str(dump())[:OUTPUT_TRACE_MAX]
    if isinstance(value, Mapping):
        return json.dumps(value, default=str)[:OUTPUT_TRACE_MAX]
    return str(value)[:OUTPUT_TRACE_MAX]


def _event(h: Hooks, level: int, event: str, **fields: Any) -> None:
    (h.on_event or _log_event)(level, event, fields)


def _grade(role: Role, prompt: str, res: Generated[Any], obs: Observation, h: Hooks) -> Generated[Any]:
    """Score an `ok` reply with the role's graders and write each score on its observation.

    The scores ride in the returned stamp under `"scores"`. Failed graders are also emitted by name
    as `llm_call_graded`.
    """
    scores = score_all(role, prompt, res.value, h.graders)
    for s in scores:
        obs.score(name=f"{role.name}.{s.name}", value=s.value, comment=s.reason or None)
    failed = [s.name for s in scores if not s.passed]
    if failed:
        _event(
            h,
            logging.INFO,
            "llm_call_graded",
            role=role.name,
            failed=failed,
            reasons={s.name: s.reason for s in scores if not s.passed},
        )
    return replace(res, stamp={**res.stamp, "scores": scores})


def _gated(role: Role, req: GenRequest, h: Hooks) -> Generated[Any] | None:
    """Return the `gated` result when the admit hook refuses the call, else None."""
    refused = h.admit(role) if h.admit is not None else None
    if refused is None:
        return None
    return Generated(value=None, outcome="gated", model=req.model, reason=refused[:REASON_MAX])


def _observed[T](role: Role, res: Generated[T], obs: Observation) -> Generated[T]:
    """Write the reply on its observation and stamp the observation on the reply."""
    obs.update(
        output=_trace_output(res.value),
        usage_details=dict(res.usage or {}),
        metadata={
            "role": role.name,
            "outcome": res.outcome,
            "retries": res.retries,
            "finish_reason": res.finish_reason,
            "problems": len(res.problems),
        },
    )
    return replace(res, stamp={**res.stamp, "generation": obs})


async def _report(role: Role, prompt: str, res: Generated[Any], h: Hooks) -> None:
    """Hand the scores, and an `ok` reply, to the consumer's hooks."""
    if res.stamp.get("scores") and h.on_scores is not None:
        await h.on_scores(role, res.stamp["scores"])
    if res.outcome == "ok" and h.on_ok is not None:
        h.on_ok(role, prompt, res)


async def generate[T](
    role: Role,
    *,
    prompt: str,
    output: type[T] = str,  # type: ignore[assignment]
    generator: Generator,
    hooks: Hooks | None = None,
    system: str = "",
    images: Sequence[tuple[bytes, str]] = (),
    validate: Callable[[T], Sequence[Any]] | None = None,
    retry_message: Callable[[Sequence[Any]], str] | None = None,
    settings: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    web: bool = False,
) -> Generated[T]:
    """Run one generation for `role` through `generator`.

    The path and the hooks are described in the module docstring. `web=True` says the role's tier
    fetches pages the prompt never rendered: like an image, an input no text grader sees, so
    `call.graders.TEXT_ONLY_GRADERS` are dropped for this call. The role itself is unchanged.

    Returns:
        The result, with a named outcome. A model, wire or gate failure is an outcome, never an
        exception.

    Raises:
        RuntimeError: Only when a trace hook swallowed a generator exception, leaving no result.
    """
    h = hooks or Hooks()
    seen = graders_for_inputs(tuple(role.graders), has_images=bool(images), has_web=web)
    if seen != tuple(role.graders):
        role = replace(role, graders=seen)
    req = request_for(role, prompt=prompt, system=system, images=images, settings=settings)
    with llm_feature(role.category):
        gated = _gated(role, req, h)
        if gated is not None:
            return gated
        res: Generated[T] | None = None
        trace = h.trace or _null_trace
        try:
            async with trace(role, req, prompt, {"role": role.name, **(metadata or {})}) as obs:
                res = await generator.generate(req, output, validate=validate, retry_message=retry_message)
                res = _observed(role, res, obs)
                if res.outcome == "ok" and role.graders:
                    res = _grade(role, prompt, res, obs, h)
                if res.outcome not in ("ok", "gated"):
                    raise _NotOk(f"{res.outcome}: {res.reason}")
        except _NotOk:
            _event(
                h,
                logging.WARNING,
                "llm_call_failed",
                role=role.name,
                outcome=res.outcome if res else None,
                reason=res.reason if res else None,
            )
        if res is None:  # reachable only when a trace hook swallowed a generator exception
            raise RuntimeError(f"llm_call {role.name}: the generator returned nothing")
        await _report(role, prompt, res, h)
        return res


__all__ = [
    "OUTPUT_TRACE_MAX",
    "AdmitHook",
    "Hooks",
    "TraceHook",
    "generate",
    "json_output",
    "request_for",
    "require_ok",
    "tracer_trace",
]
