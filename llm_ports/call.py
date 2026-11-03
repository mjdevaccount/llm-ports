"""call -- THE one generation path: role -> request -> gate -> trace -> Generator -> graders.

    res = await call.generate(CATEGORIZE, prompt=text, output=Reply, generator=gen, hooks=HOOKS)
    if res.outcome != "ok": ...          # gated | unavailable | request_fault | truncated | invalid

Per call: the role resolves to one `GenRequest` (`request_for`) -> `context.llm_feature(category)`
-> the `admit` hook (a refusal returns `gated` BEFORE any trace opens: a switched-off category is
not an error) -> one trace observation (the `trace` hook) around the `Generator` -> every `ok`
reply scored by the graders its role names, each written as a score on the observation and
carried under `stamp["scores"]` -> `on_scores` (a gauge) -> `on_ok` (a sampler). Feature code
carries no model literal, no timeout and no framework import: the role does, and the Generator is
passed in (an adapter, a test stub, a recording replay).

The hooks are where a consumer's own planes plug in -- a category switchboard, a tracing backend,
a metrics sink, an LLM-judge sampler. Every hook is optional; with none, the path is role ->
Generator -> graders, which is already more than a bare client call: the outcome is NAMED, and a
grader failure is a recorded score, never a silent drop.

NEVER RAISES for a model, wire or gate failure. A non-`ok`, non-`gated` outcome ends the
observation in error (`_NotOk` is raised INSIDE the trace block only) and emits `llm_call_failed`.
A caller whose ledger is an `except` uses `require_ok`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from typing import Any

from llm_ports import ports
from llm_ports.call_graders import CallGrader, CallScore, graders_for_inputs, score_all
from llm_ports.context import llm_feature
from llm_ports.failure import GenerationFailed
from llm_ports.observation import Observation, observe
from llm_ports.ports import Generated, Generator, GenRequest, Tracer
from llm_ports.reference import NoopTracer
from llm_ports.roles import Role

OUTPUT_TRACE_MAX = 4000
_log = logging.getLogger("llm_ports.call")


#: `(role, request, prompt, metadata) -> async context manager yielding an Observation`. On an
#: exception inside the block the hook ends the observation in error and lets it propagate.
TraceHook = Callable[[Role, GenRequest, str, Mapping[str, Any]], AbstractAsyncContextManager[Observation]]
#: `(role) -> None` to admit, or the refusal reason (the outcome becomes `gated`).
AdmitHook = Callable[[Role], str | None]
#: `(logging level, event, fields)`. Default: the stdlib logger `llm_ports.call`.
EventHook = Callable[[int, str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class Hooks:
    admit: AdmitHook | None = None
    trace: TraceHook | None = None
    graders: Mapping[str, CallGrader] | None = None  # None = call_graders.GRADERS
    on_scores: Callable[[Role, tuple[CallScore, ...]], Awaitable[Any]] | None = None
    on_ok: Callable[[Role, str, Generated[Any]], Any] | None = None
    on_event: EventHook | None = None


def tracer_trace(tracer: Tracer, *, provider: str = "openai") -> TraceHook:
    """A `trace` hook over any `Tracer` port adapter (Langfuse, OTel, a test recorder): one
    generation per call (`observation.observe`), NAMED BY THE ROLE'S CATEGORY, ended in error on a
    non-ok outcome."""

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


#: The trace when no hook is given: the port path over the sink-less tracer, so the package has ONE
#: null sink (`reference.NoopGeneration`) at either layer.
_null_trace = tracer_trace(NoopTracer())


class _NotOk(Exception):
    """Raised INSIDE the trace block only, so the observation ends in error; never escapes."""


def request_for(
    role: Role,
    *,
    prompt: str,
    system: str = "",
    images: Sequence[tuple[bytes, str]] = (),
    settings: Mapping[str, Any] | None = None,
) -> GenRequest:
    """Pure. The role resolved into one wire request. `settings` override the role's per call
    (a seed for a vote, a tighter max_tokens) -- the role stays the default, never a literal."""
    tier = role.resolved_tier
    return GenRequest(
        model=role.resolved_model,
        prompt=prompt,
        system=system,
        images=tuple(images),
        settings={**role.model_settings(), **(settings or {})},
        box=tier.box,
        timeout_s=role.resolved_timeout_s,
        output_retries=role.output_retries,
        label=role.name,
    )


def json_output(schema: Mapping[str, Any], name: str) -> type[dict[str, Any]]:
    """A hand-written JSON schema as `output=`: the grammar goes to the model unchanged and the
    value is a plain dict. `name` is what a test stub dispatches on. A port value type
    (`ports.JsonOutput`); the Generator adapter translates it."""
    return ports.json_output(schema, name)


def require_ok[T](res: Generated[T]) -> T:
    """The value, or the failure RAISED as `GenerationFailed(outcome, reason)` -- `gated` included.
    For a caller whose `except Exception` is its ledger, where the never-raise contract would
    otherwise turn a dead model into an unrecorded None."""
    if res.outcome != "ok" or res.value is None:
        raise GenerationFailed(res.outcome, res.reason)
    return res.value


def trace_output(value: Any) -> str:
    """What the observation records as output: JSON for a model or a mapping, capped."""
    if value is None:
        return ""
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        return str(dump())[:OUTPUT_TRACE_MAX]
    if isinstance(value, Mapping):
        return json.dumps(value, default=str)[:OUTPUT_TRACE_MAX]
    return str(value)[:OUTPUT_TRACE_MAX]


def _event(h: Hooks, level: int, event: str, **fields: Any) -> None:
    if h.on_event is not None:
        h.on_event(level, event, fields)
    else:
        _log.log(level, "%s %s", event, fields)


def _grade(role: Role, prompt: str, res: Generated[Any], obs: Observation, h: Hooks) -> Generated[Any]:
    """Score an `ok` reply with the role's graders onto its observation; the scores ride in the
    stamp. Pure but for the scores written; a failed grader is also emitted by name."""
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
    """One generation for `role` through `generator`. See the module docstring. `web=True` says the
    role's tier fetches pages the prompt never rendered: like an image, an input no text grader
    sees, so `call_graders.TEXT_ONLY_GRADERS` are dropped for THIS call (the role is untouched)."""
    h = hooks or Hooks()
    seen = graders_for_inputs(tuple(role.graders), has_images=bool(images), has_web=web)
    if seen != tuple(role.graders):
        role = replace(role, graders=seen)
    req = request_for(role, prompt=prompt, system=system, images=images, settings=settings)
    with llm_feature(role.category):
        if h.admit is not None:
            refused = h.admit(role)
            if refused is not None:
                return Generated(value=None, outcome="gated", model=req.model, reason=refused[:200])
        res: Generated[T] | None = None
        trace = h.trace or _null_trace
        try:
            async with trace(role, req, prompt, {"role": role.name, **(metadata or {})}) as obs:
                res = await generator.generate(req, output, validate=validate, retry_message=retry_message)
                obs.update(
                    output=trace_output(res.value),
                    usage_details=dict(res.usage or {}),
                    metadata={
                        "role": role.name,
                        "outcome": res.outcome,
                        "retries": res.retries,
                        "finish_reason": res.finish_reason,
                        "problems": len(res.problems),
                    },
                )
                res = replace(res, stamp={**res.stamp, "generation": obs})
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
        if res is None:  # only reachable if a trace hook swallowed a generator crash
            raise RuntimeError(f"llm_call {role.name}: the generator returned nothing")
        if res.stamp.get("scores") and h.on_scores is not None:
            await h.on_scores(role, res.stamp["scores"])
        if res.outcome == "ok" and h.on_ok is not None:
            h.on_ok(role, prompt, res)
        return res


__all__ = [
    "OUTPUT_TRACE_MAX",
    "AdmitHook",
    "EventHook",
    "Hooks",
    "Observation",
    "TraceHook",
    "generate",
    "json_output",
    "request_for",
    "require_ok",
    "trace_output",
    "tracer_trace",
]
