"""The single generation path: role, request, gate, trace, `Generator`, graders.

Example:
    res = await call.generate(CATEGORIZE, prompt=text, output=Reply, generator=gen, hooks=HOOKS)
    if res.outcome != "ok": ...  # gated | unavailable | request_fault | truncated | invalid | refused
    if res.failure: res.failure.kind  # the detail: rate_limited, context_overflow, internal, ...

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

Deadline and fallback (ADR-0021, ADR-0022): the call's `Deadline` (the role's timeout as
`total_s`) starts when `generate` is entered and bounds everything: admit, the slot wait, every wire
request, re-ask and backoff, every fallback tier, and grading. A tier that fails with a kind in its
`fallback_on` hands the call to the next tier of its `fallback` chain with the REMAINING deadline;
the result says which tier answered (`answered.tier`) and carries every earlier tier's failure in
`attempts`. A `Generator` that does not honour `GenRequest.deadline` is cut off `DEADLINE_GRACE_S`
after it, as a `timeout` with `phase="total"`.

Invariants:
    - `generate` never raises for a model, wire or gate failure; the outcome names it.
    - A deadline expiry is a `timeout` result; a caller's cancellation propagates.
    - Fallback starts only after the tier's own wire policy gave up, and only for a listed kind.
    - A non-`ok`, non-`gated` outcome ends the observation in error and emits `llm_call_failed`.
      `_NotOk` is raised inside the trace block only and never escapes.
    - A caller whose ledger is an `except` uses `require_ok`.
    - The observation's metadata carries the failure kind, the cost and its source, who answered,
      the unified finish, the attempt count and every usage bucket; an unknown stays None.

Rationale: ADR-0001, ADR-0019, ADR-0020, ADR-0021, ADR-0022.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from typing import Any

from llm_ports import ports
from llm_ports.call._fallback import Route, chain_of, timed_out
from llm_ports.call._preflight import finished, prepared
from llm_ports.call.graders import CallGrader, CallScore, graders_for_inputs, score_all
from llm_ports.call.models import missing, needs
from llm_ports.call.roles import Role
from llm_ports.observability import (
    EventHook,
    NoopTracer,
    Observation,
    PortObservation,
    llm_feature,
    log_events,
    observe,
)
from llm_ports.ports import Deadline, Generated, Generator, GenRequest, TokenCounter, Tracer
from llm_ports.ports.outcomes import REASON_MAX, GenerationFailed

#: The longest output text written on a trace observation, in characters.
OUTPUT_TRACE_MAX = 4000
#: Where events go when `Hooks.on_event` is None.
_log_event = log_events("llm_ports.call")
#: How long after its deadline a `Generator` that does not honour `GenRequest.deadline` is cut off,
#: in seconds. A generator that honours it (the pydantic-ai adapter) ends the call AT the deadline.
DEADLINE_GRACE_S = 0.1


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
        on_event: Receives `llm_call_failed`, `llm_call_graded`, `llm_fallback` and
            `llm_deadline_expired`; without one, they go to the stdlib logger `llm_ports.call`.
        generator_for: Tier name to the `Generator` that serves it (e.g.
            `call.providers.Backends.generator_for`), for a fallback tier on another provider.
            None runs every tier on the `generator` passed to `generate` (the model id is on the
            request).
    """

    admit: AdmitHook | None = None
    trace: TraceHook | None = None
    graders: Mapping[str, CallGrader] | None = None
    on_scores: Callable[[Role, tuple[CallScore, ...]], Awaitable[Any]] | None = None
    on_ok: Callable[[Role, str, Generated[Any]], Any] | None = None
    on_event: EventHook | None = None
    generator_for: Callable[[str], Generator] | None = None


def tracer_trace(tracer: Tracer, *, provider: str | None = None) -> TraceHook:
    """Return a `trace` hook over any `Tracer` port adapter (Langfuse, OTel, `TeeTracer`, a recorder).

    The hook opens one generation per call (`observability.observe`), named by the role's category,
    and ends it in error on a non-`ok` outcome. The end carries the call's result contract (usage,
    cost, finish, who answered, failure) through the one end payload `observability.record` uses.

    Args:
        tracer: The sink.
        provider: The provider named on the generation; None takes the tier's provider from a
            schema 2 table, else `"unknown"` (1.x defaulted to `"openai"`).
    """

    def trace(
        role: Role, req: GenRequest, prompt: str, meta: Mapping[str, Any]
    ) -> AbstractAsyncContextManager[Observation]:
        info = role.resolved_tier.info
        return observe(
            tracer,
            name=role.category,
            model=req.model,
            provider=provider or (info.provider if info is not None else None) or "unknown",
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
    info = tier.info
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
        accept_truncated=role.accept_truncated,
        concurrency_group=None if info is None else (info.group or info.key),
        policy=role.policy if role.policy is not None else tier.wire_policy,
        stream_internally=role.stream_internally,
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


#: Warnings already emitted once in this process (`llm_call_warning`), by key.
_WARNED: set[str] = set()


def _warn_once(h: Hooks, key: str, setting: str, detail: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    _event(h, logging.WARNING, "llm_call_warning", kind="other", setting=setting, detail=detail)


def _unsupported(role: Role, req: GenRequest, output: type[Any], web: bool, h: Hooks) -> Generated[Any] | None:
    """Return the `request_fault` result when the tier's model lacks a capability the call needs.

    Nothing is sent: the refusal is decided from the table before the wire. A schema 1 tier states
    no capability, so nothing is checked, and `llm_call_warning` says so once per process.
    """
    info = role.resolved_tier.info
    if info is None:
        _warn_once(h, "capabilities", "capabilities", "a schema 1 tier table states no capability; none is checked")
        return None
    if role.model is not None and role.model != info.id:
        _warn_once(h, f"model:{role.name}", "model", f"role {role.name!r} overrides the table's model id")
    wanted = needs(structured=output is not str, images=bool(req.images), web=web)
    lacks = missing(info, wanted)
    if not lacks:
        return None
    reason = f"unsupported: model {info.key!r} ({info.id}) does not state {', '.join(lacks)}"
    _event(h, logging.WARNING, "llm_call_failed", role=role.name, outcome="request_fault", reason=reason)
    return Generated(value=None, outcome="request_fault", model=req.model, reason=reason[:REASON_MAX])


def _stamped[T](role: Role, res: Generated[T]) -> Generated[T]:
    """Stamp the fingerprint of the tier table that routed the call (`stamp["tiers"]`)."""
    fp = role.resolved_tier.table_fingerprint
    return res if fp is None else replace(res, stamp={**res.stamp, "tiers": fp})


def _gated(role: Role, req: GenRequest, h: Hooks) -> Generated[Any] | None:
    """Return the `gated` result when the admit hook refuses the call, else None."""
    refused = h.admit(role) if h.admit is not None else None
    if refused is None:
        return None
    return Generated(
        value=None,
        outcome="gated",
        model=req.model,
        answered=ports.Answered(requested=req.model),
        failure=ports.failure("gated", reason=refused[:REASON_MAX]),
    )


def _result_metadata(res: Generated[Any]) -> dict[str, Any]:
    """The result contract as flat observation metadata: kind, cost, who answered, finish, usage."""
    u, cost, who = res.usage_v2, res.cost, res.answered
    return {
        "kind": res.failure.kind if res.failure is not None else None,
        "cost_usd": str(cost.usd) if cost is not None and cost.usd is not None else None,
        "cost_source": cost.source if cost is not None else None,
        "answered_model": who.model if who is not None else None,
        "answered_deployment": who.deployment if who is not None else None,
        "finish": res.finish.unified if res.finish is not None else None,
        "attempts": len(res.attempts),
        "answered_tier": who.tier if who is not None else None,
        "limiter_wait_s": res.stamp.get("limiter_wait_s"),
        "wire_requests": res.stamp.get("wire_requests"),
        "usage_input": u.input if u is not None else None,
        "usage_output": u.output if u is not None else None,
        "usage_cache_read": u.cache_read if u is not None else None,
        "usage_cache_write": u.cache_write if u is not None else None,
        "usage_reasoning": u.reasoning if u is not None else None,
        "usage_requests": u.requests if u is not None else None,
        "usage_unmeasured": u.unmeasured if u is not None else None,
    }


def _observed[T](
    role: Role, res: Generated[T], obs: Observation, extra: Mapping[str, Any] | None = None
) -> Generated[T]:
    """Write the reply on its observation and stamp the observation on the reply.

    The tier answering is stamped on `answered` here, before the observation ends, so the trace
    names the tier (the fallback loop stamps the same name later).
    """
    who = res.answered
    if who is not None and who.tier is None:
        res = replace(res, answered=replace(who, tier=role.resolved_tier.name))
    obs.update(
        output=_trace_output(res.value),
        usage_details=dict(res.usage or {}),
        metadata={
            "role": role.name,
            "outcome": res.outcome,
            "retries": res.retries,
            "finish_reason": res.finish_reason,
            "problems": len(res.problems),
            **_result_metadata(res),
            **(extra or {}),
        },
    )
    if isinstance(obs, PortObservation):
        obs.attach(res)
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
    output: type[T] = str,  # type: ignore[assignment]  # omitted, T is str; no TypeVar default spells it
    generator: Generator,
    hooks: Hooks | None = None,
    system: str = "",
    images: Sequence[tuple[bytes, str]] = (),
    validate: Callable[[T], Sequence[Any]] | None = None,
    retry_message: Callable[[Sequence[Any]], str] | None = None,
    settings: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    web: bool = False,
    counter: TokenCounter | None = None,
) -> Generated[T]:
    """Run one generation for `role` through `generator`.

    The path and the hooks are described in the module docstring. `web=True` says the role's tier
    fetches pages the prompt never rendered: like an image, an input no text grader sees, so
    `call.graders.TEXT_ONLY_GRADERS` are dropped for this call. The role itself is unchanged.
    `counter` counts the prompt for the role's pre-flight (`Role.context`, ADR-0024); the role's
    `cache_prefix` is translated by the tier's `cache` capability.

    Returns:
        The result, with a named outcome. A model, wire or gate failure is an outcome, never an
        exception; so is the deadline running out (`timeout`, `phase="total"`).

    Raises:
        RuntimeError: Only when a trace hook swallowed a generator exception, leaving no result.
        asyncio.CancelledError: The caller cancelled the call.
    """
    h = hooks or Hooks()
    seen = graders_for_inputs(tuple(role.graders), has_images=bool(images), has_web=web)
    if seen != tuple(role.graders):
        role = replace(role, graders=seen)
    loop = asyncio.get_running_loop()
    clocks = Deadline(total_s=role.resolved_timeout_s).start(loop.time())
    route = Route(role, clocks)
    try:
        async with asyncio.timeout_at(clocks.at + DEADLINE_GRACE_S):
            for tier_role in chain_of(role):
                if not route.next_tier(tier_role, loop.time(), h.on_event or _log_event):
                    break
                req = request_for(tier_role, prompt=prompt, system=system, images=images, settings=settings)
                req = replace(req, deadline=clocks)
                req, pre, res = await prepared(tier_role, req, counter, h.on_event or _log_event)
                if res is None:
                    gen = h.generator_for(tier_role.tier) if h.generator_for is not None else generator
                    res = await _one(tier_role, req, prompt, output, gen, h, validate, retry_message, metadata, web)
                    res = finished(res, pre, h.on_event or _log_event, tier_role.name)
                if not route.settle(tier_role, res):
                    break
    except TimeoutError:
        route.expire(loop.time(), h.on_event or _log_event)
    return await _finish(role, prompt, route, h)


async def _finish(role: Role, prompt: str, route: Route, h: Hooks) -> Generated[Any]:
    """Stamp, report and return the route's result; an expiry with no result is a `timeout`."""
    res = route.result if route.result is not None else timed_out(role, route)
    res = _stamped(role, res)
    if route.result is None:
        _event(
            h,
            logging.WARNING,
            "llm_call_failed",
            role=role.name,
            outcome=res.outcome,
            kind="timeout",
            reason=res.reason,
        )
    await _report(role, prompt, res, h)
    return res


async def _one[T](
    role: Role,
    req: GenRequest,
    prompt: str,
    output: type[T],
    generator: Generator,
    h: Hooks,
    validate: Callable[[T], Sequence[Any]] | None,
    retry_message: Callable[[Sequence[Any]], str] | None,
    metadata: Mapping[str, Any] | None,
    web: bool,
) -> Generated[T]:
    """One tier's call: the capability check, the gate, the trace, the generator, the graders."""
    with llm_feature(role.category):
        refused = _unsupported(role, req, output, web, h) or _gated(role, req, h)
        if refused is not None:
            return _stamped(role, refused)
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
                kind=res.failure.kind if res and res.failure else None,
                reason=res.reason if res else None,
            )
        if res is None:  # reachable only when a trace hook swallowed a generator exception
            raise RuntimeError(f"llm_call {role.name}: the generator returned nothing")
        return res


__all__ = [
    "DEADLINE_GRACE_S",
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
