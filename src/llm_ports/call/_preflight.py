"""The pre-flight context check both call paths run before a request is sent (ADR-0024).

`preflight` counts the request with the caller's `TokenCounter` and reads the verdict against the
role's budget (the model table's `context_window`, less the role's reply ceiling and a margin):

- `Role.context="off"` (the default) does nothing and counts nothing.
- `check` refuses an overflow before the wire: `request_fault` / `context_overflow`,
  `provider_code="preflight"`, zero generation requests.
- `trim` cuts the history (`context.trim`), counts again to confirm, cuts every turn the policy
  allows when the first cut was not enough, and refuses what still does not fit.

An unknown verdict (no counter, a model that cannot be counted, a count request that failed, or
no window in the table) SENDS the request: a missing optimisation is never an outage. Each count
request is reported (`count_requests`), and so is what trimming removed.

Invariants:
    - Never raises for a counter failure: a counter that raises is a failed count, `unknown`.
    - A count is never estimated.

Rationale: ADR-0024.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_ports.call.context import Budget, Fit, TrimPolicy, Trimmed, fit, trim
from llm_ports.call.roles import Role
from llm_ports.observability import EventHook
from llm_ports.ports import CallWarning, Counted, Failure, Message, TokenCounter, Tool, failure

#: Models already warned about in this process for an unknown fit (`llm_call_warning`, once each).
_UNKNOWN_SEEN: set[str] = set()


@dataclass(frozen=True, slots=True)
class Preflight:
    """What the pre-flight decided.

    Attributes:
        messages: The history to send (trimmed when it was).
        refused: The `context_overflow` failure; None when the call proceeds.
        count_requests: Count requests made.
        trimmed: What trimming removed; None when nothing was trimmed.
        fit: The last verdict; None when no count was taken.
        warnings: What the result should carry.
    """

    messages: tuple[Message, ...]
    refused: Failure | None = None
    count_requests: int = 0
    trimmed: Trimmed | None = None
    fit: Fit | None = None
    warnings: tuple[CallWarning, ...] = field(default=())

    def stamp(self) -> dict[str, Any]:
        """The result stamp entries: `count_requests` always, `trimmed` when something was cut."""
        out: dict[str, Any] = {"count_requests": self.count_requests}
        if self.trimmed is not None:
            out["trimmed"] = self.trimmed.as_stamp()
        return out


async def _count(
    counter: TokenCounter | None,
    model: str,
    system: str,
    messages: Sequence[Message],
    tools: Sequence[Tool],
    emit: EventHook,
    role: str,
) -> tuple[Counted, int]:
    if counter is None:
        return Counted(None, "unsupported"), 0
    try:
        got = await counter.count(model, system=system, messages=messages, tools=tools)
    except Exception as exc:  # a counter that broke its never-raise contract is a failed count
        got = Counted(None, "provider", failure=failure("internal", reason=f"{type(exc).__name__}: {exc}"))
    if got.failure is not None:
        emit(logging.WARNING, "llm_count_failed", {"role": role, "model": model, "kind": got.failure.kind})
    return got, 1


def _overflow(f: Fit, model: str) -> Failure:
    return failure(
        "context_overflow",
        provider_code="preflight",
        reason=f"preflight: {f.tokens} input tokens over the {f.limit} limit of {model}",
    )


async def preflight(
    role: Role,
    counter: TokenCounter | None,
    *,
    system: str,
    messages: Sequence[Message],
    tools: Sequence[Tool] = (),
    emit: EventHook,
) -> Preflight:
    """Check (and on `trim`, cut) one request against its model's window; the module docstring states the rules."""
    msgs = tuple(messages)
    if role.context == "off":
        return Preflight(msgs)
    model = role.resolved_model
    info = role.resolved_tier.info
    reserve = role.max_tokens or (info.max_output if info is not None else None) or 0
    budget = Budget(info.context_window if info is not None else None, reserve_output=reserve)
    counted, n = await _count(counter, model, system, msgs, tools, emit, role.name)
    verdict = fit(counted, budget)
    if verdict.verdict == "unknown":
        why = "no exact count" if counted.tokens is None else "no context_window in the table"
        warn = CallWarning("other", "context", f"{why} for {model}; sent unchecked")
        if model not in _UNKNOWN_SEEN:
            _UNKNOWN_SEEN.add(model)
            emit(logging.WARNING, "llm_call_warning", {"kind": "other", "setting": "context", "detail": warn.detail})
        return Preflight(msgs, count_requests=n, fit=verdict, warnings=(warn,))
    if verdict.verdict == "fits":
        return Preflight(msgs, count_requests=n, fit=verdict)
    if role.context == "check":
        emit(logging.WARNING, "llm_context_overflow", {"role": role.name, "where": "preflight", "tokens": verdict.tokens, "limit": verdict.limit})
        return Preflight(msgs, refused=_overflow(verdict, model), count_requests=n, fit=verdict)
    policy = role.trim or TrimPolicy()
    cut = trim(msgs, policy, over_by_tokens=verdict.over_by)
    again, m = await _count(counter, model, system, cut.messages, tools, emit, role.name)
    n += m
    verdict = fit(again, budget)
    if verdict.verdict == "over":
        cut = trim(msgs, policy, over_by_tokens=None)
        again, m = await _count(counter, model, system, cut.messages, tools, emit, role.name)
        n += m
        verdict = fit(again, budget)
    if cut.changed:
        emit(logging.INFO, "llm_trimmed", {"role": role.name, **cut.as_stamp()})
    warns = (
        (CallWarning("clamped", "messages", f"history trimmed: {cut.dropped_messages} messages dropped, {cut.cut_tool_outputs} tool outputs cut"),)
        if cut.changed
        else ()
    )
    if verdict.verdict == "over":
        emit(logging.WARNING, "llm_context_overflow", {"role": role.name, "where": "preflight", "tokens": verdict.tokens, "limit": verdict.limit})
        return Preflight(msgs, refused=_overflow(verdict, model), count_requests=n, trimmed=cut, fit=verdict, warnings=warns)
    return Preflight(cut.messages, count_requests=n, trimmed=cut if cut.changed else None, fit=verdict, warnings=warns)


def optimistic(pre: Preflight, res_failure: Failure | None, emit: EventHook, role: str) -> None:
    """Emit `llm_count_optimistic` when a call the pre-flight passed still overflowed on the wire."""
    if res_failure is None or res_failure.kind != "context_overflow" or res_failure.provider_code == "preflight":
        return
    emit(logging.WARNING, "llm_context_overflow", {"role": role, "where": "wire"})
    if pre.fit is not None and pre.fit.verdict == "fits":
        emit(logging.WARNING, "llm_count_optimistic", {"role": role, "tokens": pre.fit.tokens, "limit": pre.fit.limit})


__all__: list[str] = []
