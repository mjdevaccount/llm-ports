"""The one end payload every observation path sends: `generate`, `observe`, and `record`.

`end_payload` turns what a call produced (or what a caller says an outside call produced) into the
keyword arguments of `Generation.end`, so no sink sees a field from one path that another omits.
`deliver` hands them to a generation; to a 1.x `Generation` whose `end` does not take the 1.24
keywords (`usage_v2`, `cost`, `finish`, `answered`, `failure`) it hands the 1.x five only.

Invariants:
    - Unmeasured is absent, never zero: a side nobody reported is None in `usage`; with neither
      side reported, `usage` is None and `metadata["tokens_unmeasured"]` is True.
    - `deliver` never raises (a `Generation` that raises is the sink's fault, not the call's).

Rationale: ADR-0007, ADR-0019, ADR-0025.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Mapping
from typing import Any

from llm_ports.ports import Answered, Cost, Failure, Finish, Generation, Usage

UNMEASURED_KEY = "tokens_unmeasured"
#: Characters of a failure's reason kept as a status message.
STATUS_MAX = 500
#: The keywords `Generation.end` gained in 1.24.
_V2_KEYS = ("usage_v2", "cost", "finish", "answered", "failure")
_ACCEPTS: dict[type, bool] = {}


def legacy_usage(usage: Usage | None) -> dict[str, int | None] | None:
    """The 1.x `{"input", "output"}` pair of `usage`; None when neither side was measured. Pure."""
    if usage is None or (usage.input is None and usage.output is None):
        return None
    return {"input": usage.input, "output": usage.output}


def end_payload(
    *,
    output: Any = None,
    usage: Usage | None = None,
    pair: Mapping[str, int | None] | None = None,
    cost: Cost | None = None,
    finish: Finish | None = None,
    answered: Answered | None = None,
    failure: Failure | None = None,
    metadata: Mapping[str, Any] | None = None,
    level: str | None = None,
    status_message: str | None = None,
) -> dict[str, Any]:
    """The keyword arguments of `Generation.end` for one finished call. Pure.

    Args:
        output: What the call produced.
        usage: Its usage, summed over every request; None when unknown.
        pair: The 1.x `{"input", "output"}` counts when the caller has only those; `usage`'s
            pair otherwise.
        cost: What it cost; None when not priced.
        finish: How its last reply ended.
        answered: Who answered.
        failure: Its failure; None when it succeeded.
        metadata: Extra fields for the sink.
        level: The level; None derives it: "ERROR" with a failure, else "DEFAULT".
        status_message: The status line; None derives it from the failure when the level is not
            "DEFAULT".

    Returns:
        The keyword arguments: `output`, `usage` and `metadata` always (as 1.x sent them);
        `level` and `status_message` only off the "DEFAULT" level; each 1.24 keyword only when
        it carries a value.
    """
    usage_pair = dict(pair) if pair is not None else legacy_usage(usage)
    if usage_pair is not None and usage_pair.get("input") is None and usage_pair.get("output") is None:
        usage_pair = None
    meta = dict(metadata or {})
    if usage_pair is None and (usage is not None or pair is not None):
        meta[UNMEASURED_KEY] = True
    if level is None:
        level = "ERROR" if failure is not None else "DEFAULT"
    if status_message is None and failure is not None and level != "DEFAULT":
        status_message = f"{failure.kind}: {failure.reason or ''}".rstrip(": ")[:STATUS_MAX]
    out: dict[str, Any] = {"output": output, "usage": usage_pair, "metadata": meta or None}
    if level != "DEFAULT" or status_message is not None:
        out["level"] = level
        out["status_message"] = status_message
    v2 = {"usage_v2": usage, "cost": cost, "finish": finish, "answered": answered, "failure": failure}
    out.update({k: v for k, v in v2.items() if v is not None})
    return out


def _accepts_v2(gen: Generation) -> bool:
    """Whether `gen.end` takes the 1.24 keywords (cached per class)."""
    kind = type(gen)
    if kind not in _ACCEPTS:
        try:
            params = inspect.signature(gen.end).parameters
        except (TypeError, ValueError):  # a builtin or C callable: assume the 1.x five only
            _ACCEPTS[kind] = False
        else:
            kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            _ACCEPTS[kind] = kw or all(k in params for k in _V2_KEYS)
    return _ACCEPTS[kind]


def deliver_or_raise(gen: Generation, payload: Mapping[str, Any]) -> None:
    """End `gen` with `payload`; a 1.x `Generation` gets its five keywords only. What `end` raises propagates."""
    kw = dict(payload) if _accepts_v2(gen) else {k: v for k, v in payload.items() if k not in _V2_KEYS}
    gen.end(**kw)


def deliver(gen: Generation, payload: Mapping[str, Any]) -> None:
    """`deliver_or_raise`, never raising (a `Generation` that raises is the sink's fault, not the call's)."""
    with contextlib.suppress(Exception):
        deliver_or_raise(gen, payload)


__all__: list[str] = []
