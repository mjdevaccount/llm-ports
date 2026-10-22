"""context -- WHICH LLM category is running, and WHOSE traffic it is, as context variables every
plane reads (the tracer, the category switchboard, a cost ledger, the work row).

A category tag is not a tracing concern: the cost ledger, the switchboard and the tracer all read
it, and none of them owns it.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_feature_var: ContextVar[str | None] = ContextVar("langfuse_feature", default=None)


@contextmanager
def llm_feature(name: str) -> Iterator[None]:
    """Tag every LLM generation emitted within this block with a category. Propagates
    through async/await; nests (inner wins). No-op cost when tracing is disabled.

    SAFE AROUND ASYNC GENERATORS. `ContextVar.reset(token)` raises ValueError when the token
    was created in a DIFFERENT Context — which is exactly what happens if this wraps a body
    that `yield`s and the generator is later closed (or garbage-collected) from somewhere other
    than the task that entered. That is not hypothetical here: the copilot brains are streaming
    async generators consumed by a StreamingResponse. Falling back to clearing the var keeps a
    stale category from leaking into whatever runs next, without turning generator teardown into
    an exception.
    """
    token = _feature_var.set(name)
    try:
        yield
    finally:
        try:
            _feature_var.reset(token)
        except ValueError:
            _feature_var.set(None)

def current_feature() -> str | None:
    """The active category, if any — for callers that also persist it (e.g. a transport that writes a work row)."""
    return _feature_var.get()


# ── traffic source: live production calls vs an eval REPLAY ───────────────────────────────────
#: What a generation with no source set is: the role's own producer, serving the operator.
LIVE_SOURCE = "live"
_source_var: ContextVar[str | None] = ContextVar("llm_source", default=None)


@contextmanager
def llm_source(name: str) -> Iterator[None]:
    """Tag every generation inside this block as traffic of `name` -- an eval harness sets e.g.
    `"prompt_eval"` around a golden replay so its failed graders write NO live gauge: on the live
    role's series a `rate()` alert would read a replay as production breaking. Same teardown rule as
    `llm_feature` (a token from another Context is cleared, not reset)."""
    token = _source_var.set(name)
    try:
        yield
    finally:
        try:
            _source_var.reset(token)
        except ValueError:
            _source_var.set(None)


def current_source() -> str:
    """`"live"` unless an `llm_source` block says otherwise; never None, so a label is always set."""
    return _source_var.get() or LIVE_SOURCE

