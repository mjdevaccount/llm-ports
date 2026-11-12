"""Context variables naming the running LLM category and whose traffic it is.

Every plane reads them: the tracer, the category switchboard, a cost ledger, a work row. None of
those planes owns the category tag, so it lives here rather than in tracing.

Invariants:
    - Both tags propagate through async/await and nest (the inner block wins).
    - Leaving a block never raises, even when the block is torn down in another `Context`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_feature_var: ContextVar[str | None] = ContextVar("llm_feature", default=None)


@contextmanager
def llm_feature(name: str) -> Iterator[None]:
    """Tag every LLM generation emitted within this block with the category `name`.

    Propagates through async/await and nests (the inner block wins). Costs nothing when tracing
    is disabled.

    Safe around async generators. `ContextVar.reset(token)` raises `ValueError` when the token was
    created in a different `Context`, which happens when the block wraps a body that yields and
    the generator is closed or garbage-collected outside the task that entered it (a streaming
    response that consumes an async generator, for example). On that error the variable is cleared
    instead, so a stale category never leaks into whatever runs next.
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
    """Return the active category, or None outside any `llm_feature` block."""
    return _feature_var.get()


# ── traffic source: live calls vs an eval replay ───────────────────────────────────────────────
#: The source of a generation that no `llm_source` block tagged: the role's own live traffic.
LIVE_SOURCE = "live"
_source_var: ContextVar[str | None] = ContextVar("llm_source", default=None)


@contextmanager
def llm_source(name: str) -> Iterator[None]:
    """Tag every generation inside this block as traffic of `name`.

    An eval harness sets e.g. `"prompt_eval"` around a golden replay, so its failed graders write
    no live gauge and a rate alert on the live series never reads a replay as a live failure.
    Teardown follows `llm_feature`: a token from another `Context` is cleared, not reset.
    """
    token = _source_var.set(name)
    try:
        yield
    finally:
        try:
            _source_var.reset(token)
        except ValueError:
            _source_var.set(None)


def current_source() -> str:
    """Return the active traffic source: `LIVE_SOURCE` unless an `llm_source` block says otherwise.

    Never None, so a metric label is always set.
    """
    return _source_var.get() or LIVE_SOURCE


__all__ = ["LIVE_SOURCE", "current_feature", "current_source", "llm_feature", "llm_source"]
