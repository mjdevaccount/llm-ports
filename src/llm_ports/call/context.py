"""Context: whether a request fits its model's window, what is cut when it does not, and cache hints.

Pure functions, no I/O. The call path (`call.chat`, `call.generate`) counts a request with a
`TokenCounter` before it is sent, reads the verdict from `fit`, and on `Role.context="trim"` cuts
the history with `trim`.

- A count is exact or absent (`ports.Counted`): no character-ratio estimate is ever a token count.
  An unknown fit sends the request; it never blocks.
- Trimming is deterministic, structural and reported. Order: tool outputs over the policy's
  budget are cut in the middle (`cut_middle`); then whole turns are dropped oldest-first between
  `keep_first` and `keep_last`, never splitting a tool call from its result. The system prompt is
  never cut.
- A cache hint is a request, not a guarantee (`cache_plan`); `Usage.cache_read` on the result is
  the proof.

Rationale: ADR-0024.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from llm_ports.call.models import ModelInfo
from llm_ports.ports import OMITTED_MARKER, CallWarning, Counted, Message, Part
from llm_ports.prompts import Pack

#: What a role does about context before a call: nothing, refuse an overflow, or trim to fit.
ContextMode = Literal["off", "check", "trim"]


@dataclass(frozen=True, slots=True)
class Budget:
    """The input tokens a call may send.

    Attributes:
        context_window: The model's window, from its `ModelInfo`; None when unknown.
        reserve_output: The tokens held for the reply (the role's `max_tokens`).
        margin: The fraction of the rest held back for counting error.
    """

    context_window: int | None
    reserve_output: int
    margin: float = 0.05

    @property
    def input_limit(self) -> int | None:
        """`(context_window - reserve_output) * (1 - margin)`, rounded down; None when the window is unknown."""
        if self.context_window is None:
            return None
        return max(0, math.floor((self.context_window - self.reserve_output) * (1.0 - self.margin)))


@dataclass(frozen=True, slots=True)
class Fit:
    """Whether a counted request fits its budget.

    Attributes:
        verdict: `fits`, `over`, or `unknown` (no count, or no window).
        tokens: The count; None when there was none.
        limit: The budget's input limit; None when the window is unknown.
    """

    verdict: Literal["fits", "over", "unknown"]
    tokens: int | None
    limit: int | None

    @property
    def over_by(self) -> int | None:
        """Tokens over the limit; None unless the verdict is `over`."""
        if self.verdict != "over" or self.tokens is None or self.limit is None:
            return None
        return self.tokens - self.limit


def fit(counted: Counted, budget: Budget) -> Fit:
    """Return whether `counted` fits `budget`. Pure; an absent count or window is `unknown`."""
    limit = budget.input_limit
    if counted.tokens is None or limit is None:
        return Fit("unknown", counted.tokens, limit)
    return Fit("fits" if counted.tokens <= limit else "over", counted.tokens, limit)


@dataclass(frozen=True, slots=True)
class TrimPolicy:
    """How a history is cut to fit.

    Attributes:
        keep_first: Leading messages never dropped (the task statement).
        keep_last: Trailing messages never dropped.
        tool_output_chars: The longest tool result kept whole; a longer one is cut in the middle.
        drop: What is dropped after the tool outputs: whole turns, oldest first.
    """

    keep_first: int = 1
    keep_last: int = 4
    tool_output_chars: int = 4000
    drop: Literal["oldest_turns"] = "oldest_turns"


@dataclass(frozen=True, slots=True)
class Trimmed:
    """What `trim` returned and what it removed.

    Attributes:
        messages: The cut history.
        dropped_messages: Whole messages dropped.
        cut_tool_outputs: Tool results cut in the middle.
        chars_removed: Characters removed, over both.
    """

    messages: tuple[Message, ...]
    dropped_messages: int
    cut_tool_outputs: int
    chars_removed: int

    @property
    def changed(self) -> bool:
        """True when anything was removed."""
        return bool(self.dropped_messages or self.cut_tool_outputs)

    def as_stamp(self) -> dict[str, int]:
        """The counts, as `stamp["trimmed"]` carries them."""
        return {
            "dropped_messages": self.dropped_messages,
            "cut_tool_outputs": self.cut_tool_outputs,
            "chars_removed": self.chars_removed,
        }


def cut_middle(text: str, max_chars: int) -> tuple[str, int]:
    """Cut `text` to at most `max_chars`, keeping its head and its tail. Pure.

    The tail usually holds the answer or the error, so the middle goes; the marker
    `[... N chars omitted]` says how much.

    Returns:
        The text, and the characters removed (0 when it already fit).
    """
    if len(text) <= max_chars:
        return text, 0
    keep = max(0, max_chars - len(OMITTED_MARKER.format(n=len(text))))
    head = keep - keep // 2
    tail = keep // 2
    removed = len(text) - head - tail
    return text[:head] + OMITTED_MARKER.format(n=removed) + (text[len(text) - tail :] if tail else ""), removed


def _cut_tools(messages: Sequence[Message], max_chars: int) -> tuple[list[Message], int, int]:
    out: list[Message] = []
    cut = removed = 0
    for m in messages:
        parts: list[Part] = []
        for p in m.parts:
            if p.kind == "tool_result" and p.text is not None and len(p.text) > max_chars:
                text, n = cut_middle(p.text, max_chars)
                parts.append(replace(p, text=text))
                cut += 1
                removed += n
            else:
                parts.append(p)
        out.append(replace(m, parts=tuple(parts)) if parts != list(m.parts) else m)
    return out, cut, removed


def _chars(m: Message) -> int:
    return sum(len(p.text or "") + len(p.data or b"") for p in m.parts)


def _turns(messages: Sequence[Message]) -> list[tuple[int, int]]:
    """Split into turns `[start, end)`: each starts at a user message; a tool result stays with its call."""
    starts = [i for i, m in enumerate(messages) if m.role == "user"]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    return [(s, e) for s, e in zip(starts, [*starts[1:], len(messages)], strict=True) if s < e]


def trim(messages: Sequence[Message], policy: TrimPolicy, *, over_by_tokens: int | None) -> Trimmed:
    """Cut `messages` by `policy`. Pure and deterministic.

    Tool results over `policy.tool_output_chars` are cut in the middle first. Then whole turns
    (a user message and every message up to the next one) are dropped oldest-first, while every
    message of the turn lies between the first `keep_first` and the last `keep_last`. A turn is
    never split, so a tool call is never separated from its result.

    Args:
        messages: The history.
        policy: What may be cut.
        over_by_tokens: How far over the budget the history is. Turns are dropped until at least
            that many characters are gone (a token is at least one character, so fewer cannot be
            enough; the caller counts again to confirm). None drops every turn the policy allows.

    Returns:
        The cut history and what was removed.
    """
    out, cut, removed = _cut_tools(messages, policy.tool_output_chars)
    if over_by_tokens is not None and removed >= over_by_tokens:
        return Trimmed(tuple(out), 0, cut, removed)
    floor, ceil = policy.keep_first, len(out) - policy.keep_last
    drop: set[int] = set()
    for start, end in _turns(out):
        if over_by_tokens is not None and removed >= over_by_tokens:
            break
        if start >= floor and end <= ceil:
            drop.update(range(start, end))
            removed += sum(_chars(out[i]) for i in range(start, end))
    kept = tuple(m for i, m in enumerate(out) if i not in drop)
    return Trimmed(kept, len(drop), cut, removed)


def cache_plan(
    info: ModelInfo | None, *, wanted: bool, settings: Mapping[str, Any]
) -> tuple[bool, dict[str, Any], tuple[CallWarning, ...]]:
    """Translate a cache hint by the model's stated capability. Pure.

    Args:
        info: The model; None (a schema 1 tier) states nothing, and the hint passes to the adapter.
        wanted: Whether the call asked for a cache breakpoint.
        settings: The call's settings.

    Returns:
        Whether the adapter still places breakpoints, the settings to send, and warnings. A model
        without `cache` drops the hint with a `CallWarning(kind="unsupported")`. A self-hosted
        model with `cache` caches prefixes by slot: it gets `extra_body.cache_prompt` and no
        breakpoint. A hosted model with `cache` keeps the hint for the adapter.
    """
    out = dict(settings)
    if not wanted or info is None:
        return wanted, out, ()
    if "cache" not in info.supports:
        warn = CallWarning("unsupported", "cache", f"model {info.key!r} does not state cache; the hint is dropped")
        return False, out, (warn,)
    if info.self_hosted:
        out["extra_body"] = {**(out.get("extra_body") or {}), "cache_prompt": True}
        return False, out, ()
    return True, out, ()


def volatile_system(pack: Pack, contexts: Sequence[Mapping[str, Any]], *, template: str = "system") -> bool:
    """True when the pack's system template renders differently across `contexts`.

    A system text that interpolates a per-call value defeats a cache breakpoint after it: lint a
    pack whose role sets `cache_prefix` with two representative contexts.

    Args:
        pack: The pack.
        contexts: Render contexts; two or more.
        template: The system template's name in the manifest.

    Returns:
        False when the pack has no such template.
    """
    if template not in pack.manifest.templates:
        return False
    return len({pack.render(template, **ctx) for ctx in contexts}) > 1


__all__ = [
    "Budget",
    "ContextMode",
    "Fit",
    "TrimPolicy",
    "Trimmed",
    "cache_plan",
    "cut_middle",
    "fit",
    "trim",
    "volatile_system",
]
