"""Labelled LLM replies on the learning-loop core: topics, the review-queue draw, and judge labels.

A label on a model reply is a `learn.Example` whose topic is `llm:<role>` (`topic`), whose
subject is the reply's id in the label store, and whose source is `human` (ground truth),
`model` (a stronger model labelling offline) or `judge` (the sampled production judge). Counting
and agreement are the core's (`learn.examples.coverage`, `learn.examples.agreement`); this module
holds only what is specific to model replies:

- `draw`: which unlabelled production replies to put in front of a human next. Per role it takes
  only what closes the role's coverage gap, deterministically, and stratifies by the judge's
  verdict so both classes reach the human: an agreement statistic needs both, and a judge that
  fails most replies would otherwise fill the queue with its fails.
- `label_of`: any `Judge`'s verdict as an example. The offline model labeller is a `Judge` with a
  stronger model behind it, so no second judging path exists.
- `LabelQueue`: the port a review surface implements to receive the drawn replies.

Invariants:
    - Pure, except `LabelQueue` implementations: no model, no store, no clock.
    - A model's verdict is never a `human` example.

Rationale: ADR-0010.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from llm_ports.learn import Example, StoreOutcome, Written
from llm_ports.learn.examples import Coverage
from llm_ports.ports import JudgeVerdict

#: The topic prefix of a labelled model reply.
PREFIX = "llm:"


def topic(role: str) -> str:
    """Return the topic of `role`'s replies (`llm:<role>`)."""
    return f"{PREFIX}{role}"


def role_of(name: str) -> str | None:
    """Return the role a topic names, or None for a topic that is not a model reply's."""
    return name[len(PREFIX) :] or None if name.startswith(PREFIX) else None


@dataclass(frozen=True)
class Candidate:
    """An unlabelled production reply that could be put in front of a human.

    Attributes:
        role: The role that produced it.
        subject: Its id in the label store.
        trace: The trace it belongs to ("" where the store does not scope ids by trace).
        judged: The judge's verdict on it (`pass`, `fail`), or None when it was not judged.
    """

    role: str
    subject: str
    trace: str = ""
    judged: str | None = None


@dataclass(frozen=True)
class Queued:
    """What `LabelQueue.queued` returns: the subjects a topic's queue holds.

    Attributes:
        pending: Subjects waiting for a human.
        done: Subjects a human has completed.
        outcome: `ok`, `unavailable` when the queue could not be read, or `request_fault` for a
            topic that is not a model reply's.
        reason: One line naming the failure; None on `ok`.

    Invariants:
        - An unreadable queue is `unavailable`, never an empty one.
        - A topic with no queue yet is `ok` and empty.
    """

    pending: frozenset[str] = frozenset()
    done: frozenset[str] = frozenset()
    outcome: StoreOutcome = "ok"
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """True when `outcome == "ok"`."""
        return self.outcome == "ok"

    @property
    def seen(self) -> frozenset[str]:
        """Every subject the queue holds, pending or done: what `draw` should skip."""
        return self.pending | self.done


@runtime_checkable
class LabelQueue(Protocol):
    """A review surface where a human labels queued replies (an annotation queue)."""

    async def enqueue(self, topic: str, subjects: Sequence[tuple[str, str]]) -> Written:
        """Put replies in front of a human labeller.

        Args:
            topic: The replies' topic (`topic(role)`); the queue is per topic.
            subjects: `(subject, trace)` pairs.

        Returns:
            How many were newly queued; a subject the queue already holds is skipped, not added
            twice and not an error. Never raises.
        """
        ...

    async def queued(self, topic: str) -> Queued:
        """Read the subjects a topic's queue holds, so a caller can skip them and bound the backlog.

        Returns:
            The pending and done subjects, or an `unavailable` outcome. Never raises.
        """
        ...


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _stratified(pool: Sequence[Candidate], n: int) -> list[Candidate]:
    """Take `n` from `pool` round-robin over the judge's verdict, rarest stratum first."""
    strata: dict[str | None, list[Candidate]] = defaultdict(list)
    for c in sorted(pool, key=lambda c: _digest(f"{c.role}\n{c.subject}")):
        strata[c.judged].append(c)
    ordered = sorted(strata.values(), key=len)
    out: list[Candidate] = []
    while len(out) < n and any(ordered):
        for s in ordered:
            if s and len(out) < n:
                out.append(s.pop(0))
    return out


def draw(
    candidates: Iterable[Candidate],
    coverage_now: Mapping[str, Coverage],
    *,
    per_run: int,
    seen: Iterable[str] = (),
) -> list[Candidate]:
    """Return the replies to queue for a human next.

    Per role, at most `min(per_run, gap)` replies, none in `seen` (already labelled or queued).
    A role whose topic is absent from `coverage_now` is not drawn. Within a role the draw is
    deterministic (a hash of role and subject) and round-robin across the judge's verdicts,
    rarest first, so a queue holds both classes whenever the candidates do.

    Args:
        candidates: Unlabelled production replies.
        coverage_now: `learn.examples.coverage` keyed by topic, which carries each role's gap.
        per_run: The most replies one run queues per role.
        seen: Subjects to skip.

    Returns:
        The replies to queue, grouped by role in the order of `coverage_now`. Never raises.
    """
    skip = set(seen)
    by_role: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        if c.subject not in skip:
            by_role[c.role].append(c)
            skip.add(c.subject)
    out: list[Candidate] = []
    for name, cov in coverage_now.items():
        role = role_of(name)
        n = min(per_run, cov.gap)
        if role is not None and n > 0 and by_role.get(role):
            out += _stratified(by_role[role], n)
    return out


def label_of(
    verdict: JudgeVerdict,
    *,
    role: str,
    subject: str,
    trace: str = "",
    by: str = "",
    source: Literal["model", "judge"] = "model",
    at: datetime | None = None,
) -> Example:
    """Return a `Judge`'s verdict on one reply as an example.

    Args:
        verdict: Any `Judge`'s verdict.
        role: The labelled reply's role.
        subject: The reply's id in the label store.
        trace: The reply's trace id, kept in `ref["trace"]` when given.
        by: The judging model's name; the verdict's judge when "".
        source: `model` for an offline labeller, `judge` for the production judge. Never `human`.
        at: When it was given.

    Returns:
        The example, `good` when the verdict passed, `bad` otherwise.

    Raises:
        ValueError: `source` is not `model` or `judge`: a model's verdict is never ground truth.
    """
    if source not in ("model", "judge"):
        raise ValueError(f"a model's verdict is never a {source!r} label")
    return Example(
        topic=topic(role),
        subject=subject,
        value="good" if verdict.passed else "bad",
        source=source,
        by=by or verdict.judge,
        at=at,
        reason=verdict.reason,
        ref={"trace": trace} if trace else {},
    )


__all__ = ["PREFIX", "Candidate", "LabelQueue", "Queued", "draw", "label_of", "role_of", "topic"]
