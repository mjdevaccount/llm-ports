"""Read labelled examples: how many per topic, who agrees with the human, which to show a judge.

- `coverage`: per topic, how many distinct subjects each source labelled, against a target. Every
  topic asked about appears, an unlabelled one included: the gap is the point.
- `agreement`: per topic, how well each measured source (a judge, a model labeller) agrees with
  the reference source (a human), as Cohen's kappa and Gwet's AC1 on the subjects both labelled.
- `few_shot`: a deterministic, class-balanced pick of examples, to put graded examples of a
  topic in front of a judge.

Invariants:
    - Pure: no store, no clock.
    - One source's labels on one subject count once: the latest given wins (a relabel replaces).
    - A statistic the examples cannot define is None, never a number.

Rationale: ADR-0010.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from llm_ports.learn._types import Example, Source
from llm_ports.learn.stats import YES, cohen_kappa, gwet_ac1

SOURCES: tuple[Source, ...] = ("human", "model", "judge", "outcome")
_EPOCH = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True)
class Coverage:
    """One topic's labelled examples.

    Attributes:
        topic: The topic.
        counts: Distinct subjects labelled, per source (every source present, zeros included).
        target: The reference-source labels the topic should have.
        reference: The source the target counts (`human` unless told otherwise).
        positive: Of the reference-labelled subjects, how many carry the positive class.
    """

    topic: str
    counts: Mapping[str, int]
    target: int
    reference: Source = "human"
    positive: int = 0

    @property
    def have(self) -> int:
        """Subjects the reference source labelled."""
        return self.counts.get(self.reference, 0)

    @property
    def gap(self) -> int:
        """Reference labels still missing to reach `target`; never negative."""
        return max(0, self.target - self.have)

    @property
    def covered(self) -> bool:
        """True when the topic has its target of reference labels."""
        return self.have >= self.target


@dataclass(frozen=True)
class Agreement:
    """How one source agrees with the reference source on one topic.

    Attributes:
        n: Subjects both labelled.
        agree: The share of those on which they agree; None when `n` is 0.
        kappa: Cohen's kappa; None where it is undefined (`stats.cohen_kappa`).
        ac1: Gwet's AC1; None under two subjects.
    """

    n: int
    agree: float | None
    kappa: float | None
    ac1: float | None


def latest(examples: Iterable[Example]) -> dict[tuple[str, str, str], Example]:
    """Return one example per `(topic, subject, source)`: the latest given (by `at`).

    An example with no `at` sorts before every dated one; among equals the later in `examples` wins.
    """
    out: dict[tuple[str, str, str], Example] = {}
    for ex in examples:
        k = (ex.topic, ex.subject, ex.source)
        prev = out.get(k)
        if prev is None or (ex.at or _EPOCH) >= (prev.at or _EPOCH):
            out[k] = ex
    return out


def coverage(
    examples: Iterable[Example],
    topics: Iterable[str],
    *,
    target: int | Mapping[str, int],
    reference: Source = "human",
    positive: str = "good",
) -> dict[str, Coverage]:
    """Return each topic's `Coverage`.

    Args:
        examples: Examples from any source.
        topics: The topics to report; a topic with no example reports zeros. A labelled topic
            missing from `topics` is reported too, so an example is never silently dropped.
        target: The reference labels each topic should have, one number for all or a per-topic
            mapping (a topic missing from the mapping gets 0).
        reference: The source the target counts.
        positive: The class counted in `Coverage.positive`.

    Returns:
        `{topic: Coverage}`, in the order of `topics`, then any extra labelled topics sorted.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(SOURCES, 0))
    good: dict[str, int] = defaultdict(int)
    for (topic, _subject, source), ex in latest(examples).items():
        counts[topic][source] += 1
        if source == reference and ex.value == positive:
            good[topic] += 1
    order = list(dict.fromkeys(topics))
    order += sorted(set(counts) - set(order))

    def _target(topic: str) -> int:
        return target.get(topic, 0) if isinstance(target, Mapping) else target

    return {t: Coverage(t, dict(counts[t]), _target(t), reference, good[t]) for t in order}


def agreement(
    examples: Iterable[Example],
    *,
    reference: Source = "human",
    measured: Sequence[Source] = ("judge", "model"),
    positive: str = "good",
) -> dict[str, dict[str, Agreement]]:
    """Return, per topic, each measured source's `Agreement` with the reference source.

    Labels are read as binary: `positive` or not. Only subjects both sources labelled count. A
    topic with no reference label is absent; a source with no shared subject reports `n == 0`.

    Returns:
        `{topic: {source: Agreement}}`.
    """
    by: dict[tuple[str, str], dict[str, Example]] = defaultdict(dict)
    for (topic, subject, source), ex in latest(examples).items():
        by[(topic, source)][subject] = ex
    out: dict[str, dict[str, Agreement]] = {}
    for (topic, source), ref in sorted(by.items()):
        if source != reference:
            continue
        out[topic] = {}
        for m in measured:
            theirs = by.get((topic, m), {})
            pairs = [
                (_yes(theirs[s].value, positive), _yes(ref[s].value, positive)) for s in sorted(ref) if s in theirs
            ]
            agree = sum(a == b for a, b in pairs) / len(pairs) if pairs else None
            out[topic][m] = Agreement(len(pairs), agree, cohen_kappa(pairs), gwet_ac1(pairs))
    return out


def few_shot(examples: Iterable[Example], k: int, *, topic: str, source: Source = "human") -> list[Example]:
    """Return up to `k` examples of one topic from one source, balanced across classes.

    The pick is deterministic (a hash of the subject orders each class) and round-robin across
    the label values, rarest class first, so a judge shown the examples sees every class the
    examples hold. Numeric labels are one class each, so a numeric topic is simply a stable pick.

    Args:
        examples: Examples from any topic and source.
        k: The most to return.
        topic: The topic to pick from.
        source: The source to pick from (`human` by default: ground truth).

    Returns:
        The picked examples, at most `k`. Never raises.
    """
    classes: dict[str, list[Example]] = defaultdict(list)
    mine = [ex for (t, _s, src), ex in latest(examples).items() if t == topic and src == source]
    for ex in sorted(mine, key=lambda e: _digest(e.subject)):
        classes[str(ex.value)].append(ex)
    ordered = sorted(classes.values(), key=len)
    out: list[Example] = []
    while len(out) < k and any(ordered):
        for bucket in ordered:
            if bucket and len(out) < k:
                out.append(bucket.pop(0))
    return out


def _yes(value: str | float, positive: str) -> str:
    """Return `stats.YES` when `value` is the positive class, else `no`."""
    return YES if value == positive else "no"


def _digest(text: str) -> str:
    """Return a stable hex digest of `text`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["SOURCES", "Agreement", "Coverage", "agreement", "coverage", "few_shot", "latest"]
