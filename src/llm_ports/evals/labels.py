"""Labelled data: how much there is per role, which reply to label next, and who agrees with whom.

A `Label` (`ports`) is one verdict on one reply from one labeller: a human (ground truth), a
stronger model labelling offline, or the sampled production judge. This module reads labels from
any `LabelStore` and answers three questions:

- `coverage`: per role, how many distinct replies each labeller has labelled, against a target.
  Every role asked about appears, a role with no label included: the gap is the point.
- `draw`: which unlabelled production replies to put in front of a human next. Per role it takes
  only what closes the gap, deterministically, and stratifies by the judge's verdict so both
  classes reach the human: an agreement statistic needs both, and a judge that fails most replies
  would otherwise fill the queue with its fails.
- `agreement`: per role, how often the judge and the model labeller agree with the human, as
  Cohen's kappa and Gwet's AC1 (`calibration`), on the replies both labelled.

`label_of` turns any `Judge`'s verdict into a label: the offline model labeller is a `Judge`
with a stronger model behind it, so no second judging path exists.

Invariants:
    - Pure: no model, no store, no clock.
    - One labeller's labels on one reply count once: the latest given wins (a relabel replaces).
    - A statistic the labels cannot define is None, never a number.

Rationale: ADR-0010.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from llm_ports.evals.calibration import YES, cohen_kappa, gwet_ac1
from llm_ports.ports import JudgeVerdict, Label, Labeller
from llm_ports.prompts import sha

#: The labellers measured against the human.
MEASURED: tuple[Labeller, ...] = ("judge", "model")
_EPOCH = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True)
class Coverage:
    """One role's labelled data.

    Attributes:
        role: The role.
        human: Distinct replies a human labelled.
        model: Distinct replies the model labeller labelled.
        judge: Distinct replies the production judge labelled.
        target: The human labels the role should have.
        human_good: Of the human-labelled replies, how many were good.
    """

    role: str
    human: int
    model: int
    judge: int
    target: int
    human_good: int = 0

    @property
    def gap(self) -> int:
        """Human labels still missing to reach `target`; never negative."""
        return max(0, self.target - self.human)

    @property
    def covered(self) -> bool:
        """True when the role has its target of human labels."""
        return self.human >= self.target


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
class Agreement:
    """How one labeller agrees with the human on one role.

    Attributes:
        n: Replies both labelled.
        agree: The share of those on which they agree; None when `n` is 0.
        kappa: Cohen's kappa; None where it is undefined (see `calibration.cohen_kappa`).
        ac1: Gwet's AC1; None under two replies.
    """

    n: int
    agree: float | None
    kappa: float | None
    ac1: float | None


def latest(labels: Iterable[Label]) -> dict[tuple[str, str, str], Label]:
    """Return one label per `(role, subject, labeller)`: the latest given (by `at`).

    A label with no `at` sorts before every dated one; among equals the later in `labels` wins.
    """
    out: dict[tuple[str, str, str], Label] = {}
    for lab in labels:
        k = (lab.role, lab.subject, lab.labeller)
        prev = out.get(k)
        if prev is None or (lab.at or _EPOCH) >= (prev.at or _EPOCH):
            out[k] = lab
    return out


def coverage(labels: Iterable[Label], roles: Iterable[str], *, target: int | Mapping[str, int]) -> dict[str, Coverage]:
    """Return each role's `Coverage`.

    Args:
        labels: Labels from any labeller.
        roles: The roles to report; a role with no label reports zeros. A labelled role missing
            from `roles` is reported too, so a label is never silently dropped.
        target: The human labels each role should have, one number for all or a role mapping
            (a role missing from the mapping gets 0).

    Returns:
        `{role: Coverage}`, in the order of `roles`, then any extra labelled roles sorted.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"human": 0, "model": 0, "judge": 0, "good": 0})
    for (role, _subject, who), lab in latest(labels).items():
        counts[role][who] += 1
        if who == "human" and lab.good:
            counts[role]["good"] += 1
    order = list(dict.fromkeys(roles))
    order += sorted(set(counts) - set(order))

    def _target(role: str) -> int:
        return target.get(role, 0) if isinstance(target, Mapping) else target

    return {
        r: Coverage(
            r, counts[r]["human"], counts[r]["model"], counts[r]["judge"], _target(r), counts[r]["good"]
        )
        for r in order
    }


def _stratified(pool: Sequence[Candidate], n: int) -> list[Candidate]:
    """Take `n` from `pool` round-robin over the judge's verdict, rarest stratum first."""
    strata: dict[str | None, list[Candidate]] = defaultdict(list)
    for c in sorted(pool, key=lambda c: sha(f"{c.role}\n{c.subject}")):
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
    A role absent from `coverage_now` is not drawn. Within a role the draw is deterministic (a
    hash of role and subject) and round-robin across the judge's verdicts, rarest first, so a
    queue holds both classes whenever the candidates do.

    Args:
        candidates: Unlabelled production replies.
        coverage_now: The current `coverage`, which carries each role's gap.
        per_run: The most replies one run queues per role.
        seen: Subjects to skip.
    """
    skip = set(seen)
    by_role: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        if c.subject not in skip:
            by_role[c.role].append(c)
            skip.add(c.subject)
    out: list[Candidate] = []
    for role, cov in coverage_now.items():
        n = min(per_run, cov.gap)
        if n > 0 and by_role.get(role):
            out += _stratified(by_role[role], n)
    return out


def agreement(labels: Iterable[Label]) -> dict[str, dict[str, Agreement]]:
    """Return, per role, each measured labeller's `Agreement` with the human.

    Only replies both the human and that labeller labelled count. A role with no human label is
    absent; a labeller with no shared reply reports `n == 0`.

    Returns:
        `{role: {"judge": Agreement, "model": Agreement}}`.
    """
    by: dict[tuple[str, str], dict[str, Label]] = defaultdict(dict)
    for (role, subject, who), lab in latest(labels).items():
        by[(role, who)][subject] = lab
    out: dict[str, dict[str, Agreement]] = {}
    for (role, who), human in sorted(by.items()):
        if who != "human":
            continue
        out[role] = {}
        for m in MEASURED:
            theirs = by.get((role, m), {})
            pairs = [
                (YES if theirs[s].good else "no", YES if human[s].good else "no") for s in sorted(human) if s in theirs
            ]
            agree = sum(a == b for a, b in pairs) / len(pairs) if pairs else None
            out[role][m] = Agreement(len(pairs), agree, cohen_kappa(pairs), gwet_ac1(pairs))
    return out


def label_of(
    verdict: JudgeVerdict,
    *,
    role: str,
    subject: str,
    trace: str = "",
    by: str = "",
    labeller: Labeller = "model",
    at: datetime | None = None,
) -> Label:
    """Return a `Judge`'s verdict on one reply as a `Label`.

    Args:
        verdict: Any `Judge`'s verdict.
        role: The labelled reply's role.
        subject: The reply's id in the label store.
        trace: The reply's trace id.
        by: The judging model's name.
        labeller: `model` for an offline labeller, `judge` for the production judge. Never `human`.
        at: When it was given.

    Raises:
        ValueError: `labeller` is `human`: a model's verdict is never ground truth.
    """
    if labeller == "human":
        raise ValueError("a model's verdict is never a human label")
    return Label(
        role=role,
        subject=subject,
        value="good" if verdict.passed else "bad",
        labeller=labeller,
        trace=trace,
        by=by or verdict.judge,
        reason=verdict.reason,
        at=at,
    )


__all__ = [
    "MEASURED",
    "Agreement",
    "Candidate",
    "Coverage",
    "agreement",
    "coverage",
    "draw",
    "label_of",
    "latest",
]
