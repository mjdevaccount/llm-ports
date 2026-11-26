"""Claim-level grounding: the label scheme, the case rule, the fact table, and how a judge is measured.

A reply is graded one claim at a time against the facts its Input carried. Each claim gets one of
four labels (`LABELS`):

- `supported`: a fact states it.
- `derived`: it follows from cited facts by a stated operation (arithmetic, a date step, a rule).
- `not_in_input`: no fact states it or licenses it.
- `contradicted`: a fact states otherwise.

A case fails when a MATERIAL claim is `not_in_input` or `contradicted` (`case_label`); an
immaterial slip is counted, not failed. A case may also be `unjudgeable` (the Input the reply was
written from is not on file), which no rule here turns into a pass or a fail.

The module also holds the measurement side: `read_cases` loads gold cases from JSON lines (any
family that carries the core keys; extra keys ride along), `agreement` scores a judge's labels
against the gold per label class with a 95% interval, `split` assigns a case to train / dev / test
by a stable hash of its id, `check` names what a gold row must not hold, and `flips` counts
verdicts a reworded prompt changed.

Invariants:
    - Pure: no model, no clock, no network. `read_cases` reads only the text it is given.
    - Never raises on data: a malformed row is a named problem in `Loaded.problems`, not an error.
    - The label vocabulary is closed: an unknown label is a problem, never a fifth class.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_ports.learn.examples import Agreement

ClaimLabel = Literal["supported", "derived", "not_in_input", "contradicted"]
CaseLabel = Literal["pass", "fail", "unjudgeable"]
Split = Literal["train", "dev", "test"]

#: The four claim labels, in the order reports list them.
LABELS: tuple[ClaimLabel, ...] = ("supported", "derived", "not_in_input", "contradicted")
#: The labels under which a claim is grounded.
GROUNDED: frozenset[str] = frozenset({"supported", "derived"})
#: The case labels.
CASE_LABELS: tuple[CaseLabel, ...] = ("pass", "fail", "unjudgeable")
#: The keys every case row must carry.
CASE_KEYS: tuple[str, ...] = ("case_id", "input", "output", "claims", "case_label")
#: The keys every claim must carry.
CLAIM_KEYS: tuple[str, ...] = ("id", "quote", "text", "label")


@dataclass(frozen=True)
class Fact:
    """One fact of a case's Input.

    Attributes:
        id: The id a claim cites it by.
        text: The fact as the judge reads it.
    """

    id: str
    text: str


@dataclass(frozen=True)
class Claim:
    """One claim of a reply, with its gold label.

    Attributes:
        id: The claim's id within its case.
        quote: The reply fragment the claim was read from.
        text: The claim, stated plainly.
        label: Its gold label (`LABELS`).
        facts: The fact ids it rests on.
        material: Whether a wrong label on it changes what a reader does. Absent reads as True.
        rule: The consumer's inference-rule id that licenses or refuses it, if any.
        op: The operation a `derived` claim performs on its facts, as text code can re-check.
        extra: Every other key of the claim row, unread here.
    """

    id: str
    quote: str
    text: str
    label: ClaimLabel
    facts: tuple[str, ...] = ()
    material: bool = True
    rule: str | None = None
    op: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        """True when the label is in `GROUNDED`."""
        return self.label in GROUNDED


@dataclass(frozen=True)
class Case:
    """One judged reply with its Input, its facts and its gold claims.

    Attributes:
        case_id: Unique within a fixture.
        input: Everything the model was given, as text.
        output: The reply.
        claims: The gold claims.
        case_label: The gold case label (`CASE_LABELS`).
        facts: The Input's facts, when the row lists them.
        extra: Every other key of the row, unread here.
    """

    case_id: str
    input: str
    output: str
    claims: tuple[Claim, ...]
    case_label: CaseLabel
    facts: tuple[Fact, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Loaded:
    """What `read_cases` read.

    Attributes:
        cases: The rows that parsed, in file order.
        problems: One line per row that did not, naming the line and the defect.
    """

    cases: tuple[Case, ...]
    problems: tuple[str, ...]


def case_label(claims: Iterable[Claim]) -> Literal["pass", "fail"]:
    """Return `fail` when any material claim is ungrounded, else `pass`.

    Args:
        claims: The case's claims, gold or judged.

    Returns:
        The case label the claims imply. Never `unjudgeable`: that is a property of the Input.
    """
    return "fail" if any(c.material and not c.grounded for c in claims) else "pass"


def fact_table(facts: Iterable[Fact]) -> str:
    """Return the facts as numbered lines, `<id> <text>`, one per fact, for a judge's Input.

    A judge that must name the fact ids it checked can only do so against ids it was shown.
    """
    return "\n".join(f"{f.id} {f.text}" for f in facts)


def _facts(raw: Any) -> tuple[Fact, ...]:
    if isinstance(raw, Mapping):  # {key: value}: the key is the id
        return tuple(Fact(str(k), f"{k} = {v}") for k, v in raw.items())
    if isinstance(raw, list):  # [{id, span|text, ...}]
        return tuple(Fact(str(f["id"]), str(f.get("span") or f.get("text") or "")) for f in raw)
    return ()


def _claim(raw: Mapping[str, Any], aliases: Mapping[str, str]) -> Claim:
    missing = [k for k in CLAIM_KEYS if k not in raw]
    if missing:
        raise ValueError(f"claim {raw.get('id')!r} lacks {missing}")
    if raw["label"] not in LABELS:
        raise ValueError(f"claim {raw['id']!r} label {raw['label']!r} not in {LABELS}")
    rule = raw.get("rule")
    known = {*CLAIM_KEYS, "facts", "material", "rule", "op"}
    return Claim(
        id=str(raw["id"]),
        quote=str(raw["quote"]),
        text=str(raw["text"]),
        label=raw["label"],
        facts=tuple(str(f) for f in raw.get("facts") or ()),
        material=raw.get("material") is not False,
        rule=aliases.get(rule, rule) if rule else None,
        op=raw.get("op"),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def _case(row: Mapping[str, Any], aliases: Mapping[str, str]) -> Case:
    missing = [k for k in CASE_KEYS if k not in row]
    if missing:
        raise ValueError(f"lacks {missing}")
    if row["case_label"] not in CASE_LABELS:
        raise ValueError(f"case_label {row['case_label']!r} not in {CASE_LABELS}")
    claims = tuple(_claim(c, aliases) for c in row["claims"])
    ids = [c.id for c in claims]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate claim ids")
    known = {*CASE_KEYS, "facts"}
    return Case(
        case_id=str(row["case_id"]),
        input=str(row["input"]),
        output=str(row["output"]),
        claims=claims,
        case_label=row["case_label"],
        facts=_facts(row.get("facts")),
        extra={k: v for k, v in row.items() if k not in known},
    )


def read_cases(lines: Iterable[str], *, rule_aliases: Mapping[str, str] | None = None) -> Loaded:
    """Parse gold cases from JSON lines; blank lines are skipped.

    Any fixture family that carries `CASE_KEYS` per row and `CLAIM_KEYS` per claim loads; every
    other key is kept in `extra`. Facts are read from a list of `{id, span|text}` or a mapping of
    `{id: value}`.

    Args:
        lines: The file's lines.
        rule_aliases: Old rule id -> current id, applied to every claim's `rule` (a fixture written
            under an older rule set loads without editing the data).

    Returns:
        The `Loaded` cases and problems. Never raises: a bad line, a missing key, an unknown label,
        a duplicate claim id or a duplicate case id is a problem naming its line.
    """
    aliases = rule_aliases or {}
    cases: list[Case] = []
    problems: list[str] = []
    seen: set[str] = set()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            case = _case(json.loads(line), aliases)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            problems.append(f"line {n}: {exc}")
            continue
        if case.case_id in seen:
            problems.append(f"line {n}: duplicate case_id {case.case_id!r}")
            continue
        seen.add(case.case_id)
        cases.append(case)
    return Loaded(tuple(cases), tuple(problems))


def check(case: Case, *, known: Iterable[str] = ()) -> list[str]:
    """Return what in `case` a gold file must not hold; empty when it is sound.

    A claim's `quote` must occur in the reply verbatim (else the claim was read from text the
    reply does not contain), every fact id a claim cites must resolve, and the gold `case_label`
    must be the one its claims imply (`unjudgeable` is exempt: it is a property of the Input).

    Args:
        case: A loaded case.
        known: Ids beyond `case.facts` a claim may cite (values the consumer derives from facts).

    Returns:
        One line per defect, naming the claim. Never raises.
    """
    ids = {f.id for f in case.facts} | set(known)
    out = [f"{c.id}: quote not in the output" for c in case.claims if c.quote not in case.output]
    if case.facts:  # a case that lists no facts cannot resolve a citation either way
        out += [f"{c.id}: cites unknown fact {r!r}" for c in case.claims for r in c.facts if r not in ids]
    if case.case_label != "unjudgeable" and case.case_label != case_label(case.claims):
        out.append(f"case_label {case.case_label!r} but its claims imply {case_label(case.claims)!r}")
    return out


def agreement(pairs: Sequence[tuple[str, str]]) -> dict[str, Agreement]:
    """Score `(judged, gold)` claim labels per label class, one class against the rest.

    Cohen's kappa is defined for two classes; four labels are scored as four binary questions
    ("is it `<label>`?"), each with its 95% interval, plus `grounded` (supported or derived
    against the other two), the question a case verdict turns on.

    Args:
        pairs: `(judged label, gold label)` per claim. A judged label outside `LABELS` (an
            abstention) is left out of every class.

    Returns:
        `{label: Agreement}` for each of `LABELS`, then `"grounded"`.
    """
    kept = [(a, b) for a, b in pairs if a in LABELS and b in LABELS]
    out: dict[str, Agreement] = {
        lab: Agreement.of([(_yes(a == lab), _yes(b == lab)) for a, b in kept]) for lab in LABELS
    }
    out["grounded"] = Agreement.of([(_yes(a in GROUNDED), _yes(b in GROUNDED)) for a, b in kept])
    return out


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def split(case_id: str, *, dev: float = 0.2, test: float = 0.2) -> Split:
    """Return the split a case belongs to, from a stable hash of its id.

    The same id lands in the same split on every machine and every run, so a tuning round that
    reads `dev` never sees `test`, and a case added later does not move the others.

    Args:
        case_id: The case's id.
        dev: The share of cases in `dev`.
        test: The share in `test`; the rest is `train`.

    Returns:
        `train`, `dev` or `test`.
    """
    u = int(hashlib.blake2b(case_id.encode(), digest_size=8).hexdigest(), 16) / 2**64
    return "test" if u < test else "dev" if u < test + dev else "train"


def flips(first: Mapping[str, str], second: Mapping[str, str]) -> tuple[int, int]:
    """Count the verdicts that changed between two runs over the same cases.

    The paraphrase check: run the judge, reword its prompt without changing its meaning, run it
    again. A judge whose verdicts move under a rewording is reading the wording, not the reply.

    Args:
        first: `{case_id: verdict}` of the first run.
        second: The same for the reworded run.

    Returns:
        `(flipped, compared)` over the case ids both runs answered.
    """
    both = first.keys() & second.keys()
    return sum(first[k] != second[k] for k in both), len(both)


__all__ = [
    "CASE_KEYS",
    "CASE_LABELS",
    "CLAIM_KEYS",
    "GROUNDED",
    "LABELS",
    "Case",
    "CaseLabel",
    "Claim",
    "ClaimLabel",
    "Fact",
    "Loaded",
    "Split",
    "agreement",
    "case_label",
    "check",
    "fact_table",
    "flips",
    "read_cases",
    "split",
]
