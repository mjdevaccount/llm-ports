"""The learning loop's values: a labelled example, a gate clause, a verdict, a ledger row, a run.

Invariants:
    - Standard library only: this package imports nothing else from `llm_ports` and no vendor.
    - Every value is frozen; `Verdict.as_dict` and `Verdict.from_dict` round-trip through JSON.

Rationale: ADR-0010.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

#: A gate's decision. `insufficient` = a sample floor not met; `could_not_run` = nothing measured.
Decision = Literal["pass", "fail", "insufficient", "could_not_run"]
DECISIONS: tuple[Decision, ...] = ("pass", "fail", "insufficient", "could_not_run")
#: Who produced a label: a person, a model labelling offline, a production judge, or the world
#: (an outcome resolved mechanically, such as a forward return).
Source = Literal["human", "model", "judge", "outcome"]
#: How a store call ended.
StoreOutcome = Literal["ok", "unavailable", "request_fault"]
#: A clause value: a number, a label, or None when it could not be computed.
Value = float | int | str | bool | None


def _finite(value: Value) -> Value:
    """Return `value`, or None for a NaN or infinite float (a non-number is not evidence)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class Clause:
    """One condition of a gate, evaluated.

    Attributes:
        name: The clause's name, unique within its gate (`n_at_least_200`, `kappa`, ...).
        passed: True or False when evaluated; None when it could not be (missing evidence).
        value: The measured value; None when it could not be computed.
        threshold: The bar it was held to; None for a boolean check.
        op: How `value` was compared to `threshold` (`>=`, `<=`, `==`); "" for a boolean check.
        floor: True for a sample-size clause: failing it means "not enough evidence yet"
            (`insufficient`), not "measured and failed".

    Invariants:
        - A None `passed` is never read as a pass (`gate.decide` fails it closed).
    """

    name: str
    passed: bool | None
    value: Value = None
    threshold: Value = None
    op: str = ""
    floor: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the clause as a JSON-able mapping."""
        return {
            "name": self.name,
            "passed": self.passed,
            "value": _finite(self.value),
            "threshold": _finite(self.threshold),
            "op": self.op,
            "floor": self.floor,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Clause:
        """Rebuild a clause from `as_dict` output.

        Raises:
            KeyError: `raw` has no `name`.
        """
        passed = raw.get("passed")
        return cls(
            name=str(raw["name"]),
            passed=passed if isinstance(passed, bool) else None,
            value=raw.get("value"),
            threshold=raw.get("threshold"),
            op=str(raw.get("op", "")),
            floor=bool(raw.get("floor", False)),
        )


@dataclass(frozen=True)
class Verdict:
    """What a gate decided about one subject, with every clause it read.

    Attributes:
        gate: The gate's name.
        subject: What was judged (a model, a symbol, a role); "" when the gate has one subject.
        decision: `pass`, `fail`, `insufficient` or `could_not_run`.
        clauses: Every clause, in the order the gate evaluated them.
        blocker: The first clause that kept the decision from `pass`; None on `pass` and on
            `could_not_run`.
        reason: One line for a person: why it passed, what blocked it, or why nothing ran.

    Invariants:
        - `decision == "pass"` exactly when there is at least one clause and every clause passed.
    """

    gate: str
    subject: str
    decision: Decision
    clauses: tuple[Clause, ...] = ()
    blocker: str | None = None
    reason: str = ""

    @property
    def passed(self) -> bool:
        """True when `decision == "pass"`."""
        return self.decision == "pass"

    def as_dict(self) -> dict[str, Any]:
        """Return the verdict as a JSON-able mapping (what a ledger row stores under `gate`)."""
        return {
            "gate": self.gate,
            "subject": self.subject,
            "decision": self.decision,
            "blocker": self.blocker,
            "reason": self.reason,
            "clauses": [c.as_dict() for c in self.clauses],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Verdict | None:
        """Rebuild a verdict from `as_dict` output; None when `raw` is not one.

        Never raises: a mapping with an unknown decision or a malformed clause is not a verdict.
        """
        decision = raw.get("decision")
        if decision not in DECISIONS:
            return None
        try:
            clauses = tuple(Clause.from_dict(c) for c in raw.get("clauses", ()))
        except (KeyError, TypeError, AttributeError):
            return None
        blocker = raw.get("blocker")
        return cls(
            gate=str(raw.get("gate", "")),
            subject=str(raw.get("subject", "")),
            decision=cast(Decision, decision),
            clauses=clauses,
            blocker=str(blocker) if blocker is not None else None,
            reason=str(raw.get("reason", "")),
        )


@dataclass(frozen=True)
class Example:
    """One labelled example: something a loop learns from, with where it came from.

    Attributes:
        topic: What is learned (`llm:narrative.research`, `trend:1d`, `picker:SPY`).
        subject: The thing labelled, unique within the topic (an observation id, `SPY@2026-11-06`).
        value: The label: a class (`good`, `bad`) or a number (a forward return).
        source: `human`, `model`, `judge` or `outcome`.
        by: Who labelled: a person's id, a model's name, a grader's name; "" when unknown.
        at: When the label was given; None when the store does not say.
        reason: The labeller's reason; "" when none was given.
        ref: Provenance: trace id, table and key, code version. Not part of equality.

    Invariants:
        - A `model` or `judge` label is measured against `human` labels, never counted as one.
    """

    topic: str
    subject: str
    value: str | float
    source: Source
    by: str = ""
    at: datetime | None = None
    reason: str = ""
    ref: Mapping[str, str] = field(default_factory=dict[str, str], compare=False, hash=False)


@dataclass(frozen=True)
class ExamplesRead:
    """What `ExampleStore.examples` returns.

    Attributes:
        examples: Every example read; `()` exactly when `outcome != "ok"`.
        outcome: `ok`, or `unavailable` when the store could not be read.
        reason: One line naming the failure; None on `ok`.

    Invariants:
        - A store that could not be read never reads as a store with no examples.
    """

    examples: tuple[Example, ...]
    outcome: StoreOutcome
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """True when `outcome == "ok"`."""
        return self.outcome == "ok"


@dataclass(frozen=True)
class Written:
    """What a store or ledger write returns.

    Attributes:
        written: Items the store accepted.
        outcome: `ok` when every item landed; `unavailable` when the store failed (`written`
            counts what landed first); `request_fault` when the store refused the input.
        reason: One line naming the failure; None on `ok`.
    """

    written: int
    outcome: StoreOutcome
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """True when `outcome == "ok"`."""
        return self.outcome == "ok"


@dataclass(frozen=True)
class LedgerRow:
    """One recorded run of a loop.

    Attributes:
        kind: The loop's run kind.
        subject: What the run was about; "" when the loop has one subject.
        params: The knobs the run used.
        metrics: What it measured; a gated run carries `verdict` (the decision) and `gate`
            (`Verdict.as_dict`).
        at: When it was recorded.
    """

    kind: str
    subject: str
    params: Mapping[str, Any]
    metrics: Mapping[str, Any]
    at: datetime

    @property
    def verdict(self) -> Verdict | None:
        """The recorded verdict, or None for an ungated (informational) run."""
        raw = self.metrics.get("gate")
        return Verdict.from_dict(raw) if isinstance(raw, Mapping) else None


@dataclass(frozen=True)
class RowRead:
    """What `RunLedger.latest` returns.

    Attributes:
        row: The newest matching row; None when there is none or the ledger could not be read.
        outcome: `ok`, or `unavailable` when the ledger could not be read.
        reason: One line naming the failure; None on `ok`.

    Invariants:
        - "No row yet" (`ok`, None) is distinct from "could not read" (`unavailable`).
    """

    row: LedgerRow | None
    outcome: StoreOutcome
    reason: str | None = None

    @property
    def ok(self) -> bool:
        """True when `outcome == "ok"`."""
        return self.outcome == "ok"


__all__ = [
    "DECISIONS",
    "Clause",
    "Decision",
    "Example",
    "ExamplesRead",
    "LedgerRow",
    "RowRead",
    "Source",
    "StoreOutcome",
    "Value",
    "Verdict",
    "Written",
]
