"""One gate for every loop: evaluated clauses in, a verdict out.

A gate is a function from evidence to clauses; `decide` turns the clauses into a `Verdict` the
same way for every loop, so a verdict means the same thing on every page that shows one. The
first rule that applies wins:

1. no clauses: `could_not_run`.
2. a `veto` clause evaluated False: `fail` (it disqualifies, whatever the sample).
3. a `floor` clause not True: `insufficient` (not enough evidence yet; a `bar` measured on too
   small a sample is noise, so it cannot `fail`).
4. with `unevaluated="fail"` (default), any clause not True: `fail`. With
   `unevaluated="insufficient"`, a clause evaluated False: `fail`, else a clause that is None:
   `insufficient`.
5. otherwise `pass`.

So a promote / hold / reject gate maps onto it exactly: disqualifiers are `veto`, unproven
conditions are `floor`, `unevaluated="insufficient"`; then `pass` is promote, `insufficient` is
hold and `fail` is reject, with reject winning over hold.

The clause helpers (`at_least`, `at_most`, `equals`, `check`) build a `Clause` from a measured
value. A value that is None, NaN or infinite gives `passed=None`, which never passes.

Invariants:
    - Pure: no clock, no I/O.
    - `decide` never passes a verdict with an unevaluated clause, and never passes zero clauses.
    - The blocker is the FIRST clause, in the order given, that the deciding rule names.

Rationale: ADR-0010.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

from llm_ports.learn._types import Clause, Decision, Role, Value, Verdict

#: How `decide` reads a clause that could not be evaluated (`passed is None`).
Unevaluated = Literal["fail", "insufficient"]


def _number(value: Value) -> float | None:
    """Return `value` as a finite float, or None (None, NaN, infinite, or not a number)."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def at_least(name: str, value: Value, threshold: float, *, role: Role = "bar") -> Clause:
    """Return a clause that passes when `value >= threshold`.

    Args:
        name: The clause's name.
        value: The measured value; None, NaN or infinite leaves the clause unevaluated.
        threshold: The bar.
        role: What failing it means (`bar`, `floor` for a sample size, `veto`).

    Returns:
        The evaluated clause. Never raises.
    """
    number = _number(value)
    passed = None if number is None else number >= threshold
    return Clause(name, passed, number, threshold, ">=", role)


def at_most(name: str, value: Value, threshold: float, *, role: Role = "bar") -> Clause:
    """Return a clause that passes when `value <= threshold`.

    Args:
        name: The clause's name.
        value: The measured value; None, NaN or infinite leaves the clause unevaluated.
        threshold: The bar.
        role: What failing it means (`bar`, `floor`, `veto`).

    Returns:
        The evaluated clause. Never raises.
    """
    number = _number(value)
    passed = None if number is None else number <= threshold
    return Clause(name, passed, number, threshold, "<=", role)


def equals(name: str, value: Value, expected: Value, *, role: Role = "bar") -> Clause:
    """Return a clause that passes when `value == expected`.

    Args:
        name: The clause's name.
        value: The measured value; None leaves the clause unevaluated.
        expected: What it must equal.
        role: What failing it means (`bar`, `floor`, `veto`).

    Returns:
        The evaluated clause. Never raises.
    """
    passed = None if value is None else value == expected
    return Clause(name, passed, value, expected, "==", role)


def check(name: str, passed: bool | None, value: Value = None, *, role: Role = "bar") -> Clause:
    """Return a clause from a condition the caller already evaluated.

    Args:
        name: The clause's name.
        passed: True, False, or None when it could not be evaluated.
        value: What was measured, for the record; optional.
        role: What failing it means (`bar`, `floor`, `veto`).

    Returns:
        The clause. Never raises.
    """
    return Clause(name, passed, value, role=role)


def decide(
    gate: str, clauses: Iterable[Clause], *, subject: str = "", unevaluated: Unevaluated = "fail"
) -> Verdict:
    """Return the verdict of one gate over its evaluated clauses (rules in the module docstring).

    Args:
        gate: The gate's name.
        clauses: Every clause, in the order to report them; the blocker is the first one the
            deciding rule names.
        subject: What was judged; "" when the gate has one subject.
        unevaluated: How a `bar` or `veto` clause that is None reads: `fail` (default, fail
            closed as a failure) or `insufficient` (fail closed as unproven). A `floor` that is
            None is always `insufficient`.

    Returns:
        The verdict. Never raises.

    Invariants:
        - An unevaluated clause never passes, under either policy.
    """
    evaluated = tuple(clauses)
    if not evaluated:
        return Verdict(gate, subject, "could_not_run", reason="the gate has no clauses")
    rules: list[tuple[Decision, list[Clause]]] = [
        ("fail", [c for c in evaluated if c.role == "veto" and c.passed is False]),
        ("insufficient", [c for c in evaluated if c.role == "floor" and c.passed is not True]),
    ]
    if unevaluated == "fail":
        rules.append(("fail", [c for c in evaluated if c.passed is not True]))
    else:
        rules.append(("fail", [c for c in evaluated if c.passed is False]))
        rules.append(("insufficient", [c for c in evaluated if c.passed is None]))
    for decision, named in rules:
        if named:
            first = named[0]
            return Verdict(gate, subject, decision, evaluated, first.name, _why(first))
    return Verdict(gate, subject, "pass", evaluated, None, f"all {len(evaluated)} clauses passed")


def could_not_run(gate: str, reason: str, *, subject: str = "") -> Verdict:
    """Return the verdict of a gate that measured nothing.

    Args:
        gate: The gate's name.
        reason: Why nothing was measured (the store was down, the data was empty, ...).
        subject: What would have been judged.

    Returns:
        A `could_not_run` verdict with no clauses. Never raises.
    """
    return Verdict(gate, subject, "could_not_run", reason=reason)


def _show(value: Value) -> str:
    """Return `value` for a person: numbers compactly (`-1`, `0.05`), anything else as is."""
    if isinstance(value, float | int) and not isinstance(value, bool):
        return f"{value:g}"
    return str(value)


def _why(clause: Clause) -> str:
    """Return one line saying why `clause` blocked."""
    if clause.passed is None:
        return f"{clause.name}: not evaluated (no value)"
    if clause.op:
        return f"{clause.name}: {_show(clause.value)} not {clause.op} {_show(clause.threshold)}"
    return f"{clause.name}: failed"


__all__ = ["Unevaluated", "at_least", "at_most", "check", "could_not_run", "decide", "equals"]
