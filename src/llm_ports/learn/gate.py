"""One gate for every loop: evaluated clauses in, a verdict out.

A gate is a function from evidence to clauses; `decide` turns the clauses into a `Verdict` the
same way for every loop, so a verdict means the same thing on every page that shows one:

- `pass`: at least one clause, and every clause passed.
- `insufficient`: a floor clause (a sample size) did not pass: not enough evidence yet.
- `fail`: the floors held and another clause did not pass: measured, and it did not clear.
- `could_not_run`: nothing was measured (`could_not_run`), or a gate with no clauses.

The clause helpers (`at_least`, `at_most`, `equals`, `check`) build a `Clause` from a measured
value. A value that is None, NaN or infinite gives `passed=None`, and `decide` fails a None
closed: missing evidence is never a pass.

Invariants:
    - Pure: no clock, no I/O.
    - `decide` never passes a verdict with an unevaluated clause, and never passes zero clauses.
    - The blocker is the FIRST clause, in the order given, that kept the verdict from passing.

Rationale: ADR-0010.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from llm_ports.learn._types import Clause, Value, Verdict


def _number(value: Value) -> float | None:
    """Return `value` as a finite float, or None (None, NaN, infinite, or not a number)."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def at_least(name: str, value: Value, threshold: float, *, floor: bool = False) -> Clause:
    """Return a clause that passes when `value >= threshold`.

    Args:
        name: The clause's name.
        value: The measured value; None, NaN or infinite leaves the clause unevaluated.
        threshold: The bar.
        floor: True for a sample-size clause (failing it is `insufficient`, not `fail`).

    Returns:
        The evaluated clause. Never raises.
    """
    number = _number(value)
    passed = None if number is None else number >= threshold
    return Clause(name, passed, number, threshold, ">=", floor)


def at_most(name: str, value: Value, threshold: float) -> Clause:
    """Return a clause that passes when `value <= threshold`.

    Args:
        name: The clause's name.
        value: The measured value; None, NaN or infinite leaves the clause unevaluated.
        threshold: The bar.

    Returns:
        The evaluated clause. Never raises.
    """
    number = _number(value)
    passed = None if number is None else number <= threshold
    return Clause(name, passed, number, threshold, "<=")


def equals(name: str, value: Value, expected: Value) -> Clause:
    """Return a clause that passes when `value == expected`.

    Args:
        name: The clause's name.
        value: The measured value; None leaves the clause unevaluated.
        expected: What it must equal.

    Returns:
        The evaluated clause. Never raises.
    """
    passed = None if value is None else value == expected
    return Clause(name, passed, value, expected, "==")


def check(name: str, passed: bool | None, value: Value = None) -> Clause:
    """Return a clause from a condition the caller already evaluated.

    Args:
        name: The clause's name.
        passed: True, False, or None when it could not be evaluated.
        value: What was measured, for the record; optional.

    Returns:
        The clause. Never raises.
    """
    return Clause(name, passed, value)


def decide(gate: str, clauses: Iterable[Clause], *, subject: str = "") -> Verdict:
    """Return the verdict of one gate over its evaluated clauses.

    Args:
        gate: The gate's name.
        clauses: Every clause, in the order to report them; the first failing one is the blocker.
        subject: What was judged; "" when the gate has one subject.

    Returns:
        `insufficient` when a floor clause did not pass, else `fail` when any clause did not pass,
        else `pass`; `could_not_run` for no clauses. Never raises.

    Invariants:
        - An unevaluated clause (`passed is None`) blocks exactly like a failed one.
    """
    evaluated = tuple(clauses)
    if not evaluated:
        return Verdict(gate, subject, "could_not_run", reason="the gate has no clauses")
    floors = [c for c in evaluated if c.floor and c.passed is not True]
    if floors:
        first = floors[0]
        return Verdict(gate, subject, "insufficient", evaluated, first.name, _why(first))
    blocking = [c for c in evaluated if c.passed is not True]
    if blocking:
        first = blocking[0]
        return Verdict(gate, subject, "fail", evaluated, first.name, _why(first))
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


def _why(clause: Clause) -> str:
    """Return one line saying why `clause` blocked."""
    if clause.passed is None:
        return f"{clause.name}: not evaluated (no value)"
    if clause.op:
        return f"{clause.name}: {clause.value} not {clause.op} {clause.threshold}"
    return f"{clause.name}: failed"


__all__ = ["at_least", "at_most", "check", "could_not_run", "decide", "equals"]
