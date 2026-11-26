"""Derivation re-check: a figure computed from facts is checked by code, never by a judge.

A `derived` claim names its operation as an op chain, `expr = numeric = result`: the expression
over fact ids, the same expression with each id replaced by its value, and the value it gives
(`abs(a - b) / a * 100 = abs(120.0 - 114.0) / 120.0 * 100 = 5.0`). `recheck` evaluates the
chain in a closed arithmetic grammar, holds the three parts to each other, and compares the figure
the reply shows with the value code computed. A comparison chain (`a < b < c = True`) holds when
it is true. Text after a colon is a comment (`a > b: why it matters`).

Three readings license a shown figure: the value itself, the value rounded to the decimals the
figure is written with or at its percent or scale marker (the number law's `rerenders`), and the
value within the claim's stated tolerance. A figure outside the tolerance but within `band` times
it is `abstain`: close enough that a wrong tolerance and a wrong figure look alike, so neither code
nor a judge should call it. Anything code cannot evaluate is `unchecked`, never a guess.

Invariants:
    - Pure: no I/O, no `eval`. Never raises on data.
    - The grammar is closed: numbers, fact ids, `+ - * /`, unary `-` and `+`, `abs`, `sqrt`, and a
      chain of `<`, `<=`, `>`, `>=`. Any other node is `unchecked`.

Rationale: ADR-0030.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from llm_ports.grounding.numbers import _PERCENT, _SCALE, NUM, _decimal, _exp, canon, rerenders

Verdict = Literal["holds", "breaks", "abstain", "unchecked"]
Value = float | bool

#: The four verdicts, in the order reports list them.
VERDICTS: tuple[Verdict, ...] = ("holds", "breaks", "abstain", "unchecked")
#: Where the op chain's parts are split: `expr = numeric = result`.
SEPARATOR = " = "
#: Where a chain's trailing comment starts (`a > b: why it matters`); no arithmetic holds a colon.
COMMENT = ":"

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_COMPARE: dict[type[ast.cmpop], Callable[[float, float], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_CALLS: dict[str, Callable[[float], float]] = {"abs": abs, "sqrt": math.sqrt}
#: The typographic minus sign a reply may print before a figure.
_MINUS = "\u2212"
#: How close two evaluations of one chain must be to be one value: float noise, not rounding.
_SAME = 1e-9


class _Outside(Exception):
    """A node outside the grammar, an unknown name, or arithmetic with no value (private)."""


@dataclass(frozen=True)
class Tolerance:
    """How far a shown figure may sit from the computed value and still state it.

    Attributes:
        absolute: The largest allowed `|shown - value|`, or None.
        relative: The largest allowed `|shown - value| / |value|`, or None.
    """

    absolute: float | None = None
    relative: float | None = None

    @classmethod
    def of(cls, raw: Mapping[str, object] | None) -> Tolerance | None:
        """Read `{"abs": x}` or `{"rel": x}`; None for no mapping or no number under either key."""
        if not raw:
            return None
        found = {k: v for k, v in raw.items() if k in ("abs", "rel") and isinstance(v, int | float)}
        if not found:
            return None
        a, r = found.get("abs"), found.get("rel")
        return cls(float(a) if a is not None else None, float(r) if r is not None else None)

    def allows(self, shown: float, value: float, *, widen: float = 1.0) -> bool:
        """Return whether `shown` is within this tolerance of `value`, each bound times `widen`."""
        gap = abs(shown - value)
        if self.absolute is not None and gap <= self.absolute * widen + _SAME:
            return True
        return self.relative is not None and gap <= self.relative * widen * abs(value) + _SAME


@dataclass(frozen=True)
class Recheck:
    """What code found when it re-ran one claim's operation.

    Attributes:
        verdict: `holds`, `breaks`, `abstain` or `unchecked` (`VERDICTS`).
        value: What the chain computes, when code could evaluate it.
        reason: One line naming what held or broke.
    """

    verdict: Verdict
    value: Value | None
    reason: str


def _node(node: ast.AST, names: Mapping[str, float]) -> Value:
    if isinstance(node, ast.Expression):
        return _node(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise _Outside(f"unknown name {node.id!r}")
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_number(node.left, names), _number(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_number(node.operand, names))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALLS:
        if len(node.args) != 1 or node.keywords:
            raise _Outside(f"{node.func.id} takes one argument")
        return _CALLS[node.func.id](_number(node.args[0], names))
    if isinstance(node, ast.Compare) and all(type(o) in _COMPARE for o in node.ops):
        left, ok = _number(node.left, names), True
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = _number(right_node, names)
            ok = ok and _COMPARE[type(op)](left, right)
            left = right
        return ok
    raise _Outside(f"{type(node).__name__} is outside the grammar")


def _number(node: ast.AST, names: Mapping[str, float]) -> float:
    value = _node(node, names)
    if isinstance(value, bool):
        raise _Outside("a comparison inside arithmetic")
    return value


def evaluate(expr: str, names: Mapping[str, float] | None = None) -> Value | None:
    """Evaluate `expr` in the closed grammar; None when it falls outside it.

    Args:
        expr: An arithmetic expression or a comparison chain.
        names: Values for the names `expr` uses; a name not here makes the result None.

    Returns:
        A float, a bool for a comparison chain, or None for a parse error, a node outside the
        grammar, an unknown name, a division by zero, a square root of a negative, or a value that
        is not finite. Never raises.
    """
    try:
        value = _node(ast.parse(expr.strip(), mode="eval"), names or {})
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError, _Outside):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _readable(stated: str) -> bool:
    return stated in ("True", "False") or _decimal(stated.lstrip("+")) is not None


def _agrees(stated: str, value: Value) -> bool:
    """Return whether the chain's written result is `value`, to the decimals it is written with."""
    if isinstance(value, bool):
        return stated in ("True", "False") and (stated == "True") == value
    d = _decimal(stated.lstrip("+"))
    if d is None:
        return False
    exponent = d.as_tuple().exponent
    half = Decimal(5).scaleb(exponent - 1) if isinstance(exponent, int) else Decimal(0)
    return abs(float(d) - value) <= float(half) + _SAME * max(1.0, abs(value))


def _same(a: Value, b: Value) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return abs(a - b) <= _SAME * max(1.0, abs(a), abs(b))


def _numeric(facts: Mapping[str, object]) -> dict[str, float]:
    return {k: float(v) for k, v in facts.items() if isinstance(v, int | float) and not isinstance(v, bool)}


def _shown(shown: str) -> tuple[str, str, list[float]] | None:
    """Return the shown figure's run, the text after it, and the values it may state."""
    m = NUM.search(shown)
    d = _decimal(m.group()) if m else None
    if m is None or d is None:
        return None
    after = shown[m.end() :]
    sign = -1.0 if shown[: m.start()].rstrip("$€£ ").endswith(("-", _MINUS)) else 1.0
    base = sign * float(d)
    if _PERCENT.match(after):
        return m.group(), after, [base, base / 100]
    scale = _SCALE.match(after)
    if scale:
        return m.group(), after, [base * 10.0 ** _exp(scale)]
    return m.group(), after, [base]


def _nearest(candidates: list[float], value: float) -> float:
    """Return the reading of a shown figure nearest `value` in ratio, the one a tolerance judges.

    A percent figure reads as the value itself or as a fraction times 100. A tolerance is stated
    for one reading, so it judges only the reading the figure plainly means ("0.30%" beside 0.097
    is 0.30, not 0.003): the reading whose ratio to the value is nearest 1.
    """
    if len(candidates) == 1 or value == 0:
        return min(candidates, key=lambda c: abs(c - value))
    return min(candidates, key=lambda c: abs(math.log(abs(c / value))) if c * value > 0 else math.inf)


def recheck(
    op: str,
    shown: str | None = None,
    *,
    tol: Tolerance | None = None,
    facts: Mapping[str, object] | None = None,
    band: float = 2.0,
) -> Recheck:
    """Re-run one claim's op chain and judge the figure the reply shows against it.

    Args:
        op: `expr = numeric = result`, `expr = result` when the expression has no names, or a
            bare comparison chain (`750.5 < 751`). Text after a colon is a comment and is dropped.
        shown: The figure as the reply writes it (`"$1.2M"`, `"5.0%"`), or None when the claim
            shows none (a comparison, or a figure the claim only relies on).
        tol: How far the shown figure may sit from the value; None licenses only the value and
            its roundings.
        facts: The Input's facts by id. When given, `expr` is evaluated over them too, so a chain
            that substituted the wrong value breaks.
        band: The abstain band, as a multiple of `tol`: outside `tol` but inside `band * tol` is
            `abstain`. 1.0 turns the band off.

    Returns:
        The `Recheck`: `breaks` when the chain's parts disagree, a comparison is false, or the
        figure is outside the band; `holds` when the figure is the value, a rounding of it, or
        within `tol`; `abstain` inside the band; `unchecked` when code cannot evaluate the chain
        or find a figure in `shown`. Never raises.
    """
    parts = [p.strip() for p in op.split(COMMENT, 1)[0].split(SEPARATOR)]
    if not all(parts):
        return Recheck("unchecked", None, f"not an op chain: {op!r}")
    expr, stated, numeric = parts[0], parts[-1], parts[1] if len(parts) > 2 else parts[0]
    value = evaluate(numeric)
    if value is None:
        return Recheck("unchecked", None, f"outside the grammar: {numeric!r}")
    if len(parts) == 1:  # a bare comparison states its own result; a bare sum states none
        if not isinstance(value, bool):
            return Recheck("unchecked", value, f"no result stated: {op!r}")
        stated = str(value)
    if not _readable(stated):
        return Recheck("unchecked", value, f"the chain's result {stated!r} is not a number")
    if facts is not None:
        over_facts = evaluate(expr, _numeric(facts))
        if over_facts is None:
            return Recheck("unchecked", value, f"{expr!r} does not evaluate over the facts")
        if not _same(over_facts, value):
            return Recheck("breaks", value, f"{expr!r} gives {over_facts} over the facts, not {value}")
    if not _agrees(stated, value):
        return Recheck("breaks", value, f"the chain states {stated}; it computes {value}")
    if isinstance(value, bool):
        return Recheck("holds" if value else "breaks", value, f"{expr} is {value}")
    if shown is None:
        return Recheck("holds", value, "the chain computes; the claim shows no figure")
    read = _shown(shown)
    if read is None:
        return Recheck("unchecked", value, f"no figure in {shown!r}")
    run, after, candidates = read
    if any(_same(c, value) for c in candidates) or (
        all(c * value >= 0 for c in candidates) and rerenders(run, after, {canon(repr(abs(value)))})
    ):
        return Recheck("holds", value, f"{shown} shows {value}")
    near = _nearest(candidates, value)
    if tol is not None and tol.allows(near, value):
        return Recheck("holds", value, f"{shown} is within tolerance of {value}")
    if tol is not None and band > 1.0 and tol.allows(near, value, widen=band):
        return Recheck("abstain", value, f"{shown} is outside tolerance of {value}, inside the band")
    return Recheck("breaks", value, f"{shown} does not state {value}")


__all__ = ["COMMENT", "SEPARATOR", "VERDICTS", "Recheck", "Tolerance", "Value", "Verdict", "evaluate", "recheck"]
