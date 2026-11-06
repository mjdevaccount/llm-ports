"""Calibrate a judge in both directions: agreement with human labels, and replies it must fail.

A judge earns trust two ways, and each measures something the other cannot:

- AGREEMENT with a human labeller on the same items. `cohen_kappa` is the chance-corrected rate
  to read; `gwet_ac1` is its prevalence-robust companion, for an item whose labels are nearly all
  one answer (where kappa's chance correction degenerates). Raw agreement flatters a lopsided item.
- SEEDED-BAD replies: corrupted copies of real replies that any honest judge must fail.
  `seed_placeholder` blanks every string (the schema-answered-honestly law), `seed_figure` appends
  a figure no input carried (the number law). A judge that passes these passes everything.

`tally` and `verdict` read the rows of a replay that judged good samples beside their seeded-bad
copies. A row is a mapping with:

- `role`: the judged role's name.
- `kind`: `"good"` for a real reply, else the seed's name (`"tbd"`, `"figure"`, ...).
- `after`: the replayed verdict (`pass`, `fail` or `could_not_run`, as `judge.verdict_label`).
- `before`: the verdict previously stored for a good sample; None on a seeded row.
- `held_out`: whether the good sample was outside the rows read while building the judge.

Invariants:
    - Pure: no model, no database, no clock.
    - A statistic that the data cannot define is None, never a number.
    - A `could_not_run` row is never a catch: a judge that did not answer measured nothing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

#: The positive label of a binary item.
YES = "yes"
#: What `seed_placeholder` writes into every string.
PLACEHOLDER = "TBD"
#: What `seed_figure` appends: two figures that appear in no input.
FIGURE = " Target 7,431.5 by the 19th, up 38.2%."
#: `verdict` exit codes: calibrated in both directions, a finding, did not finish looking.
CALIBRATED, FINDING, BLIND = 0, 1, 2

Pairs = Sequence[tuple[str, str]]


def _decided(pairs: Pairs) -> tuple[int, float] | None:
    """Return `(n, observed agreement)`, or None below two pairs."""
    n = len(pairs)
    if n < 2:
        return None
    return n, sum(1 for a, b in pairs if a == b) / n


def cohen_kappa(pairs: Pairs) -> float | None:
    """Return Cohen's kappa over binary `(judge, human)` pairs.

    Both labels are `"yes"` or anything else (read as no); abstentions belong outside `pairs`.

    Returns:
        Kappa, or None when it is undefined: under two pairs, or both labellers one-sided on the
        same answer (chance agreement 1.0).
    """
    got = _decided(pairs)
    if got is None:
        return None
    n, po = got
    pa = sum(1 for a, _ in pairs if a == YES) / n
    pb = sum(1 for _, b in pairs if b == YES) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def gwet_ac1(pairs: Pairs) -> float | None:
    """Return Gwet's AC1 over binary `(judge, human)` pairs.

    Chance agreement is `2 * pi * (1 - pi)`, with `pi` the pooled share of `"yes"`, so the statistic
    stays defined and bounded where one answer dominates.

    Returns:
        AC1, or None under two pairs.
    """
    got = _decided(pairs)
    if got is None:
        return None
    n, po = got
    pi = sum((a == YES) + (b == YES) for a, b in pairs) / (2 * n)
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe)


def _strings(obj: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], str]]:
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, (*path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, (*path, i))


def _put(obj: Any, path: tuple[Any, ...], value: str) -> None:
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value


def _json(reply: str) -> Any:
    try:
        got = json.loads(reply)
    except ValueError:
        return None
    return got if isinstance(got, dict | list) else None


def seed_placeholder(reply: str, placeholder: str = PLACEHOLDER) -> str | None:
    """Return a copy of the reply with every string replaced by `placeholder`.

    A JSON object or list keeps its shape; any other reply becomes `placeholder` itself.

    Returns:
        The seeded reply, or None for a JSON reply with no string to replace.
    """
    obj = _json(reply)
    if obj is None:
        return placeholder
    paths = [p for p, _ in _strings(obj) if p]
    if not paths:
        return None
    for p in paths:
        _put(obj, p, placeholder)
    return json.dumps(obj)


def seed_figure(reply: str, figure: str = FIGURE) -> str | None:
    """Return a copy of the reply with `figure` appended to its longest free-text string.

    A free-text string holds a space; a non-JSON reply has `figure` appended to the whole text.

    Returns:
        The seeded reply, or None for a JSON reply with no free text to carry the figure.
    """
    obj = _json(reply)
    if obj is None:
        return reply + figure
    free = [(p, s) for p, s in _strings(obj) if p and " " in s]
    if not free:
        return None
    path, s = max(free, key=lambda x: len(x[1]))
    _put(obj, path, s + figure)
    return json.dumps(obj)


def tally(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Count a replay's rows per role, sorted by role.

    Per role: `n` good samples, `before_pass` and `after_pass` among them, `after_could_not_run`,
    `held_out`, and per seed kind `<kind>_n` and `<kind>_fail` (a `could_not_run` is not a fail).
    """
    t: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        d = t[r["role"]]
        if r["kind"] == "good":
            d["n"] += 1
            d["before_pass"] += r["before"] == "pass"
            d["after_pass"] += r["after"] == "pass"
            d["after_could_not_run"] += r["after"] == "could_not_run"
            d["held_out"] = r["held_out"]
        else:
            d[f"{r['kind']}_n"] += 1
            d[f"{r['kind']}_fail"] += r["after"] == "fail"
    return {k: dict(v) for k, v in sorted(t.items())}


def _measured(rows: Sequence[Mapping[str, Any]], *, good: bool) -> bool:
    return any((r["kind"] == "good") == good and r["after"] != "could_not_run" for r in rows)


def verdict(results: Sequence[Mapping[str, Any]], *, floor: float = 0.0) -> tuple[int, str]:
    """Return the replay's exit code and the reason.

    - `BLIND` (2), did not finish looking: nothing was replayed, or some role has no answered good
      sample or no answered seeded-bad one. Such a replay proves nothing in that direction.
    - `FINDING` (1): a seeded-bad reply passed, or a role's pass rate over its answered good samples
      is under `floor`.
    - `CALIBRATED` (0): measured and clean in both directions.
    """
    if not results:
        return BLIND, "nothing was replayed"
    blind = sum(r["after"] == "could_not_run" for r in results)
    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in results:
        by_role[r["role"]].append(r)
    dark = sorted(
        f"{k} ({'good' if not _measured(rs, good=True) else 'seeded-bad'})"
        for k, rs in by_role.items()
        if not (_measured(rs, good=True) and _measured(rs, good=False))
    )
    if dark:
        return BLIND, f"BLIND: {blind}/{len(results)} could_not_run; unmeasured direction: {', '.join(dark)}"
    missed = sorted({r["role"] for r in results if r["kind"] != "good" and r["after"] == "pass"})
    low = []
    for k, rs in sorted(by_role.items()):
        good = [r for r in rs if r["kind"] == "good" and r["after"] != "could_not_run"]
        rate = sum(r["after"] == "pass" for r in good) / len(good)
        if rate < floor:
            low.append(f"{k} {rate:.2f}")
    if missed or low:
        return FINDING, (
            f"seeded-bad PASSED in {', '.join(missed) or '-'}; under floor {floor}: "
            f"{', '.join(low) or '-'}; {blind}/{len(results)} could_not_run"
        )
    return CALIBRATED, f"calibrated both directions; {blind}/{len(results)} could_not_run"


__all__ = [
    "BLIND",
    "CALIBRATED",
    "FIGURE",
    "FINDING",
    "PLACEHOLDER",
    "YES",
    "Pairs",
    "cohen_kappa",
    "gwet_ac1",
    "seed_figure",
    "seed_placeholder",
    "tally",
    "verdict",
]
