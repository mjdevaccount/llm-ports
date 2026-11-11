"""Golden-set evaluation of a prompt pack: fire each item through the call path, grade on one board.

A golden item is a mapping with an `id` and the pack's template context under `slots` (omitted
for a pack with no slots). An item that also carries `user` fires as a system and user pair: the
rendered pack is the system prompt and `user` is the prompt, the shape of a producer whose fixed
instructions ride as the system so a prefix cache holds them. Every other key of the item (a
stratum, the keys a reply may cite) rides onto the graded row, where a grader can read it. The
flow:

    fire:   rendered = pack.render_source(source, **item.get("slots", {}))
            result   = generate(role, prompt=rendered, output=output(item), generator=generator)
                       or, with `user`: generate(role, prompt=user, system=rendered, ...)
    grade:  the role's call graders (`call.graders.board_graders`) plus any extra row graders,
            through one `Evaluator`, over `{**item, prompt, value, outcome, reason, retries}` rows

`fire_pack(resolved, ...)` is the whole run for a resolved pack (`prompts.resolve`, or
`resolve_pack` here, which falls back to the built-in call graders): it reads the golden set the
manifest names (a JSON list of items, a JSON object with an `items` list, or JSON lines), fires it
through the pack's own source and output type, grades it on the pack's graders, and stamps the run
and every row with the pack's `identity`. A run that cannot start (no golden set, a missing,
unreadable or empty file, an item with no `id`) is `could_not_run` with the reason, never a pass
over zero items.

`source` is the template text under evaluation, so a candidate revision is graded before it is
installed. `output(item)` is the `output=` type the producer itself compiles: a grammar that
differs from the live one grades a different model. `generator_for()` returns a fresh `Generator`
per item and the list its raw attempts land in (`adapters.pydantic_ai.replay.recording` builds
one), so a recording keeps every attempt and not only the last.

Invariants:
    - `fire` and `fire_pack` never raise for a model or wire failure: the item's outcome names it.
    - A row whose model never answered fails every call grader with its reason; the board names the
      outage instead of grading an absence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from llm_ports.call import generate as call_generate
from llm_ports.call.graders import GRADERS, CallGrader, board_graders
from llm_ports.call.roles import Role
from llm_ports.evals._gauntlet import GauntletEvaluator
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import Evaluator, Generated, Generator, Grader
from llm_ports.prompts import AnyGrader, GraderRegistry, Pack, Registry, ResolvedPack, resolve

#: `(item) -> the output= type` the producer generates under.
OutputFor = Callable[[Mapping[str, Any]], Any]
#: `() -> (a fresh Generator, the list its attempts land in)`.
GeneratorFor = Callable[[], tuple[Generator, list[Any]]]
#: The call path: `call.generate`, or a consumer's facade with the same keyword arguments.
Generate = Callable[..., Awaitable[Generated[Any]]]
#: Items fired concurrently by default.
PARALLEL = 2
_REASON_CAP = 200


async def fire_one(
    item: Mapping[str, Any],
    *,
    role: Role,
    pack: Pack,
    source: str,
    output: OutputFor,
    generator_for: GeneratorFor,
    gate: asyncio.Semaphore,
    generate: Generate = call_generate,
    on_event: EventHook | None = None,
) -> dict[str, Any]:
    """Fire one golden item and return its result.

    Returns:
        `{id, verdict, outcome, reason, retries, prompt, system, value, attempts, scores}`:
        `verdict` is `ok`, `gated` or `error`; `prompt` and `system` are what the model was sent
        (`system` empty unless the item carries `user`); `scores` holds each call grader's score
        as a dict.
    """
    rendered = pack.render_source(source, **(item.get("slots") or {}))
    user = item.get("user")
    prompt, system = (rendered, "") if user is None else (str(user), rendered)
    gen, attempts = generator_for()
    sys_kw = {"system": system} if system else {}  # a facade with no `system` keyword still fits
    async with gate:
        res = await generate(role, prompt=prompt, output=output(item), generator=gen, **sys_kw)
    verdict = res.outcome if res.outcome in ("ok", "gated") else "error"
    (on_event or log_events("llm_ports.evals.golden"))(
        logging.INFO,
        "golden_item",
        {"pack": pack.id, "identity": pack.identity, "id": item["id"], "outcome": res.outcome, "retries": res.retries},
    )
    return {
        "id": item["id"],
        "verdict": verdict,
        "attempts": attempts,
        "value": res.value,
        "prompt": prompt,
        "system": system,
        "reason": res.reason,
        "outcome": res.outcome,
        "retries": res.retries,
        "scores": [vars(s) for s in res.stamp.get("scores", ())],
    }


async def fire(
    items: Sequence[Mapping[str, Any]],
    *,
    role: Role,
    pack: Pack,
    source: str,
    output: OutputFor,
    generator_for: GeneratorFor,
    generate: Generate = call_generate,
    parallel: int = PARALLEL,
    on_event: EventHook | None = None,
) -> dict[str, dict[str, Any]]:
    """Fire every item, at most `parallel` at a time; return each result (`fire_one`) by item id."""
    gate = asyncio.Semaphore(parallel)
    outs = await asyncio.gather(
        *[
            fire_one(
                it,
                role=role,
                pack=pack,
                source=source,
                output=output,
                generator_for=generator_for,
                gate=gate,
                generate=generate,
                on_event=on_event,
            )
            for it in items
        ]
    )
    return {o["id"]: o for o in outs}


def rows_of(items: Sequence[Mapping[str, Any]], outs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the graded row per item: the item's own keys except `slots`, under the reply.

    An item with no result is `outcome="not_run"` with no value.
    """
    rows = []
    for it in items:
        o = outs.get(it["id"]) or {}
        rows.append(
            {
                **{k: v for k, v in it.items() if k != "slots"},
                "prompt": o.get("prompt"),
                "value": o.get("value"),
                "outcome": o.get("outcome", "not_run"),
                "reason": o.get("reason"),
                "retries": o.get("retries"),
            }
        )
    return rows


def monitors(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the board's rates: rows `answered`, and their `mean_retries` (None when none answered)."""
    answered = [r for r in rows if r.get("value") is not None]
    retries = [r["retries"] for r in answered if isinstance(r.get("retries"), int)]
    return {"answered": len(answered), "mean_retries": round(sum(retries) / len(retries), 3) if retries else None}


def grade(
    items: Sequence[Mapping[str, Any]],
    outs: Mapping[str, Mapping[str, Any]],
    role: Role,
    evaluator: Evaluator,
    *,
    extra: Mapping[str, Grader] | None = None,
    table: Mapping[str, CallGrader] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Grade fired items on one board: the role's call graders plus `extra` row graders.

    Args:
        items: The golden items.
        outs: `fire`'s results by item id.
        role: The judged role; its graders name the call graders, its pack (else its name) the
            board's surface.
        evaluator: The board's `Evaluator`.
        extra: Row graders beyond the call graders, by name.
        table: The call-grader table; None means `call.graders.GRADERS`.

    Returns:
        The board and the graded rows.
    """
    graders = {**board_graders(role, table), **(extra or {})}
    rows = rows_of(items, outs)
    return evaluator.run(role.pack or role.name, rows, graders, monitors=monitors), rows


def citations_resolve(cited: str, known: str) -> Grader:
    """Return a row grader: every key the reply lists under `cited` is one the row lists under `known`.

    It reads the raw reply, before any production step drops unknown citations, so a reply that
    cites what its evidence never carried fails here.
    """

    def grader(row: dict[str, Any]) -> tuple[bool, str]:
        value = row.get("value")
        if not isinstance(value, Mapping):
            return False, "no reply to read citations from"
        keys = value.get(cited) or []
        if not isinstance(keys, list):
            return False, f"{cited} is {type(keys).__name__}, not a list"
        have = {str(k) for k in row.get(known) or []}
        unknown = [k for k in keys if str(k) not in have]
        return (False, f"cited keys resolve to nothing: {unknown[:3]}") if unknown else (True, "")

    return grader


class _Chained:
    """A grader registry over the caller's, falling back to the built-in call graders."""

    def __init__(self, first: GraderRegistry | Registry | None) -> None:
        self._first = first

    def get(self, name: str) -> AnyGrader:
        first = self._first
        try:
            fn = None if first is None else first[name] if isinstance(first, Mapping) else first.get(name)
        except KeyError:
            fn = None
        return fn if fn is not None else GRADERS[name]


def resolve_pack(
    pack: Pack, *, registry: GraderRegistry | Registry | None = None, import_roots: Sequence[str] = ()
) -> ResolvedPack:
    """Return `prompts.resolve(pack, ...)` with `call.graders.GRADERS` behind `registry`.

    A grader name the caller's registry lacks resolves to the built-in grader of that name.

    Raises:
        PromptPackInvalid: A manifest pointer does not resolve; the message names the field.
    """
    return resolve(pack, registry=_Chained(registry), import_roots=import_roots)


@dataclass(frozen=True)
class GoldenRun:
    """One golden run of a resolved pack.

    Attributes:
        status: `ran`, or `could_not_run` when nothing was fired.
        reason: Why the run could not run; empty when it ran.
        stamp: The pack's stamp (`pack`, `version`, `sha`, `identity`), the role's `tier` and
            `model`, and `table_fingerprint` (None when the caller passed none). Two runs are
            comparable only when `identity` matches.
        rows: The graded rows (`rows_of`), each carrying `identity`.
        outs: `fire`'s results by item id.
        board: The evaluator's board; None when the run could not run.
    """

    status: Literal["ran", "could_not_run"]
    reason: str
    stamp: Mapping[str, str | None]
    rows: list[dict[str, Any]] = field(default_factory=list)
    outs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    board: Mapping[str, Any] | None = None


def read_golden(path: Path) -> list[dict[str, Any]]:
    """Read a golden set: a JSON list of items, a JSON object with an `items` list, or JSON lines (`.jsonl`).

    Returns:
        The items, in file order.

    Raises:
        ValueError: The file is not one of those shapes, or an item is not a mapping with an `id`.
        OSError: The file cannot be read.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        doc: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        doc = json.loads(text)
    items = doc.get("items") if isinstance(doc, dict) else doc
    if not isinstance(items, list):
        raise ValueError("not a list of items, nor an object with an `items` list")
    for i, it in enumerate(items):
        if not isinstance(it, dict) or "id" not in it:
            raise ValueError(f"item {i} is not a mapping with an `id`")
    return items


async def fire_pack(
    resolved: ResolvedPack,
    *,
    role: Role,
    generator_for: GeneratorFor,
    generate: Generate = call_generate,
    parallel: int = PARALLEL,
    on_event: EventHook | None = None,
    evaluator: Evaluator | None = None,
    table_fingerprint: str | None = None,
) -> GoldenRun:
    """Fire a resolved pack's golden set through its own source and output type, and grade it.

    Args:
        resolved: The pack with its manifest resolved (`prompts.resolve` or `resolve_pack`).
        role: The role the items fire under; its tier and model are stamped on the run.
        generator_for: A fresh generator per item and the list its attempts land in.
        generate: The call path.
        parallel: Items fired concurrently.
        on_event: The event sink; defaults to the stdlib logger `llm_ports.evals.golden`. Emits one
            `golden_item` per item and one `golden_run` per run (a WARNING when it could not run).
        evaluator: The board's evaluator; defaults to `GauntletEvaluator()`.
        table_fingerprint: The tier table's fingerprint, stamped on the run; None when unknown.

    Returns:
        The run. No golden set declared, or a missing, unreadable or empty one, is `could_not_run`
        with the reason, never a pass over zero items. The board grades the pack's own graders.

    Invariants:
        - Never raises for a model, wire or golden-file failure.
    """
    pack = resolved.pack
    emit = on_event or log_events(__name__)
    stamp: dict[str, str | None] = {
        **pack.stamp(),
        "tier": role.tier,
        "model": role.resolved_model,
        "table_fingerprint": table_fingerprint,
    }

    def could_not_run(reason: str) -> GoldenRun:
        emit(logging.WARNING, "golden_run", {**stamp, "status": "could_not_run", "reason": reason})
        return GoldenRun("could_not_run", reason, stamp)

    if resolved.golden is None:
        return could_not_run("the pack declares no golden set")
    if not resolved.golden.is_file():
        return could_not_run(f"golden file missing: {resolved.golden.name}")
    try:
        items = read_golden(resolved.golden)
    except (OSError, ValueError) as exc:  # a JSON or UTF-8 decode error is a ValueError
        return could_not_run(f"golden file unreadable: {type(exc).__name__}: {exc}"[:_REASON_CAP])
    if not items:
        return could_not_run("golden set is empty")
    output: Any = resolved.output if resolved.output is not None else str
    outs = await fire(
        items,
        role=role,
        pack=pack,
        source=pack.source,
        output=lambda _item: output,
        generator_for=generator_for,
        generate=generate,
        parallel=parallel,
        on_event=on_event,
    )
    board_role = replace(role, graders=tuple(resolved.graders))
    board, rows = grade(items, outs, board_role, evaluator or GauntletEvaluator(), table=resolved.graders)
    rows = [{**r, "identity": pack.identity} for r in rows]
    emit(logging.INFO, "golden_run", {**stamp, "status": "ran", "n": len(rows), "rows_failed": board["rows_failed"]})
    return GoldenRun("ran", "", stamp, rows, outs, board)


__all__ = [
    "PARALLEL",
    "Generate",
    "GeneratorFor",
    "GoldenRun",
    "OutputFor",
    "citations_resolve",
    "fire",
    "fire_one",
    "fire_pack",
    "grade",
    "monitors",
    "read_golden",
    "resolve_pack",
    "rows_of",
]
