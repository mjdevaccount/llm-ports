"""Golden-set evaluation of a prompt pack: fire each item through the call path, grade on one board.

A golden item is a mapping with an `id` and the pack's template context under `slots`. Every other
key of the item (a stratum, the keys a reply may cite) rides onto the graded row, where a grader
can read it. The flow:

    fire:   prompt = pack.render_source(source, **item["slots"])
            result = generate(role, prompt=prompt, output=output(item), generator=generator)
    grade:  the role's call graders (`call.graders.board_graders`) plus any extra row graders,
            through one `Evaluator`, over `{**item, prompt, value, outcome, reason, retries}` rows

`source` is the template text under evaluation, so a candidate revision is graded before it is
installed. `output(item)` is the `output=` type the producer itself compiles: a grammar that
differs from the live one grades a different model. `generator_for()` returns a fresh `Generator`
per item and the list its raw attempts land in (`adapters.pydantic_ai.replay.recording` builds
one), so a recording keeps every attempt and not only the last.

Invariants:
    - `fire` never raises for a model or wire failure: the item's outcome names it.
    - A row whose model never answered fails every call grader with its reason; the board names the
      outage instead of grading an absence.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from llm_ports.call import generate as call_generate
from llm_ports.call.graders import CallGrader, board_graders
from llm_ports.call.roles import Role
from llm_ports.observability import EventHook, log_events
from llm_ports.ports import Evaluator, Generated, Generator, Grader
from llm_ports.prompts import Pack

#: `(item) -> the output= type` the producer generates under.
OutputFor = Callable[[Mapping[str, Any]], Any]
#: `() -> (a fresh Generator, the list its attempts land in)`.
GeneratorFor = Callable[[], tuple[Generator, list[Any]]]
#: The call path: `call.generate`, or a consumer's facade with the same keyword arguments.
Generate = Callable[..., Awaitable[Generated[Any]]]
#: Items fired concurrently by default.
PARALLEL = 2


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
        `{id, verdict, outcome, reason, retries, prompt, value, attempts, scores}`: `verdict` is
        `ok`, `gated` or `error`; `scores` holds each call grader's score as a dict.
    """
    prompt = pack.render_source(source, **item["slots"])
    gen, attempts = generator_for()
    async with gate:
        res = await generate(role, prompt=prompt, output=output(item), generator=gen)
    verdict = res.outcome if res.outcome in ("ok", "gated") else "error"
    (on_event or log_events("llm_ports.evals.golden"))(
        logging.INFO, "golden_item", {"pack": pack.id, "id": item["id"], "outcome": res.outcome, "retries": res.retries}
    )
    return {
        "id": item["id"],
        "verdict": verdict,
        "attempts": attempts,
        "value": res.value,
        "prompt": prompt,
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


__all__ = [
    "PARALLEL",
    "Generate",
    "GeneratorFor",
    "OutputFor",
    "citations_resolve",
    "fire",
    "fire_one",
    "grade",
    "monitors",
    "rows_of",
]
