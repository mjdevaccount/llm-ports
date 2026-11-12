"""Conformance checks for the remaining ports: each takes a factory and returns the cases it checked.

Every check builds what it tests through `make()`, calls the port the way the library does, and
raises `ContractViolation` naming the case on the first broken invariant. An exception where
the port promises never to raise is the case `never_raises`.

Rationale: ADR-0001, ADR-0016, ADR-0025.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_ports.observability._payload import deliver_or_raise, end_payload
from llm_ports.ports import (
    Answered,
    Chunk,
    CitedAnswer,
    Citer,
    Claim,
    Cost,
    Embedded,
    Embedder,
    Evaluator,
    GatewayReader,
    Generation,
    Grounder,
    GroundingVerdict,
    Health,
    Judge,
    JudgeVerdict,
    Limiter,
    Pricer,
    Probe,
    PromptSource,
    Reranked,
    Reranker,
    Retriever,
    Tracer,
    Usage,
    failure,
)
from llm_ports.testing.conformance._core import call, call_sync, expect, is_instance, tag

_NEVER = "never_raises"
_BOARD_KEYS = frozenset({"surface", "n", "rows_failed", "by_grader", "monitors"})


async def check_embedder(make: Callable[[], Embedder]) -> tuple[str, ...]:
    """One vector per text in one space; no texts is not an error; the stamp is constant."""
    emb = make()
    is_instance("is_embedder", emb, Embedder)
    stamp = emb.stamp
    expect("stamp_constant", bool(stamp) and emb.stamp == stamp, f"stamp {stamp!r} then {emb.stamp!r}")
    got: Embedded = await call(_NEVER, "embed", emb.embed(["alpha", "beta"]))
    if got.outcome == "ok":
        dims = {len(v) for v in got.vectors}
        expect("one_vector_each", len(got.vectors) == 2 and len(dims) == 1, f"2 texts gave {len(got.vectors)}")
    else:
        expect(
            "failure_named", got.failure is not None and not got.vectors, f"{got.outcome} with vectors or no failure"
        )
    await call(_NEVER, "embed([])", emb.embed([], input_type="query"))
    return ("is_embedder", "stamp_constant", "one_vector_each", _NEVER)


async def check_reranker(make: Callable[[], Reranker]) -> tuple[str, ...]:
    """One score per document in input order; no documents is an `ok` with no scores."""
    rr = make()
    is_instance("is_reranker", rr, Reranker)
    got: Reranked = await call(_NEVER, "rerank", rr.rerank("alpha", ["alpha beta", "gamma", "alpha"]))
    if got.outcome == "ok":
        expect("one_score_each", len(got.scores) == 3, f"3 documents gave {len(got.scores)} scores")
    else:
        expect("failure_named", got.failure is not None and not got.scores, f"{got.outcome} with scores or no failure")
    empty: Reranked = await call(_NEVER, "rerank([])", rr.rerank("alpha", []))
    expect("empty_is_ok", empty.outcome == "ok" and empty.scores == (), f"no documents gave {empty.outcome}")
    return ("is_reranker", "one_score_each", "empty_is_ok", _NEVER)


def _full_payload() -> dict[str, Any]:
    return end_payload(
        output="out",
        usage=Usage(input=3, output=None, requests=1),
        cost=Cost(None, "unknown", reason="unknown_model"),
        answered=Answered(requested="m", model="m"),
        failure=failure("server", reason="conformance"),
        metadata={"outcome": "unavailable"},
    )


async def check_generation(make: Callable[[], Generation]) -> tuple[str, ...]:
    """`end` takes the full 1.24 payload (or the 1.x five) and `score` takes metadata; neither raises."""
    gen = make()
    is_instance("is_generation", gen, Generation)
    call_sync(_NEVER, "end", lambda: deliver_or_raise(gen, _full_payload()))
    call_sync(_NEVER, "score", lambda: gen.score("conformance", 1.0, None, {"by": "check"}))
    return ("is_generation", _NEVER)


async def check_tracer(make: Callable[[], Tracer]) -> tuple[str, ...]:
    """`start_generation` returns a `Generation` that passes `check_generation`; nothing raises."""
    tracer = make()
    is_instance("is_tracer", tracer, Tracer)
    meta = {"role": f"conformance-{tag()}"}

    def opened() -> Generation:
        return call_sync(
            _NEVER,
            "start_generation",
            lambda: tracer.start_generation(
                name="conformance",
                model="m",
                provider="p",
                input="in",
                metadata=meta,
                session_id=None,
                prompt_name=None,
            ),
        )

    await check_generation(opened)
    ok = opened()
    call_sync(_NEVER, "end", lambda: deliver_or_raise(ok, end_payload(output="ok", usage=None)))
    return ("is_tracer", "is_generation", _NEVER)


async def check_pricer(make: Callable[[], Pricer]) -> tuple[str, ...]:
    """An unknown model and unmeasured usage both price to `usd=None`; neither raises."""
    pricer = make()
    is_instance("is_pricer", pricer, Pricer)
    nobody = f"no-such-model-{tag()}"
    unknown = call_sync(
        _NEVER, "price", lambda: pricer.price(Usage(input=10, output=10, requests=1), Answered(nobody, nobody))
    )
    expect("unknown_is_none", unknown.usd is None, f"an unknown model priced at {unknown.usd}")
    blank = call_sync(_NEVER, "price", lambda: pricer.price(Usage(requests=1, unmeasured=1), Answered("m", "m")))
    expect("unmeasured_is_none", blank.usd is None, f"unmeasured usage priced at {blank.usd}")
    return ("is_pricer", "unknown_is_none", "unmeasured_is_none", _NEVER)


async def check_probe(make: Callable[[], Probe]) -> tuple[str, ...]:
    """`probe` returns a `Health` in a known state and never raises."""
    probe = make()
    is_instance("is_probe", probe, Probe)
    got: Health = await call(_NEVER, "probe", probe.probe())
    expect("known_state", got.state in ("up", "down", "unknown"), f"state {got.state!r}")
    return ("is_probe", "known_state", _NEVER)


async def check_retriever(make: Callable[[], Retriever]) -> tuple[str, ...]:
    """A blank query is `[]`; a search returns at most `k` chunks."""
    ret = make()
    is_instance("is_retriever", ret, Retriever)
    blank = await call(_NEVER, "search('')", ret.search("  "))
    expect("blank_is_empty", blank == [], f"a blank query gave {len(blank)} chunks")
    got = await call(_NEVER, "search", ret.search("conformance", k=2))
    expect("at_most_k", len(got) <= 2 and all(isinstance(c, Chunk) for c in got), f"k=2 gave {len(got)}")
    return ("is_retriever", "blank_is_empty", "at_most_k", _NEVER)


async def check_citer(make: Callable[[], Citer]) -> tuple[str, ...]:
    """No chunks yield no claims; every citation is a chunk id it was given."""
    citer = make()
    is_instance("is_citer", citer, Citer)
    none: CitedAnswer = await call(_NEVER, "answer([])", citer.answer("q", []))
    expect("no_chunks_no_claims", none.claims == (), f"no chunks gave {len(none.claims)} claims")
    chunks = [Chunk(id="c1", text="alpha is one"), Chunk(id="c2", text="beta is two")]
    got: CitedAnswer = await call(_NEVER, "answer", citer.answer("what is alpha?", chunks))
    expect("cites_known_ids", got.unknown_ids() == (), f"cited unknown ids {got.unknown_ids()}")
    return ("is_citer", "no_chunks_no_claims", "cites_known_ids")


async def check_grounder(make: Callable[[], Grounder]) -> tuple[str, ...]:
    """A claim with no spans is never supported."""
    g = make()
    is_instance("is_grounder", g, Grounder)
    got: GroundingVerdict = call_sync(_NEVER, "verdict", lambda: g.verdict(Claim("alpha is 1"), []))
    expect("no_spans_unsupported", not got.supported, "a claim with no spans was supported")
    return ("is_grounder", "no_spans_unsupported")


async def check_evaluator(make: Callable[[], Evaluator]) -> tuple[str, ...]:
    """The board has every key, always."""
    ev = make()
    is_instance("is_evaluator", ev, Evaluator)
    board = call_sync(_NEVER, "run", lambda: ev.run("conformance", [{"id": 1}], {"g": lambda _r: (False, "no")}))
    expect("board_shape", _BOARD_KEYS <= set(board), f"board keys {sorted(board)}")
    return ("is_evaluator", "board_shape")


async def check_prompt_source(make: Callable[[], PromptSource], name: str) -> tuple[str, ...]:
    """`get(name)` serves non-empty text under that name.

    Args:
        make: Builds the source.
        name: A prompt the source has (at least as a repo seed).
    """
    src = make()
    is_instance("is_prompt_source", src, PromptSource)
    got = call_sync(_NEVER, "get", lambda: src.get(name))
    expect("text_not_empty", bool(got.text) and got.name == name, f"get({name!r}) served {got}")
    return ("is_prompt_source", "text_not_empty")


async def check_judge(make: Callable[[], Judge]) -> tuple[str, ...]:
    """A judgement is a `JudgeVerdict` with a numeric score (a judge may raise when it cannot answer)."""
    j = make()
    is_instance("is_judge", j, Judge)
    got = await j.judge(prompt="2+2?", output="4", rubric="correct arithmetic")
    expect("verdict_shape", isinstance(got, JudgeVerdict) and isinstance(got.score, float | int), f"{got!r}")
    return ("is_judge", "verdict_shape")


async def check_gateway_reader(make: Callable[[], GatewayReader]) -> tuple[str, ...]:
    """No headers report no cost and no deployment."""
    r = make()
    is_instance("is_gateway_reader", r, GatewayReader)
    expect("absent_is_none", call_sync(_NEVER, "cost", lambda: r.cost_from_headers({})) is None, "cost from {}")
    who = call_sync(_NEVER, "answered", lambda: r.answered_from_headers({}))
    expect("absent_is_none", who == (None, None), f"answered from {{}} = {who}")
    return ("is_gateway_reader", "absent_is_none")


async def check_limiter(make: Callable[[], Limiter]) -> tuple[str, ...]:
    """A free slot is held at once and released: the wait is a non-negative number of seconds."""
    lim = make()
    is_instance("is_limiter", lim, Limiter)
    group = f"conformance-{tag()}"
    for _ in range(2):  # a slot not released would block the second take on a one-slot group
        async with lim.slot(group) as waited:
            expect("wait_is_seconds", isinstance(waited, float | int) and waited >= 0, f"waited {waited!r}")
    return ("is_limiter", "wait_is_seconds")


__all__ = [
    "check_citer",
    "check_embedder",
    "check_evaluator",
    "check_gateway_reader",
    "check_generation",
    "check_grounder",
    "check_judge",
    "check_limiter",
    "check_pricer",
    "check_probe",
    "check_prompt_source",
    "check_reranker",
    "check_retriever",
    "check_tracer",
]
