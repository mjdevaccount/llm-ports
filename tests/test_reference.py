"""reference + the seam doubles: each framework-free impl satisfies its port and keeps its laws, and
each law test FIRES on an impl that breaks it. Moved from trade-data's adapters/ai_ours_port.py and
its contract/mirror tests (v0.9.0)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from llm_ports import ports
from llm_ports.ports import Chunk, Claim, Prompt, Seed
from llm_ports.reference import (
    BOARD_EXAMPLES,
    GauntletEvaluator,
    NoopGeneration,
    NoopTracer,
    NumberLawGrounder,
    RepoPromptSource,
)
from llm_ports.testing import StubCiter, StubPromptSource

SPAN = Chunk(id="a", text="SPY closed at 580.5, up 3%", title="Close")


# ---- Evaluator ----


def _evaluator_laws(e: ports.Evaluator) -> None:
    assert isinstance(e, ports.Evaluator)
    rows = [{"id": 1, "x": 1}, {"id": 2, "x": 0}]
    graders = {"x_pos": lambda r: (r["x"] > 0, "x<=0")}
    board = e.run("t", rows, graders)
    assert set(board) == {"surface", "n", "rows_failed", "by_grader", "monitors"}
    assert board["n"] == 2 and board["rows_failed"] == 1
    cell = board["by_grader"]["x_pos"]
    assert cell == {"pass": 1, "fail": 1, "examples": [{"id": 2, "reason": "x<=0"}]}
    assert board["monitors"] == {}, "no monitors passed -> the key is present and empty, never absent"
    with_m = e.run("t", rows, graders, monitors=lambda rs: {"seen": len(rs), "ids": [r["id"] for r in rs]})
    assert with_m["monitors"] == {"seen": 2, "ids": [1, 2]}
    assert {k: v for k, v in with_m.items() if k != "monitors"} == {k: v for k, v in board.items() if k != "monitors"}


def test_the_gauntlet_keeps_the_evaluator_laws() -> None:
    _evaluator_laws(GauntletEvaluator())


def test_the_evaluator_laws_fire_on_an_adapter_that_drops_monitors() -> None:
    class Dropping:
        def run(self, surface, rows, graders, *, id_field="id", monitors=None):
            return {**GauntletEvaluator().run(surface, rows, graders, id_field=id_field), "monitors": {}}
    with pytest.raises(AssertionError):
        _evaluator_laws(Dropping())


def test_the_board_names_at_most_three_wounds_and_counts_a_row_once() -> None:
    rows = [{"key": i} for i in range(5)]
    graders = {"never": lambda r: (False, f"row {r['key']}"), "also_never": lambda r: (False, "x")}
    board = GauntletEvaluator().run("s", rows, graders, id_field="key")
    assert board["rows_failed"] == 5, "a row failing two graders is ONE failed row"
    assert [x["id"] for x in board["by_grader"]["never"]["examples"]] == list(range(BOARD_EXAMPLES))


def test_a_grader_that_raises_raises_out_of_the_board() -> None:
    def boom(_row: dict[str, Any]) -> tuple[bool, str]:
        raise RuntimeError("grader bug")
    with pytest.raises(RuntimeError, match="grader bug"):
        GauntletEvaluator().run("s", [{"id": 1}], {"boom": boom})


# ---- Grounder ----


def test_the_number_law_grounder_keeps_the_grounder_laws() -> None:
    g = NumberLawGrounder()
    assert isinstance(g, ports.Grounder)
    v = g.verdict(Claim("SPY at 580.5"), [])
    assert v.supported is False and v.checker == "number_law" and v.reason
    assert g.verdict(Claim("SPY at 580.5, up 3 percent", ("a",)), [SPAN]).supported is True
    bad = g.verdict(Claim("SPY at 581", ("a",)), [SPAN])
    assert bad.supported is False and "581" in bad.reason


def test_the_number_law_grounder_names_its_blind_spot() -> None:
    g = NumberLawGrounder()
    # a spelled number is a number no span rendered, even when '3%' is cited
    assert g.verdict(Claim("up three percent", ("a",)), [SPAN]).supported is False
    none = g.verdict(Claim("SPY will rally hard", ("a",)), [SPAN])
    assert none.supported is True and "no_numbers" in none.reason


# ---- PromptSource ----


def _prompt_laws(s: ports.PromptSource, name: str) -> Prompt:
    assert isinstance(s, ports.PromptSource)
    p = s.get(name)
    assert isinstance(p, Prompt) and p.name == name and p.text
    assert (p.version is None) == (p.source == "repo"), "version None <=> served from the repo seed"
    assert isinstance(p.config, Mapping)
    return p


def test_the_repo_and_stub_sources_keep_the_prompt_source_laws() -> None:
    _prompt_laws(StubPromptSource({"demo": "You are {{ who }}."}), "demo")
    p = _prompt_laws(RepoPromptSource(lambda n: Seed("You are {{ who }}.", "0123456789abcdef", {"model": "m"},
                                                     system="sys")), "demo")
    assert p.system == "sys" and p.config == {"model": "m"}


def test_the_prompt_source_laws_fire_on_a_versioned_repo_prompt() -> None:
    class Lying:
        def get(self, name: str, *, label: str = "production") -> Prompt:
            return Prompt(name, "t", 3, "repo")
    with pytest.raises(AssertionError, match="version None"):
        _prompt_laws(Lying(), "demo")


def test_a_name_without_a_seed_raises_never_invents_text() -> None:
    with pytest.raises(KeyError):
        StubPromptSource({}).get("missing")

    def no_seed(name: str) -> Seed:
        raise KeyError(name)
    with pytest.raises(KeyError):
        RepoPromptSource(no_seed).get("missing")


def test_the_stub_source_records_the_label_asked_for() -> None:
    assert StubPromptSource({"d": "t"}).get("d", label="staging").config == {"label": "staging"}


# ---- Citer ----


async def test_the_stub_citer_keeps_the_citer_laws() -> None:
    c = StubCiter()
    assert isinstance(c, ports.Citer)
    empty = await c.answer("q", [])
    assert empty.claims == () and empty.unknown_ids() == ()
    ans = await c.answer("q", [SPAN])
    assert ans.unknown_ids() == () and set(ans.chunks) == {"a"} and ans.model == ""


# ---- Tracer ----


def test_the_noop_tracer_is_a_tracer_whose_generation_swallows_everything() -> None:
    t = NoopTracer()
    assert isinstance(t, ports.Tracer)
    g = t.start_generation(name="n", model="m", provider="p", input="i", metadata={}, session_id=None,
                           prompt_name=None)
    assert isinstance(g, NoopGeneration) and isinstance(g, ports.Generation)
    assert g.end(output="o", usage=None, level="ERROR", status_message="x") is None
    assert g.score("s", 1.0, "c") is None
