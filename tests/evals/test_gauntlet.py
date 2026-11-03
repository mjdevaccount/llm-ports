"""GauntletEvaluator keeps the Evaluator port's laws, and each law test FIRES on an impl that breaks it."""

from __future__ import annotations

from typing import Any

import pytest

from llm_ports import ports
from llm_ports.evals import BOARD_EXAMPLES, GauntletEvaluator


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
