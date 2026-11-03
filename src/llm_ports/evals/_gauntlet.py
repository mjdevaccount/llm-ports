"""`GauntletEvaluator`: an `Evaluator` that runs deterministic graders over rows into one board.

The board is the shape the port promises. Pure: rows in, board out.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from llm_ports.ports import Grader

BOARD_EXAMPLES = 3  # (id, reason) pairs a board keeps per failing grader

_Monitors = Callable[[list[dict[str, Any]]], dict[str, Any]]  # a surface's own rates over all rows


class GauntletEvaluator:
    """An `Evaluator` that runs deterministic graders over a set of rows into one board.

    Every reader of a surface's quality (a CI golden gate, a scheduled scorer, an ad-hoc script)
    gets the same numbers. The board's shape:

        {"surface": ..., "n": ..., "rows_failed": N,
         "by_grader": {grader: {"pass": n, "fail": n, "examples": [...]}},
         "monitors": {...}}

    `examples` carries at most `BOARD_EXAMPLES` `(id, reason)` pairs per grader.
    """

    def run(
        self,
        surface: str,
        rows: list[dict[str, Any]],
        graders: Mapping[str, Grader],
        *,
        id_field: str = "id",
        monitors: _Monitors | None = None,
    ) -> dict[str, Any]:
        """Grade every row with every grader and return the board.

        `id_field` names the row key reported in `examples`; `monitors`, when given, computes the
        board's `monitors` entry over all rows. A grader that raises is a bug in the consumer's
        code: the exception propagates unchanged, and where it lands is the caller's concern.
        """
        by_grader: dict[str, dict[str, Any]] = {name: {"pass": 0, "fail": 0, "examples": []} for name in graders}
        failed_rows: set[Any] = set()
        for row in rows:
            for name, fn in graders.items():
                passed, reason = fn(row)
                cell = by_grader[name]
                if passed:
                    cell["pass"] += 1
                else:
                    cell["fail"] += 1
                    failed_rows.add(row.get(id_field))
                    if len(cell["examples"]) < BOARD_EXAMPLES:
                        cell["examples"].append({"id": row.get(id_field), "reason": reason})
        return {
            "surface": surface,
            "n": len(rows),
            "rows_failed": len(failed_rows),
            "by_grader": by_grader,
            "monitors": monitors(rows) if monitors is not None else {},
        }


__all__ = ["BOARD_EXAMPLES", "GauntletEvaluator"]
