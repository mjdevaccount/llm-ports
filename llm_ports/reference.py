"""Framework-free implementations behind the ports that have one.

A consumer gets a working default before, or instead of, a framework adapter.

- `GauntletEvaluator`: an `Evaluator` that runs deterministic graders over rows into one board,
  the shape the port promises. Pure: rows in, board out.
- `NumberLawGrounder`: a `Grounder` that applies the number law (`numbers.uncited`): every figure
  in a claim must appear in a cited span. It judges numbers only, a necessary check and never a
  sufficient one.
- `RepoPromptSource`: a `PromptSource` with no managed store; every name is its repo `Seed`.
- `NoopTracer`: a `Tracer` with no sink; every generation is a `NoopGeneration`. It is also the
  null trace `call.generate` uses when no `trace` hook is given, so the package has one null sink.

The test doubles (`StubCiter`, `StubPromptSource`, ...) live in `llm_ports.testing`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from llm_ports import numbers
from llm_ports.ports import Chunk, Claim, Grader, GroundingVerdict, Prompt, Seed

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


class NumberLawGrounder:
    """A `Grounder` that applies the number law (`numbers.uncited`).

    Every digit run and spelled number in the claim must appear in a cited span.

    Invariants:
        - It judges numbers only: a necessary check, never a sufficient one. A numberless claim
          passes with reason `no_numbers` because this checker has nothing to say about it, not
          because a span entails it. Entailment is another `Grounder` behind the same port.
        - No spans means unsupported, always.
        - `score` is None: the verdict has no scale.

    Rationale: ADR-0003.
    """

    checker = "number_law"

    def verdict(self, claim: Claim, spans: Sequence[Chunk]) -> GroundingVerdict:
        """Return supported only when every number in the claim appears in one of `spans`."""
        if not spans:
            return GroundingVerdict(False, "no evidence spans", checker=self.checker)
        missing = numbers.uncited(claim.text, ((s.title, s.text) for s in spans))
        if missing:
            return GroundingVerdict(False, "number_not_in_span: " + ", ".join(missing[:4]), checker=self.checker)
        if not (numbers.digits(claim.text) or numbers.spelled(claim.text)):
            return GroundingVerdict(True, "no_numbers: nothing for number_law to judge", checker=self.checker)
        return GroundingVerdict(True, "every number cited", checker=self.checker)


class RepoPromptSource:
    """A `PromptSource` for a process with no managed store.

    Every name is its repo seed (`seed_for`, the same callable a managed-store adapter mirrors),
    served as `Prompt(repo)` with the seed's config and system template: what a managed adapter
    serves when its store is down, without the round-trip.
    """

    def __init__(self, seed_for: Callable[[str], Seed]) -> None:
        self._seed_for = seed_for

    def get(self, name: str, *, label: str = "production") -> Prompt:
        """Return the repo seed for `name` as a prompt.

        `label` is accepted and ignored: a repo has one copy. A name without a seed raises from
        `seed_for` (the port's law: no seed, no text).
        """
        s = self._seed_for(name)
        return Prompt(name, s.text, None, "repo", config=dict(s.config), system=s.system)


class NoopGeneration:
    """A `Generation` that records nothing.

    It is what a sink-less `Tracer` returns, and so what `call.generate` observes into when no
    `trace` hook is given.
    """

    def end(self, **_kw: Any) -> None:
        """Record nothing."""
        return None

    def score(self, *_a: Any, **_kw: Any) -> None:
        """Record nothing."""
        return None


class NoopTracer:
    """A `Tracer` for a process with no observability sink.

    Every `start_generation` returns a `NoopGeneration`, so a producer never branches on whether a
    sink is configured.
    """

    def start_generation(self, **_kw: Any) -> NoopGeneration:
        """Return a `NoopGeneration`."""
        return NoopGeneration()


__all__ = [
    "BOARD_EXAMPLES",
    "GauntletEvaluator",
    "NoopGeneration",
    "NoopTracer",
    "NumberLawGrounder",
    "RepoPromptSource",
]
