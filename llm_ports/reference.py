"""reference -- the framework-free impl behind each port that has one, so a consumer has a working
default before (or instead of) a framework adapter.

- `GauntletEvaluator`  -- `Evaluator`: deterministic graders over rows -> ONE board (the shape the
                          port promises). Pure: rows in, board out.
- `NumberLawGrounder`  -- `Grounder`: the number law (`numbers.uncited`) -- every figure in a claim
                          must appear in a cited span. NUMBERS ONLY: necessary, never sufficient.
- `RepoPromptSource`   -- `PromptSource` with no managed store: every name is its repo `Seed`.
- `NoopTracer`         -- `Tracer` with no sink: every generation is a `NoopGeneration`. Also the
                          null trace `call.generate` uses when no `trace` hook is given -- ONE null
                          sink in the package.

The test doubles (`StubCiter`, `StubPromptSource`, ...) live in `llm_ports.testing`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from llm_ports import numbers
from llm_ports.ports import Chunk, Claim, Grader, GroundingVerdict, Prompt, Seed

BOARD_EXAMPLES = 3            # (id, reason) pairs kept per grader: the wound named, never the corpse


class GauntletEvaluator:
    """`Evaluator` = the gauntlet: run the deterministic graders over a set of rows and produce ONE
    board, so every reader of a surface's quality (a CI golden gate, a scheduled scorer, an ad-hoc
    script) gets the same numbers:

        {"surface": ..., "n": ..., "rows_failed": N,
         "by_grader": {grader: {"pass": n, "fail": n, "examples": [...]}},
         "monitors": {...}}

    `examples` carries at most `BOARD_EXAMPLES` (id, reason) pairs per grader. A grader that RAISES
    is a bug in the consumer's code and raises out of here; where that lands is the caller's job."""

    def run(self, surface: str, rows: list[dict[str, Any]], graders: Mapping[str, Grader],
            *, id_field: str = "id",
            monitors: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None) -> dict[str, Any]:
        by_grader: dict[str, dict[str, Any]] = {
            name: {"pass": 0, "fail": 0, "examples": []} for name in graders
        }
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
    """`Grounder` = the number law (`numbers.uncited`): every digit run and spelled number in the
    claim must appear in a cited span. It judges NUMBERS ONLY -- a necessary check, never
    sufficient: a numberless claim passes with reason `no_numbers` because this checker has nothing
    to say about it, not because the span entails it. Entailment is another Grounder behind the
    same port. No spans -> unsupported, always. `score` is None -- no scale."""

    checker = "number_law"

    def verdict(self, claim: Claim, spans: Sequence[Chunk]) -> GroundingVerdict:
        if not spans:
            return GroundingVerdict(False, "no evidence spans", checker=self.checker)
        missing = numbers.uncited(claim.text, ((s.title, s.text) for s in spans))
        if missing:
            return GroundingVerdict(False, "number_not_in_span: " + ", ".join(missing[:4]),
                                    checker=self.checker)
        if not (numbers.digits(claim.text) or numbers.spelled(claim.text)):
            return GroundingVerdict(True, "no_numbers: nothing for number_law to judge",
                                    checker=self.checker)
        return GroundingVerdict(True, "every number cited", checker=self.checker)


class RepoPromptSource:
    """`PromptSource` for a process with NO managed store: every name is its repo SEED (`seed_for`,
    the same callable a managed-store adapter mirrors), served as `Prompt(repo)` with the seed's
    config and system template -- what a managed adapter serves when its store is down, minus the
    round-trip. `label` is accepted and ignored: a repo has one copy. A name without a seed RAISES
    from `seed_for` (the port's law: no seed, no text)."""

    def __init__(self, seed_for: Callable[[str], Seed]) -> None:
        self._seed_for = seed_for

    def get(self, name: str, *, label: str = "production") -> Prompt:
        s = self._seed_for(name)
        return Prompt(name, s.text, None, "repo", config=dict(s.config), system=s.system)


class NoopGeneration:
    """`Generation` that records nothing: what a sink-less `Tracer` hands back, and so what
    `call.generate` observes into when no `trace` hook is given."""

    def end(self, **_kw: Any) -> None:
        return None

    def score(self, *_a: Any, **_kw: Any) -> None:
        return None


class NoopTracer:
    """`Tracer` for a process with no observability sink: every `start_generation` is a
    `NoopGeneration`, so a producer never branches on whether a sink is configured."""

    def start_generation(self, **_kw: Any) -> NoopGeneration:
        return NoopGeneration()


__all__ = ["BOARD_EXAMPLES", "GauntletEvaluator", "NoopGeneration", "NoopTracer", "NumberLawGrounder",
           "RepoPromptSource"]
