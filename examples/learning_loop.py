"""A learning loop from evidence to a recorded verdict, on the standard library alone.

The loop learns a decision threshold from labelled outcomes, holds the fit to a gate, applies it
only when the gate passes, and leaves exactly one ledger row per run, whatever happens. A loop
holds only what is specific to it (`evidence`, `fit`, `clauses`, `apply`); `run` does the rest.

The second half reads labelled examples the way a judge or a coverage report does: how many
human labels each topic has, how far a judge agrees with them, and a balanced few-shot pick.
"""

import asyncio
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_ports.learn import Clause, Example, Fit, Verdict, examples, gate, run
from llm_ports.learn.stores import JsonlRunLedger, MemoryExampleStore


@dataclass
class Threshold:
    """Learn the score above which outcomes are mostly wins, and adopt it only when earned."""

    scores: Sequence[tuple[float, bool]]  # (score, won)
    kind: str = "threshold"
    subject: str = "demo"
    params: Mapping[str, Any] = field(default_factory=lambda: {"min_n": 20, "min_hit": 0.6})
    adopted: float | None = None

    async def evidence(self) -> Sequence[tuple[float, bool]]:
        return self.scores  # a real loop reads a repository here

    def fit(self, evidence: Sequence[tuple[float, bool]]) -> Fit:
        best = max(
            sorted({s for s, _ in evidence}), key=lambda t: _hit([w for s, w in evidence if s >= t]), default=None
        )
        above = [w for s, w in evidence if best is not None and s >= best]
        return Fit(model={"threshold": best}, metrics={"n": len(evidence), "n_above": len(above), "hit": _hit(above)})

    def clauses(self, fit: Fit, evidence: Sequence[tuple[float, bool]]) -> Sequence[Clause]:
        return [
            gate.at_least("n", fit.metrics["n"], self.params["min_n"], role="floor"),
            gate.at_least("hit_rate", fit.metrics["hit"], self.params["min_hit"]),
        ]

    async def apply(self, verdict: Verdict, fit: Fit) -> str:
        if not verdict.passed:
            return "held"  # the incumbent stays
        self.adopted = fit.model["threshold"]
        return "stored"


class Unreachable(Threshold):
    """The same loop, with its store down."""

    async def evidence(self) -> Sequence[tuple[float, bool]]:
        raise ConnectionError("the outcomes table is unreachable")


def _hit(wins: Sequence[bool]) -> float | None:
    return sum(wins) / len(wins) if wins else None


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ledger = JsonlRunLedger(Path(tmp) / "runs.jsonl")

        # Too little evidence: `insufficient`, nothing adopted, and still one row.
        early = Threshold([(0.9, True), (0.2, False)])
        first = await run(early, ledger)
        assert (first.verdict.decision, first.verdict.blocker, first.applied) == ("insufficient", "n", "held")
        assert early.adopted is None

        # Enough evidence: the gate passes and the loop adopts the fitted threshold.
        data = [(s / 10, s >= 6) for s in range(10)] * 3
        later = Threshold(data)
        second = await run(later, ledger)
        assert second.verdict.passed and second.applied == "stored" and later.adopted == 0.6

        # A stage that raises is a recorded `could_not_run`, never an exception.
        broken = await run(Unreachable([]), ledger)
        assert broken.verdict.decision == "could_not_run" and broken.applied == ""
        assert broken.verdict.reason == "evidence: ConnectionError: the outcomes table is unreachable"

        # The ledger answers "what is the last decisive verdict?" without a later failure hiding it.
        decisive = await ledger.latest("threshold", decisions=("pass", "fail"))
        assert decisive.row is not None and decisive.row.verdict == second.verdict
        assert decisive.row.metrics["learn_model"] == {"threshold": 0.6}
        assert len(Path(ledger.path).read_text(encoding="utf-8").splitlines()) == 3, "one row per run"
        for got in (first, second, broken):
            print(f"{got.kind}: {got.verdict.decision:<13} applied={got.applied or '-':<7} {got.verdict.reason}")

    # Labelled examples: one contract over every store.
    store = MemoryExampleStore(
        [Example("llm:summary", f"r{i}", "good" if i % 3 else "bad", "human", by="operator") for i in range(9)]
        + [Example("llm:summary", f"r{i}", "good", "judge", by="9b-judge") for i in range(9)]
    )
    read = await store.examples(["llm:summary", "llm:triage"])
    assert read.ok
    cov = examples.coverage(read.examples, ["llm:summary", "llm:triage"], target=20)
    assert (cov["llm:summary"].have, cov["llm:summary"].gap) == (9, 11)
    assert cov["llm:triage"].have == 0, "a topic with no labels is reported, not dropped"

    judge = examples.agreement(read.examples)["llm:summary"]["judge"]
    assert judge.n == 9 and judge.agree is not None and round(judge.agree, 2) == 0.67  # the judge passes everything

    shots = examples.few_shot(read.examples, 4, topic="llm:summary")
    assert {e.value for e in shots} == {"good", "bad"}, "a judge shown examples sees every class"
    for topic, c in cov.items():
        print(f"{topic}: {c.have} human labels, gap {c.gap}")
    print(f"judge vs human on llm:summary: agree {judge.agree:.2f}, kappa {judge.kappa}")


asyncio.run(main())
