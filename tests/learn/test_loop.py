"""loop.run: one ledger row per run, every exit included, never raises."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_ports.learn import Clause, Fit, RowRead, Written, run
from llm_ports.learn import gate as g
from llm_ports.learn.stores import MemoryRunLedger


@dataclass
class Mean:
    """A loop that learns the mean of its evidence and passes when n >= floor and mean >= bar."""

    xs: list[float]
    floor: int = 3
    bar: float = 0.0
    fail_at: str = ""
    kind: str = "mean"
    subject: str = "SPY"
    params: Mapping[str, Any] = field(default_factory=lambda: {"window": 3})

    async def evidence(self) -> list[float]:
        if self.fail_at == "evidence":
            raise ConnectionError("db down")
        return self.xs

    def fit(self, evidence: list[float]) -> Fit:
        if self.fail_at == "fit":
            raise ZeroDivisionError("empty")
        mean = sum(evidence) / len(evidence) if evidence else None
        return Fit(model={"mean": mean}, metrics={"n": len(evidence), "verdict": "shadowed"})

    def clauses(self, fit: Fit, evidence: list[float]) -> Sequence[Clause]:
        if self.fail_at == "clauses":
            raise KeyError("n")
        return [
            g.at_least("n", fit.metrics["n"], self.floor, floor=True),
            g.at_least("mean", fit.model["mean"], self.bar),
        ]


def _clock() -> Any:
    ticks = iter([10.0, 10.25])
    return lambda: next(ticks)


async def test_a_passing_run_records_verdict_model_and_metrics() -> None:
    ledger = MemoryRunLedger()
    got = await run(Mean([1.0, 2.0, 3.0]), ledger, clock=_clock())
    assert (got.verdict.decision, got.recorded.ok, got.seconds) == ("pass", True, 0.25)
    (row,) = ledger.rows
    assert (row.kind, row.subject, row.params) == ("mean", "SPY", {"window": 3})
    assert row.metrics["verdict"] == "pass", "a reserved key replaces the fit's metric of the same name"
    assert row.metrics["model"] == {"mean": 2.0} and row.metrics["n"] == 3 and row.metrics["seconds"] == 0.25
    assert row.verdict == got.verdict


async def test_an_insufficient_run_is_recorded_too() -> None:
    ledger = MemoryRunLedger()
    got = await run(Mean([5.0]), ledger)
    assert (got.verdict.decision, got.verdict.blocker) == ("insufficient", "n")
    assert len(ledger.rows) == 1


async def test_a_failed_gate_names_its_blocker() -> None:
    got = await run(Mean([-1.0, -2.0, -3.0]), MemoryRunLedger())
    assert (got.verdict.decision, got.verdict.blocker) == ("fail", "mean")


async def test_a_stage_that_raises_is_a_recorded_could_not_run() -> None:
    for stage, exc in (("evidence", "ConnectionError: db down"), ("fit", "ZeroDivisionError"), ("clauses", "KeyError")):
        ledger = MemoryRunLedger()
        got = await run(Mean([1.0, 2.0, 3.0], fail_at=stage), ledger)
        assert got.verdict.decision == "could_not_run"
        assert got.verdict.reason.startswith(f"{stage}: {exc}")
        assert (got.fit is None) == (stage != "clauses"), "a fit that finished is kept for the record"
        (row,) = ledger.rows
        assert row.metrics["verdict"] == "could_not_run"


class _Raising:
    async def append(
        self, kind: str, *, subject: str, params: Mapping[str, Any], metrics: Mapping[str, Any]
    ) -> Written:
        raise RuntimeError("pool exhausted")

    async def latest(self, kind: str, *, subject: str | None = None, decisions: Sequence[str] | None = None) -> RowRead:
        return RowRead(None, "ok")


class _Refusing(_Raising):
    async def append(
        self, kind: str, *, subject: str, params: Mapping[str, Any], metrics: Mapping[str, Any]
    ) -> Written:
        return Written(0, "unavailable", "db down")


async def test_a_ledger_failure_is_returned_not_raised() -> None:
    got = await run(Mean([1.0, 2.0, 3.0]), _Raising())
    assert got.verdict.passed and got.recorded.outcome == "unavailable"
    assert "RuntimeError: pool exhausted" in (got.recorded.reason or "")
    refused = await run(Mean([1.0, 2.0, 3.0]), _Refusing())
    assert refused.recorded.reason == "db down"
