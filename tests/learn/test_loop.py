"""loop.run: one ledger row per run, every exit included, never raises."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_ports.learn import VERDICT_KEY, Clause, Fit, RowRead, Verdict, Written, record, run
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
    applied: list[str] = field(default_factory=list[str])

    async def evidence(self) -> list[float]:
        if self.fail_at == "evidence":
            raise ConnectionError("db down")
        return self.xs

    def fit(self, evidence: list[float]) -> Fit:
        if self.fail_at == "fit":
            raise ZeroDivisionError("empty")
        mean = sum(evidence) / len(evidence) if evidence else None
        return Fit(model={"mean": mean}, metrics={"n": len(evidence), "learn_model": "shadowed", "verdict": "kept"})

    def clauses(self, fit: Fit, evidence: list[float]) -> Sequence[Clause]:
        if self.fail_at == "clauses":
            raise KeyError("n")
        return [
            g.at_least("n", fit.metrics["n"], self.floor, role="floor"),
            g.at_least("mean", fit.model["mean"], self.bar),
        ]

    async def apply(self, verdict: Verdict, fit: Fit) -> str:
        if self.fail_at == "apply":
            raise PermissionError("read-only replica")
        self.applied.append(verdict.decision)
        return "stored" if verdict.passed else "held"


def _clock() -> Any:
    ticks = iter([10.0, 10.25])
    return lambda: next(ticks)


async def test_a_passing_run_records_verdict_model_and_metrics() -> None:
    ledger = MemoryRunLedger()
    got = await run(Mean([1.0, 2.0, 3.0]), ledger, clock=_clock())
    assert (got.verdict.decision, got.recorded.ok, got.seconds) == ("pass", True, 0.25)
    (row,) = ledger.rows
    assert (row.kind, row.subject, row.params) == ("mean", "SPY", {"window": 3})
    assert row.metrics["learn_model"] == {"mean": 2.0}, "a reserved key replaces the fit's metric of the same name"
    assert row.metrics["verdict"] == "kept", "a consumer's own verdict key is never touched"
    assert "gate" not in row.metrics
    assert row.metrics["n"] == 3 and row.metrics["learn_seconds"] == 0.25
    assert (row.metrics["learn_apply"], got.applied) == ("stored", "stored")
    assert row.verdict == got.verdict and row.decision == "pass"


async def test_an_insufficient_run_is_recorded_too() -> None:
    ledger = MemoryRunLedger()
    loop = Mean([5.0])
    got = await run(loop, ledger)
    assert (got.verdict.decision, got.verdict.blocker) == ("insufficient", "n")
    assert len(ledger.rows) == 1
    assert (loop.applied, got.applied) == (["insufficient"], "held"), "apply sees every decided verdict"


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
        assert got.applied == "", "nothing is applied without a decision"
        (row,) = ledger.rows
        assert row.decision == "could_not_run"


async def test_an_apply_that_raises_keeps_the_verdict() -> None:
    ledger = MemoryRunLedger()
    got = await run(Mean([1.0, 2.0, 3.0], fail_at="apply"), ledger)
    assert (got.verdict.decision, got.applied) == ("pass", "apply_failed")
    (row,) = ledger.rows
    assert row.decision == "pass" and row.metrics["learn_apply"] == "apply_failed"
    assert row.metrics["learn_apply_reason"].startswith("apply: PermissionError: read-only replica")


async def test_a_loop_may_choose_the_unevaluated_policy() -> None:
    @dataclass
    class Unknown(Mean):
        unevaluated: str = "insufficient"

    got = await run(Unknown([]), MemoryRunLedger())
    assert (got.verdict.decision, got.verdict.blocker) == ("insufficient", "n")
    unknown_mean = await run(Unknown([], floor=0), MemoryRunLedger())
    assert (unknown_mean.verdict.decision, unknown_mean.verdict.blocker) == ("insufficient", "mean")
    strict = await run(Mean([], floor=0), MemoryRunLedger())
    assert strict.verdict.decision == "fail"


async def test_record_writes_a_verdict_row_outside_a_loop() -> None:
    ledger = MemoryRunLedger()
    v = g.decide("card-system", [g.check("sealed_gate_passed", True)], subject="SPY")
    got = await record(ledger, "card_system", v, metrics={"n": 4})
    (row,) = ledger.rows
    assert got.ok and row.subject == "SPY" and row.metrics[VERDICT_KEY] == v.as_dict() and row.verdict == v
    assert (await record(_Raising(), "k", v)).outcome == "unavailable"


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


class _Wrapped:
    """Delegates to a `Mean`; subclasses replace one attribute with a property that raises."""

    def __init__(self, inner: Mean) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _BadParams(_Wrapped):
    @property
    def params(self) -> Mapping[str, Any]:
        raise RuntimeError("config gone")


class _BadKind(_Wrapped):
    @property
    def kind(self) -> str:
        raise RuntimeError("no name")


class _BadFit(Mean):
    def fit(self, evidence: list[float]) -> Fit:
        return Fit(model={"mean": 1.0}, metrics=[1, 2, 3])  # type: ignore[arg-type]  # a malformed fit, by design


async def test_a_loop_attribute_that_raises_is_a_recorded_could_not_run() -> None:
    ledger = MemoryRunLedger()
    got = await run(_BadParams(Mean([1.0, 2.0, 3.0])), ledger)
    assert got.verdict.decision == "could_not_run"
    assert got.verdict.reason.startswith("loop: RuntimeError: config gone")
    assert (got.kind, got.subject, got.applied) == ("mean", "SPY", "")
    (row,) = ledger.rows
    assert row.decision == "could_not_run"


async def test_a_loop_whose_kind_cannot_be_read_is_recorded_under_its_class() -> None:
    ledger = MemoryRunLedger()
    got = await run(_BadKind(Mean([1.0, 2.0, 3.0])), ledger)
    assert (got.kind, got.verdict.decision) == ("_BadKind", "could_not_run")
    assert len(ledger.rows) == 1


async def test_a_fit_with_non_mapping_metrics_is_a_recorded_could_not_run() -> None:
    ledger = MemoryRunLedger()
    got = await run(_BadFit([1.0, 2.0, 3.0]), ledger)
    assert got.verdict.decision == "could_not_run"
    assert got.verdict.reason.startswith("fit: ")
    assert got.fit is None and got.applied == ""
    (row,) = ledger.rows
    assert row.metrics["learn_model"] == {}


async def test_an_unknown_unevaluated_policy_is_the_loops_error_not_a_silent_fail() -> None:
    @dataclass
    class Misspelt(Mean):
        unevaluated: str = "insuficient"

    got = await run(Misspelt([1.0, 2.0, 3.0]), MemoryRunLedger())
    assert got.verdict.decision == "could_not_run"
    assert "unevaluated must be" in got.verdict.reason
