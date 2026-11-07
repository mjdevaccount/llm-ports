"""stores: memory and JSON-lines ledgers and example stores, and union."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from llm_ports.learn import Example, ExamplesRead, RunLedger, Source, WritableExampleStore
from llm_ports.learn import gate as g
from llm_ports.learn import stores

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _now() -> datetime:
    return T0


async def _exercise_ledger(ledger: RunLedger) -> None:
    v = g.decide("k", [g.check("ok", True)])
    first = await ledger.append("k", subject="SPY", params={"w": 1}, metrics={"verdict": "pass", "gate": v.as_dict()})
    assert first.ok
    assert (await ledger.append("k", subject="SPY", params={}, metrics={"verdict": "could_not_run"})).ok
    assert (await ledger.append("k", subject="QQQ", params={}, metrics={})).ok
    newest = await ledger.latest("k", subject="SPY")
    assert newest.ok and newest.row is not None and newest.row.metrics["verdict"] == "could_not_run"
    decisive = await ledger.latest("k", subject="SPY", decisions=("pass", "fail"))
    assert decisive.row is not None and decisive.row.params == {"w": 1} and decisive.row.verdict == v
    assert (await ledger.latest("k")).row is not None
    none_yet = await ledger.latest("other")
    assert none_yet.ok and none_yet.row is None


async def test_memory_ledger() -> None:
    await _exercise_ledger(stores.MemoryRunLedger(now=_now))


async def test_jsonl_ledger(tmp_path: Path) -> None:
    ledger = stores.JsonlRunLedger(tmp_path / "sub" / "runs.jsonl", now=_now)
    await _exercise_ledger(ledger)
    assert (await stores.JsonlRunLedger(ledger.path).latest("k")).row is not None


async def test_jsonl_ledger_refuses_what_is_not_json(tmp_path: Path) -> None:
    got = await stores.JsonlRunLedger(tmp_path / "r.jsonl").append("k", subject="", params={"x": object()}, metrics={})
    assert got.outcome == "request_fault" and got.written == 0


async def test_an_unreadable_ledger_is_unavailable_not_empty(tmp_path: Path) -> None:
    (tmp_path / "r.jsonl").write_text("{not json" + chr(10), encoding="utf-8")
    got = await stores.JsonlRunLedger(tmp_path / "r.jsonl").latest("k")
    assert got.outcome == "unavailable" and "not JSON" in (got.reason or "")
    assert (await stores.JsonlRunLedger(tmp_path).latest("k")).outcome == "unavailable", "a directory is unreadable"


async def test_an_unwritable_ledger_is_unavailable(tmp_path: Path) -> None:
    got = await stores.JsonlRunLedger(tmp_path).append("k", subject="", params={}, metrics={})
    assert got.outcome == "unavailable"


EX = [
    Example("a", "1", "good", "human", at=T0, ref={"trace": "t1"}),
    Example("a", "2", "bad", "judge", by="9b"),
    Example("b", "SPY@d", 0.012, "outcome", at=T0),
]


async def _exercise_store(store: WritableExampleStore) -> None:
    assert (await store.put(EX)).written == 3
    got = await store.examples(["a"])
    assert got.ok and {e.subject for e in got.examples} == {"1", "2"}
    assert {e.subject for e in (await store.examples(["a", "b"], source="human")).examples} == {"1"}
    assert {e.subject for e in (await store.examples(["a", "b"], since=T0)).examples} == {"1", "SPY@d"}


async def test_memory_store() -> None:
    await _exercise_store(stores.MemoryExampleStore())


async def test_jsonl_store_round_trips_every_field(tmp_path: Path) -> None:
    store = stores.JsonlExampleStore(tmp_path / "ex.jsonl")
    await _exercise_store(store)
    read = (await stores.JsonlExampleStore(store.path).examples(["a", "b"])).examples
    assert read == tuple(EX)
    assert read[0].ref == {"trace": "t1"}


async def test_an_unreadable_example_store_is_unavailable(tmp_path: Path) -> None:
    assert (await stores.JsonlExampleStore(tmp_path).examples(["a"])).outcome == "unavailable"
    assert (await stores.JsonlExampleStore(tmp_path).put(EX)).outcome == "unavailable"


class _Down:
    async def examples(
        self, topics: Sequence[str], *, since: datetime | None = None, source: Source | None = None
    ) -> ExamplesRead:
        return ExamplesRead((), "unavailable", "down")


async def test_union_reads_every_store_and_fails_whole() -> None:
    both = stores.union(stores.MemoryExampleStore(EX[:1]), stores.MemoryExampleStore(EX[1:]))
    assert len((await both.examples(["a", "b"])).examples) == 3
    partial = await stores.union(stores.MemoryExampleStore(EX), _Down()).examples(["a"])
    assert partial.outcome == "unavailable" and partial.examples == ()
    assert "store 1 (_Down): down" in (partial.reason or "")
