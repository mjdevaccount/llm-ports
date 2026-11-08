"""The standard-library model registries, the ledger refs and `Fit.artifact_uri` in a run's row."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_ports.learn import Clause, Fit, Verdict, gate, run
from llm_ports.learn.registry import JsonlModelRegistry, MemoryModelRegistry
from llm_ports.learn.stores import JsonlRunLedger, MemoryRunLedger


async def test_refusals_are_request_faults() -> None:
    reg = MemoryModelRegistry()
    assert (await reg.register("", model={})).reason == "empty model name"
    assert (await reg.register("m", model={}, tags={"k": 1})).reason == "tags must map str to str"  # type: ignore[dict-item]  # a wrong tag type is the input under test
    assert (await reg.register("m", model={})).ref == "1"
    assert (await reg.set_alias("m", "", 1)).reason == "empty alias"
    assert (await reg.set_alias("m", "champion", 2)).outcome == "request_fault"


async def test_jsonl_keeps_every_alias_move_and_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "models.jsonl"
    reg = JsonlModelRegistry(path)
    for i in (1, 2):
        await reg.register("m", model={"i": i})
    await reg.set_alias("m", "champion", 1)
    await reg.set_alias("m", "champion", 2)
    with path.open("a", encoding="utf-8") as fh:
        for junk in (
            {"op": "register", "name": "m", "version": 7, "model": {}, "at": "2026-01-01T00:00:00+00:00"},
            {"op": "register", "name": "m", "version": 3, "model": [], "at": "2026-01-01T00:00:00+00:00"},
            {"op": "alias", "name": "m", "alias": "champion", "version": 9},
            {"op": "alias", "name": "m", "alias": 5, "version": 1},
            {"op": "drop", "name": "m", "version": 1},
            {"name": 1},
        ):
            fh.write(json.dumps(junk) + "\n")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [e["version"] for e in events if e.get("op") == "alias" and e.get("alias") == "champion"][:2] == [1, 2]
    champion = (await reg.by_alias("m", "champion")).version
    assert champion is not None and champion.model == {"i": 2} and champion.tags == {}
    assert (await reg.register("m", model={})).ref == "3"  # the out-of-sequence lines never took a number


async def test_jsonl_unwritable_after_a_good_read(tmp_path: Path) -> None:
    path = tmp_path / "models.jsonl"
    reg = JsonlModelRegistry(path)
    await reg.register("m", model={})
    path.chmod(0o444)
    try:
        assert (await reg.register("m", model={})).outcome == "unavailable"
        assert (await reg.set_alias("m", "champion", 1)).outcome == "unavailable"
    finally:
        path.chmod(0o644)
    assert (await reg.set_alias("m", "", 1)).outcome == "request_fault"


async def test_ledgers_mint_a_ref_the_row_carries(tmp_path: Path) -> None:
    for ledger in (MemoryRunLedger(), JsonlRunLedger(tmp_path / "r.jsonl")):
        wrote = await ledger.append("k", subject="", params={}, metrics={})
        row = (await ledger.latest("k")).row
        assert wrote.ref and row is not None and row.ref == wrote.ref


async def test_ref_is_not_part_of_equality() -> None:
    wrote = await MemoryRunLedger().append("k", subject="", params={}, metrics={})
    assert wrote.ref is not None
    assert wrote == type(wrote)(1, "ok")


class _Loop:
    kind, subject, params = "art", "", {}

    def __init__(self, fit: Any) -> None:
        self._fit = fit

    async def evidence(self) -> int:
        return 1

    def fit(self, evidence: int) -> Any:
        return self._fit

    def clauses(self, fit: Fit, evidence: int) -> Sequence[Clause]:
        return [gate.at_least("n", 1, 1)]

    async def apply(self, verdict: Verdict, fit: Fit) -> str:
        return "stored"


class _DuckFit:
    model: Mapping[str, Any] = {}
    metrics: Mapping[str, Any] = {"learn_artifact_uri": "posing"}


async def test_run_records_the_artifact_uri_and_the_row_ref() -> None:
    ledger = MemoryRunLedger()
    got = await run(_Loop(Fit({"w": 1}, {}, artifact_uri="file:///m.onnx")), ledger)
    assert ledger.rows[-1].metrics["learn_artifact_uri"] == "file:///m.onnx"
    assert got.recorded.ref == ledger.rows[-1].ref
    await run(_Loop(_DuckFit()), ledger)
    assert "learn_artifact_uri" not in ledger.rows[-1].metrics
    assert ledger.rows[-1].decision == "pass"


async def test_an_unreadable_jsonl_registry_refuses_to_move_an_alias(tmp_path: Path) -> None:
    got = await JsonlModelRegistry(tmp_path).set_alias("m", "champion", 1)
    assert got.outcome == "unavailable" and got.reason


async def test_a_wrong_typed_artifact_uri_or_run_ref_is_refused_not_raised(tmp_path: Path) -> None:
    for registry in (MemoryModelRegistry(), JsonlModelRegistry(tmp_path / "r.jsonl")):
        assert (await registry.register("m", model={}, artifact_uri=tmp_path)).outcome == "request_fault"  # type: ignore[arg-type]  # a Path is the input under test
        assert (await registry.register("m", model={}, artifact_uri="")).outcome == "request_fault"
        assert (await registry.register("m", model={}, run_ref=7)).outcome == "request_fault"  # type: ignore[arg-type]  # an int is the input under test
