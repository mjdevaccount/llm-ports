"""The conformance suites pass every in-repo backend and catch a backend that breaks a contract."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from llm_ports.learn import Decision, ExamplesRead, RowRead, Source, VersionRead, Written
from llm_ports.learn.registry import JsonlModelRegistry, MemoryModelRegistry
from llm_ports.learn.stores import JsonlExampleStore, JsonlRunLedger, MemoryExampleStore, MemoryRunLedger, union
from llm_ports.testing.conformance import (
    ContractViolation,
    check_example_store,
    check_model_registry,
    check_run_ledger,
    check_unavailable_model_registry,
    check_unavailable_run_ledger,
    check_unreadable_example_store,
)


async def test_memory_and_jsonl_ledgers_pass(tmp_path: Path) -> None:
    for ledger in (MemoryRunLedger(), JsonlRunLedger(tmp_path / "runs.jsonl")):
        assert len(await check_run_ledger(ledger)) == 9


async def test_a_ledger_passes_twice_on_the_same_store(tmp_path: Path) -> None:
    ledger = JsonlRunLedger(tmp_path / "runs.jsonl")
    assert await check_run_ledger(ledger) == await check_run_ledger(ledger)


async def test_an_unwritable_jsonl_ledger_reports_it(tmp_path: Path) -> None:
    assert await check_unavailable_run_ledger(JsonlRunLedger(tmp_path)) == (
        "write_fails_loud",
        "unreadable_is_not_empty",
    )


async def test_memory_and_jsonl_example_stores_pass(tmp_path: Path) -> None:
    for store in (MemoryExampleStore(), JsonlExampleStore(tmp_path / "ex.jsonl")):
        assert len(await check_example_store(store)) == 6


async def test_unreadable_example_stores_and_their_union_report_it(tmp_path: Path) -> None:
    for store in (JsonlExampleStore(tmp_path), union(MemoryExampleStore(), JsonlExampleStore(tmp_path))):
        assert await check_unreadable_example_store(store) == ("unreadable_is_not_empty",)


async def test_memory_and_jsonl_registries_pass(tmp_path: Path) -> None:
    for registry in (MemoryModelRegistry(), JsonlModelRegistry(tmp_path / "models.jsonl")):
        assert len(await check_model_registry(registry)) == 8


async def test_an_unreadable_jsonl_registry_reports_it(tmp_path: Path) -> None:
    assert len(await check_unavailable_model_registry(JsonlModelRegistry(tmp_path))) == 2


# --- the other direction: a backend that breaks a contract fails its suite --------------------


class _IgnoresDecisions(MemoryRunLedger):
    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        return await super().latest(kind, subject=subject)


class _EmptyWhenBroken:
    async def append(self, kind: str, *, subject: str, params: Any, metrics: Any) -> Written:
        return Written(0, "unavailable", "down")

    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        return RowRead(None, "ok")  # the bug: an unreadable ledger read as an empty one


class _Raises(MemoryRunLedger):
    async def append(self, kind: str, *, subject: str, params: Any, metrics: Any) -> Written:
        raise ConnectionError("socket closed")


class _IgnoresSince(MemoryExampleStore):
    async def examples(
        self, topics: Sequence[str], *, since: datetime | None = None, source: Source | None = None
    ) -> ExamplesRead:
        return await super().examples(topics, source=source)


class _AliasAnything(MemoryModelRegistry):
    async def set_alias(self, name: str, alias: str, version: int) -> Written:
        got = await super().set_alias(name, alias, version)
        return Written(1, "ok") if got.outcome == "request_fault" else got


class _UnreadableIsEmpty(MemoryModelRegistry):
    async def register(self, name: str, **_: Any) -> Written:
        return Written(0, "unavailable", "down")

    async def by_alias(self, name: str, alias: str) -> VersionRead:
        return VersionRead(None, "unavailable", "down")  # this read is right, so `get` is reached

    async def get(self, name: str, version: int) -> VersionRead:
        return VersionRead(None, "ok")


class _OldestWins(MemoryRunLedger):
    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        newest = await super().latest(kind, subject=subject, decisions=decisions)
        hits = [r for r in self.rows if r.kind == kind and subject in (None, r.subject)]
        return RowRead(hits[0], "ok") if newest.row is not None and decisions is None else newest


class _IgnoresSubject(MemoryRunLedger):
    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        return await super().latest(kind, decisions=decisions)


class _KindsLeak(MemoryRunLedger):
    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        got = await super().latest(kind, subject=subject, decisions=decisions)
        return RowRead(self.rows[-1], "ok") if got.row is None and decisions is None and self.rows else got


class _Forgets(MemoryRunLedger):
    async def latest(
        self, kind: str, *, subject: str | None = None, decisions: Sequence[Decision] | None = None
    ) -> RowRead:
        return RowRead(None, "ok")


class _WrongRef(MemoryRunLedger):
    async def append(self, kind: str, *, subject: str, params: Any, metrics: Any) -> Written:
        await super().append(kind, subject=subject, params=params, metrics=metrics)
        return Written(1, "ok", ref="not-the-row")


class _NamesShareNumbers(MemoryModelRegistry):
    total = 0

    async def register(self, name: str, **kwargs: Any) -> Written:
        got = await super().register(name, **kwargs)
        if got.ok:
            self.total += 1
        return Written(1, "ok", ref=str(self.total)) if got.ok else got


class _CountsFrom0(MemoryModelRegistry):
    async def register(self, name: str, **kwargs: Any) -> Written:
        got = await super().register(name, **kwargs)
        return Written(1, "ok", ref=str(int(str(got.ref)) - 1)) if got.ok else got


@pytest.mark.parametrize(
    ("ledger", "invariant"),
    [
        (_Forgets, "round_trip"),
        (_OldestWins, "newest_wins"),
        (_IgnoresSubject, "subject_filter"),
        (_KindsLeak, "kinds_isolated"),
        (_WrongRef, "ref_links_row"),
    ],
)
async def test_a_ledger_breaking_one_invariant_fails_on_it(ledger: type[MemoryRunLedger], invariant: str) -> None:
    with pytest.raises(ContractViolation) as info:
        await check_run_ledger(ledger())
    assert info.value.invariant == invariant


@pytest.mark.parametrize(
    ("registry", "invariant"),
    [(_NamesShareNumbers, "names_isolated"), (_CountsFrom0, "versions_number_from_1")],
)
async def test_a_registry_breaking_one_invariant_fails_on_it(
    registry: type[MemoryModelRegistry], invariant: str
) -> None:
    with pytest.raises(ContractViolation) as info:
        await check_model_registry(registry())
    assert info.value.invariant == invariant


async def test_a_ledger_ignoring_the_decision_filter_fails() -> None:
    with pytest.raises(ContractViolation) as info:
        await check_run_ledger(_IgnoresDecisions())
    assert info.value.invariant == "decision_filter"


async def test_a_ledger_reading_empty_when_broken_fails() -> None:
    with pytest.raises(ContractViolation, match="unreadable_is_not_empty"):
        await check_unavailable_run_ledger(_EmptyWhenBroken())


async def test_a_ledger_that_raises_fails_with_the_cause() -> None:
    with pytest.raises(ContractViolation, match="append raised ConnectionError") as info:
        await check_run_ledger(_Raises())
    assert isinstance(info.value.__cause__, ConnectionError)


async def test_a_ledger_claiming_a_broken_write_fails() -> None:
    with pytest.raises(ContractViolation, match="write_fails_loud"):
        await check_unavailable_run_ledger(MemoryRunLedger())


async def test_a_store_ignoring_since_fails() -> None:
    with pytest.raises(ContractViolation, match="since_filter"):
        await check_example_store(_IgnoresSince())


async def test_a_working_store_is_not_unreadable() -> None:
    with pytest.raises(ContractViolation, match="unreadable_is_not_empty"):
        await check_unreadable_example_store(MemoryExampleStore())


async def test_a_registry_aliasing_a_missing_version_fails() -> None:
    with pytest.raises(ContractViolation, match="alias_needs_a_version"):
        await check_model_registry(_AliasAnything())


async def test_a_registry_reading_empty_when_broken_fails() -> None:
    with pytest.raises(ContractViolation, match="unreadable_is_not_empty: get:"):
        await check_unavailable_model_registry(_UnreadableIsEmpty())
