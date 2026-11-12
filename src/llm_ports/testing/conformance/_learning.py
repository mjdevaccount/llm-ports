"""Conformance suites for the learning-loop ports: `RunLedger`, `ExampleStore`, `ModelRegistry`.

A backend (a database table, a tracker, a feature store) that passes its suite can stand in for
the standard-library implementations without a consumer noticing. Each `check_*` coroutine runs
every invariant against the backend it is given and returns the names of the invariants it
checked; the first violation raises `ContractViolation` naming the invariant. They need no test
framework, so a backend's own test suite (or a smoke check against a live store) calls them as
`await check_run_ledger(ledger)`.

Each run writes under fresh, random kinds, topics and names, so a suite runs against a shared or
long-lived store without reading another run's rows.

The `*_unavailable` / `*_unreadable` checks take a backend whose storage is broken (a closed
connection, an unreachable server) and check that the failure is reported, never raised and
never read as "empty".

Invariants:
    - A backend that raises anything fails the check as a `ContractViolation` (never-raise,
      ADR-0001), with the backend's exception as its cause.

Rationale: ADR-0016.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from llm_ports.learn import (
    VERDICT_KEY,
    Clause,
    Decision,
    Example,
    ExampleStore,
    ModelRegistry,
    RunLedger,
    Verdict,
    VersionRead,
    WritableExampleStore,
)
from llm_ports.testing.conformance._core import call as _call
from llm_ports.testing.conformance._core import expect as _expect
from llm_ports.testing.conformance._core import present as _present
from llm_ports.testing.conformance._core import tag as _tag


def _verdict(decision: Decision) -> dict[str, Any]:
    clauses = () if decision == "could_not_run" else (Clause("n", decision == "pass", 10, 5, ">="),)
    return Verdict("contract", "", decision, clauses, reason=decision).as_dict()


async def check_run_ledger(ledger: RunLedger) -> tuple[str, ...]:
    """Check a working `RunLedger` against the port's contract.

    Returns:
        The invariants checked, in order.

    Raises:
        ContractViolation: The first invariant the ledger breaks.
    """
    kind, other = f"contract-{_tag()}", f"contract-{_tag()}"
    subject = 'it\'s "quoted"'  # a query-backed ledger must not splice a subject into its query
    params: dict[str, Any] = {"window": 20, "name": "w", "nested": {"a": [1, 2.5, None]}}
    metrics: dict[str, Any] = {"n": 12, "err": 0.25, "flag": True, "label": "x", VERDICT_KEY: _verdict("pass")}

    got = await _call("empty_reads_ok_none", "latest", ledger.latest(kind))
    _expect("empty_reads_ok_none", got.ok and got.row is None, f"a kind never written read as {got}")

    wrote = await _call("append_ok", "append", ledger.append(kind, subject=subject, params=params, metrics=metrics))
    _expect("append_ok", wrote.ok and wrote.written == 1, f"append returned {wrote}")
    _expect("append_ok", wrote.ref is None or (isinstance(wrote.ref, str) and wrote.ref != ""), f"ref {wrote.ref!r}")

    got = await _call("round_trip", "latest", ledger.latest(kind, subject=subject))
    row = _present("round_trip", got.row if got.ok else None, f"the row just written read as {got}")
    _expect("round_trip", (row.kind, row.subject) == (kind, subject), f"kind/subject {row.kind!r}/{row.subject!r}")
    _expect("round_trip", dict(row.params) == params, f"params {dict(row.params)!r} != {params!r}")
    _expect("round_trip", dict(row.metrics) == metrics, f"metrics {dict(row.metrics)!r} != {metrics!r}")
    _expect("round_trip", row.at.tzinfo is not None, f"`at` {row.at!r} has no time zone")
    _expect("ref_links_row", wrote.ref is None or row.ref == wrote.ref, f"row ref {row.ref!r} != {wrote.ref!r}")

    for sub, seq, more in ((subject, 2, {VERDICT_KEY: _verdict("could_not_run")}), ("b", 3, {"n": 1})):
        later = await _call(
            "newest_wins", "append", ledger.append(kind, subject=sub, params={"seq": seq}, metrics=more)
        )
        _expect("newest_wins", later.ok, f"append returned {later}")
    newest = await _call("newest_wins", "latest", ledger.latest(kind))
    _expect("newest_wins", newest.row is not None and newest.row.params == {"seq": 3}, f"newest read {newest}")
    mine = await _call("subject_filter", "latest", ledger.latest(kind, subject=subject))
    _expect("subject_filter", mine.row is not None and mine.row.params == {"seq": 2}, f"subject read {mine}")

    decisive = await _call("decision_filter", "latest", ledger.latest(kind, decisions=("pass", "fail")))
    _expect("decision_filter", decisive.row is not None and decisive.row.params == params, f"read {decisive}")
    none = await _call("decision_filter", "latest", ledger.latest(kind, decisions=("insufficient",)))
    _expect("decision_filter", none.ok and none.row is None, f"no insufficient row was written; read {none}")

    apart = await _call("kinds_isolated", "latest", ledger.latest(other))
    _expect("kinds_isolated", apart.ok and apart.row is None, f"another kind read {apart}")

    odd = await _call(
        "non_json_never_raises", "append", ledger.append(other, subject="", params={}, metrics={"x": {1, 2}})
    )
    _expect("non_json_never_raises", odd.outcome in ("ok", "request_fault"), f"a set value answered {odd}")
    return (
        "empty_reads_ok_none",
        "append_ok",
        "round_trip",
        "ref_links_row",
        "newest_wins",
        "subject_filter",
        "decision_filter",
        "kinds_isolated",
        "non_json_never_raises",
    )


async def check_unavailable_run_ledger(ledger: RunLedger) -> tuple[str, ...]:
    """Check that a `RunLedger` whose storage is broken reports it.

    Returns:
        The invariants checked.

    Raises:
        ContractViolation: A write that claims success, or an unreadable ledger read as empty.
    """
    wrote = await _call("write_fails_loud", "append", ledger.append("contract", subject="", params={}, metrics={}))
    _expect("write_fails_loud", wrote.outcome == "unavailable" and wrote.written == 0, f"append returned {wrote}")
    _expect("write_fails_loud", bool(wrote.reason), "an unavailable write names no reason")
    got = await _call("unreadable_is_not_empty", "latest", ledger.latest("contract"))
    _expect("unreadable_is_not_empty", got.outcome == "unavailable" and got.row is None, f"latest returned {got}")
    return ("write_fails_loud", "unreadable_is_not_empty")


def _key(e: Example) -> tuple[str, str, str | float, str, str, datetime | None, str]:
    return (e.topic, e.subject, e.value, e.source, e.by, e.at, e.reason)


async def check_example_store(store: WritableExampleStore) -> tuple[str, ...]:
    """Check a working, writable `ExampleStore` against the port's contract.

    Returns:
        The invariants checked, in order.

    Raises:
        ContractViolation: The first invariant the store breaks.
    """
    topic, other = f"contract:{_tag()}", f"contract:{_tag()}"
    t0 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    rows = [
        Example(topic, "s1", "good", "human", by="p", at=t0, reason="clear"),
        Example(topic, "s2", 0.5, "outcome", at=t0 + timedelta(days=1)),
        Example(topic, "s3", "bad", "model", by="m"),
        Example(other, "s1", "bad", "human", at=t0),
    ]
    empty = await _call("empty_reads_ok", "examples", store.examples([topic]))
    _expect("empty_reads_ok", empty.ok and empty.examples == (), f"a topic never written read as {empty}")

    wrote = await _call("put_counts", "put", store.put(rows))
    _expect("put_counts", wrote.ok and wrote.written == len(rows), f"put of {len(rows)} returned {wrote}")
    none = await _call("put_counts", "put", store.put([]))
    _expect("put_counts", none.ok and none.written == 0, f"put of nothing returned {none}")

    got = await _call("round_trip", "examples", store.examples([topic]))
    _expect("round_trip", got.ok, f"read returned {got}")
    want = sorted((_key(e) for e in rows[:3]), key=lambda k: k[1])
    seen = sorted((_key(e) for e in got.examples), key=lambda k: k[1])
    _expect("round_trip", seen == want, f"read {seen!r}, wrote {want!r}")

    both = await _call("topic_filter", "examples", store.examples([topic, other]))
    _expect("topic_filter", len(both.examples) == len(rows), f"two topics read {len(both.examples)} examples")
    human = await _call("source_filter", "examples", store.examples([topic], source="human"))
    _expect("source_filter", [e.subject for e in human.examples] == ["s1"], f"human read {human.examples!r}")
    later = await _call("since_filter", "examples", store.examples([topic], since=t0 + timedelta(hours=1)))
    _expect("since_filter", [e.subject for e in later.examples] == ["s2"], f"since read {later.examples!r}")
    return ("empty_reads_ok", "put_counts", "round_trip", "topic_filter", "source_filter", "since_filter")


async def check_unreadable_example_store(store: ExampleStore) -> tuple[str, ...]:
    """Check that an `ExampleStore` whose storage is broken never reads as empty.

    Returns:
        The invariants checked.

    Raises:
        ContractViolation: The read raised or claimed success.
    """
    got = await _call("unreadable_is_not_empty", "examples", store.examples(["contract"]))
    _expect("unreadable_is_not_empty", got.outcome == "unavailable" and got.examples == (), f"read returned {got}")
    _expect("unreadable_is_not_empty", bool(got.reason), "an unavailable read names no reason")
    return ("unreadable_is_not_empty",)


async def check_model_registry(registry: ModelRegistry) -> tuple[str, ...]:
    """Check a working `ModelRegistry` against the port's contract.

    Returns:
        The invariants checked, in order.

    Raises:
        ContractViolation: The first invariant the registry breaks.
    """
    name, other = f"contract-{_tag()}", f"contract-{_tag()}"
    model: dict[str, Any] = {"w": [0.5, -1.0], "bias": 0.1, "label": "logit"}

    miss = await _call("missing_reads_ok_none", "by_alias", registry.by_alias(name, "champion"))
    _expect("missing_reads_ok_none", miss.ok and miss.version is None, f"an unknown model read as {miss}")
    miss = await _call("missing_reads_ok_none", "get", registry.get(name, 1))
    _expect("missing_reads_ok_none", miss.ok and miss.version is None, f"an unknown version read as {miss}")

    one = await _call(
        "versions_number_from_1",
        "register",
        registry.register(
            name, model=model, artifact_uri="s3://contract/m/1", run_ref="run-1", tags={"by": "contract"}
        ),
    )
    two = await _call("versions_number_from_1", "register", registry.register(name, model={"w": [1.0]}))
    _expect("versions_number_from_1", (one.ref, two.ref) == ("1", "2") and one.ok and two.ok, f"got {one}, {two}")

    got = await _call("round_trip", "get", registry.get(name, 1))
    mv = _present("round_trip", got.version if got.ok else None, f"version 1 read as {got}")
    _expect("round_trip", (mv.name, mv.version, dict(mv.model)) == (name, 1, model), f"read {mv!r}")
    _expect("round_trip", (mv.artifact_uri, mv.run_ref) == ("s3://contract/m/1", "run-1"), f"read {mv!r}")
    _expect("round_trip", dict(mv.tags).get("by") == "contract" and mv.at.tzinfo is not None, f"read {mv!r}")

    await _call("alias_moves", "set_alias", registry.set_alias(name, "champion", 1))
    first = await _call("alias_moves", "by_alias", registry.by_alias(name, "champion"))
    moved = await _call("alias_moves", "set_alias", registry.set_alias(name, "champion", 2))
    second = await _call("alias_moves", "by_alias", registry.by_alias(name, "champion"))
    ok = first.version is not None and first.version.version == 1 and moved.ok
    _expect("alias_moves", ok and second.version is not None and second.version.version == 2, f"{first}, {second}")
    still = await _call("versions_immutable", "get", registry.get(name, 1))
    _expect("versions_immutable", still.version is not None and dict(still.version.model) == model, f"{still}")

    bad = await _call("alias_needs_a_version", "set_alias", registry.set_alias(name, "champion", 99))
    _expect("alias_needs_a_version", bad.outcome == "request_fault", f"alias to a missing version returned {bad}")
    kept = await _call("alias_needs_a_version", "by_alias", registry.by_alias(name, "champion"))
    _expect("alias_needs_a_version", kept.version is not None and kept.version.version == 2, f"alias moved: {kept}")
    unset = await _call("missing_reads_ok_none", "by_alias", registry.by_alias(name, "challenger"))
    _expect("missing_reads_ok_none", unset.ok and unset.version is None, f"an unset alias read as {unset}")

    for bad_version in ("1", 1.0, True):  # `Written.ref` is "1": passing it back must not raise or match
        typed = await _call("version_is_an_int", "get", registry.get(name, bad_version))  # type: ignore[arg-type]  # a wrong type is the input under test
        _expect("version_is_an_int", typed.outcome == "request_fault", f"get({bad_version!r}) returned {typed}")
        moved = await _call(
            "version_is_an_int",
            "set_alias",
            registry.set_alias(name, "champion", bad_version),  # type: ignore[arg-type]  # a wrong type is the input under test
        )
        _expect("version_is_an_int", moved.outcome == "request_fault", f"set_alias to {bad_version!r}: {moved}")

    fresh = await _call("names_isolated", "register", registry.register(other, model={}))
    _expect("names_isolated", fresh.ok and fresh.ref == "1", f"a new name's first version is {fresh}")
    odd = await _call("non_json_never_raises", "register", registry.register(other, model={"x": {1, 2}}))
    _expect("non_json_never_raises", odd.outcome == "request_fault", f"a set value answered {odd}")
    return (
        "missing_reads_ok_none",
        "versions_number_from_1",
        "round_trip",
        "alias_moves",
        "versions_immutable",
        "alias_needs_a_version",
        "version_is_an_int",
        "names_isolated",
        "non_json_never_raises",
    )


async def check_unavailable_model_registry(registry: ModelRegistry) -> tuple[str, ...]:
    """Check that a `ModelRegistry` whose storage is broken reports it.

    Returns:
        The invariants checked.

    Raises:
        ContractViolation: A write that claims success, or an unreadable registry read as empty.
    """
    wrote = await _call("write_fails_loud", "register", registry.register("contract", model={}))
    _expect("write_fails_loud", wrote.outcome == "unavailable" and wrote.written == 0, f"register returned {wrote}")
    reads: tuple[tuple[str, Callable[[], Awaitable[VersionRead]]], ...] = (
        ("by_alias", lambda: registry.by_alias("contract", "champion")),
        ("get", lambda: registry.get("contract", 1)),
    )
    for what, read in reads:
        got = await _call("unreadable_is_not_empty", what, read())
        _expect("unreadable_is_not_empty", got.outcome == "unavailable" and got.version is None, f"{what}: {got}")
    return ("write_fails_loud", "unreadable_is_not_empty")


__all__ = [
    "check_example_store",
    "check_model_registry",
    "check_run_ledger",
    "check_unavailable_model_registry",
    "check_unavailable_run_ledger",
    "check_unreadable_example_store",
]
