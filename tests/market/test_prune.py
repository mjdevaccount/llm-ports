"""`prune_mirror`: a retired action's enabled row is retired; a live row and a foreign one are not."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from llm_ports.market import ActionRegistry, ActionSpec, MirrorRow, PrunableMirror, prune_mirror, sync_all
from tests.market._events import CACHE, Events


class Mirror:
    """A mirror over a dict: `retire_action` deletes the row."""

    def __init__(self, rows: Sequence[MirrorRow], *, fail_read: bool = False, fail_on: str | None = None) -> None:
        self.rows = {r.name: r for r in rows}
        self.retired: list[str] = []
        self.fail_read = fail_read
        self.fail_on = fail_on

    async def sync_action(self, spec: ActionSpec) -> None:
        self.rows.setdefault(spec.name, MirrorRow(spec.name, enabled=True))

    async def mirrored(self) -> Sequence[MirrorRow]:
        if self.fail_read:
            raise ConnectionError("store down")
        return list(self.rows.values())

    async def retire_action(self, name: str) -> None:
        if name == self.fail_on:
            raise PermissionError("no grant")
        self.retired.append(name)
        del self.rows[name]


async def _h(_session: object, key: str) -> None: ...


def _registry(*actions: str) -> ActionRegistry:
    reg = ActionRegistry("main")
    for action in actions:
        reg.action(feature="f", action=action, desired_ttl="6h", **CACHE, model="m")(_h)
    return reg


def test_the_test_mirror_is_a_prunable_mirror() -> None:
    mirror: PrunableMirror = Mirror([])  # mypy checks the structure
    assert callable(mirror.retire_action)


async def test_a_retired_action_is_pruned_and_a_live_one_is_untouched() -> None:
    mirror = Mirror([MirrorRow("f.live", True), MirrorRow("f.gone", True), MirrorRow("f.off", True)])
    events = Events()
    report = await prune_mirror(_registry("live", "off"), mirror, on_event=events)
    assert report.pruned == ("f.gone",) and report.count == 1 and report.held == () and report.error is None
    assert set(mirror.rows) == {"f.live", "f.off"} and mirror.retired == ["f.gone"]
    assert events.seen == [
        (logging.INFO, "market_action_pruned", {"registry": "main", "action": "f.gone", "origin": None})
    ]


async def test_a_disabled_unregistered_row_is_left_alone() -> None:
    mirror = Mirror([MirrorRow("f.gone", False)])
    report = await prune_mirror(_registry("live"), mirror, on_event=Events())
    assert report.count == 0 and mirror.retired == []


async def test_a_registered_row_is_never_retired_whatever_its_switch() -> None:
    rows = [MirrorRow("f.a", True, origin="pack1"), MirrorRow("f.b", False, origin="pack1")]
    mirror = Mirror(rows)
    report = await prune_mirror(_registry("a", "b"), mirror, unloaded=["pack1"], on_event=Events())
    assert report.count == 0 and report.held == () and mirror.retired == []


async def test_a_pack_that_failed_to_load_keeps_its_rows() -> None:
    """The direction that matters: a daemon missing one pack must not wipe that pack's rows."""
    mirror = Mirror([MirrorRow("p.x", True, origin="broken"), MirrorRow("f.gone", True, origin="main")])
    events = Events()
    report = await prune_mirror(_registry("live"), mirror, unloaded=["broken"], on_event=events)
    assert report.held == ("p.x",) and report.pruned == ("f.gone",)
    assert "p.x" in mirror.rows and mirror.retired == ["f.gone"]
    assert events.seen[-1] == (
        logging.WARNING,
        "market_prune_held",
        {"registry": "main", "actions": ("p.x",), "unloaded": ("broken",)},
    )


async def test_with_no_origin_on_record_any_failed_pack_holds_every_unregistered_row() -> None:
    mirror = Mirror([MirrorRow("p.x", True), MirrorRow("f.gone", True)])
    report = await prune_mirror(_registry("live"), mirror, unloaded=["broken"], on_event=Events())
    assert report.held == ("p.x", "f.gone") and report.count == 0 and mirror.retired == []


async def test_a_failed_read_is_reported_never_raised() -> None:
    events = Events()
    report = await prune_mirror(_registry("live"), Mirror([], fail_read=True), on_event=events)
    assert report.error == "ConnectionError: store down" and report.count == 0
    assert events.seen == [
        (
            logging.WARNING,
            "market_prune_failed",
            {"registry": "main", "action": None, "reason": "ConnectionError: store down"},
        )
    ]


async def test_a_failed_retire_is_reported_and_the_rest_still_prune() -> None:
    mirror = Mirror([MirrorRow("f.a", True), MirrorRow("f.b", True)], fail_on="f.a")
    events = Events()
    report = await prune_mirror(_registry("live"), mirror, on_event=events)
    assert report.failed == (("f.a", "PermissionError: no grant"),) and report.pruned == ("f.b",)
    assert [e[1] for e in events.seen] == ["market_prune_failed", "market_action_pruned"]


async def test_without_a_hook_the_prune_logs_each_row(caplog: Any) -> None:
    caplog.set_level(logging.INFO, logger="llm_ports.market")
    await prune_mirror(_registry("live"), Mirror([MirrorRow("f.gone", True)]))
    assert "market_action_pruned" in caplog.text and "f.gone" in caplog.text


async def test_sync_then_prune_leaves_exactly_the_declared_actions() -> None:
    mirror = Mirror([MirrorRow("f.gone", True)])
    reg = _registry("a", "b")
    await sync_all(reg, mirror)
    await prune_mirror(reg, mirror, on_event=Events())
    assert set(mirror.rows) == {"f.a", "f.b"}
