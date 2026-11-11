"""`sync_all`: every spec synced, paused ones included; a failed sync is reported, never raised."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from llm_ports.market import CODE_OWNED, OPERATOR_TUNED, ActionRegistry, ActionSpec, sync_all
from tests.market._events import CACHE


class Mirror:
    def __init__(self, fail_on: str | None = None) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.fail_on = fail_on

    async def sync_action(self, spec: ActionSpec) -> None:
        if spec.name == self.fail_on:
            raise PermissionError("no grant")
        row = self.rows.setdefault(spec.name, {f: getattr(spec, f) for f in OPERATOR_TUNED if f != "enabled"})
        row.update({f: getattr(spec, f) for f in CODE_OWNED})


async def _h(_session: object, key: str) -> None: ...


def _registry() -> ActionRegistry:
    reg = ActionRegistry("main")
    for action in ("a", "b", "c"):
        reg.action(feature="f", action=action, desired_ttl="6h", **CACHE, model="m")(_h)
    return reg


async def test_sync_all_syncs_every_spec_and_never_clobbers_operator_fields() -> None:
    mirror = Mirror()
    mirror.rows["f.a"] = {"lane": 0, "value_weight": 9.0, "desired_ttl": "1h"}
    report = await sync_all(_registry(), mirror)
    assert report.synced == ("f.a", "f.b", "f.c") and report.failed == ()
    assert mirror.rows["f.a"]["lane"] == 0 and mirror.rows["f.a"]["model"] == "m"
    assert mirror.rows["f.b"]["lane"] == 2


async def test_a_failed_sync_is_reported_and_the_rest_still_sync() -> None:
    seen: list[tuple[int, str, dict[str, Any]]] = []

    def on_event(level: int, event: str, fields: Mapping[str, Any]) -> None:
        seen.append((level, event, dict(fields)))

    report = await sync_all(_registry(), Mirror(fail_on="f.b"), on_event=on_event)
    assert report.synced == ("f.a", "f.c")
    assert report.failed == (("f.b", "PermissionError: no grant"),)
    assert seen == [
        (
            logging.WARNING,
            "market_sync_failed",
            {"registry": "main", "action": "f.b", "reason": "PermissionError: no grant"},
        )
    ]
