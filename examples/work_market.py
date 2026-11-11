"""The work market's declaration side: actions on a registry, a kill list, a cached gate, a mirror sync.

A scheduler (yours) claims due work when its model server is free. This example builds what it
reads: the registry its producers declared on, the operator's paused set, the per-category gate
`call.generate` admits by, and the store copy an operator tunes.
"""

import asyncio
import os

from llm_ports.market import ActionRegistry, ActionSpec, Gate, GateCache, KillSwitch, sync_all

ACTIONS = ActionRegistry("reports")


async def due_weeks(session: object, ttl_s: float) -> list[str]:
    return ["2026-W45"]


@ACTIONS.action(feature="report", action="weekly", lane=3, cadence="7d", stale_fn=due_weeks, cache_table="report_out")
async def weekly(session: object, work_key: str) -> dict[str, str]:
    return {"week": work_key}


@ACTIONS.action(feature="report", action="digest", lane=2, max_silence_h=48, source="mail")
async def digest(session: object, work_key: str) -> dict[str, str]:
    return {"digest": work_key}


# A producer nothing could judge stale is refused at import, not discovered months later.
try:
    ACTIONS.action(feature="report", action="orphan")(digest)
except ValueError as exc:
    assert "declare desired_ttl, cadence or max_silence_h" in str(exc)

# The setting's NAME is yours; unset means the default, set (even empty) overrides it.
paused = KillSwitch(lambda: os.environ.get("REPORTS_PAUSED"), default={"report.digest"})
assert paused.is_disabled("report", "digest") or "REPORTS_PAUSED" in os.environ
assert paused.unknown(ACTIONS) == frozenset()


class Mirror:
    """Your store copy of the registry (a table, a file): code fields overwrite, tuning survives."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    async def sync_action(self, spec: ActionSpec) -> None:
        row = self.rows.setdefault(spec.name, {"lane": spec.lane, "enabled": True})
        row.update(handler=spec.handler_path, model=spec.model, est_cost_s=spec.est_cost_s)


async def read_gates() -> dict[str, Gate]:
    return {"report.digest": Gate(enabled=False), "report.weekly": Gate(call_cap=100, used=3)}


async def main() -> None:
    mirror = Mirror()
    report = await sync_all(ACTIONS, mirror)
    assert report.synced == ("report.weekly", "report.digest") and report.failed == ()

    gates = GateCache(read_gates, window="daily")
    await gates.refresh()
    assert gates.check("report.digest") == "category_disabled: report.digest is switched off"
    assert gates.check("report.weekly") is None

    # One reconcile pass, the way a scheduler would run it: skip paused actions, ask the rest
    # what is due.
    due = {
        spec.name: await spec.stale_fn(None, 0.0)
        for spec in ACTIONS.registered()
        if spec.stale_fn is not None and not paused.is_disabled(spec.feature, spec.action)
    }
    assert due == {"report.weekly": ["2026-W45"]}
    print(due)


asyncio.run(main())
