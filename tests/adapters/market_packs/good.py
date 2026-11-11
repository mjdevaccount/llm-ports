"""A well-formed pack: two actions and the hook that returns them."""

from __future__ import annotations

from llm_ports.adapters.pluggy import hookimpl
from llm_ports.market import ActionRegistry

PACK = ActionRegistry("reports")


@PACK.action(feature="report", action="weekly", cadence="7d", lane=3)
async def weekly(_session: object, work_key: str) -> dict[str, str]:
    return {"key": work_key}


@PACK.action(feature="report", action="digest", max_silence_h=48)
async def digest(_session: object, work_key: str) -> dict[str, str]:
    return {"key": work_key}


@hookimpl
def market_actions() -> ActionRegistry:
    return PACK
