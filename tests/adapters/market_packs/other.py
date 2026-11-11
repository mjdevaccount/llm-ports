"""A second well-formed pack, with keys of its own."""

from __future__ import annotations

from llm_ports.adapters.pluggy import hookimpl
from llm_ports.market import ActionRegistry

PACK = ActionRegistry("audit")


@PACK.action(feature="audit", action="nightly", cadence="1d")
async def nightly(_session: object, work_key: str) -> dict[str, str]:
    return {"key": work_key}


@hookimpl
def market_actions() -> ActionRegistry:
    return PACK
