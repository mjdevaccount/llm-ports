"""Action packs from other distributions, found by entry point and loaded with pluggy (extra `[market]`).

A distribution that declares background actions for a scheduler it does not run lists a module
under the `market.ENTRY_POINT_GROUP` entry-point group. The module declares its actions on its own
`ActionRegistry` (the pack) and returns it from a `market_actions` hook:

    PACK = ActionRegistry("reports")

    @PACK.action(feature="report", action="weekly", cadence="7d", lane=3)
    async def weekly(session, work_key): ...

    @hookimpl
    def market_actions() -> ActionRegistry:
        return PACK

    # pyproject.toml
    [project.entry-points."llm_ports.market"]
    reports = "reports_pkg.market"

The scheduler's process calls `load_packs(registry)` once at boot. pluggy validates each module's
hook against the spec when it registers (an implementation with an argument the spec does not
declare is refused), keeps the plugin-to-distribution map, and blocks a pack by name.

Invariants:
    - `load_packs` never raises. Each entry point ends as one `PackLoad`: `ok`, `failed` (with the
      stage that failed) or `blocked`, each emitted as `market_pack_loaded`, `market_pack_failed`
      or `market_pack_blocked`.
    - A pack joins all or nothing (`ActionRegistry.include`): one conflicting or refused action
      adds none of the pack's actions.
    - Whether a failed pack stops the process is the caller's decision; `failed` names the packs
      that did not load.

Rationale: ADR-0018.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from llm_ports._extras import extra
from llm_ports.market import ENTRY_POINT_GROUP, ActionRegistry
from llm_ports.observability import EventHook, log_events

with extra("market"):
    import pluggy

#: pluggy's project name: hook markers match plugins by it.
PROJECT: str = "llm_ports.market"
#: Marks a pack module's `market_actions` implementation.
hookimpl: pluggy.HookimplMarker = pluggy.HookimplMarker(PROJECT)
_hookspec = pluggy.HookspecMarker(PROJECT)
_log_event = log_events("llm_ports.market")

#: How one entry point ended.
PackOutcome = Literal["ok", "failed", "blocked"]
#: Where a failed pack stopped: importing its module, validating its hooks, building its pack,
#: or joining the registry.
PackStage = Literal["import", "validate", "pack", "join"]


class PackSpec:
    """The hooks a pack module may implement."""

    @_hookspec
    def market_actions(self) -> ActionRegistry:  # type: ignore[empty-body]  # a hook spec has no body
        """Return the pack's actions."""


@dataclass(frozen=True)
class PackLoad:
    """One entry point, loaded or not.

    Attributes:
        name: The entry point's name (the pack's name, and its pluggy plugin name).
        dist: The distribution that declares it, or None when unknown.
        outcome: `ok`, `failed` or `blocked`.
        actions: The `feature.action` names the pack added; empty unless `ok`.
        stage: Where a failed pack stopped; None unless `failed`.
        reason: Why it failed; None unless `failed`.
    """

    name: str
    dist: str | None
    outcome: PackOutcome
    actions: tuple[str, ...] = ()
    stage: PackStage | None = None
    reason: str | None = None


def _dist(ep: importlib.metadata.EntryPoint) -> str | None:
    dist = ep.dist
    return dist.name if dist is not None else None


def load_packs(
    registry: ActionRegistry,
    *,
    group: str = ENTRY_POINT_GROUP,
    block: Iterable[str] = (),
    entry_points: Iterable[importlib.metadata.EntryPoint] | None = None,
    on_event: EventHook | None = None,
) -> tuple[PackLoad, ...]:
    """Load every pack listed under `group` into `registry`. Never raises.

    Args:
        registry: The scheduler's registry; each pack joins it through `include`.
        group: The entry-point group to read.
        block: Pack names to skip (pluggy's blocked set).
        entry_points: The entry points to load instead of the installed ones (tests, a fixed list).
        on_event: Receives the `market_pack_*` events; without one, the stdlib logger
            `llm_ports.market`.

    Returns:
        One `PackLoad` per entry point, in the order read.
    """
    emit = on_event or _log_event
    pm = pluggy.PluginManager(PROJECT)
    pm.add_hookspecs(PackSpec)
    for name in block:
        pm.set_blocked(name)
    eps = importlib.metadata.entry_points(group=group) if entry_points is None else entry_points
    loads: list[PackLoad] = []
    for ep in eps:
        load = _load_one(pm, ep, registry)
        loads.append(load)
        if load.outcome == "ok":
            emit(
                logging.INFO,
                "market_pack_loaded",
                {"pack": load.name, "dist": load.dist, "actions": list(load.actions)},
            )
        elif load.outcome == "blocked":
            emit(logging.INFO, "market_pack_blocked", {"pack": load.name, "dist": load.dist})
        else:
            fields = {"pack": load.name, "dist": load.dist, "stage": load.stage, "reason": load.reason}
            emit(logging.WARNING, "market_pack_failed", fields)
    return tuple(loads)


def _load_one(pm: pluggy.PluginManager, ep: importlib.metadata.EntryPoint, registry: ActionRegistry) -> PackLoad:
    name, dist = ep.name, _dist(ep)

    def fail(stage: PackStage, exc: BaseException | str) -> PackLoad:
        reason = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        return PackLoad(name, dist, "failed", stage=stage, reason=reason[:300])

    if pm.is_blocked(name):
        return PackLoad(name, dist, "blocked")
    try:
        module = ep.load()
    except Exception as exc:
        return fail("import", exc)
    try:
        pm.register(module, name=name)
    except Exception as exc:
        return fail("validate", exc)
    impls = [i for i in pm.hook.market_actions.get_hookimpls() if i.plugin_name == name]
    if not impls:
        pm.unregister(name=name)
        return fail("validate", "implements no market_actions hook")
    try:
        pack = impls[0].function()
    except Exception as exc:
        pm.unregister(name=name)
        return fail("pack", exc)
    if not isinstance(pack, ActionRegistry):
        pm.unregister(name=name)
        return fail("pack", f"market_actions returned {type(pack).__name__}, not an ActionRegistry")
    try:
        added = registry.include(pack)
    except ValueError as exc:
        pm.unregister(name=name)
        return fail("join", exc)
    return PackLoad(name, dist, "ok", actions=tuple(spec.name for spec in added))


def failed(loads: Iterable[PackLoad]) -> tuple[PackLoad, ...]:
    """The loads that failed (a blocked pack is a decision, not a failure)."""
    return tuple(load for load in loads if load.outcome == "failed")


__all__ = ["PROJECT", "PackLoad", "PackOutcome", "PackSpec", "PackStage", "failed", "hookimpl", "load_packs"]
