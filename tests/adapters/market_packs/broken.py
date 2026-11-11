"""Packs that fail at each stage after import, one hook implementation per class."""

from __future__ import annotations

from llm_ports.adapters.pluggy import hookimpl
from llm_ports.market import ActionRegistry


class NoHook:
    """Implements nothing."""


class ExtraArgument:
    @hookimpl
    def market_actions(self, registry: ActionRegistry) -> ActionRegistry:
        return registry


class Raises:
    @hookimpl
    def market_actions(self) -> ActionRegistry:
        raise RuntimeError("pack build failed")


class NotARegistry:
    @hookimpl
    def market_actions(self) -> object:
        return {"report.weekly": None}


class Conflicts:
    @hookimpl
    def market_actions(self) -> ActionRegistry:
        pack = ActionRegistry("conflicts")

        async def weekly(_session: object, work_key: str) -> None: ...

        pack.action(feature="report", action="weekly", cadence="1d")(weekly)
        return pack


class Misspelled:
    @hookimpl
    def market_action(self) -> ActionRegistry:  # the spec's hook is market_actions
        return ActionRegistry("misspelled")


no_hook = NoHook()
misspelled = Misspelled()
extra_argument = ExtraArgument()
raises = Raises()
not_a_registry = NotARegistry()
conflicts = Conflicts()


__all__ = ["conflicts", "extra_argument", "misspelled", "no_hook", "not_a_registry", "raises"]
