"""The operator's kill list: `feature.action` names that are paused without a rebuild.

The consumer hands in a reader for the raw setting (an environment variable, a config key) and a
default set. The setting is a comma-separated list of names. Unset (the reader returns None) means
the default; set, even to the empty string, overrides it, so re-enabling a paused action is a
configuration edit, never a code change.

Invariants:
    - The setting is read and parsed once per `KillSwitch` and then cached; `reset` forces a
      re-read.
    - A non-empty paused set is emitted once per parse as `market_actions_disabled`, with
      `source` `configured` or `default`.
    - A paused action stays registered; the consumer's scheduler skips it.
    - Constructed with a `registry`, every parse also checks the names against it (`unknown`): a
      paused name that matches no action is `market_kill_unknown` at WARNING each time the setting
      is read, so a typo cannot quietly pause nothing.

Rationale: ADR-0018.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from llm_ports.market._registry import ActionRegistry
from llm_ports.observability import EventHook, log_events

_log_event = log_events("llm_ports.market")


def parse_names(raw: str) -> frozenset[str]:
    """Split a comma-separated list of `feature.action` names; blanks are dropped.

    Args:
        raw: The setting's value.

    Returns:
        The names, stripped. Never raises.
    """
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


class KillSwitch:
    """The paused set, read once from a consumer's setting.

    Args:
        read: Returns the raw setting, or None when it is unset.
        default: The paused set while the setting is unset.
        on_event: Receives `market_actions_disabled` and `market_kill_unknown`; without one, the
            stdlib logger `llm_ports.market`.
        registry: The actions the names must match; each parse runs `unknown` against it. None
            checks nothing (call `unknown` yourself).
    """

    __slots__ = ("_default", "_emit", "_keys", "_read", "_registry")

    def __init__(
        self,
        read: Callable[[], str | None],
        *,
        default: Iterable[str] = (),
        on_event: EventHook | None = None,
        registry: ActionRegistry | None = None,
    ) -> None:
        self._read = read
        self._registry = registry
        self._default = frozenset(default)
        self._emit = on_event or _log_event
        self._keys: frozenset[str] | None = None

    @property
    def default(self) -> frozenset[str]:
        """The paused set while the setting is unset."""
        return self._default

    def names(self) -> frozenset[str]:
        """The paused `feature.action` names, parsed on first use."""
        if self._keys is None:
            raw = self._read()
            keys = self._default if raw is None else parse_names(raw)
            if keys:
                source = "default" if raw is None else "configured"
                self._emit(logging.INFO, "market_actions_disabled", {"actions": sorted(keys), "source": source})
            self._keys = keys
            if self._registry is not None:
                self.unknown(self._registry)
        return self._keys

    def is_disabled(self, feature: str, action: str) -> bool:
        """Whether `feature.action` is paused."""
        return f"{feature}.{action}" in self.names()

    def unknown(self, registry: ActionRegistry) -> frozenset[str]:
        """Paused names that match no action in `registry`: a typo pauses nothing.

        A non-empty result is emitted as `market_kill_unknown` (WARNING).
        """
        known = {spec.name for spec in registry.values()}
        stray = frozenset(n for n in self.names() if n not in known)
        if stray:
            self._emit(logging.WARNING, "market_kill_unknown", {"registry": registry.name, "names": sorted(stray)})
        return stray

    def reset(self) -> None:
        """Forget the parsed set; the next read parses the setting again (and emits again)."""
        self._keys = None


__all__ = ["KillSwitch", "parse_names"]
