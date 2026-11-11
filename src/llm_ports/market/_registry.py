"""The action registry: the `action` decorator, the duplicate guard, and enumeration in declared order.

A feature declares a background action by decorating its handler with `registry.action(...)`. The
decorator builds an `ActionSpec`, refuses one nothing could judge (`refusal`), and adds it under
`(feature, action)`. A scheduler then enumerates `registered()` and calls `fn` and `stale_fn`.

The registry is a `MutableMapping[(feature, action), ActionSpec]`, so test teardown keeps working:
`reg.pop(key, None)`, `del reg[key]`, `reg.clear()`. A raw item write passes the same guard as
`add`. Packs from other distributions join through `include` (or the pluggy adapter's
`load_packs`, which calls it).

Invariants:
    - A refused declaration raises `ValueError` at decoration (import time) and registers nothing;
      the refusal is emitted as `market_action_refused` first, so a failed boot names it.
    - An equal spec under a known key (the same module imported twice) replaces the held one, so
      the newest callables run; a different spec under a known key raises. A spec is never
      silently swapped for a different one.
    - `include` is all-or-nothing: a pack with one conflicting key adds none of its actions.

Rationale: ADR-0018.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from typing import TypeVar

from llm_ports.market._spec import SILENCE_CLOCKS, ActionSpec, Handler, StaleFn, refusal
from llm_ports.observability import EventHook, log_events

H = TypeVar("H", bound=Handler)
Key = tuple[str, str]

#: Where events go when no `on_event` hook is given.
_log_event = log_events("llm_ports.market")


class ActionRegistry(MutableMapping[Key, ActionSpec]):
    """Insertion-ordered `(feature, action) -> ActionSpec` table with the declare-time guard.

    Args:
        name: Names the registry in errors and events (a pack's name, or the process's).
        clocks: The silence clocks a spec may declare.
        on_event: Receives `market_action_refused`; without one, the stdlib logger
            `llm_ports.market`.
    """

    __slots__ = ("_clocks", "_emit", "_items", "name")

    def __init__(
        self, name: str = "actions", *, clocks: tuple[str, ...] = SILENCE_CLOCKS, on_event: EventHook | None = None
    ) -> None:
        self.name: str = name
        self._clocks = clocks
        self._emit = on_event or _log_event
        self._items: dict[Key, ActionSpec] = {}

    @property
    def clocks(self) -> tuple[str, ...]:
        """The silence clocks a spec may declare."""
        return self._clocks

    # -- the guard -----------------------------------------------------------------------------
    def add(self, spec: ActionSpec) -> ActionSpec:
        """Declare `spec` and return it.

        Raises:
            ValueError: The spec is refused (`refusal`), or a different spec holds its key.
        """
        reason = refusal(spec, self._clocks)
        if reason is not None:
            self._emit(
                logging.WARNING, "market_action_refused", {"registry": self.name, "action": spec.name, "reason": reason}
            )
            raise ValueError(f"market action {spec.name}: {reason}")
        held = self._items.get(spec.key)
        if held is not None and held != spec:
            raise ValueError(f"market action {spec.name} registered twice with a different spec in {self.name!r}")
        self._items[spec.key] = spec
        return spec

    def action(
        self,
        *,
        feature: str,
        action: str,
        model: str | None = None,
        lane: int = 2,
        value_weight: float = 1.0,
        desired_ttl: str | None = None,
        est_cost_s: float = 40.0,
        cadence: str | None = None,
        cache_table: str | None = None,
        age_col: str | None = None,
        stale_fn: StaleFn | None = None,
        source: str | Iterable[str] | None = None,
        max_silence_h: float | None = None,
        silence_clock: str = "wall",
    ) -> Callable[[H], H]:
        """Return a decorator that declares the handler it wraps; the handler is returned unchanged.

        The arguments are `ActionSpec`'s fields; `source` also takes one id as a string.

        Raises:
            ValueError: At decoration, when the spec is refused or a different spec holds the key.
        """
        srcs = (source,) if isinstance(source, str) else tuple(source or ())

        def deco(fn: H) -> H:
            self.add(
                ActionSpec(
                    feature=feature,
                    action=action,
                    handler_path=f"{fn.__module__}.{fn.__qualname__}",
                    fn=fn,
                    model=model,
                    lane=lane,
                    value_weight=value_weight,
                    desired_ttl=desired_ttl,
                    est_cost_s=est_cost_s,
                    cadence=cadence,
                    cache_table=cache_table,
                    age_col=age_col,
                    stale_fn=stale_fn,
                    max_silence_h=max_silence_h,
                    silence_clock=silence_clock,
                    source=srcs,
                )
            )
            return fn

        return deco

    def include(self, other: ActionRegistry) -> tuple[ActionSpec, ...]:
        """Add every spec of `other` (a pack), all or nothing, and return the specs added.

        Raises:
            ValueError: A spec of `other` is refused here (a clock this registry does not accept),
                or a different spec holds one of its keys; nothing is added.
        """
        specs = tuple(other.values())
        for spec in specs:
            reason = refusal(spec, self._clocks)
            held = self._items.get(spec.key)
            if reason is not None or (held is not None and held != spec):
                self._emit(
                    logging.WARNING,
                    "market_action_refused",
                    {"registry": self.name, "action": spec.name, "reason": reason or f"conflicts with {self.name!r}"},
                )
                raise ValueError(
                    f"pack {other.name!r} cannot join {self.name!r}: {spec.name}: {reason or 'a different spec holds the key'}"
                )
        for spec in specs:
            self._items[spec.key] = spec
        return specs

    def registered(self) -> list[ActionSpec]:
        """Every spec, in declaration order."""
        return list(self._items.values())

    def find(self, feature: str, action: str) -> ActionSpec | None:
        """The spec under `(feature, action)`, or None."""
        return self._items.get((feature, action))

    def empty(self) -> ActionRegistry:
        """A fresh registry with the same name, clocks and event hook (for test isolation)."""
        return ActionRegistry(self.name, clocks=self._clocks, on_event=self._emit)

    # -- MutableMapping. A raw write passes the guard; del / pop / clear stay raw for teardown. --
    def __getitem__(self, key: Key) -> ActionSpec:
        return self._items[key]

    def __setitem__(self, key: Key, spec: ActionSpec) -> None:
        if key != spec.key:
            raise ValueError(f"{self.name}: key {key!r} is not the spec's own key {spec.key!r}")
        self.add(spec)

    def __delitem__(self, key: Key) -> None:
        del self._items[key]

    def __iter__(self) -> Iterator[Key]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"ActionRegistry({self.name!r}, n={len(self._items)})"


__all__ = ["ActionRegistry", "Key"]
