"""A read-only, hashable mapping, so a frozen dataclass that holds a declared table stays hashable.

`types.MappingProxyType` is read-only but unhashable, and a third-party frozendict would be a
fourth core dependency; Python's own `frozendict` arrives in 3.15, above this package's floor.

Invariants:
    - `freeze` turns every nested dict into a `FrozenMap`, every list or tuple into a tuple and
      every set into a frozenset, so the whole value hashes when its leaves do.
    - A `FrozenMap` equals any mapping with the same items, a plain dict included.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Set
from typing import Any


class FrozenMap(Mapping[str, Any]):
    """A read-only mapping that hashes by its items."""

    __slots__ = ("_items",)

    def __init__(self, items: Mapping[str, Any]) -> None:
        self._items: dict[str, Any] = dict(items)

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(frozenset(self._items.items()))

    def __eq__(self, other: object) -> bool:
        return dict(self._items) == dict(other) if isinstance(other, Mapping) else NotImplemented

    def __repr__(self) -> str:
        return f"FrozenMap({self._items!r})"


def freeze(value: Any) -> Any:
    """Return `value` with every container replaced by its read-only, hashable form. Pure."""
    if isinstance(value, Mapping):
        return FrozenMap({str(k): freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze(v) for v in value)
    if isinstance(value, Set):
        return frozenset(freeze(v) for v in value)
    return value


__all__: list[str] = []  # a helper with no surface: `Tier.policy` is typed `object`
