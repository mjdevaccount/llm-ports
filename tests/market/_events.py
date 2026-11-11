"""A recording `on_event` hook for the market tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class Events:
    """Records every `(level, event, fields)` it receives."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, str, dict[str, Any]]] = []

    def __call__(self, level: int, event: str, fields: Mapping[str, Any]) -> None:
        self.seen.append((level, event, dict(fields)))


#: The output columns a spec with a `desired_ttl` must name.
CACHE: dict[str, str] = {"cache_table": "out.items", "age_col": "made_at"}
