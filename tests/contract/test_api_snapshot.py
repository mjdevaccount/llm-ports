"""The public surface, snapshotted: a change to any public module's `__all__` is a reviewed diff.

1.x is additive, so a name may appear here but never disappear. Regenerate deliberately, after
checking the diff is the change you meant:

    uv run python -m tests.contract.test_api_snapshot
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from tests.contract.test_public_api import MODULES

SNAPSHOT = Path(__file__).with_name("api_surface.json")


def _public(name: str) -> bool:
    return not any(part.startswith("_") for part in name.split("."))


def surface() -> dict[str, list[str]]:
    """Every public module's `__all__`, sorted."""
    return {m: sorted(importlib.import_module(m).__all__) for m in MODULES if _public(m)}


def removed(snap: dict[str, list[str]], now: dict[str, list[str]]) -> dict[str, list[str]]:
    """Every snapshot name the live surface no longer has, by module; a deleted module loses all."""
    gone = {m: sorted(set(names) - set(now.get(m, []))) for m, names in snap.items()}
    return {m: n for m, n in gone.items() if n}


def test_the_public_surface_matches_the_snapshot() -> None:
    snap: dict[str, list[str]] = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    now = surface()
    assert not removed(snap, now), f"1.x is additive; these public names disappeared: {removed(snap, now)}"
    assert now == snap, "the public surface changed: regenerate the snapshot deliberately (see the module docstring)"


def test_a_deleted_module_and_a_dropped_name_are_each_removed() -> None:
    now = surface()
    snap = {k: list(v) for k, v in now.items()}  # a snapshot taken of the live tree
    assert removed(snap, now) == {}
    without_ports = {k: v for k, v in now.items() if k != "llm_ports.ports"}
    assert removed(snap, without_ports) == {"llm_ports.ports": sorted(now["llm_ports.ports"])}
    dropped = {**now, "llm_ports.ports": now["llm_ports.ports"][1:]}
    assert removed(snap, dropped) == {"llm_ports.ports": [now["llm_ports.ports"][0]]}
    assert removed(snap, {**now, "llm_ports.brand_new": ["X"]}) == {}  # an addition is not a removal


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(surface(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
