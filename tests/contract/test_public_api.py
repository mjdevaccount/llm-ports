"""The public surface: every module declares `__all__`, and every name in it resolves.

A leading underscore is the only privacy marker, so `__all__` is the semver-covered surface. A
name listed but not bound is an `ImportError` waiting for the first `from x import *`; a module
without `__all__` has no declared surface at all.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pkgutil
from pathlib import Path
from types import ModuleType

import pytest

import llm_ports


def _modules() -> list[str]:
    names = [llm_ports.__name__]
    names += [m.name for m in pkgutil.walk_packages(llm_ports.__path__, prefix=f"{llm_ports.__name__}.")]
    return sorted(names)


MODULES = _modules()


def _import(name: str) -> ModuleType:
    return importlib.import_module(name)


def test_the_walk_sees_every_layer() -> None:
    top = {m.split(".")[1] for m in MODULES if m.count(".") >= 1}
    assert top >= {"adapters", "call", "evals", "grounding", "observability", "ports", "prompts", "provenance", "rag"}
    assert "testing" in top


@pytest.mark.parametrize("name", MODULES)
def test_every_module_declares_its_surface(name: str) -> None:
    module = _import(name)
    assert isinstance(getattr(module, "__all__", None), list), f"{name} has no __all__ list"


@pytest.mark.parametrize("name", MODULES)
def test_every_exported_name_resolves_once(name: str) -> None:
    exported = _import(name).__all__
    assert len(exported) == len(set(exported)), f"{name}.__all__ lists a name twice"
    missing = [n for n in exported if not hasattr(_import(name), n)]
    assert not missing, f"{name}.__all__ names what it does not bind: {missing}"


def _is_public(name: str) -> bool:
    return not any(part.startswith("_") for part in name.split("."))


@pytest.mark.parametrize("name", [m for m in MODULES if m.rsplit(".", 1)[-1].startswith("_") and "__" not in m])
def test_a_private_module_is_reexported_by_its_package(name: str) -> None:
    """A private module's surface is reached through its package, or a public package beside it.

    Shared adapter internals (`adapters._index_order`) are exported by the adapter packages that
    use them; a helper with no surface (`_extras`) declares an empty `__all__`.
    """
    parent = name.rsplit(".", 1)[0]
    homes = [_import(m) for m in MODULES if (m == parent or m.startswith(parent + ".")) and _is_public(m)]
    unexported = [n for n in _import(name).__all__ if not any(n in getattr(h, "__all__", ()) for h in homes)]
    assert not unexported, f"{name} defines {unexported}, which no public module of {parent} exports"


# ── the snapshot: a change to the public surface is a deliberate diff of public_api.json ─────────

SNAPSHOT = Path(__file__).with_name("public_api.json")


def surface() -> dict[str, list[str]]:
    """Every public module's `__all__`, sorted; a module whose extra is absent is skipped."""
    out: dict[str, list[str]] = {}
    for name in MODULES:
        if not _is_public(name):
            continue
        try:
            out[name] = sorted(_import(name).__all__)
        except ModuleNotFoundError:
            continue
    return out


def surface_drift(want: dict[str, list[str]], got: dict[str, list[str]]) -> list[str]:
    """Every difference between the snapshot `want` and the live `got`, as one line each.

    A snapshot module that no longer exists is drift; one that exists but whose extra is absent
    (so `surface()` skipped it) is not.
    """
    out: list[str] = []
    for name, names in want.items():
        if name in got:
            if got[name] != names:
                out.append(f"{name}: names changed")
        elif importlib.util.find_spec(name) is None:
            out.append(f"{name}: a public module was removed")
    out += [f"{name}: a public module was added" for name in got if name not in want]
    return out


def test_the_public_surface_matches_the_snapshot() -> None:
    # Regenerate on a deliberate change: `python -m tests.contract.test_public_api`.
    want = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    drift = surface_drift(want, surface())
    assert not drift, f"the public surface changed; regenerate the snapshot on purpose: {drift}"


def test_a_deleted_module_an_added_one_and_a_renamed_name_are_each_drift() -> None:
    got = surface()
    want = {k: list(v) for k, v in got.items()}  # a snapshot taken of the live tree
    want_gone = {**want, "llm_ports.ports_deleted_since": ["X"]}
    assert surface_drift(want_gone, got) == ["llm_ports.ports_deleted_since: a public module was removed"]
    skipped = {k: v for k, v in got.items() if k != "llm_ports.ports"}  # present, its extra absent
    assert surface_drift(want, skipped) == []
    assert surface_drift(want, {**got, "llm_ports.brand_new": []}) == ["llm_ports.brand_new: a public module was added"]
    renamed = {**got, "llm_ports.ports": [*got["llm_ports.ports"][:-1], "renamed"]}
    assert surface_drift(want, renamed) == ["llm_ports.ports: names changed"]
    assert surface_drift(want, got) == []


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(surface(), indent=1, sort_keys=True) + chr(10), encoding="utf-8")
