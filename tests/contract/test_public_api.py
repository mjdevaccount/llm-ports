"""The public surface: every module declares `__all__`, and every name in it resolves.

A leading underscore is the only privacy marker, so `__all__` is the semver-covered surface. A
name listed but not bound is an `ImportError` waiting for the first `from x import *`; a module
without `__all__` has no declared surface at all.
"""

from __future__ import annotations

import importlib
import pkgutil
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


@pytest.mark.parametrize("name", [m for m in MODULES if m.rsplit(".", 1)[-1].startswith("_") and "__" not in m])
def test_a_private_module_is_reexported_by_its_package(name: str) -> None:
    package = _import(name.rsplit(".", 1)[0])
    unexported = [n for n in _import(name).__all__ if n not in package.__all__]
    assert not unexported, f"{name} defines {unexported}, which its package does not export"
