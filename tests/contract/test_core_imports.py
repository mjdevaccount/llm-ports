"""The core needs no framework: `pip install llm-ports` imports with pydantic, jinja2 and pyyaml.

Every module outside `adapters` (and the `testing.langfuse` rig) is imported in a fresh
interpreter, which then reports every vendor module it loaded; there must be none. The `core-only`
CI job runs this file in an environment with no extra installed, and again with only `http`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from llm_ports import _extras

VENDORS = (
    "httpx",
    "httpx2",
    "langfuse",
    "lettucedetect",
    "numpy",
    "openai",
    "opentelemetry",
    "pandas",
    "pydantic_ai",
    "pydantic_evals",
    "river",
    "scipy",
    "scoringrules",
    "sklearn",
    "sqlglot",
    "torch",
    "transformers",
    "venn_abers",
)
#: Modules that need an extra, by path prefix.
NEEDS_AN_EXTRA = ("llm_ports.adapters", "llm_ports.testing.langfuse")
#: Each adapter, the extra that installs it, and the vendor module its import needs first.
ADAPTERS = {
    "llm_ports.adapters.http": ("http", "httpx"),
    "llm_ports.adapters.langfuse": ("langfuse", "langfuse"),
    "llm_ports.adapters.otel": ("otel", "opentelemetry"),
    "llm_ports.adapters.pydantic_ai": ("pydantic-ai", "pydantic_ai"),
    "llm_ports.adapters.pydantic_evals": ("evals", "pydantic_evals"),
    "llm_ports.adapters.river": ("river", "river"),
    "llm_ports.adapters.scoringrules": ("scoringrules", "scoringrules"),
    "llm_ports.adapters.sklearn": ("sklearn", "sklearn"),
    "llm_ports.adapters.sqlglot": ("sql", "sqlglot"),
    "llm_ports.adapters.venn_abers": ("venn-abers", "numpy"),
    "llm_ports.testing.langfuse": ("langfuse", "httpx"),  # needs httpx at import, the SDK when built
}


def _package_dir() -> Path:
    spec = importlib.util.find_spec("llm_ports")
    assert spec is not None and spec.submodule_search_locations
    return Path(next(iter(spec.submodule_search_locations)))


def _core_modules() -> list[str]:
    """Every module of the package that needs no extra, read from the files, so nothing is imported."""
    root = _package_dir()
    names = []
    for path in sorted(root.rglob("*.py")):
        parts = ("llm_ports", *path.relative_to(root).with_suffix("").parts)
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        if not name.startswith(NEEDS_AN_EXTRA):
            names.append(name)
    return names


CORE = _core_modules()


def _loaded_vendors(*modules: str) -> list[str]:
    """Import `modules` in a fresh interpreter and return the vendor modules it loaded."""
    script = (
        "import importlib, json, sys\n"
        f"for m in {list(modules)!r}: importlib.import_module(m)\n"
        f"print(json.dumps(sorted({{n.split('.')[0] for n in sys.modules}} & set({list(VENDORS)!r}))))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def test_the_walk_sees_the_core_layers_and_no_adapter() -> None:
    top = {m.split(".")[1] for m in CORE if "." in m}
    assert top >= {
        "call",
        "evals",
        "forecast",
        "grounding",
        "learn",
        "observability",
        "ports",
        "prompts",
        "provenance",
        "rag",
        "testing",
    }
    assert not [m for m in CORE if m.startswith(NEEDS_AN_EXTRA)]


def test_the_core_loads_no_vendor_module() -> None:
    assert _loaded_vendors(*CORE) == []


@pytest.mark.skipif(not _installed("httpx"), reason="the `http` extra is not installed")
def test_the_http_adapters_load_httpx_and_nothing_else() -> None:
    assert _loaded_vendors("llm_ports.adapters.http") == ["httpx"]


@pytest.mark.parametrize("module", sorted(ADAPTERS))
def test_an_adapter_without_its_extra_names_the_install_line(module: str) -> None:
    extra, vendor = ADAPTERS[module]
    if _installed(vendor):
        pytest.skip(f"the `{extra}` extra is installed")
    out = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False)
    assert out.returncode != 0
    assert f"pip install 'llm-ports[{extra}]'" in out.stderr


def test_extra_rewrites_a_missing_module_and_keeps_its_name_and_cause() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"^nope_x is not installed; .*llm-ports\[sql\]") as info:
        with _extras.extra("sql"):
            import nope_x  # type: ignore[import-not-found]  # noqa: F401
    assert info.value.name == "nope_x"
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_extra_lets_any_other_error_through_unchanged() -> None:
    with pytest.raises(ValueError, match=r"^boom$"):
        with _extras.extra("sql"):
            raise ValueError("boom")
