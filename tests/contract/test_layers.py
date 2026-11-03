"""The layering contracts in `pyproject.toml` (`[tool.importlinter]`) hold, run on every pytest leg.

Layers bottom-up: ports < grounding | prompts | observability < call < evals | rag | provenance <
adapters | testing. Adapters do not import each other, and only adapters and testing import a
vendor library.
"""

from __future__ import annotations

from pathlib import Path

from importlinter.cli import lint_imports

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_the_import_contracts_hold() -> None:
    assert lint_imports(config_filename=str(PYPROJECT), no_cache=True, no_logo=True) == 0, (
        "see the import-linter report above"
    )
