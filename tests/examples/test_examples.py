"""Every example in `examples/` runs to completion, and every Python block in the docs runs (the
README quickstart included).

An example asserts its own claims, so a run that exits 0 is a run whose claims held. The one
example that needs a live endpoint runs only when its environment is set.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
#: Examples that need a live model, and the variables that switch them on.
LIVE = {"openai_compatible.py": ("LLM_PORTS_BASE_URL", "LLM_PORTS_MODEL")}
SCRIPTS = sorted([*EXAMPLES.glob("*.py"), *EXAMPLES.glob("*/main.py")])
MARKDOWN = [ROOT / "README.md", ROOT / "docs"]


def _runnable(example: CodeExample) -> bool:
    """A block marked `{.python test="skip"}` is a fragment; a `--8<--` block includes an example
    file, which `test_an_example_runs` already runs."""
    return example.prefix_settings().get("test") != "skip" and not example.source.lstrip().startswith("--8<--")


BLOCKS = [e for e in find_examples(*MARKDOWN) if _runnable(e)]


def test_the_examples_are_found() -> None:
    names = {p.relative_to(EXAMPLES).as_posix() for p in SCRIPTS}
    assert {"named_outcomes.py", "slots_grounded_prose.py", "prompt_pack/main.py"} <= names


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.relative_to(EXAMPLES).as_posix())
def test_an_example_runs(script: Path) -> None:
    needs = LIVE.get(script.name, ())
    if needs and not all(os.environ.get(v) for v in needs):
        pytest.skip(f"needs {', '.join(needs)}")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip(), "an example prints what it shows"


def test_the_readme_quickstart_is_collected() -> None:
    assert any(e.path.name == "README.md" for e in BLOCKS)


@pytest.mark.parametrize("example", BLOCKS, ids=str)
def test_a_markdown_python_block_runs(example: CodeExample, eval_example: EvalExample) -> None:
    eval_example.run(example)
