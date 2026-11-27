"""The published-vocabulary gate fires: the pre-commit `no-market-vocabulary` hook, run through
pre-commit's own pygrep engine, goes red on a planted fixture, stays quiet on near-misses, covers
every published root, excludes only the fixture, and is invoked by CI.

A guard nothing can trip is indistinguishable from one that works, so this test proves both
directions against the hook exactly as `.pre-commit-config.yaml` declares it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from pre_commit.languages import pygrep

ROOT = Path(__file__).parents[2]
PLANTED = "tests/fixtures/vocabulary/planted.txt"
HOOK_ID = "no-market-vocabulary"

#: Words that sit next to a refused term and must still pass: a test double, an English word,
#: a substring (a refused word is bounded by letters, so it may not continue into a lowercase
#: one), a public repository path, a date and a scale letter.
NEAR_MISSES = (
    "class _Spy(WrapperModel): a test spy",
    "an edge case at the Edge of the map",
    "spyware, stickers, a dealership, an update",
    "a Dealership in town, worth it, north of here, a setup_upgrade",
    "the edge_city and home_cinema",
    "alpha, beta and gamma hold",
    "the work market and its kill switch",
    "a trade-off between edge_weights and home pages",
    "narrow it down",
    "https://github.com/mjdevaccount/llm-ports and mjdevaccount/ci-kit",
    "see mjdevaccount/llm-ports.",
    "## [1.32.0] - 2026-11-27",
    "-19.8B of flow",
)


def _hook() -> dict[str, Any]:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = [h for repo in config["repos"] for h in repo["hooks"] if h["id"] == HOOK_ID]
    assert len(hooks) == 1, f"exactly one {HOOK_ID} hook, found {len(hooks)}"
    hook: dict[str, Any] = hooks[0]
    assert hook["language"] == "pygrep" and not hook.get("args"), "the engine below runs it as declared"
    return hook


def _run(path: Path) -> int:
    """Pre-commit's own engine over one file: 1 when any line matched, else 0."""
    return pygrep.main([_hook()["entry"], str(path)])


def test_the_planted_fixture_goes_red_on_every_line(tmp_path: Path) -> None:
    lines = (ROOT / PLANTED).read_bytes().splitlines()
    assert len(lines) == 55, "one line per refused term: add a line when the list grows"
    assert _run(ROOT / PLANTED) == 1, "MUST FIRE: the planted fixture passed the gate"
    # pygrep compiles the entry as bytes with no flags; each line alone must be a finding
    for i, line in enumerate(lines):
        one = tmp_path / f"line{i}.txt"
        one.write_bytes(line + b"\n")
        assert _run(one) == 1, f"planted line {i + 1} passed the gate: {line!r}"


@pytest.mark.parametrize("line", NEAR_MISSES)
def test_near_misses_pass(tmp_path: Path, line: str) -> None:
    clean = tmp_path / "clean.md"
    clean.write_text(line + "\n", encoding="utf-8")
    assert _run(clean) == 0, f"a neutral line was refused: {line!r}"


def test_the_gate_covers_every_published_root_and_excludes_only_the_fixture() -> None:
    hook = _hook()
    files, exclude = re.compile(hook["files"]), re.compile(hook["exclude"])
    for path in (
        "src/llm_ports/x.py",
        "docs/x.md",
        "tests/x.py",
        "examples/x.py",
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "CLAUDE.md",
        "mkdocs.yml",
        "pyproject.toml",
    ):
        assert files.search(path) and not exclude.search(path), path
    assert files.search(PLANTED) and exclude.search(PLANTED)
    try:
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout: the tracked-file check needs `git ls-files`")
    assert PLANTED in tracked, "the fixture is committed, so CI sees what this test sees"
    assert [p for p in tracked if exclude.search(p)] == [PLANTED], "the exclusion widened"


def test_ci_runs_the_gate_over_every_file() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pre-commit run --all-files" in ci, "no CI step invokes the hook"
