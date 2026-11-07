"""`llm_ports.learn` lifts out whole: copied alone into an empty tree, it imports and runs a loop.

The import-linter contract `learn-standalone` stops `learn` importing the rest of `llm_ports`.
This proves the other half: copied with no other `llm_ports` module beside it, the package runs a
loop end to end on the standard library alone, in an isolated interpreter (`-I`), and loads no
module outside the standard library.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import llm_ports.learn

SCRIPT = """
import asyncio, json, sys
sys.path.insert(0, sys.argv[2])
from llm_ports.learn import Fit, gate, run
from llm_ports.learn.stores import JsonlRunLedger

class Loop:
    kind, subject, params = "demo", "", {}
    async def evidence(self):
        return [1, 2, 3]
    def fit(self, xs):
        return Fit({"mean": sum(xs) / len(xs)}, {"n": len(xs)})
    def clauses(self, fit, xs):
        return [gate.at_least("n", fit.metrics["n"], 3, floor=True)]

got = asyncio.run(run(Loop(), JsonlRunLedger(sys.argv[1])))
third = sorted(
    m.split(".")[0] for m in sys.modules
    if m.split(".")[0] not in sys.stdlib_module_names and m.split(".")[0] != "llm_ports"
)
print(json.dumps({"decision": got.verdict.decision, "recorded": got.recorded.ok, "third_party": third}))
"""


def test_learn_runs_alone_on_the_standard_library(tmp_path: Path) -> None:
    pkg = tmp_path / "tree" / "llm_ports"
    shutil.copytree(Path(llm_ports.learn.__file__).parent, pkg / "learn", ignore=shutil.ignore_patterns("__pycache__"))
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    script = tmp_path / "demo.py"
    script.write_text(SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", str(script), str(tmp_path / "runs.jsonl"), str(tmp_path / "tree")],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got == {"decision": "pass", "recorded": True, "third_party": []}
    assert (tmp_path / "runs.jsonl").read_text(encoding="utf-8").count('"kind": "demo"') == 1
