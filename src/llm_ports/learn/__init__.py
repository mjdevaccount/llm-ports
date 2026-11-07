"""The learning-loop core: labelled examples, one gate, one run ledger, one loop runner.

Every learning loop is the same moves: read evidence, fit or measure, hold the result to a gate,
and record the run. This package holds the parts that do not depend on what is learned:

- values: `Example`, `Clause`, `Verdict`, `LedgerRow` (and the store results).
- `gate`: clause helpers and `decide`, the one way a verdict is reached.
- `stats`: agreement (`cohen_kappa`, `gwet_ac1`) and false-discovery control (`bh_fdr`).
- `examples`: `coverage`, `agreement`, `few_shot` over labelled examples.
- ports: `RunLedger`, `ExampleStore`, `WritableExampleStore`, `Loop`; `stores` implements them
  with the standard library; `loop.run` drives a `Loop` and records exactly one row.

Invariants:
    - Standard library only: imports nothing else from `llm_ports` and no third-party package,
      so it can be lifted into any codebase (enforced by an import-linter contract).

Rationale: ADR-0010.
"""

from llm_ports.learn import examples, gate, loop, stats, stores
from llm_ports.learn._protocols import ExampleStore, Fit, Loop, RunLedger, WritableExampleStore
from llm_ports.learn._types import (
    DECISIONS,
    Clause,
    Decision,
    Example,
    ExamplesRead,
    LedgerRow,
    RowRead,
    Source,
    StoreOutcome,
    Value,
    Verdict,
    Written,
)
from llm_ports.learn.loop import Run, run

__all__ = [
    "DECISIONS",
    "Clause",
    "Decision",
    "Example",
    "ExampleStore",
    "ExamplesRead",
    "Fit",
    "LedgerRow",
    "Loop",
    "RowRead",
    "Run",
    "RunLedger",
    "Source",
    "StoreOutcome",
    "Value",
    "Verdict",
    "WritableExampleStore",
    "Written",
    "examples",
    "gate",
    "loop",
    "run",
    "stats",
    "stores",
]
