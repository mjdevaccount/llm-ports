"""The learning-loop core: labelled examples, one gate, one run ledger, one loop runner.

Every learning loop is the same moves: read evidence, fit or measure, hold the result to a gate,
and record the run. This package holds the parts that do not depend on what is learned:

- values: `Example`, `Clause`, `Verdict`, `LedgerRow` (and the store results).
- `gate`: clause helpers and `decide`, the one way a verdict is reached.
- `stats`: agreement (`cohen_kappa`, `gwet_ac1`) and false-discovery control (`bh_fdr`).
- `examples`: `coverage`, `agreement`, `few_shot` over labelled examples.
- ports: `RunLedger`, `ExampleStore`, `WritableExampleStore`, `ModelRegistry`, `Loop`; `stores`
  and `registry` implement them with the standard library; `loop.run` drives a `Loop` (evidence,
  fit, clauses, apply) and records exactly one row.

Invariants:
    - Standard library only: imports nothing else from `llm_ports` and no third-party package,
      so it can be lifted into any codebase (enforced by an import-linter contract).

Rationale: ADR-0010, ADR-0016 (the seams to bigger engines).
"""

from llm_ports.learn import examples, gate, loop, registry, stats, stores
from llm_ports.learn._protocols import ExampleStore, Fit, Loop, ModelRegistry, RunLedger, WritableExampleStore
from llm_ports.learn._types import (
    DECISIONS,
    ROLES,
    VERDICT_KEY,
    Clause,
    Decision,
    Example,
    ExamplesRead,
    LedgerRow,
    ModelVersion,
    Role,
    RowRead,
    Source,
    StoreOutcome,
    Value,
    Verdict,
    VersionRead,
    Written,
)
from llm_ports.learn.loop import Run, record, run

__all__ = [
    "DECISIONS",
    "ROLES",
    "VERDICT_KEY",
    "Clause",
    "Decision",
    "Example",
    "ExampleStore",
    "ExamplesRead",
    "Fit",
    "LedgerRow",
    "Loop",
    "ModelRegistry",
    "ModelVersion",
    "Role",
    "RowRead",
    "Run",
    "RunLedger",
    "Source",
    "StoreOutcome",
    "Value",
    "Verdict",
    "VersionRead",
    "WritableExampleStore",
    "Written",
    "examples",
    "gate",
    "loop",
    "record",
    "registry",
    "run",
    "stats",
    "stores",
]
