---
status: accepted
date: 2026-11-07
---

# A standalone learning-loop core inside llm-ports

Decided in 1.5.0.

## Context and problem statement

A consuming platform runs dozens of learning loops (signal weights, calibrations, judges, grids)
and each re-wrote the same parts: a gate with its own verdict shape, its own way of recording a
run, its own idea of "not enough evidence". Three gate shapes and an `earned()` boolean per fit
meant no page could read a verdict generically, a clause that could not be evaluated was handled
differently in each, and a loop that stopped deciding left no trace. LLM judges need the same
parts: labelled examples, agreement statistics, a trust gate. Where should one shared core live,
and what must it guarantee?

## Decision drivers

- Usable in an unrelated codebase: no vendor, no dependency on the rest of this library.
- One verdict vocabulary (`pass | fail | insufficient | could_not_run`) that can express every
  existing gate exactly, including a promote / hold / reject gate, without changing a threshold.
- Missing evidence never passes, and a run that could not decide is recorded, not silent.
- Cheap to adopt: a new loop is four methods.

## Considered options

1. A new repository and distribution for the core.
2. A subpackage of llm-ports, enforced standalone.
3. MLflow tracking and registry as the ledger and gate store.

## Decision outcome

Option 2: `llm_ports.learn`.

- llm-ports already has the release, typing, import-layer and executed-docs machinery a published
  core needs; a new repository would rebuild it and add a second pin to every consumer.
- An import-linter `forbidden` contract stops `learn` importing any other `llm_ports` module or a
  third-party package, and `tests/contract/test_learn_standalone.py` copies the package alone into
  an empty tree and runs a loop there, so "lift it out" is executed, not claimed. Splitting it into
  its own distribution later is a file move.
- Its Protocols (`RunLedger`, `ExampleStore`, `Loop`) live in `learn/`, not `ports/`: a
  deliberate exception to "a port lives in `ports/`", because `ports/` carries LLM types and the
  standalone contract forbids reaching it.
- `learn` sits in the bottom layer beside `ports`; `evals.calibration` re-exports `cohen_kappa`
  and `gwet_ac1` from `learn.stats`, so no consumer changes.
- The gate gives each clause a role (`bar`, `floor`, `veto`) and takes an `unevaluated` policy
  (`fail` or `insufficient`); with those, a strict AND, a sample floor and a promote / hold /
  reject gate are all the same `decide`.
- A recorded verdict lives under its own metric key, `learn_verdict`, because an existing ledger
  already uses `gate` and `verdict` for other shapes.
- `bh_fdr` defaults to Benjamini-Yekutieli (valid under any dependence between tests);
  `dependent=False` gives Benjamini-Hochberg.

MLflow (option 3) is rejected as the ledger: it is a new service with its own storage, backup and
auth, a second registry beside the consumer's, and the "models" these loops fit are small
parameter rows, not artifacts. An MLflow `RunLedger` adapter stays possible as an optional extra.

### Consequences

- Good: one verdict type and one run record for every loop; a new loop is four methods; the core
  runs anywhere with a JSON-lines ledger and zero setup.
- Good: a loop that cannot decide, or whose action fails, leaves a row saying so.
- Bad: the package name says "llm" for a core that is not LLM-specific; the trigger to split it
  out is a consumer that objects to the name.
- Bad: a database ledger is the consumer's adapter to write (two methods) until a portable
  Postgres adapter ships as an extra.
