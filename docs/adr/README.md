# Architecture decision records

Why the library is shaped the way it is, with the evidence that decided it. Each record follows
[MADR 4](https://adr.github.io/madr/). Docstrings state the contract and point here with
`Rationale: ADR-NNNN.`; what changed release by release is in [`CHANGELOG.md`](https://github.com/mjdevaccount/llm-ports/blob/main/CHANGELOG.md).

The first eight were recorded together in 0.12.0, after the fact: each decision had shipped
earlier, and its reasoning lived in docstrings until then. The release that made each decision is
named in its record.

| ADR | Decision |
|---|---|
| [0001](0001-named-outcomes-never-raise.md) | A model call ends in a named outcome and never raises; `Judge` is the one port that raises |
| [0002](0002-grounded-prose-by-construction.md) | Grounded prose by construction: the model cites a value by id, code prints the value |
| [0003](0003-the-number-law-compares-by-value.md) | One number law, comparing figures by value, digits and spelled-out |
| [0004](0004-digit-free-context.md) | Context a digit-free writer reads is digit-free too |
| [0005](0005-character-whitelist-grammar.md) | The digit-free grammar is a character whitelist, not a negated class |
| [0006](0006-data-rights-in-prov-o-terms.md) | Data rights and lineage in PROV-O terms, with one deterministic print gate |
| [0007](0007-one-observation-lifecycle.md) | One observation lifecycle; unmeasured usage is unmeasured, never zero |
| [0008](0008-prompt-packs-named-by-content-sha.md) | Prompt packs live in the repo and are named by a content sha |
| [0009](0009-httpx2-client-under-the-pydantic-ai-adapter.md) | The pydantic-ai adapter hands its provider an httpx2 client and bridges a legacy httpx transport |
| [0010](0010-a-standalone-learning-loop-core.md) | A standalone learning-loop core (`llm_ports.learn`): one gate, one run ledger port, one labelled-example contract |
| [0011](0011-label-names-follow-the-stores-own-rule.md) | Label score names follow Langfuse's score-config rule (`label-<role>`); reads accept the old `:` spelling |
| [0012](0012-a-judge-earns-trust-from-labels.md) | A judge earns trust from human labels per judge model (`calibration.trust`, `kappa_ci`, `score` metadata) |
| [0013](0013-forecast-scoring-in-a-standalone-core.md) | Forecast scoring in a standalone core (`llm_ports.forecast`); fitters and non-binary scores in four adapters |
