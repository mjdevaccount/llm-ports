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
| [0014](0014-a-persons-label-relayed-by-code.md) | A person's label can be relayed by code (`put` with `by`), on the same score config, completing its queue item |
| [0015](0015-a-combiner-is-judged-walk-forward.md) | A combiner is replayed walk-forward, shrunk toward a prior, and tuned only on the past (`forecast.walk`, `PriorPool`, `adapters.optuna`) |
| [0016](0016-seams-to-bigger-ml-engines.md) | Seams to bigger ML engines: run ids, artifact URIs, a `ModelRegistry` port, conformance suites, and MLflow as the first engine adapter |
| [0017](0017-prompt-tuning-on-gepa-core.md) | Prompt tuning runs on the `gepa` core behind a caller's evaluator; the study UI is launched on demand |
| [0018](0018-the-work-market-declaration-side.md) | The work market's declaration side is a core subpackage (`llm_ports.market`); packs from other distributions load through pluggy entry points (`adapters.pluggy`) |
| [0019](0019-a-result-names-what-was-measured-what-it-cost-and-who-answered.md) | A result names what was measured (`Usage`, summed over every request), what it cost (`Cost`: gateway, then a `Pricer`, then unknown) and who answered (`Answered`) |
| [0020](0020-a-failure-has-a-kind.md) | A failure has a kind (`Failure.kind`) under coarse outcomes; `refused` is the seventh outcome, a cut reply is `truncated`, a bug is `internal` |
| [0021](0021-one-retry-owner-and-a-total-deadline.md) | One retry owner (the adapter's loop, keyed on `Failure.kind`, a `WirePolicy`) and a total `Deadline` over the whole call; a cancel propagates; a slot is taken inside the deadline |
| [0022](0022-the-model-table.md) | The model table: schema 2 states each model's capabilities and its provider by name; capability is checked before the wire; providers and packs load through one strict plugin contract; a never-read gate fails closed |
| [0026](0026-a-pack-has-an-identity-beside-its-content-sha.md) | A pack has an `identity` (`byte_sha` over the manifest and the closed template set) beside its frozen `content_sha`; CRLF is refused; the manifest is resolved by the library |
