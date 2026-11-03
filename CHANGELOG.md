# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Why a design is the way it is lives in
[`docs/adr/`](https://github.com/mjdevaccount/llm-ports/tree/main/docs/adr).

## [Unreleased]

## [1.0.0rc1] - 2026-11-03

### Added

- A documentation site (mkdocs-material + mkdocstrings, `mkdocs.yml`): concepts, a guide per
  example, an API reference page per subpackage generated from the docstrings, the ADRs and this
  changelog. The new `docs` CI job builds it with `--strict`; a release tag deploys it to GitHub
  Pages.
- `examples/`: named outcomes as a fallback `match`, grounded prose with slots, a prompt pack,
  the print gate, a judge and an adversary, chunk/embed/score, OpenTelemetry spans (all
  network-free), and an OpenAI-compatible endpoint (run when its env vars are set). Each asserts
  its claims; `tests/examples/` runs them, and every Python block in the README and docs, on every
  `pytest` leg.
- Strict typing: `mypy --strict` over the package and `pyright --verifytypes` at 100% (the public
  surface fully typed), run by the new `types` CI job. `CaptureLike` members are read-only
  properties, so any capture with those attributes satisfies it; `StubEmbedder` and `StubReranker`
  take `fail: Outcome`; `PackManifest.model_config` is a `ClassVar`.
- CI: `vulture` (dead code) in `lint`; the `core-only` job (the core with no extra, then with
  `http` only).
- Releases publish to PyPI through trusted publishing (no stored token), with PEP 740
  attestations; a tag still attaches the wheel and sdist to a GitHub release, independently of the
  upload. A manual run of `release.yml` uploads a `.devN` pre-release as the dry run.
- Package metadata: project URLs, classifiers and keywords.
- CI: `actionlint` over the workflows; `ci.yml` is callable (`workflow_call`), and a release runs
  all of it before building.

### Changed

- The README is rewritten for a first reader: the one idea, a 30-line quickstart that needs no
  network or key (run in CI), the concepts with their ADRs, and the layer diagram.
- **Breaking: the package is layered** (`src/llm_ports/`, PyPA src layout). The 33 flat modules
  move into subpackages, bottom-up: `ports` (with `ports.outcomes`, was `failure`) <
  `observability` (`context`, `observation`, `NoopTracer`) | `grounding` (`numbers`, `slots`, the
  belt, `NumberLawGrounder`) | `prompts` (`packs`, `manifest`, `RepoPromptSource`) < `call`
  (`generate`, `call.roles`, `call.graders`, `call.schema`) < `evals` (`GauntletEvaluator`,
  `evals.judge`, `evals.refute`) | `rag` (`rag.chunking`, `rag.metrics`) | `provenance` (was
  `content`) < `adapters` (`pydantic_ai`, `pydantic_evals`, `langfuse`, `otel`, `http`,
  `lettucedetect`, `sqlglot`) | `testing` (`testing.langfuse`). `reference` is dissolved into
  the packages above. The root keeps its names. The map is `docs/REDESIGN.md` §1.
- Every module declares `__all__`, sorted (ruff `RUF022`); a test asserts each name resolves.
- import-linter contracts (`pyproject.toml`) hold the layers, keep adapters independent and keep
  vendor imports inside `adapters` and `testing`; `tests/contract/` runs them on every leg.
- Tests mirror the package; fixtures live in `tests/fixtures/`.
- **Breaking: every framework is an extra.** `pip install llm-ports` installs pydantic, jinja2 and
  pyyaml; pydantic-ai and httpx move to the `pydantic-ai` and `http` extras (`all` = every extra
  but `lettucedetect`). An adapter imported without its extra fails with the install line.
  `tests/contract/test_core_imports.py` and the `core-only` CI job hold the core vendor-free.
- **Breaking: the tier key `box` is renamed `self_hosted`** (`Tier.self_hosted`,
  `GenRequest.self_hosted`), default true. `GenRequest.label` defaults to `"chat"`.
- **Breaking: names moved to the layer that owns them.** `outcome_for_status` and `REQUEST_FAULT`
  -> `ports.outcomes`; `strict_json_schema` -> `adapters.pydantic_ai`; `EventHook` (and the new
  `log_events`) -> `observability`, so `call` no longer exports `EventHook` or `Observation`;
  `IndexOrderedTransport` / `INDEXED_LISTS` are imported from `adapters.http` (or
  `adapters.pydantic_ai`), the module `index_order` is private; the Protocols live in
  `ports._protocols`, still exported from `ports`.
- **Breaking: helpers that were never the contract are private:** `capture`, `record_response`,
  `Capture` (`CaptureLike` stays public), `call.trace_output`, the prompt source's
  `kind` / `wire` / `served`.
- **Breaking: `evals.judge.rubric` takes no domain vocabulary.** The number law's examples are the
  consumer's, passed as `examples=numbers.LawExamples(paraphrase=..., fact_names=...)`; with the
  0.12 examples the text is byte-identical to 0.12 (the golden test).
- `load_tiers` takes a mapping as well as a path; checks and errors name it `<mapping>`.
- `__version__` is read from the installed metadata.
- Logger names follow the modules (`llm_ports.adapters.langfuse.tracer`, `.prompt_source`,
  `llm_ports.call`).
- Long functions split on merit (`call.generate`, the pydantic-ai generator); magic numbers named
  (`CHAIN_LIMIT`, `DEFAULT_TIMEOUT_S`, `SYSTEM_CAP`, `SCHEMA_CAP`).
- `uv.lock` refreshed to the latest releases (pydantic-ai-slim and pydantic-evals 2.65.0).

### Deprecated

- The tier key `box`: a table that still sets it loads, with a `DeprecationWarning` pointing at
  the `load_tiers` caller; a tier setting both `box` and `self_hosted` is refused.

### Fixed

- The `pydantic-ai` extra names httpx: the adapter builds an httpx client, and from pydantic-ai
  2.6x / openai 3.4x neither depends on httpx (both moved to httpx2). The run's usage is read as
  the property pydantic-ai made it, so 2.65 no longer warns on `usage()`.

## [0.12.0] - 2026-11-03

No behaviour change and no public name change; the release carries documentation, tooling and
dependency floors.

### Added

- `numbers.law_text(mode, *, web)`: the number law's one English statement, beside the check it
  states (`numbers.loose`). `judge_sampling.rubric` composes it; its output is byte-identical to
  0.11.0 for every grounding mode and web flag, pinned by `tests/test_rubric_golden.py`.
- Architecture decision records `docs/adr/0001`–`0008` (MADR), holding the rationale and the
  measured evidence that used to live in docstrings.
- `CONTRIBUTING.md`: the docstring rules, where history goes, how to release.
- A `pre-commit` hook that rejects deployment vocabulary in `llm_ports/`, run in CI.
- CI: pytest on Python 3.12, 3.13, 3.14 and 3.15 (Ubuntu) and 3.13 (Windows); the suite at every
  dependency's declared lower bound (`--resolution lowest-direct`); a branch-coverage floor with
  Codecov upload over OIDC; `ruff format --check`; `twine check --strict` on the build; `zizmor`
  over the workflows; every workflow defaults to `permissions: {}`.

### Changed

- Every docstring states the contract in Google style, with invariants in an `Invariants:`
  section; ruff's `D` rules (Google convention) enforce the layout.
- The tree is formatted with `ruff format`.
- Dependency floors match what is tested: `pydantic>=2.12` (already required by
  `pydantic-ai-slim>=2.57`) and, for the `sql` extra, `sqlglot>=30.22`. Older sqlglot drops
  `LIMIT ALL` at parse time, so `sql_guard.guard` capped the query instead of refusing it.

### Fixed

- `content.print_gate` and `DataClass.allows` document the `ValueError` they raise for an audience
  outside `AUDIENCES`; the old text said `print_gate` never raised.

## [0.11.0] - 2026-11-03

### Added

- `ports.Reranker`, the tenth port, and `Reranked` (scores in input order, order-only; never
  raises for a model, wire or gate failure). Also exported from the package root.
- `reranker_http`: `HttpReranker` over the Cohere/Jina-shaped `/rerank` wire, `payload`,
  `scores_from`. Every document is scored exactly once (indices exactly `0..n-1`) and every score
  is finite, or the result is `invalid`.
- `observation`: `observe(tracer, ...)` opens one generation around a block and ends it once
  (`ERROR` and `latency_ms` on an exception); `PortObservation` (was `call._PortObservation`) with
  `usage_counts()` and `fail(exc, metadata)`; `opt_int`. A call that reported neither usage count
  stamps `tokens_unmeasured: true`. `call.tracer_trace` is built on `observe`.
- `langfuse_tracer` (extra `langfuse`): `LangfuseTracer(client, ..., on_event=)`,
  `LangfuseGeneration`, `usage_details`, `trace_id`.
- `langfuse_prompt_source`: `LangfusePromptSource(client, seed_for, ..., on_event=)`; the repo
  leads and the label mirrors. Duck-typed client, no SDK import.
- `otel_gen_ai` (extra `otel`): `gen_ai_span`, `record_gen_ai` (back-dated), `GenAiRecorder`;
  attribute names from `opentelemetry-semantic-conventions`, emitting `gen_ai.provider.name`
  beside the deprecated `gen_ai.system`.
- `sql_guard` (extra `sql`): `guard(sql, policy)`, `SqlGuardPolicy` (with `dialect`, default
  postgres), `SqlGuardError`, `FORBIDDEN_NODES`, `POSTGRES_DENY_FUNCTIONS`.
- `testing.StubReranker`; `testing_langfuse` (extra `langfuse`): `LangfuseRig`, `observed`,
  `FakeLangfusePrompts`, `RefusingLangfuse`.

### Changed

- `index_order.IndexOrderedTransport` moved out of `pydantic_ai_embedder` (still importable
  there, the same object) and also orders `/rerank` replies' `results`.

## [0.10.0] - 2026-11-03

### Added

- `pydantic_evals_judge` (extra `evals`): `PydanticEvalsJudge(role, model=|model_for=)`,
  `JUDGE_PROFILE`, `clean_reason`, `verdict_of`, `judge_inputs`.
- `judge_sampling`: `Seen`, `seen_for`, `schema_of`, `judge_input`, `reply_text`, `sample_key`,
  `selected`, `rubric(role, seen, *, platform, mode, today, tz_label)`, `verdict_label`.
- `lettucedetect_grounder` (extra `lettucedetect`): `LettuceDetectGrounder`, `load_detector`,
  `Detector`, `MODEL`, `MAX_CONTEXT_CHARS`, `WINDOW_REASON`.
- `grounder_http`: `HttpGrounder`, `GrounderUnavailable`, `payload`.
- `claim_refute`: `refute_slots`, `refute_prompt(pack, ...)`, `refute(generate, role, *, pack, ...)`,
  `RefuteClaim`, `GenerateFn`.
- `json_schema`: `research_schema`, `RESEARCH_ANALYSIS`, `ANALYSIS_KEY`, `without_analysis`,
  `research(generate, role, ...)`.
- `testing.StubDetector`.

## [0.9.0] - 2026-11-03

### Added

- `chunking`: `chunk_text`, `break_floor`, `max_chars_for_chunks`, `content_hash`,
  `should_ingest`, `CHUNK_CHARS`, `CHUNK_OVERLAP`.
- `retrieval_metrics`: `first_rank`, `hit_at_k`, `reciprocal_rank`, `agree_at_k`, `mean`, `clears`.
- `reference`: `GauntletEvaluator`, `NumberLawGrounder`, `RepoPromptSource`, `NoopTracer`,
  `NoopGeneration`.
- `pydantic_ai_replay`: `Recorder`, `replay`, `Recording`, `key`, `NotRecorded`.
- `testing.StubCiter`, `testing.StubPromptSource`; `ports.Seed` (also exported from the root).
- `call_graders.TEXT_ONLY_GRADERS`, `graders_for_inputs`.

### Changed

- `call.generate` drops `TEXT_ONLY_GRADERS` for a call with `images=` or the new `web=True`, so an
  image or a web tier no longer fails `numbers_from_inputs` by construction. With no `trace` hook
  the call observes into `tracer_trace(NoopTracer())`, the package's one null sink.

## [0.8.1] - 2026-11-03

### Fixed

- `IndexOrderedTransport` re-sorts an `/embeddings` reply's `data` by `index`. pydantic-ai drops
  `index`, so a server answering out of order put every vector on its neighbour's row, which the
  count check cannot see. `openai_compatible_embeddings` always wraps its transport with it.

## [0.8.0] - 2026-11-03

### Added

- `ports.Embedder`, the ninth port, and `Embedded(vectors, outcome, stamp, reason, input_tokens)`:
  one vector per text in one named space; `input_tokens=None` when the wire reported no usage.
- `pydantic_ai_embedder.PydanticAIEmbedder` on pydantic-ai's `OpenAIEmbeddingModel`: count and
  dimension checks (a short batch or a wrong dimension is `invalid`, never a partial result),
  batching, an asymmetric query prefix, and the generator's never-raise outcome classification.
- `openai_compatible_embeddings(base_url, model_name, key, transport=)`.
- `testing.StubEmbedder`: deterministic hashed unit vectors.

## [0.7.0] - 2026-11-03

### Added

- `content`: `ContentItem` (`prov:Entity`), `Activity` (`prov:Activity`), `DataClass` (rights
  `public < owned < licensed < scraped`, `private`, `link_required`; `join` takes the most
  restrictive), `CEILING` per audience, and `print_gate(item, audience) -> GateVerdict`: ok, or one
  named refusal (`rights`, `unclaimed`, `uncited`, `unsourced`, `link`).

## [0.6.0] - 2026-10-31

### Added

- `slots.to_prose(text)`: context a digit-free grammar can read with no figure to reach for. Every
  digit run is removed whole (`numbers.DIGITS`), every character outside `PROSE_CHARS` becomes a
  space.

### Fixed

- `__version__` matches the package version (it said 0.3.0).

## [0.5.0] - 2026-10-31

### Added

- `call_graders.nullable_paths(model)`: the nullable leaf paths of a pydantic model.

### Changed

- `call_graders.schema_complete` reads a None on a nullable pydantic field as an answer; a None on
  a required field still fails, and a dict value is graded as before.

## [0.4.0] - 2026-10-29

### Added

- `numbers.loose(text, allowed, words_ok)`: the number law in one function. Digit runs by value
  not in `allowed`, then spelled numbers and number-word tails not in `words_ok`, in order and
  named.
- `grounding.loose`, `slots.loose`; `grounding.belt` and `slots.belt` count `loose`.

### Changed

- `numbers.uncited` is `loose` sorted into the cited-spans shape (same results on every pinned case).
- `call_graders.numbers_from_inputs` scores the distinct figures of `loose`, so spelled numbers
  count; the prompt's own words and a `%` license them.

## [0.3.0] - 2026-10-28

### Added

- `slots`: grounded prose by construction. `PROSE_CHARS`, `prose_pattern`, `slot_pattern` (the
  digit-free grammar), `SLOT`, `ID`, `cited`, `fill`, `problems` (unknown slot, stray brace, cited
  twice), `belt` over filled text.
- `numbers.canon` and `numbers.digits`: digit runs compare by value ("2.20" == "2.2").

### Changed

- `grounding.belt` compares digit runs by value.

## [0.2.0] - 2026-10-23

### Added

- `grounding`: `Problem(path, reason, detail="")`, `describe`, `retry_message`,
  `belt(text, allowed, words_ok, *, strip_links)`, `allowed_numbers(texts)`.
- `failure_reason(exc, *, with_cause=False)`: an optional ` <- <cause reason>` tail, cut by the
  same body-safe rule and within `REASON_MAX`.

## [0.1.1] - 2026-10-22

### Added

- `py.typed` (PEP 561).
- A release workflow: a `v*` tag builds the wheel and sdist from the tagged commit, refuses a
  tag that does not match the package version, and attaches both to a GitHub release.

## [0.1.0] - 2026-10-22

### Added

- The ports (Protocols and frozen value types), roles over a consumer's tier table, prompt packs
  named by content sha, per-call graders, `generate()` with named outcomes and `Hooks`, the
  pydantic-ai `Generator` adapter, and test doubles.

[Unreleased]: https://github.com/mjdevaccount/llm-ports/compare/v1.0.0rc1...HEAD
[1.0.0rc1]: https://github.com/mjdevaccount/llm-ports/compare/v0.12.0...v1.0.0rc1
[0.12.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/mjdevaccount/llm-ports/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mjdevaccount/llm-ports/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/mjdevaccount/llm-ports/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mjdevaccount/llm-ports/releases/tag/v0.1.0
