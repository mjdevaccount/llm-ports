# Changelog

Releases before 0.9.0: the GitHub release notes (`gh release view vX.Y.Z -R mjdevaccount/llm-ports`).

## 0.11.0 -- 2026-11-03

Third wave out of trade-data: the reranker port and its HTTP adapter, the Langfuse tracer and prompt source, OpenTelemetry GenAI spans, the generation-observation lifecycle, and the read-only SQL guard. trade-data keeps only its wire (client construction, flags, its category semantics, its schema allowlist).

- `ports`: `Reranker` (the tenth port) + `Reranked` (scores in input order, ORDER-ONLY; never raises for a model, wire or gate failure). Also exported from the package root.
- `reranker_http` (new): `HttpReranker` over the Cohere/Jina-shaped `/rerank` wire, `payload`, `scores_from`. Laws: every document scored exactly once (indices exactly 0..n-1 -- a duplicate index no longer passes a count check), every score finite; else `invalid`.
- `index_order` (new): `IndexOrderedTransport` moved here from `pydantic_ai_embedder` (still importable there -- the same object) and extended to `/rerank` replies' `results`, so the index is read in one place for both wires.
- `observation` (new): `observe(tracer, ...)` -- one generation around a block, ended once (ERROR + `latency_ms` on an exception); `PortObservation` (public; was `call._PortObservation`) with `usage_counts()` and `fail(exc, metadata)`; `opt_int`. Neither usage count reported now also stamps `tokens_unmeasured: true` in the metadata. `call.tracer_trace` is built on `observe`; `call.Observation` is re-exported from here.
- `langfuse_tracer` (new, extra `langfuse`): `LangfuseTracer(client, ..., on_event=)`, `LangfuseGeneration`, `usage_details`, `trace_id`. `provider` rides `metadata["provider"]` unless the caller set it.
- `langfuse_prompt_source` (new): `LangfusePromptSource(client, seed_for, ..., on_event=)` -- the repo leads, the label mirrors; duck-typed client, no SDK import.
- `otel_gen_ai` (new, extra `otel`): `gen_ai_span(tracer, ...)`, `record_gen_ai(tracer, ...)` (back-dated), `GenAiRecorder`; attribute names from `opentelemetry-semantic-conventions`; emits `gen_ai.provider.name` beside the deprecated `gen_ai.system`.
- `sql_guard` (new, extra `sql`): `guard(sql, policy)`, `SqlGuardPolicy` (adds `dialect`, default postgres), `SqlGuardError`, `FORBIDDEN_NODES`, `POSTGRES_DENY_FUNCTIONS`.
- `testing`: `StubReranker`. `testing_langfuse` (new, extra `langfuse`): `LangfuseRig` (a real SDK 5 client exporting to memory), `observed`, `FakeLangfusePrompts`, `RefusingLangfuse` (every call raises).
- Dev group gains langfuse, opentelemetry-sdk, opentelemetry-semantic-conventions and sqlglot (those adapters' tests run in CI).

## 0.10.0 -- 2026-11-03

Second wave out of trade-data: the judge, the judge sampler's pure half, the research and refute passes, and the two grounder adapters. trade-data keeps only its wire (transport, roles, calendar, flags, tables).

- `pydantic_evals_judge` (new, extra `evals`): `PydanticEvalsJudge(role, model=|model_for=)`, `JUDGE_PROFILE`, `clean_reason`, `verdict_of`, `judge_inputs`.
- `judge_sampling` (new): `Seen`, `seen_for`, `schema_of`, `judge_input`, `reply_text`, `sample_key`, `selected`, `rubric(role, seen, *, platform, mode, today, tz_label)`, `verdict_label`.
- `lettucedetect_grounder` (new, extra `lettucedetect`): `LettuceDetectGrounder`, `load_detector`, `Detector`, `MODEL`, `MAX_CONTEXT_CHARS`, `WINDOW_REASON`.
- `grounder_http` (new): `HttpGrounder`, `GrounderUnavailable`, `payload`.
- `claim_refute`: `refute_slots`, `refute_prompt(pack, ...)`, `refute(generate, role, *, pack, ...)`, `RefuteClaim`, `GenerateFn`.
- `json_schema`: `research_schema`, `RESEARCH_ANALYSIS`, `ANALYSIS_KEY`, `without_analysis`, `research(generate, role, ...)`.
- `testing`: `StubDetector`.
- Dev group gains `pydantic-evals` (the judge's tests run in CI); `lettucedetect` stays out of it (its tests run over `StubDetector`).

## 0.9.0 -- 2026-11-03

Generic LLM/RAG mechanics moved out of trade-data (its first wave; trade-data keeps only the EDGE wiring).

- `chunking` (new): `chunk_text`, `break_floor`, `max_chars_for_chunks` (tight worst-case bound), `content_hash`, `should_ingest`, `CHUNK_CHARS` / `CHUNK_OVERLAP`.
- `retrieval_metrics` (new): `first_rank`, `hit_at_k`, `reciprocal_rank`, `agree_at_k`, `mean`, `clears`.
- `reference` (new): `GauntletEvaluator`, `NumberLawGrounder`, `RepoPromptSource`, `NoopTracer` / `NoopGeneration`.
- `pydantic_ai_replay` (new): `Recorder`, `replay`, `Recording`, `key`, `NotRecorded`.
- `testing`: `StubCiter`, `StubPromptSource`.
- `ports`: `Seed` (a prompt's repo copy; also exported from the package root).
- `call_graders`: `TEXT_ONLY_GRADERS`, `graders_for_inputs`.
- `call.generate`: drops `TEXT_ONLY_GRADERS` for a call with `images=` or the new `web=True`, so an image or a web tier no longer fails `numbers_from_inputs` by construction. With no `trace` hook the call now observes into `tracer_trace(NoopTracer())` -- one null sink in the package (the private `_NullObservation` is gone).
