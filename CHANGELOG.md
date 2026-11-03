# Changelog

Releases before 0.9.0: the GitHub release notes (`gh release view vX.Y.Z -R mjdevaccount/llm-ports`).

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
