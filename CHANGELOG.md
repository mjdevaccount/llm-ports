# Changelog

Releases before 0.9.0: the GitHub release notes (`gh release view vX.Y.Z -R mjdevaccount/llm-ports`).

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
