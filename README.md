# llm-ports

[![ci](https://github.com/mjdevaccount/llm-ports/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mjdevaccount/llm-ports/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mjdevaccount/llm-ports/graph/badge.svg)](https://codecov.io/gh/mjdevaccount/llm-ports)

Thin Protocols for LLM work, and the one call path around them. **Our seam, their impl:** the
ports are ours and tiny; structured output, output retries and validation come from
[pydantic-ai](https://ai.pydantic.dev/) behind the `Generator` port, never hand-rolled here.

```
pip install "llm-ports @ https://github.com/mjdevaccount/llm-ports/releases/download/v0.11.0/llm_ports-0.11.0-py3-none-any.whl"
```

## What is in it

| module | what |
|---|---|
| `ports` | zero-I/O Protocols: `Generator`, `Embedder`, `Reranker`, `Tracer`/`Generation`, `PromptSource`, `Judge`, `Retriever`, `Citer`, `Grounder`, `Evaluator`; value types `GenRequest`, `Generated[T]`, `Embedded`, `Reranked`, `Prompt`, `Chunk`, `Claim`, `CitedAnswer` |
| `roles` | `Role` (one generation surface, `feature.action`) resolved against a **consumer-owned TOML tier table** (`load_tiers`) |
| `packs` | prompt packs: `<root>/<id>/pack.yaml` + jinja templates, with a frozen **content sha** (sha256[:16] of the sources) |
| `call` | `generate(role, prompt=, output=, generator=, hooks=)` -- the one call path; outcome is always NAMED: `ok \| gated \| unavailable \| request_fault \| truncated \| invalid` |
| `call_graders` | per-call graders (`schema_complete` -- a None on a pydantic field whose annotation admits None is an honest answer, not a miss; `numbers_from_inputs` -- digits by value AND spelled numbers, `echo`); a grader result is a recorded score, a crashed grader scores `-1.0`; `graders_for_inputs` drops `TEXT_ONLY_GRADERS` for a call with an image or a web tier (`call.generate(images=, web=True)` applies it) |
| `pydantic_ai_generator` | the first `Generator` adapter (`PydanticAIGenerator`, `openai_compatible(base_url, key)` for any OpenAI-wire endpoint, e.g. a LiteLLM gateway) |
| `grounding` | the shapes a grounded generator shares: `Problem(path, reason, detail)`, `retry_message` (the ONE retry text), `loose` (the figures no input rendered, NAMED, so a verifier can refuse one by name), `belt` (`loose` counted; 0 by construction) and `allowed_numbers` |
| `slots` | grounded prose BY CONSTRUCTION: code renders each value under a digit-free id, the model writes `{id}` under a grammar with no digit token (`slot_pattern` / `prose_pattern`), code `fill`s it, and `belt` over the filled text is 0. Use it for any prose a reader takes numbers from |
| `content` | the OUTPUT shape in W3C PROV-O words: `ContentItem` (prov:Entity), `Activity` (prov:Activity), `uses` (prov:used); `DataClass{rights, private, link_required}` with `join` (most restrictive wins) and an audience `CEILING`; `print_gate(item, audience)` -> ok or ONE named refusal (`rights \| unclaimed \| uncited \| unsourced \| link`) |
| `pydantic_ai_embedder` | the first `Embedder` adapter (`PydanticAIEmbedder` on pydantic-ai's `embeddings`, `openai_compatible_embeddings(base_url, model)`): a `stamp` per space, count + dims laws, index order (`IndexOrderedTransport`), batching, the asymmetric query prefix, named outcomes |
| `testing` | `StubGenerator`, `StubEmbedder`, `StubReranker`, `RecordingTracer`, `StubCiter`, `StubPromptSource`, `StubDetector` for consumers' tests; `testing_langfuse` (extra `[langfuse]`): `LangfuseRig` (a real SDK 5 client exporting to memory), `FakeLangfusePrompts`, `RefusingLangfuse` |
| `reranker_http` | the first `Reranker` adapter (`HttpReranker` over the Cohere/Jina `/rerank` wire): scores in input order via `index_order.IndexOrderedTransport` (shared with the embeddings wire), every document scored exactly once and finite or `invalid`, named outcomes |
| `observation` | `observe(tracer, ...)`: one generation around a block, ended once (ERROR + latency on an exception); `PortObservation` (`usage_counts()`); `opt_int`; unmeasured usage is flagged `tokens_unmeasured`, never zero tokens |
| `langfuse_tracer` | the `Tracer` on the Langfuse SDK 5 (extra `[langfuse]`): a root generation in the current OTel trace, usage honesty, NUMERIC scores, the labelled prompt link; fails open with one held warning |
| `langfuse_prompt_source` | the `PromptSource` on Langfuse prompt management: the repo seed leads, the label mirrors (pushed when absent/stale, served when the store is down or disagrees); text and chat prompts |
| `otel_gen_ai` | OpenTelemetry GenAI spans (extra `[otel]`): `gen_ai_span` (a child of the ambient span), `record_gen_ai` (back-dated by the measured latency), none-is-not-zero usage; no prompt bodies, ever |
| `sql_guard` | a read-only AST guard for agent-written SQL (extra `[sql]`, sqlglot): one SELECT, no write/utility node, schema allowlist, function denylist, LIMIT enforced, SQL regenerated from the tree; a policy may tighten, never loosen |
| `pydantic_evals_judge` | the first `Judge` adapter (`PydanticEvalsJudge(role, model_for=...)` on pydantic-evals' `LLMJudge`, extra `[evals]`): the grade on the NATIVE grammar (`JUDGE_PROFILE`), `clean_reason` refuses leaked tool-call markup or a contradicting leaked verdict, multimodal `images=`; raises, never a default verdict |
| `judge_sampling` | a sampled judge's pure half: `Seen` (what the judged call saw beside the prompt) + `judge_input`, a deterministic `sample_key` + `selected` (1-in-N), the four-law `rubric` (platform, grounding mode, calendar injected), `verdict_label` (`pass \| fail \| could_not_run`) |
| `lettucedetect_grounder` | `LettuceDetectGrounder` (token-span hallucination detector behind `Grounder`, extra `[lettucedetect]`, lazy import, injectable detector); a context past the model's window is judged and NAMED `context_over_window` |
| `grounder_http` | `HttpGrounder`: the `Grounder` port over HTTP to a grounder sidecar (`POST {url}/verdict`); every non-answer raises `GrounderUnavailable`, never a guessed verdict |
| `reference` | the framework-free impl behind a port: `GauntletEvaluator` (rows + graders -> ONE board), `NumberLawGrounder` (the number law; numbers only), `RepoPromptSource` (every name its repo `Seed`), `NoopTracer` (no sink; also `call.generate`'s null trace -- one null sink) |
| `chunking` | `chunk_text` (paragraph > line > sentence breaks in the window's tail third, overlap, a per-chunk header), `max_chars_for_chunks` (the TIGHT worst-case char bound for n chunks, derived from the same floor), `content_hash` + `should_ingest` (skip-unchanged) |
| `retrieval_metrics` | `first_rank`, `hit_at_k`, `reciprocal_rank` over an ANY-OF relevant set, `agree_at_k` (top-k Jaccard), `mean`, `clears` (a parity gate's floor, rounded so a reported metric compares as written) |
| `pydantic_ai_replay` | record a live model once, replay in CI: `Recorder` / `replay` keyed by the first prompt's sha + attempt (a changed prompt is `NotRecorded`), `Recording` (every attempt kept parsed) |
| `numbers` | THE number law: `loose(text, allowed, words_ok)` names every digit run (by value) and spelled number no input licensed; every belt, grader and validator in the package reads a number through it |
| `claim_refute` | the adversarial pass: `verdicts_schema` (capped grammar), `refute_slots` / `refute_prompt(pack, ...)` (each claim idx-numbered, the consumer's pack rendered), `refute(generate, role, ...)` through the consumer's `generate`, `match_verdicts` (by idx), `refutation` (a cut reason flagged) |
| `json_schema` | `strict_json_schema`, `extract_json_object`; the research grammar `research_schema` (an `analysis` scratchpad first, every field required) + `research(generate, role, ...)` / `without_analysis` (scratchpad popped) |
| `failure`, `context` | small EDGE-free helpers the call path and graders use |

The package root imports no framework; import `llm_ports.pydantic_ai_generator` explicitly.

## Use

Bind your tier table to your roles once, by subclass (no process-global registry):

```toml
# llm_tiers.toml
[ladder]
gateway_s = 60.0
[models]
cheap = "gpt-5-nano"
[tiers.gateway]
model = "cheap"
timeout = "gateway_s"
box = false          # cloud tier: no self-hosted thinking switch
```

```python
from pathlib import Path
from llm_ports import Hooks, generate, load_tiers, roles
from llm_ports.pydantic_ai_generator import PydanticAIGenerator, openai_compatible

TABLE = load_tiers(Path(__file__).with_name("llm_tiers.toml"))


class Role(roles.Role):
    TIERS = TABLE.tiers


CATEGORIZE = Role("categorize.batch", tier="gateway", temperature=0.0, graders=("schema_complete",))
gen = PydanticAIGenerator(model_for=openai_compatible("http://litellm:4000/v1", api_key))

res = await generate(CATEGORIZE, prompt=text, output=MyOutput, generator=gen)
if res.outcome != "ok":
    ...  # gated / unavailable / request_fault / truncated / invalid -- never an exception
```

`Generator.generate` never raises for a model, wire or gate failure; it returns a named outcome
with a short reason (exception class, status code or the first message line cut before any `body:`
payload, capped at 200 chars -- never a traceback). A consumer that wants the class name alone splits on `:`.

Embeddings ride the same shape:

```python
from llm_ports.pydantic_ai_embedder import PydanticAIEmbedder, openai_compatible_embeddings

emb = PydanticAIEmbedder(
    model_for=openai_compatible_embeddings("http://litellm:4000/v1", "embeddings", api_key),
    stamp="Qwen3-Embedding-0.6B-Q8_0@1024",
    dims=1024,
    query_instruct="Given a question, retrieve relevant passages",
)
res = await emb.embed(chunks)  # Embedded(vectors, outcome, stamp, reason, input_tokens)
q = await emb.embed([question], input_type="query")
```

Store `res.stamp` with every vector: two stamps are two spaces and never compare. A wrong count or
dimension is `invalid` (nothing to store), never a partial batch.

## Hooks (the consumer's platform, plugged in)

`Hooks(admit=, trace=, graders=, on_scores=, on_ok=, on_event=)`, all optional:

- `admit(role) -> None | reason` -- a cap / kill switch; a reason makes the outcome `gated`.
- `trace(role, req, prompt, meta)` -- async CM yielding an `Observation`; `tracer_trace(tracer)`
  builds one over any `Tracer` port adapter (Langfuse, OTel, `testing.RecordingTracer`).
- `graders` -- the grader table (default `call_graders.GRADERS`); add your own.
- `on_scores(role, scores)` -- persist / count grader scores.
- `on_ok(role, prompt, res)` -- e.g. sample for an LLM judge.
- `on_event(level, event, fields)` -- default is the stdlib logger `llm_ports.call`.

## Known

- pydantic-ai 2.59 emits a deprecation warning for `httpx.AsyncClient` on OpenAI providers
  (`httpx2` in v3). The `<3` pin holds it; revisit at pydantic-ai 3.
- Version is locked to the pydantic-ai range of its first consumer (`>=2.57,<3`).

MIT.
