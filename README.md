# llm-ports

Thin Protocols for LLM work, and the one call path around them. **Our seam, their impl:** the
ports are ours and tiny; structured output, output retries and validation come from
[pydantic-ai](https://ai.pydantic.dev/) behind the `Generator` port, never hand-rolled here.

```
pip install "llm-ports @ https://github.com/mjdevaccount/llm-ports/releases/download/v0.5.0/llm_ports-0.5.0-py3-none-any.whl"
```

## What is in it

| module | what |
|---|---|
| `ports` | zero-I/O Protocols: `Generator`, `Tracer`/`Generation`, `PromptSource`, `Judge`, `Retriever`, `Citer`, `Grounder`, `Evaluator`; value types `GenRequest`, `Generated[T]`, `Prompt`, `Chunk`, `Claim`, `CitedAnswer` |
| `roles` | `Role` (one generation surface, `feature.action`) resolved against a **consumer-owned TOML tier table** (`load_tiers`) |
| `packs` | prompt packs: `<root>/<id>/pack.yaml` + jinja templates, with a frozen **content sha** (sha256[:16] of the sources) |
| `call` | `generate(role, prompt=, output=, generator=, hooks=)` -- the one call path; outcome is always NAMED: `ok \| gated \| unavailable \| request_fault \| truncated \| invalid` |
| `call_graders` | per-call graders (`schema_complete` -- a None on a pydantic field whose annotation admits None is an honest answer, not a miss; `numbers_from_inputs` -- digits by value AND spelled numbers, `echo`); a grader result is a recorded score, a crashed grader scores `-1.0` |
| `pydantic_ai_generator` | the first `Generator` adapter (`PydanticAIGenerator`, `openai_compatible(base_url, key)` for any OpenAI-wire endpoint, e.g. a LiteLLM gateway) |
| `grounding` | the shapes a grounded generator shares: `Problem(path, reason, detail)`, `retry_message` (the ONE retry text), `loose` (the figures no input rendered, NAMED, so a verifier can refuse one by name), `belt` (`loose` counted; 0 by construction) and `allowed_numbers` |
| `slots` | grounded prose BY CONSTRUCTION: code renders each value under a digit-free id, the model writes `{id}` under a grammar with no digit token (`slot_pattern` / `prose_pattern`), code `fill`s it, and `belt` over the filled text is 0. Use it for any prose a reader takes numbers from |
| `testing` | `StubGenerator`, `RecordingTracer` for consumers' tests |
| `numbers` | THE number law: `loose(text, allowed, words_ok)` names every digit run (by value) and spelled number no input licensed; every belt, grader and validator in the package reads a number through it |
| `failure`, `context`, `json_schema`, `claim_refute` | small EDGE-free helpers the call path and graders use |

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
