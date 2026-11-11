# Concepts

## The call path

Every generation goes through one function, `llm_ports.generate`:

```
Role ──▶ request_for ──▶ admit? ──▶ trace ──▶ Generator.generate ──▶ graders ──▶ on_scores / on_ok
          (tier → model,     │          (one observation,                (scores ride
           timeout, budget)  └─ gated    opened and ended once)            in the stamp)
```

The `Generator` is a port: an adapter (`PydanticAIGenerator`), a test double (`StubGenerator`) or
anything with the same method. It never raises for a model, wire or gate failure; it returns a
`Generated` with a named outcome and a short reason (the exception class, a status code, or the
first message line, capped at 200 characters, never a traceback).

| outcome | meaning |
|---|---|
| `ok` | a value that parsed and passed its validator |
| `gated` | a hook refused the call before anything was sent |
| `unavailable` | the model or the wire failed: retry elsewhere |
| `request_fault` | the server rejected the request itself (400, 413, 422): change the request |
| `truncated` | the reply hit its token ceiling |
| `invalid` | a reply that never parsed or validated, re-asks included |

## Roles and the tier table

A `Role` names one generation surface, `feature.action`, and resolves it against a tier table the
consumer owns. Bind the table to your roles once, by subclass, so two consumers in one process
never share a table:

```toml
# llm_tiers.toml
[ladder]
gateway_s = 60.0
[models]
cheap = "gpt-5-nano"
[tiers.gateway]
model = "cheap"
timeout = "gateway_s"
self_hosted = false  # a cloud tier: no self-hosted thinking switch (default true)
```

```{.python test="skip"}
from pathlib import Path

from llm_ports import Role, load_tiers

TABLE = load_tiers(Path(__file__).with_name("llm_tiers.toml"))


class AppRole(Role):
    TIERS = TABLE.tiers


CATEGORIZE = AppRole("categorize.batch", tier="gateway", temperature=0.0, graders=("schema_complete",))
```

`load_tiers` also takes the parsed table as a mapping (from your own config loader); errors and
the `check_ladder` hook then name the source `<mapping>`. The pre-1.0 key `box` still loads, with
a `DeprecationWarning`; a tier that sets both `box` and `self_hosted` is refused.

### Schema 2: what each model can do, and who serves it

`schema_version = 2` adds `[providers.<name>]` (a registered `kind`, and by name only the URL key
and the secret it needs), `[models.<key>]` (the wire id, its provider, and `supports` from a closed
capability vocabulary), `[policies.<name>]`, and per tier ordered `fallback` tiers and the
`fallback_on` failure kinds (ADR-0022). A refusal is `TableInvalid`, naming the path:

```toml
schema_version = 2

[ladder]
gateway_s = 60.0

[providers.gateway]
kind = "openai_compatible"
base_url = "gateway"      # a key your code resolves to a URL
secret = "gateway_key"    # a name your code resolves to a key

[models.cheap]
id = "gpt-5-nano"
provider = "gateway"
supports = ["json_schema", "vision"]
thinking = "unified"
self_hosted = false

[tiers.gateway]
model = "cheap"
timeout = "gateway_s"
```

`call.generate` refuses a request the tier's model does not state (images without `vision`, a
schema without `json_schema` or `json_object`, `web=True` without `web`) as `request_fault`
before anything is sent, and stamps every result with the table's `fingerprint()` as
`stamp["tiers"]`. `TABLE.require(ROLE, "vision")` at import proves the role's tier, and every
tier it can fall back to, can take what the role sends. `call.providers.build(TABLE, urls=...,
secrets=..., deps=BuildDeps(), factories=...)` builds the clients once at boot and raises there for
an unknown kind, an unresolved URL key or a secret that resolves to nothing;
`backends.generator_for(tier)` is the `generator=` for a call.

## Hooks: the consumer's platform, plugged in

`Hooks(admit=, trace=, graders=, on_scores=, on_ok=, on_event=)`, all optional:

- `admit(role) -> None | reason`: a cap or a kill switch; a reason makes the outcome `gated`.
- `trace(role, req, prompt, meta)`: an async context manager yielding an `Observation`;
  `tracer_trace(tracer)` builds one over any `Tracer` adapter (Langfuse, OpenTelemetry,
  `testing.RecordingTracer`).
- `graders`: the grader table (default `call.graders.GRADERS`); add your own.
- `on_scores(role, scores)`: persist or count grader scores.
- `on_ok(role, prompt, res)`: for example, sample the reply for an LLM judge.
- `on_event(level, event, fields)`: structured events; the default writes to the stdlib logger
  `llm_ports.call`.

## Embeddings

Embeddings ride the same shape: a named outcome, never an exception, and a `stamp` that names the
vector space.

```{.python test="skip"}
from llm_ports.adapters.pydantic_ai import PydanticAIEmbedder, openai_compatible_embeddings

emb = PydanticAIEmbedder(
    model_for=openai_compatible_embeddings("http://localhost:4000/v1", "embeddings", api_key),
    stamp="Qwen3-Embedding-0.6B-Q8_0@1024",
    dims=1024,
    query_instruct="Given a question, retrieve relevant passages",
)
res = await emb.embed(chunks)  # Embedded(vectors, outcome, stamp, reason, input_tokens)
q = await emb.embed([question], input_type="query")
```

Store `res.stamp` with every vector: two stamps are two spaces and never compare. A wrong count
or dimension is `invalid` (nothing to store), never a partial batch.

## Grounding

The number law (`grounding.numbers`) is the one rule every belt, grader and validator reads a
figure through: a number in the output must be one some input rendered, compared by value, digits
and spelled-out alike. `grounding.slots` makes it hold by construction for prose: the model writes
`{id}` under a grammar with no digit token, and code prints the value. See the
[slots guide](guides/slots.md) and ADRs [0002](adr/0002-grounded-prose-by-construction.md) to
[0005](adr/0005-character-whitelist-grammar.md).

## Testing

`llm_ports.testing` holds a double for every port (`StubGenerator`, `StubEmbedder`,
`StubReranker`, `RecordingTracer`, `StubCiter`, `StubPromptSource`, `StubDetector`), so a
consumer's tests run with no network. `testing.langfuse` (the `langfuse` extra) runs a real
Langfuse SDK client that exports to memory.
