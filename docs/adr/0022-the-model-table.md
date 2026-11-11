---
status: accepted
date: 2026-11-11
---

# The model table: schema 2 states what each model can do and who serves it, by name; providers and packs load through one strict plugin contract

Decided in the release after 1.22.0. Amends ADR-0018 (duplicate declarations, pack loading).

## Context and problem statement

A 1.x tier table maps a tier to a model id string and a timeout. Nothing in it says what a model
can do, so a consumer inferred capability from a model's name (a `-web` suffix), and a request
with images sent to a text-only model failed on the wire, after a trace had opened and the call
had been paid for. The thinking switch was spelled inside `Role.model_settings()` for one server
family (`extra_body.chat_template_kwargs`), so a model that spells it differently could not be
listed. Where a model is served, and with which key, lived in the consumer's wiring code, outside
the reviewed table.

The 1.22 pack loader (`load_packs`) shared a defect class with every plugin loader the estate had
reviewed: it never raised, so a pack that failed to import was a log line and the scheduler ran
without it; load order followed install order; an equal re-declaration with a different handler
replaced the held one silently; and a misspelled hook name loaded as a plugin that did nothing.
Providers by kind are the second entry-point group, and the third use of the estate's plugin
contract, so they and packs should share one loader that fails loudly.

## Decision

1. **`schema_version = 2`.** A table without the key is version 1 and loads to the same values as
   1.22.0 (pinned by a snapshot); a version 1 table that carries a version 2 section is refused,
   so a half-migrated file is not read as the old shape. Version 2 adds:
   - `[providers.<name>]`: a registered `kind`, and by NAME only a `base_url` key and a `secret`
     name. A value holding `://` is refused: URLs and keys stay out of a reviewed, fingerprinted
     table.
   - `[models.<key>]` (`ModelInfo`): the wire id, its provider, `supports` from a closed
     capability vocabulary, `thinking` style, and limits that are None when unknown, never a
     guessed number.
   - `[policies.<name>]`, carried read-only until the resilience row types them.
   - Per tier: `policy`, ordered `fallback` tier names and `fallback_on` failure kinds (default:
     overloaded, server, transport, timeout). Cycles, self-fallback and unknown names are refused.
   Every refusal is `TableInvalid(ValueError)` naming the dotted path.
2. **A fingerprint.** `TierTable.fingerprint()` is the sha256 (first 16 hex digits) of the parsed
   table's canonical JSON. Each loaded tier carries it, and `call.generate` stamps it on every
   result as `stamp["tiers"]`, so a graded result names the table that routed it.
3. **Capability before the wire.** `call.generate` derives what a request needs (`vision` for
   images, `web`, `json_schema` for structured output, with `json_object` standing in) and
   returns `request_fault` with an `unsupported: ...` reason when the tier's model does not state
   it, with zero wire requests. `TierTable.require(role, *caps)` proves at import that a role's
   tier and every tier it can fall back to state what the role needs. A schema 1 table states
   nothing and is not checked; one `llm_call_warning` per process says so.
4. **Provider detail lives in the adapter.** A schema 2 role sends `thinking: bool`; the
   `openai_compatible` factory spells it per `ModelInfo.thinking` (`unified`: the framework's own
   setting; `chat_template`: `extra_body.chat_template_kwargs.enable_thinking`; `none`: dropped,
   with one `llm_call_warning` per model when a role asked to think). Schema 1 roles keep the
   1.22 spelling.
5. **Providers by kind, built at boot.** `call.providers.build(table, urls=, secrets=, deps=,
   factories=)` resolves each used provider's names through the caller's mappings and raises at
   boot for an unknown kind, an unresolved URL key, or a secret that resolves to nothing or to a
   value no header can carry (whitespace, a control or a non-ASCII character: httpx quotes a
   refused header, value and all, in its error). It
   returns `Backends` (`generator_for(tier)`, `embedder_for(model_key)`, `probe_for(provider)`,
   and typed placeholders for the chat and fallback rows), after emitting `llm_backends_built`
   with the fingerprint and the plugin origins. Only `openai_compatible` is built; other kinds are
   seams until a consumer uses one.
6. **`Probe`.** A `ports.Probe` returns `Health(state, models, reason, latency_s)`. `unknown` means
   the probe itself failed (including a request it could not send), the credential was refused
   (401/403: auth, not liveness), the backend has no endpoint to judge, or a 200 body is not a
   model list; `down` means it could not be reached or answered with an error. A breaker opens on
   `down` only. A reason names an exception by its type, never its message. `adapters.http.HttpProbe`
   is the OpenAI-wire implementation (`GET /models`, 5 s deadline).
7. **One strict plugin contract for both groups** (`llm_ports.market`, `llm_ports.providers`):
   hook specs added before any plugin registers; built-ins registered first by explicit name;
   entry points loaded sorted by name; a pin (`None` all, `""` built-ins only, `"a,b"` exactly
   those, a missing pinned name raises); `check_pending` after each registration; any failure
   raises `PluginLoadError` naming the group, plugin and stage (`load_packs(strict=False)` keeps
   the 1.22 record-and-continue behaviour for a diagnostic); a duplicate key names both plugins;
   the loader freezes the registry when a strict load finishes; `plugins_loaded` reports names and origins. The same ten
   conformance tests run against both groups.
8. **The gate fails closed until read.** A `GateCache` that has never completed a read refuses
   every category with `gate_unread`; `fail_open=True` admits instead. Each judgement from
   never-read gates emits `market_gate_unread` with `serving` `closed` or `open`. A gate nobody
   refreshed is not a gate, and nothing refreshes it but the consumer, at boot and on a schedule.

## Considered options

- **A bundled capability database** (LiteLLM's model price/context file). Thousands of entries,
  megabytes, and stale for a self-hosted model; the consumer knows its ten models.
- **Capability booleans on a model class.** Cannot differ per model version or per deployment.
- **Inferring capability from the model name.** The defect this replaces.
- **URLs and keys in the table.** The table is reviewed in a pull request and fingerprinted; a
  value there leaks into both.
- **Detect an unknown provider kind at `load_tiers`.** Kinds arrive from plugins, which load after
  the table may already be parsed; the kind is checked at `build`, which is still boot.
- **Keep `load_packs` lenient by default.** A producer that silently failed to load is the failure
  the registry exists to prevent; the lenient mode stays as an opt-in for diagnostics.
- **A required `on_unreadable` argument on `GateCache`.** A required argument breaks 1.x callers;
  `fail_open=False` keeps the safe default without breaking a call site.

## Consequences

- A request a model cannot serve costs nothing and is named; a table edit that removes a needed
  capability fails the boot through `require`, not a call.
- Behaviour changes for 1.22 callers, each stated first in the changelog: `load_packs` raises by
  default; an equal re-declaration with a different spec object raises `DuplicateAction`; a
  `desired_ttl` needs `cache_table` and `age_col`; a never-read `GateCache` refuses; the
  `market` extra needs pluggy 1.7; a strict `load_packs` freezes the registry it loaded into;
  every result carries `stamp["tiers"]`. A version 1 table with a `[providers]` or `[policies]`
  section still loads to its 1.22 values (the sections are ignored, with a `llm_table_ignored`
  warning), since 1.x is additive.
- `TierTable.models` stays key to id string for 1.x; `ModelInfo` lives beside it in
  `model_info`.
- Fallback (the resilience row) and the chat port (the chat row) read `Tier.fallback`,
  `Tier.fallback_on`, `Tier.policy` and `Built.chat`, which are typed placeholders here.
