# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Why a design is the way it is lives in
[`docs/adr/`](https://github.com/mjdevaccount/llm-ports/tree/main/docs/adr).

## [Unreleased]

### Behaviour changes (read first)

- **`PydanticAIGenerator` now retries a failed wire request by default.** Its default policy is
  `WirePolicy.default()`: `rate_limited` 3 tries (0.5 s doubling to 8 s, with jitter, honouring
  the server's `retry-after-ms` / `Retry-After` up to 120 s), and `overloaded`, `server`,
  `timeout` and `transport` 2 tries. `quota`, `auth` and a request-side kind are never retried.
  1.x sent one request; `PydanticAIGenerator(policy=WirePolicy.none())` keeps that. Every
  retried request is an `Attempt` on the result (ADR-0021).
- **A call's timeout is now a wall-clock bound on the WHOLE call.** `generate` starts a
  `Deadline` from the role's timeout and ends the call at it, across the slot wait, every wire
  request, every backoff and every fallback tier; an expiry is `unavailable` with kind
  `timeout` and `phase="total"`. 1.x's timeout bounded each HTTP phase, which a slow trickle
  never tripped.
- **A tier table's `[policies.<name>]` is now validated at load.** A value a `RetryRule` cannot
  hold (an unknown kind such as `quota`, a missing `wait_max_s`, a negative wait) is
  `TableInvalid` naming its path. 1.23.0 stored any table.

### Added

- **Wire policy** (ADR-0021): `WirePolicy` (kind to `RetryRule`, plus `max_total_attempts`) and
  `RetryRule`, exported from `llm_ports`, `llm_ports.ports` and `llm_ports.call.resilience`. One
  retry loop (`adapters._retry`, tenacity underneath) is shared by the generator, the embedder
  and the reranker; the provider SDK's retries stay 0. A `Role.policy`, else the tier's declared
  `policy` (`Tier.wire_policy`), else the generator's default applies. `x-should-retry: false`
  ends the retry; a server hint above `retry_after_max_s` stops it with the hint kept.
- **Deadline**: `Deadline(total_s, attempt_s=None, first_event_s=None)` on
  `GenRequest.deadline`; `GenRequest.clocks` reads it (or `Deadline(total_s=timeout_s)`).
  `attempt_s` bounds one request (a retried `timeout`), `first_event_s` the wait for a stream's
  first event. `call.DEADLINE_GRACE_S` is how long after its deadline the facade cuts off a
  generator that ignores it.
- **Fallback**: `generate` walks a role's tier and its `fallback` chain; a failure whose kind is
  in the tier's `fallback_on` hands the call to the next tier with what is left of the deadline.
  `answered.tier` names who answered, `attempts` carries every earlier tier's failure (each
  `Attempt.tier` set), `stamp["fallback_depth"]` how far it went. `Hooks.generator_for` serves a
  fallback tier from another provider's generator. Events `llm_fallback`, `llm_deadline_expired`.
- **Limiter**: the `Limiter` port (`slot(group)`, yields the seconds waited) and the in-process
  `call.resilience.LocalLimiter(slots, default=None, max_wait_s=None)` with
  `LocalLimiter.for_models`. The slot is taken inside the deadline and kept across retries;
  `stamp["limiter_wait_s"]` reports the wait. A wait past `max_wait_s` is `rate_limited` with
  `provider_code="local_limiter"`. `GenRequest.concurrency_group` (the model's `group`, else its
  key) names the slot.
- `PydanticAIGenerator(policy=, limiter=)`, `.stats()` (`clients_built`, `clients_open`,
  `requests_sent`, `calls_in_flight`) and `.aclose()`. `openai_compatible(..., client=)` takes a
  caller's `httpx2.AsyncClient`; without one it builds ONE per event loop and reuses it (1.x
  opened a client per call). It now returns an `OpenAICompatible` (still a `model_for`).
- `Role.stream_internally` / `GenRequest.stream_internally`: the adapter requests a stream and
  assembles it, so a cancelled or timed-out call closes a connection the server can stop on.
- `PydanticAIEmbedder(policy=)`; `HttpReranker(policy=, on_event=)`; `Reranked.attempts`.
- `BuildDeps.policy`, `.limiter` and `.pricer` are typed (`WirePolicy`, `Limiter`, `Pricer`) and
  `OpenAICompatibleFactory` passes them, with `on_event`, to every generator and embedder it
  builds.
- `RetryPolicy.wire()`: the 1.x policy as a `WirePolicy`.

### Changed

- `PydanticAIEmbedder(retry=RetryPolicy(...))` runs through the shared loop (`RetryPolicy.wire()`);
  a server's `Retry-After` is now honoured up to `wait_max_s`. `RetryPolicy` is deprecated in
  favour of `WirePolicy`; passing both `retry=` and `policy=` is a `ValueError`.
- The `pydantic-ai` floor is 2.64 (`first_event_timeout`); the `http` extra now carries
  `tenacity>=9` (the reranker's retry loop).

## [1.23.0] - 2026-11-12

### Behaviour fixes (read first)

- **A reply the token ceiling cut is no longer `ok`.** A `finish_reason` of `length` is now
  `truncated` (kind `truncated`) on the success path too, even when the cut text parsed; the cut
  value is kept on the result. `Role.accept_truncated=True` (or `GenRequest.accept_truncated`)
  restores `ok`, with a `CallWarning` on the result (ADR-0020).
- **A caller's bug is no longer a model outage.** An exception raised by the caller's own code (a
  `KeyError` in a `validate` function, a bug in a `model_for`) is now `request_fault` with kind
  `internal`, never `unavailable`: it is not retried, failed over or counted against a backend.
  It is reported as the event `llm_call_internal_error` with its traceback (ADR-0020). Who raised
  decides, so a provider's fault stays `unavailable` as in 1.x: an exception raised inside a
  provider client while it read a response that arrived (a 200 with `choices: []`) is kind
  `malformed_response` (retryable); pydantic-ai's `UsageLimitExceeded`, an application's
  `RunCancelled` and `ConcurrencyLimitExceeded` (the agent's own queue) are `caller_limit` (not
  retryable); a `FallbackExceptionGroup` is its worst member. A caller's configuration is
  `internal` too, with no request sent: pydantic-ai's `UserError` (an output mode the model
  lacks), and an HTTP client's `InvalidURL` or `UnsupportedProtocol` (a base URL such as
  `http://[::1/v1`, `ftp://...` or one with no scheme), which 1.x reported as a retryable
  `unavailable`.

### Changed

- `adapters.pluggy.load_packs` raises `PackLoadError` (an `ImportError`) on the first pack that
  fails, naming the group, the pack and the stage (`import`, `register`, `hook`, `duplicate`,
  `pin`, `pending`). 1.22 recorded the failure and carried on, so a scheduler booted without a
  producer it was meant to run. `load_packs(..., strict=False)` keeps the old record-and-continue
  behaviour for a diagnostic. Entry points now load sorted by name, not in install order, and a
  misspelled hook name fails at `pending` instead of loading a plugin that does nothing (ADR-0022).
- A strict `load_packs` freezes the `ActionRegistry` it loaded into when it finishes, so a
  declaration after the composition root raises `RuntimeError` (declare the process's own actions
  first, or pass them as `builtins`); calling `load_packs` on an already frozen registry raises
  before loading anything. `strict=False` leaves the registry open.
- `market_pack_failed` keeps 1.22's `stage` vocabulary (`PackLoad.stage`: `import`, `validate`,
  `pack`, `join`, plus `pin`) and adds `plugin_stage` in `PluginLoadError.stage`'s vocabulary.
- `ActionRegistry`: re-declaring a held `feature.action` with any spec object other than the
  identical one raises `DuplicateAction` (a `ValueError`) naming both origins, even when the
  metadata is equal. 1.22 replaced the held spec, so an equal declaration with a different handler
  swapped the handler in silently. Re-adding the identical object stays a no-op.
- A `desired_ttl` needs both `cache_table` and `age_col` (the output row its age is measured
  against); without them nothing can judge the TTL, so the declaration is refused at import.
- `GateCache` fails closed until its first successful read: `check` (and `admit`) return
  `gate_unread: ...` for every category, so a boot that forgot `refresh()` refuses instead of
  admitting ungated. `GateCache(..., fail_open=True)` admits instead. Either way, each judgement
  from never-read gates emits `market_gate_unread` (WARNING) with `serving: "closed"` or
  `"open"`. A failed boot read keeps the gate closed until a later `refresh()` succeeds, and the
  cache never reads on its own: call `await gates.refresh()` at boot and on a schedule.
- The `market` extra needs `pluggy>=1.7` (was `>=1.5`).
- `call.providers.build` refuses a secret whose value holds whitespace, a control or a non-ASCII
  character (`TableInvalid` at `providers.<name>.secret`, naming the secret, never its value): no
  header can carry it, and httpx quotes a refused header, value and all, in its error and its
  DEBUG trace. A trailing newline from a secret file is the usual cause.
- `adapters.pydantic_ai.openai_compatible` and `openai_compatible_embeddings` raise `ValueError`
  at construction for an `api_key` that holds whitespace, a control or a non-ASCII character (the
  message never shows the value). 1.22 accepted it and every call then failed with an error, and a
  DEBUG log line, quoting `Bearer <key>`. `build`, the `openai_compatible` kind, these two and
  `HttpProbe` share one validator.
- Every `call.generate` result carries `stamp["tiers"]`, the fingerprint of the tier table that
  routed it (absent only for a hand-built `Tier`).
- **A prompt pack now fails at load where it used to load** (ADR-0026):
  - a template that `{% include %}`s, `{% import %}`s or `{% extends %}` a file not listed in
    `templates` or the new `includes`;
  - a dynamic include whose target cannot be read statically;
  - a pack file holding a carriage return (CRLF or a lone CR);
  - a `graders` list naming the same grader twice;
  - an `includes` entry that is not a relative POSIX path inside the pack (`..`, a leading `/`, a
    drive, a `\`, an empty or hidden segment, or `pack.yaml`), or that resolves outside it.

  Each raises `PromptPackInvalid` naming the file or field. Templates render from exactly the
  manifest's files; nothing else in the pack directory is read. A partial in a subdirectory or
  with any extension (`partials/h.j2`, `rubric.md`) loads once it is listed in `includes`.
- `PackManifest.golden` is resolved **against the pack directory** (it was documented as
  repo-relative in 1.22.0, which nothing read) and must stay inside it.
- `call.graders.GRADERS` is a read-only mapping (`types.MappingProxyType`): assigning into it raises
  `TypeError`. **Consumers must** stop writing into it (`monkeypatch.setitem(GRADERS, ...)` now
  fails) and pass a table instead: `{**GRADERS, "mine": fn}` to `score_all` / `score_one` /
  `board_graders` / `call.Hooks(graders=...)` / `prompts.resolve(registry=)` /
  `evals.golden.resolve_pack(registry=)`. A test that needs the default table changed replaces the
  module global: `monkeypatch.setattr(llm_ports.call.graders, "GRADERS", {**GRADERS, "mine": fn})`
  (`score_one` reads it at call time).
- `Pack.stamp()` gains the key `identity`. The `sha` key, `stamp_str()` and `content_sha` are
  unchanged for every pack, byte for byte.
- `PackManifest.graders` accepts bare grader names (resolved through a registry) as well as dotted
  paths.
- `LangfusePromptSource` re-pushes a mirror whose `config.identity` differs from the seed's even
  when its content sha matches. `get`'s default `label` is unchanged (`production`).
- The `pydantic-ai` and `evals` extras need pydantic-ai-slim / pydantic-evals **2.62** or newer
  (was 2.57): 2.62 is the first release with every `ModelAPIError` category class the failure
  table reads (`ModelAuthenticationError`, `ModelNotFoundError` and `ModelRequestRejectedError`
  arrived in 2.62). Verified against 2.62.0 and 2.66.0.
- `PydanticAIEmbedder`'s retry predicate is `Failure.retryable`: `auth`, `not_found`, `quota` and
  `internal` are no longer retried.
- `Generated.model` is documented as the requested id (deprecated in favour of `answered`);
  `usage` and `finish_reason` are derived from `usage_v2` and `finish`. Their 1.22.0 values are
  pinned by a golden test.

### Added

- Tier table `schema_version = 2` (ADR-0022): `[providers.<name>]` (a registered `kind`, and by
  name only a `base_url` key and a `secret` name), `[models.<key>]` (`call.models.ModelInfo`: wire
  id, provider, `supports` from the closed `Capability` vocabulary, `thinking` style, limits that
  are None when unknown, `slots`, `group`, `self_hosted`, `usd_per_request` as a `Decimal`),
  `[policies.<name>]` (read-only), and per tier `policy`, `fallback` and `fallback_on`. Every
  refusal is `TableInvalid(ValueError)` naming the dotted path. A table without `schema_version`
  loads to the same values as 1.22.0, a `[providers]` or `[policies]` section in it included
  (ignored, as 1.22 ignored it, and named once per load by a `llm_table_ignored` WARNING).
  `Tier.policy` is read-only and hashable (nested tables read as mappings, arrays as tuples), so a
  `Tier` hashes as it did in 1.22. New `Tier` fields (`info`, `policy`, `fallback`,
  `fallback_on`, `table_fingerprint`) and `TierTable` fields (`model_info`, `providers`,
  `policies`, `schema_version`) all default, so 1.x construction is unchanged.
- `TierTable.fingerprint()` and `TierTable.require(role, *caps)`, which proves at import that a
  role's tier and every fallback tier state the capabilities it needs.
- The capability pre-check in `call.generate`: a request whose model does not state `vision`,
  `web` or structured output (`json_schema`, or `json_object` in its place) ends `request_fault`
  with an `unsupported: ...` reason and no wire request. A schema 1 table is not checked, and one
  `llm_call_warning` per process says so; a role that overrides its tier's model id is named once.
- On a schema 2 tier, `Role.model_settings()` sends a provider-neutral `thinking: bool`.
- `call.providers`: `ProviderFactory`, `ProviderRegistry` (a duplicate kind raises naming both
  plugins; frozen after load), `build(table, urls=, secrets=, deps=, factories=)`, which raises at
  boot for an unknown kind, an unresolved URL key or a secret that resolves to nothing, and returns
  `Backends` (`generator_for`, `chat_for`, `embedder_for`, `reranker_for`, `probe_for`, `aclose`).
  `build` emits `llm_backends_built` with the fingerprint and the plugin origins; no secret value
  is logged. `BuildDeps` and `Built` hold typed placeholders (`policy`, `limiter`, `pricer`,
  `chat`) for the resilience, result-contract and chat rows.
- `adapters.pydantic_ai.OpenAICompatibleFactory`, the `openai_compatible` kind (generator,
  embedder per model that states `embed`, probe), and `translate_settings`, which spells
  `thinking` per `ThinkingStyle` (`unified`, `chat_template`, `none`; `none` warns once per model
  as `llm_call_warning` when a role asked to think).
- `ports.Probe`, `ports.Health`, `ports.HealthState`; `adapters.http.HttpProbe` (`GET` one URL
  under a 5 s deadline: `up` with the listed model ids, `down` when unreachable or an error
  status, `unknown` when the probe itself failed or could not send its request, the credential
  was refused (401/403, reason `auth: HTTP <status>`), the endpoint is missing (404/405), or a
  200 body is not a model list; never raises). A `Health.reason` names an exception by its type
  only, never its message, and a secret no header can carry is refused before any request.
- `adapters.pluggy`: `load_providers` for the second entry-point group `llm_ports.providers`
  (hook `register_providers(register)`, marker `provider_hookimpl`, `builtin(name, *factories)`),
  on the same loader as packs; `load_packs(builtins=, pin=, strict=)`; `PluginLoadError`,
  `PackLoadError`, `PluginStage`; `PackLoad.origin`; `packs_loaded(loads)` returning a
  `call.providers.Loaded`; the event `plugins_loaded` (group, names, origins); the conventional
  pin setting names `PACKS_PIN_ENV` and `PROVIDERS_PIN_ENV` (the library reads no environment).
- `market`: `DuplicateAction`, `ActionRegistry(sources=...)` (a `source` id outside it is refused),
  `ActionRegistry.freeze()`, `.frozen`, `.origin(feature, action)`, `add(..., origin=)` and
  `include(..., origin=)`; `KillSwitch(..., registry=)` checks the paused set against the
  registry on every parse; `GATE_UNREAD`; `LastKnown.known`.
- `tests/contract/api_surface.json`: the public `__all__` of every module, snapshotted; a name
  that disappears fails the suite.
- `prompts.byte_sha(parts)`: sha256 over named byte strings, length-prefixed so a re-split, rename
  or reorder never collides. Pinned vectors freeze it.
- `Pack.identity`: `byte_sha` over the manifest's bytes and every file of the closed template set,
  by sorted path. `Pack.files()` lists that set; `Pack.seed()` builds the repo `Seed` a managed
  mirror copies, carrying `content_sha`, `identity` and the manifest's label in its config.
- `PackManifest.includes`, `PackManifest.schema_version` (only `prompts.SCHEMA_VERSION` = 1 loads),
  and `label`, the neutral alias of `langfuse_label` (YAML key and read-only property).
- `LangfusePromptSource(..., label_from_seed=True)` opts in to serving and mirroring under a seed's
  `config["label"]` (the manifest label `Pack.seed()` carries) instead of `get`'s `label`.
- `prompts.resolve(pack, registry=, import_roots=)` and `Pack.resolved(...)`, returning
  `ResolvedPack`: the `output_schema` type imported only from the caller's module roots, each grader
  looked up in a `GraderRegistry` (or a plain mapping), the golden path kept inside the pack
  directory, and `output_schema_sha`. Every miss is `PromptPackInvalid` naming the field.
  `prompts.schema_sha` hashes an output type's canonical JSON schema.
- `prompts.scan(root, registry=, import_roots=)` reports resolution failures beside load failures,
  and `prompts.registry(root, registry=, import_roots=)` raises on the first one, so a mistyped
  grader fails when the registry is built. Without the arguments both behave as in 1.22.0.
  `prompts.served_shas(root, key="identity")` serves identities for `unevaluated`.
- `evals.golden.fire_pack(resolved, role=, generator_for=, ...)`: reads the pack's golden set
  (`evals.golden.read_golden`: a JSON list, an object with `items`, or JSON lines), fires it, grades
  it on the pack's graders and returns a `GoldenRun` stamped with `identity`, `output_schema_sha`,
  the role's tier and model, and the table fingerprint the caller passes. It never raises for a
  golden-set defect: no golden set, no graders, a missing file (the reason names the manifest value
  and the resolved path), an unreadable or empty file, or an item whose slots do not render each
  return `could_not_run` with the reason before anything fires, and emit a `golden_run` WARNING.
  `evals.golden.resolve_pack` resolves with `call.graders.GRADERS` behind the caller's registry.
  The `golden_item` event gains `identity`.
- `Outcome` gains a seventh member, `refused`: the model or its provider declined (a
  `content_filter` finish, an explicit `refusal`, pydantic-ai's `ContentFilterError`).
- `ports.Failure` and `ports.failure(...)`: the failure's `kind` (18 kinds, `FailureKind`), with
  `outcome` and `retryable` derived from it (`OUTCOME_FOR`, `RETRYABLE`), and `retry_after_s`,
  `status`, `provider_code`, `phase`, `in_stream`, `reason`. On `Generated`, `Embedded` and
  `Reranked` as `failure`; the 1.x `reason` is derived from it (ADR-0020).
- The result contract (ADR-0019): `Usage` (inclusive buckets: `input`, `output`, `cache_read`,
  `cache_write`, `reasoning`, `requests`, `unmeasured`, `details`; unknown is None), `Cost`
  (`usd` as a `Decimal` or None, `source`, `priced_model`, `reason`), `Answered` (who answered:
  the body's `model`, the gateway's `deployment`, `response_id`), `Finish` (`unified`, `raw`),
  `CallWarning`, `Attempt`. `Generated` gains `usage_v2` (summed over every request of the call),
  `cost`, `finish`, `answered`, `attempts`, `warnings`, `failure`; `Embedded` gains `usage_v2`,
  `cost`, `answered`, `failure`; `Reranked` gains `answered`, `failure`.
- The `Pricer` and `GatewayReader` ports, and `PydanticAIGenerator(pricer=, gateway=, extra=,
  on_event=)`, `PydanticAIEmbedder(pricer=, gateway=, on_event=)`. Cost precedence: the gateway's
  cost header, then the pricer, then unknown. A cost never fails a call: a refused cost header, a
  custom `GatewayReader` that raises, or a sum that overflows leaves the gateway cost unknown and
  emits `llm_gateway_header_invalid`.
- `adapters.genai_prices.GenaiPricesPricer` (new extra `[prices]`, `genai-prices>=0.2.3`, also in
  `[all]`): prices a call from genai-prices' dated table, with per-request `overrides` for a
  self-hosted model; an unknown model is `Cost(None, "unknown", reason="unknown_model")`.
- `adapters.litellm_proxy` (no extra, no vendor import): reads a LiteLLM proxy's cost, deployment
  and attempt headers (a `GatewayReader`), and `classify_gateway` names a spent budget or quota 429
  `quota` (never retryable) while a parallel-limit 429 stays `rate_limited` with its wait. A cost
  header outside 0 or [`COST_MIN_USD`, `COST_MAX_USD`) = [1e-12, 1e7) dollars is implausible and
  treated as absent; a signed zero is 0.
- The shared classification table `adapters._classify.classify_core` (exported from
  `adapters.pydantic_ai` with `WireFacts`, `Classifier`, `kind_for_status`, `retry_after_s`,
  `is_context_overflow`, `raised_in_provider`, `PROVIDER_MODULES`, `CALLER_CONFIG_ERRORS`, `worst`,
  `SEVERITY`), used by
  the generator, the embedder and the HTTP reranker; and
  `adapters.pydantic_ai.classify_failure`, `openai_finish`, `OPENAI_FINISH`, `parse_usage`,
  `WireReply`.
- Events: `llm_cost_unknown` (once per model per process), `llm_usage_unmeasured`,
  `llm_pricer_failed`, `llm_gateway_header_invalid`, `llm_call_internal_error`. The
  `llm_call_failed` event carries `kind`; the call's observation metadata carries the kind, the
  cost and its source, who answered, the unified finish, the attempt count and every usage bucket.
- `testing.Reply.failed(kind, **detail)`: a `StubGenerator` reply function returns it (or a
  `ports.failure(...)`) to fail a call with that kind. `testing.Reply` is now a `Protocol` with the
  same call shape, so any function that was a `Reply` still is. The stubs fill `failure` and
  `answered`; `StubGenerator(model=)` (default `testing.STUB_MODEL`) is the model it reports as
  having answered, never the request's id.

### Fixed

- `adapters.langfuse.tracer.usage_details` no longer books an unreported side as 0: only the
  measured sides are sent.


## [1.22.0] - 2026-11-11

### Added

- `llm_ports.market`, the declaration side of a capacity-driven work market (ADR-0018):
  `ActionRegistry` with the `action(...)` decorator and `ActionSpec`, which refuse at import an
  action with none of `desired_ttl`, `cadence` or `max_silence_h` (and a malformed interval, an
  unknown silence clock, a negative lane, a non-positive weight or cost); `KillSwitch`, the
  operator's paused `feature.action` set read from a setting the caller names; `GateCache` and
  `LastKnown`, cached control reads that serve the last known value on a failed read, with
  `GateCache.admit` usable as `call.Hooks.admit`; `ActionMirror` and `sync_all`, the store copy an
  operator tunes. Every component reports through an injected `on_event` hook.
- `adapters.pluggy.load_packs` (extra `[market]`): action packs from other distributions, listed
  under the `llm_ports.market` entry-point group, validated by pluggy and joined all or nothing;
  each entry point ends `ok`, `failed` (with its stage) or `blocked`.

## [1.21.0] - 2026-11-09

### Fixed

- `adapters.gepa.optimize` no longer spins on a dead reflection model. GEPA logs a failed
  reflection and starts the next iteration, so `raise_on_exception` never sees it, and a caller
  whose evaluator caches spends no budget: a reflection model out of daily quota was called every
  few seconds for hours. `max_reflection_failures` (default 5) consecutive failed `reflect` calls
  now stop the search as `Unavailable` naming the last error, through GEPA's own `stop_callbacks`.

### Added

- `optimize(timeout_s=...)`: a wall-clock bound (GEPA's `TimeoutStopCondition`);
  `Optimized.stopped` is `"timeout"` when it ended the search, else `"budget"`.

### Changed

- Releasing is the version-bump PR alone: `tag-version.yml` (ci-kit's `ci-version-tag`) tags
  `vX.Y.Z` at the bump commit when it merges and dispatches `release.yml` on the tag; a daily run
  re-dispatches a tag whose release never ran and fails when a release left its GitHub release or
  PyPI files missing. `release.yml` and `ci.yml` read `github.ref_type` instead of the event, so a
  dispatched tag run is a release, not a dev upload, and the docs site deploys only from the newest
  release tag.

## [1.20.0] - 2026-11-09

### Added

- `adapters.gepa` (extra `[gepa]`, in `all`): `optimize(seed, train, val, evaluate=, reflect=)`
  runs GEPA reflective prompt evolution over a caller's batch evaluator (`Rollout(output, score,
  feedback)` per item) and reflection callable, and returns `Optimized | Insufficient |
  Unavailable`. `Optimized.improved` compares the best candidate to the seed on the validation
  items. GEPA swallows a failed reflection, so `optimize` counts them: every call failing is
  `Unavailable`, some failing is `Optimized.reflection_failures`. GEPA's log lines are kept on the
  outcome, never printed. `RolloutAdapter` is the `GEPAAdapter` underneath.
- `adapters.optuna.dashboard_app(storage)` / `serve_dashboard(storage, host=, port=)` (extra
  `[optuna-dashboard]`): optuna-dashboard over any Optuna storage, a `SqlJournal` table included,
  on demand and bound to loopback by default.

## [1.19.0] - 2026-11-08

### Added

- `forecast.skill`: `brier_skill(obs, fct, ref, clusters=None)`, a voice's Brier skill over a
  per-event REFERENCE forecast (a symbol's base rate, a prior) on the pairs both scored, with a
  seeded cluster-bootstrap interval (events sharing a day's tape resample together). `Score`'s
  `brier_skill_score` stays skill over climatology. Returns a `Skill` (exported from
  `llm_ports.forecast`). Standard library only, like the rest of `forecast`.

### Fixed

- `release.yml`: the PyPI upload uses `skip-existing`, and a follow-up step fails the job unless
  PyPI lists every built file. v1.18.0 shipped its wheel but not its sdist (PyPI refused the
  sdist's attestation with "Failed to refresh TUF"), and a rerun could not recover because it
  failed first on the wheel that had already uploaded. Now a rerun uploads only the missing files.

## [1.18.0] - 2026-11-08

### Added

- `forecast.score`: `mce(curve)`, the largest bin gap of a reliability curve, and
  `decompose(curve)`, Murphy's decomposition of the Brier score into `reliability`,
  `resolution` and `uncertainty` (a `Decomposition`, exported from `llm_ports.forecast`). It is
  exact on a curve over cells that each forecast one rate. Consumers that hand-rolled the
  decomposition beside `reliability()` and `ece()` can now take all three from the core.

## [1.17.0] - 2026-11-08

### Added

- `forecast.platt`: `PlattCalibrator`, a `Calibrator` that fits `sigmoid(a * logit(p) + b)` by
  ridge-stabilised Newton steps with the standard library, and `Platt`, the fitted map (applied
  and stored as JSON without a fitting library). A fitted slope `a <= 0` inverts the voice's
  ranking, so the fit is refused (`insufficient_.reason` starts `inverted slope`) and the
  values are kept in `fitted_`.

## [1.16.0] - 2026-11-08

### Added

- `evals.golden`: a golden item may carry `user`, and then fires as a system and user pair (the
  rendered pack as the system prompt, `user` as the prompt), the shape of a producer whose fixed
  instructions ride as the system for the prefix cache. `slots` is optional for a pack with none.
  `fire_one`'s result names the `system` it sent (empty for a slotted item, which still sends
  none, so a consumer facade with no `system` keyword keeps working).

## [1.15.0] - 2026-11-08

Seams for bigger ML engines (ADR-0016): every addition is optional, so no consumer changes.

### Added

- `learn.Written.ref` and `learn.LedgerRow.ref`: the id a backend gave what it wrote (not part of
  equality). `MemoryRunLedger` and `JsonlRunLedger` mint a UUID per row and read it back, so
  `Run.recorded.ref` links a run to whatever it produced.
- `learn.Fit.artifact_uri`: where a model too big for `model` lives; `loop.run` records it in the
  row as `learn_artifact_uri` (reserved), only when set.
- `learn.ModelRegistry` port (`register`, `set_alias`, `by_alias`, `get`; versions from 1 per
  name, aliases move) with `ModelVersion` and `VersionRead`, and `learn.registry`:
  `MemoryModelRegistry` and `JsonlModelRegistry` (an append-only event log). An `artifact_uri`
  or `run_ref` of the wrong type, or a `version` that is not an `int` (`Written.ref`'s string
  passed back, a `bool`), is a `request_fault`, never a raise.
- `testing.conformance`: `check_run_ledger`, `check_example_store`, `check_model_registry` and
  their broken-storage checks (`check_unavailable_run_ledger`, `check_unreadable_example_store`,
  `check_unavailable_model_registry`), raising `ContractViolation` naming the broken invariant.
  Every in-repo backend passes them.
- `adapters.mlflow` (new extra `[mlflow]`: `mlflow-skinny`, SQLAlchemy, Alembic):
  `MlflowRunLedger` (one MLflow run per row; scalar params and numeric metrics logged, the whole
  row as JSON split over run tags, read back from the search result with no artifact store) and
  `MlflowModelRegistry` (MLflow model versions and aliases; a version with no `artifact_uri` gets
  a backing run as its source; a `runs:/<id>/...` URI passes its run id, as a server requires).
  The ledger matches a row's own kind, so prefixes whose experiment names collide never
  cross-read, and refuses a subject over 8000 characters (MLflow would cut its tag). Both pass the conformance suites against a real `mlflow server`
  and a local SQLite store, write nothing to stdout or stderr, and are safe to share between
  threads. Full `mlflow` is a dev dependency only, for the server tests.
- `learn.Run.as_dict()` (a run as plain fields, for an admin endpoint's response) and
  `learn.failed(run)`: red when nothing was measured (`could_not_run`), the ledger row did not
  land, or `apply` raised; `fail` and `insufficient` stay green.
- Guide `docs/guides/ml-engines.md` and example `examples/ml_engine_seams.py`.

## [1.14.0] - 2026-11-08

### Added

- `adapters.optuna.SqlJournal` and `journal_storage(engine, table, create=)`: an Optuna journal
  backend on one SQL table `(n bigint primary key, log text)` the caller owns, schema-qualified in
  every statement. Appends are gapless across processes (the primary key fails a racing append,
  which retries from the new end). For PostgreSQL behind a transaction-mode pooler, where Optuna's
  RDB storage cannot be pinned to a schema (its tables follow the session search path and its
  Alembic check reads the default schema).

## [1.13.0] - 2026-11-08

### Added

- `forecast.walk`: `Event` (`x`, `forecast_time`, `resolve_time`, `resolution`) and
  `walk_forward`, a combiner replayed in forecast-time order that hears each outcome only at its
  resolve time and returns one probability (None = abstained) per event; `holdout_start` (the
  time split) and `compare` (several combiners scored on the same resolved, answered events).
- `forecast.PriorPool`: wraps any `Combiner` with a prior voice; shrinks the combined log-odds
  toward the prior's (`shrink`), extremizes (`alpha`), and answers with the prior while the inner
  combiner abstains.
- `forecast.BestVoice`: the best-single-voice baseline, online (lowest running Brier score among
  the voices that spoke, once each has `min_n` outcomes).
- `adapters.optuna.tune` (extra `[optuna]`, also in `forecast`): a seeded TPE search over a
  combiner's parameters whose objective is the walk-forward Brier score on the train split
  (abstention scored as 0.5); the best parameters are replayed over every event and scored on the
  holdout only. Returns `Tuned`, `Insufficient` or `Unavailable`, never raises; the study goes to
  any Optuna storage. Rationale: ADR-0015.

## [1.12.0] - 2026-11-08

### Added

- `LangfuseExampleStore.put` relays a person's label: a `human` example with `by` naming the
  person is written as `label-<role>` against the role's categorical score config (created when
  missing), keyed by the person, and the reply's annotation-queue item is marked `COMPLETED`
  (`labels.COMPLETED`). A consumer's own label surface feeds the same store, `coverage` and
  `agreement` as the annotation UI. Rationale: ADR-0014.

### Changed

- `put` refuses a `human` label with an empty `by` (was: every `human` label) as a
  `request_fault`; an `outcome` label is still refused.

## [1.11.0] - 2026-11-07

### Added

- `llm_ports.forecast`, standard library only: `Forecast` and `Resolved` (Metaculus / Good
  Judgment, schema.org `Claim` / `ClaimReview` and W3C PROV field names; `resolution` is
  `yes` / `no` / `annulled`); `score(obs, fct)` (`brier_score`, `log_score`, `brier_skill_score`,
  `overconfidence`; scoringrules' names and argument order), `reliability` and `ece`; the
  `Calibrator` port (the scikit-learn estimator protocol: `fit(X, y)`, `predict_proba(X)`,
  `get_params` / `set_params`) with `Insufficient` and `PiecewiseLinear` (a fitted monotone map,
  JSON round-trip); `HedgeMap`, a scikit-learn estimator for the probability each source's hedge
  word has earned (beta-binomial shrinkage toward the word's pooled rate); the `Combiner` port
  (River's `learn_one(x, y)` / `predict_proba_one(x)`) and `LogitPool` (equal-weight log-odds
  pool, optional extremizing). A `forecast-standalone` import contract keeps it liftable.
  Rationale: ADR-0013.
- Adapters, one extra each and grouped as `forecast` (not in `all`): `adapters.scoringrules`
  (`brier_score`, `interval_score`, `crps_normal`), `adapters.sklearn.IsotonicCalibrator`
  (optional `sample_weight`), `adapters.venn_abers.VennAbersCalibrator` (small-n, with a
  `(p0, p1)` bracket from `predict_interval`) and `adapters.river.RiverLogitPool` (online
  logistic logit pool with silent-voice flags).
- The forecasts guide and `examples/forecasts.py`.

### Changed

- The coverage floor rises to 97.9%.

## [1.10.0] - 2026-11-07

### Added

- `llm_ports.evals.calibration.trust` and `Trust`: whether a judge's verdicts may act, from its
  agreement with human labels (at least `MIN_LABELLED` pairs, kappa or its interval's lower end at
  `MIN_KAPPA`) and its seeded-bad replay. Never raises; a judge with no labels, or a replay row it
  cannot read, is untrusted.
  Rationale: ADR-0012.
- `llm_ports.learn.stats.kappa_ci` (Fleiss-Cohen-Everitt 95% interval, `Z95`) and
  `Agreement.kappa_low` / `kappa_high`, filled by `examples.agreement`.
- `llm_ports.learn.examples.pairs`: per topic, the `(measured, reference)` pairs behind
  `agreement`, optionally for one labeller (`by`, one judge model), so a caller pools them.
- The trace port's `score` takes an optional `metadata`, stored on the score (Langfuse adapter,
  `PortObservation`, `RecordingTracer.score_metadata`), so a judge label records its model.
- The learning-loops guide maps predict / resolve / score / consume onto the loop stages and
  shows a judge earning trust; `examples/learning_loop.py` runs the consume step.

## [1.9.1] - 2026-11-07

### Fixed

- `llm_ports.adapters.langfuse.labels`: label score names are `label-<role>`, `label.model-<role>`
  and `label.judge-<role>`. The `:` spelling is not a valid Langfuse score-config name, so the
  first `enqueue` for any role failed with a 400 and created no queue. Reads accept both spellings,
  so scores already written under `label.judge:<role>` stay readable.
- A role whose label name Langfuse would refuse is now a `request_fault` from `enqueue` and `put`
  before any write (new `CONFIG_NAME`), and an `unavailable` reason carries the server's own
  validation message instead of only the status code.
- `examples` splits its score-name filter into `NAMES_PER_READ` names per request, so the query
  string stays bounded as roles grow. Rationale: ADR-0011.

## [1.9.0] - 2026-11-07

### Added

- `llm_ports.adapters.langfuse.experiments.seed` and its result `Seeded`: write a golden set into a
  Langfuse dataset, creating it when missing. An item's id is the caller's or `input_id(input)` (a
  SHA-256 of the canonical JSON), and Langfuse upserts by id, so a re-seed rewrites rather than
  duplicates. A failed write is an `invalid` outcome naming the shortfall; an unreadable dataset is
  `unavailable`. Never raises.

## [1.8.0] - 2026-11-07

### Added

- Two built-in call graders in `llm_ports.call.graders`, for research (web-fetching) tiers:
  `cites_url` counts the distinct http(s) URLs in the reply's string fields and passes at one or
  more; `no_refusal` counts stock ungrounded-refusal phrases ("I don't have real-time access",
  "as of my training", "please consult ...") and passes at zero. Both are in `GRADERS`, so a role
  names them by string.
- `WEB_ONLY_GRADERS` (`{"cites_url"}`): `graders_for_inputs` drops them for a call without
  `web=True`, the mirror of `TEXT_ONLY_GRADERS`, so a text-only call is never failed for citing no
  pages.

## [1.7.0] - 2026-11-07

### Added

- `llm_ports.evals.labels.LabelQueue.queued(topic)` and its result type `Queued` (`pending`,
  `done`, `seen`, a named outcome): what a topic's review queue already holds, so a scheduled draw
  can skip it and bound the backlog. `LangfuseExampleStore.queued` reads the role's annotation
  queue items (paged); a role with no queue reads empty and creates nothing.

### Changed

- `LangfuseExampleStore.enqueue` skips a reply the queue already holds (pending or done) and
  returns how many were newly queued, so re-running the same deterministic draw adds no duplicate
  items. A custom `LabelQueue` implementation now also needs `queued`.

## [1.6.0] - 2026-11-07

### Added

- `llm_ports.evals.labels`: labelled model replies on the learning-loop core. A label is a
  `learn.Example` on topic `llm:<role>` (`topic`, `role_of`); `draw` picks which unlabelled
  production replies to queue for a human, closing each role's `learn.examples.coverage` gap and
  stratified by the judge's verdict; `label_of` turns any `Judge`'s verdict into a `model` or
  `judge` example (never `human`); `LabelQueue` is the port a review surface implements.
- `llm_ports.adapters.langfuse.labels.LangfuseExampleStore`: a `learn.WritableExampleStore` and a
  `LabelQueue` over Langfuse scores (`label:<role>` human, given in the annotation UI;
  `label.model:<role>` and `label.judge:<role>` written by `put` under a stable id) and one
  annotation queue per role, created on first use.
- `llm_ports.adapters.langfuse.experiments`: `run` and `record_rows` record a graded evaluation as
  a Langfuse dataset experiment; a run that graded fewer items than it was handed is `invalid`.

## [1.5.0] - 2026-11-07

### Added

- `llm_ports.learn`: the learning-loop core, standard library only (ADR-0010). `Clause` (with a
  `role`: `bar`, `floor`, `veto`), `Verdict` (`pass | fail | insufficient | could_not_run`, every
  clause kept, `blocking`), `Example` (a label from a `human`, `model`, `judge` or `outcome`) and
  `LedgerRow`. `gate`: `at_least`, `at_most`, `equals`, `check`, `decide(gate, clauses,
  unevaluated=)`, `could_not_run`. `loop.run(loop, ledger)` drives a `Loop`'s `evidence`, `fit`,
  `clauses` and `apply` and records exactly one row per run under `learn_verdict`, never raising;
  `loop.record` records a gate run outside a loop. `examples`: `coverage`, `agreement`,
  `few_shot`. `stats`: `cohen_kappa`, `gwet_ac1`, `bh_fdr` (Benjamini-Yekutieli by default).
  `stores`: `MemoryRunLedger`, `JsonlRunLedger`, `MemoryExampleStore`, `JsonlExampleStore`,
  `union`. Guide `docs/guides/learning-loops.md`, example `examples/learning_loop.py`.

### Changed

- `evals.calibration.cohen_kappa` and `gwet_ac1` now live in `learn.stats` and are re-exported
  unchanged.

## [1.4.0] - 2026-11-06

### Added

- `llm_ports.evals.judge.JudgeLane` and `Sample`: run a sampled judge as background tasks off the
  caller's await path. `start(role, prompt, result, mod=, run=, eligible=, seen=)` draws the reply
  (`should_sample`), reads what the call saw, and schedules the consumer's `run(sample)` in a fresh
  `contextvars.Context`, holding a strong reference until it finishes; `pending()` and `drain()`
  serve shutdown and tests. Events `<name>_sampled|_seen_failed|_sampler_failed` go to `on_event`
  (the stdlib logger by default). Merged into `evals.judge`: there is still one judge module.
- `llm_ports.evals.golden`: golden-set evaluation of a prompt pack. `fire(items, role=, pack=,
  source=, output=, generator_for=, generate=)` renders each item's slots through the pack and
  fires it through the call path (never raises for a model or wire failure); `rows_of`,
  `monitors`, `grade(rows, role=, extra=)` (the role's call graders plus extra row graders on one
  `Evaluator` board) and `citations_resolve(cited, known)`.
- `llm_ports.evals.calibration`: calibrate a judge in both directions. `cohen_kappa` and
  `gwet_ac1` over binary `(judge, human)` pairs (None where undefined); `seed_placeholder` and
  `seed_figure` build seeded-bad replies a judge must fail; `tally` and `verdict` read a replay of
  good samples beside their seeded copies (exit codes `CALIBRATED`, `FINDING`, `BLIND`).
- `llm_ports.adapters.pydantic_ai.replay.recording(make)` and `recorder(make)`: a generator plus
  the list (or dict) its raw attempts land in, so a golden run records every attempt.

## [1.3.0] - 2026-11-06

### Added

- `llm_ports.grounding.numbers.shorten(text, width, placeholder="…")`: cut a text for display on
  a word boundary, never inside a digit run. When the first word alone is too long the hard cut
  steps back to the start of any number it would split, so a shortened "25bp" can never read
  "2…". Every digit run in the result is a whole digit run of the input. Pure, no I/O.

## [1.2.0] - 2026-11-06

### Added

- `llm_ports.call.resilience`: `RetryPolicy` (tries per request, backoff start and cap) and
  `FailureBudget` (a long run's per-item outcomes; `exhausted` after `max_consecutive` failures in
  a row, or a failure rate above `max_rate` once `min_attempts` items were tried). Pure, no I/O.
- `PydanticAIEmbedder(retry=RetryPolicy(...))` re-sends ONE request that failed `unavailable`
  (timeout, refused, 5xx) with exponential backoff (tenacity, now in the `pydantic-ai` extra); the
  batches that landed are not redone, and `request_fault` / `gated` are never retried. Default
  None: each request is sent once, as in 1.1.0.
- `Embedded.retries`: the requests re-sent during the call, on success and failure alike.

## [1.1.0] - 2026-11-04

### Added

- `openai_compatible` takes `profile=` (a pydantic-ai `ModelProfileSpec`) and `settings=`
  (model-level `ModelSettings`) and passes both to the `OpenAIChatModel` it yields, so a caller can
  move an agent whose output type it does not construct onto native structured output without
  rebuilding the model on the builder's client. Both default to None; with neither given the model
  is built exactly as in 1.0.1.

## [1.0.1] - 2026-11-03

### Fixed

- `openai_compatible` and `openai_compatible_embeddings` hand pydantic-ai's OpenAI provider an
  `httpx2.AsyncClient`, so pydantic-ai 2.65's `PydanticAIDeprecationWarning` (a legacy
  `httpx.AsyncClient` on an OpenAI-compatible provider, removed in pydantic-ai 3) no longer fires
  on every call. `transport=` still takes a legacy httpx transport, bridged onto the httpx2
  client; it now also takes an httpx2 one. `classify` maps an `httpx2.HTTPStatusError`.
  (ADR-0009)

### Changed

- The `pydantic-ai` extra names `httpx2>=2.12` (already installed by the OpenAI SDK). The default
  transport under both builders is httpx2's, which verifies TLS against the OS trust store instead
  of `certifi`.
- pytest fails on a `DeprecationWarning` or `UserWarning` raised at a call this package makes.

## [1.0.0] - 2026-11-03

The first stable release: the API of 1.0.0rc1, unchanged. From here the public surface (every
name without a leading underscore, `__all__` per package) is covered by semantic versioning.

### Changed

- Development status is Production/Stable; the README installs without `--pre`.
- No change was needed here to adopt rc1 in its first consumer (mjdevaccount/trade-data#2755).

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

[Unreleased]: https://github.com/mjdevaccount/llm-ports/compare/v1.8.0...HEAD
[1.8.0]: https://github.com/mjdevaccount/llm-ports/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/mjdevaccount/llm-ports/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/mjdevaccount/llm-ports/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/mjdevaccount/llm-ports/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/mjdevaccount/llm-ports/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/mjdevaccount/llm-ports/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/mjdevaccount/llm-ports/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/mjdevaccount/llm-ports/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/mjdevaccount/llm-ports/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/mjdevaccount/llm-ports/compare/v1.0.0rc1...v1.0.0
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
