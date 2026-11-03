# Redesign: llm-ports → 1.0

Status: accepted · Date: 2026-11-03 · Baseline: v0.11.0 (`ecd91d0`)

> Name decided (§7): the distribution stays `llm-ports`, the import package stays `llm_ports`.

## 0. Why, and what the audit found

The library is a real production seam (trade-data at 0.11.0, finance-ops at 0.6.0) and a public
portfolio piece. The ideas are distinctive; the packaging hides them. Audit against `origin/main`:

| Claim | Verdict |
|---|---|
| ~30 flat modules, no layering | **Confirmed**: 33 `.py` modules flat in `llm_ports/`; ports, call path, 11 vendor adapters, a reference-impl grab-bag (`reference.py`) and test doubles side by side. |
| Docstrings narrate deployment incidents | **Confirmed**, ~20 sites, e.g. `numbers.py:15,33,44`; `slots.py:3,13-18,50,119-123`; `judge_sampling.py:122,164`; `ports.py:53,100,106,280-281`; `content.py:32`; `packs.py:82`. |
| README is one dense table, no runnable example | **Confirmed**: 30-row module table, ALL-CAPS emphasis; the first code block needs a TOML file plus a LiteLLM gateway. |
| Installed only by wheel URL | **Confirmed** (`pyproject` comment, `release.yml`); Renovate reaches it only via a custom regex manager. |
| Name states the pattern, not the value | Confirmed (§7). |
| `judge_sampling._number_law` restates `numbers.py` | **Confirmed**: `judge_sampling.py:129-152` restates in English what `numbers.loose` enforces, and the rubric carries trading words. |

Further findings:

1. **Core imports vendors.** `pydantic-ai-slim[openai]` is a hard dependency; `json_schema.py:12`
   imports `pydantic_ai.profiles.openai`; `reranker_http.py:33` imports `REQUEST_FAULT` from
   `pydantic_ai_generator`; outcome classification lives inside the pydantic-ai adapter.
2. **Deployment vocabulary in the API**: `GenRequest.box`, `Tier.box`, TOML key `box`,
   `GenRequest.label = "box-chat"`; `Chunk.source` documented with consumer feed names.
3. **Misplaced shared types**: `EventHook` is defined in `call` but imported by both Langfuse
   adapters; `reference.NoopTracer` is the call path's null sink, living in "reference".
4. **`__all__` exists in 28 modules but `RUF022` is disabled.**

### What the 1.0 story leads with

1. **Named outcomes**: every call ends `ok | gated | unavailable | request_fault | truncated | invalid`, never raises.
2. **Grounded by construction**: the model writes `{id}` under a grammar with no digit token; code fills the numbers.
3. **The number law**: figures compared by value (digits and spelled-out); one law, every grader reads through it.
4. **Data rights in PROV-O terms**: `print_gate` returns ok or exactly one named refusal.
5. **Prompt packs named by content sha**; judge sampling and claim refutation.
6. **One observation lifecycle** across Langfuse and OpenTelemetry; unmeasured usage is unmeasured, never zero.

## 1. Target package layout

`src/` layout. Layers bottom-up, enforced by import-linter (§8).

```
src/llm_ports/
  __init__.py          curated surface (same names as 0.11's root)
  py.typed
  ports/               L0  Protocols + frozen value types, zero I/O
  observability/       L1  observe(), PortObservation, NoopTracer, context vars, EventHook
  grounding/           L1  number law, Problem/belt, slots, NumberLawGrounder
  prompts/             L1  packs, manifest, RepoPromptSource
  call/                L2  generate(), Hooks, roles, graders, schema helpers
  evals/               L3  GauntletEvaluator, judge sampling, claim refutation
  rag/                 L3  chunking, retrieval metrics
  provenance/          L3  ContentItem/Activity/DataClass, print_gate
  adapters/            L4  everything needing an extra; never imported by L0-L3
    pydantic_ai/  pydantic_evals.py  langfuse/  otel.py  http/  lettucedetect.py  sqlglot.py
  testing/             L4  stubs (core), testing.langfuse (extra)
```

**Reviewable rule:** anything importing a vendor lives under `adapters/` or `testing/`. Plain
`pip install llm-ports` imports with only pydantic, jinja2, pyyaml. Each adapter package turns a
missing-vendor `ImportError` into the install line (``install `llm-ports[langfuse]` ``).

### Module map (0.11 → 1.0)

| 0.11 | 1.0 | Note |
|---|---|---|
| `__init__` | `llm_ports` | same 36 names; `__version__` from `importlib.metadata` |
| `ports` | `ports` (`_types`, `_protocols`) | |
| `failure` | `ports.outcomes` | gains vendor-neutral `REQUEST_FAULT`, `outcome_for_status(code)` |
| `call` | `call` (`_generate`) | |
| `roles` | `call.roles` | |
| `call_graders` | `call.graders` | |
| `json_schema` | `call.schema` | `strict_json_schema` → `adapters.pydantic_ai` (wraps pydantic-ai's transformer) |
| `context`, `observation` | `observability` | `EventHook` moves here from `call` |
| `reference` | **dissolved** | Noop* → `observability`; `NumberLawGrounder` → `grounding`; `RepoPromptSource` → `prompts`; `GauntletEvaluator` → `evals` |
| `packs`, `manifest` | `prompts` | |
| `numbers` | `grounding.numbers` | gains `law_text(mode, *, web)` (§3) |
| `grounding` | `grounding` | |
| `slots` | `grounding.slots` | stays a module (`slots.loose`/`slots.belt` read module-qualified) |
| `content` | `provenance` | |
| `judge_sampling` | `evals.judge` | rubric domain words become a parameter |
| `claim_refute` | `evals.refute` | |
| `chunking`, `retrieval_metrics` | `rag.chunking`, `rag.metrics` | |
| `pydantic_ai_generator`, `_embedder`, `_replay` | `adapters.pydantic_ai` | pydantic-ai-specific half of `classify` stays here |
| `pydantic_evals_judge` | `adapters.pydantic_evals` | |
| `langfuse_tracer`, `langfuse_prompt_source` | `adapters.langfuse` | |
| `otel_gen_ai` | `adapters.otel` | |
| `reranker_http`, `grounder_http`, `index_order` | `adapters.http` | reranker takes `REQUEST_FAULT` from `ports.outcomes` |
| `lettucedetect_grounder` | `adapters.lettucedetect` | |
| `sql_guard` | `adapters.sqlglot` | docs page titled "SQL guard" |
| `testing`, `testing_langfuse` | `testing`, `testing.langfuse` | |

### Public surface

`__all__` per package, sorted, RUF022 on. The root keeps 0.11's names, so root importers
(finance-ops) keep working. A leading underscore is the only privacy marker; anything without one
is semver-covered. Helpers no consumer imports get the underscore in 1.0 (candidates:
`call.trace_output`; pydantic-ai `capture`/`record_response`/`Capture`/`CaptureLike`; langfuse
prompts `kind`/`wire`/`served`). The floor is everything trade-data or finance-ops imports today
(Appendix A).

## 2. Compatibility: 1.0.0, clean break, no shims

**Recommendation: clean break**; trade-data adopts `1.0.0rc1` the same day, finance-ops within the week.

1. Nothing breaks until a consumer bumps: both pin an immutable wheel URL with a lock hash; old
   wheels keep resolving.
2. One active consumer, same owner; trade-data already funnels most imports through its own
   re-export layer (`service/platform/*.py`, `service/ai/ports.py`), so the change is mechanical.
3. Shims for 33 modules are 33 clutter files in the tree being cleaned.

**One allowance, in data not code:** `load_tiers` accepts TOML key `box` as an alias for
`self_hosted` with a `DeprecationWarning` through 1.x.

**Hotfixes in the gap:** cut `release/0.x` from the last 0.x tag before the layout PR.

**trade-data** (`origin/main` `6afcef33`): 95 import lines in 69 files (Appendix A), plus
`pyproject.toml` (dependency, `[tool.uv.sources]` URL, import-linter contracts naming `llm_ports`),
`tests/test_import_contracts.py`, `Dockerfile.grounder` (bake step imports the lettucedetect
grounder), identity asserts in `tests/platform/test_grounding.py` and
`tests/test_platform_numbers.py`, `vulture_whitelist.py` comments, the `box` → `self_hosted` sites
(`service/adapters/pydantic_ai_generator.py`, `service/adapters/pydantic_evals_judge.py`,
`service/platform/llm_judge_rules.py`), `uv.lock`.

**finance-ops** (`origin/master` `364365a`) pins **v0.6.0**. Sites:
`src/finance_ops/clients/llm_categorizer.py:35` (root names, unchanged), `:36` `packs` →
`llm_ports.prompts`, `:37` `call_graders.CallScore` → `llm_ports.call.graders`, `:205`
`pydantic_ai_generator` → `llm_ports.adapters.pydantic_ai`; `tests/test_llm_categorizer.py:18`.
Its `.github/workflows/pin-drift.yml` goes red when `v1.0.0` is tagged and stays red until it
adopts — the forcing function.

## 3. Docstring voice, and where history goes

Rules (`CONTRIBUTING.md`):

1. **A docstring states the contract**: what it does, args, returns, invariants, what it raises (or
   that it never raises). Present tense, Google style (mkdocstrings renders `Args:`/`Returns:`/
   `Raises:`); invariants in an `Invariants:` section, not capitals.
2. **No history, no deployment detail**: no incident dates, replay counts, host or consumer names,
   model sizes, ticket ids. A **compatibility fact** stays ("llama.cpp's grammar compiler ignores
   `maxLength` beside `pattern`"); the **reason** for a design choice moves out with a one-line
   pointer ("Rationale: ADR-0002").
3. **Emphasis by structure, not capitals.**
4. **Examples use neutral domains** (orders, documents), never trading vocabulary.

**Where history goes:** what changed → `CHANGELOG.md` (Keep a Changelog; backfill 0.1–0.8 from
`gh release view`). Why, with evidence → `docs/adr/NNNN-*.md` (MADR). Keep the evidence — it is
the portfolio's best material — reworded without consumer names ("on a production narrator, 40 of
64 replies invented a figure with the instruction in place; under the digit-free grammar, 0").

Seed ADRs: 0001 named outcomes, never raise (and why `Judge` raises) · 0002 grounded prose by
construction · 0003 the number law compares by value · 0004 digit-free context (`to_prose`) ·
0005 a character-whitelist grammar, not a negated class · 0006 data rights in PROV-O terms ·
0007 one observation lifecycle; unmeasured is not zero · 0008 prompt packs named by content sha.

**Gate:** a `pre-commit` `pygrep` hook over the package rejecting deployment vocabulary
(`EDGE|edge-host|\bbox[_-]|Tradier|trade[-_.]data|narrator|\b\d+B\b|20\d\d-\d\d-\d\d`), plus
ruff `D` (google convention).

**One source for the number law:** `grounding/numbers.py` owns the check (`loose`) and its English
statement (`law_text(mode, *, web)`); `evals.judge.rubric` composes `law_text(...)`. Trading
example words become `rubric(..., examples: Sequence[str] = ())`, passed by trade-data. A golden
test pins `rubric()` with trade-data's arguments byte-identical to 0.11 for every `(mode, web)`.

## 4. README outline

1. One line: *"LLM calls that end in a named outcome, and prose whose numbers can't be invented."*
   Badges: PyPI, Python versions, CI, coverage, docs, license.
2. `pip install llm-ports`; extras on one line.
3. Quickstart, ~30 lines, no network or key, executed in CI by `pytest-examples`:

```python
import asyncio
from pydantic import BaseModel, Field
from llm_ports import Role as BaseRole, generate, load_tiers
from llm_ports.grounding import slots
from llm_ports.testing import StubGenerator  # real use: adapters.pydantic_ai.PydanticAIGenerator

TIERS = load_tiers(
    {
        "models": {"small": "gpt-5-nano"},
        "ladder": {"fast": 30.0},
        "tiers": {"default": {"model": "small", "timeout": "fast"}},
    }
)


class Role(BaseRole):
    TIERS = TIERS.tiers


SUMMARIZE = Role("orders.summarize", tier="default")
facts = {"count": "3", "total": "$1,284.50"}  # code renders every value, once


class Summary(BaseModel):  # the grammar has no digit token:
    text: str = Field(pattern=slots.slot_pattern(facts, 200))  # the model can only cite {id}


async def main() -> None:
    model = StubGenerator(Summary(text="Your {count} items came to {total}."))
    res = await generate(SUMMARIZE, prompt="Summarize the order.", output=Summary, generator=model)
    if res.outcome == "ok":
        prose = slots.fill(res.value.text, facts)
        assert slots.belt(prose, facts) == 0  # zero invented figures, by construction
        print(prose)

    down = StubGenerator(lambda req, out: 1 / 0)  # the wire fails ...
    res = await generate(SUMMARIZE, prompt="Summarize the order.", output=Summary, generator=down)
    print(res.outcome, res.reason)  # ... 'unavailable', never a traceback


asyncio.run(main())
```

   (Needs `load_tiers` to accept a mapping — new in PR 5.)
4. Concepts, a paragraph each, linking docs and ADRs.
5. Extras table: `pydantic-ai` (generator, embedder, replay) · `http` (reranker, grounder client,
   index-ordered transport) · `langfuse` (tracer, prompt source, rig) · `otel` (GenAI spans) ·
   `evals` (pydantic-evals judge) · `sql` (SQL guard) · `lettucedetect` (span grounder; large,
   torch) · `all` (everything but `lettucedetect`).
6. Design: hexagonal — ports are ours, adapters are theirs — with a small layer diagram.
7. Links: docs · changelog · ADRs · license (MIT).

## 5. Docs site and examples

**mkdocs-material + mkdocstrings[python]**, GitHub Pages via `actions/deploy-pages`, built with
`mkdocs build --strict`. Material's maintainers announced Zensical as successor; it reads the same
`mkdocs.yml`, so a later move is a build-command change. Rejected: Sphinx/Furo (RST rewrite).
Nav: Home (README) · Concepts · Guides (one per example) · API reference (one
`::: llm_ports.<subpkg>` page per subpackage) · ADRs · Changelog.

`examples/` (CI runs each; all network-free except the last): `quickstart.py` ·
`named_outcomes.py` (fallback policy as a `match` on `res.outcome`) · `slots_grounded_prose.py` ·
`prompt_pack/` · `print_gate.py` · `judge_and_refute.py` · `rag_chunk_and_score.py` ·
`observability_otel.py` (in-memory exporter) · `openai_compatible.py` (env-configured; skipped in CI).

## 6. PyPI: trusted publishing

GitHub Actions OIDC, no stored token. One-time operator setup on PyPI and TestPyPI: a pending
publisher (owner `mjdevaccount`, repo, workflow `release.yml`, environment `pypi` / `testpypi`).
A pending publisher does **not** reserve the name — publish promptly once it exists.

`release.yml` on a `v*` tag: reuse `ci.yml` via `workflow_call` → `build` (tag == version check,
`uv build`, `twine check --strict`) → `publish` (`environment: pypi`, `id-token: write`,
`pypa/gh-action-pypi-publish` pinned by SHA; PEP 740 attestations default) → `github-release`
(`dist/*` attached; existing release assets untouched). `workflow_dispatch` publishes a `.devN`
to TestPyPI as the dry run.

`llm-ports` was unclaimed on PyPI on 2026-11-03 (`https://pypi.org/pypi/llm-ports/json` → 404;
unclaimed is not guaranteed registrable — the TestPyPI dry run proves it).

## 7. Name decision

**Decided (operator, 2026-11-03): keep `llm-ports`** (import `llm_ports`). "Ports" names the
ports-and-adapters design engineers recognise; the README's first line carries the value. No repo
rename, no estate reference changes, and no import-prefix change beyond the 1.0 subpackage paths.
Rejected: renames such as `groundwire` and `grounded-llm` — a metaphor that needs explaining, or
a rename cost bought for a marginal gain.

## 8. CI hardening

| Job | What |
|---|---|
| `lint` | `ruff check` + `ruff format --check` (E,F,W,I,B,UP,RUF with RUF022 on, + D google, PT, SIM, TID252); `pre-commit run --all-files`; `lint-imports`; `zizmor` |
| `types` | `mypy --strict src/`; `pyright --verifytypes llm_ports --ignoreexternal` at 100% or a recorded floor that only ratchets up |
| `test` | 3.12, 3.13, 3.14, 3.15 on ubuntu + 3.13 on windows; `uv sync --frozen --all-extras`; `pytest --cov`. 3.12 floor (PEP 695). 3.15 required once its wheels exist |
| `test-lowest` | 3.12, `uv pip install --resolution lowest-direct -e '.[all]'` — proves lower bounds |
| `core-only` | `uv pip install .` (no extras); `tests/test_core_imports.py` imports every non-adapter module in a subprocess and asserts no vendor module is loaded |
| `examples` | run `examples/*.py`; `pytest-examples` over README and docs |
| `docs` | `mkdocs build --strict`; deploy on tag |
| `gitleaks`, `dup-code` | keep the ci-kit reusable workflows |

import-linter: `layers` (adapters | testing > evals | rag | provenance > call > grounding |
prompts | observability > ports); `independence` among `adapters.*`; `forbidden` vendor imports
outside `adapters`/`testing`. Coverage via Codecov OIDC (tokenless, public) with a badge;
`--cov-fail-under` starts at PR 1's measured value and only rises. Metadata: `Typing :: Typed`,
version classifiers, `Production/Stable` at 1.0, `project.urls`; add `SECURITY.md`,
`CONTRIBUTING.md`. Workflows default `permissions: {}`, each job grants only what it needs.

## 9. Execution plan (ordered PRs, each shippable)

| PR | Scope | Accept when |
|---|---|---|
| 0 | this document | operator approves; name decision recorded in §7 |
| 1 | CI hardening, no API change: matrix, `test-lowest`, coverage + Codecov, `ruff format`, `twine check`, `zizmor` | every leg green on `main`; coverage badge renders; floor recorded |
| 2 | Voice and history → **0.12.0**: contract-only docstrings, ADR 0001–0008, Keep-a-Changelog backfill, `pygrep` hook + ruff `D`, `numbers.law_text` single source | hook passes; `rubric()` golden byte-identical to 0.11 for all `(mode, web)`; tests green; tag `v0.12.0`, cut `release/0.x` |
| 3 | Publishing plumbing (operator first: PyPI/TestPyPI pending publishers; `pypi`/`testpypi` environments): pyproject urls/classifiers, new `release.yml` (trusted publishing; GitHub release assets kept for wheel-URL consumers) | `workflow_dispatch` publishes a `.devN` to TestPyPI with no secret; installing it from TestPyPI works; the next tag lands on PyPI with attestations |
| 4 | Layout (breaking; `main` = `1.0.0.dev0`, unreleased): `src/llm_ports/` via `git mv` + import rewrites only; tests mirrored; `__all__` everywhere; import-linter contracts | every moved file a ≥90% rename in `git diff -M --summary`; test count unchanged and green; `lint-imports` green; a test asserts every public module defines `__all__` and every name resolves |
| 5 | 1.0 API cleanup (breaking): extras split (core = pydantic, jinja2, pyyaml); `outcome_for_status`/`REQUEST_FAULT` → `ports.outcomes`; `strict_json_schema` → adapter; `EventHook` → `observability`; `box` → `self_hosted` (TOML alias + warning), label default `"chat"`; `rubric(examples=)`; `load_tiers` accepts a mapping; underscore private helpers | `core-only` green; rubric golden with trade-data's examples; a `box` tier table loads with `DeprecationWarning`; `HttpReranker` imports with only `http` installed |
| 6 | Strict typing | `mypy --strict` green; `pyright --verifytypes` at the gate |
| 7 | Docs, examples, README | `mkdocs build --strict`; `examples` green; README quickstart runs under `pytest-examples`; Pages live |
| 8 | Release + adoption: tag `v1.0.0rc1` → PyPI; trade-data PR (dependency `llm-ports[pydantic-ai,http,langfuse,otel,sql,evals]==1.0.0rc1`, drop `[tool.uv.sources]`, rewrite the 95 lines per Appendix A, import-linter names, `Dockerfile.grounder`, `box` sites, rubric `examples=`); then `v1.0.0` from rc1's commit, trade-data lock-only bump; finance-ops PR (5 lines + dependency) | trade-data full CI green incl. `test_import_contracts`; grounder image builds; nightly eval board matches the last 0.11 run; finance-ops CI and pin-drift green |

## Appendix A — trade-data import sites by 0.11 module (`origin/main` `6afcef33`)

| 0.11 module → 1.0 | Sites |
|---|---|
| root (module aliases) → per-module homes | `service/adapters/pydantic_ai_embedder.py:35`, `service/adapters/pydantic_ai_generator.py:31`, `service/observability/gen_ai.py:23`, `service/platform/llm_call.py:57,58,59`, `service/platform/llm_judge.py:38`, `service/platform/llm_roles.py:28`, `service/platform/prompt_packs.py:21`, `tests/platform/test_retrieval_metrics.py:7`, `tests/test_stamp_characterization.py:35` |
| `ports` → `ports` | `service/ai/ports.py:11`, `service/adapters/pydantic_ai_embedder.py:36`, `tests/_fake_embedder.py:17`, `tests/adapters/test_pydantic_evals_judge.py:17` |
| `call` → `call` | `service/platform/llm_call.py:60` |
| `call_graders` → `call.graders` | `service/platform/call_graders.py:16` |
| `roles` → `call.roles` | `service/platform/llm_roles.py:29` |
| `json_schema` → `call.schema` / `adapters.pydantic_ai` | `service/platform/llm_json.py:4`, `scripts/llm_judge_calibrate.py:151` |
| `failure` → `ports.outcomes` | `service/platform/llm_failure.py:7` |
| `context` → `observability` | `service/platform/llm_context.py:8` |
| `observation` → `observability` | `service/platform/llm_trace.py:31` |
| `reference` → `observability` / `grounding` / `prompts` / `evals` | `scripts/ai_spike_cite.py:40`, `scripts/day_brief_eval.py:29`, `service/adapters/ai_ours_port.py:23`, `service/platform/llm_trace.py:32`, `service/platform/prompt_specs.py:40`, `tests/adapters/test_langfuse_tracer_port.py:16`, `tests/ai/test_board_single_source.py:26`, `tests/ai/test_evaluator_oracle.py:21`, `tests/ai/test_ports_contract.py:18`, `tests/narrator/test_narrator_agent.py:162`, `tests/platform/test_llm_trace.py:27`, `tests/platform/test_prompt_specs.py:17`, `tests/test_gen_ai_spans.py:29`, `tests/test_llm_ledger_instrumentation.py:10` |
| `packs`, `manifest` → `prompts` | `service/platform/prompt_packs.py:22,23`, `service/platform/stamp.py:55`, `service/schemas/prompt_pack.py:4` |
| `numbers` → `grounding.numbers` | `service/platform/numbers.py:4` |
| `grounding` → `grounding` | `service/platform/grounding.py:10`, `tests/platform/test_grounding.py:23` |
| `slots` → `grounding.slots` | `service/platform/slots.py:6`, `tests/test_platform_numbers.py:24` |
| `content` → `provenance` | `service/platform/content.py:13` |
| `judge_sampling` → `evals.judge` | `scripts/llm_judge_calibrate.py:233,250`, `service/platform/llm_call.py:61`, `service/platform/llm_judge_rules.py:32`, `tests/platform/test_llm_call.py:15`, `tests/platform/test_llm_judge.py:23`, `tests/platform/test_llm_judge_rules.py:16` |
| `claim_refute` → `evals.refute` | `service/platform/claim_refute.py:4`, `tests/test_decision_autopsy.py:21`, `tests/test_investigate.py:718` |
| `chunking` → `rag.chunking` | `service/rag/text.py:21` |
| `retrieval_metrics` → `rag.metrics` | `scripts/rag_eval.py:41`, `service/platform/retrieval_metrics.py:6` |
| `pydantic_ai_generator` → `adapters.pydantic_ai` | `service/adapters/pydantic_ai_generator.py:32`, `tests/platform/test_ai_seam_extractable.py:15` |
| `pydantic_ai_replay` → `adapters.pydantic_ai.replay` | `service/evals/day_brief_graders.py:18`, `service/evals/recorded_model.py:19`, `tests/adapters/test_pydantic_ai_generator.py:249`, `tests/day_brief/test_day_brief_graders.py:15` |
| `pydantic_evals_judge` → `adapters.pydantic_evals` | `service/adapters/pydantic_evals_judge.py:29`, `tests/adapters/test_pydantic_evals_judge.py:18` |
| `langfuse_tracer` → `adapters.langfuse` | `service/adapters/langfuse_tracer_port.py:23`, `tests/test_partial_aggregate_honesty.py:19,25,30` |
| `langfuse_prompt_source` → `adapters.langfuse` | `service/adapters/langfuse_prompt_port.py:21` |
| `otel_gen_ai` → `adapters.otel` | `service/observability/gen_ai.py:24` |
| `reranker_http` → `adapters.http` | `service/adapters/box_rerank.py:27` |
| `grounder_http` → `adapters.http` | `service/adapters/grounder_http_port.py:21`, `tests/adapters/test_grounder_http_port.py:10`, `tests/ai/test_ports_contract.py:16`, `tests/test_grounder_sidecar.py:11`, `tests/usecases/test_day_brief_ground.py:12` |
| `lettucedetect_grounder` → `adapters.lettucedetect` | `service/grounder_sidecar.py:64,94`, `tests/adapters/test_lettucedetect_live.py:28`, `tests/ai/test_ports_contract.py:17`, `tests/test_grounder_sidecar.py:12` |
| `sql_guard` → `adapters.sqlglot` | `service/platform/sql_guard.py:13`, `tests/platform/test_sql_guard.py:8` |
| `testing` → `testing` | `tests/adapters/test_box_rerank.py:14`, `tests/ai/test_ports_contract.py:19`, `tests/test_grounder_sidecar.py:13` |
| `testing_langfuse` → `testing.langfuse` | `tests/adapters/test_langfuse_prompt_port.py:13,14`, `tests/adapters/test_langfuse_tracer_port.py:17,18`, `tests/ai/test_ports_contract.py:20` |
