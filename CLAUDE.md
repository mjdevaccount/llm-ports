# CLAUDE.md

Agent operating manual for `llm-ports`. Rules, not a tour: the tree is in `src/`, the design in
`docs/REDESIGN.md`, the reasons in `docs/adr/`. Human setup and release steps: `CONTRIBUTING.md`.

## What this is

A typed Python library for calling LLMs where every failure is a **named outcome, never an
exception** (ADR-0001), and generated prose is **grounded by construction**: numbers are compared
by value against their sources (ADR-0003) and the model never sees a digit it could copy wrong
(ADR-0004). Vendor SDKs live only behind adapters, so the core installs light and each integration
is an optional extra.

## Commands (CI runs exactly these, `.github/workflows/ci.yml`)

```bash
uv sync                                   # locked dev environment
uv run pytest -q --cov                    # suite + coverage floor (fail_under in pyproject.toml)
uv run ruff check . && uv run ruff format --check .
uv run pre-commit run --all-files         # docstring-vocabulary hook
```

The import-linter contracts run inside pytest (`tests/contract/test_layers.py`), so a green suite
means the layers hold. CI also runs the suite at every dependency's declared **lower** bound
(`test-lowest`): a floor in `pyproject.toml` is a claim CI proves, so raise it when you use a newer
API, never to silence a resolver.

## Layering (enforced, `[tool.importlinter]` in `pyproject.toml`)

- Bottom-up: `ports` < `grounding | prompts | observability` < `call` < `evals | rag | provenance`
  < `adapters | testing`. A lower layer never imports a higher one.
- **Only `adapters/` and `testing/` import a vendor library** (`pydantic_ai`, `httpx`, `langfuse`,
  `opentelemetry`, `sqlglot`, ...). Core code depends on a port; the adapter implements it.
- Adapters do not import each other. Two grandfathered edges are listed in `ignore_imports`; the
  list only shrinks.
- Every public module defines `__all__`; a test checks every name resolves. Private helpers are
  `_`-prefixed modules or names.
- A new vendor integration = a new `adapters/<vendor>/` package + an optional extra in
  `[project.optional-dependencies]` + its own entry in the independence contract.

## Code rules

- **Never raise across a port.** Return the outcome type the port declares; a new failure mode is a
  new named outcome, not an exception.
- **Docstrings state the contract** (Google style, `ruff` `D` rules): what it does, `Args`,
  `Returns`, `Raises` or "never raises", `Invariants`. No history, no incident dates, no consumer,
  host or service names: the pre-commit hook rejects deployment vocabulary in `src/llm_ports/`.
  Full rules: `CONTRIBUTING.md` §Docstrings.
- **History has two homes:** what changed goes in `CHANGELOG.md` (Keep a Changelog, under
  `Unreleased`); why, with evidence, goes in a new `docs/adr/NNNN-title.md` (MADR). A docstring
  keeps a one-line `Rationale: ADR-NNNN.` pointer.
- Tests mirror `src/` layout. Use the doubles in `llm_ports.testing` rather than ad-hoc mocks of
  vendor clients.
- Optional-extra code imports its vendor lazily or only inside its adapter package, so
  `import llm_ports` works on a bare install.

## Dependencies

- Declared ranges are the **widest range CI proves**: the floor is tested by `test-lowest`, an upper
  cap is a major-version guard only. Never cap below the latest release without an ADR saying why.
- `uv.lock` tracks latest; Renovate (`renovate.json`) refreshes it. A lock change ships with a
  green suite on every leg.

## Git and releases

- Branch + PR, never push `main`. Conventional Commits. `main` is the 1.0 line; `release/0.x`
  takes fixes for 0.x consumers only.
- Release = bump `version`, move `Unreleased` in `CHANGELOG.md`, merge, tag `vX.Y.Z` on `main`.
  `release.yml` publishes to PyPI (trusted publishing) and a GitHub release. A published version
  is never replaced; a fix is a new version.
