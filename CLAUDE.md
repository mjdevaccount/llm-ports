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
uv run vulture                            # dead code (whitelist: [tool.vulture] in pyproject.toml)
uv run --group types mypy                 # --strict over src/ ([tool.mypy] in pyproject.toml)
uv run --group types pyright --verifytypes llm_ports --ignoreexternal   # public surface 100% typed
uv run --group docs mkdocs build --strict # the docs site (mkdocs.yml); any warning fails
```

The import-linter contracts run inside pytest (`tests/contract/test_layers.py`), so a green suite
means the layers hold. CI also runs the suite at every dependency's declared **lower** bound
(`test-lowest`): a floor in `pyproject.toml` is a claim CI proves, so raise it when you use a newer
API, never to silence a resolver. The `core-only` job installs the package with no extra (then
with `http` only) and runs `tests/contract/test_core_imports.py`: a bare install imports the whole
core and no vendor module.

## Layering (enforced, `[tool.importlinter]` in `pyproject.toml`)

- Bottom-up: `ports` < `grounding | prompts | observability` < `call` < `evals | rag | provenance`
  < `adapters | testing`. A lower layer never imports a higher one.
- **Only `adapters/` and `testing/` import a vendor library** (`pydantic_ai`, `httpx`, `langfuse`,
  `opentelemetry`, `sqlglot`, ...). Core code depends on a port; the adapter implements it.
- Adapters do not import each other, with no `ignore_imports` exception. Code two adapters share
  is a private module directly under `adapters/` (`adapters/_index_order.py`), outside the
  independence contract.
- Every public module defines `__all__`; a test checks every name resolves. Private helpers are
  `_`-prefixed modules or names.
- A new vendor integration = a new `adapters/<vendor>/` package + an optional extra in
  `[project.optional-dependencies]` (and in `all`, unless it is heavy) + its own entry in the
  independence contract + its row in `ADAPTERS` in `tests/contract/test_core_imports.py`.

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
- **Strictly typed.** `mypy --strict` passes over `src/` and `pyright --verifytypes` scores 100%:
  annotate every public attribute and class variable, never leave an inferred type on the public
  surface. A `# type: ignore[code]` names its code and says why on the same line; a vendor with no
  type information goes in `[[tool.mypy.overrides]]`, not in an ignore.
- **Docs and examples execute.** Each `examples/` script asserts its own claims and runs in
  `tests/examples/` on every `pytest` leg (one needing a live endpoint runs only when its env vars
  are set); every ```` ```python ```` block in `README.md` and `docs/` runs under
  `pytest-examples`. A fragment that cannot run is fenced ```` ```{.python test="skip"} ````. A
  guide (`docs/guides/`) includes its example with `--8<--`, never a pasted copy, and the API
  reference is generated from docstrings (`docs/api/`, mkdocstrings). A new public subpackage gets
  an API page and a `nav` entry in `mkdocs.yml`.
- Tests mirror `src/` layout. Use the doubles in `llm_ports.testing` rather than ad-hoc mocks of
  vendor clients.
- A vendor import sits inside its adapter, in a `with extra("<extra>"):` block (`llm_ports._extras`),
  so a missing extra fails with its install line and `import llm_ports` works on a bare install.

## Dependencies

- Declared ranges are the **widest range CI proves**: the floor is tested by `test-lowest`, an upper
  cap is a major-version guard only. Never cap below the latest release without an ADR saying why.
- `uv.lock` tracks latest; Renovate (`renovate.json`) refreshes it. A lock change ships with a
  green suite on every leg.

## Git and releases

- Branch + PR, never push `main`. Conventional Commits. `main` is the 1.0 line; `release/0.x`
  takes fixes for 0.x consumers only.
- Release = bump `version`, move `Unreleased` in `CHANGELOG.md`, merge. `tag-version.yml` tags
  `vX.Y.Z` at the bump commit and dispatches `release.yml`; never push a main tag by hand
  (a `release/0.x` fix is still tagged by hand).
  `release.yml` publishes to PyPI (trusted publishing), a GitHub release and the docs site
  (GitHub Pages, from `ci`'s strict build). A published version
  is never replaced; a fix is a new version.
