# Contributing

## Setup

```
uv sync                      # the locked dev environment
uv run pre-commit install    # the docstring-vocabulary hook, on every commit
uv run pytest -q --cov       # the suite, with the coverage floor
uv run ruff check . && uv run ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs the same commands on Python 3.12–3.14 and Windows, plus the
suite at every dependency's declared lower bound, `twine check --strict` on the build, and
`zizmor` over the workflows.

## Docstrings

A docstring states the contract. Its reader is someone calling the function, not someone
auditing why it was written.

1. **Contract only.** What it does, `Args:`, `Returns:`, `Raises:` (or that it never raises), in
   the present tense, Google style. Rules that always hold go in an `Invariants:` section.
2. **No history and no deployment detail.** No incident dates, replay counts, host, service or
   consumer names, model sizes, or ticket ids. A *compatibility fact* stays ("llama.cpp's grammar
   compiler ignores `maxLength` beside `pattern`"). The *reason* for a design choice moves to an
   ADR, and the docstring keeps a one-line pointer (`Rationale: ADR-0002.`).
3. **Emphasis by structure, not capitals.** An invariant goes in `Invariants:`; a warning goes in
   its own sentence. Capitals are for constants and acronyms.
4. **Neutral examples.** Orders, documents, tickets, invoices.

`ruff` enforces the Google layout (`D` rules). A `pre-commit` hook (`.pre-commit-config.yaml`)
rejects deployment vocabulary in `src/llm_ports/`.

## Where history goes

- **What changed**: `CHANGELOG.md` ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/)).
- **Why, with the evidence**: `docs/adr/NNNN-title.md` ([MADR](https://adr.github.io/madr/)).
  Evidence is stated without naming the system it came from ("on a production summarizer, 40 of
  64 replies invented a figure with the instruction in place").

## Releasing

Bump `version` in `pyproject.toml`, move `Unreleased` in `CHANGELOG.md` under the new version, and
merge. Then push a `vX.Y.Z` tag on `main`. `release.yml` runs the whole of `ci.yml`, checks the tag
against the version, builds, runs `twine check --strict`, then, independently of each other,
uploads to PyPI (trusted publishing through the `pypi` environment, with PEP 740 attestations) and
attaches the wheel and sdist to a GitHub release. The release assets never wait on the upload:
consumers that pin the wheel URL get it whatever PyPI did. A published file is never replaced.

To dry-run the pipeline, run `release.yml` from `main` by hand (`gh workflow run release.yml`). It
uploads a dev release, `X.Y.Z.devN` with N the run number, which installers take only when asked
(`pip install --pre`).
