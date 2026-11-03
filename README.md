# llm-ports

**LLM calls that end in a named outcome, and prose whose numbers can't be invented.**

[![PyPI](https://img.shields.io/pypi/v/llm-ports?include_prereleases)](https://pypi.org/project/llm-ports/)
[![Python](https://img.shields.io/pypi/pyversions/llm-ports)](https://pypi.org/project/llm-ports/)
[![ci](https://github.com/mjdevaccount/llm-ports/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mjdevaccount/llm-ports/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mjdevaccount/llm-ports/graph/badge.svg)](https://codecov.io/gh/mjdevaccount/llm-ports)
[![docs](https://img.shields.io/badge/docs-mjdevaccount.github.io-blue)](https://mjdevaccount.github.io/llm-ports/)
[![license](https://img.shields.io/pypi/l/llm-ports)](https://github.com/mjdevaccount/llm-ports/blob/main/LICENSE)

```
pip install "llm-ports[pydantic-ai]"
```

The core needs only pydantic, jinja2 and pyyaml. Each integration is an extra: `pydantic-ai`
(generator, embedder, replay) · `http` (reranker, grounder client) · `langfuse` (tracer, prompt
source, test rig) · `otel` (GenAI spans) · `evals` (pydantic-evals judge) · `sql` (SQL guard) ·
`lettucedetect` (span grounder; large, pulls torch) · `all` (everything but `lettucedetect`).
Importing an adapter without its extra fails with the line that installs it.

## Quickstart

No network and no key: `StubGenerator` stands in for the model.

```python
import asyncio

from pydantic import BaseModel, Field

from llm_ports import Role as BaseRole
from llm_ports import generate, load_tiers
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
        print(prose)  # Your 3 items came to $1,284.50.

    down = StubGenerator(lambda req, out: 1 / 0)  # the wire fails ...
    res = await generate(SUMMARIZE, prompt="Summarize the order.", output=Summary, generator=down)
    print(res.outcome, res.reason)  # ... 'unavailable', never a traceback


asyncio.run(main())
```

## Concepts

**Named outcomes.** A model call ends in one of `ok | gated | unavailable | request_fault |
truncated | invalid`, with a short reason, and never raises for a model, wire or gate failure.
A fallback policy is a `match` on the outcome, not a tower of `except` clauses.
([ADR-0001](https://github.com/mjdevaccount/llm-ports/blob/main/docs/adr/0001-named-outcomes-never-raise.md))

**Grounded prose by construction.** Code renders every figure once, under a digit-free id. The
output grammar has no digit token, so the model cites `{total}` and cannot write a number at all;
code fills the slots, and a belt over the filled text counts the figures no source rendered. That
count is zero by construction, so a non-zero count is an alarm, not an error rate.
([ADR-0002](https://github.com/mjdevaccount/llm-ports/blob/main/docs/adr/0002-grounded-prose-by-construction.md),
[ADR-0003](https://github.com/mjdevaccount/llm-ports/blob/main/docs/adr/0003-the-number-law-compares-by-value.md))

**Roles and tiers.** A `Role` names one generation surface (`feature.action`) and resolves to a
tier in a table the consumer owns: which model, whether it is self-hosted, which timeout rung.
Feature code carries no model or timeout literal.

**Prompt packs.** A prompt is a versioned directory of templates, named by the sha of its text, so
an output row records exactly which prompt produced it and an edited prompt is never mistaken for
the one that was evaluated.
([ADR-0008](https://github.com/mjdevaccount/llm-ports/blob/main/docs/adr/0008-prompt-packs-named-by-content-sha.md))

**Provenance and the print gate.** Every output is a PROV-O entity carrying its claims and the
evidence each cites; `print_gate` answers whether it may be shown to an audience, with one named
refusal when it may not.
([ADR-0006](https://github.com/mjdevaccount/llm-ports/blob/main/docs/adr/0006-data-rights-in-prov-o-terms.md))

**One observation per call.** A call is traced as one generation, opened once and ended once, with
unmeasured usage flagged rather than recorded as zero.
([ADR-0007](https://github.com/mjdevaccount/llm-ports/blob/main/docs/adr/0007-one-observation-lifecycle.md))

## Design

Hexagonal: the ports are ours and small, the adapters are thin wrappers over the libraries that
already do the work (structured output, retries and validation come from pydantic-ai behind the
`Generator` port). The layers, enforced by import-linter:

```
adapters | testing                      vendor SDKs live only here, one extra each
evals | rag | provenance
call                                    generate(): the one call path
grounding | prompts | observability
ports                                   Protocols and value types, zero I/O
```

## Links

[Documentation](https://mjdevaccount.github.io/llm-ports/) ·
[examples](https://github.com/mjdevaccount/llm-ports/tree/main/examples) ·
[changelog](https://github.com/mjdevaccount/llm-ports/blob/main/CHANGELOG.md) ·
[decision records](https://github.com/mjdevaccount/llm-ports/tree/main/docs/adr) ·
[contributing](https://github.com/mjdevaccount/llm-ports/blob/main/CONTRIBUTING.md)

### Known

- Dependency ranges are the widest range CI proves. Every floor is installed and tested by the
  `test-lowest` job (`--resolution lowest-direct`); an upper bound is a major-version guard only
  (`pydantic-ai-slim<3`, `langfuse<6`); `uv.lock` tracks the latest releases (Renovate refreshes
  it) and the `pytest` matrix runs on it.
- pydantic-ai (2.57 through 2.65) emits a `PydanticAIDeprecationWarning` when an OpenAI-compatible
  provider is given an `httpx.AsyncClient`; it takes `httpx2.AsyncClient` and drops httpx in v3.
  The adapters build an httpx client because a consumer's `transport=` is an httpx transport, so
  moving to httpx2 changes that argument's type.

MIT licensed.
