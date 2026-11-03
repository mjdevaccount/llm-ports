"""A fallback policy written as a `match` on the outcome, because every failure has a name.

`generate` never raises for a model, wire or gate failure. Each one comes back as an outcome
(`ok | gated | unavailable | request_fault | truncated | invalid`) with a short reason, so the
policy is plain control flow: retry elsewhere on `unavailable`, give up on `gated`, shrink the
request on `request_fault`, and so on.
"""

import asyncio

from llm_ports import Generated, GenRequest, Hooks, Role, generate, load_tiers
from llm_ports.testing import StubGenerator

TABLE = load_tiers(
    {
        "models": {"local": "qwen3-8b", "cloud": "gpt-5-nano"},
        "ladder": {"fast": 30.0},
        "tiers": {
            "local": {"model": "local", "timeout": "fast"},
            "cloud": {"model": "cloud", "timeout": "fast", "self_hosted": False},
        },
    }
)


class Tagger(Role):
    TIERS = TABLE.tiers


LOCAL = Tagger("tickets.tag", tier="local")
CLOUD = Tagger("tickets.tag", tier="cloud")


def offline(req: GenRequest, out: type) -> str:
    raise ConnectionError("connection refused")


def answer(req: GenRequest, out: type) -> str:
    return f"billing (from {req.model})"


async def tag(ticket: str, *, budget_left: bool) -> str:
    hooks = Hooks(admit=lambda role: None if budget_left else "daily budget spent")
    res: Generated[str] = await generate(LOCAL, prompt=ticket, generator=StubGenerator(offline), hooks=hooks)
    match res.outcome:
        case "ok":
            return res.value or ""
        case "unavailable":  # the local box is down: the same role on the cloud tier
            res = await generate(CLOUD, prompt=ticket, generator=StubGenerator(answer), hooks=hooks)
            return res.value if res.outcome == "ok" and res.value else "untagged"
        case "gated":  # a cap said no; nothing was sent
            return f"skipped: {res.reason}"
        case _:  # request_fault / truncated / invalid: the reply is not usable as-is
            return "untagged"


async def main() -> None:
    print(await tag("I was charged twice", budget_left=True))
    print(await tag("I was charged twice", budget_left=False))


asyncio.run(main())
