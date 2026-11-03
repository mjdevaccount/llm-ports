"""A real model over any OpenAI-compatible endpoint: OpenAI, a LiteLLM gateway, llama.cpp, vLLM.

Needs the `pydantic-ai` extra and three environment variables; the test suite skips it when they
are unset:

    LLM_PORTS_BASE_URL   e.g. https://api.openai.com/v1 or http://localhost:4000/v1
    LLM_PORTS_MODEL      e.g. gpt-5-nano
    LLM_PORTS_API_KEY    optional for a local server
"""

import asyncio
import os

from pydantic import BaseModel

from llm_ports import Role, generate, load_tiers
from llm_ports.adapters.pydantic_ai import PydanticAIGenerator, openai_compatible

if not os.environ.get("LLM_PORTS_BASE_URL") or not os.environ.get("LLM_PORTS_MODEL"):
    raise SystemExit("set LLM_PORTS_BASE_URL and LLM_PORTS_MODEL to run this example")

TABLE = load_tiers(
    {
        "models": {"chat": os.environ["LLM_PORTS_MODEL"]},
        "ladder": {"gateway": 60.0},
        "tiers": {"default": {"model": "chat", "timeout": "gateway", "self_hosted": False}},
    }
)


class AppRole(Role):
    TIERS = TABLE.tiers


CLASSIFY = AppRole("tickets.classify", tier="default", graders=("schema_complete",))


class Ticket(BaseModel):
    category: str
    urgent: bool


async def main() -> None:
    model = PydanticAIGenerator(
        model_for=openai_compatible(os.environ["LLM_PORTS_BASE_URL"], os.environ.get("LLM_PORTS_API_KEY"))
    )
    res = await generate(
        CLASSIFY, prompt="Classify: 'I was charged twice and need it fixed today.'", output=Ticket, generator=model
    )
    print(res.outcome, res.value, res.usage, res.reason or "")


asyncio.run(main())
