"""A sampled judge and an adversarial pass, through the same `generate` as every other call.

The judge's pure half decides which replies are graded (a deterministic 1-in-N draw, so a rerun
grades the same ones) and builds the instruction (`rubric`) from what the judged call saw. The
refute pass asks a model to refute each claim against the evidence; verdicts come back by `idx`,
so a reworded echo of a claim is still attributed to the right one.
"""

import asyncio
import datetime as dt
from functools import partial
from pathlib import Path

from llm_ports import Role, generate, load_tiers, prompts
from llm_ports.evals import judge, refute
from llm_ports.testing import StubGenerator

TABLE = load_tiers(
    {"models": {"m": "gpt-5-nano"}, "ladder": {"fast": 30.0}, "tiers": {"default": {"model": "m", "timeout": "fast"}}}
)


class AppRole(Role):
    TIERS = TABLE.tiers


SUMMARIZE = AppRole("orders.summarize", tier="default")
ADVERSARY = AppRole("orders.refute", tier="default")
PACKS = Path(__file__).with_name("prompt_pack") / "packs"

# --- the judge: which reply to grade, and the instruction it reads
prompt, reply = "Summarize the order.", "Your 3 items came to $1,284.50."
key = judge.sample_key(SUMMARIZE, prompt, judge.reply_text(reply))
print("sampled at 1-in-1:", judge.selected(key, 1))
instruction = judge.rubric(
    SUMMARIZE,
    judge.seen_for(output=str),
    platform="shop assistant",
    mode="slots",
    today=dt.date(2026, 11, 3),
    tz_label="UTC",
)
print(instruction.splitlines()[0])


# --- the adversary: refute each claim against the evidence
def adversary(req, out):
    return {
        "verdicts": [
            {"idx": 2, "claim": "ships Friday", "verdict": "refute", "reason": "evidence says Monday"},
            {"idx": 1, "claim": "3 items", "verdict": "uphold", "reason": "line items: 3"},
        ]
    }


async def main() -> None:
    claims = ["The order has 3 items.", "It ships Friday."]
    res = await refute.refute(
        partial(generate, generator=StubGenerator(adversary)),
        ADVERSARY,
        pack=prompts.load("orders.refute", PACKS),
        claims=claims,
        evidence='{"items": 3, "ships": "Monday"}',
        max_items=4,
        claim_cap=120,
        noun="claims",
        subject="one order",
        evidence_label="ORDER",
        name="verdicts",
    )
    verdicts = refute.match_verdicts(claims, (res.value or {}).get("verdicts"))
    for claim, v in zip(claims, verdicts, strict=True):
        print(f"{claim!r}: {v['verdict'] if v else 'unexamined'}")


asyncio.run(main())
