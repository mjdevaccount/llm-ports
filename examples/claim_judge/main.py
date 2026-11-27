"""Judge a reply claim by claim: code settles what it can find, the model is asked about the rest.

An extraction model read an invoice and returned JSON. `claim_judge.grade` splits the reply into
one claim per field (`decompose.leaves`), numbers the Input's lines as facts, settles in code every
value it finds verbatim or every figure the number law licenses, and asks the judge only about what
remains, one claim per call, under a grammar that fits the question. A verdict must quote the fact
it rests on, and a quote that is not in the facts abstains.

No model runs here: the "judge" is a stub that answers like a careless local model, calling every
value it cannot see verbatim "not_in_input". Code-settling is what keeps it from failing the case.
"""

import asyncio
import json
from functools import partial
from pathlib import Path

from llm_ports import Role, call, load_tiers, prompts
from llm_ports.evals import claim_judge as cj
from llm_ports.evals import decompose as d
from llm_ports.ports import GenRequest
from llm_ports.testing import StubGenerator

INVOICE = """Vendor: Acme Supplies
Invoice no: INV-2041
Subtotal: 90.00
Tax: 10.00
Total due: 100.00
Due date: 2026-03-02"""

REPLY = json.dumps(
    {
        "vendor": "Acme Supplies",
        "invoice_no": "INV-2041",
        "total": 100.0,
        "status": "unpaid",
        "currency": "EUR",
        "line_items": [],
    }
)

TABLE = load_tiers(
    {"models": {"m": "local-8b"}, "ladder": {"fast": 30.0}, "tiers": {"default": {"model": "m", "timeout": "fast"}}}
)


class AppRole(Role):
    TIERS = TABLE.tiers


JUDGE = AppRole("invoice.claim", tier="default")
SCHEMA = {"properties": {"status": {"enum": ["paid", "unpaid", "partial"]}}}
KINDS: dict[str, d.Kind] = {"vendor": "extract", "invoice_no": "extract", "total": "extract", "status": "classify"}


def kind_of(leaf: d.Leaf) -> d.Kind:
    return KINDS.get(leaf.field, "free")


def careless(req: GenRequest, out: type) -> dict[str, object]:
    """A judge that says `not_in_input` to anything whose words it cannot see, and quotes nothing."""
    contradiction = "consistent" in json.dumps(out.json_schema)  # type: ignore[attr-defined]
    if contradiction:
        return {"fact_ids": [], "quote": "", "label": "consistent", "reason": "no fact rules it out"}
    if "currency" in req.prompt.split("CLAIM", 1)[1]:
        return {"fact_ids": [], "quote": "", "label": "not_in_input", "reason": "no fact names a currency"}
    return {"fact_ids": ["F1"], "quote": "Acme Supplies", "label": "supported", "reason": "F1"}


async def main() -> None:
    pack = prompts.load("invoice.claim", Path(__file__).with_name("packs"))
    stub = StubGenerator(careless)
    judge = cj.GenerateClaimJudge(partial(call.generate, generator=stub), JUDGE, pack=pack)
    policy = cj.Policy(kind_of=kind_of, allowed=d.enums(SCHEMA))
    ruling = await cj.grade(REPLY, INVOICE, judge=judge, policy=policy)

    for v in ruling.verdicts:
        print(f"{v.path:>12}: {v.outcome:<8} {v.label or '-':<13} {v.checker}")
    print(ruling.label, ruling.counts())

    settled = {v.path for v in ruling.verdicts if v.outcome == "settled"}
    assert settled == {"vendor", "invoice_no", "total"}  # found in the Input: never sent to the model
    assert len(stub.requests) == 3  # status, currency, line_items: only these reached the judge
    assert ruling.label == "fail" and ruling.failing == ("currency",)  # the one value the reply added

    # The same reply without the added currency passes.
    clean = json.dumps({k: v for k, v in json.loads(REPLY).items() if k != "currency"})
    assert (await cj.grade(clean, INVOICE, judge=judge, policy=policy)).label == "pass"


asyncio.run(main())
