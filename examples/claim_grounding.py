"""Measure a judge claim by claim: gold cases, a judge's labels, agreement per label, and code's recheck.

No model runs here. The "judge" is a table of labels a judge returned, so the example shows the
measurement itself: which mistake class the judge makes, not just how often it is wrong.
"""

import json

from llm_ports.evals import claims
from llm_ports.evals.settle import settle
from llm_ports.grounding import numbers
from llm_ports.grounding.derive import Tolerance, recheck

GOLD = [
    {
        "case_id": f"invoice-{i}",
        "input": "F1 subtotal = 90.00\nF2 tax = 10.00\nF3 due = 2026-03-02",
        "output": "Total 100.00, due Monday 2 March.",
        "facts": {"subtotal": "90.00", "tax": "10.00", "due": "2026-03-02"},
        "claims": [
            {
                "id": "C1",
                "quote": "Total 100.00",
                "text": "the total is 100.00",
                "label": "derived",
                "facts": ["subtotal", "tax"],
                "op": "90.00 + 10.00 = 100.00",
            },
            {
                "id": "C2",
                "quote": "due",
                "text": "payment is due on 2026-03-02",
                "label": "supported",
                "facts": ["due"],
            },
            {
                "id": "C3",
                "quote": "Monday",
                "text": "2026-03-02 is a Monday",
                "label": "derived",
                "facts": ["due"],
                "op": "weekday(2026-03-02) = Monday",
                "material": False,
            },
        ],
        "case_label": "pass",
    }
    for i in range(12)
]

loaded = claims.read_cases(json.dumps(row) for row in GOLD)
assert loaded.problems == ()

# A judge that reads every figure but refuses every inference: it calls each `derived` claim invented.
judged = {"supported": "supported", "derived": "not_in_input"}
pairs = [(judged[c.label], c.label) for case in loaded.cases for c in case.claims]
agree = claims.agreement(pairs)

assert agree["supported"].kappa == 1.0  # it reads stated facts perfectly...
assert (
    agree["derived"].kappa is not None and abs(agree["derived"].kappa) < 1e-9
)  # ...never says `derived`: chance agreement on that class
assert agree["grounded"].kappa is None or agree["grounded"].kappa < 0.6


def _k(x: float | None) -> str:
    return "undefined" if x is None else f"{x:.2f}"


for label, a in agree.items():
    print(f"{label:>13}: n={a.n} kappa={_k(a.kappa)} [{_k(a.kappa_low)}, {_k(a.kappa_high)}]")

# The case rule: the material derivation the judge refused fails every case; the gold says pass.
case = loaded.cases[0]
as_judged = [claims.Claim(c.id, c.quote, c.text, judged[c.label], material=c.material) for c in case.claims]
assert claims.case_label(case.claims) == "pass" and claims.case_label(as_judged) == "fail"

# Splits are stable: the same case is always in the same split.
assert claims.split("invoice-0") == claims.split("invoice-0")
print("splits:", sorted({claims.split(c.case_id) for c in loaded.cases}))

# Arithmetic is code's: the judge names the op, code recomputes it and rules where it can.

total = "subtotal + tax = 90.00 + 10.00 = 100.00"
facts = {"subtotal": 90.0, "tax": 10.0}
assert recheck(total, "100.00", facts=facts).verdict == "holds"
assert recheck(total, "110.00", facts=facts).verdict == "breaks"  # a planted wrong total
assert recheck(total, "100.4", tol=Tolerance(absolute=0.25)).verdict == "abstain"  # too close to call
assert recheck("weekday(2026-03-02) = Monday").verdict == "unchecked"  # outside the grammar: never guessed

# A judge that calls the wrong total `derived` is overruled; the right one is confirmed.
wrong = claims.Claim("C1", "Total 110.00", "the total is 110.00", "derived", op=total, extra={"shown": "110.00"})
right = claims.Claim("C1", "Total 100.00", "the total is 100.00", "derived", op=total, extra={"shown": "100.00"})
allowed = numbers.digits(GOLD[0]["input"])
assert settle(wrong, allowed=allowed, facts=facts).label == "contradicted"
assert settle(right, allowed=allowed, facts=facts).label == "derived"
print("recheck:", recheck(total, "110.00", facts=facts).reason)
