"""Measure a judge claim by claim: gold cases, a judge's labels, agreement per label with its interval.

No model runs here. The "judge" is a table of labels a judge returned, so the example shows the
measurement itself: which mistake class the judge makes, not just how often it is wrong.
"""

import json

from llm_ports.evals import claims

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
