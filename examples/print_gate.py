"""The print gate: may this generated item be shown to this audience? One named answer.

Every output is a `ContentItem` (a PROV-O entity): a body, its claims, and the evidence chunks
each claim cites. Its `DataClass` is the join of its sources' classes (most restrictive wins).
`print_gate` checks, in order: rights against the audience's ceiling, a claim decomposition,
every claim cited, every cited chunk sourced and dated, and a link where the source requires one.
"""

from datetime import date

from llm_ports import ContentItem, DataClass, print_gate
from llm_ports.ports import Chunk, Claim

docs = Chunk(id="d1", text="Plan B is 12 dollars a month.", source="pricing page", as_of=date(2026, 11, 1))
forum = Chunk(id="f1", text="Support answers within a day.", source="user forum", as_of=date(2026, 10, 30))

owned = DataClass("owned")
scraped = DataClass("scraped", link_required=True)

item = ContentItem(
    kind="answer",
    id="a-1",
    data_class=owned.join(scraped),  # one scraped source makes the whole answer scraped
    body="Plan B costs 12 dollars a month, and support answers within a day.",
    claims=(Claim("Plan B costs 12 dollars a month.", ("d1",)), Claim("Support answers within a day.", ("f1",))),
    evidence={"d1": docs, "f1": forum},
)

for audience in ("operator", "public"):
    print(audience, print_gate(item, audience))
# The operator may read scraped text, but the joined class owes a link per cited chunk: `link`.
assert print_gate(item, "operator").reason == "link"
# Scraped is above the public ceiling: `rights`, before any other rule runs.
assert print_gate(item, "public").reason == "rights"

only_docs = ContentItem(
    kind="answer",
    id="a-2",
    data_class=owned,
    body="Plan B costs 12 dollars a month.",
    claims=(Claim("Plan B costs 12 dollars a month.", ("d1",)),),
    evidence={"d1": docs},
)
assert print_gate(only_docs, "public").ok
