"""content -- the OUTPUT shape: one `ContentItem` per printed thing, and the rule that says who may see it.

The input side already has shapes (`Chunk`, `Claim`, `CitedAnswer`) and the process side has one
(`Role` + pack sha + `Generated.stamp`). This module is the third: what a consumer PRINTS, in the
W3C PROV-O vocabulary (https://www.w3.org/TR/prov-o/) so a lineage reader needs no translation:

    ContentItem  = prov:Entity            the printed thing (a card, a note, a brief edition)
    Activity     = prov:Activity          the generation that made it (trace, role, pack@sha, model, git sha)
    item.uses    = prov:used              the source ids / item ids it was made FROM
    item.as_of   = prov:generatedAtTime   when it was made (the time it speaks for)

`DataClass` is the rights + privacy class of a source or an item. An item's class is the JOIN of
the classes it uses: the most restrictive rights tier, private if any input is private, a link
owed if any input owes one. `print_gate(item, audience)` is the deterministic rule over all of it:
`ok`, or `refused` with ONE named reason. It never raises and never reads anything but the item.

Zero I/O, no framework. ODRL 2.2 supplies the words (permission/prohibition/duty), not an engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from llm_ports.ports import Chunk, Claim

#: Rights tiers, LEAST to MOST restrictive. The join of two classes takes the later one.
#:   public    free to print anywhere, cited (FRED non-copyright series, SEC EDGAR)
#:   owned     ours: the operator's own lists, journal, decisions, our signals
#:   licensed  a vendor licence for personal use (Tradier/OPRA non-pro, Yahoo RSS, vendor emails)
#:   scraped   taken from a page with no licence (yfinance, ForexFactory, mn.co, a web search)
RIGHTS: tuple[str, ...] = ("public", "owned", "licensed", "scraped")
Rights = Literal["public", "owned", "licensed", "scraped"]

#: Who an item is printed FOR, least to most exposed.
AUDIENCES: tuple[str, ...] = ("operator", "shared", "public")
Audience = Literal["operator", "shared", "public"]

#: The rights tiers each audience may see. `operator` sees everything; `shared` (a signed-in viewer
#: who is not the operator) and `public` see what we may redistribute, and never a private item.
CEILING: Mapping[str, frozenset[str]] = {
    "operator": frozenset(RIGHTS),
    "shared": frozenset({"public", "owned"}),
    "public": frozenset({"public", "owned"}),
}

#: Why `print_gate` refused, in the order it checks (the first that applies is the one reported):
#:   rights     the item's class is above the audience's ceiling (or it is private and the audience is not the operator)
#:   unclaimed  the body was never decomposed into claims, so nothing about its citations can be checked
#:   uncited    a claim cites nothing, or cites an id that is not in the item's evidence
#:   unsourced  a cited chunk carries no source or no as_of
#:   link       the class owes a link to the original and a cited chunk has none
REFUSALS: tuple[str, ...] = ("rights", "unclaimed", "uncited", "unsourced", "link")
Refusal = Literal["rights", "unclaimed", "uncited", "unsourced", "link"]


@dataclass(frozen=True)
class DataClass:
    """Rights tier + privacy + attribution duty. Order-comparable on rights through `rank`."""

    rights: str
    private: bool = False          # PII / the operator's personal data: never leaves the operator
    link_required: bool = False    # the licence owes a link back to the original (an ODRL duty)

    def __post_init__(self) -> None:
        if self.rights not in RIGHTS:
            raise ValueError(f"rights {self.rights!r} not in {RIGHTS}")

    @property
    def rank(self) -> int:
        return RIGHTS.index(self.rights)

    def join(self, *others: DataClass) -> DataClass:
        """The most restrictive of self and others: max rights, any private, any link owed."""
        every = (self, *others)
        return DataClass(rights=max(every, key=lambda c: c.rank).rights,
                         private=any(c.private for c in every),
                         link_required=any(c.link_required for c in every))

    def allows(self, audience: str) -> bool:
        if audience not in CEILING:
            raise ValueError(f"audience {audience!r} not in {AUDIENCES}")
        return self.rights in CEILING[audience] and (audience == "operator" or not self.private)


PUBLIC = DataClass("public")
OWNED = DataClass("owned")
LICENSED = DataClass("licensed")
SCRAPED = DataClass("scraped")


def join_all(classes: Iterable[DataClass]) -> DataClass | None:
    """The join of a possibly-empty set; None when there is nothing to join (UNKNOWN, never public)."""
    got = tuple(classes)
    return got[0].join(*got[1:]) if got else None


@dataclass(frozen=True)
class Activity:
    """prov:Activity -- one generation. Every field is optional: a deterministic writer has no
    model or pack, and an empty string says 'not recorded', never 'none'."""

    id: str = ""                   # the trace / audit id that names this generation
    role: str = ""                 # feature.action
    pack: str = ""
    pack_sha: str = ""
    model: str = ""
    git_sha: str = ""              # the code that ran it
    started_at: datetime | None = None
    ended_at: datetime | None = None

    def prov(self) -> dict[str, Any]:
        out: dict[str, Any] = {"prov:type": "prov:Activity"}
        if self.id:
            out["@id"] = self.id
        if self.started_at:
            out["prov:startedAtTime"] = self.started_at.isoformat()
        if self.ended_at:
            out["prov:endedAtTime"] = self.ended_at.isoformat()
        assoc = {k: v for k, v in (("role", self.role), ("pack", self.pack), ("pack_sha", self.pack_sha),
                                   ("model", self.model), ("git_sha", self.git_sha)) if v}
        if assoc:
            out["prov:qualifiedAssociation"] = assoc
        return out


@dataclass(frozen=True)
class ContentItem:
    """prov:Entity -- one printed thing.

    claims    the body decomposed into claims, or None when it never was (`print_gate` says
              `unclaimed` rather than guessing either way)
    evidence  id -> Chunk, the universe the claims may cite
    uses      prov:used -- the source ids (or item ids) it was made from
    verdict   the writer's own check (ok | degraded | refused ...), as written; None = none recorded
    status    what the reader is shown: ok | unavailable | ...
    """

    kind: str
    id: str
    data_class: DataClass
    as_of: datetime | date | None = None
    body: str = ""
    claims: tuple[Claim, ...] | None = None
    evidence: Mapping[str, Chunk] = field(default_factory=dict)
    uses: tuple[str, ...] = ()
    activity: Activity | None = None
    verdict: str | None = None
    status: str = "ok"

    def prov(self) -> dict[str, Any]:
        """The PROV-JSON-shaped view of this item (vocabulary only: no namespace document)."""
        out: dict[str, Any] = {"@id": f"{self.kind}:{self.id}", "prov:type": "prov:Entity",
                               "kind": self.kind, "data_class": self.data_class.rights,
                               "prov:used": list(self.uses)}
        if self.as_of is not None:
            out["prov:generatedAtTime"] = self.as_of.isoformat()
        if self.activity is not None:
            out["prov:wasGeneratedBy"] = self.activity.prov()
        return out


@dataclass(frozen=True)
class GateVerdict:
    ok: bool
    reason: str | None = None      # one of REFUSALS when not ok
    detail: str = ""


def print_gate(item: ContentItem, audience: str) -> GateVerdict:
    """PURE. May this item be printed for this audience? The first rule that fails is the reason."""
    if not item.data_class.allows(audience):
        why = "private" if item.data_class.private and audience != "operator" else item.data_class.rights
        return GateVerdict(False, "rights", f"{why} is above the {audience} ceiling")
    if item.claims is None:
        return GateVerdict(False, "unclaimed", "body has no claim decomposition")
    for i, c in enumerate(item.claims):
        if not c.evidence_ids:
            return GateVerdict(False, "uncited", f"claim {i} cites nothing")
        unknown = [e for e in c.evidence_ids if e not in item.evidence]
        if unknown:
            return GateVerdict(False, "uncited", f"claim {i} cites unknown id(s) {unknown}")
    cited = {e for c in item.claims for e in c.evidence_ids}
    for e in sorted(cited):
        ch = item.evidence[e]
        if not ch.source or ch.as_of is None:
            return GateVerdict(False, "unsourced", f"chunk {e} has no {'source' if not ch.source else 'as_of'}")
        if item.data_class.link_required and not ch.url:
            return GateVerdict(False, "link", f"chunk {e} owes a link and has none")
    return GateVerdict(True)
