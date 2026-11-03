from datetime import UTC, date, datetime

import pytest

from llm_ports import content as c
from llm_ports.ports import Chunk, Claim

CH = {"a": Chunk(id="a", text="t", source="edgar_8k", url="https://sec.gov/x", as_of=date(2026, 11, 2))}


def _item(**kw):
    base = {"kind": "brief", "id": "1", "data_class": c.OWNED, "claims": (Claim("x", ("a",)),), "evidence": CH}
    return c.ContentItem(**{**base, **kw})


def test_rights_are_ordered_and_the_join_takes_the_most_restrictive() -> None:
    assert [c.DataClass(r).rank for r in c.RIGHTS] == [0, 1, 2, 3]
    j = c.PUBLIC.join(c.LICENSED, c.DataClass("owned", private=True), c.DataClass("public", link_required=True))
    assert j == c.DataClass("licensed", private=True, link_required=True)
    assert c.join_all([]) is None  # nothing to join is UNKNOWN, never public
    assert c.join_all([c.SCRAPED, c.PUBLIC]) == c.SCRAPED


def test_an_unknown_rights_tier_or_audience_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        c.DataClass("free")
    with pytest.raises(ValueError):
        c.PUBLIC.allows("everyone")


@pytest.mark.parametrize(
    ("cls", "audience", "ok"),
    [
        (c.SCRAPED, "operator", True),
        (c.LICENSED, "operator", True),
        (c.LICENSED, "shared", False),
        (c.SCRAPED, "public", False),
        (c.OWNED, "shared", True),
        (c.PUBLIC, "public", True),
        (c.DataClass("owned", private=True), "shared", False),
        (c.DataClass("owned", private=True), "operator", True),
    ],
)
def test_the_ceiling(cls, audience, ok) -> None:
    assert cls.allows(audience) is ok


def test_a_fully_cited_item_passes() -> None:
    assert c.print_gate(_item(), "public") == c.GateVerdict(True)


@pytest.mark.parametrize(
    ("kw", "audience", "reason"),
    [
        ({"data_class": c.LICENSED}, "shared", "rights"),
        ({"data_class": c.DataClass("public", private=True)}, "public", "rights"),
        ({"claims": None}, "operator", "unclaimed"),
        ({"claims": (Claim("x"),)}, "operator", "uncited"),
        ({"claims": (Claim("x", ("zz",)),)}, "operator", "uncited"),
        ({"evidence": {"a": Chunk(id="a", text="t", as_of=date(2026, 1, 1))}}, "operator", "unsourced"),
        ({"evidence": {"a": Chunk(id="a", text="t", source="s")}}, "operator", "unsourced"),
        (
            {
                "data_class": c.DataClass("licensed", link_required=True),
                "evidence": {"a": Chunk(id="a", text="t", source="yahoo_rss", as_of=date(2026, 1, 1))},
            },
            "operator",
            "link",
        ),
    ],
)
def test_each_refusal_fires_for_its_own_reason(kw, audience, reason) -> None:
    v = c.print_gate(_item(**kw), audience)
    assert (v.ok, v.reason) == (False, reason)
    assert v.reason in c.REFUSALS and v.detail


def test_rights_outranks_every_citation_reason() -> None:
    assert c.print_gate(_item(data_class=c.SCRAPED, claims=None), "shared").reason == "rights"


def test_prov_uses_the_prov_o_vocabulary() -> None:
    at = datetime(2026, 11, 2, 21, tzinfo=UTC)
    act = c.Activity(id="tr-1", role="day_brief.edition", pack="day_brief", pack_sha="abc", started_at=at)
    p = _item(uses=("tear_feed",), activity=act, as_of=at).prov()
    assert p["prov:type"] == "prov:Entity" and p["@id"] == "brief:1" and p["prov:used"] == ["tear_feed"]
    assert p["prov:generatedAtTime"] == at.isoformat()
    g = p["prov:wasGeneratedBy"]
    assert g["@id"] == "tr-1" and g["prov:qualifiedAssociation"] == {
        "role": "day_brief.edition",
        "pack": "day_brief",
        "pack_sha": "abc",
    }
    assert "prov:endedAtTime" not in g  # unrecorded is absent, never a fake value
