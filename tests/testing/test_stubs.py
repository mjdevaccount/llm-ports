"""The test doubles keep the laws of the ports they stand in for."""

from __future__ import annotations

from llm_ports import ports
from llm_ports.ports import Chunk
from llm_ports.testing import StubCiter

SPAN = Chunk(id="a", text="SPY closed at 580.5, up 3%", title="Close")


async def test_the_stub_citer_keeps_the_citer_laws() -> None:
    c = StubCiter()
    assert isinstance(c, ports.Citer)
    empty = await c.answer("q", [])
    assert empty.claims == () and empty.unknown_ids() == ()
    ans = await c.answer("q", [SPAN])
    assert ans.unknown_ids() == () and set(ans.chunks) == {"a"} and ans.model == ""
