"""failure: the persisted reason line, and the exception chain the outcome map walks."""

from __future__ import annotations

from llm_ports.failure import REASON_MAX, exception_chain, failure_reason


class _Http(Exception):
    status_code = 502


def test_a_status_carrying_error_is_class_and_status_only() -> None:
    assert failure_reason(_Http("status_code: 502, body: {'message': 'secret'}")) == "_Http: status 502"


def test_the_body_is_cut_and_the_line_capped() -> None:
    assert failure_reason(RuntimeError("reset, body: {'x': 1}")) == "RuntimeError: reset"
    assert len(failure_reason(ValueError("x" * 1000))) == REASON_MAX


def test_the_chain_walks_cause_then_context_and_stops_on_a_cycle() -> None:
    root = KeyError("root")
    mid = RuntimeError("mid")
    mid.__cause__ = root
    top = ValueError("top")
    top.__context__ = mid
    root.__context__ = top
    assert [type(e).__name__ for e in exception_chain(top)] == ["ValueError", "RuntimeError", "KeyError"]
