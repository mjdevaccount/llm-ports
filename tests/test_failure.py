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


def test_with_cause_names_the_wrapper_and_its_cause_by_the_same_cut() -> None:
    try:
        try:
            raise ValueError("Invalid JSON: EOF while parsing a string")
        except ValueError as inner:
            raise RuntimeError("Exceeded maximum output retries (1)") from inner
    except RuntimeError as exc:
        r = failure_reason(exc, with_cause=True)
        assert failure_reason(exc) == "RuntimeError: Exceeded maximum output retries (1)"
    assert r == "RuntimeError: Exceeded maximum output retries (1) <- ValueError: Invalid JSON: EOF while parsing a string"


def test_with_cause_never_carries_the_cause_body_and_stays_within_the_cap() -> None:
    top = RuntimeError("x" * 1000)
    top.__cause__ = _Http("status_code: 502, body: {'message': 'secret'}")
    r = failure_reason(top, with_cause=True)
    assert r.endswith(" <- _Http: status 502") and "secret" not in r and len(r) <= REASON_MAX


def test_with_cause_and_no_cause_is_the_plain_reason() -> None:
    assert failure_reason(KeyError("k"), with_cause=True) == failure_reason(KeyError("k"))
