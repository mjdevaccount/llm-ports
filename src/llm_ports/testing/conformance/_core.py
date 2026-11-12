"""What every conformance suite shares: the violation, and the helpers that raise it."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from llm_ports.testing._violation import ContractViolation


async def call[T](invariant: str, what: str, pending: Awaitable[T]) -> T:
    """Await one call; an exception is a never-raise violation."""
    try:
        return await pending
    except Exception as exc:
        raise ContractViolation(invariant, f"{what} raised {type(exc).__name__}: {exc}") from exc


def call_sync[T](invariant: str, what: str, fn: Callable[[], T]) -> T:
    """Run one synchronous call; an exception is a never-raise violation."""
    try:
        return fn()
    except Exception as exc:
        raise ContractViolation(invariant, f"{what} raised {type(exc).__name__}: {exc}") from exc


def expect(invariant: str, ok: bool, detail: str) -> None:
    """Raise `ContractViolation(invariant, detail)` unless `ok`."""
    if not ok:
        raise ContractViolation(invariant, detail)


def present[T](invariant: str, value: T | None, detail: str) -> T:
    """Return `value`, or raise `ContractViolation` when it is None."""
    if value is None:
        raise ContractViolation(invariant, detail)
    return value


def tag() -> str:
    """A fresh random tag, so a run never reads another run's rows."""
    return uuid.uuid4().hex[:12]


def is_instance(invariant: str, obj: Any, port: type) -> None:
    """Raise unless `obj` passes the port's runtime `isinstance` check."""
    expect(invariant, isinstance(obj, port), f"{type(obj).__name__} does not satisfy {port.__name__}")


__all__: list[str] = []
