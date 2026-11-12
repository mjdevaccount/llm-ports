"""`StubGenerator`: a `Generator` that answers from a reply function, a constant, or a SCRIPT.

A script is a sequence of replies (`Reply.ok`, `Reply.problems`, `Reply.failed`, or a
`Generated`), one per model request, played in order. The stub runs the re-ask loop the real
adapter runs, so a test sees real `retries` and `usage_v2` summed over the re-asks. A script that
runs out raises `ContractViolation("script_exhausted")` in the call; one with replies left over
fails `assert_spent()` (the `stub_generator` fixture calls it at teardown).

Invariants:
    - `generate` never raises for a model failure; a script running out is a test failure and
      does raise.
    - A re-ask happens when the reply has problems (its own, or `validate`'s) and
      `req.output_retries` re-asks remain; after the last one the value is kept with its problems.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from llm_ports.ports import (
    Answered,
    Failure,
    FailureKind,
    Finish,
    Generated,
    GenRequest,
    Usage,
    failure,
)
from llm_ports.ports._usage import UnifiedFinish
from llm_ports.ports.outcomes import failure_reason
from llm_ports.testing._violation import ContractViolation

#: The model id a `StubGenerator` reports as having answered unless told otherwise; never the
#: request's id, since a real result's `answered.model` is what the response said.
STUB_MODEL = "stub-generator"


@dataclass(frozen=True, slots=True)
class ScriptedReply:
    """One scripted model reply: a value, with what the provider would have reported about it.

    Build one with `Reply.ok` or `Reply.problems`.

    Attributes:
        value: The parsed value.
        problems: Problems the reply carries regardless of `validate` (a validator not satisfied).
        usage: The request's usage; None is unmeasured.
        finish: How the reply ended; None (a bare 1.x reply value) reports no finish.
        answered_model: The model the reply says answered; None uses the stub's.
    """

    value: Any
    problems: tuple[Any, ...] = ()
    usage: Usage | None = None
    finish: UnifiedFinish | None = None
    answered_model: str | None = None


class _ReplyNamespace(type(Protocol)):  # type: ignore[misc]  # Protocol's metaclass, named at runtime
    """Gives `Reply` constructors that are not protocol members, so any plain function is a `Reply`."""

    def ok(
        cls,
        value: Any,
        *,
        usage: Usage | None = None,
        finish: UnifiedFinish = "stop",
        answered_model: str | None = None,
    ) -> ScriptedReply:
        """A reply that parses into `value`. Pure."""
        return ScriptedReply(value, (), usage, finish, answered_model)

    def problems(cls, value: Any, problems: Sequence[Any], *, usage: Usage | None = None) -> ScriptedReply:
        """A reply whose `value` the validator is not satisfied with: it earns a re-ask. Pure."""
        return ScriptedReply(value, tuple(problems), usage, "stop")

    def failed(cls, kind: FailureKind, **detail: Any) -> Failure:
        """Return the `Failure` a reply returns to fail its call with `kind`.

        Pure. `detail` is `ports.failure`'s keyword arguments (`status`, `retry_after_s`, ...).

        Raises:
            ValueError: `kind` is not a `FailureKind`.
        """
        return failure(kind, **detail)


class Reply(Protocol, metaclass=_ReplyNamespace):
    """A `StubGenerator` reply function: `(request, output type)` to a reply.

    A reply is a plain value, a `ScriptedReply` (`Reply.ok`, `Reply.problems`), a `Failure`
    (`Reply.failed(kind, **detail)`) or a whole `Generated`. Any function of that shape is a `Reply`.

    Example:
        gen = StubGenerator([Reply.problems("draft", ["too short"]), Reply.ok("final")])
    """

    def __call__(self, req: GenRequest, output: type[Any], /) -> Any:
        """Answer `req` for `output`."""
        ...


_REPLY_TYPES = (ScriptedReply, Failure, Generated)


def _is_script(reply: Any) -> bool:
    """A non-empty sequence of reply objects (never a str, never a plain list value) is a script."""
    if isinstance(reply, (str, bytes)) or not isinstance(reply, Sequence) or not reply:
        return False
    return all(isinstance(r, _REPLY_TYPES) for r in reply)


def _plus(total: Usage | None, more: Usage | None) -> Usage | None:
    if more is None:
        return total
    return more if total is None else total + more


class StubGenerator:
    """A `Generator` with no wire; the module docstring states the contract.

    Args:
        reply: A function of `(request, output type)`, a script (a sequence of reply objects), or
            any other value, returned for every request.
        model: The model id the stub reports in `answered.model`.

    Attributes:
        requests: Every `GenRequest` received, in call order (a re-ask does not add one).
    """

    def __init__(self, reply: Reply | Sequence[Any] | Any, *, model: str = STUB_MODEL) -> None:
        self._script: list[Any] | None = list(cast("Sequence[Any]", reply)) if _is_script(reply) else None
        self._played = 0
        if self._script is not None:
            self._reply: Reply = self._next
        elif callable(reply):
            self._reply = reply
        else:
            self._reply = lambda _req, _out: reply
        self._model = model
        self.requests: list[GenRequest] = []

    def _next(self, _req: GenRequest, _out: type[Any]) -> Any:
        script = self._script or []
        if self._played >= len(script):
            raise ContractViolation(
                "script_exhausted", f"request {self._played + 1} has no reply; the script holds {len(script)}"
            )
        self._played += 1
        return script[self._played - 1]

    @property
    def unused(self) -> tuple[Any, ...]:
        """The scripted replies no request has played yet; `()` for a reply function or a constant."""
        return tuple((self._script or [])[self._played :])

    def assert_spent(self) -> None:
        """Raise `ContractViolation("script_unused")` when scripted replies were never played."""
        if self.unused:
            raise ContractViolation(
                "script_unused", f"{len(self.unused)} of {len(self._script or [])} scripted replies never played"
            )

    def _ask(self, req: GenRequest, output: type[Any]) -> Any:
        try:
            return self._reply(req, output)
        except ContractViolation:
            raise
        except Exception as exc:
            return failure("server", reason=failure_reason(exc))

    async def generate[T](
        self,
        req: GenRequest,
        output: type[T],
        *,
        validate: Callable[[T], Sequence[Any]] | None = None,
        retry_message: Callable[[Sequence[Any]], str] | None = None,
    ) -> Generated[T]:
        """Answer `req`, re-asking while problems remain and re-asks are left; `retry_message` is unused."""
        del retry_message
        self.requests.append(req)
        total: Usage | None = None
        last = max(0, req.output_retries)
        for n in range(last + 1):
            got = self._ask(req, output)
            if isinstance(got, Generated):
                return got
            if isinstance(got, Failure):
                return _failed(req, got, n, total, Answered(requested=req.model, model=self._model))
            reply = got if isinstance(got, ScriptedReply) else ScriptedReply(got)
            total = _plus(total, reply.usage)
            answered = Answered(requested=req.model, model=reply.answered_model or self._model)
            try:
                found = tuple(validate(reply.value)) if validate is not None else ()
            except Exception as exc:  # a bug in the caller's validator is never a model outage
                return _failed(req, failure("internal", reason=failure_reason(exc)), n, total, answered)
            problems = reply.problems + found
            if problems and n < last:
                continue
            return Generated(
                value=reply.value,
                outcome="ok",
                model=req.model,
                problems=problems,
                retries=n,
                usage_v2=total,
                finish=Finish(reply.finish, reply.finish) if reply.finish is not None else None,
                answered=answered,
            )
        raise AssertionError("unreachable: the loop's last pass returns")  # pragma: no cover


def _failed(req: GenRequest, got: Failure, n: int, total: Usage | None, answered: Answered) -> Generated[Any]:
    return Generated(
        value=None,
        outcome=got.outcome,
        model=req.model,
        retries=n or None,
        usage_v2=total,
        answered=answered,
        failure=got,
    )


__all__ = ["STUB_MODEL", "Reply", "ScriptedReply", "StubGenerator"]
