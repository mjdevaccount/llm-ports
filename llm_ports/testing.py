"""testing -- the doubles a consumer's tests need, so each one does not hand-roll its own.

    gen = StubGenerator(lambda req, output: {"category": "groceries"})
    tracer = RecordingTracer()
    res = await call.generate(ROLE, prompt="...", output=Reply, generator=gen,
                              hooks=Hooks(trace=tracer_trace(tracer)))
    assert gen.requests[0].model == ROLE.resolved_model
    assert tracer.generations[0].ended["usage"] is None

`StubGenerator` satisfies the `Generator` port without a framework: its reply function returns a
VALUE (wrapped `ok`), a `Generated` (returned as-is, any outcome), or raises (returned as
`unavailable` -- the port never raises). `RecordingTracer` satisfies `Tracer` and keeps every
generation, its end and its scores. Both pass the ports' `isinstance` checks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_ports.failure import failure_reason
from llm_ports.ports import Generated, GenRequest

Reply = Callable[[GenRequest, type[Any]], Any]


class StubGenerator:
    """A `Generator` over a reply function. Records every request in `requests`. `validate` runs on
    an `ok` value and its problems are kept (no re-ask: a stub has no model to re-ask)."""

    def __init__(self, reply: Reply | Any) -> None:
        self._reply: Reply = reply if callable(reply) else (lambda _req, _out: reply)
        self.requests: list[GenRequest] = []

    async def generate[T](self, req: GenRequest, output: type[T], *,
                          validate: Callable[[T], Sequence[Any]] | None = None,
                          retry_message: Callable[[Sequence[Any]], str] | None = None) -> Generated[T]:
        del retry_message
        self.requests.append(req)
        try:
            got = self._reply(req, output)
        except Exception as exc:
            return Generated(value=None, outcome="unavailable", model=req.model,
                             reason=failure_reason(exc), retries=None)
        if isinstance(got, Generated):
            return got
        problems = tuple(validate(got)) if validate is not None else ()
        return Generated(value=got, outcome="ok", model=req.model, problems=problems)


@dataclass
class RecordedGeneration:
    """One `start_generation` call: what it was opened with, how it ended, what it was scored."""

    opened: dict[str, Any]
    ended: dict[str, Any] | None = None
    scores: list[tuple[str, float, str | None]] = field(default_factory=list)

    def end(self, *, output: Any = None, usage: Mapping[str, int | None] | None = None,
            level: str = "DEFAULT", status_message: str | None = None,
            metadata: Mapping[str, Any] | None = None) -> None:
        self.ended = {"output": output, "usage": usage, "level": level,
                      "status_message": status_message, "metadata": metadata}

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        self.scores.append((name, value, comment))


class RecordingTracer:
    """A `Tracer` that keeps every generation in `generations`, in open order."""

    def __init__(self) -> None:
        self.generations: list[RecordedGeneration] = []

    def start_generation(self, *, name: str, model: str, provider: str, input: Any,
                         metadata: Mapping[str, Any], session_id: str | None,
                         prompt_name: str | None) -> RecordedGeneration:
        g = RecordedGeneration(opened={"name": name, "model": model, "provider": provider, "input": input,
                                       "metadata": dict(metadata), "session_id": session_id,
                                       "prompt_name": prompt_name})
        self.generations.append(g)
        return g


__all__ = ["RecordedGeneration", "RecordingTracer", "Reply", "StubGenerator"]
