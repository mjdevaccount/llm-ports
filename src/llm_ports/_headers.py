"""The one rule for a value sent as an HTTP header, a bearer secret above all (internal).

A header value holding whitespace, a control or a non-ASCII character is refused by the HTTP stack
when the request is sent, and the refusal quotes the header, value and all: in the exception
message and in httpcore's DEBUG trace. A secret read from a file with its trailing newline leaks
that way. Every place that puts a caller's secret in a header checks it here first, before any
request, and names the secret, never its value.

Invariants:
    - Pure, no I/O, no vendor import: the core (`call.providers.build`) and the adapters share it.
"""

from __future__ import annotations

#: What a refusal says is wrong, without the value.
HEADER_VALUE_RULE = "holds whitespace, a control or a non-ASCII character, so no header can carry it"


def header_value_ok(value: str) -> bool:
    """Whether `value` is visible ASCII only (`!` through `~`), so a header carries it as given. Pure."""
    return all("!" <= c <= "~" for c in value)


# Internal: callers import by path; it is no part of the public surface.
__all__: list[str] = []
