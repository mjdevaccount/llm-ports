r"""`byte_sha`: one sha256 over a list of named byte strings, length-prefixed so it cannot collide.

Each part is hashed as `name + NUL + decimal length of data + NUL + data`, in the given order, and
the hex digest is cut to `n` characters. The length prefix is what `content_sha` lacks: there,
`["a\nb", "c"]` and `["a", "b\nc"]` join to the same text; here no two part lists that differ by
re-splitting, renaming or reordering hash alike.

Invariants:
    - Pure: no I/O.
    - The algorithm never changes; tests pin it on fixed inputs (a pack's `identity` is built on it
      and persisted beside output rows).

Rationale: ADR-0026.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

_NUL = b"\0"


def byte_sha(parts: Sequence[tuple[str, bytes]], *, n: int = 16) -> str:
    """Return sha256 over the named parts, length-prefixed, in order; the first `n` hex characters.

    Args:
        parts: `(name, data)` pairs. The order is part of the input: the caller sorts when the
            order carries no meaning.
        n: Hex characters kept, at most 64.

    Returns:
        The digest, `n` lowercase hex characters.

    Raises:
        ValueError: A part name contains NUL, which would make the encoding ambiguous.
    """
    h = hashlib.sha256()
    for name, data in parts:
        if "\0" in name:
            raise ValueError(f"part name {name!r} contains NUL")
        h.update(name.encode() + _NUL + str(len(data)).encode() + _NUL + data)
    return h.hexdigest()[:n]


__all__ = ["byte_sha"]
