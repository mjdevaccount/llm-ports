"""The install line for a missing optional dependency (internal).

Every module that imports a vendor library does so inside `with extra("<name>"):`. Without the
extra installed, the import fails with the line that fixes it rather than a bare module name.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def extra(name: str) -> Iterator[None]:
    """Re-raise a `ModuleNotFoundError` inside the block with the extra that installs the module.

    Args:
        name: The `llm-ports` extra that provides the block's imports, e.g. `"pydantic-ai"`.

    Raises:
        ModuleNotFoundError: The block's import failed. The message names the missing module and
            the install line; `name` and the cause are kept.
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{exc.name} is not installed; install it with `pip install 'llm-ports[{name}]'`", name=exc.name
        ) from exc


# Internal: the adapters import `extra` by its path; it is no part of the public surface.
__all__: list[str] = []
