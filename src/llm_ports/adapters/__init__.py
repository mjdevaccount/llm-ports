"""Adapters: every module that imports a vendor library lives here, one extra each.

Import an adapter by its path (`llm_ports.adapters.pydantic_ai`, `llm_ports.adapters.langfuse`,
...); nothing here is imported by the rest of the package.
"""

__all__: list[str] = []
