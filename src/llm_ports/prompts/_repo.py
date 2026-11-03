"""`RepoPromptSource`: a `PromptSource` with no managed store; every name is its repo `Seed`."""

from __future__ import annotations

from collections.abc import Callable

from llm_ports.ports import Prompt, Seed


class RepoPromptSource:
    """A `PromptSource` for a process with no managed store.

    Every name is its repo seed (`seed_for`, the same callable a managed-store adapter mirrors),
    served as `Prompt(repo)` with the seed's config and system template: what a managed adapter
    serves when its store is down, without the round-trip.
    """

    def __init__(self, seed_for: Callable[[str], Seed]) -> None:
        self._seed_for = seed_for

    def get(self, name: str, *, label: str = "production") -> Prompt:
        """Return the repo seed for `name` as a prompt.

        `label` is accepted and ignored: a repo has one copy. A name without a seed raises from
        `seed_for` (the port's law: no seed, no text).
        """
        s = self._seed_for(name)
        return Prompt(name, s.text, None, "repo", config=dict(s.config), system=s.system)


__all__ = ["RepoPromptSource"]
