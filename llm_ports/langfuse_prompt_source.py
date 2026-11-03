"""LangfusePromptSource -- the `PromptSource` port over Langfuse prompt management.

    source = LangfusePromptSource(Langfuse(...), seed_for)
    prompt = source.get("summarize")          # Prompt(text, version, source="langfuse:v7" | "repo")

THE REPO LEADS, the managed label MIRRORS, never leads. Per `get(name, label)`:

  1. `get_prompt(name, label, fallback=<seed>, cache_ttl_seconds=ttl)` -- the SDK's own cache
     (an expired entry is served and refreshed on a background thread).
  2. If the reply is the fallback, or its `config.content_sha` is not the seed's sha, the managed
     copy is absent or stale: `create_prompt(...)` the seed under the label, refetch uncached.
  3. Still the fallback, or text that is not the seed's: serve the SEED as `Prompt(source="repo")`.
     Otherwise `Prompt(source="langfuse:vN")` with the managed version and its stored config.

`seed_for(name) -> Seed` is injected: the adapter knows nothing about where a repo prompt lives or
how it is fingerprinted. A seed with a non-empty `system` (or `kind="chat"`) is a CHAT prompt:
pushed as `type="chat"` with the message list, fetched as chat, compared message for message on
role + content. Langfuse refuses a type change on an existing name, which is why `kind` can force
chat for a seed with no system message.

The client is duck-typed: anything with the SDK's `get_prompt` / `create_prompt` signatures. Its
construction is the consumer's, so this module imports no SDK.

FAILS SOFT by contract: unreachable, a refused push, a malformed reply -- each serves the seed as
`source == "repo"` plus a warning event through `on_event` (default: the stdlib logger
`llm_ports.langfuse_prompt_source`), never a raise. The SDK never caches a fallback reply, so the
down-verdict is held per name for one ttl here: a store that is down costs one round-trip per ttl,
not one per call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from llm_ports.call import EventHook
from llm_ports.ports import Prompt, Seed

DEFAULT_CACHE_TTL_S = 300
EVENT = "langfuse_prompt_source"
_ERR_CAP = 200
_log = logging.getLogger("llm_ports.langfuse_prompt_source")


def _log_event(level: int, event: str, fields: Mapping[str, Any]) -> None:
    _log.log(level, "%s %s", event, dict(fields))


def kind(seed: Seed) -> str:
    """ "chat" or "text": the seed's forced `kind`, else chat exactly when it has a system message."""
    return seed.kind or ("chat" if seed.system else "text")


def wire(seed: Seed) -> str | list[dict[str, str]]:
    """What the store holds for this seed: the text, or the chat list -- a system message only
    when the seed has one, so a system-less chat seed is a one-message list, never ""."""
    if kind(seed) == "text":
        return seed.text
    msgs = [{"role": "system", "content": seed.system}] if seed.system else []
    return [*msgs, {"role": "user", "content": seed.text}]


def served(p: Any) -> Any:
    """A reply's prompt in `wire`'s shape: a chat reply compared on role + content only (the SDK may
    carry more keys per message). A message that is not a mapping RAISES (-> the seed serves)."""
    body = p.prompt
    if isinstance(body, list):
        return [{"role": m["role"], "content": m["content"]} for m in body]
    return body


def _is_fallback(p: Any) -> bool:
    return bool(getattr(p, "is_fallback", False))


class LangfusePromptSource:
    """`PromptSource` whose managed store is Langfuse. One instance per client; the SDK caches."""

    def __init__(
        self,
        client: Any,
        seed_for: Callable[[str], Seed],
        *,
        cache_ttl_s: int = DEFAULT_CACHE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
        on_event: EventHook | None = None,
    ) -> None:
        self._client = client
        self._seed_for = seed_for
        self._ttl = cache_ttl_s
        self._clock = clock
        self._on_event = on_event or _log_event
        self._down_until: dict[str, float] = {}

    def _down(self, name: str) -> bool:
        return self._clock() < self._down_until.get(name, 0.0)

    def _fetch(self, name: str, label: str, seed: Seed, ttl: int) -> Any:
        return self._client.get_prompt(name, label=label, type=kind(seed), fallback=wire(seed), cache_ttl_seconds=ttl)

    def get(self, name: str, *, label: str = "production") -> Prompt:
        seed = self._seed_for(name)
        repo = Prompt(name, seed.text, None, "repo", config=dict(seed.config), system=seed.system)
        if self._down(name):
            return repo
        want = wire(seed)
        try:
            p = self._fetch(name, label, seed, self._ttl)
            if _is_fallback(p) or (p.config or {}).get("content_sha") != seed.sha:
                self._client.create_prompt(
                    name=name,
                    type=kind(seed),
                    prompt=want,
                    labels=[label],
                    tags=list(seed.tags),
                    config={**seed.config, "content_sha": seed.sha},
                )
                p = self._fetch(name, label, seed, 0)
            got = served(p)
        except Exception as exc:  # down or refusing: the seed serves, visibly
            self._on_event(
                logging.WARNING,
                EVENT,
                {
                    "action": "mirror_failed",
                    "name": name,
                    "label": label,
                    "error": f"{type(exc).__name__}: {exc}"[:_ERR_CAP],
                },
            )
            self._down_until[name] = self._clock() + self._ttl
            return repo
        if _is_fallback(p) or got != want:
            self._down_until[name] = self._clock() + self._ttl
            return repo
        self._down_until.pop(name, None)
        # `got == want`, so the seed's text/system ARE the managed copy's: serve them with its version.
        return Prompt(
            name, seed.text, int(p.version), f"langfuse:v{p.version}", config=dict(p.config or {}), system=seed.system
        )


__all__ = ["DEFAULT_CACHE_TTL_S", "EVENT", "LangfusePromptSource", "kind", "served", "wire"]
