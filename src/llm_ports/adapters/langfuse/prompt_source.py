"""The `PromptSource` port over Langfuse prompt management.

Example:
    source = LangfusePromptSource(Langfuse(...), seed_for)
    prompt = source.get("summarize")  # Prompt(text, version, source="langfuse:v7" | "repo")

The repository seed leads; the managed label mirrors it and never leads. Per `get(name, label)`:

1. `get_prompt(name, label, fallback=<seed>, cache_ttl_seconds=ttl)` goes through the SDK's own
   cache (an expired entry is served and refreshed on a background thread).
2. If the reply is the fallback, or its `config.content_sha` is not the seed's sha, or the seed's
   config carries an `identity` the reply's config does not, the managed copy is absent or stale:
   the seed is pushed under the label with `create_prompt(...)` and refetched uncached.
3. If the reply is still the fallback, or its text is not the seed's, the seed is served as
   `Prompt(source="repo")`. Otherwise the result is `Prompt(source="langfuse:vN")` with the managed
   version and its stored config.

`seed_for(name) -> Seed` is injected: the adapter knows nothing about where a repository prompt
lives or how it is fingerprinted (`prompts.Pack.seed` builds one). With no `label` passed, `get`
serves under the seed's `config["label"]`, else `production`. A seed with a non-empty `system` (or `kind="chat"`) is a chat
prompt: pushed as `type="chat"` with the message list, fetched as chat, and compared message by
message on role and content. Langfuse refuses a type change on an existing name, so `kind` can
force chat for a seed with no system message.

The client is duck-typed: anything with the SDK's `get_prompt` / `create_prompt` signatures. The
caller builds it, so this module imports no SDK.

Invariants:
    - Fails soft: an unreachable store, a refused push or a malformed reply each serves the seed
      as `source == "repo"` and emits a warning event through `on_event` (default: the stdlib
      logger `llm_ports.adapters.langfuse.prompt_source`); none of them raises.
    - The SDK never caches a fallback reply, so the down verdict is held per name for one TTL
      here: a store that is down costs one round-trip per TTL, not one per call.

Rationale: ADR-0008.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from llm_ports.observability import EventHook, log_events
from llm_ports.ports import Prompt, Seed

DEFAULT_CACHE_TTL_S = 300
#: The label served when neither the caller nor the seed's config names one.
DEFAULT_LABEL = "production"
EVENT = "langfuse_prompt_source"
_ERR_CAP = 200


def _kind(seed: Seed) -> str:
    """Return "chat" or "text": the seed's forced `kind`, else "chat" exactly when it has a system message."""
    return seed.kind or ("chat" if seed.system else "text")


def _wire(seed: Seed) -> str | list[dict[str, str]]:
    """Return what the store holds for a seed: the text, or the chat message list.

    The list has a system message only when the seed has one, so a system-less chat seed is a
    one-message list, never "".
    """
    if _kind(seed) == "text":
        return seed.text
    msgs = [{"role": "system", "content": seed.system}] if seed.system else []
    return [*msgs, {"role": "user", "content": seed.text}]


def _served(p: Any) -> Any:
    """Return a reply's prompt in `_wire`'s shape.

    A chat reply is reduced to role and content per message (the SDK may carry more keys). A
    message that is not a mapping raises, and `LangfusePromptSource.get` then serves the seed.
    """
    body = p.prompt
    if isinstance(body, list):
        return [{"role": m["role"], "content": m["content"]} for m in body]
    return body


def _stale(config: Any, seed: Seed) -> bool:
    """True when the managed copy's config names another sha, or another identity than the seed's."""
    if config.get("content_sha") != seed.sha:
        return True
    return "identity" in seed.config and config.get("identity") != seed.config["identity"]


def _is_fallback(p: Any) -> bool:
    return bool(getattr(p, "is_fallback", False))


class LangfusePromptSource:
    """`PromptSource` whose managed store is Langfuse.

    One instance per client; the SDK caches.

    Args:
        client: A Langfuse client, or anything with its `get_prompt` / `create_prompt` signatures.
        seed_for: Returns the repository seed for a prompt name.
        cache_ttl_s: The SDK cache TTL, and how long a down verdict is held per name.
        clock: A monotonic clock, injectable for tests.
        on_event: The warning sink; defaults to the stdlib logger `llm_ports.adapters.langfuse.prompt_source`.
    """

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
        self._on_event = on_event or log_events(__name__)
        self._down_until: dict[str, float] = {}

    def _down(self, name: str) -> bool:
        return self._clock() < self._down_until.get(name, 0.0)

    def _fetch(self, name: str, label: str, seed: Seed, ttl: int) -> Any:
        return self._client.get_prompt(name, label=label, type=_kind(seed), fallback=_wire(seed), cache_ttl_seconds=ttl)

    def get(self, name: str, *, label: str | None = None) -> Prompt:
        """Return prompt `name`, mirrored to the managed store under `label`.

        Serves the managed version when it matches the seed, else the seed as `source == "repo"`;
        the module docstring gives the steps. A store failure never raises.

        Args:
            name: The prompt name.
            label: The managed label; None takes the seed's `config["label"]`, else `production`.
        """
        seed = self._seed_for(name)
        label = label or str(seed.config.get("label") or DEFAULT_LABEL)
        repo = Prompt(name, seed.text, None, "repo", config=dict(seed.config), system=seed.system)
        if self._down(name):
            return repo
        want = _wire(seed)
        try:
            p = self._fetch(name, label, seed, self._ttl)
            if _is_fallback(p) or _stale(p.config or {}, seed):
                self._client.create_prompt(
                    name=name,
                    type=_kind(seed),
                    prompt=want,
                    labels=[label],
                    tags=list(seed.tags),
                    config={**seed.config, "content_sha": seed.sha},
                )
                p = self._fetch(name, label, seed, 0)
            got = _served(p)
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
        # `got == want`, so the seed's text and system are the managed copy's: serve them with its version.
        return Prompt(
            name, seed.text, int(p.version), f"langfuse:v{p.version}", config=dict(p.config or {}), system=seed.system
        )


__all__ = ["DEFAULT_CACHE_TTL_S", "DEFAULT_LABEL", "EVENT", "LangfusePromptSource"]
