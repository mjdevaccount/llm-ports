"""Diff the pinned GenAI names in `observability/semconv.py` against the convention's HEAD.

`semconv.py` pins every attribute and metric name to one commit of
open-telemetry/semantic-conventions-genai, and the unit tests test OUR constants, so an upstream
rename passes them. This script is the outside witness: it fetches the registry and metric files
at the default branch's HEAD and reports every pinned `gen_ai.*` name HEAD no longer defines (or
marks deprecated).

Run by `.github/workflows/semconv-drift.yml` on a schedule; on drift the workflow opens an issue.

Exit codes:
    0  no drift: every pinned name is defined at HEAD.
    1  drift: the names are printed, one per line.
    2  could not look (a fetch failed, a file did not parse): never read as "no drift".
"""

from __future__ import annotations

import re
import sys
import urllib.request
from collections.abc import Callable, Iterable

from llm_ports.observability import semconv

REPO = semconv.SEMCONV_SOURCE.split("@", 1)[0]
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
TIMEOUT_S = 30
_DEFINED = re.compile(r"^\s*-\s*(?:key|name):\s*([\w.]+)\s*$", re.MULTILINE)
_DEPRECATED = re.compile(
    r"^\s*-\s*(?:key|name):\s*([\w.]+)\s*$(?:(?!^\s*-\s*(?:key|name):).)*?^\s*deprecated:", re.M | re.S
)


def fetch(path: str, ref: str = "main") -> str:
    """Return one file of the convention repo at `ref`."""
    url = RAW.format(repo=REPO, ref=ref, path=path)
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:  # a fixed https URL
        return str(resp.read().decode("utf-8"))


def pinned() -> set[str]:
    """The pinned names the GenAI repo owns (`error.type` lives in the general registry)."""
    return {n for n in semconv.CONVENTION_ATTRIBUTES | semconv.METRICS if n.startswith("gen_ai.")}


def drift(texts: Iterable[str], names: set[str]) -> list[str]:
    """The pinned names not defined, or defined as deprecated, across `texts`. Pure."""
    blob = "\n".join(texts)
    defined = set(_DEFINED.findall(blob))
    if not defined:
        raise ValueError("no `- key:` or `- name:` entries parsed: the file layout changed")
    deprecated = set(_DEPRECATED.findall(blob))
    return sorted(n for n in names if n not in defined or n in deprecated)


def main(get: Callable[[str], str] = fetch) -> int:
    """Print the drift and return the exit code (see the module docstring)."""
    try:
        texts = [get(p) for p in semconv.SEMCONV_FILES]
        found = drift(texts, pinned())
    except Exception as exc:
        print(f"could not check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for name in found:
        print(name)
    print(f"{len(found)} of {len(pinned())} pinned names drifted from {REPO}@main", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
