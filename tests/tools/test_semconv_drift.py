"""tools/semconv_drift.py: no drift, drift (renamed and deprecated), and could-not-look are three exits."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "semconv_drift.py"
_spec = importlib.util.spec_from_file_location("semconv_drift", _PATH)
assert _spec is not None and _spec.loader is not None
drift_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drift_tool)


def _all_defined(extra: str = "") -> str:
    names = sorted(drift_tool.pinned())
    return "attributes:\n" + "".join(f"  - key: {n}\n    brief: x\n" for n in names) + extra


def test_every_pinned_name_defined_is_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert drift_tool.main(lambda _p: _all_defined()) == 0


def test_a_renamed_name_is_exit_1_and_printed(capsys: pytest.CaptureFixture[str]) -> None:
    # MUST FIRE: the pinned-names unit test still passes on a rename (it tests our constants).
    text = _all_defined().replace("key: gen_ai.usage.cache_read.input_tokens", "key: gen_ai.usage.cache.read")
    assert drift_tool.main(lambda _p: text) == 1
    assert capsys.readouterr().out.split() == ["gen_ai.usage.cache_read.input_tokens"]


def test_a_deprecated_name_is_drift() -> None:
    text = _all_defined().replace(
        "  - key: gen_ai.response.id\n    brief: x\n",
        "  - key: gen_ai.response.id\n    deprecated:\n      reason: renamed\n",
    )
    assert drift_tool.drift([text], drift_tool.pinned()) == ["gen_ai.response.id"]


def test_could_not_look_is_exit_2_never_no_drift() -> None:
    def down(_p: str) -> str:
        raise OSError("network down")

    assert drift_tool.main(down) == 2
    assert drift_tool.main(lambda _p: "nothing parsable") == 2
