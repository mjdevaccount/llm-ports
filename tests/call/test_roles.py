"""roles: the tier table is the consumer's DATA, checked at load; a Role binds one by subclass."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import pytest

from llm_ports.call import roles
from tests.conftest import FIXTURES, TABLE, Role


def test_the_table_resolves_models_and_rungs() -> None:
    assert TABLE.tiers["local"] == roles.Tier("local", "small-chat", 120.0, True)
    assert TABLE.tiers["local-web"].timeout_s == 260.0
    assert TABLE.tiers["cloud"].self_hosted is False


def test_an_unknown_model_or_rung_is_refused_at_load(tmp_path: Path) -> None:
    bad = tmp_path / "t.toml"
    bad.write_text('[ladder]\na = 1\n[models]\nm = "x"\n[tiers.t]\nmodel = "nope"\ntimeout = "a"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="tier 't' names model 'nope'"):
        roles.load_tiers(bad)


def test_the_consumers_ladder_check_runs_and_can_refuse() -> None:
    seen: list[dict[str, float]] = []

    def must_clear(ladder, path):
        seen.append(dict(ladder))
        if not ladder["web_deadline_s"] < ladder["web_timeout_s"]:
            raise ValueError(f"{path.name}: web_timeout_s must clear web_deadline_s")

    assert roles.load_tiers(FIXTURES / "llm_tiers.toml", check_ladder=must_clear).tiers
    assert seen and seen[0]["web_deadline_s"] == 240.0

    def refuse(_ladder, path):
        raise ValueError(f"{path.name}: must clear")

    with pytest.raises(ValueError, match="must clear"):
        roles.load_tiers(FIXTURES / "llm_tiers.toml", check_ladder=refuse)


def test_the_base_class_has_no_table_so_cannot_be_instantiated() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        roles.Role("a.b", tier="local")


def test_two_consumers_never_share_a_table() -> None:
    class Other(roles.Role):
        TIERS: ClassVar[Mapping[str, roles.Tier]] = {"solo": roles.Tier("solo", "m", 1.0, self_hosted=False)}

    assert Other("a.b", tier="solo").resolved_model == "m"
    with pytest.raises(ValueError, match="unknown tier"):
        Other("a.b", tier="local")
    with pytest.raises(ValueError, match="unknown tier"):
        Role("a.b", tier="solo")


def test_an_unknown_tier_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown tier 'huge'"):
        Role("x.y", tier="huge")


def test_category_defaults_to_the_name_and_overrides_win() -> None:
    assert Role("a.b", tier="local").category == "a.b"
    r = Role("a.b", tier="local", model="pinned", timeout_s=5.0, category="c.d")
    assert (r.category, r.resolved_model, r.resolved_timeout_s) == ("c.d", "pinned", 5.0)


def test_cloud_tiers_never_send_the_thinking_kwarg() -> None:
    assert "extra_body" not in Role("t.cloud", tier="cloud", max_tokens=5).model_settings()
    boxed = Role("t.box", tier="local", thinking=True, settings={"extra_body": {"keep": 1}}).model_settings()
    assert boxed["extra_body"] == {"keep": 1, "chat_template_kwargs": {"enable_thinking": True}}


def test_model_settings_carry_budget_sampling_and_extras() -> None:
    s = Role("t.s", tier="cloud", max_tokens=300, temperature=0.2, settings={"seed": 7}).model_settings()
    assert s == {"seed": 7, "max_tokens": 300, "temperature": 0.2}


TABLE_AS_MAPPING = {
    "ladder": {"fast": 30.0},
    "models": {"small": "gpt-5-nano"},
    "tiers": {"default": {"model": "small", "timeout": "fast", "self_hosted": False}},
}


def test_a_table_loads_from_a_mapping_as_from_a_file() -> None:
    table = roles.load_tiers(TABLE_AS_MAPPING)
    assert table.tiers["default"] == roles.Tier("default", "gpt-5-nano", 30.0, self_hosted=False)


def test_a_mapping_is_named_as_the_mapping_source_in_checks_and_errors() -> None:
    seen: list[Path] = []
    roles.load_tiers(TABLE_AS_MAPPING, check_ladder=lambda _ladder, path: seen.append(path))
    assert seen == [roles.MAPPING_SOURCE]
    bad = {**TABLE_AS_MAPPING, "tiers": {"t": {"model": "nope", "timeout": "fast"}}}
    with pytest.raises(ValueError, match=r"^<mapping>: tier 't' names model 'nope'"):
        roles.load_tiers(bad)


def test_self_hosted_defaults_to_true() -> None:
    table = roles.load_tiers({**TABLE_AS_MAPPING, "tiers": {"t": {"model": "small", "timeout": "fast"}}})
    assert table.tiers["t"].self_hosted is True


@pytest.mark.parametrize("box", [True, False])
def test_the_old_box_key_reads_as_self_hosted_with_a_deprecation_warning(tmp_path: Path, box: bool) -> None:
    old = tmp_path / "old.toml"
    old.write_text(
        f'[ladder]\na = 1\n[models]\nm = "x"\n[tiers.t]\nmodel = "m"\ntimeout = "a"\nbox = {str(box).lower()}\n',
        encoding="utf-8",
    )
    with pytest.warns(
        DeprecationWarning, match="old.toml: tier 't' key 'box' is deprecated; rename it to 'self_hosted'"
    ) as w:
        table = roles.load_tiers(old)
    assert table.tiers["t"].self_hosted is box
    assert w[0].filename == __file__, "the warning points at the caller of load_tiers"


def test_the_old_box_key_beside_self_hosted_is_refused() -> None:
    both = {**TABLE_AS_MAPPING, "tiers": {"t": {"model": "small", "timeout": "fast", "box": True, "self_hosted": True}}}
    with pytest.raises(ValueError, match="sets both self_hosted and its old name 'box'"):
        roles.load_tiers(both)
