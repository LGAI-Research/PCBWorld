"""Rule YAML schemas — ``RewardRuleSchema`` / ``MaskingRuleSchema`` / ``DrcRuleSchema``.

Every tracked rule file in the public tree loads complete, every legacy rule name resolves, and a rule with an unknown, missing or
mistyped key — including a typo in a masking ``when:`` — is refused with a message naming the
file and the key. An untracked rule file that predates the schema is migrated the way the old
loader read it (``configs/ckpt_migrations.yaml``) with a warning; a tracked one never is.
"""

from __future__ import annotations

from typing import Any, Literal

import warnings

import pytest
import yaml

from configs.loader import _is_tracked, check_mapping
from pcb_world.core import masking, reward_config
from pcb_world.engine.drc_config import DrcRuleSchema, load_drc_config
from tests._rule_helpers import (
    LOADERS, REPO, RULE_PENALTIES, tracked_rule_files, yaml_lines,
)

RULE_FILES = tracked_rule_files("configs", "experiments", "tests", "pcb_world", "methods", "eval")
SHIPPED = {
    "reward": REPO / "configs" / "reward" / "pcbworld_reward.yaml",
    "masking": REPO / "configs" / "masking" / "pcbworld_masking.yaml",
    "drc": REPO / "configs" / "drc" / "pcbworld_drc.yaml",
}


@pytest.mark.parametrize("kind,path", RULE_FILES, ids=[str(p.relative_to(REPO)) for _, p in RULE_FILES])
def test_every_tracked_rule_file_is_complete(kind, path):
    LOADERS[kind](path)


def test_every_rule_kind_has_tracked_files():
    assert {kind for kind, _ in RULE_FILES} == set(LOADERS)
    assert {p for _, p in RULE_FILES} >= set(SHIPPED.values())


@pytest.mark.parametrize("name", sorted(reward_config._NAME_COMPAT))
def test_legacy_reward_names_resolve(name):
    assert reward_config.get_reward_config(name).name == reward_config._NAME_COMPAT[name]


@pytest.mark.parametrize("name", sorted(masking._NAME_COMPAT))
def test_legacy_masking_names_resolve(name):
    assert masking.get_masking_rule(name).name == masking._NAME_COMPAT[name]


def _broken(tmp_path, kind, edit):
    """Write the shipped *kind* rule with *edit* applied to its parsed dict; return the path."""
    raw = yaml.safe_load(SHIPPED[kind].read_text())
    edit(raw)
    path = tmp_path / f"broken_{kind}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _assert_refused(kind, path, *needles):
    with pytest.raises(ValueError) as exc:
        LOADERS[kind](path)
    for needle in (str(path), *needles):
        assert needle in str(exc.value)


@pytest.mark.parametrize("kind,edit,key", [
    ("reward", lambda r: r.update(potentail=r["potential"]), "potentail"),
    ("reward", lambda r: r["potential"].update(via_pnalty=0.1), "via_pnalty"),
    ("masking", lambda r: r.update(action={}), "action"),
    ("masking", lambda r: r["actions"].update(net_selct={"when": {}}), "net_selct"),
    ("drc", lambda r: r["global_minima"].update(clearence=0.2), "clearence"),
], ids=["reward-top", "reward-potential", "masking-top", "masking-action", "drc-minima"])
def test_unknown_key_is_refused(tmp_path, kind, edit, key):
    _assert_refused(kind, _broken(tmp_path, kind, edit), "unknown key", key)


# Keys with a legacy fill in configs/ckpt_migrations.yaml.
FILLED_KEY_CASES = [
    ("reward", lambda r: r.pop("mode"), "mode"),
    ("reward", lambda r: r["potential"].pop("mask_reject_penalty"), "mask_reject_penalty"),
    ("masking", lambda r: r["actions"]["make_line"].pop("when"), "when"),
    ("masking", lambda r: r.pop("actions"), "actions"),
]
FILLED_KEY_IDS = ["reward-mode", "reward-penalty", "masking-when", "masking-actions"]


# (A reward rule without `potential:` is the old format: converted below, refused when tracked.)
def test_missing_key_is_refused(tmp_path):
    _assert_refused("drc", _broken(tmp_path, "drc", lambda r: r["global_minima"].pop("hole_to_hole")),
                    "missing required key", "hole_to_hole")


@pytest.mark.parametrize("kind,edit,key", FILLED_KEY_CASES, ids=FILLED_KEY_IDS)
def test_tracked_rule_file_is_never_filled(tmp_path, monkeypatch, kind, edit, key):
    monkeypatch.setattr("configs.loader._is_tracked", lambda path: True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _assert_refused(kind, _broken(tmp_path, kind, edit), "missing required key", key)


@pytest.mark.parametrize("kind,edit,key", FILLED_KEY_CASES, ids=FILLED_KEY_IDS)
def test_untracked_rule_file_gets_legacy_fill_with_warning(tmp_path, kind, edit, key):
    path = _broken(tmp_path, kind, edit)
    with pytest.warns(UserWarning, match="update the file to match") as record:
        LOADERS[kind](path)
    message = str(record[0].message)
    assert str(path) in message and key in message


def test_legacy_fill_restores_old_loader_defaults(tmp_path):
    # An old checkpoint's rule file without the penalties loads with the values the loader used to default to.
    path = tmp_path / "old_reward.yaml"
    path.write_text("potential:\n  completion_bonus: 5.0\n")
    with pytest.warns(UserWarning, match="name.*mode.*potential.step_penalty"):
        cfg = reward_config.load_reward_config(path)
    assert (cfg.name, cfg.mode) == ("yaml_custom", "terminal")
    assert {k: getattr(cfg, k) for k in RULE_PENALTIES} == RULE_PENALTIES
    path = tmp_path / "old_masking.yaml"
    path.write_text("actions:\n  net_select: {}\n")
    with pytest.warns(UserWarning, match=r"actions\.net_select\.when"):
        rule = masking.load_masking_rule(path)
    assert rule.build_mask(masking.MaskContext(has_net=True, is_routing=True))[masking.ACT_NET_SELECT]


def test_legacy_masking_null_action_and_missing_actions_read_as_the_old_loader(tmp_path):
    # The old loader read a null action entry as always enabled and a missing ``actions:`` as all masked.
    path = tmp_path / "null_action.yaml"
    path.write_text("name: m\nactions:\n  net_select:\n")
    with pytest.warns(UserWarning, match=r"actions\.net_select\.when"):
        mask = masking.load_masking_rule(path).build_mask(masking.MaskContext(has_net=True, is_routing=True))
    assert mask.tolist() == [i == masking.ACT_NET_SELECT for i in range(masking.NUM_ACTIONS)]
    path = tmp_path / "no_actions.yaml"
    path.write_text("name: m\n")
    with pytest.warns(UserWarning, match=r"\['actions'\]"):
        assert not masking.load_masking_rule(path).build_mask(masking.MaskContext()).any()


def test_legacy_reward_use_drc_is_dropped_untracked_and_refused_tracked(tmp_path, monkeypatch):
    path = _broken(tmp_path, "reward", lambda r: r["potential"].update(use_drc=True))
    with pytest.warns(UserWarning, match=r"potential\.use_drc"):
        cfg = reward_config.load_reward_config(path)
    assert "use_drc" not in cfg.rule.potential
    monkeypatch.setattr("configs.loader._is_tracked", lambda path: True)
    _assert_refused("reward", path, "unknown key", "use_drc")


_OLD_DEFAULT_POTENTIAL = {"completion_bonus": 5.0, "unconnected_penalty": 1.0, "wirelength_penalty": 0.0,
                          "via_penalty": 0.0, "drc_penalty": 0.0, "step_penalty": 0.01,
                          "truncation_mode": "none", "wire_via_emission": "per_step"}
# Golden: what the loader before the `potential:`-only schema (713a186a0^ YamlRewardConfig) built
# from each old-format rule — name, mode, PotentialReward arguments, the four penalties.
OLD_FORMAT_GOLDEN = [
    ("bare", {"step_reward": {}},
     ("yaml_custom", "terminal", _OLD_DEFAULT_POTENTIAL, (0.01, 0.01, 0.2, 0.09))),
    ("full", {"name": "old", "mode": "final_only", "step_penalty": 0.02, "invalid_action_penalty": 0.03,
              "parse_fail_penalty": 0.1, "mask_reject_penalty": 0.05, "step_reward": {"drc_penalty": 9.0},
              "final_reward": {"completion_bonus": 3.0, "unconnected_penalty": 2.0, "wirelength_penalty": 0.001,
                               "via_penalty": 0.1, "use_drc": True, "drc_penalty": 0.7,
                               "truncation_mode": "full", "wire_via_emission": "on_net_end"}},
     ("old", "terminal", {"completion_bonus": 3.0, "unconnected_penalty": 2.0, "wirelength_penalty": 0.001,
                          "via_penalty": 0.1, "drc_penalty": 0.7, "step_penalty": 0.02,
                          "truncation_mode": "full", "wire_via_emission": "on_net_end"}, (0.02, 0.03, 0.1, 0.05))),
    ("drc-default", {"final_reward": {"use_drc": True}},
     ("yaml_custom", "terminal", {**_OLD_DEFAULT_POTENTIAL, "drc_penalty": 0.5}, (0.01, 0.01, 0.2, 0.09))),
    ("drc-off", {"final_reward": {"use_drc": False, "drc_penalty": 0.7}},
     ("yaml_custom", "terminal", _OLD_DEFAULT_POTENTIAL, (0.01, 0.01, 0.2, 0.09))),
    ("name-mode-only", {"name": "x", "mode": "per_step"},
     ("x", "per_step", _OLD_DEFAULT_POTENTIAL, (0.01, 0.01, 0.2, 0.09))),
]


@pytest.mark.parametrize("raw,golden", [c[1:] for c in OLD_FORMAT_GOLDEN], ids=[c[0] for c in OLD_FORMAT_GOLDEN])
def test_old_reward_format_converts_as_the_old_loader_did(tmp_path, raw, golden):
    """An old checkpoint's rule in the pre-`potential:` format loads, with a warning, into the
    reward the old loader built from it; a tracked rule in that format is refused."""
    from pcb_world.core.reward import PotentialReward

    path = tmp_path / "old_format.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.warns(UserWarning, match=r"format before `potential:`"):
        cfg = reward_config.load_reward_config(path)
    name, mode, potential, penalties = golden
    got_potential = {k: v for k, v in cfg.rule.potential.items()
                     if k not in ("invalid_action_penalty", "parse_fail_penalty", "mask_reject_penalty")}
    assert (cfg.name, cfg.mode, got_potential) == (name, mode, potential)
    assert (cfg.step_penalty, cfg.invalid_action_penalty, cfg.parse_fail_penalty,
            cfg.mask_reject_penalty) == penalties
    assert vars(cfg.build_reward()) == vars(PotentialReward(**potential))


def test_old_format_conversion_names_the_keys_it_ignores(tmp_path):
    """A mistyped new-format rule (`potentials:`) reads as the old format: the warning names it."""
    path = tmp_path / "typo.yaml"
    path.write_text("name: r\nmode: terminal\npotentials: {step_penalty: 0.5}\n")
    with pytest.warns(UserWarning, match=r"ignored top-level keys \['potentials'\]") as rec:
        reward_config.load_reward_config(path)
    assert "legacy keys" not in str(rec[0].message)


def test_tracked_old_format_reward_rule_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "old_format.yaml"
    path.write_text("name: r\nmode: terminal\nstep_penalty: 0.01\n")
    monkeypatch.setattr("configs.loader._is_tracked", lambda path: True)
    _assert_refused("reward", path, "old reward format", "potential:")


def test_rule_tracking_without_git_names_git_and_the_file(monkeypatch):
    monkeypatch.setenv("PATH", "")
    path = REPO / "configs" / "reward" / "no_git_probe.yaml"
    with pytest.raises(RuntimeError, match="git") as exc:
        _is_tracked(path)
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("value", ["nan", "inf", float("inf"), float("-inf")])
def test_drc_minimum_non_finite_is_refused(tmp_path, value):
    _assert_refused("drc", _broken(tmp_path, "drc", lambda r: r["global_minima"].update(clearance=value)),
                    "clearance", "finite")


def test_rule_schemas_do_not_alias_the_parsed_dict():
    raw = yaml.safe_load(SHIPPED["masking"].read_text())
    rule = masking.MaskingRuleSchema.from_dict(raw, "m")
    raw["actions"]["net_select"]["when"]["has_net"] = True
    assert rule.actions["net_select"]["has_net"] is False
    raw = yaml.safe_load(SHIPPED["reward"].read_text())
    rule = reward_config.RewardRuleSchema.from_dict(raw, "r")
    raw["potential"]["step_penalty"] = 99.0
    assert rule.potential["step_penalty"] != 99.0
    with pytest.raises(TypeError):
        rule.potential["step_penalty"] = 1.0
    rule = DrcRuleSchema.from_dict(yaml.safe_load(SHIPPED["drc"].read_text()), "d")
    with pytest.raises(TypeError):
        rule.global_minima["clearance"] = 1.0


def test_trainer_reads_truncation_mode_from_the_rule_without_building_the_reward(tmp_path, monkeypatch):
    # A terminal rule without ``truncation_mode`` takes the PotentialReward default (none), which
    # needs the truncation bootstrap; the check must not construct the reward.
    from types import SimpleNamespace

    from methods.rl_agent.training.loop import PPOTrainer

    path = tmp_path / "terminal.yaml"
    path.write_text("name: r\nmode: terminal\npotential:\n" + yaml_lines(RULE_PENALTIES))

    def no_build(*args, **kwargs):
        raise AssertionError("PotentialReward built")
    monkeypatch.setattr(reward_config, "PotentialReward", no_build)
    trainer = SimpleNamespace(args=SimpleNamespace(reward_rule=str(path), truncation_bootstrap=False))
    with pytest.raises(ValueError, match="truncation_mode=none"):
        PPOTrainer._check_truncation_consistency(trainer)
    trainer.args.truncation_bootstrap = True
    PPOTrainer._check_truncation_consistency(trainer)


@pytest.mark.parametrize("kind,edit,key", [
    ("reward", lambda r: r["potential"].update(step_penalty="0.1"), "step_penalty"),
    ("reward", lambda r: r["potential"].update(wirelength_bbox_normalize="yes"), "wirelength_bbox_normalize"),
    ("masking", lambda r: r["actions"]["net_select"]["when"].update(has_net="no"), "has_net"),
    ("drc", lambda r: r["global_minima"].update(clearance=True), "clearance"),
], ids=["reward-float", "reward-bool", "masking-bool", "drc-float"])
def test_mistyped_value_is_refused(tmp_path, kind, edit, key):
    _assert_refused(kind, _broken(tmp_path, kind, edit), "must be", key)


def test_masking_when_key_typo_is_refused(tmp_path):
    # Before the schema a misspelled ``when`` silently left the action always enabled.
    def edit(r):
        r["actions"]["start_route"] = {"wehn": r["actions"]["start_route"]["when"]}
    _assert_refused("masking", _broken(tmp_path, "masking", edit), "actions.start_route", "wehn")


def test_masking_when_field_typo_is_refused(tmp_path):
    # Before the schema a misspelled context field silently left the action always masked.
    def edit(r):
        r["actions"]["make_line"]["when"] = {"is_routnig": True}
    _assert_refused("masking", _broken(tmp_path, "masking", edit), "actions.make_line.when", "is_routnig")


def test_invalid_reward_mode_is_refused(tmp_path):
    _assert_refused("reward", _broken(tmp_path, "reward", lambda r: r.update(mode="sometimes")), "sometimes")


def test_drc_minimum_accepts_number_strings(tmp_path):
    # YAML 1.1 (PyYAML) reads ``1e-3`` as a string; it is a number to the DRC schema.
    path = tmp_path / "drc.yaml"
    path.write_text(SHIPPED["drc"].read_text().replace("uvia_drill:           0.10", "uvia_drill: 1e-1"))
    assert yaml.safe_load(path.read_text())["global_minima"]["uvia_drill"] == "1e-1"
    assert load_drc_config(path)["min_uvia_drill_mm"] == 0.1


def test_drc_minimum_non_number_string_is_refused(tmp_path):
    _assert_refused("drc", _broken(tmp_path, "drc", lambda r: r["global_minima"].update(uvia_drill="thin")),
                    "uvia_drill", "must be a number")


def test_drc_name_and_description_are_optional(tmp_path):
    path = _broken(tmp_path, "drc", lambda r: (r.pop("name"), r.pop("description")))
    assert len(load_drc_config(path)) == 9


def test_loading_a_reward_rule_does_not_build_the_reward(tmp_path):
    # Validation runs before CLI overrides: a value only the override makes valid must not fail the load.
    path = tmp_path / "reward.yaml"
    path.write_text("name: r\nmode: per_step\npotential:\n  truncation_mode: bogus\n" + yaml_lines(RULE_PENALTIES))
    cfg = reward_config.load_reward_config(path)
    with pytest.raises(ValueError, match="truncation_mode"):
        cfg.build_reward()
    assert cfg.with_overrides(truncation_mode="none").build_reward().truncation_mode == "none"


@pytest.mark.parametrize("hint", [Literal["a"], Any, int | str], ids=["literal", "any", "union"])
def test_check_mapping_refuses_unsupported_hints(hint):
    with pytest.raises(ValueError, match="unsupported type hint"):
        check_mapping({"k": "a"}, {"k": hint}, where="here")


def test_check_mapping_requires_every_key_by_default():
    with pytest.raises(ValueError, match=r"missing required key\(s\) \['b'\]"):
        check_mapping({"a": 1}, {"a": int, "b": int}, where="here")
    assert check_mapping({"a": 1}, {"a": int, "b": int}, required=("a",), where="here") == {"a": 1}
