"""configs.loader.ckpt — the one reader of a checkpoint's ``config`` (schema 2) / ``args``.

Golden per-generation fixtures (``tests/fixtures/ckpt_args/``: the KDD paper checkpoint, June /
August / September d2b runs, the October iclr1007 run) pin the resolved env + policy config and
the source of every key that did not come straight from the checkpoint. The other tests pin each
loader step: renames, weight derivations (and their contradictions), the legacy table (a
missing key is filled at any recorded version), the field-set check, strict mode and the once-per-process warning; and the
schema-2 ``config`` the trainers write now (read as stored, accepted by ``--strict-ckpt``).
"""
from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from configs.loader import ckpt as ck
from configs.loader.schema import NO_FLAG_FIELDS, RLEnvConfig, RLPolicyConfig
from tests.helpers.ckpt_payload import GENERATIONS, ckpt_config, ckpt_payload, fixture
from tests.helpers.ckpt_payload import quiet_load as _load


@pytest.mark.parametrize("generation", GENERATIONS)
def test_generation_resolves_to_its_golden_config(generation):
    expected = fixture(generation)["expected"]
    values, sources = _load(ckpt_payload(generation))
    assert {k: s for k, s in sources.items() if s != "ckpt"} == expected["sources"]
    assert asdict(RLEnvConfig.from_checkpoint(values, values["max_steps"])) == expected["env"]
    assert asdict(RLPolicyConfig.from_checkpoint(values)) == expected["policy"]


def test_every_ckpt_field_gets_a_source_and_values_are_read_only():
    values, sources = _load(ckpt_payload("kdd_2026_05"))
    assert set(sources) == ck.CKPT_FIELDS
    with pytest.raises(TypeError):
        values["corner_mode"] = 45


def test_paper_checkpoint_sources():
    """The KDD file predates most knobs: the grid size maps to a grid<N> ring, the tokenizer
    layout comes from the weights, reject_if_stuck from the shove_iter_limit proxy."""
    values, sources = _load(ckpt_payload("kdd_2026_05"))
    assert (values["directional_candidates"], sources["directional_candidates"]) == ("grid10", "renamed")
    assert "directional_grid_size" not in values
    assert (values["n_max_slots"], sources["n_max_slots"]) == (64, "derived")
    assert (values["reject_if_stuck"], sources["reject_if_stuck"]) == (False, "legacy:partial")
    assert (values["shove_iter_limit"], sources["shove_iter_limit"]) == (250, "legacy")
    assert sources["trim_at_contact"] == "legacy"
    assert values["corner_mode"] == 90


def test_slot_table_size_comes_from_the_weights():
    """June d2b stores no n_max_slots and has a 1024-row slot table — the old legacy value
    (64) built a model its weights could not load into."""
    values, sources = _load(ckpt_payload("d2b_2026_06"))
    assert (values["n_max_slots"], sources["n_max_slots"]) == (1024, "derived")


def test_shove_iter_limit_present_means_reject_if_stuck_true():
    values, sources = _load(ckpt_payload("d2b_2026_08"))
    assert (values["reject_if_stuck"], sources["reject_if_stuck"]) == (True, "legacy:proxy")


def test_cli_forms_become_their_fields():
    values, _ = _load(ckpt_payload(no_drc_tokens=True, no_mask_start_point=True))
    assert values["emit_drc_tokens"] is False and values["mask_start_point"] is False
    assert values["no_drc_tokens"] is True   # the stored dest stays (training namespaces)


def test_newer_directional_key_wins_over_the_grid_size():
    values, _ = _load(ckpt_payload(directional_grid_size=16))
    assert values["directional_candidates"] == "multi_resolution"
    values, _ = _load(ckpt_payload(directional_grid_size=16, directional_candidates="none"))
    assert values["directional_candidates"] == "grid16"


@pytest.mark.parametrize("generation,source", [("d2b_2026_06", "board"),
                                               ("d2b_2026_08", "board_or_yaml")])
def test_yaml_drc_fallback_flag_reads_as_rules_source(generation, source):
    """Checkpoints store the bool ``use_yaml_drc_fallback`` that ``rules_source`` replaced:
    off -> board, on -> board_or_yaml (the same behaviour), and the env is built with it."""
    values, sources = _load(ckpt_payload(generation))
    assert (values["rules_source"], sources["rules_source"]) == (source, "renamed")
    assert "use_yaml_drc_fallback" not in values
    assert RLEnvConfig.from_checkpoint(values, 64).env.rules_source == source


def test_removed_key_is_dropped():
    values, _ = _load(ckpt_payload(allow_incomplete_net_end=True))
    assert "allow_incomplete_net_end" not in values


@pytest.mark.parametrize("key,stored", [("shape_obs", False), ("n_max_slots", 64),
                                        ("legacy_edge_encoding", True), ("legacy_action_history", True),
                                        ("use_critic", False)])
def test_args_contradicting_the_weights_raise(key, stored):
    with pytest.raises(ck.CkptConfigError, match=f"contradiction: args say {key}"):
        _load(ckpt_payload(**{key: stored}))


def test_single_entry_history_weights_derive_window_one():
    values, sources = _load(ckpt_payload(drop=("action_history_len",),
                                         shapes={"tokenizer.vocab.history_age_proj.weight": None}))
    assert values["legacy_action_history"] is True
    assert (values["action_history_len"], sources["action_history_len"]) == (1, "derived")


def test_unknown_tokenizer_layout_raises():
    with pytest.raises(ck.CkptConfigError, match="legacy_pad_layer_encoding"):
        _load(ckpt_payload(shapes={"tokenizer.vocab.pad_proj.weight": [256, 7]}))


def test_a_key_missing_everywhere_raises(monkeypatch):
    monkeypatch.setattr(ck, "LEGACY", {k: v for k, v in ck.LEGACY.items() if k != "obstacle_obs"})
    with pytest.raises(ck.CkptConfigError, match=r"\['obstacle_obs'\].*ckpt_migrations.yaml"):
        _load(ckpt_payload("kdd_2026_05"))


@pytest.mark.parametrize("version", ["0.9.0", "1.0.1", "1.6.4", "9.9.9", "unknown", None])
def test_key_missing_from_a_checkpoint_is_filled_at_any_version(version, monkeypatch):
    """provenance.repo_version is the writing tree's README marker, and public and private
    version lines differ: a key missing from an ``args`` checkpoint is filled from the table
    with a warning at any version, and only --strict-ckpt refuses it."""
    payload = ckpt_payload(drop=("offboard_mask", "trim_at_contact"))
    payload["provenance"] = None if version is None else {"repo_version": version}
    monkeypatch.setattr(ck, "_warned", set())
    with pytest.warns(UserWarning, match=r"offboard_mask \(legacy\).*trim_at_contact \(legacy\)"):
        values, sources = ck.load_ckpt_config(payload, strict=False)
    assert (values["offboard_mask"], values["trim_at_contact"]) == (False, False)
    with pytest.raises(ck.CkptConfigError, match=r"--strict-ckpt.*offboard_mask \(legacy\)"):
        ck.load_ckpt_config(payload, strict=True)


def _trainer_payload(module: str, *argv: str):
    """A checkpoint payload the way ``RLTrainer`` writes it, for a run resolved from ``argv``."""
    from configs.loader.record import collect_provenance
    from tests.helpers.ckpt_golden import BOARD, train_run

    run = train_run(module.removeprefix("train_"), ["--board", BOARD, *argv])
    model = RLPolicyConfig.from_namespace(run, use_critic=module == "train_ppo").build()
    return run, {**ck.ckpt_entries(run.ckpt_config()), "policy_state_dict": model.state_dict(),
                 "obs_probe": {"stub": 1}, "provenance": collect_provenance(),
                 "trainer": module.removeprefix("train_")}


@pytest.mark.parametrize("module,critic", [("train_ppo", True), ("train_grpo", False)])
def test_every_trainer_writes_a_checkpoint_the_loader_reads(module, critic):
    """A checkpoint of either trainer loads as stored — schema 2, no warning, accepted by
    --strict-ckpt; the GRPO parser has no --detach-critic (PPO-only)."""
    run, payload = _trainer_payload(module)
    values, sources = ck.load_ckpt_config(payload, strict=True)
    assert values["use_critic"] is critic and values["detach_critic"] is False
    assert sources["use_critic"] == "derived"
    # The no-flag fields (engine knobs, promoted constants) have no flag, yet the checkpoint
    # config records them with the run's values.
    assert {k: values[k] for k in NO_FLAG_FIELDS} == {k: getattr(run, k) for k in NO_FLAG_FIELDS}
    stored = [k for k in NO_FLAG_FIELDS if k in ck.CKPT_FIELDS]
    assert {k: sources[k] for k in stored} == dict.fromkeys(stored, "ckpt")


@pytest.mark.parametrize("module", ["train_ppo", "train_grpo"])
def test_new_checkpoint_args_read_by_a_pre_schema_2_loader(module, monkeypatch):
    """The trainer's checkpoint also stores its values as ``args`` (what loaders before schema 2
    read, plus ``use_yaml_drc_fallback`` for ``rules_source``): read through the ``args`` path
    alone they resolve to the values of its schema-2 config."""
    from methods.rl_agent.training.loop import GRPOTrainer, PPOTrainer

    run, base = _trainer_payload(module, "--rules-source", "board_or_yaml")
    cls = PPOTrainer if module == "train_ppo" else GRPOTrainer
    t = cls.__new__(cls)
    t.args = run
    t.policy = SimpleNamespace(state_dict=lambda: base["policy_state_dict"])
    t.optimizer = SimpleNamespace(state_dict=dict)
    payload = {**base, **t._train_ckpt_payload(3, {})}
    assert payload["config"] == run.ckpt_config()
    assert payload["args"]["use_yaml_drc_fallback"] is True
    want, _ = ck.load_ckpt_config(payload, strict=True)       # `config` wins over `args`
    del payload["config"]
    monkeypatch.setattr(ck, "_warned", set())
    with pytest.warns(UserWarning, match="pre-schema-2"):
        got, sources = ck.load_ckpt_config(payload, strict=False)
    assert {k: got[k] for k in ck.CKPT_FIELDS} == {k: want[k] for k in ck.CKPT_FIELDS}
    assert sources["rules_source"] == "ckpt" and "use_yaml_drc_fallback" not in got
    payload["args"]["use_yaml_drc_fallback"] = False   # contradicts rules_source
    with pytest.raises(ck.CkptConfigError, match="contradict"):
        ck.load_ckpt_config(payload, strict=False)


def test_schema_2_keeps_a_recorded_estimate_source(tmp_path, monkeypatch):
    """A value a schema-2 config recorded as legacy* (directly, or through ``init_from`` for a key
    a BC checkpoint took from its base) stays an estimate: --init-ckpt from it takes a different
    command-line value with a warning; --strict-ckpt refuses."""
    import torch

    from configs.loader.resolve import ConfigError
    from methods.rl_agent.training.train_ppo import parse_train_args

    run, payload = _trainer_payload("train_ppo")
    payload["config"]["sources"]["obstacle_obs"] = "legacy"
    assert ck.load_ckpt_config(payload, strict=True)[1]["obstacle_obs"] == "legacy"
    payload["config"]["sources"]["obstacle_obs"] = "init_from"
    payload["config"]["init_from"] = {"ckpt": "base.pt", "sources": {"obstacle_obs": "legacy:proxy"}}
    assert ck.load_ckpt_config(payload, strict=True)[1]["obstacle_obs"] == "legacy:proxy"
    path = tmp_path / "bc.pt"
    torch.save(payload, path)
    flip = "--no-obstacle-obs" if run.obstacle_obs else "--obstacle-obs"
    argv = ["--board", "b.kicad_pcb", "--init-ckpt", str(path), flip]
    with pytest.warns(UserWarning, match="obstacle_obs: the checkpoint's .* is not recorded in it"):
        assert parse_train_args(argv).obstacle_obs is not run.obstacle_obs
    with pytest.raises(ConfigError, match="may not set 'obstacle_obs'"):
        parse_train_args([*argv, "--strict-ckpt"])
    payload["config"]["init_from"]["sources"]["obstacle_obs"] = "ckpt"   # recorded by the base
    torch.save(payload, path)
    with pytest.raises(ConfigError, match="may not set 'obstacle_obs'"):
        parse_train_args(argv)


def test_schema_2_config_without_a_source_is_a_writer_bug():
    _, payload = _trainer_payload("train_ppo")
    del payload["config"]["sources"]["obstacle_obs"]
    with pytest.raises(ck.CkptConfigError, match="'obstacle_obs' without a source"):
        ck.load_ckpt_config(payload, strict=False)
    payload["config"]["sources"]["obstacle_obs"] = "init_from"
    payload["config"]["init_from"] = {"ckpt": "base.pt", "sources": {}}
    with pytest.raises(ck.CkptConfigError, match="'obstacle_obs' without a source"):
        ck.load_ckpt_config(payload, strict=False)


def test_schema_2_round_trip(tmp_path):
    """save -> load -> resolve gives back the run's values; the stored sources are the run's."""
    import torch

    from configs.loader.resolve import PROCESS_OPTIONS
    from methods.rl_agent.training.args import resolve_ckpt_train_config

    run, payload = _trainer_payload("train_ppo", "--gamma", "0.9", "--masking-rule", "default")
    path = tmp_path / "policy_iter_1.pt"
    torch.save(payload, path)
    stored = torch.load(path, weights_only=False)["config"]
    assert stored["sources"] == {k: e.source for k, e in run.provenance.items()}
    assert stored["sources"]["gamma"] == "cli" and stored["mode"] == "fresh"
    again = resolve_ckpt_train_config(path, strict=True)
    assert vars(again) == vars(run)
    assert {k for k, e in again.provenance.items() if e.source != "ckpt"} == set(PROCESS_OPTIONS)


def test_schema_2_config_missing_a_field_is_a_writer_bug():
    _, payload = _trainer_payload("train_ppo")
    del payload["config"]["values"]["masking_rule"]
    with pytest.raises(ck.CkptConfigError, match=r"\['masking_rule'\].*schema-2 config records every field"):
        ck.load_ckpt_config(payload, strict=False)
    payload["config"]["schema"] = 3
    with pytest.raises(ck.CkptConfigError, match="not schema 2"):
        ck.load_ckpt_config(payload, strict=False)


def test_trainers_store_their_marker(tmp_path, monkeypatch):
    import torch

    from methods.rl_agent.training.loop import GRPOTrainer, PPOTrainer

    for cls, name in ((PPOTrainer, "ppo"), (GRPOTrainer, "grpo")):
        t = cls.__new__(cls)
        t._ckpt_stamp, t.best_eval_fp = {}, float("-inf")
        monkeypatch.setattr(t, "_schedule_state", lambda: None, raising=False)
        t._save_ckpt(str(tmp_path / f"{name}.pt"), {"config": {}})
        assert torch.load(tmp_path / f"{name}.pt", weights_only=False)["trainer"] == name


def test_unknown_trainer_raises():
    payload = ckpt_payload()
    payload["trainer"] = "sac"
    with pytest.raises(ck.CkptConfigError, match="unknown checkpoint trainer 'sac'"):
        _load(payload)


def test_corner_mode_must_be_an_angle():
    with pytest.raises(ck.CkptConfigError, match="corner_mode must be one of"):
        _load(ckpt_payload(corner_mode=2))


def test_strict_rejects_a_pre_schema_2_checkpoint():
    with pytest.raises(ck.CkptConfigError, match="--strict-ckpt: pre-schema-2"):
        ck.load_ckpt_config(ckpt_payload(), strict=True)


def test_warns_once_per_distinct_resolution_and_names_missing_obs_probe(monkeypatch):
    monkeypatch.setattr(ck, "_warned", set())
    with pytest.warns(UserWarning, match=r"net_reselect \(legacy\).*no obs_probe") as rec:
        ck.load_ckpt_config(ckpt_payload("d2b_2026_08"), strict=False)
        ck.load_ckpt_config(ckpt_payload("d2b_2026_08"), strict=False)
        ck.load_ckpt_config(ckpt_payload("kdd_2026_05"), strict=False)
    assert len(rec) == 2                            # a second generation is warned too


def test_warning_names_filled_engine_knobs(monkeypatch):
    monkeypatch.setattr(ck, "_warned", set())
    with pytest.warns(UserWarning) as rec:
        ck.load_ckpt_config(ckpt_payload("kdd_2026_05"), strict=False)
    msg = str(rec[0].message)
    assert "shove_iter_limit (legacy)" in msg      # applied to the env like any knob
    assert "net_reselect (legacy)" in msg


def test_strict_flag_is_one_shared_argument():
    from configs.loader.cli import StrictArgumentParser

    p = StrictArgumentParser()
    ck.add_strict_ckpt_arg(p)
    assert p.parse_args([]).strict_ckpt is False
    assert p.parse_args(["--strict-ckpt"]).strict_ckpt is True


def test_env_record_is_filled_from_the_same_table():
    filled = ck.with_legacy_env_defaults({"max_steps": 20, "outline_obs": "arc"})
    assert filled["outline_obs"] == "arc"            # recorded value kept
    assert filled["net_reselect"] is False           # pre-knob value
    # Keys records gained in v1.6.4: what the val env was built with before.
    assert (filled["engine_seed"], filled["reject_if_stuck"], filled["reward_noise_std"],
            filled["aug_flip"]) == (77, True, 0.0, False)
    assert "masking_rule" not in filled              # always recorded: never filled


@pytest.mark.parametrize("flag,source", [(False, "board"), (True, "board_or_yaml")])
def test_env_record_yaml_drc_fallback_reads_as_rules_source(flag, source):
    filled = ck.with_legacy_env_defaults({"max_steps": 20, "use_yaml_drc_fallback": flag})
    assert filled["rules_source"] == source and "use_yaml_drc_fallback" not in filled


@pytest.mark.parametrize("code,angle", [(0, 45), (2, 90), (45, 45), (90, 90)])
def test_env_record_engine_corner_code_reads_as_degrees(code, angle):
    """Records written before corner_mode became degrees store the engine code."""
    assert ck.with_legacy_env_defaults({"max_steps": 20, "corner_mode": code})["corner_mode"] == angle


def test_from_checkpoint_reads_no_default():
    """Every field comes from the resolved config — a bare dict missing a key raises."""
    values = dict(ckpt_config())
    del values["masking_rule"]
    with pytest.raises(KeyError):
        RLEnvConfig.from_checkpoint(values, 64)


def test_ckpt_migrate_writes_the_loader_result(tmp_path, monkeypatch, capsys):
    """tools/ckpt_migrate.py writes what the loader resolves: afterwards every
    config key is stored, so a second run has nothing to add."""
    import sys

    import torch

    from tests.helpers.ckpt_payload import save_ckpt
    from tools import ckpt_migrate as tool

    ckpt = save_ckpt(tmp_path / "old.pt", "kdd_2026_05")
    expected, _ = _load(ckpt_payload("kdd_2026_05"))

    monkeypatch.setattr(sys, "argv", ["ckpt_migrate", str(ckpt)])
    tool.main()
    assert "adding" in capsys.readouterr().out
    stored = torch.load(ckpt, weights_only=False)["args"]
    assert stored == {**expected, "use_yaml_drc_fallback": False} and "directional_grid_size" not in stored
    tool.main()
    assert "complete" in capsys.readouterr().out
