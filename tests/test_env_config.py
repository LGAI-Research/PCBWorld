"""Golden-equivalence tests for configs.loader.schema (EnvConfig / RLEnvConfig).

RLEnvConfig.from_namespace() and from_checkpoint() map training-arg namespaces
and resolved checkpoint configs (configs.loader.ckpt) into one RLEnvConfig
(compared here in its flat ``to_record`` form).
These tests pin the produced dicts to known-correct values so behavior stays
stable. Imports are function-local (the repo's eval/* import at collection
time can tip a pre-existing native double-init segfault).
"""
from __future__ import annotations

from types import SimpleNamespace


# Full training-arg surface (every field from_namespace reads — it fills none).
_FULL_NS = dict(
    max_steps=300, masking_rule="strict", reward_rule="shaped",
    corner_mode=90, force_walkaround=True, no_mask_start_point=True,
    slot_perm=True, no_drc_tokens=True, via_penalty=0.1,
    wirelength_penalty=0.2, drc_penalty=0.3, drc_log_scale=0.4,
    drc_log_agg_scale=0.5, drc_log_offset=0.6, reward_step_penalty=0.7,
    wire_via_emission="both", directional_candidates="grid16", offboard_mask=True,
    rules_source="board_or_yaml", drc_config_path="/tmp/drc.yaml",
    connectivity_filter=False, pad_graze_margin_mm=0.0, outline_obs="tess",
    simplify_outline=False, action_history_len=1, net_constraint_obs=False,
    net_reselect=False, obs_format="indexed", keep_routing_fraction=None,
    # Engine knobs (no flag: the training parser puts them on the namespace).
    engine_seed=5, shove_iter_limit=9, followbranch_iter_limit=11,
    reject_if_stuck=False, trim_at_contact=True,
    # Promoted constants (no flag either).
    drc_incremental=False, drc_promoted_codes=[13],
)

#: The training-only layer as from_namespace / from_checkpoint leave it (off).
_TRAIN_LAYER_OFF = {"reward_noise_std": 0.0, "reward_noise_clip_sigma": 4.0, "aug_bbox_shifted": False, "aug_flip": False,
                    "aug_rotate": False, "aug_trans": False, "aug_zoom": False}


def test_from_namespace_matches_legacy_mapper():
    from configs.loader.schema import RLEnvConfig

    got = RLEnvConfig.from_namespace(SimpleNamespace(**_FULL_NS)).to_record()
    assert got == {
        "max_steps": 300, "masking_rule": "strict", "reward_rule": "shaped",
        "force_walkaround": True, "mask_start_point": False, "slot_perm": True,
        "emit_drc_tokens": False, "via_penalty": 0.1, "wirelength_penalty": 0.2,
        "drc_penalty": 0.3, "drc_log_scale": 0.4, "drc_log_agg_scale": 0.5,
        "drc_log_offset": 0.6, "reward_step_penalty": 0.7,
        "wire_via_emission": "both", "corner_mode": 90,   # degrees
        "directional_candidates": "grid16",
        "connectivity_filter": False,
        "pad_graze_margin_mm": 0.0,
        "offboard_mask": True,
        "rules_source": "board_or_yaml",
        "drc_config_path": "/tmp/drc.yaml", "obs_format": "indexed",  # fixed on the RL training path (not on the CLI)
        "outline_obs": "tess", "simplify_outline": False,
        "action_history_len": 1, "net_constraint_obs": False,
        "net_reselect": False,
        "keep_routing_fraction": None,
        "engine_seed": 5, "shove_iter_limit": 9, "followbranch_iter_limit": 11,
        "reject_if_stuck": False, "trim_at_contact": True,
        "drc_incremental": False, "drc_promoted_codes": [13],
        **_TRAIN_LAYER_OFF,
    }


def test_from_namespace_trainer_defaults():
    from configs.loader.schema import RLEnvConfig
    from methods.rl_agent.training.train_ppo import build_arg_parser

    ns = build_arg_parser().parse_args(["--board", "b.kicad_pcb"])
    got = RLEnvConfig.from_namespace(ns).to_record()
    assert got["corner_mode"] == 45
    assert got["mask_start_point"] is True   # --no-mask-start-point absent
    assert got["emit_drc_tokens"] is True    # --no-drc-tokens absent
    assert got["slot_perm"] is False
    assert got["via_penalty"] is None


def test_from_checkpoint_full():
    """A paper-era checkpoint (grid-size ring, pre-knob keys) with every stored
    env knob set away from its default."""
    from configs.loader.schema import RLEnvConfig
    from tests.helpers.ckpt_payload import ckpt_config

    ckpt = ckpt_config(
        "kdd_2026_05",
        masking_rule="strict", reward_rule="shaped", force_walkaround=True,
        no_mask_start_point=True, no_drc_tokens=True, via_penalty=0.1,
        wirelength_penalty=0.2, drc_penalty=0.3, drc_log_scale=0.4,
        drc_log_agg_scale=0.5, drc_log_offset=0.6, reward_step_penalty=0.7,
        # legacy pre-rename ckpt key — maps to directional_candidates="grid16"
        wire_via_emission="both", corner_mode=90, directional_grid_size=16,
        # legacy bool — maps to rules_source="board_or_yaml"
        use_yaml_drc_fallback=True, drc_config_path="/tmp/drc.yaml",
        slot_perm=True,  # must be IGNORED (eval forces slot_perm False)
    )
    got = RLEnvConfig.from_checkpoint(ckpt, max_steps=256).to_record()
    assert got == {
        "max_steps": 256, "masking_rule": "strict", "reward_rule": "shaped",
        "force_walkaround": True, "mask_start_point": False, "slot_perm": False,
        "emit_drc_tokens": False, "via_penalty": 0.1, "wirelength_penalty": 0.2,
        "drc_penalty": 0.3, "drc_log_scale": 0.4, "drc_log_agg_scale": 0.5,
        "drc_log_offset": 0.6, "reward_step_penalty": 0.7,
        "wire_via_emission": "both", "corner_mode": 90,
        "directional_candidates": "grid16",
        # pre-flag ckpt (no connectivity_filter key) -> the legacy unfiltered
        # candidate set the policy was trained on, NOT the flag's default True.
        "connectivity_filter": False,
        "pad_graze_margin_mm": 0.0,
        # pre-knob ckpt (no offboard_mask key) -> off, the trained behaviour
        "offboard_mask": False,
        "rules_source": "board_or_yaml",
        "drc_config_path": "/tmp/drc.yaml", "obs_format": "json",
        "outline_obs": "tess", "simplify_outline": False,
        "action_history_len": 1,
        # pre-knob ckpt: pinned to the all-zero NET channel used at training
        # (False regardless of the YAML)
        "net_constraint_obs": False,
        "net_reselect": False,
        # train-only augmentation: from_checkpoint pins it OFF at eval
        "keep_routing_fraction": None,
        **_TRAIN_LAYER_OFF,
        # pre-knob engine values; no shove_iter_limit key -> trained before
        # reject_if_stuck (legacy:partial)
        "engine_seed": 77, "shove_iter_limit": 250, "followbranch_iter_limit": 1000000,
        "reject_if_stuck": False, "trim_at_contact": False,
        # pre-knob constants: incremental per-step DRC, promoted dangling via/track + net conflict
        "drc_incremental": True, "drc_promoted_codes": [12, 13, 37],
    }


def test_connectivity_filter_round_trips_through_checkpoint():
    """The filter must survive the ckpt -> eval/MCTS env rebuild.

    It drops existing-copper candidates the route head is already connected to,
    so it changes the CANDIDATE POOL — the pointer index space the policy was
    trained against. Left out of the factory kwargs it silently defaulted to
    True, evaluating a ``--no-connectivity-filter`` policy on a candidate set it
    never saw. A checkpoint that predates the flag maps to False (its training
    behaviour), not to the flag's default.
    """
    from configs.loader.schema import RLEnvConfig
    from tests.helpers.ckpt_payload import ckpt_config

    for stored in (True, False):
        got = RLEnvConfig.from_checkpoint(
            ckpt_config(connectivity_filter=stored), max_steps=64).to_record()
        assert got["connectivity_filter"] is stored
    # pre-flag checkpoint -> legacy unfiltered pool
    assert RLEnvConfig.from_checkpoint(ckpt_config("kdd_2026_05"), max_steps=64
                                       ).to_record()["connectivity_filter"] is False
    # training namespace: the parser default is ON and reaches the env
    from methods.rl_agent.training.train_ppo import build_arg_parser
    trained = build_arg_parser().parse_args(["--board", "b.kicad_pcb"])
    assert RLEnvConfig.from_namespace(trained).to_record()[
        "connectivity_filter"] is True


def test_net_reselect_round_trips_through_checkpoint():
    """``net_reselect`` decides which nets the policy may select, so eval and the
    async-val watcher must rebuild the env the checkpoint was trained in. A
    checkpoint without the key was trained with net_end retiring the net."""
    from configs.loader.schema import RLEnvConfig
    from tests.helpers.ckpt_payload import ckpt_config

    for stored in (True, False):
        assert RLEnvConfig.from_checkpoint(
            ckpt_config(net_reselect=stored), max_steps=64).to_record()[
            "net_reselect"] is stored
    assert RLEnvConfig.from_checkpoint(ckpt_config(), max_steps=64).to_record()[
        "net_reselect"] is False
    ns = SimpleNamespace(**_FULL_NS)
    ns.net_reselect = True
    assert RLEnvConfig.from_namespace(ns).to_record()["net_reselect"] is True


def test_net_reselect_cli_pair():
    """The trainer's generated ``--net-reselect`` / ``--no-net-reselect`` pair reaches
    the env config."""
    import pytest

    from configs.loader.schema import RLEnvConfig
    from methods.rl_agent.training.train_ppo import build_arg_parser

    p = build_arg_parser()
    base = ["--board", "b.kicad_pcb"]
    assert p.parse_args(base).net_reselect is False
    on = p.parse_args(base + ["--net-reselect"])
    assert RLEnvConfig.from_namespace(on).to_record()["net_reselect"] is True
    assert p.parse_args(base + ["--no-net-reselect"]).net_reselect is False
    with pytest.raises(SystemExit):                      # one dest, both forms: an error, not last-wins
        p.parse_args(base + ["--net-reselect", "--no-net-reselect"])


def test_watcher_namespace_rebuilds_the_checkpoint_env(tmp_path):
    """The async-val watcher resolves the training config from the checkpoint
    (``resolve_ckpt_train_config``): from_namespace on it gives the env from_checkpoint
    gives, for every generation (the eval-only pins aside)."""
    import warnings

    from configs.loader.schema import RLEnvConfig
    from methods.rl_agent.training.args import resolve_ckpt_train_config
    from tests.helpers.ckpt_payload import GENERATIONS, ckpt_config, save_ckpt

    eval_pins = {"obs_format", "slot_perm", "keep_routing_fraction"}
    for gen in GENERATIONS:
        values = ckpt_config(gen)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run = resolve_ckpt_train_config(save_ckpt(tmp_path / f"{gen}.pt", gen), strict=False)
        ns = RLEnvConfig.from_namespace(run).to_record()
        ck = RLEnvConfig.from_checkpoint(values, values["max_steps"]).to_record()
        assert {k: v for k, v in ns.items() if k not in eval_pins} == \
            {k: v for k, v in ck.items() if k not in eval_pins}, gen


def test_corner_mode_code_is_strict():
    import pytest

    from pcb_world.core.env import corner_mode_code

    assert corner_mode_code(45) == 0
    assert corner_mode_code(90) == 2
    # Degrees only: raw engine codes, other angles and wrong types raise.
    for bad in (0, 1, 2, 3, 44, 91, "45", None):
        with pytest.raises(ValueError, match="45 or 90"):
            corner_mode_code(bad)


def test_env_core_to_env_kwargs():
    """EnvConfig (env-core) projects to the PCBWorld keyword surface."""
    from configs.loader.schema import EnvConfig

    kw = EnvConfig().to_env_kwargs()
    # The env-core fields (no RL-wrapper knobs leak in).
    assert set(kw) == {
        "max_steps", "masking_rule", "reward_rule", "reward_noise_std",
        "emit_drc_tokens", "via_penalty", "wirelength_penalty", "drc_penalty",
        "drc_log_scale", "drc_log_agg_scale", "drc_log_offset",
        "reward_step_penalty", "wire_via_emission", "corner_mode",
        "rules_source", "drc_config_path", "engine_seed",
        "shove_iter_limit", "followbranch_iter_limit", "reject_if_stuck",
        "trim_at_contact", "reward_noise_clip_sigma", "drc_incremental", "drc_promoted_codes",
        "obs_format", "outline_obs", "simplify_outline", "action_history_len",
        "net_constraint_obs", "net_reselect", "keep_routing_fraction",
    }
    for leaked in ("force_walkaround", "mask_start_point", "slot_perm",
                   "directional_candidates"):
        assert leaked not in kw


def test_rlenv_carries_env_core():
    """RLEnvConfig composes an EnvConfig + wrapper knobs."""
    from configs.loader.schema import EnvConfig, RLEnvConfig

    cfg = RLEnvConfig.from_namespace(SimpleNamespace(**_FULL_NS))
    assert isinstance(cfg.env, EnvConfig)
    assert cfg.env.corner_mode == 90         # degrees, lives on env-core
    assert cfg.slot_perm is True             # wrapper knob lives on RLEnvConfig
    assert cfg.env.to_env_kwargs()["reward_rule"] == "shaped"


def test_convenience_construction_and_flat_overrides():
    """``EnvConfig.with_overrides`` = the YAML defaults + the given fields; the flat
    ``with_flat`` routes record keys to env / reward / wrapper; unknown keys raise."""
    import pytest

    from configs.loader.schema import EnvConfig, RLEnvConfig

    assert EnvConfig.with_overrides() == EnvConfig()
    assert EnvConfig.with_overrides(max_steps=7).max_steps == 7
    with pytest.raises(TypeError):
        EnvConfig.with_overrides(nope=1)
    cfg = RLEnvConfig().with_flat({"max_steps": 7, "via_penalty": 0.5, "offboard_mask": True})
    assert (cfg.env.max_steps, cfg.env.reward.via_penalty, cfg.offboard_mask) == (7, 0.5, True)
    assert RLEnvConfig().with_flat(cfg.to_record()) == cfg     # the record round-trips
    with pytest.raises(TypeError):
        RLEnvConfig().with_flat({"nope": 1})


# NOTE: configs.loader.schema.RLPolicyConfig golden tests were intentionally NOT added
# here. RLPolicyConfig.from_namespace/from_checkpoint/build are exercised end-to-end
# by tests/test_train_decoder.py (train path) and the eval suites (checkpoint
# path). Adding more standalone test cases pushes the collected-test count past a
# threshold that tips the pre-existing native double-init landmine in
# tests/test_env/test_reward_modes.py (full-suite segfault ~64%). Until that
# landmine is fixed, keep RLPolicyConfig coverage in the build-path suites.


def test_llm_env_net_reselect_is_its_own_knob():
    """The LLM path takes net_reselect from llm.env (CLI --net_reselect), not from
    the RL env: default, and carries it to the env in the one EnvConfig."""
    from configs.loader import schema
    from configs.loader.cli import StrictArgumentParser
    from eval.args import add_env_args, llm_env_config

    assert "net_reselect" in schema._LLM["env"]
    assert schema.LLMConfig().env.net_reselect is schema._LLM["env"]["net_reselect"] is False
    p = StrictArgumentParser()
    add_env_args(p)
    assert llm_env_config(p.parse_args([])).net_reselect is False
    assert llm_env_config(p.parse_args(["--net_reselect"])).net_reselect is True


def test_every_env_field_declares_its_readback():
    """Each EnvConfig field names where ``PCBWorld.effective_config`` reads it back
    (an attribute path), or says why it has no readback — a new field without the
    declaration fails here (and the env's readback would KeyError on it)."""
    from dataclasses import fields

    from configs.loader.schema import EnvConfig

    undeclared, unreasoned = [], []
    for f in fields(EnvConfig):
        if "readback" not in f.metadata:
            undeclared.append(f.name)
        elif f.metadata["readback"] is None and not f.metadata.get("readback_reason"):
            unreasoned.append(f.name)
        elif f.metadata["readback"] is not None:
            assert isinstance(f.metadata["readback"], str) and f.metadata["readback"], f.name
    assert not undeclared, f"EnvConfig fields without a readback declaration: {undeclared}"
    assert not unreasoned, f"readback=None without readback_reason: {unreasoned}"
