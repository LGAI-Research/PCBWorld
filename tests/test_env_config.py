"""Golden-equivalence tests for configs.loader.schema (EnvConfig / RLEnvConfig).

RLEnvConfig.from_namespace() and from_checkpoint() map training-arg namespaces
and checkpoint dicts into pool kwargs through one shared schema + adapters.
These tests pin the produced dicts to known-correct values so behavior stays
stable. Imports are function-local (the repo's eval/* import at collection
time can tip a native double-init segfault).
"""
from __future__ import annotations

from types import SimpleNamespace


# Full training-arg surface (every field the training-arg mapper reads).
_FULL_NS = dict(
    max_steps=300, masking_rule="strict", reward_rule="shaped",
    corner_mode=90, force_walkaround=True, no_mask_start_point=True,
    slot_perm=True, no_drc_tokens=True, via_penalty=0.1,
    wirelength_penalty=0.2, drc_penalty=0.3, drc_log_scale=0.4,
    drc_log_agg_scale=0.5, drc_log_offset=0.6, reward_step_penalty=0.7,
    wire_via_emission="both", directional_candidates="grid16",
    use_yaml_drc_fallback=True, drc_config_path="/tmp/drc.yaml",
)


def test_from_namespace_matches_legacy_mapper():
    from configs.loader.schema import RLEnvConfig

    got = RLEnvConfig.from_namespace(SimpleNamespace(**_FULL_NS)).to_pool_kwargs()
    assert got == {
        "max_steps": 300, "masking_rule": "strict", "reward_rule": "shaped",
        "force_walkaround": True, "mask_start_point": False, "slot_perm": True,
        "emit_drc_tokens": False, "via_penalty": 0.1, "wirelength_penalty": 0.2,
        "drc_penalty": 0.3, "drc_log_scale": 0.4, "drc_log_agg_scale": 0.5,
        "drc_log_offset": 0.6, "reward_step_penalty": 0.7,
        "wire_via_emission": "both", "corner_mode": 2,  # 90 -> code 2
        "directional_candidates": "grid16", "connectivity_filter": True,
        "pad_graze_margin_mm": 0.0,
        "use_yaml_drc_fallback": True,
        "drc_config_path": "/tmp/drc.yaml", "obs_format": "indexed",  # fixed on the RL training path (not on the CLI)
        "outline_obs": "tess", "simplify_outline": False,
        "action_history_len": 1, "net_constraint_obs": False,
        "keep_routing_fraction": None,
    }

def test_from_namespace_defaults_and_corner_45():
    from configs.loader.schema import RLEnvConfig

    # Minimal namespace: required fields + corner 45 sugar -> code 0.
    ns = SimpleNamespace(max_steps=200, masking_rule="default",
                         reward_rule="drc_only_dense", corner_mode=45)
    got = RLEnvConfig.from_namespace(ns).to_pool_kwargs()
    assert got["corner_mode"] == 0
    assert got["mask_start_point"] is True   # no_mask_start_point absent
    assert got["emit_drc_tokens"] is True    # no_drc_tokens absent
    assert got["slot_perm"] is False
    assert got["via_penalty"] is None


def test_from_checkpoint_full():
    from configs.loader.schema import RLEnvConfig

    ckpt = dict(
        masking_rule="strict", reward_rule="shaped", force_walkaround=True,
        no_mask_start_point=True, no_drc_tokens=True, via_penalty=0.1,
        wirelength_penalty=0.2, drc_penalty=0.3, drc_log_scale=0.4,
        drc_log_agg_scale=0.5, drc_log_offset=0.6, reward_step_penalty=0.7,
        # older ckpt spelling — must map to directional_candidates="grid16"
        wire_via_emission="both", corner_mode=90, directional_grid_size=16,
        use_yaml_drc_fallback=True, drc_config_path="/tmp/drc.yaml",
        slot_perm=True,  # must be IGNORED (eval forces slot_perm False)
    )
    got = RLEnvConfig.from_checkpoint(ckpt, max_steps=256).to_pool_kwargs()
    assert got == {
        "max_steps": 256, "masking_rule": "strict", "reward_rule": "shaped",
        "force_walkaround": True, "mask_start_point": False, "slot_perm": False,
        "emit_drc_tokens": False, "via_penalty": 0.1, "wirelength_penalty": 0.2,
        "drc_penalty": 0.3, "drc_log_scale": 0.4, "drc_log_agg_scale": 0.5,
        "drc_log_offset": 0.6, "reward_step_penalty": 0.7,
        "wire_via_emission": "both", "corner_mode": 2,
        "directional_candidates": "grid16",
        # ckpt without a connectivity_filter key -> the unfiltered candidate set
        # the policy was trained on, NOT the flag's default True.
        "connectivity_filter": False,
        "pad_graze_margin_mm": 0.0,
        "use_yaml_drc_fallback": True,
        "drc_config_path": "/tmp/drc.yaml", "obs_format": "json",
        "outline_obs": "tess", "simplify_outline": False,
        "action_history_len": 1,
        # ckpt without the knob: pinned to the all-zero NET channel used at
        # training (False regardless of the YAML)
        "net_constraint_obs": False,
        # train-only augmentation: from_checkpoint pins it OFF at eval
        "keep_routing_fraction": None,
    }


def test_from_checkpoint_defaults_on_empty():
    from configs.loader.schema import RLEnvConfig

    # Empty ckpt -> every key falls back to the shared YAML default
    # (configs/defaults/env.yaml = the training defaults), so eval matches train.
    got = RLEnvConfig.from_checkpoint({}, max_steps=128).to_pool_kwargs()
    assert got == {
        "max_steps": 128, "masking_rule": "default_no_finish",
        "reward_rule": "drc_only_dense", "force_walkaround": False,
        "mask_start_point": True, "slot_perm": False, "emit_drc_tokens": True,
        "via_penalty": None, "wirelength_penalty": None, "drc_penalty": None,
        "drc_log_scale": None, "drc_log_agg_scale": None, "drc_log_offset": None,
        "reward_step_penalty": None, "wire_via_emission": None,
        "corner_mode": 0,  # default 45 -> code 0
        "directional_candidates": None, "connectivity_filter": False,
        "pad_graze_margin_mm": 0.0,
        "use_yaml_drc_fallback": False,
        "drc_config_path": None, "obs_format": "json",
        "outline_obs": "tess", "simplify_outline": False,  # ckpt without the flag: the representation used at training
        "action_history_len": 1, "net_constraint_obs": False,
        "keep_routing_fraction": None,
    }


def test_connectivity_filter_round_trips_through_checkpoint():
    """The filter must survive the ckpt -> eval/MCTS env rebuild.

    It drops existing-copper candidates the route head is already connected to,
    so it changes the CANDIDATE POOL — the pointer index space the policy was
    trained against. It therefore has to be carried through ``to_pool_kwargs``,
    and a checkpoint that stores no such key maps to False (the behaviour it was
    trained with), not to the flag's default.
    """
    from configs.loader.schema import RLEnvConfig

    for stored in (True, False):
        got = RLEnvConfig.from_checkpoint(
            {"connectivity_filter": stored}, max_steps=64).to_pool_kwargs()
        assert got["connectivity_filter"] is stored
    # checkpoint without the key -> unfiltered pool
    assert RLEnvConfig.from_checkpoint({}, max_steps=64).to_pool_kwargs()[
        "connectivity_filter"] is False
    # training namespace: flag default is ON, and an explicit off is honoured
    ns = SimpleNamespace(max_steps=64, masking_rule="default",
                         reward_rule="drc_only_dense", corner_mode=45)
    assert RLEnvConfig.from_namespace(ns).to_pool_kwargs()[
        "connectivity_filter"] is True
    ns.connectivity_filter = False
    assert RLEnvConfig.from_namespace(ns).to_pool_kwargs()[
        "connectivity_filter"] is False


def test_corner_mode_to_code():
    from configs.loader.schema import corner_mode_to_code

    assert corner_mode_to_code(45) == 0
    assert corner_mode_to_code(90) == 2
    assert corner_mode_to_code(0) == 0     # raw engine code passes through
    assert corner_mode_to_code(3) == 3
    assert corner_mode_to_code("45") == 0
    assert corner_mode_to_code(None) == 0  # unparsable -> 0


def test_corner_deg_to_code_strict():
    import pytest

    from configs.loader.schema import corner_deg_to_code

    assert corner_deg_to_code(45) == 0
    assert corner_deg_to_code(90) == 2
    # STRICT degrees entry: raw codes, other angles, wrong types all assert
    # (unlike the lenient checkpoint-side corner_mode_to_code).
    for bad in (0, 1, 2, 3, 44, 91, "45", None):
        with pytest.raises(AssertionError):
            corner_deg_to_code(bad)


def test_env_core_to_env_kwargs():
    """EnvConfig (env-core) projects to the PCBWorld keyword surface."""
    from configs.loader.schema import EnvConfig

    kw = EnvConfig().to_env_kwargs()
    # 26 env-core fields (no RL-wrapper knobs leak in).
    assert set(kw) == {
        "max_steps", "masking_rule", "reward_rule", "reward_noise_std",
        "emit_drc_tokens", "via_penalty", "wirelength_penalty", "drc_penalty",
        "drc_log_scale", "drc_log_agg_scale", "drc_log_offset",
        "reward_step_penalty", "wire_via_emission", "corner_mode",
        "use_yaml_drc_fallback", "drc_config_path", "engine_seed",
        "shove_iter_limit", "followbranch_iter_limit", "reject_if_stuck",
        "obs_format", "outline_obs", "simplify_outline", "action_history_len",
        "net_constraint_obs", "keep_routing_fraction",
    }
    for leaked in ("force_walkaround", "mask_start_point", "slot_perm",
                   "directional_candidates"):
        assert leaked not in kw


def test_rlenv_carries_env_core():
    """RLEnvConfig composes an EnvConfig + wrapper knobs."""
    from configs.loader.schema import EnvConfig, RLEnvConfig

    cfg = RLEnvConfig.from_namespace(SimpleNamespace(**_FULL_NS))
    assert isinstance(cfg.env, EnvConfig)
    assert cfg.env.corner_mode == 2          # 90 -> code 2, lives on env-core
    assert cfg.slot_perm is True             # wrapper knob lives on RLEnvConfig
    assert cfg.env.to_env_kwargs()["reward_rule"] == "shaped"


def test_policy_loader_delegates_to_rlenvconfig():
    """The mapper helpers in models.loader delegate and produce identical dicts."""
    from configs.loader.schema import RLEnvConfig
    from methods.rl_agent.models.loader import (
        _corner_mode_code,
        env_kwargs_from_checkpoint,
        env_kwargs_from_training_args,
    )

    ns = SimpleNamespace(**_FULL_NS)
    assert env_kwargs_from_training_args(ns) == (
        RLEnvConfig.from_namespace(ns).to_pool_kwargs()
    )
    ckpt = {"reward_rule": "shaped", "corner_mode": 90}
    assert env_kwargs_from_checkpoint(ckpt, 256) == (
        RLEnvConfig.from_checkpoint(ckpt, 256).to_pool_kwargs()
    )
    assert _corner_mode_code(90) == 2


# NOTE: this file carries no golden tests for configs.loader.schema.RLPolicyConfig.
# RLPolicyConfig.from_namespace/from_checkpoint/build are exercised end-to-end by
# tests/test_train_decoder.py (train path) and the eval suites (checkpoint path),
# and extra standalone cases here push the collected-test count past the threshold
# that tips the native double-init landmine in tests/test_env/test_reward_modes.py
# (full-suite segfault ~64%). Keep RLPolicyConfig coverage in the build-path suites.
