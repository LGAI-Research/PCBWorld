"""The result-affecting constants promoted to ``EnvConfig`` fields (2026-10-18):
``reward_noise_clip_sigma`` (was a literal 4σ), ``drc_incremental`` (was always on) and
``drc_promoted_codes`` (was a module constant of ``pcb_world.engine.drc``).

1. Default config = the behaviour before the promotion, byte for byte: a scripted rollout's
   rewards, candidate pools and DRC promotion counts hash to the digest the base commit
   (8503799da) produced.
2. A non-default value reaches the place that consumes it.
3. The training run records each one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from configs.loader.schema import NO_FLAG_FIELDS, EnvConfig, PPOConfig, RLEnvConfig
from pcb_world.engine import engine_available

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
OBSTACLE_BOARD = str(FIXTURES / "simple_obstacle_board.kicad_pcb")
GRID_BOARD = str(FIXTURES / "jumanji" / "synth_1L_grid10_5net_v15" / "test" / "board_00000.kicad_pcb")

needs_engine = pytest.mark.skipif(not engine_available(), reason="kicad_rl_router binding not built")

#: sha256 of :func:`_rollout_digest` at 8503799da, before the constants were fields.
BASE_DIGEST = "2e5ed5797c4cee5ca996a0b9ad8c6963c03e8d7738e860dd829ccc5341017897"


def _rollout_digest(**env_overrides) -> tuple[str, list]:
    """A seeded 60-step random-valid rollout on the RL wrapper: per step the action, reward,
    candidate pool, errors_and_promoted count and violation codes."""
    from methods.rl_agent.wrappers.factory import make_decoder_env
    from pcb_world.core.action_schema import ACT_NET_SELECT

    cfg = RLEnvConfig().with_flat({"max_steps": 50, **env_overrides})
    env = make_decoder_env(OBSTACLE_BOARD, cfg, seed=123, policy_net_select=True)
    rng = np.random.default_rng(7)
    env.reset()
    rec = []
    try:
        for _ in range(60):
            at = int(rng.choice(np.flatnonzero(env.action_masks())))
            if at == ACT_NET_SELECT:
                nets = np.flatnonzero(env.net_valid_mask())
                ptr = int(rng.choice(nets)) if len(nets) else 0
            else:
                ptr = int(rng.integers(0, max(len(env._cand_mm), 1)))
            cands = [tuple(round(float(v), 9) for v in c) for c in env._cand_mm]
            _, r, term, trunc, _ = env.step(np.array([at, ptr, 2], dtype=np.int64))
            drc = env.unwrapped._engine.drc_helper
            rec.append([at, ptr, round(float(r), 9), cands,
                        drc.get_count_by_severity_mode("errors_and_promoted"),
                        sorted(int(v.error_code) for v in drc.get_violations()),
                        bool(term), bool(trunc)])
            if term or trunc:
                env.reset()
    finally:
        env.close()
    return hashlib.sha256(json.dumps(rec).encode()).hexdigest(), rec


@needs_engine
def test_default_config_reproduces_the_base_rollout():
    digest, rec = _rollout_digest()
    assert any(12 in codes or 13 in codes for *_, codes, _, _ in rec)   # promotion is exercised
    assert digest == BASE_DIGEST


@needs_engine
def test_promoted_codes_reach_the_reward_and_drc_counts():
    """With no promoted codes the dangling-track/via warnings stop counting."""
    _, base = _rollout_digest()
    _, none = _rollout_digest(drc_promoted_codes=[])
    assert [s[4] for s in none] != [s[4] for s in base]
    assert all(s[4] <= b[4] for s, b in zip(none, base))


@needs_engine
@pytest.mark.parametrize("incremental", [True, False])
def test_drc_incremental_selects_the_per_step_drc(monkeypatch, incremental):
    from pcb_world.core.env import PCBWorld

    env = PCBWorld(GRID_BOARD, EnvConfig.with_overrides(max_steps=5, drc_incremental=incremental))
    try:
        env.reset(seed=0)
        eng, calls = env._engine, []
        for name in ("run_drc", "run_drc_incremental"):
            real = getattr(eng, name)
            monkeypatch.setattr(eng, name, lambda *a, _n=name, _r=real, **k: (calls.append(_n), _r(*a, **k))[1])
        env.step(_net_select(env))
        assert calls == ["run_drc_incremental" if incremental else "run_drc"]
    finally:
        env.close()


def _net_select(env) -> dict:
    from pcb_world.core.action_schema import ACT_NET_SELECT

    net = min(env._board_info.nets)
    return {"action_type": ACT_NET_SELECT, "net_id": int(net)}


@needs_engine
@pytest.mark.parametrize("clip", [4.0, 1.5])
def test_reward_noise_is_clipped_at_the_configured_sigma(clip):
    """A huge noise draw lands at +clip x std on the terminal reward (default 4: as before)."""
    from pcb_world.core.env import PCBWorld

    def episode(**kw):
        env = PCBWorld(GRID_BOARD, EnvConfig.with_overrides(
            max_steps=1, masking_rule="strict_no_finish", reward_rule="drc_sparse_error", **kw))
        try:
            env.reset(seed=0)
            env._np_random = SimpleNamespace(normal=lambda *a: 1e6)
            return env.step(_net_select(env))[1]
        finally:
            env.close()

    noisy = episode(reward_noise_std=0.5, reward_noise_clip_sigma=clip)
    assert noisy - episode() == pytest.approx(clip * 0.5)


def test_nonpositive_clip_is_refused():
    from pcb_world.core.env import PCBWorld

    with pytest.raises(ValueError, match="reward_noise_clip_sigma must be > 0"):
        PCBWorld(GRID_BOARD, EnvConfig.with_overrides(reward_noise_clip_sigma=0.0))


def test_training_run_records_the_promoted_fields():
    from methods.rl_agent.training.args import resolve_train_args
    from methods.rl_agent.training.train_ppo import build_arg_parser

    run = resolve_train_args(build_arg_parser(), PPOConfig(), ["--board", "b.kicad_pcb"])
    for key in ("reward_noise_clip_sigma", "drc_incremental", "drc_promoted_codes"):
        assert key in NO_FLAG_FIELDS
        entry = run.provenance[key]
        assert (entry.source, entry.value) == ("defaults", getattr(EnvConfig(), key))
        assert key in run.ckpt_config()["values"]


ROUTING_BOARD = FIXTURES / "simple_routing_board.kicad_pcb"


def _dangling_track_env(**env_overrides):
    """``simple_routing_board`` with half of NET1 routed: a dangling track (KiCad code 13)
    next to the clearance errors."""
    from pcb_world.core.action_schema import ACT_MAKE_LINE, ACT_NET_SELECT, ACT_START_ROUTE
    from pcb_world.core.env import PCBWorld
    from tests.test_env.test_reward_modes import _net_at

    env = PCBWorld(str(ROUTING_BOARD), EnvConfig.with_overrides(
        max_steps=20, masking_rule="default_no_finish", **env_overrides))
    env.reset()
    env.step({"action_type": ACT_NET_SELECT, "net_id": _net_at(env, 10, 10)})
    env.step({"action_type": ACT_START_ROUTE, "x_mm": 10.0, "y_mm": 10.0, "layer": 1})
    env.step({"action_type": ACT_MAKE_LINE, "x_mm": 25.0, "y_mm": 10.0, "routing_mode": 2})
    return env


@needs_engine
def test_inline_scoring_prices_promoted_warnings_like_posthoc(tmp_path):
    """Inline (live env) and post-hoc (saved board) scoring count the same promoted set."""
    import gc

    from eval.metrics import compute_metrics_inline, evaluate_one

    routed = tmp_path / "routed.kicad_pcb"
    env = _dangling_track_env()
    try:
        assert 13 in {int(v.error_code) for v in env._engine.drc_helper.get_violations()}
        inline = compute_metrics_inline(env)
        env._engine.save(str(routed))
    finally:
        env.close()
        del env
        gc.collect()
    # The fixture's own rules (see tests/test_reward_parity.py for why the .kicad_pro is copied).
    routed.with_suffix(".kicad_pro").write_bytes(ROUTING_BOARD.with_suffix(".kicad_pro").read_bytes())
    post = evaluate_one(str(routed), str(routed.with_suffix(".kicad_pro")))
    gc.collect()
    for m in (inline, post):
        m["promoted"] = (m["extras"]["drc_promoted_count"], m["drv_errors_and_promoted_count"])
    assert inline["promoted"] == post["promoted"]
    assert inline["final_potential"] == post["final_potential"]


@needs_engine
def test_inline_scoring_refuses_a_non_default_promoted_set():
    """Scoring is fixed across methods: an env built with other promoted codes would price
    its Φ and promoted counts with them, so inline scoring raises naming both sets."""
    from eval.metrics import compute_metrics_inline

    env = _dangling_track_env(drc_promoted_codes=[])
    try:
        with pytest.raises(ValueError, match=r"\[12, 13, 37\].*drc_promoted_codes=\[\]"):
            compute_metrics_inline(env)
    finally:
        env.close()


def test_reject_if_stuck_flag_reaches_the_run_record_and_checkpoint():
    """reject_if_stuck is the one engine knob with a training flag (the KDD recipes pin false)."""
    from methods.rl_agent.training.args import resolve_train_args
    from methods.rl_agent.training.train_ppo import build_arg_parser

    assert "reject_if_stuck" not in NO_FLAG_FIELDS
    run = resolve_train_args(build_arg_parser(), PPOConfig(), ["--board", "b.kicad_pcb", "--no-reject-if-stuck"])
    entry = run.provenance["reject_if_stuck"]
    assert (entry.source, entry.value) == ("cli", False)
    assert run.ckpt_config()["values"]["reject_if_stuck"] is False
