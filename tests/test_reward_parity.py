"""Reward parity guard: the training env and the offline scorer must price a
board identically.

Every board-dependent reward term (the "board-resolution hooks":
``completion_bonus_log_scale``, ``clean_completion_bonus_log_scale``,
``wirelength_bbox_normalize``, ``net_bonus_size_log_scale``) has exactly one
definition — ``PotentialReward.bind_board`` — and both ``PCBWorld`` (training)
and ``eval.metrics.compute_metrics`` / ``evaluate_one`` (offline scoring, eval
Stage 2) call it. Witness for the drift this guards: under the
bbox-normalized wirelength rule a routed board scored ``final_potential`` −302
offline vs +28 in the env, because the scorer hand-copied only one of the four
hooks.

Pinned for every reward yaml in the repo (``configs/reward/*.yaml`` and the KDD-era
rules under ``experiments/kdd/configs/reward/*.yaml``), the
hook-heavy d2b-lineage rules (``tests/fixtures/reward_rules/reward*.yaml``) and a W6-style
size-weighted ladder rule, on a small fixture board routed to completion by
the scripted trajectory of ``tests/test_env/test_reward_modes.py``:

  (a) resolved values — completion_bonus, clean_completion_bonus,
      wirelength_penalty, per-net size weights — are EXACTLY equal on both paths;
  (b) ``compute_final`` on the SAME engine snapshot (inline scorer on the live
      env) is bit-identical to the env's reward object;
  (c) e2e: the env's episode-end ``info["final_potential"]`` and its bare-board
      ``initial_potential`` equal ``evaluate_one`` on the saved board — DRC on both sides; only the
      float-sum order over reloaded tracks may differ, hence a tight approx.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path

import pytest

from configs.loader.schema import EnvConfig
from pcb_world.core.reward import PotentialReward, RewardState
from tests._rule_helpers import ENV_PENALTIES, yaml_lines
from tests.test_env.test_reward_modes import _route_all_nets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOARD = PROJECT_ROOT / "tests" / "fixtures" / "simple_routing_board.kicad_pcb"
REWARD_DIRS = (PROJECT_ROOT / "configs" / "reward",
               PROJECT_ROOT / "experiments" / "kdd" / "configs" / "reward")
HOOK_RULES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "reward_rules"

REPO_RULES = sorted(p.stem for d in REWARD_DIRS for p in d.glob("*.yaml"))
D2B_RULES = sorted(
    str(p.relative_to(PROJECT_ROOT)) for p in HOOK_RULES_DIR.glob("reward*.yaml")
)
SW_RULE = "sw_inline"  # W6-style size-weighted ladder (all four hooks on)
ALL_RULES = REPO_RULES + D2B_RULES + [SW_RULE]
assert len(REPO_RULES) >= 10, REPO_RULES  # the glob must actually find the repo rules
assert len(D2B_RULES) >= 3, D2B_RULES

SW_RULE_TEXT = (
    "name: sw_parity\nmode: per_step\npotential:\n"
    "  completion_bonus_log_scale: 2.0\n"
    "  clean_completion_bonus_log_scale: 2.0\n"
    "  net_completion_bonus: 1.0\n"
    "  net_clean_bonus: 1.0\n"
    "  net_bonus_size_log_scale: 1.0\n"
    "  unconnected_penalty: 1.0\n"
    "  wirelength_penalty: 0.5\n"
    "  wirelength_bbox_normalize: true\n"
    "  via_penalty: 0.2\n"
    "  drc_shape: log_per_net\n"
    "  drc_log_scale: 1.0\n"
    "  drc_log_agg_scale: 3.0\n"
    "  drc_log_offset: 2.0\n"
    "  step_penalty: 0.0\n"
    "  drc_severity_mode: errors_and_promoted\n"
    + yaml_lines(ENV_PENALTIES)
)


@pytest.fixture(scope="module")
def sw_rule_path(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("reward") / "reward_sw.yaml"
    p.write_text(SW_RULE_TEXT)
    return str(p)


def _rule_arg(rule: str, sw_rule_path: str) -> str:
    """Reward rule as PCBWorld / evaluate_one take it (name or yaml path)."""
    if rule == SW_RULE:
        return sw_rule_path
    if rule.endswith(".yaml"):
        return str(PROJECT_ROOT / rule)
    return rule


def _resolved(pot: PotentialReward) -> dict:
    """The board-resolved terms of a reward object (what bind_board sets)."""
    return {
        "completion_bonus": pot.completion_bonus,
        "clean_completion_bonus": pot.clean_completion_bonus,
        "wirelength_penalty": pot.wirelength_penalty,
        "net_size_weights": (
            dict(pot.net_size_weights_by_code)
            if pot.net_size_weights_by_code is not None else None
        ),
    }


def _resolved_from_metrics(result: dict) -> dict:
    """Same terms as reported by compute_metrics' ``phi_weights``."""
    w = result["phi_weights"]
    return {k: w[k] for k in _resolved(PotentialReward())}


# ---------------------------------------------------------------------------
# bind_board semantics (no C++)
# ---------------------------------------------------------------------------


def _hooked(**overrides) -> PotentialReward:
    kwargs = dict(
        completion_bonus_log_scale=2.0,
        clean_completion_bonus_log_scale=3.0,
        wirelength_penalty=0.5,
        wirelength_bbox_normalize=True,
        net_completion_bonus=1.0,
        net_clean_bonus=1.0,
        net_bonus_size_log_scale=1.0,
    )
    kwargs.update(overrides)
    return PotentialReward(**kwargs)


def test_static_bind_formulas_and_idempotence():
    r = _hooked()
    r.bind_board(net_count=7, bbox_w=40.0, bbox_h=25.0)
    assert r.completion_bonus == 2.0 * math.log1p(7)
    assert r.clean_completion_bonus == 3.0 * math.log1p(7)
    assert r.wirelength_penalty == 0.5 / 40.0
    # Rebinding derives from the config value again — no double division.
    r.bind_board(net_count=7, bbox_w=40.0, bbox_h=25.0)
    assert r.wirelength_penalty == 0.5 / 40.0
    # Another board re-resolves (the env binds once; the scorer per call).
    r.bind_board(net_count=3, bbox_w=10.0, bbox_h=80.0)
    assert r.completion_bonus == 2.0 * math.log1p(3)
    assert r.wirelength_penalty == 0.5 / 80.0


def test_static_bind_leaves_unhooked_knobs_alone():
    r = PotentialReward(completion_bonus=3.0, clean_completion_bonus=1.5,
                        wirelength_penalty=0.002)
    r.bind_board(net_count=50, bbox_w=100.0, bbox_h=100.0)
    assert (r.completion_bonus, r.clean_completion_bonus, r.wirelength_penalty) \
        == (3.0, 1.5, 0.002)
    assert r.net_size_weights_by_code is None


def test_reset_bind_weights_sorted_by_code_and_named():
    r = _hooked()
    r.bind_board(
        pad_groups={3: 4, 1: 2, 2: 3, 9: 1},
        net_names={1: "A", 2: "B", 3: "C", 9: "S"},
        routable_nets=frozenset({3, 1, 2}),  # 9 is single-pad: not routable
    )
    assert list(r.net_size_weights_by_code) == [1, 2, 3]
    assert r.net_size_weights_by_code == {
        1: math.log1p(2), 2: math.log1p(3), 3: math.log1p(4)}
    assert r.net_size_weights_by_name == {
        "A": math.log1p(2), "B": math.log1p(3), "C": math.log1p(4)}


def test_reset_bind_is_noop_without_size_scale():
    r = _hooked(net_bonus_size_log_scale=0.0)
    r.bind_board(pad_groups={1: 2}, net_names={1: "A"}, routable_nets={1})
    assert r.net_size_weights_by_code is None


def test_reset_bind_fails_loudly_on_missing_inputs():
    r = _hooked()
    with pytest.raises(RuntimeError, match="no pad groups"):
        r.bind_board(pad_groups={1: 2}, net_names={1: "A", 2: "B"},
                     routable_nets={1, 2})
    with pytest.raises(RuntimeError, match="no net name"):
        r.bind_board(pad_groups={1: 2, 2: 2}, net_names={1: "A"},
                     routable_nets={1, 2})


def test_partial_or_empty_groups_rejected():
    r = _hooked()
    with pytest.raises(ValueError, match="partial groups"):
        r.bind_board(net_count=3)
    with pytest.raises(ValueError, match="partial groups"):
        r.bind_board(pad_groups={1: 2}, routable_nets={1})
    with pytest.raises(ValueError, match="nothing to bind"):
        r.bind_board()
    with pytest.raises(ValueError, match="positive board bbox"):
        r.bind_board(net_count=3, bbox_w=0.0, bbox_h=0.0)


# ---------------------------------------------------------------------------
# env ↔ offline scorer parity (real engine, every reward rule)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", ALL_RULES)
def test_env_and_offline_scorer_price_the_board_identically(
    rule, sw_rule_path, tmp_path,
):
    from eval.metrics import compute_metrics_inline, evaluate_one
    from pcb_world.core.env import PCBWorld

    rule_arg = _rule_arg(rule, sw_rule_path)
    routed_pcb = tmp_path / "routed.kicad_pcb"

    env = PCBWorld(
        str(BOARD),
        EnvConfig.with_overrides(max_steps=200, masking_rule="default_no_finish", reward_rule=rule_arg),
    )
    try:
        env.reset()
        pot = env._potential_reward
        env_resolved = _resolved(pot)
        phi0_env = env._initial_potential

        steps = _route_all_nets(env)
        assert steps[-1][1], "scripted route must terminate the episode"
        fp_env = steps[-1][3]["final_potential"]

        # (b) inline scorer on the live env: same engine, same snapshot →
        # bit-identical Φ and identical resolved terms.
        inline = compute_metrics_inline(env, reward_config_name=rule_arg)
        snap = env._engine.get_reward_snapshot(run_drc=True)
        phi_env_same_snapshot = pot.compute_final(RewardState.from_snapshot(snap))
        env._engine.save(str(routed_pcb))
    finally:
        env.close()
        del env
        gc.collect()
    # Design rules are the fixture's own (routing changes no rule), so give the
    # routed board the fixture's .kicad_pro explicitly rather than relying on
    # the engine's project write: under xdist, a sibling worker holding the
    # fixture project's KiCad lock file opens ours read-only, and a read-only
    # PROJECT silently skips its .kicad_pro on save (same pattern as
    # tests/test_eval_routability.py).
    routed_pro = routed_pcb.with_suffix(".kicad_pro")
    routed_pro.write_bytes(BOARD.with_suffix(".kicad_pro").read_bytes())

    assert _resolved_from_metrics(inline) == env_resolved
    assert inline["final_potential"] == phi_env_same_snapshot

    # (a) + (c) offline path: reload the saved board from disk.
    res = evaluate_one(str(routed_pcb), str(routed_pro), reward_config_name=rule_arg)
    gc.collect()  # release the scorer's engine before the next PCBWorld
    assert _resolved_from_metrics(res) == env_resolved
    assert res["final_potential"] == pytest.approx(fp_env, rel=1e-9, abs=1e-9)
    assert res["initial_potential"] == pytest.approx(phi0_env, rel=1e-9, abs=1e-9)
    if rule == SW_RULE:
        # The hook-heavy rule must actually exercise every hook.
        assert env_resolved["net_size_weights"]
        assert env_resolved["clean_completion_bonus"] > 0
        assert env_resolved["wirelength_penalty"] < 0.5


def _route_and_score_inline(tmp_path, train_rule: str, score_rule: str):
    """Route the fixture board under ``train_rule``, score it inline with ``score_rule`` and save
    it -> (env Φ₀, inline result, saved board path)."""
    from eval.metrics import compute_metrics_inline
    from pcb_world.core.env import PCBWorld

    routed_pcb = tmp_path / "routed.kicad_pcb"
    env = PCBWorld(str(BOARD), EnvConfig.with_overrides(
        max_steps=200, masking_rule="default_no_finish", reward_rule=train_rule))
    try:
        env.reset()
        phi0_env = env._initial_potential
        assert _route_all_nets(env)[-1][1], "scripted route must terminate the episode"
        inline = compute_metrics_inline(env, reward_config_name=score_rule)
        env._engine.save(str(routed_pcb))
    finally:
        env.close()
        del env
        gc.collect()
    routed_pcb.with_suffix(".kicad_pro").write_bytes(BOARD.with_suffix(".kicad_pro").read_bytes())
    return phi0_env, inline, routed_pcb


def test_inline_scoring_prices_phi0_with_the_scoring_config(tmp_path):
    """A scoring config other than the env's training rule: the inline row's Φ₀ / potential gain
    are the scoring config's, as post-hoc scoring of the same routed board gives them (they were
    the training rule's Φ₀ beside the scoring config's final Φ)."""
    from eval.metrics import evaluate_one

    train, score = "drc_linear", "drc_dense_errors_only_eval"
    phi0_env, inline, routed_pcb = _route_and_score_inline(tmp_path, train, score)
    post = evaluate_one(str(routed_pcb), str(routed_pcb.with_suffix(".kicad_pro")),
                        reward_config_name=score)
    gc.collect()
    assert post["initial_potential"] != pytest.approx(phi0_env)   # the two rules price Φ₀ apart
    for key in ("initial_potential", "potential_gain", "final_potential"):
        assert inline[key] == pytest.approx(post[key], rel=1e-9, abs=1e-9), key
    assert inline["drv_errors_only_count"] == post["drv_errors_only_count"]
    assert inline["env_phi0_matches"] is False   # crash rows' env Φ₀ is not the scorer's
    # the scored row replaces the rollout's env-info Φ₀ (training rule) with the scorer's
    from eval.eval_utils import assemble_rollout_row
    row = assemble_rollout_row(inline, runtime={"initial_potential": phi0_env, "potential_gain": 0.0},
                               reward_config=score)
    assert (row["initial_potential"], row["potential_gain"]) == \
        pytest.approx((post["initial_potential"], post["potential_gain"]))


def test_inline_scoring_without_reset_drc_leaves_phi0_blank(tmp_path):
    """An env whose reward reads no DRC ran none at reset, so a DRC-reading scoring config cannot
    price its Φ₀ inline: blank with a warning, never the training rule's value."""
    import math

    from eval.eval_utils import flatten_scored_result

    with pytest.warns(UserWarning, match="reset ran none"):
        _, inline, _ = _route_and_score_inline(tmp_path, "default", "drc_dense_errors_only_eval")
    assert inline["initial_potential"] is None and inline["potential_gain"] is None
    row = flatten_scored_result(inline, reward_config="drc_dense_errors_only_eval")
    assert math.isnan(row["initial_potential"]) and math.isnan(row["potential_gain"])
    assert row["env_phi0_matches"] is False


def test_inline_scoring_with_the_training_phi_marks_the_env_phi0_as_the_scorers(tmp_path):
    """Same Φ for env and scorer: the env's Φ₀ (what crash rows carry) is the scorer's Φ₀."""
    rule = "drc_dense_errors_only_eval"
    phi0_env, inline, _ = _route_and_score_inline(tmp_path, rule, rule)
    assert inline["env_phi0_matches"] is True
    assert inline["initial_potential"] == pytest.approx(phi0_env)
