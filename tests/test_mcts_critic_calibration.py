"""How the critic's affine calibration (scale · offset · trust) enters the search.

The critic is trained on normalized rewards. The trainer only divides by the
running std of the discounted returns and does not subtract the mean
(RewardNormalizer: "mean is NOT subtracted"), so denormalization is the single
factor V_raw = sigma * V~ — regressing the realized return G on sigma*V~ with a
free intercept gives a slope near 1 (0.98~1.42), which confirms it. What was
missing was the constant: a completed board has 0 residual return while its
measured V~ is +0.920 (maytal) to -1.007 (NiMH). Hence

    boot = trust * scale * (V~(s) - offset)

What this file pins:
  1. the defaults (offset 0, trust 1) are bit-identical to the previous behavior
  2. with offset at the anchor, a completed state bootstraps to exactly 0
  3. trust scales the bootstrap linearly, and turns it off entirely at 0
  4. a terminal leaf is always path_return, independent of trust/offset (no bootstrap)

  5. a single-child root finishes in one simulation (no budget on a forced move)
  6. budget accounting of the three invalid modes (pop / drop / penalize)
  7. the rejected knobs stay removed (below)
"""
from __future__ import annotations

import pytest

from methods._shared.mcts.config import MctsConfig
from methods._shared.mcts.protocols import NodeState, StepResult
from methods._shared.mcts.search import (
    _bootstrap_from, _terminal_value, run_search,
)

SIG, K = 8.5969, 0.9278          # d2b ckpt reward-norm std, maytal completed-state V~


def test_default_is_plain_scale():
    """With offset=0 and trust=1 the bootstrap is plain scale*V, as before."""
    cfg = MctsConfig(gamma=0.9, critic_scale=5.0)
    assert _bootstrap_from(None, cfg, 0.0, 1.0, 2.0) == 10.0
    assert _bootstrap_from(None, cfg, 3.0, 0.9 ** 3, 2.0) == pytest.approx(3.0 + 0.729 * 10.0)


def test_anchor_zeroes_the_bootstrap_at_completion():
    """With the anchor as offset, the bootstrap vanishes in a completed state."""
    cfg = MctsConfig(gamma=0.9, critic_scale=SIG, critic_offset=K)
    assert _bootstrap_from(None, cfg, 4.0, 1.0, K) == pytest.approx(4.0)
    # Only a V above the anchor leaves a positive contribution.
    assert _bootstrap_from(None, cfg, 0.0, 1.0, K + 0.5) == pytest.approx(SIG * 0.5)
    assert _bootstrap_from(None, cfg, 0.0, 1.0, K - 0.5) == pytest.approx(-SIG * 0.5)


@pytest.mark.parametrize("trust", [0.0, 0.07, 0.25, 0.83, 1.0])
def test_trust_scales_the_bootstrap_linearly(trust):
    base = MctsConfig(gamma=0.9, critic_scale=SIG, critic_offset=K)
    cfg = MctsConfig(gamma=0.9, critic_scale=SIG, critic_offset=K, critic_lambda=trust)
    full = _bootstrap_from(None, base, 2.0, 0.9 ** 2, 2.0) - 2.0
    got = _bootstrap_from(None, cfg, 2.0, 0.9 ** 2, 2.0) - 2.0
    assert got == pytest.approx(trust * full)


def test_trust_zero_turns_the_critic_off():
    cfg = MctsConfig(gamma=0.9, critic_scale=SIG, critic_offset=K, critic_lambda=0.0)
    for v in (-3.0, 0.0, K, 5.0):
        assert _bootstrap_from(None, cfg, 7.0, 0.9 ** 4, v) == pytest.approx(7.0)


@pytest.mark.parametrize("cfg", [
    MctsConfig(gamma=0.9, critic_scale=SIG),
    MctsConfig(gamma=0.9, critic_scale=SIG, critic_offset=K, critic_lambda=0.5),
])
def test_terminal_leaf_never_bootstraps(cfg):
    """A true terminal has 0 residual return, so it must stay at path_return."""
    assert _terminal_value(None, cfg, 5.0) == 5.0


def test_removed_knobs_are_gone():
    """Knobs judged ineffective and reverted must not come back.

    roll_* was a leaf-evaluation rule that advanced a leaf reached over a
    ΔΦ = 0 edge to the first ΔΦ != 0 step instead of bootstrapping it in
    place. It genuinely widened the sibling-Q gap (0.001 -> 0.138 on
    net_select d0, holding under argmax too, so not sampling noise) and
    reported 0.000 on truly symmetric branches — the estimator behaved
    correctly. Yet it lost to base on all 3 boards: rout 1.000/0.447/0.816
    (base) vs 1.000/0.421/0.447 (argmax roll), wallclock 1.7~3x. The reason:
    that very decision in base collapses to argmax(g + logit) = prior
    sampling at a q-hat gap of 0.001 — on ΔΦ ≡ 0 nodes prior sampling beats
    the 1-step-lookahead ΔΦ ranking.
    """
    for gone in ("q_range_floor", "bootstrap_gamma", "terminal_bootstrap",
                 "roll_to_informative", "roll_max_steps", "roll_try_cap",
                 "roll_greedy"):
        assert not hasattr(MctsConfig(), gone), gone


# ---------------------------------------------------------------------------
# Single-child root — no budget spent on a forced move
# ---------------------------------------------------------------------------
class _OneChildEnv:
    """SearchEnv with a single legal action; ``invalid`` makes that one invalid."""

    def __init__(self, invalid=False, n_legal=1):
        self.invalid, self.n_legal = invalid, n_legal
        self.steps = 0

    def checkpoint(self):
        return NodeState(l1=0)

    def restore(self, st):
        pass

    def release(self, st):
        pass

    def step(self, a):
        self.steps += 1
        return StepResult(reward=1.0, done=False, info={}, invalid=self.invalid)

    def legal_actions(self):
        return [(2, -1, -1)][:1] * 0 + [(2, -1, -1 - i) for i in range(self.n_legal)]

    def potential(self):
        return 0.0

    def observe(self):
        return 0

    def __call__(self, obs, legal):
        return {a: 1.0 for a in legal}, 0.5


def test_single_child_root_spends_one_simulation():
    """With a single action (as at net_end) the search takes 1 step, not the
    full 32-simulation budget."""
    env = _OneChildEnv()
    cfg = MctsConfig(n_simulations=32, root_selection="gumbel", seed=0)
    action, visits = run_search(env, env, cfg)
    assert action == (2, -1, -1)
    assert env.steps == 1, env.steps          # 1 step, not 32
    assert visits == {(2, -1, -1): 1}


def test_single_child_root_still_detects_an_invalid_action():
    """A single simulation still catches the invalid action — pop empties the
    root and the search returns action=None."""
    env = _OneChildEnv(invalid=True)
    cfg = MctsConfig(n_simulations=32, root_selection="gumbel", seed=0,
                     invalid_mode="pop")
    action, _ = run_search(env, env, cfg)
    assert action is None
    assert env.steps == 1


def test_multi_child_root_still_spends_the_full_budget():
    env = _OneChildEnv(n_legal=3)
    cfg = MctsConfig(n_simulations=12, root_selection="gumbel", seed=0)
    run_search(env, env, cfg)
    assert env.steps == 12, env.steps


# ---------------------------------------------------------------------------
# Budget accounting for the three invalid_modes
# ---------------------------------------------------------------------------
class _InvalidEnv:
    """SearchEnv with 6 root children, of which the given ones are invalid."""

    def __init__(self, bad):
        self.bad = {(3, i, 2) for i in bad}
        self.steps = 0

    def checkpoint(self):
        return NodeState(l1=0)

    def restore(self, st):
        pass

    def release(self, st):
        pass

    def step(self, a):
        self.steps += 1
        return StepResult(reward=1.0, done=False, info={},
                          invalid=(a in self.bad))

    def legal_actions(self):
        return [(3, i, 2) for i in range(6)]

    def potential(self):
        return 0.0

    def observe(self):
        return 0

    def __call__(self, obs, legal):
        return {a: 1.0 for a in legal}, 0.0


def _run(mode, n_sim=8):
    env = _InvalidEnv(bad=(1, 3, 4, 5))
    cfg = MctsConfig(n_simulations=n_sim, root_selection="gumbel", seed=0,
                     invalid_mode=mode, max_depth=1)
    action, visits = run_search(env, env, cfg)
    return env, action, visits


def test_pop_does_not_spend_the_simulation_budget():
    """pop re-selects inside the same simulation, so all n_sim simulations back up.

    Side effect: one simulation may spend env.step more than once, so
    n_simulations does not bound the engine workload (measured: 735 pops per
    rollout = 21.5% of all env.step calls).
    """
    env, _a, visits = _run("pop")
    assert sum(visits.values()) == 8          # all 8 simulations back up
    assert env.steps == 6                     # 2 valid children + 4 invalid


def test_drop_spends_the_simulation_and_backs_up_nothing():
    """drop consumes the simulation — n_simulations is the step budget."""
    env, _a, visits = _run("drop")
    assert sum(visits.values()) == 4          # 4 of the 8 spent finding invalid children
    assert env.steps == 6


def test_penalize_spends_the_simulation_too():
    env, _a, visits = _run("penalize")
    assert sum(visits.values()) == 4


@pytest.mark.parametrize("mode", ["pop", "drop", "penalize"])
def test_every_mode_removes_the_invalid_child(mode):
    """All three modes remove the invalid child — it can be neither re-selected
    nor returned."""
    _env, action, visits = _run(mode)
    assert action not in {(3, i, 2) for i in (1, 3, 4, 5)}
    assert not ({(3, i, 2) for i in (1, 3, 4, 5)} & set(visits))
