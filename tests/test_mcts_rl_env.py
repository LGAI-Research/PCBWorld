"""RL binding integration: MCTS over a real KiCadRLWrapper.

Drives the branch-agnostic core through ``RLSearchEnv`` on a real board with a
uniform policy — validating that the incremental L1 restore + L2 snapshot + Φ
value + reaping all work end-to-end through the search (no neural policy needed).
"""

import os

import numpy as np
import pytest

from pcb_world.core.env import PCBWorld
from pcb_world.core.action_schema import ACT_NET_SELECT
from methods.rl_agent.wrappers.adapter import KiCadRLWrapper
from methods.rl_agent.policy.mcts_env import (
    RLSearchEnv, BaselinePolicyValue, MemoizingPolicyValue, _obs_fingerprint,
)
from methods._shared.mcts import MctsConfig, run_search

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"
_CKPT = os.path.expanduser("~/policy_best.pt")


@pytest.fixture
def rlenv():
    env = PCBWorld(board_path=BOARD, max_steps=20)
    wrapper = KiCadRLWrapper(env)
    wrapper.reset()
    yield RLSearchEnv(wrapper), env
    if env._engine is not None and env._engine.is_routing():
        env._engine.cancel_route()
    env.close()


def test_root_legal_actions_are_net_select(rlenv):
    se, _ = rlenv
    legal = list(se.legal_actions())
    assert legal, "fresh board should offer at least one legal action"
    # nothing routed yet -> the grammar only allows net_select at the root
    assert all(a[0] == ACT_NET_SELECT for a in legal)


def test_search_runs_and_reaps(rlenv):
    se, env = rlenv
    legal0 = list(se.legal_actions())
    action, visits = run_search(
        se, BaselinePolicyValue(), MctsConfig(n_simulations=16, seed=0),
    )
    assert action in legal0                        # returns a legal root action
    assert sum(visits.values()) == 16              # the search actually ran
    assert env._engine.checkpoint_count() == 0     # every checkpoint released
    assert not env._engine.is_routing()            # env left at the (root) decision state


def test_potential_is_finite(rlenv):
    se, _ = rlenv
    v = se.potential()
    assert isinstance(v, float)
    assert v == v and abs(v) < float("inf")        # not NaN / inf


def test_bounded_lookahead_runs_on_real_board(rlenv):
    """Per-edge discounted return end-to-end on the real engine with a max_depth
    cap: full budget runs, checkpoints reaped, env left at the root."""
    se, env = rlenv
    action, visits = run_search(
        se, BaselinePolicyValue(),
        MctsConfig(n_simulations=20, max_depth=3, gamma=0.9, seed=0),
    )
    assert action is not None                       # planned a legal root action
    assert sum(visits.values()) == 20              # full simulation budget ran
    assert env._engine.checkpoint_count() == 0     # every checkpoint released
    assert not env._engine.is_routing()            # env left at the (root) decision state


# ---------------------------------------------------------------------------
# DRC-visibility guard (no engine — pure construction-time check)
# ---------------------------------------------------------------------------
class _StubWrapper:
    """Minimal wrapper exposing only what ``RLSearchEnv.__init__`` reads."""

    def __init__(self, drc_active: bool, mode: str) -> None:
        cfg = type("Cfg", (), {"mode": mode, "name": "stub_rule"})()
        self.env = type("Env", (), {"_drc_active": drc_active,
                                    "_reward_config": cfg})()
        self._last_obs = {}


@pytest.mark.parametrize(
    "drc_active, mode, warns",
    [
        (True, "terminal", True),    # Φ scores DRC but per-step ΔΦ cannot carry it
        (True, "per_step", False),   # the canonical MCTS setup
        (False, "terminal", False),  # reward has no DRC term at all — consistent
        (False, "per_step", False),
    ],
)
def test_drc_visibility_guard(caplog, drc_active, mode, warns):
    """Node values are per-edge ΔΦ with no leaf Φ read, so a terminal-mode reward
    rule leaves every interior leaf DRC-blind — that must not pass silently."""
    with caplog.at_level("WARNING", logger="methods.rl_agent.policy.mcts_env"):
        RLSearchEnv(_StubWrapper(drc_active, mode))
    hit = [r for r in caplog.records if "DRC-blind" in r.message]
    assert bool(hit) is warns, caplog.text


def test_memo_new_generation_drops_untouched_entries():
    """``new_generation()`` keeps only what the last decision touched.

    Committing an action makes every sibling subtree unreachable, so their
    cached boards can never recur; the committed subtree's states were touched
    during that decision and must survive.
    """
    calls = []

    def pv(obs, legal):
        calls.append(obs)
        return {a: 1.0 for a in legal}, 0.0

    memo = MemoizingPolicyValue(pv)
    a = {"router_head": {"step": 1}}
    b = {"router_head": {"step": 2}}
    memo(a, [(0, 0, -1)])
    memo(b, [(0, 0, -1)])
    assert len(memo) == 2

    memo.new_generation()          # nothing touched since -> both live
    assert len(memo) == 2
    memo(a, [(0, 0, -1)])          # only `a` touched this generation
    memo.new_generation()
    assert len(memo) == 1
    assert memo.dropped == 1
    memo(a, [(0, 0, -1)])
    assert memo.hits == 2          # `a` still cached
    memo(b, [(0, 0, -1)])
    assert memo.misses == 3        # `b` was evicted -> recomputed


@pytest.mark.skipif(not os.path.exists(_CKPT), reason="needs ~/policy_best.pt")
def test_logit_prior_matches_evaluate_on_real_board():
    """LogitPolicyValue == the teacher-forced ``evaluate_actions_and_value`` prior
    EXACTLY on a real board — all three factors (at, ptr, and the POST-pointer mode
    for make_line/make_via), state encoded once vs per-candidate. Integration guard
    over the real wrapper masks / legal set (the model-level oracle is
    test_incremental_decode ``TestFactoredExactMode``)."""
    from pathlib import Path

    import torch

    from methods.rl_agent.rollout.transformer import load_policy_from_ckpt
    from methods.rl_agent.models.loader import env_kwargs_from_checkpoint
    from methods.rl_agent.wrappers.factory import make_decoder_env
    from methods.rl_agent.policy.agent import KiCadRLAgent
    from methods.rl_agent.policy.mcts_env import LogitPolicyValue, _SingleEnvPool
    from methods.rl_agent.models.v1.encoding import (
        stack_action_masks, stack_mode_masks,
        stack_offlayer_masks, stack_net_valid_masks,
        stack_pointer_masks,
    )

    policy, ckpt_args, _ = load_policy_from_ckpt(Path(_CKPT), torch.device("cpu"))
    agent = KiCadRLAgent(policy, device=torch.device("cpu"), deterministic=True)
    env_kwargs = env_kwargs_from_checkpoint(ckpt_args, int(ckpt_args.get("max_steps", 200)))
    for k in ("board_path", "board_paths", "boards", "n_envs", "num_envs", "group_n"):
        env_kwargs.pop(k, None)
    # crossover_board = modern-format twin of crossover_legacy (the legacy
    # fixture is refused by the strict load contract).
    w = make_decoder_env("tests/fixtures/crossover_board.kicad_pcb", **env_kwargs)

    def exact_prior(obs, legal):
        """Reference prior from evaluate_actions_and_value (per-candidate rows)."""
        pool = _SingleEnvPool(w)
        n = len(legal)
        dev = getattr(agent, "device", None)
        actions = torch.tensor([[int(a[0]), int(a[1]), int(a[2])] for a in legal],
                               dtype=torch.int64, device=dev)

        def tile(arr, dtype):
            import numpy as np
            t = torch.as_tensor(np.asarray(arr), dtype=dtype, device=dev)
            return t.expand(n, *t.shape[1:]).contiguous()

        import numpy as np
        mode_np = stack_mode_masks(pool, indices=[0])
        extra = {}
        if getattr(agent, "policy_net_select", False):
            extra = {"net_valid_mask": tile(stack_net_valid_masks(pool, indices=[0]), torch.bool),
                     "allow_net_select_lp": True}
        with torch.no_grad():
            lp, _e, _v = agent.model.evaluate_actions_and_value(
                [obs] * n, actions,
                action_masks=tile(stack_action_masks(pool, indices=[0]), torch.bool),
                pointer_masks=tile(stack_pointer_masks(pool, indices=[0]), torch.int64),
                # make_line's off-layer block: the reference path must see the
                # SAME masks as LogitPolicyValue or the parity check compares two
                # different action spaces (the blocked columns shift the softmax
                # denominator even though they are absent from `legal`).
                offlayer_masks=tile(stack_offlayer_masks(pool, indices=[0]), torch.int64),
                mode_mask=(tile(mode_np, torch.bool)
                           if mode_np is not None and np.asarray(mode_np).size else None),
                **extra)
        probs = torch.softmax(lp, dim=0)
        return {a: float(probs[i]) for i, a in enumerate(legal)}

    try:
        obs, _ = w.reset()
        se = RLSearchEnv(w)
        pool = _SingleEnvPool(w)
        logit = LogitPolicyValue(agent, w)
        saw_make_line = False
        for _ in range(8):
            legal = list(se.legal_actions())
            pl, _vl = logit(obs, legal)
            pe = exact_prior(obs, legal)
            if any(a[0] in (3, 4) for a in legal):              # make_line/make_via
                saw_make_line = True
            # EXACT on every factor now (mode is post-pointer too).
            assert max(pe, key=pe.get) == max(pl, key=pl.get)   # same top action
            assert max(abs(pe[a] - pl[a]) for a in legal) < 1e-4
            a = agent.act_from_pool(pool, [obs], [0], deterministic=True)[0]
            res = se.step(a)
            obs = se.observe()
            if res.done:
                break
        assert saw_make_line                                    # exercised the mode path
    finally:
        if w.env._engine is not None and w.env._engine.is_routing():
            w.env._engine.cancel_route()
        w.env.close()


# --- prior/value memo (MemoizingPolicyValue) ----------------------------------
# Engine-free: a fake PolicyValueFn + hand-built obs dicts exercise the cache and
# its obs fingerprint. The RL-branch integration (bit-identical to cold search)
# is covered end-to-end by mcts_compare's `--no-pv-cache` A/B; here we pin the
# unit contract.

def test_obs_fingerprint_is_content_addressed():
    a = {"g": np.arange(6, dtype=np.float32), "head": np.array([1, 2])}
    b = {"head": np.array([1, 2]), "g": np.arange(6, dtype=np.float32)}  # key order flipped
    assert _obs_fingerprint(a) == _obs_fingerprint(b)                    # order-invariant
    c = dict(a); c["g"] = c["g"].copy(); c["g"][0] = 99.0
    assert _obs_fingerprint(a) != _obs_fingerprint(c)                    # array content matters
    d = dict(a); d["head"] = np.array([1, 3])
    assert _obs_fingerprint(a) != _obs_fingerprint(d)


def test_obs_fingerprint_skips_board_static():
    base = {"g": np.arange(4, dtype=np.float32)}
    o1 = dict(base, board_static={"x": np.ones(3)})
    o2 = dict(base, board_static={"x": np.zeros(9)})   # different board_static
    assert _obs_fingerprint(o1) == _obs_fingerprint(o2)  # board_static excluded (episode-constant)


def test_memo_is_transparent_and_counts():
    calls = {"n": 0}
    def fake_pv(obs, legal):
        calls["n"] += 1
        return {legal[0]: 1.0}, float(obs["g"][0])      # value keyed to obs content

    memo = MemoizingPolicyValue(fake_pv)
    o1 = {"g": np.array([7.0])}
    r1 = memo(o1, [(0, 0, 0)])
    r1b = memo({"g": np.array([7.0])}, [(0, 0, 0)])      # same content, new object
    assert r1 == r1b and calls["n"] == 1                # second call served from cache
    assert (memo.hits, memo.misses) == (1, 1)

    o2 = {"g": np.array([8.0])}                          # different obs → recompute
    r2 = memo(o2, [(0, 0, 0)])
    assert r2 != r1 and calls["n"] == 2
    assert (memo.hits, memo.misses) == (1, 2)

    memo.reset()
    assert memo.hits == 0 and memo.misses == 0 and not memo._cache


def test_memo_result_is_identical_object_to_uncached():
    """A hit returns exactly what the wrapped provider returned (bit-identical),
    so the search sees the same priors/value it would have recomputed."""
    sentinel = ({(0, 0, 0): 0.25}, 3.14159)
    memo = MemoizingPolicyValue(lambda obs, legal: sentinel)
    obs = {"g": np.array([1.0, 2.0])}
    first = memo(obs, [(0, 0, 0)])
    second = memo(obs, [(0, 0, 0)])
    assert first is sentinel and second is sentinel      # same object both times
