"""MCTS core (methods._shared.mcts) — validated on a toy deterministic tree.

No KiCad / torch: a mock SearchEnv exercises select / expand / backprop, the
restore-to-node lockstep, terminal handling, and checkpoint reaping. The leaf
value is the discounted per-edge return (Σ γ^k ΔΦ_k + γ^depth·c·V), so the mock
env supplies each step's ΔΦ as its reward.
"""

import pytest

from methods._shared.mcts import (
    AlphaZero,
    GumbelMuZero,
    MctsConfig,
    MuZero,
    NodeState,
    SearchAlgorithm,
    StepResult,
    make_algorithm,
    run_search,
    search_iter,
)


class MockEnv:
    """Depth-2 tree. Leaf (a, b) has potential a*10 + b, so the best first action
    is 2 (its leaves are 20, 21). Each step's reward is the potential delta ΔΦ, so
    the per-edge return sums to Φ(leaf)−Φ(root). Tracks live checkpoints to assert
    reaping and the position to assert restore-to-node correctness."""

    def __init__(self):
        self.pos: tuple = ()
        self._live: dict[int, tuple] = {}
        self._next = 0
        self.max_live = 0
        self.steps = 0
        self.max_pos_len = 0      # deepest node ever realized (for max_depth tests)

    # --- SearchEnv protocol ---
    def checkpoint(self) -> NodeState:
        h = self._next
        self._next += 1
        self._live[h] = self.pos
        self.max_live = max(self.max_live, len(self._live))
        return NodeState(l1=h, l2=("L2", self.pos))   # l2 carried opaquely by the core

    def restore(self, s: NodeState) -> None:
        assert s.l1 in self._live, "restored a released handle"
        self.pos = self._live[s.l1]
        assert s.l2 == ("L2", self.pos), "L2 not in lockstep with L1"

    def release(self, s: NodeState) -> None:
        self._live.pop(s.l1, None)

    def step(self, action) -> StepResult:
        before = self.potential()
        self.steps += 1
        self.pos = self.pos + (action,)
        self.max_pos_len = max(self.max_pos_len, len(self.pos))
        return StepResult(reward=self.potential() - before,   # ΔΦ
                          done=len(self.pos) >= 2, info={})

    def legal_actions(self):
        d = len(self.pos)
        return [0, 1, 2] if d == 0 else ([0, 1] if d == 1 else [])

    def potential(self) -> float:
        p = 0.0
        if len(self.pos) >= 1:
            p += self.pos[0] * 10
        if len(self.pos) >= 2:
            p += self.pos[1]
        return float(p)

    def observe(self):
        return self.pos


def _uniform(obs, legal):
    p = 1.0 / len(legal)
    return {a: p for a in legal}, None


def _biased(obs, legal):
    """Prior ∝ (action + 1): higher action index → higher prior."""
    w = {a: float(a) + 1.0 for a in legal}
    z = sum(w.values())
    return {a: w[a] / z for a in legal}, None


def test_finds_best_action():
    env = MockEnv()
    cfg = MctsConfig(n_simulations=300, c_puct=1.5, seed=0)
    action, visits = run_search(env, _uniform, cfg)
    assert action == 2                       # leads to the highest-potential leaves
    assert visits[2] == max(visits.values())


def test_search_iter_matches_run_search():
    """search_iter drained to its ``done`` event == run_search, and it yields
    exactly one per-simulation event before ``done`` (env left at root, all
    checkpoints reaped — same post-conditions as run_search)."""
    cfg_kwargs = dict(n_simulations=50, c_puct=1.5, seed=0)

    ref_env = MockEnv()
    ref_action, ref_visits = run_search(ref_env, _uniform, MctsConfig(**cfg_kwargs))

    it_env = MockEnv()
    sim_events, done_ev = [], None
    for ev in search_iter(it_env, _uniform, MctsConfig(**cfg_kwargs)):
        if ev.get("done"):
            done_ev = ev
        else:
            sim_events.append(ev)

    assert done_ev is not None
    assert done_ev["action"] == ref_action
    assert done_ev["visits"] == ref_visits
    assert len(sim_events) == 50             # one yield per simulation
    assert sim_events[-1]["sim"] == 50 and sim_events[-1]["n"] == 50
    assert it_env.pos == ()                  # env left at decision state
    assert len(it_env._live) == 0            # every checkpoint released


def test_search_iter_early_close_restores_and_reaps():
    """Closing the generator mid-search (e.g. GUI ESC) must still restore the
    env to the decision state and free every checkpoint (finally block)."""
    env = MockEnv()
    gen = search_iter(env, _uniform, MctsConfig(n_simulations=100, seed=1))
    for _ in range(3):                       # advance a few simulations, then bail
        next(gen)
    assert env._live                         # tree is holding checkpoints mid-search
    gen.close()                              # GeneratorExit → finally cleanup
    assert env.pos == ()                     # restored to decision state
    assert len(env._live) == 0               # all checkpoints released


def test_reaping_and_env_left_at_root():
    env = MockEnv()
    cfg = MctsConfig(n_simulations=100, seed=1)
    run_search(env, _uniform, cfg)
    assert len(env._live) == 0               # every checkpoint released
    assert env.pos == ()                     # env left at the decision (root) state
    assert env.max_live > 1                  # the search really held multiple checkpoints


def test_terminal_root_returns_none():
    env = MockEnv()
    env.pos = (1, 0)                         # already terminal (no legal actions)
    action, visits = run_search(env, _uniform, MctsConfig(n_simulations=10))
    assert action is None and visits == {}
    assert len(env._live) == 0


def test_visit_counts_sum_matches_simulations():
    env = MockEnv()
    cfg = MctsConfig(n_simulations=50, seed=2)
    _, visits = run_search(env, _uniform, cfg)
    assert sum(visits.values()) == 50        # every simulation descends through the root


class FlatEnv(MockEnv):
    """All values equal (reward 0, Φ 0) — forces pure round-robin exploration, so
    an exact visit tie isolates the final-pick tie-break rule from the value."""

    def step(self, action) -> StepResult:
        self.steps += 1
        self.pos = self.pos + (action,)
        self.max_pos_len = max(self.max_pos_len, len(self.pos))
        return StepResult(reward=0.0, done=len(self.pos) >= 2, info={})

    def potential(self) -> float:
        return 0.0


def test_alphazero_final_pick_randomizes_ties_muzero_does_not():
    """Deep-comparison finding: az-general's ``getActionProb`` breaks a visit-count
    tie at the final pick UNIFORMLY AT RANDOM (``np.random.choice(bestAs)``); the
    generic ``_pick_action`` (used by MuZero / hand-built configs) keeps the
    first-scanned action instead — muzero-general's ``select_action`` tie-break. A
    flat value landscape + n_sim == #root actions forces an exact 3-way visit tie
    (round-robin exploration), isolating the final-pick rule."""
    def pick(seed, **cfg_kwargs):
        env = FlatEnv()
        cfg = MctsConfig(n_simulations=3, seed=seed, **cfg_kwargs)
        action, visits = run_search(env, _uniform, cfg)
        assert visits == {0: 1, 1: 1, 2: 1}       # confirms the 3-way tie
        return action

    # MuZero / hand-built ("custom"): deterministic — always the first action.
    assert {pick(seed) for seed in range(20)} == {0}
    assert {pick(seed, algorithm="muzero") for seed in range(20)} == {0}

    # AlphaZero: randomized — varies with the seed, same seed reproduces.
    az_actions = {pick(seed, algorithm="alphazero") for seed in range(20)}
    assert len(az_actions) > 1
    assert pick(7, algorithm="alphazero") == pick(7, algorithm="alphazero")


def test_muzero_first_pick_ignores_prior_and_randomizes():
    """Deep-comparison finding (MuZero pass): muzero-general's ``ucb_score`` uses a
    RAW ``sqrt(parent.visit_count)`` with no epsilon guard, so at a freshly-expanded
    node's very first pick (``parent.visit_count == 0``) ``pb_c`` is EXACTLY 0 for
    every child — the prior plays NO role and ``np.random.choice`` picks uniformly
    among ALL children. The generic ``_puct_select`` / ``_az_select`` floor the sqrt
    at 1, so they stay prior-driven (deterministic here) from the very first pick."""
    def pick(seed, **cfg_kwargs):
        env = MockEnv()
        cfg = MctsConfig(n_simulations=1, seed=seed, **cfg_kwargs)
        action, _ = run_search(env, _biased, cfg)   # prior: action 2 > 1 > 0
        return action

    # Generic ("custom") and AlphaZero: first pick is prior-driven — always the
    # highest-prior action, deterministic regardless of seed.
    assert {pick(seed) for seed in range(15)} == {2}
    assert {pick(seed, algorithm="alphazero") for seed in range(15)} == {2}

    # MuZero: pb_c == 0 at the very first pick → uniformly random, prior ignored.
    muzero_actions = {pick(seed, algorithm="muzero") for seed in range(30)}
    assert len(muzero_actions) > 1
    assert pick(11, algorithm="muzero") == pick(11, algorithm="muzero")   # reproducible


class ShortLongEnv(MockEnv):
    """Root action 0 (SHORT) → terminal at depth 1, Φ=10. Action 1 (LONG) → a
    non-terminal at depth 1 (Φ=1), whose only action → terminal at depth 2, Φ=10.
    Same terminal Φ via different depths; per-step reward = ΔΦ. Undiscounted the two
    paths tie (Σ ΔΦ = 10 either way); a discount γ<1 makes SHORT (γ⁰·10) beat LONG
    (γ⁰·1 + γ¹·9)."""

    def step(self, action) -> StepResult:
        before = self.potential()
        self.steps += 1
        self.pos = self.pos + (action,)
        self.max_pos_len = max(self.max_pos_len, len(self.pos))
        return StepResult(reward=self.potential() - before,   # ΔΦ
                          done=self.pos in ((0,), (1, 0)), info={})

    def legal_actions(self):
        if self.pos == ():
            return [0, 1]
        if self.pos == (1,):
            return [0]
        return []

    def potential(self) -> float:
        if self.pos in ((0,), (1, 0)):
            return 10.0            # both terminals reach the same Φ
        if self.pos == (1,):
            return 1.0             # LONG's intermediate node
        return 0.0                 # root


def test_gamma_prefers_shorter_completion():
    """SHORT (depth-1) and LONG (depth-2) reach the SAME terminal Φ=10, so their
    discounted returns tie at γ=1 (Σ ΔΦ = 10). A discount γ<1 weights later reward
    less: SHORT = 10 vs LONG = 1 + 9γ, so SHORT strictly wins."""
    env = ShortLongEnv()
    cfg = MctsConfig(n_simulations=64, c_puct=1.5, seed=0, critic_scale=0.0, gamma=0.8)
    a_on, v_on = run_search(env, _uniform, cfg)
    assert a_on == 0                        # SHORT chosen under the discount
    assert v_on[0] > v_on.get(1, 0)         # and visited more than LONG


def test_progressive_widening_admits_by_visits():
    """k = ceil(pw_base·N^pw_alpha) selectable children, admitted in prior order and
    monotonically widening with the node's visit count; pw_alpha=0 = all children."""
    from methods._shared.mcts.node import Node
    from methods._shared.mcts.search import _active_children
    node = Node()
    node.expand([0, 1, 2, 3, 4, 5],
                {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.05, 5: 0.05})
    cfg = MctsConfig(pw_alpha=0.5, pw_base=1.0)
    node.N = 1                              # k = ceil(1·1) = 1
    assert set(_active_children(node, cfg)) == {0}
    node.N = 4                              # k = ceil(1·2) = 2
    assert set(_active_children(node, cfg)) == {0, 1}
    node.N = 16                             # k = ceil(1·4) = 4
    assert set(_active_children(node, cfg)) == {0, 1, 2, 3}
    node.N = 100                            # k ≥ 6 → all
    assert set(_active_children(node, cfg)) == {0, 1, 2, 3, 4, 5}
    # disabled: every child always selectable
    assert set(_active_children(node, MctsConfig(pw_alpha=0.0))) == {0, 1, 2, 3, 4, 5}


def test_progressive_widening_runs_end_to_end():
    """A full search with progressive widening on reaps cleanly and picks a legal
    action (widening must not strand checkpoints or crash)."""
    env = MockEnv()
    cfg = MctsConfig(n_simulations=64, seed=0, pw_alpha=0.5, pw_base=1.0)
    action, visits = run_search(env, _biased, cfg)
    assert action in (0, 1, 2)
    assert sum(visits.values()) == 64
    assert len(env._live) == 0


def test_gumbel_interior_ignores_progressive_widening():
    """Progressive widening must NOT gate the Gumbel interior selector: its rule
    drives visits toward the improved policy over the FULL child set, so a widened
    (non-top-prior) child can be selected even when PW would restrict to one."""
    from methods._shared.mcts.node import Node
    from methods._shared.mcts.search import _active_children, _gumbel_interior_select

    node = Node()
    node.expand([0, 1, 2, 3], {0: 0.7, 1: 0.1, 2: 0.1, 3: 0.1})
    node.N = 1
    node.raw_value = 1.0
    node.children[0].N = 1        # top-prior child visited → Q=0 + visit penalty
    node.children[0].W = 0.0      # (others unvisited → completed with mixed value)

    cfg = MctsConfig(root_selection="gumbel", pw_alpha=2.0, pw_base=1.0)
    # PW WOULD restrict the selectable set to the single top-prior child...
    assert set(_active_children(node, cfg)) == {0}
    # ...but the Gumbel interior rule sees ALL children and widens past it.
    assert _gumbel_interior_select(node, cfg).action != 0


def test_max_depth_caps_lookahead():
    """max_depth=1 on the depth-2 tree: depth-1 nodes are bootstrap value-leaves,
    so the search never realizes (steps into) a depth-2 node."""
    env = MockEnv()
    cfg = MctsConfig(n_simulations=30, max_depth=1, seed=0)
    action, visits = run_search(env, _uniform, cfg)
    assert env.max_pos_len == 1             # never stepped past depth 1
    assert action in (0, 1, 2)
    assert len(env._live) == 0              # checkpoints reaped


class InvalidActionEnv(MockEnv):
    """Root action 0 is a no-op dead-end (StepResult.invalid=True, board
    unchanged); actions 1 and 2 are normal."""

    def __init__(self):
        super().__init__()
        self.invalid_steps = 0

    def step(self, action):
        if len(self.pos) == 0 and action == 0:
            self.invalid_steps += 1
            return StepResult(reward=0.0, done=False, info={}, invalid=True)
        return super().step(action)


def test_invalid_action_popped():
    """A StepResult.invalid child is REMOVED from the tree — never in visit counts,
    stepped exactly once, no leaked checkpoint — so it neither pollutes the parent Q
    nor stretches the min-max floor."""
    env = InvalidActionEnv()
    cfg = MctsConfig(n_simulations=40, seed=0)
    action, visits = run_search(env, _uniform, cfg)
    assert action != 0                      # dead-end avoided (1 or 2 chosen)
    assert 0 not in visits                  # dropped entirely (not just low-visited)
    assert env.invalid_steps == 1           # realized once, then removed
    assert len(env._live) == 0              # checkpoints reaped (released on pop)


def test_invalid_action_penalize_backprops_once_then_removes():
    """Under invalid_mode="penalize" a StepResult.invalid child is scored ONCE
    (penalty backed up to ancestors) and THEN removed — so it is stepped exactly
    once, leaves no checkpoint, and is still never returned/visit-counted, but
    unlike "pop" its single backprop consumed a simulation."""
    env = InvalidActionEnv()
    cfg = MctsConfig(n_simulations=40, seed=0,
                     invalid_mode="penalize", invalid_penalty=0.5)
    action, visits = run_search(env, _uniform, cfg)
    assert action != 0                      # dead-end still avoided
    assert 0 not in visits                  # removed after its one penalized backprop
    assert env.invalid_steps == 1           # realized once, then removed (no re-select)
    assert len(env._live) == 0              # checkpoints reaped


class AllInvalidBelowEnv(MockEnv):
    """Depth-1 node reached by action 1 has ONLY invalid actions below it (a true
    dead-end); action 2's subtree is normal. Tests the all-children-invalid path."""

    def step(self, action):
        if len(self.pos) == 1 and self.pos[0] == 1:
            return StepResult(reward=0.0, done=False, info={}, invalid=True)
        return super().step(action)


def test_all_invalid_children_is_dead_end():
    """A node whose every child pops becomes a dead-end (marked terminal) — the
    search must not loop or crash, and must reap every checkpoint."""
    env = AllInvalidBelowEnv()
    cfg = MctsConfig(n_simulations=50, seed=0)
    action, visits = run_search(env, _uniform, cfg)
    assert action in (0, 1, 2)
    assert len(env._live) == 0              # no leaked checkpoints despite dead-ends


# --- algorithm profiles (methods._shared.mcts.algorithms) ---------------------

def test_profiles_encode_their_module_set():
    """Each named profile emits the coherent module set that defines it."""
    az = AlphaZero().config()
    assert az.root_selection == "puct"
    assert az.dirichlet_alpha == 0.0        # az-general repo: no root noise
    assert az.value_completion is False
    assert az.algorithm == "alphazero"      # dispatches to _az_select/_az_final_action

    mz = MuZero().config()
    assert mz.root_selection == "puct"
    assert mz.dirichlet_alpha > 0.0         # muzero-general: root Dirichlet ON
    assert mz.value_completion is False
    assert mz.algorithm == "muzero"         # shares the generic PUCT/visits modules

    gz = GumbelMuZero().config()
    assert gz.root_selection == "gumbel"    # Gumbel + Sequential Halving
    assert gz.value_completion is True      # completed Q at interior (mctx)
    assert gz.dirichlet_alpha == 0.0        # Gumbel replaces Dirichlet


def test_profile_tunables_and_overrides():
    """Tunables pass through the constructor; config(**overrides) wins over the
    profile's own field values."""
    cfg = GumbelMuZero(n_simulations=128, seed=7).config(max_depth=6, gamma=0.9)
    assert cfg.n_simulations == 128 and cfg.seed == 7
    assert cfg.max_depth == 6 and cfg.gamma == 0.9   # overrides applied
    assert cfg.root_selection == "gumbel"   # profile axis preserved
    # canonical AlphaZero: opt back into root Dirichlet via the constructor
    assert AlphaZero(dirichlet_alpha=0.3).config().dirichlet_alpha == 0.3


def test_make_algorithm_by_name():
    assert isinstance(make_algorithm("muzero"), MuZero)
    assert isinstance(make_algorithm("GUMBEL"), GumbelMuZero)   # case-insensitive
    with pytest.raises(ValueError):
        make_algorithm("nope")


@pytest.mark.parametrize("profile", [AlphaZero, MuZero, GumbelMuZero, SearchAlgorithm])
def test_every_profile_runs_end_to_end(profile):
    """Each profile's config drives a full search on the toy tree, reaps cleanly,
    and returns a legal root action (Gumbel exercises Sequential Halving + the
    gumbel final pick; the rest exercise PUCT)."""
    env = MockEnv()
    cfg = profile(n_simulations=64, seed=0).config()
    action, visits = run_search(env, _uniform, cfg)
    assert action in (0, 1, 2)
    assert sum(visits.values()) == 64
    assert len(env._live) == 0
