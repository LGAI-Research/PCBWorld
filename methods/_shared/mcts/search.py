"""MCTS core — branch-agnostic PUCT search with per-node checkpoints.

One ``run_search`` call plans for one decision: it checkpoints the current state
as the root, runs ``n_simulations`` expansions (each = restore-to-a-node + one
env step), then returns the most-visited action and leaves the env back at the
decision state with every checkpoint released.

The restore-to-node is the hot path (once per simulation) — it is what the fast
incremental L1 restore was built for.
"""

from __future__ import annotations

import math
import random
from typing import Any

from methods._shared.mcts.config import MctsConfig
from methods._shared.mcts.node import Node
from methods._shared.mcts.protocols import PolicyValueFn, SearchEnv


class _MinMaxStats:
    """Running min/max of node Q values, used to normalize Q into [0, 1] inside
    PUCT so the exploration constant ``c_puct`` is robust to the (possibly large
    or unbounded) scale of Φ(board). MuZero-style: without it a high-value branch
    permanently out-weighs the exploration bonus and sibling branches are never
    tried.
    """

    __slots__ = ("lo", "hi")

    def __init__(self) -> None:
        self.lo = math.inf
        self.hi = -math.inf

    def update(self, v: float) -> None:
        if v < self.lo:
            self.lo = v
        if v > self.hi:
            self.hi = v

    def norm(self, v: float) -> float:
        if self.hi > self.lo:
            return (v - self.lo) / (self.hi - self.lo)
        return 0.0


def run_search(
    env: SearchEnv,
    policy_value: PolicyValueFn,
    cfg: MctsConfig,
    rng: random.Random | None = None,
) -> tuple[Any, dict[Any, int]]:
    """Plan one decision. Returns ``(action, visit_counts)``.

    ``action`` is ``None`` when the decision state has no legal actions. The env
    is left at the decision state; ``visit_counts`` (action → visits) is the
    MCTS policy target. Actions must be hashable.

    Thin consumer of :func:`search_iter` (single source of truth); draining the
    generator to its ``done`` event is exactly the batch search.
    """
    for ev in search_iter(env, policy_value, cfg, rng):
        if ev.get("done"):
            return ev["action"], ev["visits"]
    return None, {}


def search_iter(
    env: SearchEnv,
    policy_value: PolicyValueFn,
    cfg: MctsConfig,
    rng: random.Random | None = None,
):
    """Stepwise generator form of :func:`run_search` — one decision, one
    simulation per yield.

    Yields one event dict per simulation, then a final ``done`` event:
      - per simulation: ``{"sim": i, "n": N, "visits": {action: N}}`` (``i`` is
        1-based; ``visits`` is the live root visit distribution after that sim).
      - final:          ``{"done": True, "action": action|None, "visits": {...}}``.

    Between simulation yields the env is left at the leaf that simulation
    reached, so a caller can render it (the tree checkpoints are still live).
    The env is restored to the decision (root) state and the tree released
    exactly once — before the ``done`` yield on normal completion, or via the
    ``finally`` block if the generator is ``.close()``d early (GeneratorExit).
    """
    rng = rng or random.Random(cfg.seed)

    root = Node()
    root.state = env.checkpoint()                 # env is at the decision state
    released = False

    def _finish() -> None:
        # Restore the env to the decision state and free every checkpoint.
        nonlocal released
        released = True
        env.restore(root.state)
        _release_tree(root, env)

    try:
        _expand(root, env, policy_value, cfg)

        if not root.children:                     # terminal / no legal actions
            _finish()
            yield {"done": True, "action": None, "visits": {}}
            return

        gumbel = None
        if cfg.root_selection == "gumbel":
            gumbel = _GumbelRoot(root, cfg, rng)   # Gumbel supplies root exploration
        else:
            _add_dirichlet_noise(root, cfg, rng)

        minmax = _MinMaxStats()
        root_select = (lambda r: gumbel.select(r)) if gumbel is not None else None
        # The leaf value is the EXACT discounted return accumulated PER-EDGE inside
        # _simulate (Σ γ^k ΔΦ_k from the step rewards) + γ^depth·c·V_φ — no Φ(root)
        # subtraction and no leaf env.potential()/DRC; Φ enters only through ΔΦ.
        diag = getattr(env, "_search_diag", None)   # opt-in per-decision telemetry
        if diag is not None:
            diag.clear()
            diag.update(terminals=0, solved=0, caps=0, expands=0, leaf_vals=[])
            diag["phi_root"] = env.potential()   # telemetry only; env is at root here
        # A single-child root has no decision to make: every final-pick rule
        # returns that one action. ONE simulation is still spent, not zero — it
        # realizes the child, so an action that steps to StepResult.invalid still
        # empties the root under ``invalid_mode`` and this decision still reports
        # ``action=None`` (a dead end) exactly as a full-budget search would. The
        # remaining n_simulations-1 can only re-visit the same child and cannot
        # change the outcome. Under the default masking rule this fires on every
        # ``net_end`` state (net_end is the ONLY action allowed once the current net
        # is fully connected) — measured 95 of 494 decisions across 4 boards, each
        # previously spending the full budget on a forced move.
        n_sims = 1 if len(root.children) == 1 else cfg.n_simulations
        for i in range(n_sims):
            _simulate(root, env, policy_value, cfg, minmax, root_select, rng)
            yield {"sim": i + 1, "n": n_sims,
                   "visits": {a: c.N for a, c in root.children.items()}}

        if not root.children:                     # every root action proved invalid
            _finish()
            yield {"done": True, "action": None, "visits": {}}
            return

        if gumbel is not None:
            action = gumbel.final_action(root)
        elif cfg.algorithm == "alphazero":
            action = _az_final_action(root, cfg, rng)
        else:
            action = _pick_action(root, cfg, rng)
        visit_counts = {a: c.N for a, c in root.children.items()}

        if diag is not None:
            kids = sorted(root.children.values(), key=lambda c: c.N, reverse=True)
            # ``root_children``: the FULL visit-sorted root distribution. ``top``
            # is the 4-entry slice the log line prints; a consumer that needs more
            # (per-action-type aggregation, an animation panel) reads the full list
            # rather than the search deciding how much to keep.
            diag["root_children"] = [
                (list(c.action), c.N, round(c.Q, 3), round(c.prior, 3))
                for c in kids
            ]
            diag["top"] = diag["root_children"][:4]
            diag["chosen"] = list(action) if action is not None else None

        _finish()                                 # leave env at the decision state
        yield {"done": True, "action": action, "visits": visit_counts}
    finally:
        # Covers early .close() (GeneratorExit) before a done event fired.
        if not released:
            _finish()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _simulate(
    root: Node, env: SearchEnv, pv: PolicyValueFn, cfg: MctsConfig,
    minmax: _MinMaxStats, root_select=None, rng: random.Random | None = None,
) -> None:
    node = root
    depth = 0                  # edges from the root to `node` (for the max_depth cap)
    # return_bootstrap accumulates the EXACT discounted return along the descent:
    #   path_r = Σ_{k<depth} γ^k ΔΦ_k   (each ΔΦ = child.reward, the env's per-step
    #   reward with DRC),   gpow = γ^depth.  Absolute modes ignore both. This is the
    #   discounted return of the training MDP (γ = cfg.gamma), replacing the old
    #   single-factor γ^depth·(Φ(leaf)−Φ(root)) approximation.
    path_r = 0.0
    gpow = 1.0
    while True:
        if node.terminal:
            _backprop(node, node.terminal_value, minmax)
            return

        if node is root and root_select is not None:
            child = root_select(root)              # Gumbel / Sequential Halving
        elif root_select is not None:
            # Gumbel mode interiors: mctx's deterministic completed-Q rule.
            child = _gumbel_interior_select(node, cfg)
        elif cfg.algorithm == "alphazero":
            child = _az_select(node, cfg.c_puct, minmax, cfg)
        elif cfg.algorithm == "muzero":
            child = _muzero_select(node, cfg, minmax, rng)
        else:
            child = _puct_select(node, cfg.c_puct, minmax, cfg.value_completion, cfg)

        if not child.realized:
            # Realize the leaf: restore to its (realized) parent, step the edge.
            env.restore(node.state)
            res = env.step(child.action)
            child.state = env.checkpoint()
            child.reward = res.reward
            # Discounted return / γ^depth AT the leaf (this edge included).
            leaf_path_r = path_r + gpow * child.reward
            leaf_gpow = gpow * cfg.gamma
            at_cap = cfg.max_depth is not None and depth + 1 >= cfg.max_depth
            diag = getattr(env, "_search_diag", None)   # opt-in leaf-type telemetry
            if res.done:
                child.terminal = True
                child.terminal_value = _terminal_value(env, cfg, leaf_path_r)
                if diag is not None:
                    diag["terminals"] += 1
                    diag["leaf_vals"].append(child.terminal_value)
                    sv = getattr(env, "solved", None)
                    if sv is not None and sv():          # completion (unrouted==0) reached
                        diag["solved"] += 1
                _backprop(child, child.terminal_value, minmax)
            elif res.invalid:
                # Meaningless action (failed / no-op → state unchanged): dead-end.
                # Never expand it (the child re-offers the same actions, so
                # expanding would just loop and waste simulations). Two policies:
                if cfg.invalid_mode in ("pop", "drop"):
                    # Drop the child entirely — it can't be re-selected or returned
                    # and its value is never backed up, so it neither pollutes the
                    # parent's Q nor stretches the min-max floor. Release the
                    # checkpoint (never reaped by _release_tree once the child leaves
                    # the tree). Re-select among the survivors in this same
                    # simulation; if that empties the parent it becomes a dead-end
                    # (valued at the return TO the node).
                    env.release(child.state)
                    child.state = None
                    if diag is not None and node is root:
                        diag.setdefault("popped", {})
                        _t = int(child.action[0])
                        diag["popped"][_t] = diag["popped"].get(_t, 0) + 1
                    node.children.pop(child.action, None)
                    if not node.children:
                        env.restore(node.state)
                        node.terminal = True
                        node.terminal_value = _terminal_value(env, cfg, path_r)
                        _backprop(node, node.terminal_value, minmax)
                        return
                    if cfg.invalid_mode == "drop":
                        return          # simulation spent; nothing backed up
                    continue
                # "penalize" (once): back-propagate the penalty a SINGLE time so the
                # dead-end pessimism reaches the ancestors' Q, then REMOVE the child
                # so it can never be re-selected. Without removal a high-prior invalid
                # is re-picked as √N_parent grows (its PUCT u-term stays competitive),
                # re-spending budget and re-polluting the min-max floor on every
                # visit. Removal reuses the "pop" path, so re-selection is impossible
                # under every selector; unlike "pop" this consumes THIS simulation
                # (one backprop = one budget), so an all-invalid node is penalized one
                # child per simulation before it collapses to a dead-end.
                #
                # The value sits below the return-bootstrap dead-end base
                # (path_return to this node — the env is at the child = node's board
                # since the step was a no-op), so raw leaf_path_r would double-count
                # the failed edge's ΔΦ (0 here anyway) but path_r is the correct base.
                base = _terminal_value(env, cfg, path_r)
                child.terminal = True
                child.terminal_value = base - cfg.invalid_penalty
                _backprop(child, child.terminal_value, minmax)   # penalty → ancestors, once
                env.release(child.state)
                child.state = None
                node.children.pop(child.action, None)
                if not node.children:            # dead-end for FUTURE sims (this one
                    node.terminal = True         # already back-propagated the penalty)
                    node.terminal_value = base
            elif at_cap:
                # Depth cap: bootstrap-evaluate but do NOT expand; mark it a
                # value-leaf (re-backs its stored value if selected again).
                child.terminal = True
                child.terminal_value = _expand(
                    child, env, pv, cfg, leaf_path_r, leaf_gpow, expand=False,
                )
                if diag is not None:
                    diag["caps"] += 1
                    diag["leaf_vals"].append(child.terminal_value)
                _backprop(child, child.terminal_value, minmax)
            else:
                value = _expand(child, env, pv, cfg, leaf_path_r, leaf_gpow)
                if diag is not None:
                    diag["expands"] += 1
                    diag["leaf_vals"].append(value)
                _backprop(child, value, minmax)
            return

        # Descend into the realized child: accumulate its edge's discounted ΔΦ.
        path_r += gpow * child.reward
        gpow *= cfg.gamma
        depth += 1
        node = child


def _terminal_value(env: SearchEnv, cfg: MctsConfig, path_return: float) -> float:
    """Value backed up at a terminal leaf: the DISCOUNTED PATH RETURN ``path_return``
    = Σ γ^k ΔΦ_k, accumulated PER-EDGE from the root (each ΔΦ_k is the exact step
    reward the env produced, DRC included). The bootstrap is 0 — the episode is over,
    so there IS no remaining return — and the completion ΔΦ is simply the final edge's
    reward, so the completion bonus is captured with NO separate leaf Φ read. Exact
    discounted return (γ = cfg.gamma).

    Keeping this at zero is only consistent because the critic is calibrated to vanish
    at completion (``critic_offset`` = the V a finished board reads). Without that
    anchor a constant critic offset rides on every NON-terminal leaf and on no terminal
    one, and min-max cannot cancel what only one side carries."""
    return path_return


def _bootstrap_from(
    env: SearchEnv, cfg: MctsConfig, path_return: float, gpow: float,
    value: float | None,
) -> float:
    """Non-terminal leaf value: the discounted path return so far (``path_return``
    = Σ γ^k ΔΦ_k, per-edge) plus the discounted critic bootstrap γ^depth·c·V_φ
    (``gpow`` = γ^depth). Φ enters ONLY through the per-step ΔΦ edge rewards — no
    env.potential() read here."""
    boot = (cfg.critic_lambda * cfg.critic_scale * (float(value) - cfg.critic_offset)
            if value is not None else 0.0)
    return path_return + gpow * boot


def _expand(
    node: Node, env: SearchEnv, pv: PolicyValueFn, cfg: MctsConfig,
    path_return: float = 0.0, gpow: float = 1.0, expand: bool = True,
) -> float:
    """Evaluate ``node`` (env is positioned at it); return its value.

    With ``expand=True`` (default) also attaches children with priors (the
    normal case). With ``expand=False`` only the value is computed — used at the
    ``max_depth`` cap, where the node becomes a value-leaf with no children.

    A node with no legal actions becomes terminal. ``path_return``/``gpow`` = the
    discounted return + γ^depth accumulated to this leaf.
    """
    legal = list(env.legal_actions())
    if not legal:
        node.terminal = True
        node.terminal_value = _terminal_value(env, cfg, path_return)
        return node.terminal_value

    priors, value = pv(env.observe(), legal)
    if expand:
        node.expand(legal, priors)
    value_only = _bootstrap_from(env, cfg, path_return, gpow, value)
    # raw_value doubles as mctx's ``raw_values[node]`` — the node's own value in
    # the same units as its children's backed-up Q (consumed by completed-Q).
    node.raw_value = value_only
    return value_only


def _active_children(node: Node, cfg: MctsConfig) -> dict:
    """Children selectable at ``node`` under progressive widening: the top
    ``k = ceil(pw_base · N^pw_alpha)`` by prior (highest first), where N = node
    visits. Grows with N, so lower-prior children are DELAYED, not pruned. Returns
    the full child dict when disabled (``pw_alpha <= 0``) or when k covers all."""
    kids = node.children
    if cfg.pw_alpha <= 0.0 or len(kids) <= 1:
        return kids
    k = max(1, math.ceil(cfg.pw_base * (max(1, node.N) ** cfg.pw_alpha)))
    if k >= len(kids):
        return kids
    top = sorted(kids.items(), key=lambda kv: kv[1].prior, reverse=True)[:k]
    return dict(top)


def _puct_select(node: Node, c_puct: float, minmax: _MinMaxStats,
                 complete: bool = False, cfg: MctsConfig | None = None) -> Node:
    """Generic PUCT interior/root select — the fallback for hand-built
    ("custom") configs not tied to any one paper. AlphaZero and MuZero each
    have their OWN copy instead (``_az_select`` / ``_muzero_select`` below) —
    forked, not shared, because the references disagree with each other (and
    with this generic version) on tie-break placement and the PUCT formula
    itself (see :class:`~methods._shared.mcts.algorithms.AlphaZero` /
    :class:`~methods._shared.mcts.algorithms.MuZero`)."""
    sqrt_parent = math.sqrt(max(1, node.N))
    # Value-completion: an unvisited child inherits the PARENT's normalized value
    # instead of 0 (mctx complete_qvalues). 0 otherwise (legacy: exploration term
    # alone decides for unvisited children).
    unvisited_q = minmax.norm(node.Q) if complete else 0.0
    children = _active_children(node, cfg) if cfg is not None else node.children
    best, best_score = None, -math.inf
    for child in children.values():
        q = minmax.norm(child.Q) if child.N > 0 else unvisited_q
        u = c_puct * child.prior * sqrt_parent / (1 + child.N)
        score = q + u
        if score > best_score:
            best_score, best = score, child
    return best


def _az_select(node: Node, c_puct: float, minmax: _MinMaxStats,
               cfg: MctsConfig | None = None) -> Node:
    """AlphaZero interior/root PUCT — ``~/alpha-zero-general`` ``MCTS.search()``.

    One rule at every node (root == interior; az-general has no separate root
    treatment). An unvisited child contributes Q=0 — the exploration term alone
    decides it, as az-general's ``else: u = cpuct*P*sqrt(Ns+EPS)`` branch. Ties
    keep the FIRST child scanned: az-general's ``if u > cur_best`` has no
    explicit tie-break either, so this matches it exactly (contrast
    ``_az_final_action`` below, where az-general DOES randomize).

    Forked from ``_puct_select`` rather than calling it with ``complete=False``
    so a future AlphaZero-only change here can never silently reach MuZero.
    The two bodies compute the identical score today; that is expected to
    change as each gets its own fidelity passes.

    NB: az-general normalizes U by ``sqrt(Ns[s])`` (the node's own raw visit
    count; its values are bounded [-1, 1]). This core instead min-max-
    normalizes Q (mandatory for CADAgent's unbounded Φ — see algorithms.py
    footnote 2) and floors the sqrt at 1 for numerical stability at N=0. Both
    are a single positive multiplicative/additive term shared by every sibling
    at the comparison, so the argmax ordering — the only thing that drives the
    search — is unaffected by the difference.
    """
    sqrt_parent = math.sqrt(max(1, node.N))
    children = _active_children(node, cfg) if cfg is not None else node.children
    best, best_score = None, -math.inf
    for child in children.values():
        q = minmax.norm(child.Q) if child.N > 0 else 0.0
        u = c_puct * child.prior * sqrt_parent / (1 + child.N)
        score = q + u
        if score > best_score:
            best_score, best = score, child
    return best


def _muzero_ucb_score(node: Node, child: Node, cfg: MctsConfig,
                      minmax: _MinMaxStats) -> float:
    """muzero-general ``ucb_score``: a pb_c that GROWS with the parent's raw
    (un-floored) visit count, not a flat ``c_puct``.

    NB: ``node.N`` is used RAW, with no epsilon/floor guard — unlike
    ``_az_select``'s ``sqrt(max(1, node.N))``. This is deliberate: at
    ``node.N == 0`` (a node's very first pick right after expansion),
    ``sqrt(0) == 0`` zeroes ``pb_c`` for every child, exactly matching
    muzero-general's literal behavior (see ``_muzero_select`` for what that
    implies). Flooring here would silently diverge from the reference.
    """
    log_term = math.log((node.N + cfg.pb_c_base + 1) / cfg.pb_c_base) + cfg.pb_c_init
    pb_c = log_term * math.sqrt(node.N) / (child.N + 1)
    prior_score = pb_c * child.prior
    value_score = minmax.norm(child.Q) if child.N > 0 else 0.0
    return prior_score + value_score


def _muzero_select(node: Node, cfg: MctsConfig, minmax: _MinMaxStats,
                   rng: random.Random) -> Node:
    """MuZero interior/root select — ``~/muzero-general`` ``select_child()``.

    Two divergences from the generic ``_puct_select`` this replaces for
    MuZero:

      1. pb_c-scaled exploration (``_muzero_ucb_score``) instead of a flat
         ``c_puct`` — ``cfg.c_puct`` is UNUSED here.
      2. Ties are broken UNIFORMLY AT RANDOM among the max-scoring children
         (``np.random.choice`` over an exact-equality filter, mirroring
         muzero-general's own recompute-then-filter idiom) — the mirror image
         of az-general's placement, which randomizes the FINAL pick instead
         and is tie-break-free here (``_az_select``).

    Consequence of (1) using a RAW (un-floored) ``node.N``: at a node's very
    first pick (``node.N == 0``) every child scores 0 regardless of prior, so
    this is a uniform-random pick among ALL children, not just a tie-broken
    subset — muzero-general's literal behavior (no epsilon guard on
    ``sqrt(parent.visit_count)``, unlike az-general's ``sqrt(Ns+EPS)``).
    """
    children = _active_children(node, cfg)
    scores = {a: _muzero_ucb_score(node, c, cfg, minmax)
              for a, c in children.items()}
    best_score = max(scores.values())
    best = [a for a, s in scores.items() if s == best_score]
    action = rng.choice(best) if len(best) > 1 else best[0]
    return children[action]


def _backprop(node: Node, value: float, minmax: _MinMaxStats) -> None:
    while node is not None:
        node.N += 1
        node.W += value
        minmax.update(node.Q)
        node = node.parent


# ---------------------------------------------------------------------------
# Root noise / action selection / reaping
# ---------------------------------------------------------------------------

def _considered_visits(m: int, n_simulations: int) -> list[int]:
    """Sequential Halving visit schedule (port of mctx ``seq_halving``).

    Returns a length-``n_simulations`` list whose entry ``s`` is the visit count a
    root child must currently have to be *eligible* on simulation ``s``. Visiting
    the best-scoring eligible child each step implements Sequential Halving: all
    ``m`` actions get one visit, then the survivors get a second, etc., so the
    budget concentrates on policy-likely actions.
    """
    if m <= 1:
        return list(range(n_simulations))
    log2m = int(math.ceil(math.log2(m)))
    seq: list[int] = []
    visits = [0] * m
    num_considered = m
    while len(seq) < n_simulations:
        num_extra = max(1, int(n_simulations / (log2m * num_considered)))
        for _ in range(num_extra):
            seq.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return seq[:n_simulations]


def _gumbel(rng: random.Random) -> float:
    """One sample from the standard Gumbel(0, 1) distribution."""
    # Inverse-CDF: -log(-log(U)), U ~ Uniform(0,1).
    u = rng.random()
    while u <= 0.0:                                 # guard log(0)
        u = rng.random()
    return -math.log(-math.log(u))


def _mixed_value(children: dict[Any, Node], raw_value: float | None) -> float:
    """mctx ``_compute_mixed_value``: interpolate the node's own value estimate
    with the prior-weighted mean of its VISITED children's Q,
    ``(raw + ΣN·weighted_q) / (ΣN + 1)``. Falls back to ``weighted_q`` alone when
    the node carries no own value (horizon interiors), and to ``raw``/0 when
    nothing was visited yet."""
    visited = [c for c in children.values() if c.N > 0]
    if not visited:
        return raw_value if raw_value is not None else 0.0
    sum_n = sum(c.N for c in children.values())
    sum_probs = sum(max(c.prior, 1e-38) for c in visited)
    weighted_q = sum(max(c.prior, 1e-38) * c.Q for c in visited) / sum_probs
    if raw_value is None:
        return weighted_q
    return (raw_value + sum_n * weighted_q) / (sum_n + 1)


def _completed_q(children: dict[Any, Node], raw_value: float | None,
                 cfg: MctsConfig) -> dict[Any, float]:
    """mctx ``qtransform_completed_by_mix_value``: unvisited children are
    completed with the MIXED value, the vector is rescaled to [0, 1] over the
    completed values, then scaled by ``value_scale·(maxvisit_init + max_visit)``
    so Q outweighs the Gumbel/prior only as visits accumulate."""
    if not children:
        return {}
    mixed = _mixed_value(children, raw_value)
    completed = {a: (c.Q if c.N > 0 else mixed) for a, c in children.items()}
    lo = min(completed.values())
    hi = max(completed.values())
    denom = max(hi - lo, 1e-8)
    max_visit = max(c.N for c in children.values())
    scale = cfg.gumbel_value_scale * (cfg.gumbel_maxvisit_init + max_visit)
    return {a: scale * (v - lo) / denom for a, v in completed.items()}


def _gumbel_interior_select(node: Node, cfg: MctsConfig) -> Node:
    """mctx ``gumbel_muzero_interior_action_selection``: deterministic
    ``argmax( softmax(logits + q̂) − N/(1 + ΣN) )`` — repeated argmaxes drive the
    visitation frequencies toward the improved policy softmax(logits + q̂).

    Progressive widening is NOT applied here: this rule already drives visits
    toward the full improved policy over ALL children, and Sequential Halving at
    the Gumbel ROOT already bounds the search width — restricting the interior
    candidate set by prior would distort the improved-policy visitation target."""
    children = node.children
    qn = _completed_q(children, node.raw_value, cfg)
    logits = {a: math.log(max(c.prior, 1e-12)) + qn[a] for a, c in children.items()}
    mx = max(logits.values())
    exps = {a: math.exp(v - mx) for a, v in logits.items()}
    z = sum(exps.values()) or 1.0
    sum_n = sum(c.N for c in children.values())
    best, best_score = None, -math.inf
    for a, c in children.items():
        score = exps[a] / z - c.N / (1.0 + sum_n)
        if score > best_score:
            best_score, best = score, c
    return best


class _GumbelRoot:
    """Gumbel MuZero root selection + final pick (mctx-exact, pointer tree).

    Built once per ``run_search`` over the root's children: samples one Gumbel per
    child action, precomputes the Sequential Halving schedule, and exposes
    ``select`` (which child to descend this simulation) and ``final_action`` (the
    action to return — argmax of ``g + logit + σ(q̂)`` among the MOST-VISITED
    root children only, mctx's ``considered_visit = max(visit_counts)``).
    Children are addressed by action key, so a child popped mid-search (horizon
    invalid) simply drops out of consideration.
    """

    __slots__ = ("g", "logit", "seq", "cfg")

    def __init__(self, root: Node, cfg: MctsConfig, rng: random.Random) -> None:
        self.cfg = cfg
        actions = list(root.children.keys())
        # One Gumbel(0,1) sample per action (fixed for the whole search).
        self.g = {a: cfg.gumbel_scale * _gumbel(rng) for a in actions}
        self.logit = {a: math.log(max(root.children[a].prior, 1e-12))
                      for a in actions}
        m = min(cfg.gumbel_max_considered, len(actions))
        self.seq = _considered_visits(max(1, m), max(1, cfg.n_simulations))

    def select(self, root: Node) -> Node:
        children = root.children
        sim_index = sum(c.N for c in children.values())
        cv = self.seq[min(sim_index, len(self.seq) - 1)]
        qn = _completed_q(children, root.raw_value, self.cfg)
        best, best_score = None, -math.inf
        for a, c in children.items():
            if c.N != cv:                          # not eligible this round
                continue
            score = self.g.get(a, 0.0) + self.logit.get(a, 0.0) + qn[a]
            if score > best_score:
                best_score, best = score, c
        if best is None:        # count desync (e.g. a popped child) → least-visited
            best = min(children.values(), key=lambda c: c.N)
        return best

    def final_action(self, root: Node) -> Any:
        children = root.children
        qn = _completed_q(children, root.raw_value, self.cfg)
        # mctx: considered_visit = max(visit_counts) — only the most-visited
        # (Sequential-Halving finalist) actions compete for the final pick; the
        # policy-improvement guarantee is stated for this restricted argmax.
        max_n = max(c.N for c in children.values())
        best, best_score = None, -math.inf
        for a, c in children.items():
            if c.N != max_n:
                continue
            score = self.g.get(a, 0.0) + self.logit.get(a, 0.0) + qn[a]
            if score > best_score:
                best_score, best = score, a
        return best


def _add_dirichlet_noise(root: Node, cfg: MctsConfig, rng: random.Random) -> None:
    if cfg.dirichlet_alpha <= 0 or not root.children:
        return
    children = list(root.children.values())
    samples = [rng.gammavariate(cfg.dirichlet_alpha, 1.0) for _ in children]
    total = sum(samples) or 1.0
    frac = cfg.dirichlet_frac
    for child, s in zip(children, samples):
        child.prior = (1 - frac) * child.prior + frac * (s / total)


def _pick_action(root: Node, cfg: MctsConfig, rng: random.Random) -> Any:
    """Generic visits-based final pick — used by MuZero and "custom" configs.
    Keeps the FIRST max-N action (matches muzero-general's ``select_action``,
    plain ``numpy.argmax``). AlphaZero has its OWN copy, ``_az_final_action``
    below, because az-general randomizes this tie-break instead — see
    :class:`~methods._shared.mcts.algorithms.AlphaZero`."""
    children = list(root.children.items())
    return max(children, key=lambda kv: kv[1].N)[0]


def _az_final_action(root: Node, cfg: MctsConfig, rng: random.Random) -> Any:
    """AlphaZero final pick — ``~/alpha-zero-general`` ``MCTS.getActionProb()``.

    argmax visit count, ties broken UNIFORMLY AT RANDOM (``bestAs = argwhere(
    counts == max(counts)); np.random.choice(bestAs)``) — the generic
    ``_pick_action`` used elsewhere in this core keeps the first-scanned action
    on a tie instead, which is muzero-general's tie-break placement, not
    az-general's. The two references are NOT interchangeable here: az-general
    randomizes at the FINAL pick and is tie-break-free at interior selection;
    muzero-general is the mirror image, random at interior ``select_child``,
    first-argmax at the final pick."""
    children = list(root.children.items())
    max_n = max(c.N for _, c in children)
    ties = [a for a, c in children if c.N == max_n]
    return rng.choice(ties)


def _release_tree(root: Node, env: SearchEnv) -> None:
    """Free every node's L1 checkpoint (the env's live board is unaffected)."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.state is not None:
            env.release(node.state)
            node.state = None
        stack.extend(node.children.values())
