"""MCTS hyper-parameters."""

from __future__ import annotations

from dataclasses import dataclass

# Search-time default discount γ for the leaf value — a SEARCH REGULARIZER, a
# DIFFERENT role from the trainer's γ (which defines the learned return). Over the
# short lookahead (max_depth≈6) the training γ≈0.995 is near-inert (0.995^6≈0.97)
# and does NOT break the near-goal Q plateau — measured WORSE than γ=1 (chaotic
# net_end, no completion). A decisively stronger γ escapes it, so the CLIs default
# to this rather than inheriting the ckpt's training γ. The raw MctsConfig default
# below stays 1.0 (bit-exact legacy) — this is only the CLI/session default.
DEFAULT_SEARCH_GAMMA = 0.9


@dataclass
class MctsConfig:
    n_simulations: int = 64       # tree expansions per decision
    c_puct: float = 1.5           # exploration constant in PUCT (puct/az/muzero only)
    critic_scale: float = 1.0     # maps V_critic into raw Φ units (≈ reward-norm std)
    # Offset subtracted from the RAW critic output before scaling:
    #     boot = critic_scale * (V_tilde(s) - critic_offset)
    # The two together are the affine test-time calibration of a critic that was
    # trained on normalized rewards. Denormalization alone is exact -- the trainer
    # divides rewards by the running return std and never subtracts a mean
    # (RewardNormalizer: "mean is NOT subtracted"), so V_raw = sigma * V_tilde with
    # sigma = sqrt(reward_normalizer_state.var) from the checkpoint. What is NOT
    # exact is the constant: regressing the realized return G on V_raw with a free
    # intercept gives a slope of 0.98..1.42 (median 1.11, i.e. ~1 = denormalization
    # is right) but an intercept of -12.2..+7.3 that varies per board and flips
    # sign. Forcing the fit through the origin folds that intercept into the slope
    # and scatters it 0.47..1.71 instead.
    #
    # The physical anchor is the terminal state: a completed board has zero
    # remaining return, so a calibrated critic must read 0 there. Measured V_tilde
    # at completion is +0.920 (maytal) .. -1.007 (NiMH) -- stable within a board
    # (std 0.015..0.091 over 10 completions) but board-specific. Setting
    # critic_offset to that value makes the bootstrap vanish at completion, which
    # is what removes the spurious "do not finish" premium at its source rather
    # than papering over it at the terminal leaf.
    critic_offset: float = 0.0
    # Trust weight lambda on the critic bootstrap:
    #     boot = critic_lambda * critic_scale * (V_tilde(s) - critic_offset)
    # Bounded to [0, 1] — there is never a reason to believe the critic MORE than its
    # own calibration says. The calibration reports two things that decide it:
    #   * whether any rollout COMPLETED — only then is the terminal anchor measured at
    #     a real terminal state rather than extrapolated from the lowest-unrouted state
    #     the policy reached (the proxy can sit as far out as unrouted=7);
    #   * ``corr``, how well V ranks states at all.
    # The tiers this encodes:
    #   no completion + corr below the ranking floor -> 0    (V says nothing usable)
    #   no completion + corr above it                -> small, ranking is real but the
    #                                                   absolute anchor is unverified
    #   completion observed                          -> up to 1, anchor is measured
    # Mathematically this just scales the bootstrap, but it is kept separate from
    # ``critic_scale`` so the calibration stays a measurement and the decision of how
    # far to trust it stays legible (and ``critic_lambda=0`` turns the critic off
    # without having to know the scale).
    # BOARD-LEVEL SCALAR, deliberately. The original formulation had λ as a
    # per-STATE function λ(s_d) — trust the critic less the further the leaf sits
    # from where the calibration verified it. That was implemented and swept
    # (λ = min(1, M/scale), M ∈ {0.5, 1, 2}) and REJECTED: every arm landed in
    # 0.809..0.834, indistinguishable. The reason is structural, not a tuning
    # failure — at a ΔΦ ≡ 0 node every sibling Q is the SAME affine function of
    # its own Ṽ, so ``_completed_q``'s (v−lo)/(hi−lo) cancels λ exactly, and those
    # nodes are 54..62% of all decisions. Any λ that is constant across siblings
    # is invisible there, per-state or not. What remains is the one thing λ still
    # does: scale the critic against ΔΦ at nodes that HAVE a ΔΦ — and a single
    # number per board is enough for that.
    critic_lambda: float = 1.0
    # Discount γ for the leaf value — the SAME kind of discount the trainer used on
    # its ΔΦ return (per-step reward telescopes onto Φ). The search backs up the
    # EXACT discounted return  Σ_k γ^k ΔΦ_k + γ^depth·c·V_φ, accumulated PER-EDGE from
    # the env's own step rewards (not a single-factor γ^depth on the total, nor the
    # undiscounted Φ(leaf)−Φ(root)). γ here is a SEARCH REGULARIZER — a different role
    # from the trainer's γ (which defines the learned return); it need NOT match it.
    # The CLIs default to ``DEFAULT_SEARCH_GAMMA`` (a strong 0.9), NOT the ckpt γ.
    # Why it matters: one ratsnest edge from done, a completion is reachable within
    # max_depth from almost EVERY action, so the UNDISCOUNTED return backs up
    # ~identically to productive AND wasteful moves (start_route / redundant via) —
    # the search can't tell them apart and dithers (Φ degrades while the value stays
    # flat). Discounting makes "complete in 1 step" strictly beat "complete in k
    # steps", restoring the gradient WITHOUT a step penalty. The training γ=0.995 is
    # nearly inert over a short lookahead (0.995^6≈0.97); use ≈0.9. At γ=1 the
    # per-edge sum telescopes back to Φ(leaf)−Φ(root).
    gamma: float = 1.0
    # max_depth → cap the simulation descent at this many edges from the root. A node
    # realized AT the cap is bootstrap-evaluated (its value backed up) but NOT
    # expanded — bounded lookahead leaning on the value estimate beyond the horizon.
    max_depth: int | None = None
    # Progressive widening at INTERIOR selection: the number of a node's children
    # that are SELECTABLE grows with its visit count, k(N) = ceil(pw_base · N^pw_alpha),
    # admitting children in PRIOR order (highest first). At a wide action space (dozens
    # of routing candidates) this concentrates a small simulation budget on the most
    # policy-likely branches so they get enough visits to matter, while LOWER-prior
    # children are DELAYED — not permanently pruned; they enter once N is large enough
    # (so a low-prior but valuable move like a completing make_line is not lost, only
    # deferred). ``pw_alpha == 0`` (default) disables it (all children always
    # selectable). Applies to the PUCT-family interior selectors (puct/az/muzero)
    # ONLY — NOT to Gumbel at all: the Gumbel ROOT is already width-bounded by
    # Sequential Halving over gumbel_max_considered, and the Gumbel INTERIOR rule
    # drives visits toward the improved policy over the FULL child set (a prior
    # cutoff there would distort that target). Typical: pw_alpha≈0.5, pw_base≈2.
    pw_alpha: float = 0.0
    pw_base: float = 1.0
    # Root Dirichlet exploration noise (AlphaZero). 0 disables. Ignored under Gumbel.
    dirichlet_alpha: float = 0.0
    dirichlet_frac: float = 0.25
    seed: int | None = None       # RNG seed for sampling / noise (None → unseeded)
    # MuZero-only interior PUCT knobs (~/muzero-general ucb_score); IGNORED unless
    # algorithm="muzero". pb_c GROWS logarithmically with the parent's visit count:
    # pb_c = (log((N_parent+pb_c_base+1)/pb_c_base) + pb_c_init) · √N_parent /
    # (N_child+1). Defaults are muzero-general's own.
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    # Root action selection strategy:
    #   "puct"   → AlphaZero PUCT at the root (pairs with dirichlet_* for exploration
    #              and argmax-visits for the final pick).
    #   "gumbel" → Gumbel MuZero (Danihelka et al. 2022), mctx-EXACT: sample one
    #              Gumbel per root child, drive root visits by Sequential Halving, and
    #              pick the final action by argmax(gumbel + logit + σ(q̂)) among the
    #              MOST-VISITED root children. Interior nodes use mctx's deterministic
    #              rule argmax(softmax(logit + q̂) − N/(1+ΣN)) — NOT PUCT — so
    #              c_puct/value_completion are ignored. q̂ is completed with the MIXED
    #              value, rescaled to [0,1], scaled by
    #              gumbel_value_scale·(gumbel_maxvisit_init + max_visit). Needs a
    #              DETERMINISTIC prior; ignores dirichlet_* (Gumbel supplies noise).
    root_selection: str = "puct"
    gumbel_max_considered: int = 16   # m: actions kept in the first halving round
    # Value-completion at INTERIOR PUCT nodes. When False an unvisited child
    # contributes Q=0 to PUCT; when True it is completed with its PARENT's value
    # (node.Q, min-max normalized). PUCT modes only — IGNORED under Gumbel.
    value_completion: bool = False
    gumbel_scale: float = 1.0         # multiplier on the sampled Gumbel noise
    gumbel_value_scale: float = 0.1   # σ(q̂) magnitude (mctx qtransform value_scale)
    gumbel_maxvisit_init: float = 50.0  # visit-count offset in the σ(q̂) scaling
    # How the _simulate path treats a child that steps to StepResult.invalid (a
    # failed / no-op action → board unchanged; e.g. a make_via at an occupied
    # spot, an un-routable start_route). Both variants REMOVE the child so it can
    # never be re-selected or returned (a stationary invalid re-offers the same
    # actions — expanding it would loop and waste simulations):
    #   "pop"      → drop it silently. Its value is never backed up, so it neither
    #                pollutes the parent's Q nor stretches the min-max floor, and
    #                THIS simulation re-selects among the surviving siblings (no
    #                budget spent). Right for a pure-rollout search where no
    #                learning target needs the invalid child scored.
    #   "drop"     → remove it and END this simulation with NO backup. The
    #                budget is spent; nothing is backed up, so the invalid child
    #                still pollutes neither the parent's Q nor the min-max floor.
    #                The difference from "pop" is accounting, not values: under
    #                "pop" a single simulation may spend 1, 2, 3+ env.step calls
    #                (it re-selects until it lands on a valid child), so n_simulations
    #                does NOT bound the engine work — measured 735 pops per rollout
    #                on top of 2688 simulations, i.e. 21.5% of all env.step calls,
    #                and it concentrates late in the episode where invalids explode.
    #                Under "drop", n_simulations IS the step budget, which is what
    #                makes the knob mean what it says and keeps a congested node from
    #                buying unbounded free enumeration. The cost: a node whose
    #                children are mostly invalid burns one simulation per invalid
    #                before it can search the valid ones.
    #   "penalize" → back-propagate a penalty ONCE (value = dead-end base −
    #                invalid_penalty) so the ancestors' Q feels the dead-end, THEN
    #                remove it. Consumes THIS simulation (one backprop = one
    #                budget), so an all-invalid node is penalized one child per
    #                simulation before collapsing to a dead-end — this keeps a
    #                high-prior invalid from re-inflating the parent's value.
    invalid_mode: str = "pop"
    # Drop candidates the env is GUARANTEED to refuse from the legal set instead of
    # discovering the refusal by stepping. Domain-agnostic here: the SearchEnv
    # decides what "guaranteed" means (RL branch: the engine's ``via_on_thru_pad``
    # predicate on make_via targets, a pure geometric test on cached thru-pad
    # geometry). Measured cost of NOT doing it: make_via is 52.7% of popped
    # children on d3b — 387 wasted env.step per rollout, 11.3% of all steps — and
    # plain/best-of-N never pays it because it never enumerates candidates, so it
    # lands as an MCTS-only tax in a wallclock-matched comparison.
    #
    # Default False: it narrows the legal set relative to what the POLICY can
    # sample (the model's pointer head is unchanged), so the search prior
    # renormalizes over fewer actions than the policy's own distribution. That is
    # a deliberate search-side restriction on provably-refused actions, not the
    # bug ``legal_actions`` guards against (admitting actions the model CANNOT
    # sample). Pushing it into the model's per-type pointer mask instead would
    # also change training and the plain rollout — a larger change, not this one.
    prefilter_refused: bool = False
    # Penalty subtracted below the dead-end base under invalid_mode="penalize"
    # (in raw Φ units). UNUSED under invalid_mode="pop".
    invalid_penalty: float = 0.1
    # Dispatch tag for the reference-faithful select/final-pick module PAIR the core
    # uses, set automatically by ``SearchAlgorithm.config()``. Each named algorithm
    # gets its OWN select + final-pick functions (search.py) because az-general and
    # muzero-general disagree on tie-break placement and the PUCT formula:
    #   "alphazero" -> `_az_select` / `_az_final_action`
    #   "muzero"    -> `_muzero_select` (pb_c-scaled) / `_pick_action`
    #   "custom"    -> `_puct_select` / `_pick_action` (generic flat-c_puct PUCT)
    #   "gumbel"    -> unused for dispatch (root_selection=="gumbel" selects the path)
    algorithm: str = "custom"
