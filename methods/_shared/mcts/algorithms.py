"""Named algorithm profiles — one class per published MCTS *module set*.

The core (``run_search`` + :class:`MctsConfig`) is mechanism-complete but
flag-driven: a raw ``MctsConfig`` can express any (possibly *incoherent*) mix of
modules. Each class here fixes the coherent SET of modules that defines one
reference implementation, so you pick an algorithm by NAME instead of hand-wiring
flags, and read one docstring to understand what that algorithm actually does:

    AlphaZero()     -> MctsConfig   # ~/alpha-zero-general  (MCTS.py)
    MuZero()        -> MctsConfig   # ~/muzero-general       (self_play.py)
    GumbelMuZero()  -> MctsConfig   # ~/mctx                 (policies.py, action_selection.py)

Five module axes distinguish them (▼ = differs across the three); each class sets
the axis DEFAULTS that define it, and every axis stays per-instance tunable::

    axis          AlphaZero              MuZero                    GumbelMuZero
    ──────────────────────────────────────────────────────────────────────────────
  ▼ root select   PUCT                   pb_c-PUCT⁵                Gumbel + Seq-Halving
  ▼ root noise    none (this repo)¹      Dirichlet                 Gumbel (replaces noise)
    interior      PUCT, ties→first³      pb_c-PUCT, ties→random⁴   PUCT (+ completed Q)
  ▼ unvisited Q   0                      0                         parent value (completion)
    Q-normalize   —²                     min-max                   min-max
    backup        running mean           running mean⁶             running mean
    leaf value    NN value head          critic (return-to-go)     critic
  ▼ final pick    argmax N, ties→random³ argmax N, ties→first⁴     argmax(g + logit + σ(q̂))
                  / ∝N^(1/τ)             / ∝N^(1/τ)

  ¹ alpha-zero-general's repo omits root Dirichlet (root == interior). *Canonical*
    AlphaZero (Silver et al. 2017) DOES add it — enable with
    ``AlphaZero(dirichlet_alpha=0.3)``.
  ² az-general uses bounded win/loss values ([-1, 1]) so needs no Q-normalization;
    our Φ is UNBOUNDED, so this core ALWAYS min-max-normalizes Q (the MuZero
    adaptation) regardless of profile — a necessary divergence from az-general.
  ³ az-general (``MCTS.py``): interior selection (``search``) has no explicit
    tie-break (``if u > cur_best`` keeps the first found); the FINAL pick
    (``getActionProb``, temp=0) breaks ties with ``np.random.choice(bestAs)`` —
    found + ported (``_az_select`` / ``_az_final_action``).
  ⁴ muzero-general (``self_play.py``) is the MIRROR IMAGE of az-general: interior
    ``select_child`` breaks UCB ties with ``np.random.choice``, but the final
    ``select_action`` (temp=0) uses plain ``numpy.argmax`` (first tie) — found
    + ported (``_muzero_select`` for the interior random tie-break; the final
    pick already matched, via the shared ``_pick_action``, so MuZero keeps it).
  ⁵ pb_c GROWS logarithmically with the parent's visit count instead of being a
    flat constant (az-general/mctx's ``c_puct``):
    ``pb_c = (log((N_parent+pb_c_base+1)/pb_c_base) + pb_c_init) · √N_parent /
    (N_child+1)``. Ported EXACTLY, including the edge case this uncovered: at a
    freshly-expanded node's very FIRST pick (``N_parent == 0``, no epsilon guard
    in muzero-general unlike az-general's ``sqrt(Ns+EPS)``), ``√0 == 0`` zeroes
    pb_c for every child — the prior plays NO role and the pick is uniformly
    random among ALL children, not just the tied top scorers. Self-corrects
    after that first visit (``N_parent`` ≥ 1). ``c_puct`` is unused for MuZero.
  ⁶ muzero-general's single-player backup additionally folds
    ``reward + discount·value`` into what gets normalized and propagated — a
    return construction specific to combining ITS OWN learned model's rolled-
    out per-step rewards. This core runs the real PNS engine (no learned
    model), and CADAgent's leaf value (per-edge discounted return + calibrated
    critic bootstrap, see below) already targets the absolute value that
    composition would reconstruct, so the reward/discount fold-in is
    intentionally NOT ported — only the normalization target (``child.Q``, the
    running mean) is.

Module isolation: each reference's select/final-pick functions are COPIED
under a name for that reference rather than kept shared — so a fidelity fix
to one algorithm can never silently perturb another. Gumbel, AlphaZero, and
MuZero are each isolated this way (``_gumbel_interior_select``/``_GumbelRoot``,
``_az_select``/``_az_final_action``, ``_muzero_select`` — see each profile's
docstring for the exact PUCT/tie-break rule it uses). "custom" hand-built
configs fall back to the generic ``_puct_select`` / ``_pick_action`` pair
(flat ``c_puct``, first-scanned ties both ends) as a safe default not tied
to any one paper. Dispatch key: ``MctsConfig.algorithm``,
stamped by ``config()``.

Orthogonal axis — the LEAF VALUE — is a CADAgent EXTENSION, not an AZ/MuZero/Gumbel
distinction, and it is FIXED (not a tunable; the former ``value_mode`` menu was
removed). The references all feed the raw critic value as the leaf value;
CADAgent's potential-based reward lets us do better: the leaf value is the
per-edge discounted return along the descent plus the calibrated critic
bootstrap, Σ γ^k·ΔΦ_k + γ^depth·λ·scale·(V_critic − offset) ≈ E[Φ(terminal)],
the value the absolute backup expects — using the critic alone (the references'
choice) mixes relative/absolute values under our global min-max and is a known
bug in this env. See ``search._bootstrap_from`` for the exact formula and
:class:`MctsConfig` (``critic_scale``/``critic_offset``/``critic_lambda``) for
the calibration knobs.

Usage::

    from methods._shared.mcts import GumbelMuZero, run_search
    cfg = GumbelMuZero(n_simulations=100, seed=0).config()
    action, visits = run_search(search_env, policy_value, cfg)

    # or by name (e.g. from a CLI flag):
    from methods._shared.mcts import make_algorithm
    cfg = make_algorithm("muzero", n_simulations=64).config(max_depth=6)
"""

from __future__ import annotations

from dataclasses import dataclass

from methods._shared.mcts.config import MctsConfig


@dataclass
class SearchAlgorithm:
    """Base profile: the shared tunables + module-axis fields + ``config()``.

    Subclasses override only the module-axis DEFAULTS (``root_selection``,
    ``dirichlet_alpha``, ``value_completion``, …) that define their algorithm;
    every field is still per-instance tunable. This base is itself usable as a
    fully-manual profile (equivalent to writing an ``MctsConfig`` by hand) — the
    named subclasses are the point: they document and fix a *coherent* module set.

    ``config(**overrides)`` returns the :class:`MctsConfig` the core consumes;
    ``overrides`` forwards any remaining core fields not surfaced here.
    """

    # --- free tunables (budget / exploration / value) ---
    n_simulations: int = 64
    c_puct: float = 1.5
    critic_scale: float = 1.0
    gamma: float = 1.0                      # discount γ (see MctsConfig)
    max_depth: int | None = None
    pw_alpha: float = 0.0                   # progressive widening exponent (0 = off)
    pw_base: float = 1.0                    # progressive widening constant
    invalid_mode: str = "pop"              # invalid-child policy: "pop" | "drop" | "penalize"
    invalid_penalty: float = 0.1           # penalty under invalid_mode="penalize"
    seed: int | None = None

    # --- module-set axes (subclasses override the DEFAULTS to define the algo) ---
    root_selection: str = "puct"           # "puct" | "gumbel"
    dirichlet_alpha: float = 0.0           # root exploration noise (0 = off)
    dirichlet_frac: float = 0.25
    value_completion: bool = False         # interior unvisited Q → parent value (mctx)

    # --- Gumbel knobs (only meaningful when root_selection == "gumbel") ---
    gumbel_max_considered: int = 16
    gumbel_scale: float = 1.0
    gumbel_value_scale: float = 0.1
    gumbel_maxvisit_init: float = 50.0

    # --- MuZero knob (only meaningful when algorithm == "muzero") ---
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25

    # Dispatch tag (see MctsConfig.algorithm) — identifies which reference this
    # profile's config should route to at the core, independent of the tunables
    # above (which a profile instance may override, e.g. AlphaZero+Dirichlet).
    algorithm: str = "custom"

    def config(self, **overrides) -> MctsConfig:
        """Assemble the :class:`MctsConfig` for this profile.

        ``overrides`` wins over the profile's fields and can reach any
        ``MctsConfig`` field not exposed as a profile field.
        """
        params = dict(
            n_simulations=self.n_simulations,
            c_puct=self.c_puct,
            critic_scale=self.critic_scale,
            gamma=self.gamma,
            max_depth=self.max_depth,
            pw_alpha=self.pw_alpha,
            pw_base=self.pw_base,
            invalid_mode=self.invalid_mode,
            invalid_penalty=self.invalid_penalty,
            seed=self.seed,
            root_selection=self.root_selection,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_frac=self.dirichlet_frac,
            value_completion=self.value_completion,
            gumbel_max_considered=self.gumbel_max_considered,
            gumbel_scale=self.gumbel_scale,
            gumbel_value_scale=self.gumbel_value_scale,
            gumbel_maxvisit_init=self.gumbel_maxvisit_init,
            pb_c_base=self.pb_c_base,
            pb_c_init=self.pb_c_init,
            algorithm=self.algorithm,
        )
        params.update(overrides)
        return MctsConfig(**params)


@dataclass
class AlphaZero(SearchAlgorithm):
    """AlphaZero — one PUCT rule everywhere, temperature-visit final pick.

    Source: ``~/alpha-zero-general/MCTS.py`` (``search`` / ``getActionProb``).

    Module set:
      root / interior : a SINGLE PUCT rule at every node (the root is treated
                        like any interior node),
                        U = c_puct · P(a) · √N_parent / (1 + N_child);
                        an UNVISITED child contributes Q = 0 (exploration term
                        alone decides it).
      root noise      : NONE in this repo. Canonical AlphaZero adds root Dirichlet
                        — enable via ``AlphaZero(dirichlet_alpha=0.3)``.
      backup          : running mean of leaf values (Qsa ← incremental mean).
                        NB: az-general does NOT min-max-normalize Q (its values are
                        bounded [-1, 1]); this core always does, for unbounded Φ.
      leaf value      : the policy's value head. Here → per-edge discounted
                        return + calibrated critic bootstrap (module doc
                        explains why).
      final pick      : argmax visit counts (τ=0, ties broken UNIFORMLY AT
                        RANDOM — az-general's ``np.random.choice(bestAs)``, NOT
                        first-scanned) or sample ∝ N^(1/τ) (τ>0, unchanged).

    az-general's interior ``search`` has no explicit tie-break (``if u >
    cur_best`` keeps the first action found), but its ``getActionProb`` final
    pick DOES randomize ties. The two references disagree on WHERE they
    randomize — az-general only at the final pick, muzero-general only at
    interior selection — so this profile uses its own final-pick function,
    ``_az_final_action`` (search.py), forked from the shared PUCT/visits code
    rather than sharing it, keeping the two algorithms' tie-break rules
    independent. Dispatched via ``algorithm="alphazero"`` (below), not by any
    tunable flag.

    All base defaults already encode this profile (puct root, no Dirichlet, no
    completion); the class exists to NAME and document the module set.
    """

    algorithm: str = "alphazero"


@dataclass
class MuZero(SearchAlgorithm):
    """MuZero — pb_c-scaled PUCT + root Dirichlet + learned critic.

    Source: ``~/muzero-general/self_play.py`` (``MCTS.run`` / ``select_child`` /
    ``ucb_score`` / ``backpropagate`` / ``select_action``).

    Module set (differences from :class:`AlphaZero` marked ►):
      root select : ► pb_c-scaled PUCT (see below) with root Dirichlet
                    exploration noise ON (``add_exploration_noise``: prior ←
                    (1-frac)·prior + frac·noise — identical formula to
                    AlphaZero's, this repo's ``_add_dirichlet_noise`` is shared).
      interior    : ► pb_c-scaled PUCT, NOT a flat ``c_puct`` — ``pb_c`` grows
                    logarithmically with the parent's visit count:
                    ``pb_c = (log((N_parent+pb_c_base+1)/pb_c_base) + pb_c_init)
                    · √N_parent/(N_child+1)``. ► Ties broken UNIFORMLY AT RANDOM
                    (``np.random.choice`` among equal-max-UCB children) — the
                    mirror image of AlphaZero's tie-break placement (see
                    :class:`AlphaZero`). Value term is min-max-normalized Q (as
                    AlphaZero); unvisited child Q = 0.
      backup      : running mean (as AlphaZero) — muzero-general's own backup
                    additionally folds ``reward + discount·value`` into the
                    normalized/propagated value, a return construction specific
                    to ITS learned dynamics model; NOT ported here (see note).
      leaf value  : ► the learned critic's value head, trained to predict the
                    (reward-normalized) return-to-go — exactly what CADAgent's
                    ``critic_bootstrap`` telescopes onto Φ.
      final pick  : argmax visits (τ=0, ties → first-scanned — matches
                    muzero-general's ``select_action`` at temp=0, ``numpy.argmax``;
                    unlike AlphaZero, this needed NO fork) or ∝ N^(1/τ) (τ>0).

    Fidelity notes:
      1. Interior tie-breaks use a dedicated selector, ``_muzero_select``
         (search.py), rather than the generic ``_puct_select`` (which keeps
         the first-scanned child on a tie) — muzero-general breaks UCB/PUCT
         ties uniformly at random, requiring its own interior rule.
      2. pb_c-scaled PUCT is implemented via ``pb_c_base``/``pb_c_init``
         (defaults from muzero-general itself, universal across its game
         configs); ``c_puct`` is unused for this profile.
      3. Porting pb_c exactly (no epsilon guard on ``√N_parent``, matching
         muzero-general literally) surfaces an edge case: a freshly-expanded
         node's very FIRST pick has ``N_parent == 0`` ⇒ ``pb_c == 0`` for every
         child ⇒ the prior plays NO role and the pick is uniformly random
         among ALL children (not just tied top scorers) — self-corrects after
         that one visit, matching muzero-general's actual, literal behavior.

    In a REAL-simulator setting (we have the actual PNS engine, not MuZero's
    learned dynamics model), the learned-model distinction collapses; what
    remains — and is set here — is: root Dirichlet ON + pb_c-scaled PUCT with
    min-max Q-norm (always on in this core) + a return-to-go critic value.
    """

    algorithm: str = "muzero"
    dirichlet_alpha: float = 0.3           # muzero-general: root exploration noise ON
    dirichlet_frac: float = 0.25


@dataclass
class GumbelMuZero(SearchAlgorithm):
    """Gumbel MuZero — Gumbel + Sequential Halving at the root, completed Q inside.

    Source: ``~/mctx`` (``policies.gumbel_muzero_policy``,
    ``action_selection.gumbel_muzero_{root,interior}_action_selection``,
    ``qtransforms.qtransform_completed_by_mix_value``, ``seq_halving``).

    Module set (differences from :class:`AlphaZero` marked ►):
      root select : ► Gumbel MuZero. One Gumbel(0,1) g_a per root action; the
                    visit budget is driven by SEQUENTIAL HALVING (all m actions
                    get 1 visit, survivors a 2nd, …), and among the round's
                    eligible actions the pick maximizes g_a + logit_a + σ(q̂_a).
                    ► Replaces Dirichlet noise entirely.
      interior    : ► mctx-exact deterministic rule — argmax(softmax(logits + q̂)
                    − N/(1+ΣN)) with the SAME completed-Q transform as the root
                    (mixed-value completion, ``qtransform_completed_by_mix_value``).
                    No PUCT at interiors in this mode; ``c_puct`` and
                    ``value_completion`` are ignored.
      backup      : running mean (as AlphaZero/MuZero).
      leaf value  : critic (here → per-edge discounted return + calibrated
                    critic bootstrap, as every profile).
      final pick  : ► argmax of g_a + logit_a + σ(q̂_a) among the MOST-VISITED
                    root children only (mctx ``considered_visit = max(visits)``;
                    the policy-improvement guarantee is for this restricted
                    argmax) — NOT a plain visit-count argmax.

    Guarantees a policy improvement even at a tiny simulation budget — the regime
    this env runs in (n_sim≈64, ~100 root candidates). Needs a DETERMINISTIC prior
    (exact/factored, not sampling). σ(q̂) magnitude is tuned by
    ``gumbel_value_scale·(gumbel_maxvisit_init + max_visit)``.
    """

    algorithm: str = "gumbel"              # unused for dispatch (root_selection already
                                            # fully selects this path); stamped for symmetry
    root_selection: str = "gumbel"
    value_completion: bool = True          # moot under gumbel (interiors use mctx rule)
    dirichlet_alpha: float = 0.0           # Gumbel supplies exploration, not Dirichlet


# Name → profile class, for CLI flags / config strings.
ALGORITHMS: dict[str, type[SearchAlgorithm]] = {
    "alphazero": AlphaZero,
    "muzero": MuZero,
    "gumbel": GumbelMuZero,
}


def make_algorithm(name: str, **tunables) -> SearchAlgorithm:
    """Build a profile by name (``"alphazero"`` | ``"muzero"`` | ``"gumbel"``).

    ``tunables`` are forwarded to the profile constructor (e.g. ``n_simulations``,
    ``critic_scale``, ``seed``). Raises ``ValueError`` on an unknown name.
    """
    try:
        cls = ALGORITHMS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}"
        ) from None
    return cls(**tunables)
