"""RL binding for the branch-agnostic MCTS core (``methods._shared.mcts``).

``RLSearchEnv`` adapts a :class:`KiCadRLWrapper` to the ``SearchEnv`` protocol:
it bundles L1 (``PCBWorld.checkpoint`` — incremental restore) with L2
(``KiCadRLWrapper.snapshot_mcts_state``) and reports each step's ΔΦ as the edge
reward — node values are the per-edge discounted return, so Φ (and the DRC term
inside it) reaches the search ONLY through ``step``; there is no leaf Φ read
(``RLSearchEnv._warn_if_drc_invisible`` guards the reward rules that break this).
Prior/value providers: ``LogitPolicyValue`` (the sampler-exact logit-based prior —
the default for real planning), ``SamplingPolicyValue`` (frequency estimate from
policy samples), and ``BaselinePolicyValue`` (policy-free uniform baseline, used by
the tests). ``MemoizingPolicyValue`` wraps any of them with a cross-decision cache
keyed by (obs fingerprint, legal action set) — bit-identical to the wrapped provider,
it just skips the redundant policy forward on recurring board states.

The LLM branch will provide an analogous ``LLMSearchEnv`` (env + ``manager``
memory snapshot) — the MCTS core itself is unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Sequence

import numpy as np

from pcb_world.core.action_schema import (
    ACT_FINISH,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
)
from pcb_world.core.reward import RewardState
from methods._shared.mcts.protocols import NodeState, StepResult

logger = logging.getLogger(__name__)

# Action types that carry a pointer + routing mode vs. a bare action.
_POINTER_MODE_ACTS = (ACT_MAKE_LINE, ACT_MAKE_VIA)


class RLSearchEnv:
    """``SearchEnv`` over a single :class:`KiCadRLWrapper`.

    Actions are hashable ``(action_type, pointer_idx, routing_mode)`` tuples; the
    wrapper consumes the equivalent ``np.ndarray``.
    """

    def __init__(self, wrapper, run_drc_in_value: bool | None = None,
                 prefilter_refused: bool = False) -> None:
        self._w = wrapper
        self._env = wrapper.env          # PCBWorld (L1)
        self._obs = getattr(wrapper, "_last_obs", {})
        # Φ must match the training reward's potential. When the reward scores
        # DRC (drc_penalty>0), a Φ computed WITHOUT DRC desyncs from the per-step
        # reward and the telescoping Σr = Φ(terminal)−Φ(s_0) breaks (measured
        # ~4.0 off on a 3-net board). Default to the env's own DRC-activity flag
        # so Φ stays reward-consistent. Only :meth:`potential` reads this — it is
        # search telemetry (``phi_root``), not part of any node value.
        self._run_drc = (
            getattr(self._env, "_drc_active", False)
            if run_drc_in_value is None else run_drc_in_value
        )
        self._prefilter_refused = bool(prefilter_refused)
        self._warn_if_drc_invisible()
        # Carry the derived obs/pointer bundle in each node checkpoint instead
        # of re-deriving it per restore (see KiCadRLWrapper.snapshot_mcts_state).
        self._obs_cache = os.environ.get("MCTS_OBS_CACHE") != "0"
        self._verify_obs_cache = os.environ.get("MCTS_VERIFY_OBS_CACHE") == "1"

    def _warn_if_drc_invisible(self) -> None:
        """Warn when the reward scores DRC but the SEARCH cannot see it.

        Node values are ``Σ γ^k ΔΦ_k`` accumulated per-edge with no leaf Φ read
        (see ``mcts.search._bootstrap_from``), so DRC reaches the search only
        through the env's per-step ΔΦ — which ``PCBWorld.step`` computes with
        ``run_drc = (reward mode == "per_step") and _drc_active``. Under a
        ``terminal``-mode reward rule the first conjunct is False: Φ still
        penalizes DRC at the episode end, but every interior leaf is scored
        DRC-blind. That is a silent evaluation hole, not a speed knob.
        """
        if not getattr(self._env, "_drc_active", False):
            return                       # reward has no DRC term — consistent
        cfg = getattr(self._env, "_reward_config", None)
        mode = getattr(cfg, "mode", None)
        if mode != "per_step":
            logger.warning(
                "MCTS search is DRC-blind: reward rule %r scores DRC but its "
                "mode is %r, so per-step ΔΦ carries no DRC and the leaf value "
                "never sees it (DRC lands only on the terminal step). Use a "
                "per_step rule (e.g. --reward-rule drc_only_dense).",
                getattr(cfg, "name", "<unknown>"), mode,
            )

    # --- checkpoint: L1 (env) + L2 (wrapper) in lockstep ---
    def checkpoint(self) -> NodeState:
        return NodeState(l1=self._env.checkpoint(),
                         l2=self._w.snapshot_mcts_state(obs_cache=self._obs_cache))

    def restore(self, state: NodeState) -> None:
        # MCTS_FULL_RESTORE=1 forces the full restore (resyncWorld +
        # buildConnectivity, the validated oracle) instead of the incremental
        # diff path.
        #
        # The two are CLOSE but not yet identical. NODE::CanonicalizeOrder() (called
        # by both paths) fixed the round-trip defect that made them diverge badly —
        # a mavbridge rollout went from 355 tracks / 32 vias to 358 / 29 against the
        # full path's 359 / 28 — but that last gap is still open.
        # Treat the full path as the oracle.
        incremental = os.environ.get("MCTS_FULL_RESTORE") != "1"
        self._env.restore(state.l1, incremental=incremental)
        # The obs / candidate-pool bundle is board-derived (not path state), so
        # the checkpoint carries the one built when this node was reached and
        # restore_mcts_state reinstalls it. Re-deriving instead costs a full
        # _get_obs + engine connectivity query + candidate-pool rebuild on every
        # simulation. MCTS_OBS_CACHE=0 falls back to re-deriving.
        if self._obs_cache and self._w.restore_mcts_state(state.l2):
            self._obs = self._w._last_obs
            if self._verify_obs_cache:
                self._assert_obs_cache_matches()
            return
        self._w.restore_mcts_state(state.l2)
        self._obs = self._rederive_obs()

    def _rederive_obs(self) -> Any:
        """Rebuild the obs + pointer-decode tables from the live board."""
        raw_obs = self._env._get_obs()
        aug_obs = self._w._inject_aug(raw_obs)
        self._w._refresh_cache(aug_obs)
        return aug_obs

    def _assert_obs_cache_matches(self) -> None:
        """MCTS_VERIFY_OBS_CACHE=1: prove the cached bundle equals a fresh one.

        The cache is only sound while the engine restores the board bit-exactly.
        A congested-board restore that drifts would
        make the pointer tables describe a board that is no longer there —
        silently routing to the wrong coordinates. This re-derives and compares
        the two things an action decode actually reads.
        """
        cached_cands = list(self._w._cand_mm)
        cached_nets = list(self._w._sorted_net_codes)
        cached_obs = self._obs
        cached_fp = _obs_fingerprint(cached_obs)
        self._obs = self._rederive_obs()
        if (list(self._w._cand_mm) != cached_cands
                or list(self._w._sorted_net_codes) != cached_nets):
            raise AssertionError(
                "obs cache diverged from the restored board: candidate/net "
                "pointer tables differ — the engine restore is not bit-exact "
                "here, so the cache is unsound on this board."
            )
        if _obs_fingerprint(self._obs) != cached_fp:
            fresh, cached = self._obs, cached_obs
            keys = sorted(set(fresh) | set(cached))
            diff = [k for k in keys
                    if (k in fresh) != (k in cached)
                    or (k in fresh and k in cached
                        and _obs_fingerprint({k: fresh[k]}, skip=())
                        != _obs_fingerprint({k: cached[k]}, skip=()))]
            logger.warning(
                "obs cache: pointer tables match but the obs differs after "
                "restore on %s (action decode is unaffected; this only moves "
                "memo hits).", diff,
            )

    def release(self, state: NodeState) -> None:
        # Drop the cached obs bundle with the node so the tree bounds memory.
        if isinstance(state.l2, dict):
            state.l2.pop("obs_cache", None)
        self._env.release_checkpoint(state.l1)

    # --- stepping / observation ---
    def step(self, action: Any, committed: bool = False) -> StepResult:
        """Advance the env by ``action``.

        ``committed=True`` marks the action as part of the REAL episode (the one
        the caller executes after a decision) rather than a throwaway search
        simulation. Both run the env's per-step DRC identically — the search
        value depends on it (see :meth:`_warn_if_drc_invisible`).

        There is no transition cache. One was tried (memoize (state, action) ->
        child checkpoint, serve a hit by restoring it) and REMOVED: it never paid
        off on a dense board — 0344_mavbridge at n_sim 32 came out 11 unrouted /
        309 tracks against 2 / 358 with it off, and slower (561s vs 367s) — and the
        divergence survived every key we could think of (obs digest, exact copper,
        copper UUIDs, episode baselines, even the KIID generator position, all
        verified identical between fill and hit). Its 13% win on a small board was
        real but not worth serving silently-different transitions. If it is
        revisited, cache PYTHON data (StepResult + obs bundle + legal_actions) and
        replay real steps when the engine itself is needed, instead of restoring a
        stored board.
        """
        arr = np.asarray(action, dtype=np.int64)
        # Episode-level tracking (ratsnest early-stop / best-Φ board) counts only
        # COMMITTED steps; throwaway search steps must not pollute it. Harmless
        # when the feature is off (the env gates on its own flags too).
        self._env._track_best_active = bool(committed)
        obs, reward, terminated, truncated, info = self._w.step(arr)
        self._obs = obs
        # MCTS accumulates the potential delta ΔΦ (info["potential_diff"]), NOT
        # the raw env reward — so the path sum telescopes onto Φ exactly and
        # excludes step / invalid penalties (those are MCTS-side via invalid_*).
        # Falls back to the env reward if the key is absent.
        delta = float(info.get("potential_diff", reward))
        # A non-"valid_effective" class = the action failed or was a no-op (board
        # unchanged) → a meaningless dead-end the core should penalize + not expand.
        invalid = info.get("action_class", "valid_effective") != "valid_effective"
        return StepResult(reward=delta, done=bool(terminated or truncated),
                          info=info, invalid=bool(invalid))

    def observe(self) -> Any:
        return self._obs

    def potential(self) -> float:
        """Φ(board) — the §7 board-derived value (potential-based, exact for the
        default dense config)."""
        snap = self._env._engine.get_reward_snapshot(run_drc=self._run_drc)
        return float(self._env._reward.fn.potential(RewardState.from_snapshot(snap)))

    def solved(self) -> bool:
        """True when every net is connected (unconnected == 0). Consulted ONLY by
        the per-decision search telemetry (``env._search_diag`` SOLVED counter);
        it never affects the search result. Reads a DRC-free connectivity snapshot
        — the unconnected count needs no DRC pass, so this stays cheap."""
        snap = self._env._engine.get_reward_snapshot(run_drc=False)
        return RewardState.from_snapshot(snap).unconnected == 0

    # --- legal action enumeration ---
    # Tuples MUST match the policy's action encoding (action_schema / the
    # act_and_value assembly), where an UNUSED slot is -1 — otherwise a
    # policy-sampled action never matches a legal tuple and the prior collapses
    # to uniform (measured: MCTS then never picks `finish`, so nets never
    # complete). Canonical encoding:
    #   net_select  (0, net_idx,  -1)      make_line (3, cand_idx, mode)
    #   start_route (1, cand_idx, -1)      make_via  (4, cand_idx, mode)
    #   net_end     (2, -1,       -1)      finish    (5, -1,       mode)
    def legal_actions(self) -> Sequence[tuple[int, int, int]]:
        w = self._w
        at_mask = np.asarray(w.action_masks())
        modes = np.asarray(w.mode_mask())
        valid_modes = [m for m in range(modes.shape[0]) if modes[m]] or [0]
        # Mirror EVERY mask the policy hard-applies so the MCTS legal set equals
        # the policy's — a factored action is legal iff the model can sample it
        # with non-zero prob. The four masks and where each bites:
        #   action_masks   — action_type (all rows below already gate on it)
        #   net_valid_mask — the NET pointer (net_select only)
        #   mode_mask      — routing mode (make_line/make_via/finish)
        #   pointer mask   — the CAND pointer, shared by start_route/make_line/
        #     make_via; `start_route_pointer_indices` marks the exact
        #     (x, y, layer) of the active start origin, which the model forces to
        #     -inf on the cand-pointer head (net.py `_combined_ptr_logits`
        #     cand_block_idx). net_select uses the net pointer, so it is NOT
        #     subject to this one. Omitting it (the prior bug) re-admitted the
        #     zero-length same-point move: prior≈0 hides it under a trained
        #     policy, but a uniform prior (BaselinePolicyValue) would explore it.
        ptr_masked = {int(i) for i in np.asarray(w.start_route_pointer_indices())}
        # Candidates the engine will refuse for make_via no matter what
        # (``via_on_thru_pad``). Opt-in: see MctsConfig.prefilter_refused for why
        # this narrows the set relative to the policy's own samplable actions.
        # make_line cannot change layer (the dispatcher drops the candidate's
        # layer and fix_route is given expected_layer=current), so an off-layer
        # candidate is not a distinct target — it routes to the same (x, y) on
        # the current layer. Mirrors the policy's own make_line pointer block
        # (KiCadRLWrapper.offlayer_pointer_indices); make_via is unrestricted
        # because changing layer is its purpose.
        off_masked: set[int] = set()
        _off = getattr(w, "offlayer_pointer_indices", None)
        if _off is not None:
            off_masked = {int(i) for i in np.asarray(_off())}
        via_masked: set[int] = set()
        if self._prefilter_refused:
            fn = getattr(w, "via_blocked_pointer_indices", None)
            if fn is not None:
                via_masked = {int(i) for i in np.asarray(fn())}
        n_cand = len(w.cand_mm_list)
        out: list[tuple[int, int, int]] = []
        for at in range(at_mask.shape[0]):
            if not at_mask[at]:
                continue
            if at == ACT_NET_SELECT:
                for i, ok in enumerate(np.asarray(w.net_valid_mask())):
                    if ok:
                        out.append((at, i, -1))
            elif at == ACT_START_ROUTE:
                for i in range(n_cand):
                    if i not in ptr_masked:
                        out.append((at, i, -1))
            elif at in _POINTER_MODE_ACTS:        # make_line / make_via
                blocked = via_masked if at == ACT_MAKE_VIA else off_masked
                for i in range(n_cand):
                    if i in ptr_masked or i in blocked:
                        continue
                    for m in valid_modes:
                        out.append((at, i, m))
            elif at == ACT_FINISH:                # routing_mode only, no pointer
                for m in valid_modes:
                    out.append((at, -1, m))
            else:                                  # net_end / idle — no pointer or mode
                out.append((at, -1, -1))
        return out


class BaselinePolicyValue:
    """Policy-free baseline: uniform priors, no critic (value = Φ via the env)."""

    def __call__(self, obs, legal_actions):
        n = len(legal_actions)
        p = 1.0 / n if n else 0.0
        return {a: p for a in legal_actions}, None


class _SingleEnvPool:
    """Adapt one wrapper to the ``VecBackend.env_method`` API so the agent's
    ``act_from_pool`` (mask gathering + inference) runs over a single env."""

    def __init__(self, wrapper) -> None:
        self._w = wrapper

    def env_method(self, name, *args, indices=None, **kwargs):
        return [getattr(self._w, name)(*args, **kwargs)]


def _critic_value(agent, pool, obs) -> float | None:
    """V_critic(s) for the pool's current env state (action-independent).

    Gathers/tensorizes masks via the shared agent-glue helpers (same path as
    ``act_from_pool``, eval convention) and calls ``act_and_value`` (the model
    computes the value at the VAL token, before sampling). Returns ``None``
    when the model has no critic head.
    """
    from methods.rl_agent.policy.agent import (
        gather_mask_arrays, mask_arrays_to_tensors,
    )

    if not getattr(agent, "use_critic", False):
        return None
    masks, ptr_masks, mode_masks, nvm, off_masks = gather_mask_arrays(
        pool, [0],
        policy_net_select=getattr(agent, "policy_net_select", False),
    )
    mask_t, ptr_t, mode_t, nvm_kwargs = mask_arrays_to_tensors(
        masks, ptr_masks, mode_masks, nvm, getattr(agent, "device", None),
        off_masks=off_masks,
        mode_none_if_all_true=False,
    )
    _a, _lp, values = agent.act_and_value(
        [obs], action_masks=mask_t, pointer_masks=ptr_t, mode_mask=mode_t,
        deterministic=True, **nvm_kwargs,
    )
    return float(values[0])


#: Trust multiplier applied when the terminal anchor could NOT be measured (no
#: rollout completed, so ``critic_offset`` comes from the lowest-unrouted state the
#: policy reached instead of a real terminal). The ranking may still be usable, but
#: its zero point is extrapolated, so the bootstrap is admitted only weakly.
TRUST_UNVERIFIED = 0.25

def _critic_value_only(agent, obs) -> float | None:
    """V_tilde(s) with NO action distribution.

    ``_critic_value`` goes through ``act_and_value``, which samples an action and
    therefore trips the ``all-(-inf) cand pointer row`` guard on a FINISHED board
    (every action is masked). The critic value itself is read at the VAL token
    inside ``_encode_state`` and never touches the action heads, so read it there.
    Needed to measure the terminal anchor of the affine calibration."""
    import torch

    if not getattr(agent, "use_critic", False):
        return None
    model = getattr(agent, "model", None) or getattr(agent, "policy", None)
    enc = getattr(model, "_encode_state", None)
    if enc is None:
        return None
    with torch.no_grad():
        return float(enc([obs], action_masks=None).values[0])


def critic_scale_from_ckpt(ckpt_path, epsilon: float = 1e-8) -> float | None:
    """Exact training-time reward-norm std saved in the checkpoint.

    The PPO trainer normalizes rewards by the running std of the discounted
    return (VecNormalize-style, `RewardNormalizer`) and saves those stats as
    ``reward_normalizer_state`` (``loop._save_ckpt``) — its own comment says it
    is stored precisely so MCTS can map V_critic back into raw Φ units. Returns
    ``std = sqrt(var + eps)``, the exact ``critic_scale``. None when the ckpt has
    no such state (GRPO / no-reward-norm run).

    OPT-IN, not the default: the saved std is the training-time running average
    over the whole run, so it can be stale w.r.t. the loaded (best/late) policy —
    the default is the empirical :func:`estimate_critic_scale`, measured with the
    very policy that will do the searching. Select via
    :func:`resolve_critic_scale` (``source="ckpt"``).
    """
    import math

    import torch
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception:  # noqa: BLE001 — unreadable / not a .pt → fall back
        return None
    if not isinstance(ckpt, dict):
        return None
    rns = ckpt.get("reward_normalizer_state")
    if not isinstance(rns, dict) or "var" not in rns:
        return None
    std = math.sqrt(float(rns["var"]) + epsilon)
    return std if std > 1e-6 else None


def estimate_critic_scale(agent, wrapper, gamma: float = 0.995,
                          n_rollouts: int = 4, min_corr: float = 0.2,
                          min_samples: int = 20,
                          ratsnest_patience: int = 0,
                          ) -> tuple[float | None, float, float, dict]:
    """Empirically estimate the reward-norm std mapping V_critic into raw Phi units.

    The checkpoint trains the critic on reward-normalized returns (rewards / a
    running std), so V_critic lives in ``raw_return / std`` units. We recover
    ``std`` by regressing the realized discounted return-to-go G on V_critic --
    the ``critic_scale`` that makes ``Phi + scale*V_critic ~ Phi(terminal)``.

    **Stochastic rollouts, pooled, ended by early stopping** -- the episode
    definition the protocol itself uses. Three measured reasons, none of them a
    free parameter:

    * **Stochastic, not greedy.** The critic was fit on the ON-POLICY stochastic
      state distribution; a greedy roll visits a narrower and systematically
      different set of states, so the G-vs-V relation measured there is
      off-distribution for the very network being calibrated.
    * **Pooled over all rolls, NOT best-of-N.** Fitting against the best roll
      looks right for a search that seeks the best branch, but the critic is a
      MEAN predictor, so that fit returns std*(G_best/E[G]) -- inflated by
      exactly the selection. Measured on it150/maytal: pooled 9.75 vs best-of-N
      14.12 against a training-time std of 8.14, and in a d3b A/B the inflated
      scale cost 3 of 8 clean seeds on a board the smaller one solved 8/8.
      Pooling matches the quantity the critic actually represents.
    * **Early stopping, not a step cap.** A capped episode truncates G by a
      t-dependent amount (late t keeps fewer terms, so G compresses toward zero
      while V legitimately falls with the time feature) -- that alone manufactures
      the negative correlation. Stopping on stalled connections ends the episode
      instead of slicing it, so G is complete for the episode that happened.

    A scale only means something if V ranks states at all, so the fit is REFUSED
    (scale ``None``) when the correlation is below ``min_corr``, when the slope is
    non-positive -- a negative scale inverts the value ordering, i.e. the search
    would prefer exactly the states the critic calls bad -- or when too few
    samples survive. The caller picks the fallback.

    Mutates ``wrapper`` (resets and rolls ``n_rollouts`` times). Returns
    ``(scale_or_None, offset, trust, diag)``: ``offset`` = the terminal anchor
    (mean V_tilde at completion — the state whose remaining return is zero),
    ``trust`` = the Spearman-derived rank quality in [0,1] (0.0 when no rollout
    completed, i.e. the anchor is unverified), ``diag`` carries ``n``/``corr``/
    ``slope``/``terminated``/``slope_best`` (the same fit over the single best
    roll, kept so the selection bias above stays visible)/``reason``.
    """
    import numpy as np

    def _fit(v, g, anchor):
        """Constrained fit with the terminal anchor FIXED — returns (slope, spearman, corr).

        The search computes ``scale*(V - offset)``, so the fit must have that same
        shape or the slope is estimated under an intercept nobody uses. Fitting
        ``G = b + m*V`` freely and then substituting the anchor for ``b`` is exactly
        that mismatch: on maytal the free intercept was -4.97 while the anchor implies
        -m*anchor = -12.3. So shift first and regress through the origin on the shifted
        regressor:

            x_i = V_i - anchor,   slope = sum(x_i*G_i) / sum(x_i^2)

        which is the least-squares solution of ``G ~ slope*(V - anchor)``. The anchor
        is a physical constraint, not a free parameter: a completed board has zero
        remaining return, so the calibrated value must vanish there.

        Spearman is returned alongside Pearson because what the search consumes is a
        RANKING over siblings, and Pearson's corr had no measured predictive power for
        whether the critic helped (corr(corr, gain) = -0.047 over 12 boards)."""
        if v.size == 0:
            return 0.0, 0.0, 0.0
        x = v - anchor
        den = float((x * x).sum())
        slope = float((x * g).sum() / den) if den > 1e-12 else 0.0
        sv, sg = float(v.std()), float(g.std())
        corr = (float((((v - v.mean()) * (g - g.mean())).mean()) / (sv * sg))
                if sv > 1e-12 and sg > 1e-12 else 0.0)
        if v.size >= 3:
            rv = np.argsort(np.argsort(v)).astype(float)
            rg = np.argsort(np.argsort(g)).astype(float)
            sr, sgr = rv.std(), rg.std()
            spear = (float((((rv - rv.mean()) * (rg - rg.mean())).mean()) / (sr * sgr))
                     if sr > 1e-12 and sgr > 1e-12 else 0.0)
        else:
            spear = 0.0
        return slope, spear, corr

    pool = _SingleEnvPool(wrapper)
    prev_patience = getattr(wrapper.env, "_ratsnest_patience", 0)
    if ratsnest_patience > 0:
        wrapper.env._ratsnest_patience = int(ratsnest_patience)
    hard_cap = int(getattr(wrapper.env, "max_steps", 512) or 512) + 8
    n_roll = max(1, int(n_rollouts))
    episodes = []
    anchors: list[float] = []            # V_tilde at completion (the ONLY valid anchor)
    try:
        for _ in range(n_roll):
            obs, _ = wrapper.reset()
            rew: list[float] = []
            vc: list[float] = []
            term = False
            for _ in range(hard_cap):
                v = _critic_value(agent, pool, obs)
                if v is None:
                    return None, 0.0, 0.0, {"reason": "model has no critic",
                                            "n": 0, "n_rollouts": n_roll}
                vc.append(v)
                a = agent.act_from_pool(pool, [obs], [0], deterministic=False)[0]
                obs, r, term, trunc, _info = wrapper.step(a)
                rew.append(float(r))
                u_now = int(wrapper.env._engine.get_unrouted_count())
                if u_now == 0:
                    # Terminal anchor: a completed board has ZERO remaining return, so a
                    # calibrated critic must read 0 there. The completing step ends the
                    # episode with term=True, so this is the only place the terminal
                    # state is visible. act_and_value would trip its masking guard here.
                    vt = _critic_value_only(agent, obs)
                    if vt is not None:
                        anchors.append(vt)
                    break
                if term or trunc:
                    break
            acc = 0.0
            rtg = [0.0] * len(rew)
            for t in range(len(rew) - 1, -1, -1):        # backwards: O(T), exact
                acc = rew[t] + gamma * acc
                rtg[t] = acc
            episodes.append({"v": np.asarray(vc[:len(rtg)], dtype=float),
                             "g": np.asarray(rtg, dtype=float),
                             # Phi-shaped rewards telescope, so the undiscounted
                             # sum IS the potential gain the protocol selects on.
                             "gain": float(sum(rew)), "term": bool(term)})
    finally:
        wrapper.env._ratsnest_patience = prev_patience

    if anchors:
        offset = float(np.mean(anchors))
        anchor_src = f"terminal({len(anchors)}/{n_roll})"
        anchor_sd = float(np.std(anchors)) if len(anchors) >= 2 else float("nan")
        have_anchor = True
    else:
        # No rollout completed -> there is no state where the remaining return is
        # KNOWN to be zero, so the affine calibration has no fixed point. The
        # lowest-unrouted state reached is NOT a substitute: its residual return is
        # not zero and the proxies sit as far out as unrouted=7..29 on the boards that
        # need them most. Refuse the bootstrap entirely rather than anchor on a guess.
        offset, anchor_src, anchor_sd, have_anchor = 0.0, "none (no completion)", float("nan"), False

    if not episodes:
        return None, offset, 0.0, {"reason": "no episodes", "n": 0,
                                   "n_rollouts": n_roll, "offset": offset,
                                   "anchor": anchor_src}
    v = np.concatenate([e["v"] for e in episodes])
    g = np.concatenate([e["g"] for e in episodes])
    slope, spear, corr = _fit(v, g, offset)
    best = max(episodes, key=lambda e: e["gain"])       # diagnostic only
    slope_best, _s_best, _c_best = _fit(best["v"], best["g"], offset)
    diag = {"n": int(v.size), "n_rollouts": n_roll,
            "terminated": sum(e["term"] for e in episodes) / n_roll,
            "slope_best": slope_best, "slope": slope, "corr": corr, "spearman": spear,
            "offset": offset, "anchor": anchor_src, "anchor_sd": anchor_sd}
    # Trust lambda. Without a measured terminal anchor there is no zero point, so the
    # bootstrap is refused outright (A). With one, rank quality carries it: Spearman,
    # because the search consumes an ORDERING over siblings, not a magnitude.
    q = max(0.0, min(1.0, (spear - min_corr) / max(1e-9, 1.0 - min_corr)))
    trust = q if have_anchor else 0.0
    diag["trust"] = trust
    diag["rank_quality"] = q
    diag["anchor_verified"] = have_anchor

    if v.size < min_samples:
        diag["reason"] = f"too few samples (n={v.size} < {min_samples})"
        return None, offset, trust, diag
    if slope <= 0.0:
        diag["reason"] = (f"non-positive slope ({slope:+.3f}) -- V ranks states "
                          f"backwards")
        return None, offset, trust, diag
    if corr < min_corr:
        diag["reason"] = f"V uninformative (corr={corr:+.3f} < {min_corr})"
        return None, offset, trust, diag
    return slope, offset, trust, diag


#: Where ``critic_scale`` comes from. ``empirical`` (DEFAULT) = rollout
#: calibration with the loaded policy; ``ckpt`` = the training-time reward-norm
#: std saved in the checkpoint, falling back to empirical when absent.

CRITIC_SCALE_SOURCES = ("empirical", "ckpt")
DEFAULT_CRITIC_SCALE_SOURCE = "empirical"


def resolve_critic_scale(agent, ckpt_path, env_factory, gamma: float = 0.995,
                         source: str | None = None, n_rollouts: int = 4,
                         ratsnest_patience: int = 0
                         ) -> tuple[float, float, float, str]:
    """Resolve the critic calibration from the selected source.

    Returns ``(scale, offset, trust, label)`` — the affine calibration the
    search consumes as ``lambda*scale*(V - offset)`` plus a human-readable
    source label. The ``ckpt`` source and every fallback pin ``offset=0.0``
    (only the empirical fit measures a terminal anchor).

    ``empirical`` (the default): roll the loaded policy on a throwaway env from
    ``env_factory`` and take :func:`estimate_critic_scale`. ``ckpt``: the exact
    saved reward-norm std (:func:`critic_scale_from_ckpt`), falling back to the
    empirical calibration when the ckpt has no ``reward_normalizer_state``.

    When the empirical fit REFUSES (negative slope, uninformative V, too few
    samples) the fallback chain is ckpt std -> unscaled 1.0, and ``label`` says
    so: the search consumes this number as if it were a calibration, so a fit
    that failed its own checks must never be passed through silently.

    ``env_factory`` is called at most once and its env is ALWAYS closed here (the
    KiCad router is a per-process singleton, so callers must invoke this before
    building their live wrapper — never two live engines). ``label`` names the
    source for the caller's log line.
    """
    src = (source or DEFAULT_CRITIC_SCALE_SOURCE).lower()
    if src not in CRITIC_SCALE_SOURCES:
        raise ValueError(f"critic_scale source must be one of {CRITIC_SCALE_SOURCES}, "
                         f"got {source!r}")
    if src == "ckpt":
        scale = critic_scale_from_ckpt(ckpt_path)
        if scale is not None:
            return scale, 0.0, 1.0, "ckpt reward-norm std"
    cal = env_factory()
    try:
        scale, offset, trust, diag = estimate_critic_scale(
            agent, cal, gamma=gamma, n_rollouts=n_rollouts,
            ratsnest_patience=ratsnest_patience)
    finally:
        try:
            if cal.env._engine.is_routing():
                cal.env._engine.cancel_route()
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass
        cal.env.close()
    stat = (f"n={diag.get('n', 0)} over {diag.get('n_rollouts', 0)} rollouts, "
            f"spearman={diag.get('spearman', 0):+.2f}, corr={diag.get('corr', 0):+.2f}, "
            f"anchor={diag.get('anchor', '?')}")
    if scale is not None:
        return scale, offset, trust, f"empirical rollout calibration ({stat})"
    # Refused — an untrustworthy empirical fit is worse than no fit, because the
    # search consumes it as if it were a calibration. Prefer the exact saved std,
    # and only then leave the critic unscaled.
    why = diag.get("reason", "refused")
    fb = critic_scale_from_ckpt(ckpt_path)
    if fb is not None:
        return fb, 0.0, 0.0, f"ckpt reward-norm std [empirical REFUSED: {why}; {stat}]"
    return 1.0, 0.0, 0.0, f"unscaled 1.0 [empirical REFUSED: {why}; {stat}]"


class SamplingPolicyValue:
    """Trained decoder policy as the MCTS prior, estimated by SAMPLING.

    The model is autoregressive over ``(action_type, pointer, mode)`` and does
    not expose the full joint distribution, so the prior is estimated by sampling
    ``n_samples`` actions from the policy at the node and using their frequency
    over the legal actions. ``value`` is the critic estimate V_critic(s) (or
    ``None`` if the model has no critic) — the core discounts it into the leaf
    value as γ^depth·critic_scale·V_critic(leaf).

    Reuses ``KiCadRLAgent.act_from_pool`` (mask gathering + tokenization) so the
    inference path is identical to training/eval rollout.
    """

    def __init__(self, agent, wrapper, n_samples: int = 16) -> None:
        self._agent = agent
        self._pool = _SingleEnvPool(wrapper)
        self._k = n_samples

    def __call__(self, obs, legal_actions):
        from collections import Counter

        value = _critic_value(self._agent, self._pool, obs)
        legal = set(legal_actions)
        counts: Counter = Counter()
        for _ in range(self._k):
            a = self._agent.act_from_pool(
                self._pool, [obs], [0], deterministic=False,
            )[0]
            t = (int(a[0]), int(a[1]), int(a[2]))
            if t in legal:
                counts[t] += 1

        if not counts:  # policy never sampled a legal action → fall back to uniform
            p = 1.0 / len(legal_actions) if legal_actions else 0.0
            return {a: p for a in legal_actions}, value

        total = sum(counts.values())
        return {a: counts.get(a, 0) / total for a in legal_actions}, value


class LogitPolicyValue:
    """Sampler-exact, deterministic policy prior with the state encoded ONCE.

    The canonical MCTS prior: a single ``factored_action_logits`` call returns the
    autoregressive factor logits for ALL action types, and each legal action is
    scored as its exact joint ``P(at)·P(ptr|at)·P(mode|·)``. Cost ≈ 2–3 forwards
    (state-encode + one batch over the action types + one batched Pass-2 over the
    candidate pointers) regardless of how many candidates are legal — far cheaper
    than a per-candidate re-encode, yet numerically identical to the teacher-forced
    ``evaluate_actions_and_value`` reference (verified ~1e-7).

    The mode factor is sampler-exact for every type: ``finish`` reads the
    pre-pointer hidden (``mode_at_logits``), make_line/make_via read the
    POST-pointer hidden (``mode_pt_logits``). ``value`` is V_critic(s) or ``None``.
    """

    def __init__(self, agent, wrapper) -> None:
        self._agent = agent
        self._pool = _SingleEnvPool(wrapper)


    def __call__(self, obs, legal_actions):
        import math

        import numpy as np
        import torch

        from methods.rl_agent.models.v1.encoding import (
            stack_action_masks, stack_mode_masks, stack_net_valid_masks,
            stack_offlayer_masks,
            stack_pointer_masks,
        )
        from methods.rl_agent.models.v1.net import ACT_NET_SELECT, SLOT_USAGE

        legal = list(legal_actions)
        if not legal:
            return {}, None
        agent, pool = self._agent, self._pool
        dev = getattr(agent, "device", None)
        policy_net_select = bool(getattr(agent, "policy_net_select", False))

        amask = torch.as_tensor(
            stack_action_masks(pool, indices=[0]), dtype=torch.bool, device=dev)
        pmask = torch.as_tensor(
            stack_pointer_masks(pool, indices=[0]), dtype=torch.int64, device=dev)
        mode_np = stack_mode_masks(pool, indices=[0])
        mode_arr = np.asarray(mode_np) if mode_np is not None else None
        mmask = (
            torch.as_tensor(mode_arr, dtype=torch.bool, device=dev)
            if mode_arr is not None and mode_arr.size else None
        )
        extra = {
            "offlayer_masks": torch.as_tensor(
                stack_offlayer_masks(pool, indices=[0]),
                dtype=torch.int64, device=dev),
        }
        if policy_net_select:
            extra["net_valid_mask"] = torch.as_tensor(
                stack_net_valid_masks(pool, indices=[0]), dtype=torch.bool, device=dev)

        out = agent.model.factored_action_logits(
            [obs], action_masks=amask, pointer_masks=pmask, mode_mask=mmask,
            **extra)
        at_lp = torch.log_softmax(out["at_logits"][0], dim=-1)        # (T,)
        ptr_lp = torch.log_softmax(out["ptr_logits"][0], dim=-1)      # (T, K)
        mode_lp = torch.log_softmax(out["mode_at_logits"][0], dim=-1)  # (T, 3)
        # Exact POST-pointer mode for make_line/make_via; finish uses the
        # pre-pointer mode_lp (its sampler reads h_at, no pointer).
        mode_pt_lp = torch.log_softmax(out["mode_pt_logits"][0], dim=-1)  # (T, K, 3)

        logp: dict = {}
        for a in legal:
            at, ptr, mode = int(a[0]), int(a[1]), int(a[2])
            lp = float(at_lp[at])
            needs_ptr = bool(SLOT_USAGE[at, 0])
            needs_mode = bool(SLOT_USAGE[at, 1])
            # net_select's pointer is policy-driven only when policy_net_select
            # (mirrors evaluate's allow_net_select_lp gate).
            if at == ACT_NET_SELECT and not policy_net_select:
                needs_ptr = False
            if needs_ptr and 0 <= ptr < ptr_lp.shape[-1]:
                lp += float(ptr_lp[at, ptr])
            if needs_mode and 0 <= mode < mode_lp.shape[-1]:
                if needs_ptr and 0 <= ptr < mode_pt_lp.shape[1]:
                    lp += float(mode_pt_lp[at, ptr, mode])   # exact (post-pointer)
                else:
                    lp += float(mode_lp[at, mode])           # finish (pre-pointer)
            logp[a] = lp

        mx = max(logp.values())
        exps = {a: math.exp(v - mx) for a, v in logp.items()}
        z = sum(exps.values()) or 1.0
        prior = {a: e / z for a, e in exps.items()}
        value = float(out["values"][0]) if getattr(agent, "use_critic", False) else None
        if os.environ.get("MCTS_LOG_PRIOR"):
            import collections as _c
            by_at = _c.defaultdict(float); cnt = _c.Counter()
            for a, p in prior.items():
                by_at[int(a[0])] += p; cnt[int(a[0])] += 1
            nm = {0: "nsel", 1: "sroute", 2: "NEND", 3: "make_line", 4: "make_via", 5: "FINISH"}
            summ = " ".join(f"{nm.get(k,k)}={by_at[k]:.3f}(n{cnt[k]})"
                            for k in sorted(by_at, key=lambda k: -by_at[k]))
            print(f"    [prior] {summ}", flush=True)
        return prior, value


class CalibratedPolicyValue:
    """Remap a PolicyValue's critic output through a fitted 1-D MONOTONE map.

    The affine calibration (:func:`estimate_critic_scale`) fits ``scale*(V-offset)``
    by least squares and REFUSES on a non-positive slope. Measured on d2c it250,
    that refusal throws away boards whose ORDERING is fine: the least-squares slope
    came out negative (0144 -2.99, 0203 -1.99) while ``Spearman(V, G)`` on the same
    data was POSITIVE (+0.149, +0.277) — the line was dragged by leverage, not by an
    inverted ranking. Since the search consumes an ordering over siblings (the reason
    :func:`estimate_critic_scale` reports Spearman at all), an isotonic map keeps that
    ordering and corrects only the scale, and it degrades to a constant — which the
    global min-max cancels, i.e. exactly the critic-off arm — when V really is
    uninformative, instead of a binary refusal.

    Shape follows Iterated Bellman Calibration (van der Laan & Kallus, arXiv
    2512.23694): a post-hoc 1-D map of the ORIGINAL prediction, fitted either on
    Monte-Carlo returns or on Bellman targets ``R + gamma*V(S')``. The Bellman variant
    needs no terminal anchor, so it also covers the boards where no rollout completes
    (0232: 0 anchors, where the affine path assigns lambda=0 by rule).

    The wrapped value is consumed as ``lambda*scale*(v-offset)`` downstream, so the
    caller pairs this with ``scale=1, offset=0`` and keeps lambda as the trust knob.
    """

    def __init__(self, pv, knots_x, knots_y) -> None:
        import numpy as np
        self._pv = pv
        self._kx = np.asarray(knots_x, dtype=float)
        self._ky = np.asarray(knots_y, dtype=float)
        if self._kx.size < 2 or self._kx.size != self._ky.size:
            raise ValueError("isotonic knots must be two aligned arrays of length >= 2")

    def __call__(self, obs, legal_actions):
        import numpy as np
        prior, value = self._pv(obs, legal_actions)
        if value is None:
            return prior, value
        cal = float(np.interp(float(value), self._kx, self._ky,
                              left=float(self._ky[0]), right=float(self._ky[-1])))
        return prior, cal

    # -- delegate the memo/driver hooks so this can sit anywhere in the chain
    def new_generation(self) -> None:
        fn = getattr(self._pv, "new_generation", None)
        if fn is not None:
            fn()

    def reset(self) -> None:
        fn = getattr(self._pv, "reset", None)
        if fn is not None:
            fn()

    def __getattr__(self, name):          # anything else -> wrapped instance
        return getattr(self._pv, name)


def load_isotonic_calibrator(path: str):
    """Load ``(knots_x, knots_y, lam)`` from a ``.npz`` written by the fitter.

    Keys: ``knots_x`` / ``knots_y`` (required, aligned 1-D), ``lam`` (optional
    trust; the CLI value wins when given).
    """
    import numpy as np
    z = np.load(path)
    lam = float(z["lam"]) if "lam" in z.files else None
    return np.asarray(z["knots_x"], float), np.asarray(z["knots_y"], float), lam


def _obs_fingerprint(obs: Any, *, skip=("board_static",)) -> bytes:
    """Deterministic 16-byte digest of an observation.

    Recurses dicts (key-sorted) / lists / tuples / ndarrays / scalars, so two
    obs with identical content hash equal and different content (almost surely)
    differ. ``board_static`` is skipped: it is the ONLY episode-invariant blob
    (both obs formats label it so) and it is large, so hashing it every call is
    pure cost. Everything else — ``routing_geometry``, ``router_head``,
    ``drc_violations``, ``prev_action``, ``closed_nets``, ``_aug`` — is genuinely
    per-step state and MUST be hashed; skipping any of it would collide distinct
    board states onto one memo entry.

    Arrays are fed to blake2b through their buffer instead of ``tobytes()`` when
    already contiguous — the dynamic tables are the bulk of the digest input, so
    the copy is worth avoiding on the per-node path.
    """
    h = hashlib.blake2b(digest_size=16)

    def upd(x: Any) -> None:
        if isinstance(x, np.ndarray):
            h.update(b"a")
            h.update(str(x.dtype).encode())
            h.update(repr(x.shape).encode())
            h.update(x.data if x.flags.c_contiguous
                     else np.ascontiguousarray(x).data)
        elif isinstance(x, dict):
            h.update(b"d")
            for k in sorted(x):
                if k in skip:
                    continue
                h.update(str(k).encode())
                upd(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b"l")
            for e in x:
                upd(e)
        else:
            h.update(b"s")
            h.update(repr(x).encode())

    upd(obs)
    return h.digest()


class MemoizingPolicyValue:
    """Wrap a ``PolicyValueFn`` with a memo keyed by (obs fingerprint, legal set).

    The search wrapper rebuilds its candidate pool FROM the obs on every restore
    (``RLSearchEnv.restore`` → ``_refresh_cache(aug_obs)``), so a provider's
    prior/value is a function of the obs plus the legal action set it is asked
    about — and the key carries both, so caching it is bit-identical to
    recomputing: the search structure, expansion order, and every backed-up
    value are unchanged; only the redundant policy forward is skipped. The wins
    come from board states that recur (the committed subtree re-explored next
    decision, and within-decision transpositions). The cache is episode-scoped:
    a fresh instance per rollout, or call :meth:`reset`.

    Entries are dropped GENERATIONALLY: :meth:`new_generation`, called by the
    driver right after it commits an action, keeps only what the just-finished
    decision touched and discards the rest. The states worth keeping are exactly
    those under the committed action — the search walked them last decision, so
    they are in the touched set, while every sibling subtree the commit made
    unreachable ages out in one step. Without it the memo grows for the whole
    episode holding boards that can never recur.
    """

    def __init__(self, pv) -> None:
        self._pv = pv
        self._cache: dict[tuple, tuple] = {}
        self._touched: set[tuple] = set()
        self.hits = 0
        self.misses = 0
        self.dropped = 0
        # MCTS_AUDIT_PV_CACHE=1 recomputes the provider on every hit and reports
        # any mismatch. The memo keys on the obs digest alone, but the provider
        # also reads the LIVE masks (``_SingleEnvPool``) and builds its dict over
        # the passed ``legal_actions`` — so "pure function of the obs" is an
        # assumption about the obs being a faithful board key, and the edge cache
        # proved that assumption false for ITS purposes on a dense board. Audited on
        # 0344_mavbridge: 0 mismatches, so the memo is sound in practice — keep
        # the switch for the next board that makes anyone wonder.
        self.collisions = 0
        self._audit = os.environ.get("MCTS_AUDIT_PV_CACHE") == "1"


    def __call__(self, obs, legal_actions):
        # Key on the obs digest AND the legal set. The obs alone is NOT a
        # sufficient key by construction: the provider also reads the live masks
        # out of `_SingleEnvPool` (action / pointer / mode / net_valid) and builds
        # its returned dict over exactly the `legal_actions` passed in, so two
        # states sharing an obs digest but differing in either would be served
        # each other's prior. Auditing found no such pair (see `_audit`), but that
        # was a measurement on one board, not a guarantee — and a masking-rule
        # change could break it silently. The legal set is what the masks
        # enumerate, so keying on it makes the assumption enforced rather than
        # merely observed, for a tuple hash of a few dozen triples.
        key = (_obs_fingerprint(obs), tuple(map(tuple, legal_actions)))
        self._touched.add(key)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            if self._audit:
                fresh = self._pv(obs, legal_actions)
                if repr(fresh) != repr(cached):
                    self.collisions += 1
                    logger.error(
                        "PV MEMO COLLISION: same obs digest, different "
                        "prior/value — legal %d vs %d, value %s vs %s",
                        len(fresh[0]), len(cached[0]), fresh[1], cached[1],
                    )
            return cached
        self.misses += 1
        out = self._pv(obs, legal_actions)
        self._cache[key] = out
        return out

    def new_generation(self) -> None:
        """Drop entries the last decision never touched (call after a commit)."""
        if not self._cache:
            return
        keep = self._touched
        self.dropped += len(self._cache) - len(keep)
        self._cache = {k: v for k, v in self._cache.items() if k in keep}
        self._touched = set()

    def __len__(self) -> int:
        return len(self._cache)

    def reset(self) -> None:
        self._cache.clear()
        self._touched.clear()
        self.hits = 0
        self.misses = 0
        self.dropped = 0
