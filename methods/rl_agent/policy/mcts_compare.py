"""Compare the RL baseline rollout against MCTS on routing boards.

Loads a trained DecoderOnly PPO checkpoint, then per board runs (see ``--mode``):
  - baseline: the REAL eval/PPO rollout, run verbatim via
    ``methods.rl_agent.rollout.transformer.eval_transformer`` (NOT a re-implemented
    greedy loop) — the exact number the trained model scores under eval.
  - mcts    : ``run_search`` (policy prior + Φ/critic value) each decision, over
    one of the named algorithm profiles (``--algo``).
Reports routing completion (unrouted; lower=better), track/via counts,
wirelength, DRC violations, and Φ; with ``--mode both`` also the deltas.

The two rollouts are INDEPENDENT: ``--mode plain`` / ``--mode mcts`` runs just one
(``both`` is the default). ``--mode mcts`` skips the critic-scale calibration only
when it is not needed — it is still done for the search value.

Run:
    python -m methods.rl_agent.policy.mcts_compare <board.kicad_pcb> [more boards...]
    python -m methods.rl_agent.policy.mcts_compare --mode plain <board>   # baseline only
    python -m methods.rl_agent.policy.mcts_compare --mode mcts --algo gumbel <board>

Tunables are CLI flags (see --help); each also honors an MCTS_* env var as its
default, so `MCTS_NSIM=100 MCTS_ALGO=gumbel ... ` sets them from the environment.
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

# Support both `python -m methods.rl_agent.policy.mcts_compare` (repo root already on path) and a
# bare `python methods/rl_agent/policy/mcts_compare.py` by ensuring the repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from methods.rl_agent.rollout.transformer import (
    load_policy_from_ckpt, FinishNoProgressGuard, NoGeometryProgressGuard,
    TRACE_ACTION_NAMES,
)
from methods.rl_agent.rollout.primitive import iter_rollout
from methods.rl_agent.models.loader import env_kwargs_from_checkpoint
from methods.rl_agent.wrappers.factory import make_decoder_env
from methods.rl_agent.policy.agent import KiCadRLAgent
from methods.rl_agent.policy.mcts_env import (
    RLSearchEnv, LogitPolicyValue, MemoizingPolicyValue,
    CalibratedPolicyValue, load_isotonic_calibrator,
    resolve_critic_scale, CRITIC_SCALE_SOURCES, DEFAULT_CRITIC_SCALE_SOURCE,
)
from methods._shared.mcts import (
    DEFAULT_SEARCH_GAMMA, MctsConfig, make_algorithm, run_search,
)
# eval-pipeline scoring/selection reused verbatim so plain & MCTS are scored
# and picked EXACTLY like the uniform eval (native .kicad_pro DRC, best@k).
from pcb_world.core.masking import ACT_NET_SELECT
from eval.eval_utils import select_best_in_board
from configs.loader.schema import DEFAULTS as _EVAL_DEFAULTS


def _env(name, default, cast):
    """argparse default sourced from an MCTS_* env var."""
    raw = os.environ.get(name)
    return cast(raw) if raw not in (None, "") else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plain policy rollout vs MCTS on PCB routing boards.")
    p.add_argument("boards", nargs="*",
                   default=["tests/fixtures/simple_routing_board.kicad_pcb"],
                   help="board .kicad_pcb paths (default: simple fixture)")
    p.add_argument("--ckpt", default=str(Path.home() / "policy_best.pt"),
                   help="policy checkpoint (default: ~/policy_best.pt)")
    p.add_argument("--mode", default=_env("MCTS_MODE", "both", str),
                   choices=["plain", "mcts", "both"],
                   help="which rollout(s) to run: plain (baseline only) | mcts "
                        "(search only) | both (baseline + search + Δ, default). "
                        "plain/mcts are independent so each can run on its own "
                        "[MCTS_MODE]")
    p.add_argument("--n-sim", type=int, default=_env("MCTS_NSIM", 200, int),
                   help="MCTS simulations per decision [MCTS_NSIM]")
    p.add_argument("--dirichlet", type=float, default=_env("MCTS_DIR", None, float),
                   help="root Dirichlet exploration alpha, 0=off. Omit (None) to "
                        "let a --algo profile keep its OWN default (AlphaZero=0, "
                        "MuZero=0.3) — passing this flag OVERRIDES the profile's "
                        "default (e.g. --algo alphazero --dirichlet 0.3 for "
                        "canonical AlphaZero, or --algo muzero --dirichlet 0 for "
                        "a no-noise MuZero ablation). With no --algo, omitting "
                        "means 0 (off) [MCTS_DIR]")
    p.add_argument("--ep-cap", type=int, default=_env("MCTS_EPCAP", 0, int),
                   help="committed-step budget for BOTH greedy and MCTS "
                        "(0 = use the policy's max_steps, matching the eval "
                        "rollout) [MCTS_EPCAP]")
    # Early-stop guards: applied to BOTH the greedy and the MCTS committed
    # trajectory (parity with the canonical eval rollout). Default 0 = off.
    p.add_argument("--early-stop-finish-no-progress", type=int,
                   default=_env("MCTS_ES_FINISH", 0, int),
                   help="stop a rollout after N finish-no-progress steps "
                        "(0=off, eval default) [MCTS_ES_FINISH]")
    p.add_argument("--early-stop-no-geometry-progress", type=int,
                   default=_env("MCTS_ES_GEOM", 0, int),
                   help="stop a rollout after N no-geometry-change steps "
                        "(0=off, eval default) [MCTS_ES_GEOM]")
    p.add_argument("--early-stop-ratsnest", type=int,
                   default=_env("MCTS_ES_RATS", 0, int),
                   help="env-side stop after N steps with no change in unrouted "
                        "count (connections stalled); 0=off [MCTS_ES_RATS]")
    p.add_argument("--plain-early-stop-ratsnest", type=int,
                   default=_env("MCTS_ES_RATS_PLAIN", None, int),
                   help="separate early-stop for the PLAIN phase; default = same "
                        "as --early-stop-ratsnest. Measured on d3b (iter150 d2a, "
                        "10 boards x 5 seeds): turning the stop OFF raised plain "
                        "routability 0.627 -> 0.712 with no board losing, so a "
                        "time-matched plain should not be handicapped by a stop "
                        "the comparison does not need [MCTS_ES_RATS_PLAIN]")
    p.add_argument("--output-best-board", action="store_true",
                   default=_env("MCTS_BEST_BOARD", False, lambda s: s == "1"),
                   help="score/save the highest-Φ board seen this episode (env "
                        "rolls the live board back to it at episode end) instead of "
                        "the final board [MCTS_BEST_BOARD=1]")
    p.add_argument("--seed", type=int, default=_env("MCTS_SEED", 0, int),
                   help="seed for torch+np+MctsConfig [MCTS_SEED]")
    p.add_argument("--critic-lambda", type=float,
                   default=_env("MCTS_CLAMBDA", None, float),
                   help="lambda in [0,1] multiplying the critic bootstrap; omit to use "
                        "the value the calibration derives. The calibration sets it "
                        "from what it could actually verify: 0 when corr is at/below "
                        "the refusal floor (V ranks states no better than the guard "
                        "already rejects), a small fraction when no rollout completed "
                        "(the terminal anchor is extrapolated from the lowest-unrouted "
                        "state reached, so the ranking may be real but its zero point "
                        "is not measured), and up to 1 when completion was observed "
                        "[MCTS_CLAMBDA]")
    p.add_argument("--critic-offset", type=float,
                   default=_env("MCTS_COFFSET", 0.0, float),
                   help="offset subtracted from the RAW critic output before "
                        "--critic-scale: boot = scale*(V_tilde - offset). Denormalizing "
                        "with the ckpt reward-norm std is exact (the trainer divides by "
                        "std and never subtracts a mean), and regressing G on sigma*V "
                        "with a free intercept gives slope ~1 (0.98..1.42) -- but an "
                        "intercept of -12.2..+7.3 that flips sign per board. The anchor "
                        "is the terminal state, where the true remaining return is 0 "
                        "while the measured V_tilde is +0.920 (maytal)..-1.007 (NiMH). "
                        "Set this to that value to make the bootstrap vanish at "
                        "completion [MCTS_COFFSET]")
    p.add_argument("--critic-scale", type=float,
                   default=_env("MCTS_CSCALE", None, float),
                   help="override critic_scale with a fixed value for EVERY "
                        "board, skipping calibration: one number broadcast "
                        "instead of a per-board fit. Measured spread across "
                        "boards is up to 2.3x, so this is a control arm, not a "
                        "recommended setting [MCTS_CSCALE]")
    p.add_argument("--critic-scale-rollouts", type=int,
                   default=_env("MCTS_CSCALE_ROLLOUTS", 4, int),
                   help="stochastic rollouts per board for the empirical "
                        "critic_scale fit. The critic was trained ON-POLICY, so "
                        "the calibration rolls the policy stochastically (not "
                        "greedily) and fits a slope through the origin; more "
                        "rollouts tighten it [MCTS_CSCALE_ROLLOUTS]")
    p.add_argument("--critic-scale-source",
                   default=_env("MCTS_CSCALE_SRC", DEFAULT_CRITIC_SCALE_SOURCE, str),
                   choices=list(CRITIC_SCALE_SOURCES),
                   help="where critic_scale (V_critic → raw Φ units) comes from. "
                        "empirical (DEFAULT) = rollout calibration with the LOADED "
                        "policy on a throwaway env (tracks this checkpoint); ckpt = "
                        "the training-time reward-norm std saved in the ckpt "
                        "(reward_normalizer_state), which falls back to empirical "
                        "when the ckpt lacks it [MCTS_CSCALE_SRC]")
    p.add_argument("--critic-isotonic", default=_env("MCTS_CISO", None, str),
                   help="path to a .npz with 'knots_x'/'knots_y' (and optional "
                        "'lam'): remap V_critic through that fitted MONOTONE 1-D map "
                        "instead of the affine scale/offset. Takes precedence over "
                        "--critic-scale; scale/offset are then pinned to 1/0 and "
                        "--critic-lambda stays the trust knob. Rationale + the "
                        "measured case for it: mcts_env.CalibratedPolicyValue "
                        "[MCTS_CISO]")
    p.add_argument("--gumbel-value-scale", type=float,
                   default=_env("MCTS_GVS", None, float),
                   help="mctx qtransform value_scale (default 0.1): sets σ(q̂)'s "
                        "magnitude via value_scale·(maxvisit_init + max_visit), i.e. how "
                        "much the completed-Q outweighs the Gumbel noise AND the prior "
                        "logit in the root pick. Raise it to trust Q more WITHOUT "
                        "touching the Gumbel noise, which is the only source of MCTS's "
                        "across-seed diversity (measured seed sd 0.136 vs 0.021 for "
                        "best-of-N — and that spread is why its best-of-N oracle keeps "
                        "up while its mean does not) [MCTS_GVS]")
    p.add_argument("--gumbel-scale", type=float, default=_env("MCTS_GS", None, float),
                   help="multiplier on the sampled Gumbel noise (default 1.0). Lowering "
                        "it also raises q̂'s relative weight, but it DESTROYS the "
                        "across-seed diversity above, and Sequential Halving's FIRST "
                        "round ranks purely by g+logit (q̂ is identically 0 there — no "
                        "child is visited yet), so at a flat-prior node like net_select "
                        "(measured max/min prior 1.008) scale→0 collapses the top-m "
                        "choice to a dict-order tie-break. Prefer "
                        "--gumbel-value-scale [MCTS_GS]")
    p.add_argument("--search-via-penalty", type=float,
                   default=_env("MCTS_SEARCH_VIA_PEN", None, float),
                   help="override via_penalty in the Φ THE SEARCH OPTIMIZES (scoring is "
                        "untouched — it goes through --scoring-reward-rule). The value "
                        "normally comes from the CHECKPOINT's training args, which "
                        "override the reward rule's own (configs/loader/schema.py), so a "
                        "yaml copy cannot change it. Diagnostic: on d3b MCTS commits "
                        "ZERO make_via while a time-matched plain rollout of the SAME "
                        "policy commits 7-8, and the search diag shows SOLVED=0 with "
                        "cap=531 — the via's -0.1 is inside the horizon and its payoff "
                        "is not. Setting 0 separates 'the penalty' from 'the horizon' "
                        "[MCTS_SEARCH_VIA_PEN]")
    p.add_argument("--prefilter-refused", action="store_true",
                   default=_env("MCTS_PREFILTER", 0, int) == 1,
                   help="drop make_via candidates the engine is guaranteed to refuse "
                        "(via_on_thru_pad) from the search's legal set instead of "
                        "discovering the refusal by stepping. Measured: make_via is "
                        "52.7%% of popped children on d3b = 387 wasted env.step per "
                        "rollout (11.3%% of all steps), a cost plain/best-of-N never "
                        "pays because it never enumerates candidates. The test is the "
                        "engine's own pure-geometry predicate on cached thru-pad "
                        "geometry [MCTS_PREFILTER=1]")
    p.add_argument("--prior-net-select", action="store_true",
                   default=_env("MCTS_PRIOR_NSEL", 0, int) == 1,
                   help="skip the search at net_select states and sample the action "
                        "from the policy prior instead. Masking makes the action types "
                        "mutually exclusive per state, so at such a state EVERY legal "
                        "action is a net_select and every sibling edge has ΔΦ = 0 — the "
                        "sibling contrast is then purely the critic, min-max-stretched "
                        "to the full range, and the head xy the critic sees is the same "
                        "(0,0) not-routing sentinel for every choice (only the net slot "
                        "and the candidate pool differ). Measured: base's own Q gap "
                        "there is 0.001, so its Gumbel pick already collapses to "
                        "argmax(g + logit) = a prior sample; doing it explicitly frees "
                        "the whole n_sim budget for the decisions that have ΔΦ "
                        "[MCTS_PRIOR_NSEL=1]")
    p.add_argument("--search-diag", action="store_true",
                   default=_env("MCTS_SEARCH_DIAG", 0, int) == 1,
                   help="per-decision search telemetry: Φ(root), how many of the "
                        "n_sim simulations reached a terminal / a SOLVED (unrouted==0) "
                        "leaf / the max_depth cap, leaf-value range, and the top root "
                        "children (action,N,Q,prior). Diagnoses completion-stall "
                        "(solved==0 every decision ⇒ search never sees completion "
                        "within max_depth) [MCTS_SEARCH_DIAG=1]")
    p.add_argument("--gamma", type=float, default=_env("MCTS_GAMMA", None, float),
                   help="discount γ for the return_bootstrap value (per-edge exact "
                        "discounted return Σγ^k ΔΦ_k + γ^depth·c·V). A SEARCH "
                        "REGULARIZER (different role from the training γ): a completion "
                        "in fewer steps outranks the same Φ later, breaking the "
                        "near-goal Q plateau (flat return → prior-driven dithering). "
                        f"DEFAULT = {DEFAULT_SEARCH_GAMMA} (strong; the training γ≈0.995 "
                        "is near-inert over the short lookahead and fails to complete) "
                        "[MCTS_GAMMA]")
    p.add_argument("--algo", default=_env("MCTS_ALGO", "alphazero", str),
                   choices=["puct", "alphazero", "muzero", "gumbel"],
                   help="algorithm profile: named profiles (alphazero|muzero|gumbel) "
                        "set the coherent module set (root selection / noise / "
                        "value-completion / gumbel knobs) in one shot, overriding the "
                        "individual --dirichlet/--value-completion flags; puct = the "
                        "raw generic flat-c_puct config. Default alphazero (completed "
                        "the 55-net fancy board at n_sim=200 where gumbel plateaued) "
                        "[MCTS_ALGO]")
    p.add_argument("--gumbel-max-considered", type=int,
                   default=_env("MCTS_GUMBEL_M", 16, int),
                   help="m: actions kept in the first Sequential-Halving round "
                        "[MCTS_GUMBEL_M]")
    p.add_argument("--value-completion", action="store_true",
                   default=_env("MCTS_VC", 0, int) == 1,
                   help="complete unvisited interior Q with the parent value "
                        "(mctx-style; §3-2 companion to gumbel) [MCTS_VC=1]")
    p.add_argument("--max-depth", type=int, default=_env("MCTS_MAXD", 6, int),
                   help="lookahead depth cap [MCTS_MAXD]")
    p.add_argument("--pw-alpha", type=float, default=_env("MCTS_PW_ALPHA", 0.0, float),
                   help="progressive-widening exponent α: interior selectable "
                        "children k=ceil(pw_base·N^α) grow with visits N (prior "
                        "order), concentrating budget on likely branches while "
                        "deferring (not pruning) low-prior ones. 0=off. Try ~0.5 "
                        "[MCTS_PW_ALPHA]")
    p.add_argument("--pw-base", type=float, default=_env("MCTS_PW_BASE", 1.0, float),
                   help="progressive-widening constant C in k=ceil(C·N^α); larger "
                        "= wider from the start [MCTS_PW_BASE]")
    p.add_argument("--invalid-mode", default=_env("MCTS_INVALID_MODE", "pop", str),
                   choices=["pop", "drop", "penalize"],
                   help="policy for a child that steps to invalid (failed/no-op): "
                        "'pop' removes it and RE-SELECTS in the same simulation, so "
                        "n_simulations does not bound the engine work (measured 735 "
                        "pops per rollout on top of 2688 simulations = 21.5%% of all "
                        "env.step calls, concentrated late where invalids explode); "
                        "'drop' removes it and ENDS the simulation with no backup, so "
                        "n_simulations IS the step budget; 'penalize' backs "
                        "up a one-time penalty to ancestors' Q before removing it "
                        "[MCTS_INVALID_MODE]")
    p.add_argument("--invalid-penalty", type=float,
                   default=_env("MCTS_INVALID_PENALTY", 0.1, float),
                   help="penalty (raw Φ units) subtracted below the dead-end base "
                        "under --invalid-mode penalize [MCTS_INVALID_PENALTY]")
    p.add_argument("--no-pv-cache", action="store_true",
                   default=_env("MCTS_NO_PV_CACHE", False, lambda s: s == "1"),
                   help="disable the cross-decision prior/value memo (recompute "
                        "every forward). The memo is bit-identical to cold search "
                        "and only caches the policy forward [MCTS_NO_PV_CACHE=1]")
    p.add_argument("--corner-mode", type=int, default=_env("MCTS_CORNER", None, int),
                   choices=[0, 1, 2, 3],
                   help="PNS corner mode: 0=MITERED_45 (H/V/45), 1=ROUNDED_45, "
                        "2=MITERED_90 (H/V only, 90° corners), 3=ROUNDED_90. "
                        "Omit to INHERIT the checkpoint's training corner mode "
                        "[MCTS_CORNER]")
    p.add_argument("--save-mcts", default=None,
                   help="save the MCTS-routed board to this .kicad_pcb path "
                        "(companion .kicad_pro emitted; multi-board → board stem prefixed)")
    p.add_argument("--reward-rule", default=None,
                   help="override the checkpoint's reward rule for Φ/scoring "
                        "(e.g. drc_dense_promoted = errors+promoted vs the "
                        "ckpt's drc_only_dense = errors_only) [MCTS_REWARD]")
    p.add_argument("--masking-rule", default=None,
                   help="override the checkpoint's action-masking rule "
                        "(e.g. default_no_finish_no_via = forbid make_via too) "
                        "[MCTS_MASK]")
    # --- plain-baseline knobs (aligned with eval.pipeline) --------------------
    p.add_argument("--n-rollouts", type=int, default=_env("MCTS_NROLL", 5, int),
                   help="plain best@k rollouts per board when NOT time-matched "
                        "(i.e. --mode plain). In --mode both the plain count is "
                        "driven by the MCTS wallclock instead [MCTS_NROLL]")
    p.add_argument("--deterministic", action="store_true",
                   default=_env("MCTS_DET", 0, int) == 1,
                   help="plain baseline uses greedy argmax instead of the eval "
                        "default (stochastic sampling → best@k) [MCTS_DET=1]")
    p.add_argument("--selection-mode", default=_env("MCTS_SEL", "final_potential", str),
                   choices=["final_potential", "posthoc_drc_aware"],
                   help="best@k winner key over plain rollouts (eval.eval_utils."
                        "selection_key) [MCTS_SEL]")
    p.add_argument("--scoring-reward-rule",
                   default=_env("MCTS_SCORE_RULE", _EVAL_DEFAULTS.reward_config, str),
                   help="reward config for the FINAL native-.kicad_pro DRC scoring "
                        "of BOTH plain and MCTS boards (eval convention; separate "
                        "from --reward-rule which sets the env Φ/value the search "
                        "optimizes) [MCTS_SCORE_RULE]")
    p.add_argument("--check-angle", type=int, default=_env("MCTS_ANGLE", None, int),
                   choices=[45, 90],
                   help="track-angle DRC mode for scoring; omit to inherit the "
                        "checkpoint's stored check_angle if any, else derive from "
                        "the (ckpt-inherited) corner mode: 90 for MITERED/ROUNDED_90, "
                        "else 45 [MCTS_ANGLE]")
    return p.parse_args(argv)


def _make_env(board, env_kwargs, corner_mode=None):
    """Create the decoder env and apply the PNS corner mode once. The corner
    mode lives in m_settings (set in initRouter at construction); wrapper.reset()
    only rips up tracks, so it persists across every reset + checkpoint/restore."""
    w = make_decoder_env(board, **env_kwargs)
    if corner_mode is not None:
        w.env._engine.set_corner_mode(corner_mode)
    return w


def _derive_check_angle(args) -> int:
    """Track-angle DRC mode for scoring: explicit --check-angle, else derived
    from the corner mode (90° corners → 90, otherwise 45) so the scoring angle
    matches the geometry the board was actually routed with. Plain and MCTS
    share this one value so their DRC numbers are comparable."""
    if args.check_angle is not None:
        return int(args.check_angle)
    return 90 if args.corner_mode in (2, 3) else 45


def score_live(wrapper, reward_cfg: str, check_angle: int, rollout_idx: int = 0) -> dict:
    """Canonical eval scoring of the live, just-routed board.

    Delegates to the wrapper's own ``eval_inline_drc`` hook — the SAME
    ``eval.metrics.compute_metrics_inline`` entry the uniform eval uses in serial
    ``--inline-drc`` mode — so plain and MCTS boards are scored identically
    against the board's native ``.kicad_pro`` rules. Returns the eval metric dict
    (final_potential / routability / drv_errors_only_count / clean_pass / ...),
    tagged with ``rollout_idx`` for the best@k tie-break in ``selection_key``.
    """
    row = wrapper.eval_inline_drc(reward_config_name=reward_cfg, check_angle=check_angle)
    row["rollout_idx"] = rollout_idx
    return row


def _fmt_row(row: dict) -> dict:
    """Compact print view of an eval score row (shared by plain & MCTS)."""
    ex = row.get("extras", {}) or {}

    def _i(v):
        return int(v) if isinstance(v, (int, float)) and v == v else -1

    def _f(v, nd):
        return round(float(v), nd) if isinstance(v, (int, float)) and v == v else float("nan")

    return {
        "unrouted": _i(ex.get("unrouted_edges_remaining")),
        "rout": _f(row.get("routability"), 3),
        "tracks": _i(row.get("track_count")),
        "vias": _i(row.get("via_count")),
        "wl": _f(row.get("wirelength_mm"), 1),
        "drv_err": _i(row.get("drv_errors_only_count")),
        "drv_prom": _i(row.get("drv_errors_and_promoted_count")),
        "clean": bool(row.get("clean_pass", False)),
        "phi": _f(row.get("final_potential"), 2),
    }


def plain_rollout_once(wrapper, agent, device, *, deterministic, cap,
                       fthr, gthr, reward_cfg, check_angle, rollout_idx):
    """One plain policy rollout on the SAME wrapper the MCTS path uses.

    Reuses the canonical per-step decision loop ``iter_rollout`` (the exact
    act+mask+step primitive shared by eval and PPO) over a single-wrapper list,
    with the same FinishNoProgress / NoGeometryProgress early-stop guards, then
    scores the routed board with ``score_live``. No re-implemented greedy loop,
    no subprocess pool — same env, same scoring path as MCTS.
    """
    obs, _ = wrapper.reset()
    obs_by_slot = {0: obs}
    done = {0: False}
    fguard = FinishNoProgressGuard(threshold=fthr) if fthr > 0 else None
    gguard = NoGeometryProgressGuard(threshold=gthr) if gthr > 0 else None
    total_r = 0.0
    steps = 0
    term = trunc = False
    for batch in iter_rollout([wrapper], agent, device, obs_by_slot, [0], done,
                              want_value=False, deterministic=deterministic,
                              max_steps=cap):
        total_r += float(batch.rewards[0])
        steps += 1
        info = batch.infos[0]
        if os.environ.get("MCTS_LOG_ACTIONS"):
            _a = batch.actions[0]
            print(f"    plain d{steps-1}: a={list(_a) if hasattr(_a,'__iter__') else _a} "
                  f"u={info.get('unrouted_count', info.get('unrouted','?'))}", flush=True)
        term = bool(batch.terminateds[0])
        trunc = _guard_truncate(fguard, gguard, batch.actions[0], info,
                                term, bool(batch.truncateds[0]))
        if term or trunc:
            done[0] = True
    row = score_live(wrapper, reward_cfg, check_angle, rollout_idx=rollout_idx)
    row["episode_return"] = round(total_r, 4)
    row["steps"] = steps
    row["terminated"] = term
    row["truncated"] = trunc
    return row


def run_plain(wrapper, agent, device, args, *, reward_cfg, check_angle, cap,
              n=None, time_budget=None):
    """Best@k plain rollouts, selected by the eval ``selection_key``.

    Repeats plain rollouts on ``wrapper`` until a fixed count (``n``) OR a
    wallclock budget (``time_budget`` seconds, ≥1 rollout guaranteed) is hit,
    then picks the single winner with ``select_best_in_board`` — exactly the
    per-board best@k the uniform eval performs in aggregation. A greedy
    (``--deterministic``) plain collapses to best@1 (repeats would be identical).
    Returns ``(winner_row, candidate_count, elapsed_sec)``.
    """
    fthr = int(getattr(args, "early_stop_finish_no_progress", 0))
    gthr = int(getattr(args, "early_stop_no_geometry_progress", 0))
    torch.manual_seed(args.seed)      # reproducible sampling stream across rollouts
    np.random.seed(args.seed)
    rows: list[dict] = []
    t0 = time.perf_counter()
    i = 0
    while True:
        row = plain_rollout_once(
            wrapper, agent, device, deterministic=args.deterministic, cap=cap,
            fthr=fthr, gthr=gthr, reward_cfg=reward_cfg, check_angle=check_angle,
            rollout_idx=i,
        )
        rows.append(row)
        i += 1
        elapsed = time.perf_counter() - t0
        print(f"    plain r{row['rollout_idx']}: {_fmt_row(row)} ({elapsed:.1f}s cum)",
              flush=True)
        if args.deterministic:                     # best@1 — repeats are identical
            break
        if n is not None and i >= n:
            break
        if time_budget is not None and elapsed >= time_budget:
            break
        if n is None and time_budget is None:      # safety: never infinite
            break
    _, winner = select_best_in_board(rows, args.selection_mode)
    return winner, len(rows), time.perf_counter() - t0


def _guard_truncate(fguard, gguard, action_arr, info, term, trunc):
    """Apply the canonical finish/geometry early-stop guards to one committed
    step (mirror of _run_one_batch). Returns possibly-updated ``trunc``."""
    if (fguard or gguard) and not term and not trunc:
        info = dict(info)
        an = (TRACE_ACTION_NAMES[int(action_arr[0])]
              if 0 <= int(action_arr[0]) < len(TRACE_ACTION_NAMES) else "unknown")
        f_trig = (fguard.update(action_name=an,
                                action_class=str(info.get("action_class", "")),
                                info=info) if fguard else False)
        g_trig = gguard.update(info=info) if gguard else False
        if f_trig or g_trig:
            return True
    return trunc


def _sample_prior_action(legal, priors, rng):
    """One action sampled ∝ prior — the deferral used by ``--prior-net-select``.

    Sampling, not argmax: with q̂ tied the Gumbel root's own final pick is
    ``argmax(gumbel + logit)``, which IS an exact sample from softmax(logit), so
    sampling reproduces the behaviour the search already has there (and keeps the
    across-seed diversity the best-Φ selection lives on).
    """
    w = [max(float(priors.get(a, 0.0)), 0.0) for a in legal]
    total = sum(w)
    if total <= 0.0:
        return legal[rng.randrange(len(legal))]
    x = rng.random() * total
    acc = 0.0
    for a, wi in zip(legal, w):
        acc += wi
        if x < acc:
            return a
    return legal[-1]


def mcts_rollout(wrapper, agent, args, scale, cap, reward_cfg, check_angle,
                 critic_offset=0.0, critic_lambda=1.0):
    torch.manual_seed(args.seed)        # reproducible prior sampling (SamplingPolicyValue)
    np.random.seed(args.seed)
    # ONE noise RNG for the whole rollout (canonical AlphaZero/MuZero/mctx
    # semantics): Dirichlet/Gumbel exploration noise is drawn FRESH per decision,
    # while the rollout stays fully reproducible via the single seed. Without
    # this, run_search re-inits Random(cfg.seed) per decision, so every decision
    # re-draws the SAME noise sequence — a fixed positional bias on the sorted
    # child order rather than exploration noise.
    search_rng = random.Random(args.seed)
    wrapper.reset()
    # Count DRC passes during the whole rollout (return_bootstrap never reads Φ at a
    # leaf — DRC enters only via the per-step ΔΦ the env.step computes).
    eng = wrapper.env._engine
    drc_n = [0]
    _orig_drc = eng.run_drc
    def _counted(*a, **k):
        drc_n[0] += 1
        return _orig_drc(*a, **k)
    eng.run_drc = _counted
    se = RLSearchEnv(
        wrapper, prefilter_refused=bool(getattr(args, "prefilter_refused", False)))
    pv = LogitPolicyValue(agent, wrapper)   # sampler-exact deterministic logit prior
    iso = getattr(args, "critic_isotonic", None)
    if iso:
        # Wrap BEFORE the memo so the calibrated value is what gets cached.
        kx, ky, _lam = load_isotonic_calibrator(iso)
        pv = CalibratedPolicyValue(pv, kx, ky)
    # Cross-decision prior/value memo: the wrapper rebuilds its candidate pool FROM
    # the obs on every restore, so the prior/value is a pure function of the obs —
    # caching it by obs fingerprint is bit-identical to cold search and only skips
    # the redundant policy forward (committed-subtree states recur across decisions).
    if not getattr(args, "no_pv_cache", False):
        pv = MemoizingPolicyValue(pv)
    if args.algo in ("alphazero", "muzero", "gumbel"):
        # A named profile fixes the module axes (root selection / noise /
        # value-completion / gumbel knobs) as one coherent set; the tunables below
        # layer on top. --dirichlet OVERRIDES the profile's own dirichlet_alpha
        # (e.g. --algo muzero --dirichlet 0 for a no-noise ablation); omitting it
        # keeps the profile default.
        algo_kwargs = dict(n_simulations=args.n_sim, critic_scale=scale,
                           gamma=args.gamma, max_depth=args.max_depth,
                           pw_alpha=args.pw_alpha, pw_base=args.pw_base,
                           invalid_mode=args.invalid_mode,
                           invalid_penalty=args.invalid_penalty,
                           seed=args.seed)
        if args.dirichlet is not None:
            algo_kwargs["dirichlet_alpha"] = args.dirichlet
        if args.gumbel_value_scale is not None:
            algo_kwargs["gumbel_value_scale"] = args.gumbel_value_scale
        if args.gumbel_scale is not None:
            algo_kwargs["gumbel_scale"] = args.gumbel_scale
        cfg = make_algorithm(args.algo, **algo_kwargs).config(
            critic_offset=critic_offset,
            critic_lambda=critic_lambda)
    else:  # "puct": raw generic flat-c_puct config (root_selection defaults to puct)
        cfg = MctsConfig(n_simulations=args.n_sim, critic_scale=scale,
                         gamma=args.gamma,
                         dirichlet_alpha=args.dirichlet if args.dirichlet is not None else 0.0,
                         max_depth=args.max_depth, seed=args.seed,
                         pw_alpha=args.pw_alpha, pw_base=args.pw_base,
                         invalid_mode=args.invalid_mode,
                         invalid_penalty=args.invalid_penalty,
                         gumbel_max_considered=args.gumbel_max_considered,
                         **({} if args.gumbel_value_scale is None
                            else {"gumbel_value_scale": args.gumbel_value_scale}),
                         **({} if args.gumbel_scale is None
                            else {"gumbel_scale": args.gumbel_scale}),
                         critic_offset=critic_offset,
                         critic_lambda=critic_lambda,
                         value_completion=getattr(args, "value_completion", False))
    se._search_diag = {} if getattr(args, "search_diag", False) else None
    # Same committed-trajectory guards as the greedy/eval rollout (parity).
    fthr = int(getattr(args, "early_stop_finish_no_progress", 0))
    gthr = int(getattr(args, "early_stop_no_geometry_progress", 0))
    fguard = FinishNoProgressGuard(threshold=fthr) if fthr > 0 else None
    gguard = NoGeometryProgressGuard(threshold=gthr) if gthr > 0 else None
    arr = None
    prior_nsel = bool(getattr(args, "prior_net_select", False))
    for d in range(cap):
        t = time.perf_counter()
        deferred = False
        legal = list(se.legal_actions()) if prior_nsel else []
        if legal and all(int(a[0]) == ACT_NET_SELECT for a in legal):
            # Every legal action is a net_select ⇒ every sibling edge is ΔΦ = 0.
            # Sample ∝ prior, which is what the Gumbel root does anyway once q̂ ties.
            priors, _v = pv(se.observe(), legal)
            action, deferred = _sample_prior_action(legal, priors, search_rng), True
        else:
            action, _visits = run_search(se, pv, cfg, rng=search_rng)
        if action is None:
            u = wrapper.env._engine.get_unrouted_count()
            print(f"    mcts d{d}: STOP (no improving action) u={u} "
                  f"({time.perf_counter()-t:.1f}s)", flush=True)
            break
        # Commit: marks the step as part of the real episode (episode-level
        # best-Φ / ratsnest tracking counts only these, not search simulations).
        res = se.step(action, committed=True)
        if isinstance(pv, MemoizingPolicyValue):
            # The commit makes every sibling subtree unreachable — keep only the
            # boards this decision touched (the committed subtree recurs next).
            pv.new_generation()
        arr = np.asarray(action, dtype=np.int64)
        u = wrapper.env._engine.get_unrouted_count()
        trunc = _guard_truncate(fguard, gguard, arr, res.info,
                                res.done, False)
        dtail = ""
        # A deferred decision ran no search, so the diag still holds the PREVIOUS
        # decision's counters — printing it would attribute them to this one.
        sd = None if deferred else getattr(se, "_search_diag", None)
        if sd:
            lv = sd.get("leaf_vals") or [0.0]
            top = " ".join(
                f"{a}:N{n}/Q{q}/P{p}" for (a, n, q, p) in sd.get("top", [])
            )
            pop = sd.get("popped")
            _pnm = {0: "nsel", 1: "sroute", 2: "NEND", 3: "mline", 4: "mvia", 5: "FIN"}
            pop_s = ("  POPPED=" + " ".join(f"{_pnm.get(k,k)}x{v}"
                     for k, v in sorted(pop.items()))) if pop else ""
            dtail = (f"\n       Φroot={sd.get('phi_root', 0.0):.3f} "
                     f"term={sd.get('terminals', 0)} SOLVED={sd.get('solved', 0)} "
                     f"cap={sd.get('caps', 0)} exp={sd.get('expands', 0)} "
                     f"leafV[{min(lv):.3f},{max(lv):.3f}] top=[{top}]{pop_s}"
                     f"\n       RC={sd.get('root_children', [])}")
        print(f"    mcts d{d}: a={list(action)} u={u} "
              f"{'PRIOR ' if deferred else ''}"
              f"({time.perf_counter()-t:.1f}s){dtail}", flush=True)
        if res.done or trunc or u == 0:
            break
    if isinstance(pv, MemoizingPolicyValue):
        print(f"    [pv cache: {pv.hits} hits / {pv.misses} forwards "
              f"({100*pv.hits/max(pv.hits+pv.misses,1):.0f}% saved), "
              f"{len(pv)} live / {pv.dropped} dropped]", flush=True)
    eng.run_drc = _orig_drc          # unwrap before scoring (its DRC shouldn't count)
    print(f"    [DRC passes during MCTS rollout: {drc_n[0]}]", flush=True)
    # Score the routed board through the SAME native-.kicad_pro eval scorer the
    # plain path (and the uniform eval) use, so every reported number is comparable.
    m = score_live(wrapper, reward_cfg, check_angle, rollout_idx=0)
    return m


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cpu")
    policy, ckpt_args, _it = load_policy_from_ckpt(Path(args.ckpt), device)
    agent = KiCadRLAgent(policy, device=device, deterministic=True)
    max_steps = int(ckpt_args.get("max_steps", 200))
    env_kwargs = env_kwargs_from_checkpoint(ckpt_args, max_steps)
    for k in ("board_path", "board_paths", "boards", "n_envs", "num_envs", "group_n"):
        env_kwargs.pop(k, None)
    # Align net-selection with the policy (what it was TRAINED with) so plain and
    # MCTS run the SAME MDP — bool(None)→False would otherwise auto-advance nets.
    env_kwargs["policy_net_select"] = bool(getattr(policy, "policy_net_select", False))
    # The env factories take no default ``seed`` (a silent 0 cannot be told apart
    # from an intentional one), so this single-env analysis tool pins it.
    env_kwargs["seed"] = 0
    if args.reward_rule:                                  # env Φ/value rule the search optimizes
        print(f"reward rule override: {env_kwargs.get('reward_rule')} -> {args.reward_rule}")
        env_kwargs["reward_rule"] = args.reward_rule
    if args.search_via_penalty is not None:
        env_kwargs["via_penalty"] = float(args.search_via_penalty)
        print(f"search via_penalty override: -> {args.search_via_penalty} "
              f"(scoring unchanged)")
    if args.masking_rule:                                 # action masking
        print(f"masking rule override: {env_kwargs.get('masking_rule')} -> {args.masking_rule}")
        env_kwargs["masking_rule"] = args.masking_rule
    cap = args.ep_cap if args.ep_cap and args.ep_cap > 0 else max_steps
    # Eval knobs inherit the checkpoint's TRAINING values unless overridden on the
    # CLI: corner_mode (engine code) rides in from ``env_kwargs`` (built from the
    # ckpt), and check_angle takes the ckpt's stored value if any, else derives from
    # the (ckpt-inherited) corner mode. --corner-mode / --check-angle win.
    if args.corner_mode is None:
        args.corner_mode = env_kwargs.get("corner_mode")
    if args.check_angle is None and ckpt_args.get("check_angle") is not None:
        args.check_angle = int(ckpt_args["check_angle"])
    gamma = float(ckpt_args.get("gamma", 0.995))   # ckpt TRAINING gamma (reference)
    # Search discount γ (return_bootstrap) is a SEARCH REGULARIZER, a different role
    # from the training γ — default to the strong DEFAULT_SEARCH_GAMMA (not the ckpt
    # γ, which is near-inert over the short lookahead and fails to complete).
    if args.gamma is None:
        args.gamma = DEFAULT_SEARCH_GAMMA
    print(f"search gamma = {args.gamma:.4f} "
          f"(search regularizer; ckpt training gamma={gamma:.4f}, different role)")
    do_plain = args.mode in ("plain", "both")
    do_mcts = args.mode in ("mcts", "both")

    # Final scoring is eval-identical for BOTH paths: native .kicad_pro rules via
    # the wrapper's eval_inline_drc, one scoring reward-config + check-angle.
    score_cfg = args.scoring_reward_rule
    check_angle = _derive_check_angle(args)
    print(f"loaded policy (max_steps={max_steps}, commit_cap={cap}, "
          f"policy_net_select={env_kwargs.get('policy_net_select')})")
    print(f"scoring: reward_config={score_cfg}, check_angle={check_angle}, "
          f"selection={args.selection_mode}, "
          f"plain={'greedy@1' if args.deterministic else 'stochastic best@k'}")

    # critic_scale (reward-norm std mapping V_critic → raw Φ units). DEFAULT =
    # empirical calibration with the loaded policy on a throwaway env (closed
    # before the live env — engine singleton); --critic-scale-source ckpt opts
    # into the exact saved reward_normalizer_state std instead. MCTS-value only.
    #
    # Measured PER BOARD: the calibration is a property of (policy, board) — the
    # same estimator returns +8.8 on one checkpoint and -0.4 on another, and a
    # negative scale inverts the value ordering outright. One board's number is
    # not evidence about the next board's.
    def _board_critic_scale(board):
        """(critic_scale, critic_offset) for this board.

        The pair is one affine calibration of a critic trained on normalized
        rewards: ``boot = scale*(V_tilde - offset)``. ``scale`` is the free-intercept
        regression slope of the realized return on V_tilde (it also acts as a trust
        weight — least squares attenuates it when V ranks states poorly), and
        ``offset`` is the terminal anchor, the V_tilde a completed board reads, which
        must map to zero remaining return. Either can be pinned from the CLI."""
        if not do_mcts:
            return 1.0, 0.0, 1.0
        if getattr(args, "critic_isotonic", None):
            # The value the search receives is ALREADY calibrated (the PolicyValue
            # was wrapped), so the downstream affine must be the identity: the
            # bootstrap is lambda*1*(g(V) - 0). lambda stays the trust knob and
            # falls back to the value stored beside the knots.
            _kx, _ky, _lam = load_isotonic_calibrator(args.critic_isotonic)
            tr_ = (float(args.critic_lambda) if args.critic_lambda is not None
                   else (float(_lam) if _lam is not None else 1.0))
            print(f"  critic = isotonic map ({_kx.size} knots, "
                  f"range {_ky[0]:.3f}..{_ky[-1]:.3f})  λ = {tr_:.3f}  "
                  f"(value = Φ + λ·g(V_critic))", flush=True)
            return 1.0, 0.0, tr_
        if args.critic_scale is not None:
            off = float(args.critic_offset)
            print(f"  critic_scale = {args.critic_scale:.3f}  offset = {off:.3f}  "
                  f"(fixed override; value = Φ + {args.critic_scale:.3f}·"
                  f"(V_critic − {off:.3f}))", flush=True)
            return float(args.critic_scale), off, (
                1.0 if args.critic_lambda is None else float(args.critic_lambda))
        s_, off_, tr_, src_ = resolve_critic_scale(
            agent, args.ckpt,
            lambda: _make_env(board, env_kwargs, args.corner_mode),
            gamma=gamma, source=args.critic_scale_source,
            n_rollouts=args.critic_scale_rollouts,
            ratsnest_patience=int(args.early_stop_ratsnest),
        )
        if args.critic_offset:            # CLI offset wins over the measured anchor
            off_ = float(args.critic_offset)
        print(f"  critic_scale = {s_:.3f}  offset = {off_:.3f}  λ = {tr_:.3f}  ({src_}; "
              f"value = Φ + {s_:.3f}·V_critic)", flush=True)
        if args.critic_lambda is not None:      # CLI wins over the derived lambda
            tr_ = float(args.critic_lambda)
        return s_, off_, tr_

    plain: dict[str, tuple] = {}     # board -> (winner_row, candidate_count, elapsed)
    mcts: dict[str, dict] = {}
    mcts_t: dict[str, float] = {}

    # ONE main-process env per board, reused by MCTS then the time-matched plain
    # (single engine, sequential reset — no subprocess pool, so no fork/singleton
    # hazard and both paths share the identical env + native-.kicad_pro scoring).
    for board in args.boards:
        dir_display = args.dirichlet if args.dirichlet is not None else "profile-default"
        _inv = (args.invalid_mode if args.invalid_mode == "pop"
                else f"penalize({args.invalid_penalty})")
        print(f"\n=== {Path(board).name} (mode={args.mode}, algo={args.algo or '-'}, "
              f"n_sim={args.n_sim}, dir={dir_display}, gamma={args.gamma}, "
              f"invalid={_inv}, corner={args.corner_mode}) ===")
        # Calibrate BEFORE the live env: the throwaway calibration env is
        # closed inside resolve_critic_scale, and the router is a per-process
        # singleton — never two live engines.
        scale, coffset, ctrust = _board_critic_scale(board)
        w = _make_env(board, env_kwargs, args.corner_mode)
        # Opt-in env-side early-stop + best-Φ-board selection (applies to both the
        # MCTS and plain rollout on this board via their reset()). Set before the
        # rollout's reset so tracking initialises for this episode.
        w.env._ratsnest_patience = int(args.early_stop_ratsnest)
        w.env._output_best_board = bool(args.output_best_board)
        # Both phases share one env, so the plain phase inherits whatever the
        # MCTS phase was given unless it is re-set between them (below).
        plain_es = (int(args.plain_early_stop_ratsnest)
                    if args.plain_early_stop_ratsnest is not None
                    else int(args.early_stop_ratsnest))
        try:
            budget = None
            if do_mcts:
                t = time.perf_counter()
                mcts[board] = mcts_rollout(w, agent, args, scale, cap, score_cfg,
                                           check_angle, critic_offset=coffset,
                                           critic_lambda=ctrust)
                budget = time.perf_counter() - t
                mcts_t[board] = budget
                print(f"  mcts  {Path(board).name}: {_fmt_row(mcts[board])}  ({budget:.1f}s)",
                      flush=True)
                if args.save_mcts:                           # persist routed board
                    out = args.save_mcts if len(args.boards) == 1 else str(
                        Path(args.save_mcts).with_name(
                            f"{Path(board).stem}_{Path(args.save_mcts).name}"))
                    w.save_pcb(out)                          # PCBWorld adapter → engine.save
                    print(f"  saved MCTS-routed board -> {out}  (+ .kicad_pro)", flush=True)
            if do_plain:
                # Re-set AFTER the MCTS phase and BEFORE run_plain's first reset,
                # so the two phases can run under different early-stop settings.
                if plain_es != int(args.early_stop_ratsnest):
                    w.env._ratsnest_patience = plain_es
                    print(f"  plain early-stop-ratsnest = {plain_es} "
                          f"(mcts used {int(args.early_stop_ratsnest)})", flush=True)
                # both → best@k within the MCTS wallclock budget (compute-matched);
                # plain-only → fixed --n-rollouts best@k (eval-style).
                winner, cnt, el = run_plain(
                    w, agent, device, args, reward_cfg=score_cfg,
                    check_angle=check_angle, cap=cap,
                    n=(None if do_mcts else args.n_rollouts),
                    time_budget=(budget if do_mcts else None),
                )
                plain[board] = (winner, cnt, el)
                print(f"  plain {Path(board).name}: {_fmt_row(winner)}  "
                      f"(best@{cnt}, {el:.1f}s)", flush=True)
        finally:
            if w.env._engine.is_routing():
                w.env._engine.cancel_route()
            w.env.close()

    # --- Summary ---------------------------------------------------------------
    print("\n===== SUMMARY =====")
    for board in args.boards:
        name = Path(board).name
        pr = plain.get(board)
        mm = mcts.get(board)
        if pr is not None:
            winner, cnt, el = pr
            print(f"  {name}  plain(best@{cnt}): {_fmt_row(winner)}   ({el:.1f}s)")
        if mm is not None:
            print(f"  {name}  mcts          : {_fmt_row(mm)}   ({mcts_t[board]:.1f}s)")
        if pr is not None and mm is not None:
            pw, mw = _fmt_row(pr[0]), _fmt_row(mm)
            du = mw["unrouted"] - pw["unrouted"]
            dderr = mw["drv_err"] - pw["drv_err"]
            dphi = (mw["phi"] - pw["phi"]
                    if mw["phi"] == mw["phi"] and pw["phi"] == pw["phi"] else float("nan"))
            print(f"  {name}  Δunrouted(mcts-plain)={du:+d} (neg = MCTS routed more)  "
                  f"Δdrv_err={dderr:+d}  Δphi={dphi:+.2f}  "
                  f"clean: plain={pw['clean']} mcts={mw['clean']}")


if __name__ == "__main__":
    main()
