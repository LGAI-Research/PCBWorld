"""PCBWorld — KiCad Gymnasium routing environment.

Exposes 6 routing actions with stateless action masking
derived from observable engine state (no phase state machine).

Observations are hierarchical JSON dicts suitable for both LLM agents
and RL policies (via flattening/tokenization).

Features:
- 6 routing actions: net_select, start_route, net_end, make_line, make_via, finish
- Action masking from observable state (has_net, is_routing, net_fully_connected)
- JSON observation: board_static (static) + routing_geometry (dynamic) + router_head
"""

from __future__ import annotations

import copy
import faulthandler
import logging
import time
import weakref
from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from configs.loader.schema import EnvConfig

# Enable faulthandler so SIGSEGV prints a Python traceback even in ad-hoc
# processes. Skip when already enabled — pcb_world.diag's crash handler routes
# it to a crashlog file, and re-enabling here would retarget it to stderr.
if not faulthandler.is_enabled():
    faulthandler.enable()

from pcb_world.diag import CrashLogger
from pcb_world.engine import KiCadEngine
from pcb_world.engine.drc import DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED
from pcb_world.engine.drc_config import (
    apply_default_drc_if_fallback,
    compute_hardest_per_netclass,
)
from pcb_world.engine.pcb_file_parser import parse_pcb_file
from pcb_world.core.masking import (
    ACT_FINISH,
    ACT_IDLE,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACTION_NAMES,
    NUM_ACTIONS,
    MaskContext,
    get_masking_rule,
)
from pcb_world.core.action import ActionDispatcher
from pcb_world.core.observation import (
    BoardStatic,
    NetGeometry,
    build_json_observation,
    build_net_geometry,
    _build_board_static,
)
from pcb_world.core.indexed_obs import (
    build_indexed_observation,
    static_tables_from_dict,
)
from pcb_world.core.reward import PotentialReward, RewardState
from pcb_world.core.reward_config import get_reward_config

logger = logging.getLogger(__name__)


def _make_history_record(
    action: dict[str, Any], action_type: int, success: bool,
    net_id: int | None = None,
) -> dict:
    """Build one action-history record emitted in subsequent observations.

    ``net_id`` is the net this action pertained to: the ``net_id`` param for
    net_select, else the net selected when the action was dispatched (net_end
    records the net it closed — capture BEFORE dispatch deselects). None = no
    net context (idle, or nothing selected).
    """
    return {
        "action_type": int(action_type),
        "pointer_xy": [
            float(action.get("x_mm", 0.0)),
            float(action.get("y_mm", 0.0)),
        ],
        "pointer_layer": int(action.get("layer", 0)),
        "routing_mode": int(action.get("routing_mode", -1)),
        "has_pointer": bool(
            action_type in (ACT_START_ROUTE, ACT_MAKE_LINE, ACT_MAKE_VIA)
        ),
        "success": bool(success),
        "net_id": int(net_id) if net_id is not None else None,
    }


def _action_net_context(
    action: dict[str, Any], action_type: int, current_net_id: int | None,
) -> int | None:
    """Net an action pertains to, for the history record (see above)."""
    if action_type == ACT_NET_SELECT:
        nid = action.get("net_id")
        return int(nid) if nid is not None else None
    return current_net_id


# ---------------------------------------------------------------------------
# Internal state containers
# ---------------------------------------------------------------------------

def _potential_uses_drc(pot: PotentialReward) -> bool:
    """Return True iff the potential reads DRC state (penalty or clean bonus).

    When False, the env can skip every DRC engine call (step + episode-end
    logging) since DRC contributes nothing to reward.
    """
    if (
        getattr(pot, "net_clean_bonus", 0.0) > 0
        or getattr(pot, "clean_completion_bonus", 0.0) > 0
    ):
        return True
    shape = getattr(pot, "drc_shape", "linear")
    if shape == "log_per_net":
        return (
            getattr(pot, "drc_log_scale", 0.0) > 0
            or getattr(pot, "drc_log_agg_scale", 0.0) > 0
        )
    return getattr(pot, "drc_penalty", 0.0) > 0


@dataclass
class RewardTracker:
    """Tracks step-to-step state for the per_step potential reward."""
    fn: PotentialReward
    run_drc_on_reset: bool = False
    prev_state: RewardState | None = None

    def reset(self, engine: KiCadEngine) -> None:
        snap = engine.get_reward_snapshot(run_drc=self.run_drc_on_reset)
        self.prev_state = RewardState.from_snapshot(snap)


@dataclass
class ObsCache:
    """Cached observation components (refreshed every step)."""
    board_static: dict = field(default_factory=dict)
    net_geometry: dict[int, NetGeometry] = field(default_factory=dict)
    # indexed_v1 static table group (built once from ``board_static`` when
    # obs_format="indexed"; shared by reference across every step's obs).
    static_tables: dict | None = None


@dataclass
class Checkpoint:
    """MCTS checkpoint of full env state.

    The heavy router state (board + engine config + routing session) lives
    C++-side behind ``engine_handle`` (an opaque router-local int). The rest are
    light Python episode scalars held directly by the MCTS node. ``routing_mode``
    is a scalar because the engine exposes no read-back getter — it is
    dispatcher-owned and read by the observation.
    """
    engine_handle: int
    step_count: int
    current_net_id: int | None
    routing_mode: int
    # obs-facing recent-action records, newest first (branch-agnostic)
    action_history: tuple = ()
    # Path-dependent Python EPISODE state the engine handle does NOT cover. Without
    # these, an MCTS restore leaves the previous simulation's values in place, so a
    # net_end explored inside a simulation leaks into the committed episode:
    #   episode_closed_nets → drives all_nets_closed termination + net masking + obs
    #                         (leak ⇒ episode ends early / nets vanish from the mask),
    #   reward_prev_state    → the ΔΦ (potential_diff) baseline for the next step,
    #   wire_via_ref_state   → the on_net_end wire/via accumulation baseline.
    # Copied on capture AND restore so a restored env never aliases the node's set.
    episode_closed_nets: set = field(default_factory=set)
    reward_prev_state: Any = None
    wire_via_ref_state: Any = None
    # RAII backstop: weakref to the owning engine so a dropped / leaked
    # checkpoint still frees its C++ clones on GC. Excluded from equality / repr.
    _engine_ref: Any = field(default=None, compare=False, repr=False)
    _released: bool = field(default=False, compare=False, repr=False)

    def release(self) -> None:
        """Free the C++ checkpoint handle. Idempotent, and safe after the engine
        is closed (the router frees all handles on close)."""
        if self._released:
            return
        eng = self._engine_ref() if self._engine_ref is not None else None
        if eng is not None and getattr(eng, "_r", None) is not None:
            eng.release_checkpoint(self.engine_handle)
        self._released = True

    def __del__(self) -> None:
        # Backstop: a checkpoint dropped without an explicit release still frees
        # its C++ clones when garbage-collected.
        try:
            self.release()
        except Exception:
            pass


class PCBWorld(gym.Env):
    """KiCad routing environment with stateless action masking.

    Action masking is derived from observable state (has_net, is_routing,
    net_fully_connected) — no separate phase state machine.

    Args:
        board_path: Path to .kicad_pcb file.
        max_steps: Maximum steps per episode.
        masking_rule: Name of the masking rule to use (e.g. "strict", "relaxed").
        render_mode: "rgb_array" or None.
        use_yaml_drc_fallback: When the board has no companion ``.kicad_pro``
            AND no legacy setup tokens (the "real default" fallback case),
            True substitutes ``drc_config_path``'s global minima into BDS
            (with a ``UserWarning``); ``drc_config_path`` is **required**
            in that case — there is no implicit default YAML. False
            (default) raises ``ValueError`` — every dataset we ship should
            already carry rules, so reaching the fallback path is treated
            as a load failure rather than silently routing on KiCad
            compile-time defaults. Authoritative loads (pro or legacy
            setup) are never touched and never trigger either path.
        drc_config_path: Explicit YAML supplying the fallback global minima
            when ``use_yaml_drc_fallback`` is True. No implicit default
            (e.g. ``configs/drc/default.yaml``).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        board_path: str,
        max_steps: int = 200,
        masking_rule: str = "default",
        render_mode: str | None = None,
        reward_rule: str = "drc_only_dense",
        reward_noise_std: float = 0.0,
        emit_drc_tokens: bool = True,
        via_penalty: float | None = None,
        wirelength_penalty: float | None = None,
        drc_penalty: float | None = None,
        drc_log_scale: float | None = None,
        drc_log_agg_scale: float | None = None,
        drc_log_offset: float | None = None,
        reward_step_penalty: float | None = None,
        wire_via_emission: str | None = None,
        corner_mode: int = 0,
        use_yaml_drc_fallback: bool = False,
        drc_config_path: str | None = None,
        env_config: "EnvConfig | None" = None,
        engine_seed: int | None = 77,
        seed: int | None = None,
        shove_iter_limit: int = 250,
        followbranch_iter_limit: int = 1_000_000,
        reject_if_stuck: bool = True,
        simplify_outline: bool = False,
        obs_format: str = "json",
        outline_obs: str = "tess",
        action_history_len: int = 1,
        net_constraint_obs: bool = False,
        early_stop_ratsnest_patience: int = 0,
        output_best_board: bool = False,
        target_nets: "set[int] | frozenset[int] | list[int] | None" = None,
        preserve_nontarget_routing: bool = True,
        keep_routing_fraction: "tuple[float, float] | list[float] | None" = None,
        **unexpected: object,
    ) -> None:
        super().__init__()

        # Two known-removed kwargs get a tailored message; anything else is
        # rejected outright (a typo'd kwarg silently swallowed here is exactly
        # the fallback class the factory layer already refuses).
        if "use_drc" in unexpected:
            import warnings
            warnings.warn(
                "PCBWorld.use_drc was removed. DRC is now always included "
                "in the potential (weight controlled by drc_penalty); DRC "
                "runs every step in mode='per_step', only at episode end in "
                "mode='terminal'. The argument is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        if "config" in unexpected:
            raise TypeError(
                "PCBWorld.config (untyped override dict) was removed. Its keys "
                "were silently dropped on a typo or wrong layer, so it is a hard "
                "error rather than a no-op: pass an EnvConfig via env_config=, or "
                "the explicit keyword args directly."
            )
        if unknown := sorted(set(unexpected) - {"use_drc", "config"}):
            raise TypeError(
                f"PCBWorld got unknown kwarg(s): {unknown} — unknown arguments "
                "are always rejected (factory-level names belong to "
                "make_decoder_env/make_decoder_env_pool, not the env core)"
            )

        # Canonical env-core config. When given, it is the single source for
        # the env-core kwargs (the RL factory and LLM worker construct via it);
        # individual kwargs above act as the field defaults / legacy path.
        if env_config is not None:
            max_steps = env_config.max_steps
            masking_rule = env_config.masking_rule
            reward_rule = env_config.reward_rule
            reward_noise_std = env_config.reward_noise_std
            emit_drc_tokens = env_config.emit_drc_tokens
            via_penalty = env_config.reward.via_penalty
            wirelength_penalty = env_config.reward.wirelength_penalty
            drc_penalty = env_config.reward.drc_penalty
            drc_log_scale = env_config.reward.drc_log_scale
            drc_log_agg_scale = env_config.reward.drc_log_agg_scale
            drc_log_offset = env_config.reward.drc_log_offset
            reward_step_penalty = env_config.reward.reward_step_penalty
            wire_via_emission = env_config.reward.wire_via_emission
            corner_mode = env_config.corner_mode
            use_yaml_drc_fallback = env_config.use_yaml_drc_fallback
            drc_config_path = env_config.drc_config_path
            engine_seed = env_config.engine_seed
            shove_iter_limit = env_config.shove_iter_limit
            followbranch_iter_limit = env_config.followbranch_iter_limit
            reject_if_stuck = env_config.reject_if_stuck
            simplify_outline = env_config.simplify_outline
            obs_format = env_config.obs_format
            outline_obs = env_config.outline_obs
            action_history_len = env_config.action_history_len
            net_constraint_obs = env_config.net_constraint_obs
            keep_routing_fraction = env_config.keep_routing_fraction

        if action_history_len < 1:
            raise ValueError(
                f"action_history_len must be >= 1, got {action_history_len}"
            )

        if obs_format not in ("json", "indexed"):
            raise ValueError(
                f"obs_format must be 'json' or 'indexed', got {obs_format!r}"
            )

        if outline_obs not in ("poly16", "tess", "arc"):
            raise ValueError(
                f"outline_obs must be 'poly16', 'tess' or 'arc', got {outline_obs!r}"
            )

        if corner_mode not in (0, 1, 2, 3):
            raise ValueError(
                f"corner_mode must be in {{0,1,2,3}} "
                f"(0=MITERED_45 default, 2=MITERED_90 no-diagonals), got {corner_mode}"
            )

        # --- Config (immutable after init) ---
        self.board_path = board_path
        self.max_steps = max_steps
        self.masking_rule = masking_rule
        self.render_mode = render_mode
        self._reward_noise_std = reward_noise_std
        self._emit_drc_tokens = emit_drc_tokens
        self._corner_mode = corner_mode
        self._obs_format = obs_format
        self._outline_obs = outline_obs
        # Per-net DRC constraint observation (NET-token tw/cl/vd channels):
        # when on, every ``board_static`` net carries its resolved netclass
        # values (see _fill_net_constraint_obs). Off (default) keeps the
        # observation byte-identical to pre-knob checkpoints.
        self._net_constraint_obs = bool(net_constraint_obs)
        # Net-subset (partial routing): restrict the problem to these net codes.
        # None = route every net (legacy). When set, only these nets appear as
        # routable targets (board_static.nets), carry ratsnest/routing_geometry,
        # count toward unrouted/termination, and gate DRC violations; other nets'
        # pads remain physical obstacles the router clears from.
        self._target_nets: frozenset[int] | None = (
            frozenset(int(n) for n in target_nets)
            if target_nets is not None else None
        )
        # Net-aware reset strip policy. When True (default), reset wipes ONLY the
        # routing of the nets being re-routed (the target subset) and keeps every
        # other pre-routed net's copper — independently of lock (keeping a net !=
        # fixing it; lock only governs shove movability). No-op vs the legacy
        # bare-board reset when target_nets is None (reroute set = all nets) or
        # when there is no non-target routing. False = always wipe all routing.
        self._preserve_nontarget_routing: bool = bool(preserve_nontarget_routing)
        # Keep-routing augmentation (opt-in, training-time): start each episode
        # from a partially pre-routed board. EVERY reset samples a fresh keep
        # set K — fraction f ~ U[lo, hi] of the routable nets, chosen uniformly
        # at random — and strips all routing EXCEPT K's (the ``keep_nets``
        # strip path), so the board file's own routing on K survives as the
        # initial state. Because an episode may physically disturb kept copper
        # (routing a neighbouring net can shove — displace or even disconnect —
        # it) and reset has no in-place way to restore what a previous reset
        # already stripped, every reset after the first RELOADS the board from
        # file (engine rebuild, see ``_reload_board_from_file``) so the sample
        # always cuts from the pristine designer routing. Whole-board semantics
        # stay unchanged (target_nets=None): kept complete nets seed born-closed
        # and are simply not selectable. Requires the board file to actually
        # carry complete routing for every sampled net — enforced loudly at
        # reset (a pristine-board check, so it is pure file validation).
        if keep_routing_fraction is not None:
            if target_nets is not None:
                raise ValueError(
                    "keep_routing_fraction is a whole-board augmentation and is "
                    "mutually exclusive with target_nets (net-subset routing)"
                )
            if len(keep_routing_fraction) != 2:
                raise ValueError(
                    f"keep_routing_fraction must be (lo, hi), "
                    f"got {keep_routing_fraction!r}"
                )
            lo, hi = float(keep_routing_fraction[0]), float(keep_routing_fraction[1])
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(
                    f"keep_routing_fraction must satisfy 0 <= lo <= hi <= 1, "
                    f"got ({lo}, {hi})"
                )
            self._keep_routing_fraction: tuple[float, float] | None = (lo, hi)
        else:
            self._keep_routing_fraction = None
        self._sampled_keep_nets: frozenset[int] | None = None
        # Opt-in early-stop + best-Φ-board selection (eval/MCTS only; off by
        # default so training/existing rollouts are byte-identical). When
        # ``output_best_board`` the env keeps a checkpoint of the highest-Φ board
        # seen this episode and, at episode end, rolls the LIVE board back to it —
        # so the scorer/artifact (which re-read the live board) get the best board,
        # not a later flailing one. ``early_stop_ratsnest_patience`` truncates the
        # episode after that many steps with no change in the unrouted-ratsnest
        # count (connections stalled). See :meth:`step`.
        self._ratsnest_patience = int(early_stop_ratsnest_patience)
        self._output_best_board = bool(output_best_board)
        self._best_ckpt: "Checkpoint | None" = None
        self._best_potential = float("-inf")
        self._rats_stagnation = 0
        self._prev_unrouted = 0
        # Only COMMITTED steps advance the episode-level tracking. A linear rollout
        # (eval/PPO) leaves this True. MCTS drives many throwaway search steps
        # through the same env (restore+step per simulation) — RLSearchEnv flips
        # this to False for those so they don't pollute the stagnation counter /
        # best-board checkpoint, and True for the one committed step per decision.
        self._track_best_active = True

        # Python-side env RNG. Distinct from ``engine_seed`` below (that one is
        # KiCad's C++ KIID/UUID generator). Drives the keep-routing draw and the
        # terminal reward noise; gymnasium's ``reset(seed=None)`` leaves an
        # existing generator alone, so seeding once here survives every later
        # reset. None = gymnasium's lazy entropy seeding (unreproducible).
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # --- Engine & static board data ---
        # engine_seed (default 77; decided once here, never re-seeded): makes routing +
        # UUID-keyed DRC reproducible across runs/processes for a fixed action sequence.
        # None = default entropy seeding. Global generator → one env per process for
        # clean determinism (the standard vectorized-RL layout).
        # Kwargs kept verbatim for ``_reload_board_from_file`` (keep-routing
        # augmentation) so a reloaded engine is constructed EXACTLY like this
        # one — same engine_seed ⇒ same UUID stream from scratch each episode.
        self._engine_ctor_kwargs = dict(
            engine_seed=engine_seed,
            shove_iter_limit=shove_iter_limit,
            followbranch_iter_limit=followbranch_iter_limit,
            reject_if_stuck=reject_if_stuck,
            simplify_outline=simplify_outline,
            # Pro-less boards are refused by the engine unless this opt-in
            # says the YAML fallback below will substitute the rules.
            allow_default_rules=use_yaml_drc_fallback,
        )
        self._drc_config_path = drc_config_path
        self._use_yaml_drc_fallback = bool(use_yaml_drc_fallback)
        # True while the live board equals the file byte-for-byte (no strip,
        # no episode). Any reset flips it False; only an engine (re)build
        # turns it back on. Drives the keep-routing reload-per-reset.
        self._board_pristine = True
        self._engine = KiCadEngine(board_path, **self._engine_ctor_kwargs)
        # Net-subset (partial routing): scope the engine's unrouted/completion
        # count to the target nets so termination + reward Φ completion fire on
        # "all target nets connected". No-op (whole-board) when target_nets=None.
        self._engine.set_target_nets(self._target_nets)

        # Strict by default: raise if neither .kicad_pro nor legacy
        # setup tokens are present, since that almost always means a
        # load failure. Set ``use_yaml_drc_fallback=True`` AND an explicit
        # ``drc_config_path`` to opt in to substituting that YAML instead
        # (warns on trigger; a missing path also raises). Authoritative
        # loads are unaffected either way.
        apply_default_drc_if_fallback(
            self._engine,
            config_path=drc_config_path,
            use_yaml=use_yaml_drc_fallback,
        )

        # Aggregate per-netclass DRC info into a single "hardest setting"
        # dict for env-level use (prompt generation, logging, stricter-than
        # -Default invariants). Pure info — not pushed into the engine or
        # BDS, so the actual per-class rules the DRC engine enforces stay
        # untouched.
        self._hardest_design_rules = compute_hardest_per_netclass(
            self._engine.get_design_rules()
        )

        parsed = parse_pcb_file(board_path, self._engine, outline_mode=outline_obs)
        meta = self._engine.get_board_meta()
        # Cache the static parse pieces (pads / edges / net_names / obstacles are
        # routing-independent) + meta so set_target_nets() can rebuild board_info
        # with a new net filter without re-parsing.
        self._parsed = parsed
        self._meta = meta
        self._board_info = BoardStatic.from_board(
            meta=meta,
            pads=parsed["board_snapshot"].pads,
            board_edges=parsed["board_edges"],
            net_names=parsed["net_names"],
            board_constraints=self._hardest_design_rules,
            obstacles=parsed.get("obstacles", []),
            target_nets=self._target_nets,
        )
        if self._net_constraint_obs:
            self._fill_net_constraint_obs()
        # Net-subset DRC (Level 1): restrict the Python DRC cache to violations
        # touching a target net. DRC reports nets by NAME, so map target codes →
        # names via the (already target-filtered) board_info.nets. No-op when
        # target_nets is None.
        if self._target_nets is not None:
            self._engine.drc_helper.set_target_net_names(
                frozenset(nc.net_name for nc in self._board_info.nets.values())
            )
        # Nets that need routing work (≥2 pads, target-scoped). Single-pad nets
        # never carry ratsnest edges, so they are excluded from the "every net
        # was opened and closed once" early-termination check (see step §3).
        # Engine-owned definition, shared with eval.metrics (reward parity).
        self._routable_nets: frozenset[int] = self._engine.get_routable_nets()
        # "Already complete" nets: pad >=2 but no ratsnest — either
        # born-connected (e.g. stacked same-coordinate pads, physically joined
        # without copper) or already fully routed. Nothing to route, so they are
        # unselectable in net_valid_mask yet still in routable; left unseeded the
        # all-nets-closed termination (closed superset of routable) never fires
        # and the endgame reaches net_valid all-False. Recomputed from the
        # CURRENT state (post strip / keep) each reset and seeded closed —
        # caching once goes stale under preserve/keep resets. "closed" =
        # consumed by net_end + complete at reset time (seeded).
        self._born_closed_nets: frozenset[int] | None = None
        # Per-episode: nets closed by a successful net_end (select → ... →
        # net_end consumes the net; see net_valid_mask). Reset() re-seeds it
        # with ``_born_closed_nets``.
        self._episode_closed_nets: set[int] = set()

        # Seed PNS's "current via size" from the Default netclass, matching
        # what pcbnew uses when the user invokes ``add via`` without an
        # explicit preset. This is the same intent the old code had (apply
        # the board's declared via_size / via_drill) but sourced from the
        # engine so modern KiCad 9 boards (which dropped the legacy setup
        # tokens) work correctly instead of silently falling back to
        # ``SIZES_SETTINGS`` defaults — ``m_viaDrill=0.25 mm`` — that don't
        # always match the board's Default netclass drill and produce
        # avoidable DRC violations on ``make_via``.
        #
        # Track width is intentionally NOT set here: ``initRouter`` already
        # copies ``bds.GetCurrentTrackWidth()`` (which is the Default
        # netclass's track_width by KiCad convention) into PNS.
        _default_nc = self._engine.get_design_rules().default_netclass
        if _default_nc.via_diameter_mm > 0:
            self._engine.set_via_diameter(_default_nc.via_diameter_mm)
        if _default_nc.via_drill_mm > 0:
            self._engine.set_via_drill(_default_nc.via_drill_mm)

        # --- Action dispatcher (owns current_net_id, routing_mode) ---
        self._masking_rule_instance = get_masking_rule(masking_rule)
        self._dispatcher = ActionDispatcher()

        # --- Reward tracking ---
        # Overrides are applied to a config COPY before build so the config
        # and the built reward always agree; value validation lives in
        # PotentialReward.__init__ (no duplicate checks here).
        self._reward_config = get_reward_config(reward_rule).with_overrides(
            step_penalty=reward_step_penalty,
            via_penalty=via_penalty,
            wirelength_penalty=wirelength_penalty,
            drc_penalty=drc_penalty,
            drc_log_scale=drc_log_scale,
            drc_log_agg_scale=drc_log_agg_scale,
            drc_log_offset=drc_log_offset,
            wire_via_emission=wire_via_emission,
        )
        self._potential_reward = self._reward_config.build_reward()
        # Board-resolution hooks (completion / clean-completion log scale,
        # bbox-normalized wirelength): the definitions live in
        # PotentialReward.bind_board — shared with the offline scorer. Bound
        # after with_overrides, so a CLI --wirelength-penalty override is
        # normalized too. The per-reset group (size weights) binds in reset().
        self._potential_reward.bind_board(
            net_count=meta.net_count, bbox_w=meta.bbox_w, bbox_h=meta.bbox_h,
        )
        self._step_penalty = self._potential_reward.step_penalty
        # DRC runs every step in per_step mode (for Φ delta); only at episode end
        # in terminal mode. If the potential has zero DRC weight, skip the engine
        # entirely (no step DRC, no episode-end logging DRC).
        self._drc_active = _potential_uses_drc(self._potential_reward)
        # per_step-mode DRC uses the incremental engine path (bit-exact with
        # full DRC on a forward rollout, ~100x faster vs the stock provider). The
        # episode-end / terminal DRC stays full (ground truth). Toggle off to compare.
        self._drc_incremental = True
        self._reward = RewardTracker(
            fn=self._potential_reward,
            run_drc_on_reset=(self._reward_config.mode == "per_step" and self._drc_active),
        )
        # Baseline state for the ``on_net_end`` wire/via accumulation mode.
        # Reset to None on every env reset; advanced to the current state
        # whenever a flush event (net_end / episode terminated) fires.
        self._wire_via_ref_state: RewardState | None = None

        # Ratsnest edge count at the start of the episode (= initial
        # ``unconnected`` count). Used to expose ``info["ratsnest_reduction"]``.
        # Captured in reset(); 0 means a board with no targets (degenerate case).
        self._initial_unconnected: int = 0

        # Per-net pad-group counts of the bare board, captured in reset(). This
        # is the baseline the eval-time routability metric is measured against;
        # the env itself only carries it so inline DRC eval (which cannot reset
        # the live engine) can reach it.
        self._initial_pad_groups: dict[int, int] = {}

        # Potential of the bare (no-track) board, captured in reset(). Exposed in
        # the terminal info so every per-rollout row carries a baseline and the
        # headline potential can be reported as a (non-negative) gain over it.
        self._initial_potential: float = float("nan")

        # --- Observation cache ---
        # board_static carries a single ``board_constraints`` key populated
        # from ``board_info.board_constraints`` (the strictest-per-netclass
        # snapshot seeded by ``from_board`` above). Downstream consumers
        # (obs dict, LLM prompt builders, loggers) read it from there.
        self._obs_cache = ObsCache(
            board_static=_build_board_static(self._board_info),
        )
        if self._obs_format == "indexed":
            # Static table group: built ONCE from the canonical dict and
            # shared by reference in every step's indexed obs.
            self._obs_cache.static_tables = static_tables_from_dict(
                self._obs_cache.board_static,
            )

        # --- Recent-action tracking (for the obs action-history tokens) ---
        # Newest first; empty at episode start. Each entry is a
        # ``_make_history_record`` record (incl. net context + success flag).
        self._action_history_len = int(action_history_len)
        self._action_history: deque[dict] = deque(maxlen=self._action_history_len)

        # --- Gym spaces ---
        bbox = self._engine.get_board_bbox()
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(NUM_ACTIONS),
            "x_mm": spaces.Box(
                low=bbox.x_mm - 1.0,
                high=bbox.x_mm + bbox.width_mm + 1.0,
                shape=(), dtype=np.float32,
            ),
            "y_mm": spaces.Box(
                low=bbox.y_mm - 1.0,
                high=bbox.y_mm + bbox.height_mm + 1.0,
                shape=(), dtype=np.float32,
            ),
            "layer": spaces.Discrete(max(meta.copper_layers, 2) + 1),  # 1-indexed human layers
            "net_id": spaces.Discrete(max(meta.net_count, 1)),
            "routing_mode": spaces.Discrete(3),
        })
        self.observation_space = spaces.Dict({
            "board_static": spaces.Space(),
            "routing_geometry": spaces.Space(),
            "router_head": spaces.Space(),
        })

        # --- Gym standard ---
        self._step_count = 0
        self._renderer = None

        # --- Crash logger (survives SIGSEGV; dir resolved by pcb_world.diag) ---
        self._crash_logger = CrashLogger(env_id=id(self) % 10000)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        """Reset environment: delete all tracks, rebuild connectivity.

        ``options['preserve_routing']`` (default False): when True, skip
        the track / via deletion step so any pre-existing routing on the
        loaded board is kept. Used by interactive viewers when a board
        is first loaded — the user wants to *see* what's already drawn,
        not be handed a bare board. ``:restart`` and RL training
        pipelines leave the option False so reset() still produces the
        clean unrouted slate they expect.
        """
        super().reset(seed=seed)
        opts = options or {}
        preserve_routing = bool(opts.get("preserve_routing", False))
        # Per-reset override of the net-aware strip policy (default = ctor flag).
        preserve_nontarget = bool(
            opts.get("preserve_nontarget_routing", self._preserve_nontarget_routing)
        )
        # Explicit "keep" (preserve) set: net codes whose routing survives this
        # reset. Orthogonal to lock — a kept net stays drawn but is still movable
        # unless separately locked (lock ⊆ keep). ``None`` = no explicit keep set.
        keep_nets = opts.get("keep_nets")
        if keep_nets is not None:
            keep_nets = {int(n) for n in keep_nets}

        # Keep-routing augmentation: every reset cuts a FRESH keep set K out
        # of the pristine file routing. The first reset already sits on the
        # just-constructed engine (board == file); every later reset must
        # first reload the board from file, because the previous reset's
        # strip removed the non-kept routing for good and the episode may
        # have shoved (displaced / disconnected) even the kept copper — an
        # in-place restore does not exist. The draw comes from np_random,
        # seeded at construction (see __init__'s ``seed``), so the K sequence
        # is reproducible per env.
        if self._keep_routing_fraction is not None:
            if keep_nets is not None:
                raise ValueError(
                    "reset(options={'keep_nets': ...}) conflicts with the "
                    "keep_routing_fraction augmentation (the env owns the keep "
                    "set); construct without keep_routing_fraction to drive "
                    "keep_nets manually"
                )
            if not self._board_pristine:
                self._reload_board_from_file()
            self._sampled_keep_nets = self._sample_keep_nets()
            keep_nets = set(self._sampled_keep_nets)

        # Cancel any active routing/dragging
        if self._engine.is_routing():
            self._engine.cancel_route()
        if self._engine.is_dragging():
            self._engine.cancel_drag()

        if not preserve_routing:
            if keep_nets is not None:
                # Keep-set strip: wipe every net EXCEPT the preserved ones — the
                # re-route set is the complement. Lock is not consulted here
                # (kept-but-unlocked nets are preserved too); it only governs
                # shove movability, applied separately via lock_net.
                reroute = [
                    int(c) for c in self._engine.get_net_names()
                    if int(c) not in keep_nets
                ]
                if reroute:
                    self._engine.delete_routing_of_nets(reroute)
            elif preserve_nontarget and self._target_nets is not None:
                # Net-aware strip: wipe ONLY the nets being re-routed (the target
                # subset) to a clean slate; every other pre-routed net's copper is
                # kept (as a movable obstacle unless separately locked). One C++
                # pass over the board, keyed on net — not on lock.
                self._engine.delete_routing_of_nets(sorted(self._target_nets))
            else:
                # Bare-board strip: delete all routing. Guarded against infinite
                # loop — if delete_track_by_index returns False or the count fails
                # to drop, raise instead of spinning forever.
                remaining = self._engine.get_track_count()
                while remaining > 0:
                    ok = self._engine.delete_track_by_index(0)
                    new_remaining = self._engine.get_track_count()
                    if not ok or new_remaining >= remaining:
                        raise RuntimeError(
                            f"delete_track_by_index(0) failed to reduce count "
                            f"({remaining} -> {new_remaining}); aborting reset"
                        )
                    remaining = new_remaining

                # Delete all existing vias (same guard as tracks).
                remaining = self._engine.get_via_count()
                while remaining > 0:
                    ok = self._engine.delete_via_by_index(0)
                    new_remaining = self._engine.get_via_count()
                    if not ok or new_remaining >= remaining:
                        raise RuntimeError(
                            f"delete_via_by_index(0) failed to reduce count "
                            f"({remaining} -> {new_remaining}); aborting reset"
                        )
                    remaining = new_remaining

        self._engine.build_connectivity()
        # From here the live board has diverged from the file (strip above,
        # then an episode) — the keep-routing path must reload before the
        # next sample.
        self._board_pristine = False

        # Reset engine configuration
        self._engine.set_routing_mode(2)
        self._engine.set_corner_mode(self._corner_mode)
        self._engine.set_track_width(0)
        self._engine.reset_via_mode()
        self._engine.clear_drc_cache()
        # Drop any checkpoints carried over from the previous episode and re-seed the
        # handle epoch, so stale handles become invalid and the engine snapshot starts
        # fresh each episode (mirrors clear_drc_cache resetting the DRC snapshot).
        self._engine.reset_checkpoints()

        # Reset state
        self._step_count = 0
        self._dispatcher.reset()
        # Routability baseline: how the pads are grouped on the just-reset board
        # (post keep/strip + build_connectivity, pre routing). The same capture
        # feeds the per-reset reward hook (size-weighted ladder weights, defined
        # in PotentialReward.bind_board) — keep-routing augmentation makes them
        # episode-dependent, and the bind must precede self._reward.reset() so
        # Φ₀ already uses this episode's weights.
        self._initial_pad_groups = self._engine.get_pad_groups()
        self._potential_reward.bind_board(
            pad_groups=self._initial_pad_groups,
            net_names={c: n.net_name for c, n in self._board_info.nets.items()},
            routable_nets=self._routable_nets,
        )
        self._reward.reset(self._engine)
        self._wire_via_ref_state = None
        self._crash_logger.on_reset(self.board_path)
        self._action_history.clear()
        # Seed the "already complete" nets — judged from the CURRENT ratsnest
        # (after this reset's strip / keep + build_connectivity) and recomputed
        # every reset (NOT cached). A routable net with no ratsnest (already
        # fully routed at episode start, or born-connected) is seeded closed so
        # the all-nets-closed termination can complete; a net that still has
        # ratsnest is never frozen closed.
        #   Must NOT cache once: under a preserve/keep reset a net that was
        #   routed at the first reset would be frozen into born_closed; a later
        #   restart that wipes it re-creates its ratsnest, yet it would stay
        #   stuck closed -> net_valid all-False -> the dead-pointer-row guard
        #   fires. Recomputing from the current state each reset avoids that.
        rats_nets = {r.net_code for r in self._engine.get_ratsnest()}
        self._born_closed_nets = frozenset(self._routable_nets - rats_nets)
        # Keep-routing augmentation invariant: every sampled kept net must be
        # COMPLETE on the loaded board (no ratsnest after the strip — i.e. in
        # born_closed). A kept net with missing or partial routing means the
        # board file does not carry the finished designer routing the
        # augmentation assumes; fail loudly instead of silently training on a
        # half-kept net.
        if self._sampled_keep_nets:
            incomplete = self._sampled_keep_nets - self._born_closed_nets
            if incomplete:
                names = self._engine.get_net_names()
                detail = ", ".join(
                    f"{c} ({names.get(c, '?')})" for c in sorted(incomplete)
                )
                # This check always runs on a just-loaded (pristine) board —
                # the reload-per-reset contract makes it pure file validation.
                # The path ends in a raise, so info["keep_nets"] (success path
                # only) never carries the sample: the message itself carries
                # the evidence a bad-file report needs.
                raise RuntimeError(
                    f"keep_routing_fraction: sampled kept net(s) {detail} are "
                    f"not fully routed on {self.board_path!r} — the "
                    f"augmentation requires boards whose file carries complete "
                    f"routing for every net\n"
                    f"  [diag] kept={sorted(self._sampled_keep_nets)} "
                    f"stripped={sorted(self._routable_nets - self._sampled_keep_nets)} "
                    f"born_closed={sorted(self._born_closed_nets)} "
                    f"ratsnest={sorted(rats_nets)} "
                    f"routable={sorted(self._routable_nets)}"
                )
        self._episode_closed_nets = set(self._born_closed_nets)

        # Capture episode-initial unconnected count for the ratsnest metric.
        # ``self._reward.reset`` populated ``prev_state`` from the just-reset
        # engine, so its ``unconnected`` is the total target count.
        self._initial_unconnected = (
            self._reward.prev_state.unconnected
            if self._reward.prev_state is not None else 0
        )
        # Φ of the bare board (same potential used for final_potential below).
        self._initial_potential = (
            float(self._potential_reward.potential(self._reward.prev_state))
            if self._reward.prev_state is not None else float("nan")
        )

        # LAST engine step: rewind the global KIID/UUID generator to its construction-time
        # position so this episode's routing draws the same UUID stream as every other
        # episode. The generator is seeded once at ctor and otherwise advances monotonically
        # across episodes (reset does not re-seed it), which would drift the UUID
        # obstacle tie-break between episodes; this pins it. Done after every board mutation
        # above (delete/build_connectivity/reward-DRC) so those draws are discarded, mirroring
        # restore()'s final rewind. No-op under entropy seeding.
        #
        # ONLY when the board is truly bare after the strip: the rewind's collision-safety
        # relies on the previous episode's routed tracks being deleted (their UUIDs freed)
        # before the stream is re-issued. Any KEPT track (preserve_routing, or a net-aware
        # strip that kept non-target routing) would collide — rewinding could hand a new
        # track the same UUID as a surviving one, a duplicate KIID the board's id cache does
        # not reject. Skipping the rewind is the safe branch; it costs only the per-episode
        # UUID-stream determinism, which the paths that keep tracks (interactive / staged /
        # keep_routing_fraction training augmentation) don't rely on.
        board_bare = (
            self._engine.get_track_count() == 0 and self._engine.get_via_count() == 0
        )
        if not preserve_routing and board_bare:
            self._engine.rewind_kiid_to_episode_start()

        # Best-Φ-board tracking / ratsnest early-stop bookkeeping (opt-in).
        self._rats_stagnation = 0
        self._prev_unrouted = int(self._initial_unconnected)
        # Re-arm episode-level tracking. A search driver (RLSearchEnv.step) flips
        # this per step to exclude throwaway simulations, and it is NOT restored
        # when the search ends — without this, an MCTS rollout that stops on a
        # search step leaves it False and the NEXT episode on the same env runs
        # with best-Φ / early-stop silently disabled (mcts_compare reuses one env
        # for the MCTS arm and the plain arm, so the comparison would be skewed).
        self._track_best_active = True
        if self._output_best_board:
            # reset_checkpoints() above invalidated any prior handle; the old
            # Checkpoint object frees itself via RAII when dropped here.
            self._best_potential = self._initial_potential
            self._best_ckpt = self.checkpoint()
        else:
            self._best_ckpt = None
            self._best_potential = float("-inf")

        obs = self._get_obs()
        info = self._get_info()
        if self._sampled_keep_nets is not None:
            info["keep_nets"] = sorted(self._sampled_keep_nets)
        return obs, info

    def _reload_board_from_file(self) -> None:
        """Rebuild the engine from the board file (keep-routing augmentation).

        There is no in-place restore: once a reset strips the non-kept
        routing (or an episode disturbs kept copper), only a fresh load
        brings the file's designer routing back. The engine is recreated
        with the construction-time kwargs — same ``engine_seed``, so every
        episode draws the identical UUID stream from scratch (the reload
        equivalent of ``rewind_kiid_to_episode_start``). All board-derived
        static state (``_board_info``, obs caches, spaces, ``_routable_nets``)
        keys on the file contents and stays valid as-is; only the live-engine
        push-downs from ``__init__`` are replayed here. Costs one board load
        (~10 ms — measured 260822 on d3b/d2b sources).
        """
        # Checkpoint handles die with their router; drop ours first so the
        # RAII release runs against the still-live engine.
        self._best_ckpt = None
        # A renderer holds native views of the old engine — rebuild lazily.
        self._renderer = None
        self._engine.close()
        self._engine = KiCadEngine(self.board_path, **self._engine_ctor_kwargs)
        self._engine.set_target_nets(self._target_nets)
        apply_default_drc_if_fallback(
            self._engine,
            config_path=self._drc_config_path,
            use_yaml=self._use_yaml_drc_fallback,
        )
        # Re-seed PNS's current via size from the Default netclass (same
        # intent + source as the construction-time push above).
        _default_nc = self._engine.get_design_rules().default_netclass
        if _default_nc.via_diameter_mm > 0:
            self._engine.set_via_diameter(_default_nc.via_diameter_mm)
        if _default_nc.via_drill_mm > 0:
            self._engine.set_via_drill(_default_nc.via_drill_mm)
        self._board_pristine = True

    def _sample_keep_nets(self) -> frozenset[int]:
        """Draw the keep-routing augmentation set K (see ``__init__``).

        f ~ U[lo, hi], |K| = round(f · #routable), members uniform without
        replacement from the routable nets. Uses ``self.np_random``, so the
        draw is reproducible whenever the env was constructed with a ``seed``
        (or reset with one) — the training factories derive that seed per env
        and per board reload, which keeps K varying across reloads *and*
        replayable.
        """
        lo, hi = self._keep_routing_fraction
        candidates = sorted(self._routable_nets)
        frac = float(self.np_random.uniform(lo, hi))
        k = int(round(frac * len(candidates)))
        if k == 0:
            return frozenset()
        picks = self.np_random.choice(len(candidates), size=k, replace=False)
        return frozenset(candidates[int(i)] for i in picks)

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        """Execute one routing action step.

        Reward = step_penalty + potential_diff + other_penalty(*)
          step_penalty   : constant per-step cost (always paid)
          potential_diff : Φ-based shaping
                           · per_step : Φ(after) − Φ(before) per step
                                      (0 for non-valid actions; state unchanged)
                           · terminal : Φ(s_final) on episode end
                                      (compute_final on terminate / compute_truncation on truncate)
          other_penalty  : per-action-class extra cost (all from reward_config)
                           · idle / valid_effective              : 0
                           · parse_fail                          : −parse_fail_penalty
                           · mask_reject                         : −mask_reject_penalty
                           · valid_empty / valid_dispatch_fail   : −invalid_action_penalty

        (*) Sparse terminal-step convention:
              · valid_empty / valid_dispatch_fail : REPLACE — invalid_action_penalty
                  is dropped on the terminal step (terminal Φ takes over).
              · parse_fail / mask_reject          : ADD — other_penalty stays
                  on the terminal step.
              · idle / valid_effective            : indistinguishable
                  (other_penalty == 0 either way).
            Dense mode always sums all three components.
        """
        self._step_count += 1

        action_type = int(action["action_type"])
        is_parse_invalid = action.get("_parse_invalid", False)
        mode = self._reward_config.mode
        step_drc = (mode == "per_step") and self._drc_active
        # Net context for the history record — BEFORE dispatch (net_end
        # deselects; net_select reads its own param).
        action_net = _action_net_context(
            action, action_type, self._dispatcher.current_net_id,
        )

        # ------------------------------------------------------------------
        # 1. Classify action
        # ------------------------------------------------------------------
        if action_type == ACT_IDLE:
            action_class = "parse_fail" if is_parse_invalid else "idle"
        elif not self._get_action_mask()[action_type]:
            action_class = "mask_reject"
            logger.warning(
                "Invalid action %d (has_net=%s, is_routing=%s)",
                action_type,
                self._dispatcher.current_net_id is not None,
                self._engine.is_routing(),
            )
        else:
            action_class = "valid"  # refined below after dispatch

        # ------------------------------------------------------------------
        # 2. Execute (only valid actions touch the engine)
        # ------------------------------------------------------------------
        success: bool = False
        empty_action: bool = False
        dispatch_info: dict[str, Any] = {}
        before_state: RewardState | None = None
        after_state: RewardState | None = None

        if action_class == "valid":
            # Read before state (reuse cached prev_state in per_step mode)
            if mode == "per_step" and self._reward.prev_state is not None:
                before_state = self._reward.prev_state
            else:
                before_snap = self._engine.get_reward_snapshot(run_drc=step_drc)
                before_state = RewardState.from_snapshot(before_snap)

            # Crash-log action + state BEFORE the C++ call (survives SIGSEGV)
            router_state = {
                "current_net": self._dispatcher.current_net_id,
                "is_routing": self._engine.is_routing(),
                "step": self._step_count,
            }
            try:
                session = self._engine.get_routing_session_state()
                router_state["head_xy"] = list(session.route_head[:2])
                router_state["current_layer"] = session.current_layer
            except Exception:
                pass
            self._crash_logger.on_pre_step(action, router_state)

            # Dispatch + build connectivity for track-modifying actions.
            # Timed (dispatch → connectivity → after-snapshot): per-action engine
            # latency telemetry — info-only, never observable by the policy.
            _engine_t0 = time.perf_counter()
            success, dispatch_info = self._dispatch(action_type, action)
            if action_type in (ACT_START_ROUTE, ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_FINISH):
                self._engine.build_connectivity()
            # net_end consumes the net for this episode: it leaves the
            # net_select candidate pool (net_valid_mask) and counts toward
            # the all-nets-closed early termination below. router_state
            # captured current_net BEFORE dispatch (net_end deselects).
            if success and action_type == ACT_NET_END:
                closed_net = router_state.get("current_net")
                if closed_net is not None:
                    self._episode_closed_nets.add(int(closed_net))
            after_snap = self._engine.get_reward_snapshot(
                run_drc=step_drc, incremental=self._drc_incremental,
            )
            after_state = RewardState.from_snapshot(after_snap)
            engine_step_s = time.perf_counter() - _engine_t0
            self._crash_logger.on_step_time(engine_step_s)

            # Refine action_class based on dispatch outcome. "Empty" means the
            # dispatch succeeded but committed nothing: PNS FixRoute reports
            # success for a fully-redundant retrace (NODE::Add silently drops
            # duplicate segments) and for a lone pending via that never lands
            # on the board. via_count is compared alongside track_count /
            # wirelength so a committed via is never misread as a no-op.
            empty_action = (
                action_type in (ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_FINISH)
                and after_state.track_count == before_state.track_count
                and after_state.via_count == before_state.via_count
                and after_state.wirelength == before_state.wirelength
            )
            if not success:
                action_class = "valid_dispatch_fail"
            elif empty_action:
                action_class = "valid_empty"
            else:
                action_class = "valid_effective"

        # ------------------------------------------------------------------
        # 3. Episode-end determination
        # ------------------------------------------------------------------
        all_nets_closed = (
            bool(self._routable_nets)
            and self._episode_closed_nets.issuperset(self._routable_nets)
        )
        terminated = (
            after_state is not None
            and after_state.unconnected == 0
        ) or all_nets_closed
        truncated = self._step_count >= self.max_steps

        # Ratsnest-stagnation early stop (opt-in): T consecutive steps with no
        # change in the unrouted count = connections stalled → truncate. A
        # non-valid action leaves after_state None (board unchanged → also counts
        # as no progress). Tune T above the normal net_select→start_route→…→connect
        # gap so it fires on genuine stalls, not routine setup steps.
        if self._ratsnest_patience > 0 and self._track_best_active:
            cur_unrouted = (
                int(after_state.unconnected) if after_state is not None
                else self._prev_unrouted
            )
            if cur_unrouted == self._prev_unrouted:
                self._rats_stagnation += 1
            else:
                self._rats_stagnation = 0
            self._prev_unrouted = cur_unrouted
            if self._rats_stagnation >= self._ratsnest_patience:
                truncated = True

        # ------------------------------------------------------------------
        # 4. Episode-end DRC + final_state (used for terminal-mode Φ + info)
        # ------------------------------------------------------------------
        drc_violations = 0
        drc_per_net: dict[int, int] = {}
        drc_errors = 0
        drc_errors_per_net: dict[int, int] = {}
        drc_promoted = 0
        drc_promoted_per_net: dict[str, int] = {}
        final_state: RewardState | None = None

        if terminated or truncated:
            if self._drc_active:
                if not step_drc:
                    self._engine.run_drc()
                severity_mode = self._potential_reward.drc_severity_mode
                drc_violations = self._engine.drc_helper.get_count_by_severity_mode(severity_mode)
                drc_per_net = self._engine.drc_helper.get_counts_by_net_by_severity_mode(severity_mode)
                drc_errors = self._engine.drc_helper.get_error_count()
                drc_errors_per_net = self._engine.drc_helper.get_error_counts_by_net()
                drc_promoted = self._engine.drc_helper.get_count_by_severity_mode(
                    DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED,
                )
                drc_promoted_per_net = (
                    self._engine.drc_helper.get_counts_by_net_by_severity_mode(
                        DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED,
                    )
                )
            # Build final_state — use after_state for valid actions, else fresh snap
            if after_state is not None:
                base_state = after_state
            else:
                snap = self._engine.get_reward_snapshot(run_drc=False)
                base_state = RewardState.from_snapshot(snap)
            # Overlay ONLY the episode-end DRC counts; every other field
            # (via_count, per-net connectivity, …) carries over. Re-listing the
            # fields by hand silently dropped via_count, so terminal-mode
            # via_penalty was never charged at episode end (caught by
            # tests/test_reward_parity.py). per_step rewards never read
            # final_state; its per_step info["final_potential"] is reached only
            # when the terminal step was a non-valid action (after_state None).
            final_state = replace(
                base_state,
                drc_violations=drc_violations,
                drc_violations_per_net=drc_per_net,
                drc_errors=drc_errors,
                drc_errors_per_net=drc_errors_per_net,
                drc_promoted=drc_promoted,
                drc_promoted_per_net=drc_promoted_per_net,
            )

        # ------------------------------------------------------------------
        # 5. Reward = step_penalty + potential_diff + other_penalty
        # ------------------------------------------------------------------

        # 5a. step_penalty (always paid)
        step_pen = -self._step_penalty

        # 5b. potential_diff
        potential_diff = 0.0
        if mode == "per_step":
            # Per-step Φ delta (0 for non-valid actions; state did not change)
            if action_class.startswith("valid_"):
                if self._potential_reward.wire_via_emission == "on_net_end":
                    if self._wire_via_ref_state is None:
                        self._wire_via_ref_state = before_state
                    flush = (
                        action_type == ACT_NET_END
                        or after_state.unconnected == 0
                    )
                    per_step_reward, self._wire_via_ref_state = (
                        self._potential_reward.compute_dense_netend(
                            before_state,
                            after_state,
                            self._wire_via_ref_state,
                            flush_wire_via=flush,
                        )
                    )
                else:
                    per_step_reward = self._potential_reward.compute_dense(
                        before_state, after_state,
                    )
                # compute_dense* internally subtracts step_penalty;
                # isolate the Φ-only piece since step_penalty is component (5a).
                potential_diff = per_step_reward + self._potential_reward.step_penalty
                self._reward.prev_state = after_state
        else:  # terminal
            if terminated:
                potential_diff = self._potential_reward.compute_final(final_state)
            elif truncated:
                potential_diff = self._potential_reward.compute_truncation(final_state)
            # Optional Gaussian noise on terminal Φ (terminal mode only, legacy)
            if (terminated or truncated) and self._reward_noise_std > 0:
                noise = self.np_random.normal(0, self._reward_noise_std)
                noise = np.clip(noise, -4 * self._reward_noise_std,
                                4 * self._reward_noise_std)
                potential_diff += noise

        # 5c. other_penalty (action-class specific; all values from reward_config)
        if action_class == "parse_fail":
            other_pen = -self._reward_config.parse_fail_penalty
        elif action_class == "mask_reject":
            other_pen = -self._reward_config.mask_reject_penalty
        elif action_class in ("valid_empty", "valid_dispatch_fail"):
            other_pen = -self._reward_config.invalid_action_penalty
        else:
            other_pen = 0.0

        # On terminal-mode episode end, drop invalid_action_penalty
        # (valid_empty / valid_dispatch_fail) so the terminal Φ replaces
        # per-step shaping. parse_fail / mask_reject keep their other_penalty
        # even at terminal (the LLM policy still needs the penalty signal).
        if (
            mode == "terminal"
            and (terminated or truncated)
            and action_class in ("valid_empty", "valid_dispatch_fail")
        ):
            other_pen = 0.0

        reward = step_pen + potential_diff + other_pen

        # ------------------------------------------------------------------
        # 6. Build obs / info
        # ------------------------------------------------------------------
        self._action_history.appendleft(
            _make_history_record(action, action_type, success, action_net)
        )
        obs = self._get_obs()
        info = self._get_info()
        info["action_type"] = action_type
        info["action_success"] = success
        info["action_class"] = action_class
        # Φ-delta component of the reward (no step / invalid penalties), exposed
        # so search (MCTS) can accumulate a value that telescopes onto Φ.
        info["potential_diff"] = potential_diff

        if action_class == "parse_fail":
            info["parse_invalid"] = True
            info["error"] = "parse_invalid"
        elif action_class == "idle":
            info["idle"] = True
        elif action_class == "mask_reject":
            info["error"] = "invalid_action"
        elif action_class.startswith("valid_"):
            info["dispatch_info"] = dispatch_info
            info["empty_action"] = empty_action
            info["step_time_s"] = engine_step_s
            info["track_count"] = after_state.track_count
            info["via_count"] = after_state.via_count
            info["wirelength"] = after_state.wirelength
            # Success = fully routed (unconnected == 0), matching the post-hoc
            # scorer (eval/metrics.evaluate_one). NOT merely "terminated": the
            # all-nets-closed voluntary finish (give-up) terminates with
            # unconnected > 0 and must not count as success.
            info["success"] = bool(terminated and after_state.unconnected == 0)
            # Fraction by which the KiCad ratsnest shrank. Signed: dangling
            # copper adds ratsnest edges, so this goes negative when the board
            # grows more islands than it closes connections. It is a cheap
            # per-step progress signal, NOT the routability metric — that one
            # is pad-group based and is computed at eval time only.
            info["ratsnest_reduction"] = (
                1.0 if self._initial_unconnected == 0
                else (self._initial_unconnected - after_state.unconnected)
                     / self._initial_unconnected
            )

        if terminated or truncated:
            info["drc_violations"] = drc_violations
            # Errors-only count (severity-mode independent) — the clean_pass
            # ingredient: clean_pass = fully-routed AND drc_errors == 0.
            if self._drc_active:
                info["drc_errors"] = drc_errors
            if mode == "per_step" and after_state is not None:
                info["final_potential"] = self._potential_reward.potential(after_state)
            elif final_state is not None:
                info["final_potential"] = self._potential_reward.potential(final_state)
            # Baseline + gain over the bare board (always carried on the row).
            info["initial_potential"] = self._initial_potential
            if "final_potential" in info:
                info["potential_gain"] = info["final_potential"] - self._initial_potential
            if truncated and not terminated:
                info["TimeLimit.truncated"] = True

        # ------------------------------------------------------------------
        # 7. Best-Φ-board selection (opt-in, eval/MCTS only) — track the
        #    episode-wide highest-Φ board, and on episode end roll the LIVE board
        #    back to it so the scorer/artifact (which re-read the live board) get
        #    the best board rather than a later degraded one. Uses the env's
        #    reward Φ (potential(after_state)) — the same Φ the reward optimizes;
        #    a close proxy for the eval scorer's final_potential (different
        #    reward_config, but aligned). No effect unless output_best_board.
        # ------------------------------------------------------------------
        if self._output_best_board and self._track_best_active and after_state is not None:
            fp_now = float(self._potential_reward.potential(after_state))
            if not (terminated or truncated):
                if fp_now > self._best_potential:
                    if self._best_ckpt is not None:
                        self.release_checkpoint(self._best_ckpt)
                    self._best_ckpt = self.checkpoint()
                    self._best_potential = fp_now
            elif self._best_ckpt is not None and fp_now < self._best_potential - 1e-9:
                # Ended below an earlier peak → restore the best board and re-derive
                # the board-describing outputs from it (obs + terminal info fields).
                self.restore(self._best_ckpt)
                obs = self._get_obs()
                snap = self._engine.get_reward_snapshot(run_drc=self._drc_active)
                st = RewardState.from_snapshot(snap)
                info["track_count"] = st.track_count
                info["via_count"] = st.via_count
                info["wirelength"] = st.wirelength
                info["unrouted_count"] = int(st.unconnected)
                info["ratsnest_reduction"] = (
                    1.0 if self._initial_unconnected == 0
                    else (self._initial_unconnected - st.unconnected)
                         / self._initial_unconnected
                )
                info["final_potential"] = self._best_potential
                info["potential_gain"] = self._best_potential - self._initial_potential
                info["success"] = bool(st.unconnected == 0)
                if self._drc_active:
                    severity_mode = self._potential_reward.drc_severity_mode
                    info["drc_violations"] = (
                        self._engine.drc_helper.get_count_by_severity_mode(severity_mode)
                    )
                    info["drc_errors"] = self._engine.drc_helper.get_error_count()
                info["output_best_board_restored"] = True
                info["best_potential"] = self._best_potential

        self._crash_logger.on_post_step(success, info)

        return obs, reward, terminated, truncated, info

    def _dispatch(
        self, action_type: int, action: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Dispatch to the appropriate routing action function."""
        if action_type == ACT_NET_SELECT:
            # Seed the engine with the selected net's DRC params before the
            # dispatcher runs. External observers (LLM prompt / RL tokenizer)
            # stay untouched — width/via sizing is the engine's concern.
            self._apply_net_constraints(int(action["net_id"]))
        result = self._dispatcher.dispatch(self._engine, action_type, action)
        return result.success, result.info

    # ------------------------------------------------------------------
    # Net-class aware engine parameters
    # ------------------------------------------------------------------

    # Netclass fields resolved for the engine push (net_select) AND the
    # per-net constraint observation: (key, netclass attr, BDS global-min
    # attr). The via-drill floor lives under the legacy "through hole" name.
    _NET_RULE_FIELDS = (
        ("track_width", "track_width_mm", "min_track_width_mm"),
        ("clearance", "clearance_mm", "min_clearance_mm"),
        ("via_diameter", "via_diameter_mm", "min_via_diameter_mm"),
        ("via_drill", "via_drill_mm", "min_through_hole_mm"),
    )

    def _resolve_net_rule_values(
        self, net_id: int, rules,
    ) -> tuple[dict[str, float], dict[str, float], bool]:
        """Resolve the netclass DRC values (``_NET_RULE_FIELDS``) for ``net_id``.

        Returns ``(raw, clamped, is_default)``. ``raw`` prefers the matched
        netclass value; a field the class leaves unset (``-1.0`` — KiCad
        "inherit") falls back to the Default netclass, mirroring KiCad's own
        resolution rules. ``clamped`` lifts each raw value up to the BDS
        global minimum when one is declared (``floor > 0``): KiCad doesn't
        always adjust netclass values to the min on load, so a raw class
        value can sit below the DRC floor — using it unchanged would route
        into immediate-violation territory. A negative floor (unset) leaves
        the value alone. ``clamped`` is the effective value — what the
        engine is driven with and what the observation exposes.
        """
        nc = self._resolve_netclass(net_id, rules)
        default = rules.default_netclass
        is_default = getattr(nc, "name", "") == getattr(default, "name", "")
        raw: dict[str, float] = {}
        clamped: dict[str, float] = {}
        for key, attr, floor_attr in self._NET_RULE_FIELDS:
            v = getattr(nc, attr, -1.0)
            if v < 0:
                v = getattr(default, attr, -1.0)
            raw[key] = v
            # Direct attribute access on purpose: every floor exists on the
            # binding's RLDesignRules — a missing one is a contract break
            # that must fail loud, not silently skip the clamp.
            floor = getattr(rules, floor_attr)
            clamped[key] = floor if (floor > 0 and v < floor) else v
        return raw, clamped, is_default

    def _apply_net_constraints(self, net_id: int) -> None:
        """Push ``track_width`` / ``via_diameter`` / ``via_drill`` from the
        selected net's netclass into the engine (resolution + clamping:
        ``_resolve_net_rule_values``; clearance is not pushed — the DRC
        engine enforces it natively per netclass).

        Nets that resolve to the **Default** netclass take the fast path:
        we do nothing, since ``__init__`` already seeded the router with
        those values via ``initRouter`` + the default-netclass via sizes.
        Re-pushing identical values mid-episode was observed to perturb
        PNS size-cache state on boards that sit on the DRC boundary (see
        ``test_via_strategy_drc_zero``), so we only override when a
        non-Default class is actually in effect — BUT a raw default value
        can itself sit below the DRC floor (e.g. default track_width 0.2 <
        min 0.3), so for Default we still push ONLY the fields the clamp
        actually RAISED above their raw value (never an identical re-push).
        """
        rules = self._engine.get_design_rules()
        raw, clamped, is_default = self._resolve_net_rule_values(net_id, rules)

        tw, tw_raw = clamped["track_width"], raw["track_width"]
        if tw > 0 and (not is_default or tw != tw_raw):
            self._engine.set_track_width(tw)
        vd, vd_raw = clamped["via_diameter"], raw["via_diameter"]
        if vd > 0 and (not is_default or vd != vd_raw):
            self._engine.set_via_diameter(vd)
        dr, dr_raw = clamped["via_drill"], raw["via_drill"]
        if dr > 0 and (not is_default or dr != dr_raw):
            self._engine.set_via_drill(dr)

    def _fill_net_constraint_obs(self) -> None:
        """Populate ``NetContext.constraints`` for every net in
        ``board_info.nets`` (the ``net_constraint_obs`` knob) with the
        resolved-and-clamped values from ``_resolve_net_rule_values`` —
        identical to what ``net_select`` pushes into the engine, so the
        policy observes the exact widths / clearances it will actually
        route with. A non-positive resolved value (field unset ``-1``
        through the whole netclass → Default chain, or declared as an
        explicit ``0`` — with no BDS floor lifting it either way) has no
        usable observation value — loud failure instead of a silent 0.
        """
        rules = self._engine.get_design_rules()
        for code, ctx in self._board_info.nets.items():
            raw, clamped, _is_default = self._resolve_net_rule_values(
                code, rules,
            )
            bad = {k: ("unset" if raw[k] < 0 else f"{raw[k]!r}")
                   for k, v in clamped.items() if v <= 0}
            if bad:
                raise RuntimeError(
                    f"net_constraint_obs: net {code} ({ctx.net_name!r}) "
                    f"resolves no positive value for {bad} (netclass → "
                    "Default resolution shown; no BDS minimum lifts it)"
                )
            ctx.constraints = clamped

    def _resolve_netclass(self, net_id: int, rules):
        """Return the ``RLNetClassInfo`` that applies to ``net_id``.

        Prefers the authoritative engine-side lookup
        (``engine.get_netclass_for_net``) which mirrors KiCad's own
        ``NETINFO_ITEM::GetNetClass()`` — that's correct even when the
        net is assigned via ``.kicad_pro`` ``netclass_patterns`` rather
        than by sharing a name with the class.

        Falls back to a name-equality heuristic (and finally to
        ``default_netclass``) if the engine call is unavailable
        (older binding without the method, or an empty-name response
        signalling lookup failure).
        """
        # Engine-side lookup (authoritative).
        lookup = getattr(self._engine, "get_netclass_for_net", None)
        if lookup is not None:
            try:
                nc = lookup(net_id)
            except Exception:  # noqa: BLE001 — guard against binding quirks
                nc = None
            if nc is not None and getattr(nc, "name", ""):
                return nc

        # Fallback: name-equality heuristic (used on pre-extension builds).
        net_ctx = self._board_info.nets.get(net_id)
        if net_ctx is not None:
            net_name = net_ctx.net_name
            for nc in rules.netclasses:
                if nc.name == net_name:
                    return nc
        return rules.default_netclass

    def _get_obs(self) -> dict:
        """Build JSON observation.

        Also updates ``_obs_cache.net_geometry`` (canonical IR)
        for efficient consumption by wrappers.
        """
        snapshot = self._engine.get_board_snapshot()
        router_session = self._engine.get_routing_session_state()

        self._obs_cache.net_geometry = build_net_geometry(
            snapshot, self._board_info, layer_map=self._engine.layer_map,
            target_nets=self._target_nets,
        )

        head_xy = (router_session.route_head[0], router_session.route_head[1])
        if self._emit_drc_tokens:
            # Filter DRC tokens by the reward's severity mode so state and
            # reward share exactly the same view of "which violations count".
            drc_violations = self._engine.drc_helper.get_sorted(
                head_xy=head_xy,
                k=32,
                severity_mode=self._potential_reward.drc_severity_mode,
            )
        else:
            drc_violations = []

        if self._obs_format == "indexed":
            return build_indexed_observation(
                static_tables=self._obs_cache.static_tables,
                net_geometry=self._obs_cache.net_geometry,
                router_state=router_session,
                step_count=self._step_count,
                max_steps=self.max_steps,
                current_net_id=self._dispatcher.current_net_id,
                routing_mode=self._dispatcher.routing_mode,
                drc_violations=drc_violations,
                action_history=list(self._action_history),
                closed_nets=sorted(self._episode_closed_nets),
            )

        return build_json_observation(
            snapshot=snapshot,
            router_state=router_session,
            board_info=self._board_info,
            step_count=self._step_count,
            max_steps=self.max_steps,
            current_net_id=self._dispatcher.current_net_id,
            routing_mode=self._dispatcher.routing_mode,
            board_static=self._obs_cache.board_static,
            net_geometry=self._obs_cache.net_geometry,
            drc_violations=drc_violations,
            action_history=list(self._action_history),
            closed_nets=sorted(self._episode_closed_nets),
        )

    def _get_action_mask(self) -> np.ndarray:
        """Build action mask from observable engine state."""
        ctx = MaskContext(
            has_net=self._dispatcher.current_net_id is not None,
            is_routing=self._engine.is_routing(),
            net_fully_connected=self.is_current_net_connected(),
            step_count=self._step_count,
            max_steps=self.max_steps,
            unrouted_count=self._engine.get_unrouted_count(),
        )
        return self._masking_rule_instance.build_mask(ctx)

    # ------------------------------------------------------------------
    # Public API (for wrappers and external consumers)
    # ------------------------------------------------------------------

    def set_target_nets(self, target_nets) -> None:
        """Change the routable-net subset at runtime (net codes) and rebuild the
        cached static context to match.

        Only these nets appear in ``board_static.nets`` (so only they are
        selectable / net_valid), carry ratsnest, and count toward
        unrouted/termination; every other net's pads drop to
        ``unconnected_pads`` and its ratsnest is hidden — but its **tracks/vias
        stay visible** in ``routing_geometry`` (existing copper the router must
        clear). ``None`` restores "all nets routable".

        Used by the interactive viewer's Keep/Lock flow: kept nets are removed
        from the routable set (``target_nets = all − keep``) so an incomplete
        kept net can no longer be selected or re-drawn, yet its copper is still
        shown. Rebuilds ``board_info`` from the cached static parse — no
        re-parse, no routing change; call between episodes (e.g. before/at
        reset)."""
        if self._keep_routing_fraction is not None and target_nets is not None:
            raise ValueError(
                "set_target_nets conflicts with the keep_routing_fraction "
                "augmentation (whole-board semantics; the env owns the keep set)"
            )
        self._target_nets = (
            frozenset(int(n) for n in target_nets)
            if target_nets is not None else None
        )
        self._engine.set_target_nets(self._target_nets)
        p = self._parsed
        self._board_info = BoardStatic.from_board(
            meta=self._meta,
            pads=p["board_snapshot"].pads,
            board_edges=p["board_edges"],
            net_names=p["net_names"],
            board_constraints=self._hardest_design_rules,
            obstacles=p.get("obstacles", []),
            target_nets=self._target_nets,
        )
        if self._net_constraint_obs:
            self._fill_net_constraint_obs()
        self._routable_nets = self._engine.get_routable_nets()
        # DRC name filter tracks the routable set (None → whole board).
        self._engine.drc_helper.set_target_net_names(
            frozenset(nc.net_name for nc in self._board_info.nets.values())
            if self._target_nets is not None else None
        )
        # Rebuild the cached static obs so board_static.nets reflects the change.
        self._obs_cache.board_static = _build_board_static(self._board_info)
        if self._obs_format == "indexed":
            self._obs_cache.static_tables = static_tables_from_dict(
                self._obs_cache.board_static,
            )

    @property
    def board_info(self) -> BoardStatic:
        """Static board metadata and per-net context (cached at reset)."""
        return self._board_info

    @property
    def board_static(self) -> dict:
        """Static context dict (boardlines, nets/pads, obstacles, constraints)."""
        return self._obs_cache.board_static

    @property
    def current_net_id(self) -> int | None:
        """Currently selected net ID (None if no net selected)."""
        return self._dispatcher.current_net_id

    @property
    def routing_mode(self) -> int:
        """Current routing mode (0=MarkObstacles, 1=Shove, 2=Walkaround)."""
        return self._dispatcher.routing_mode

    @property
    def phase(self) -> ActionDispatcher:
        """Deprecated. Use current_net_id and routing_mode directly.

        Returns the dispatcher which has .current_net_id and .routing_mode
        attributes for backward compatibility with code accessing
        env.phase.current_net_id.
        """
        return self._dispatcher

    @property
    def net_geometry(self) -> dict[int, NetGeometry]:
        """Per-net geometric objects (internal dataclass IR).

        .. deprecated::
            Use the observation dict returned by reset()/step() instead.
        """
        return self._obs_cache.net_geometry

    def is_current_net_connected(self) -> bool:
        """Check if the current net has zero remaining ratsnest edges."""
        net_id = self._dispatcher.current_net_id
        if net_id is None:
            return False
        ratsnest = self._engine.get_ratsnest()
        return not any(e.net_code == net_id for e in ratsnest)

    def _get_info(self) -> dict[str, Any]:
        """Build info dict."""
        is_routing = self._engine.is_routing()
        net_id = self._dispatcher.current_net_id
        if net_id is None:
            phase_name = "NET_SELECT"
        elif not is_routing:
            phase_name = "START_ROUTE"
        else:
            phase_name = "ROUTING"

        return {
            "step": self._step_count,
            "phase": phase_name,
            "current_net_id": net_id,
            "track_count": self._engine.get_track_count(),
            "unrouted_count": self._engine.get_unrouted_count(),
            "initial_unconnected": int(self._initial_unconnected),
            "is_routing": is_routing,
        }

    def action_masks(self) -> np.ndarray:
        """Return boolean action mask (SB3 MaskablePPO compatible)."""
        return self._get_action_mask()

    def action_mask_dict(self) -> dict[str, bool]:
        """Return action mask as {action_name: bool} dict (LLM agent friendly)."""
        mask = self._get_action_mask()
        return {name: bool(mask[i]) for i, name in enumerate(ACTION_NAMES)}

    def render(self) -> np.ndarray | None:
        """Render current board state."""
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            from pcb_world.rendering.renderer import PCBRenderer
            self._renderer = PCBRenderer()

        is_routing = self._engine.is_routing()
        net_id = self._dispatcher.current_net_id
        if net_id is None:
            phase_name = "NET_SELECT"
        elif not is_routing:
            phase_name = "START_ROUTE"
        else:
            phase_name = "ROUTING"

        return self._renderer.render(
            engine=self._engine,
            step_info={
                "Step": self._step_count,
                "Phase": phase_name,
                "Net": net_id or "-",
                "Tracks": self._engine.get_track_count(),
                "Unrouted": self._engine.get_unrouted_count(),
            },
        )

    # ------------------------------------------------------------------
    # Checkpoint / Restore (MCTS tree search)
    # ------------------------------------------------------------------

    def checkpoint(self) -> Checkpoint:
        """Capture full env state for MCTS tree search.

        Heavy router state (board + engine config + routing session) goes into
        the C++ handle; light Python episode state is stored as scalars on the
        returned node. Release with :meth:`release_checkpoint` when the node is
        pruned (the C++ clones are not freed by Python GC of the handle int).
        """
        ckpt = Checkpoint(
            engine_handle=self._engine.checkpoint(),
            step_count=self._step_count,
            current_net_id=self._dispatcher.current_net_id,
            routing_mode=self._dispatcher.routing_mode,
            action_history=copy.deepcopy(tuple(self._action_history)),
            # Copy the set (elements are net ids); the reward states are reassigned
            # (never mutated in place) each step, so a reference is snapshot-safe.
            episode_closed_nets=set(self._episode_closed_nets),
            reward_prev_state=self._reward.prev_state,
            wire_via_ref_state=self._wire_via_ref_state,
        )
        ckpt._engine_ref = weakref.ref(self._engine)
        return ckpt

    def restore(
        self,
        ckpt: Checkpoint,
        edge_action: dict | None = None,
        edge_success: bool = True,
        incremental: bool = True,
    ) -> None:
        """Restore env state from a checkpoint.

        ``_action_history`` (read by the next observation) is restored from the
        checkpoint itself, so restore is self-sufficient. ``edge_action`` is an
        optional override: when given, a record for that incoming MCTS edge
        action (+ ``edge_success``) is appended as the newest entry on top of
        the checkpoint's history.

        ``incremental=True`` (default) uses the fast diff-at-restore path
        (updates only the changed tracks in the PNS world, ~40x faster on large
        boards) and yields the same board as the full-swap restore. Pass
        ``incremental=False`` to force the full-swap path (the validation oracle).
        """
        if incremental:
            self._engine.restore_incremental(ckpt.engine_handle)
        else:
            self._engine.restore(ckpt.engine_handle)
        self._step_count = ckpt.step_count
        self._dispatcher.current_net_id = ckpt.current_net_id
        self._dispatcher.routing_mode = ckpt.routing_mode
        # Fresh copy of the set so subsequent net_end .add()s never mutate the
        # node's stored checkpoint (the reward states are replaced, not mutated).
        self._episode_closed_nets = set(ckpt.episode_closed_nets)
        self._reward.prev_state = ckpt.reward_prev_state
        self._wire_via_ref_state = ckpt.wire_via_ref_state

        hist: deque[dict] = deque(
            copy.deepcopy(list(ckpt.action_history)),
            maxlen=self._action_history_len,
        )
        if edge_action is not None:
            edge_type = int(edge_action["action_type"])
            hist.appendleft(_make_history_record(
                edge_action, edge_type, bool(edge_success),
                _action_net_context(edge_action, edge_type, ckpt.current_net_id),
            ))
        self._action_history = hist

    def release_checkpoint(self, ckpt: Checkpoint) -> None:
        """Release a checkpoint's C++ handle (frees its cloned board items).

        Idempotent — also runs automatically when ``ckpt`` is garbage-collected
        (RAII backstop via :meth:`Checkpoint.release`).
        """
        ckpt.release()

    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._crash_logger.close()
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def __del__(self) -> None:
        """Fallback cleanup: guarantees the native RLRouter is released
        before a new PCBWorld is constructed. Without this, GC delay can
        leave two RLRouter instances alive at once, corrupting KiCad global
        state and segfaulting the next start_route (see KiCadEngine.close)."""
        try:
            self.close()
        except Exception:
            pass
