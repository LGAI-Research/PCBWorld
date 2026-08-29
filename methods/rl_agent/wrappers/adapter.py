"""Gym wrapper for PCBWorld — Decoder-Only Transformer policy variant.

This wrapper builds no numpy feature tensors. The policy's internal
``StateTokenizer`` consumes the raw JSON observation dict directly, so the
wrapper's only jobs are:

    1. Pass through the raw dict observation from ``build_json_observation()``.
    2. Expose ``env.action_masks()`` for the policy's action-type masking.
    3. Decode the policy's ``(action_type, pointer_idx, routing_mode)`` output
       into an env action dict, using the **same** net/candidate ordering that
       the tokenizer uses (so pointer indices resolve correctly).

The net ordering MUST match :func:`StateTokenizer._sorted_net_keys` — i.e.
net_code ascending — because the tokenizer emits ``net_indices`` in that order
and the policy's pointer Categorical samples indices into that same pool.

The candidate ordering is the output of :func:`collect_raw_candidates` (with
directional candidates appended when routing) — exactly what the tokenizer
uses in ``_build_candidate_pool``. We rebuild the list here by calling the
same shared function; coordinates are preserved in millimetres.
"""

from __future__ import annotations

import copy
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Env-var hook for routing-mode ablation experiments. When set to "0", "1",
# or "2" (mark_obstacles / shove / walkaround), every routing_mode the
# policy emits is replaced by this value and the mode_mask exposes only the
# forced mode as valid. Read at module import time so forkserver/spawn
# workers inherit the override from the parent's environment. Unset = no
# override (normal behavior).
_FORCE_ROUTING_MODE: int | None = (
    int(os.environ["CADAGENT_FORCE_ROUTING_MODE"])
    if os.environ.get("CADAGENT_FORCE_ROUTING_MODE", "").strip() != ""
    else None
)
if _FORCE_ROUTING_MODE is not None and _FORCE_ROUTING_MODE not in (0, 1, 2):
    raise ValueError(
        f"CADAGENT_FORCE_ROUTING_MODE must be 0/1/2, got {_FORCE_ROUTING_MODE}",
    )

from pcb_world.core.env import PCBWorld
from pcb_world.core.masking import (
    ACT_FINISH,
    ACT_IDLE,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    NUM_ACTIONS,
)
from pcb_world.core.indexed_obs import is_indexed as _is_indexed
from pcb_world.core.action_schema import FALLBACK_ACTION, MODE_WALKAROUND
from pcb_world.vec.candidate_pool import parse_directional_mode
from methods.rl_agent.wrappers import augmentation as _aug
from methods.rl_agent.models.v1 import encoding as _ac
from methods.rl_agent.models.v1 import encoding as _mask
# State-encode pool builders, aliased to the private names ``_refresh_cache``
# uses.
from methods.rl_agent.models.v1.encoding import (
    sorted_net_codes_from_obs as _sorted_net_codes_from_obs,
    cand_mm_list_from_obs as _cand_mm_list_from_obs,
)
from methods.rl_agent.models.v1.spec import NUM_ROUTING_MODES

# Fixed upper bound for the MultiDiscrete action_space pointer slot.
# The real pointer range is per-episode/step variable — this is a stub
# only used to satisfy gym.spaces. Training code should rely on the
# policy's internal Categorical, not the action space.
_MAX_POINTER_STUB = 10_000


class KiCadRLWrapper(gym.Wrapper):
    """Wrap :class:`PCBWorld` for :class:`KiCadRLModel`.

    The wrapper exposes the raw JSON observation dict (not flattened numpy)
    because the policy owns its own tokenizer. Actions coming back from the
    policy have shape ``(3,) int64 = [action_type, pointer_idx, routing_mode]``
    with ``-1`` in unused slots (per ``SLOT_USAGE``); this wrapper decodes
    them into the env's action dict format.

    Args:
        env: A :class:`PCBWorld` instance.
        seed: Seeds the wrapper's numpy Generator (auto net selection,
            augmentation sampling, slot permutation).
        force_walkaround: If True, override the policy's ``routing_mode``
            slot to 2 (Walkaround) for all make_line / make_via / finish
            actions. Default False preserves the decoder's ability to select
            among the 3 routing modes.
        mask_start_point: If True (default), same-point masking: after
            ``start_route(x, y, l)`` that exact candidate ``(x, y, l)``
            is excluded from the pointer pool for every subsequent
            action, until ``_start_route_xy`` is cleared or overwritten.
            Layer-aware — the same xy on a *different* layer stays
            selectable (e.g. a stacked front/back pad pair, as in d3b
            board 0218 net 9). The indices of the masked cands are exposed via
            :meth:`start_route_pointer_indices` so the policy can set
            its pointer logit to ``-inf`` (hard masking, as in
            MaskablePPO).
    Every obs returned by :meth:`reset` / :meth:`step` carries the 5
    act-time mask arrays under the ``"_masks"`` key (action / pointer /
    mode / net_valid / offlayer), computed in-worker right after
    ``_refresh_cache``. Bit-identical to an ``env_method`` query because
    nothing mutates the env state between a step return and the next action —
    mask consumers (``gather_mask_arrays``) read them from the obs instead of
    issuing one IPC round-trip per mask.
    """

    # Default magnitudes for the 5-boolean aug interface. Each is the
    # internal value injected into the per-episode aug dict whenever the
    # corresponding boolean is enabled.  Kept module-level so that an
    # ablation can change them in a single place.
    _AUG_BBOX_SHIFTED_RANGE = 0.3   # scale_x, scale_y ~ U[1 - r, 1 + r]
    _AUG_FLIP_PROB = 0.5            # sign_reflect Bernoulli probability
    _AUG_ROTATE_PROB = 0.5          # axis_swap Bernoulli probability
    _AUG_TRANS_RANGE = 0.2          # nn_dx, nn_dy ~ U[-r, r]
    _AUG_ZOOM_RANGE = 0.1           # nn_zoom ~ U[1 - r, 1 + r]

    def __init__(
        self,
        env: PCBWorld,
        seed: int = 0,
        *,
        force_walkaround: bool = False,
        mask_start_point: bool = True,
        aug_bbox_shifted: bool = False,
        aug_flip: bool = False,
        aug_rotate: bool = False,
        aug_trans: bool = False,
        aug_zoom: bool = False,
        slot_perm: bool = False,
        policy_net_select: bool = False,
        directional_candidates: str | None = None,
        connectivity_filter: bool = True,
        pad_graze_margin_mm: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.env: PCBWorld = env
        self._force_walkaround = force_walkaround
        self._mask_start_point = mask_start_point
        self._policy_net_select = bool(policy_net_select)
        # Connectivity filter over existing-copper candidates (pads / vias /
        # track endpoints; see candidate_pool): drops what the route head is
        # already connected to. Injected into obs["_aug"] so the tokenizer and
        # pointer-decode paths apply the identical candidate set.
        self._connectivity_filter = bool(connectivity_filter)
        # Directional candidate mode: None = 8-dir/0.5mm, a preset name
        # (e.g. "multi_resolution") = 8-dir × that distance ladder, "grid<N>"
        # = 1-layer Grid mode. Parsed here only to fail fast on a typo —
        # consumers read the raw string from obs["_aug"].
        parse_directional_mode(directional_candidates)
        self._directional_candidates: str | None = directional_candidates
        # Pad-graze guard (mm, 0 = off): drops SYNTHESISED directional
        # candidates that land in a same-net pad's graze annulus, where a via /
        # track end would connect by a copper sliver instead of an anchor.
        self._pad_graze_margin_mm = float(pad_graze_margin_mm or 0.0)

        # RNG for external net selection, augmentation and slot permutation.
        self._rng = np.random.default_rng(seed)
        # Static board metadata for _pick_net_id().
        self._net_ids: list[int] = sorted(env.board_info.nets.keys())

        # 5-boolean aug interface.  Each flag is independent; combining
        # flip+rotate spans the D4 dihedral group, +bbox_shifted adds
        # board-frame jitter, +trans adds feature-space translation noise,
        # +zoom adds feature-space uniform scale noise.
        self._aug_bbox_shifted = bool(aug_bbox_shifted)
        self._aug_flip = bool(aug_flip)
        self._aug_rotate = bool(aug_rotate)
        self._aug_trans = bool(aug_trans)
        self._aug_zoom = bool(aug_zoom)

        # Per-episode bbox-shifted state — only meaningful
        # when ``aug_bbox_shifted`` is True.  Pads stay physically fixed;
        # edges are virtually scaled per-axis around (aug_cx, aug_cy).
        self._aug_scale_x = 1.0
        self._aug_scale_y = 1.0
        self._aug_cx = 0.0
        self._aug_cy = 0.0

        # Per-episode orthogonal axis state.  flip/rotate/trans are sampled
        # independently of bbox_shifted so they compose freely.
        self._aug_flip_x = 1
        self._aug_flip_y = 1
        self._aug_nn_dx = 0.0
        self._aug_nn_dy = 0.0
        self._aug_nn_zoom = 1.0
        self._aug_axis_swap = False

        self._aug_external = False

        # Per-episode slot permutation for net-exchangeability augmentation.
        # When enabled, every reset() samples a fresh permutation of
        # [0..N_MAX_SLOTS) which the StateTokenizer applies to every slot id
        # via aug["slot_perm"]. This breaks any spurious dependence on the
        # raw sorted-net-id ↔ slot-embedding pairing.
        self._slot_perm_enabled = bool(slot_perm)
        self._slot_perm: list[int] | None = None

        # Placeholder spaces — the real obs is a nested dict and the real
        # action is the policy's (3,) int64 tensor with per-episode-variable
        # pointer range. Gym spaces are stubs for API compatibility only.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [NUM_ACTIONS, _MAX_POINTER_STUB, NUM_ROUTING_MODES],
        )

        # Per-step cache populated in reset() / step().
        self._last_obs: dict = {}
        self._sorted_net_codes: list[int] = []
        self._cand_mm: list[tuple[float, float, int]] = []
        # Same-point masking state — the last start_route target as the full
        # candidate key (x, y, LAYER; name kept for the snapshot/serve field).
        # Set on ACT_START_ROUTE, kept through ACT_MAKE_LINE / ACT_MAKE_VIA /
        # ACT_FINISH (so a failed finish doesn't allow restarting from the
        # same pad next step), cleared on ACT_NET_END / ACT_NET_SELECT.
        self._start_route_xy: tuple[float, float, int] | None = None

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    # Pad-clearance margin (mm) for the bbox-shifted rejection sampler.
    _AUG_NEW_PAD_MARGIN = 1.0
    _AUG_NEW_MAX_TRIES = 100

    def set_augmentation(self, **kwargs) -> None:
        """Externally override the per-episode augmentation params.

        - bbox-shifted: ``scale_x``, ``scale_y``, ``aug_cx``, ``aug_cy``.
        - orthogonal axes: ``flip_x``, ``flip_y``, ``nn_dx``, ``nn_dy``,
          ``nn_zoom``, ``axis_swap``.

        Omitted keys keep their current value.  Calling this method marks
        the wrapper so that the next ``reset()`` skips its internal
        resampling once.  Useful for GRPO where all envs in a group share
        the same per-iter augmentation.
        """
        if "scale_x" in kwargs:
            self._aug_scale_x = float(kwargs["scale_x"])
            self._aug_scale_y = float(kwargs["scale_y"])
            self._aug_cx = float(kwargs["aug_cx"])
            self._aug_cy = float(kwargs["aug_cy"])
        if "flip_x" in kwargs:
            self._aug_flip_x = int(kwargs["flip_x"])
        if "flip_y" in kwargs:
            self._aug_flip_y = int(kwargs["flip_y"])
        if "nn_dx" in kwargs:
            self._aug_nn_dx = float(kwargs["nn_dx"])
        if "nn_dy" in kwargs:
            self._aug_nn_dy = float(kwargs["nn_dy"])
        if "nn_zoom" in kwargs:
            self._aug_nn_zoom = float(kwargs["nn_zoom"])
        if "axis_swap" in kwargs:
            self._aug_axis_swap = bool(kwargs["axis_swap"])

        self._aug_external = True

    def _sample_new_aug(self, raw_obs: dict) -> None:
        """Rejection-sample (scale_x, scale_y, aug_cx, aug_cy) so every pad
        of the current episode stays inside the virtually scaled edge
        rectangle with _AUG_NEW_PAD_MARGIN clearance. Math in
        :func:`methods.rl_agent.wrappers.augmentation.sample_bbox_shifted`."""
        (
            self._aug_scale_x,
            self._aug_scale_y,
            self._aug_cx,
            self._aug_cy,
        ) = _aug.sample_bbox_shifted(
            self._rng,
            raw_obs,
            range_=self._AUG_BBOX_SHIFTED_RANGE,
            margin=self._AUG_NEW_PAD_MARGIN,
            max_tries=self._AUG_NEW_MAX_TRIES,
        )

    def _resample_augmentation(self, raw_obs: dict | None = None) -> None:
        # Slot permutation is independent of the affine-aug external override.
        if self._slot_perm_enabled:
            from pcb_world.vec.slots import N_MAX_SLOTS
            self._slot_perm = self._rng.permutation(N_MAX_SLOTS).tolist()
        else:
            self._slot_perm = None

        if self._aug_external:
            self._aug_external = False
            return

        # Orthogonal axes — sampled independently of bbox_shifted so they
        # can be combined freely.
        if self._aug_rotate:
            self._aug_axis_swap = bool(
                self._rng.random() < self._AUG_ROTATE_PROB
            )
        else:
            self._aug_axis_swap = False
        if self._aug_flip:
            p = self._AUG_FLIP_PROB
            self._aug_flip_x = -1 if self._rng.random() < p else 1
            self._aug_flip_y = -1 if self._rng.random() < p else 1
        else:
            self._aug_flip_x = 1
            self._aug_flip_y = 1
        if self._aug_trans:
            r = self._AUG_TRANS_RANGE
            self._aug_nn_dx = float(self._rng.uniform(-r, r))
            self._aug_nn_dy = float(self._rng.uniform(-r, r))
        else:
            self._aug_nn_dx = 0.0
            self._aug_nn_dy = 0.0
        if self._aug_zoom:
            r = self._AUG_ZOOM_RANGE
            self._aug_nn_zoom = float(self._rng.uniform(1.0 - r, 1.0 + r))
        else:
            self._aug_nn_zoom = 1.0

        if self._aug_bbox_shifted:
            assert raw_obs is not None, "bbox_shifted aug sampling needs raw_obs"
            self._sample_new_aug(raw_obs)
        else:
            self._aug_scale_x = 1.0
            self._aug_scale_y = 1.0
            self._aug_cx = 0.0
            self._aug_cy = 0.0

    def _inject_aug(self, raw_obs: dict) -> dict:
        """Attach the current aug params under '_aug' on the obs dict.

        The StateTokenizer reads this key (if present) to parameterize its
        coordinate normalization. The value never enters the token feature
        stream — only the already-transformed coordinates do.
        """
        raw_obs["_aug"] = _aug.build_aug_dict(
            bbox_shifted=self._aug_bbox_shifted,
            scale_x=self._aug_scale_x,
            scale_y=self._aug_scale_y,
            cx=self._aug_cx,
            cy=self._aug_cy,
            axis_swap=self._aug_axis_swap,
            flip_x=self._aug_flip_x,
            flip_y=self._aug_flip_y,
            nn_dx=self._aug_nn_dx,
            nn_dy=self._aug_nn_dy,
            nn_zoom=self._aug_nn_zoom,
            slot_perm=self._slot_perm,
            directional_candidates=self._directional_candidates,
            connectivity_filter=self._connectivity_filter,
            route_start_xy=self._start_route_xy,
            cluster_keys=self._head_cluster_keys(raw_obs),
            pad_graze_margin_mm=self._pad_graze_margin_mm,
        )
        return raw_obs

    def _head_cluster_keys(self, raw_obs: dict) -> tuple | None:
        """Engine-resolved connectivity cluster of the route head, as
        ``(x, y, human_layer)`` keys — what ``collect_raw_candidates`` drops.
        Returned SORTED and as plain tuples so the obs stays JSON-dumpable
        (the e2e obs-equality guards serialize it) and byte-comparable.

        The wrapper owns the engine handle, so the query lives here and the
        candidate pool stays a pure obs function (the tokenizer runs in the
        trainer process, where no engine exists). Queried fresh every
        reset/step: the head's cluster GROWS as copper is committed, and each
        make_line commits before returning, so this is always the post-action
        truth. ``None`` when the filter is off, no route is active, the engine
        finds no copper under the head (nothing to be connected to), or the
        wrapped env exposes no engine at all (minimal test doubles — the filter
        simply stays inactive there).
        """
        if not self._connectivity_filter:
            return None
        rh = raw_obs.get("router_head") or {}
        if not rh.get("is_routing", False):
            return None
        xy = rh.get("current_xy")
        layer = rh.get("current_layer", -1)
        if xy is None or layer is None or int(layer) < 0:
            return None
        engine = getattr(self.env, "_engine", None)
        query = getattr(engine, "get_connected_points", None)
        if query is None:
            return None
        pts = query(float(xy[0]), float(xy[1]), int(layer))
        if not pts:
            return None
        return tuple(sorted(
            {(round(float(x), 3), round(float(y), 3), int(lay)) for x, y, lay in pts}
        ))

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        raw_obs, info = self.env.reset(seed=seed, options=options)
        self._start_route_xy = None
        self._resample_augmentation(raw_obs)
        # CRITICAL: aug must be attached BEFORE _refresh_cache so the
        # wrapper's `_cand_mm` is built with the same directional_candidates /
        # transform parameters the tokenizer sees. Otherwise the wrapper falls
        # back to the default mode (8-dir × 0.5mm) while the tokenizer uses
        # the configured mode → pointer_idx maps to mismatched coordinates.
        aug_obs = self._inject_aug(raw_obs)
        self._refresh_cache(aug_obs)
        self._attach_masks(aug_obs)
        return aug_obs, info

    def step(self, action):
        at, ptr, mode = self._unpack_action(action)
        # _decode_action turns an out-of-range pointer into the idle fallback
        # (FALLBACK_ACTION) so the env never sees garbage coords — the same
        # invalid-input path the LLM branch uses.
        env_action = self._decode_action(at, ptr, mode)

        # Track _start_route_xy for same-point masking, keyed on the DECODED
        # action so it stays in sync with what the env actually executes. This
        # MUST happen before env.step so the NEXT step's
        # start_route_pointer_indices sees the just-committed start point.
        # An OOR pointer decodes to idle, which leaves the start point
        # unchanged.
        decoded_at = env_action["action_type"]
        if decoded_at == ACT_START_ROUTE:
            self._start_route_xy = (
                float(env_action["x_mm"]), float(env_action["y_mm"]),
                int(env_action["layer"]),
            )
        elif decoded_at not in (ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_FINISH, ACT_IDLE):
            self._start_route_xy = None

        raw_obs, reward, terminated, truncated, info = self.env.step(env_action)
        # See reset() for why aug must be attached before _refresh_cache.
        aug_obs = self._inject_aug(raw_obs)
        self._refresh_cache(aug_obs)
        self._attach_masks(aug_obs)
        return aug_obs, reward, terminated, truncated, info

    def _attach_masks(self, obs: dict) -> None:
        """Embed the 5 act-time mask arrays under ``obs["_masks"]``.

        Must run after :meth:`_refresh_cache` (pointer indices read
        ``_cand_mm``). The state right after a step/reset return equals the
        state at the next act time (no mutation happens in between), so
        these arrays are bit-identical to what an ``env_method`` query
        would return at act time (regression guard:
        tests/test_mask_in_obs.py).
        """
        obs["_masks"] = {
            "action": self.action_masks(),
            "pointer": self.start_route_pointer_indices(),
            "mode": self.mode_mask(),
            "net_valid": self.net_valid_mask(),
            "offlayer": self.offlayer_pointer_indices(),
        }

    # ------------------------------------------------------------------
    # MCTS checkpoint (RL L2): wrapper-owned path-dependent state
    # ------------------------------------------------------------------

    def snapshot_mcts_state(self, obs_cache: bool = True) -> dict:
        """Capture the RL-wrapper per-env state that an MCTS restore of the
        inner ``PCBWorld`` (L1) does not cover. Parallel to the LLM manager's
        ``snapshot_memory``.

        The policy is Markov, so the only path-dependent state is the
        auto-net-select RNG (used only when ``_policy_net_select=False``) and
        ``_start_route_xy`` (same-point masking); augmentation params are
        per-episode constants.

        ``obs_cache`` additionally carries the derived obs bundle
        (``_last_obs`` + the pointer-decode tables ``_sorted_net_codes`` /
        ``_cand_mm``). It is pure recomputation — the board restores bit-exactly
        and the bundle is a function of the board — but a re-derivation costs a
        full ``PCBWorld._get_obs`` plus an engine connectivity query plus a
        candidate-pool rebuild, and MCTS restores once per simulation. Measured
        on 0344_mavbridge (gumbel n_sim 32): ``_get_obs`` alone was 14.3% of the
        rollout, half of it on the restore path. Carrying the bundle trades that
        for one reference per live tree node, dropped by
        ``RLSearchEnv.release``. Pass ``obs_cache=False`` for a snapshot that
        re-derives on restore instead.
        """
        snap = {
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "start_route_xy": self._start_route_xy,
        }
        # _last_obs only exists once _refresh_cache has run (first reset)
        if obs_cache and getattr(self, "_last_obs", None) is not None:
            # references, not copies: _refresh_cache rebinds these three
            # together and never mutates them in place
            snap["obs_cache"] = (
                self._last_obs, self._sorted_net_codes, self._cand_mm,
            )
        return snap

    def restore_mcts_state(self, snap: dict) -> bool:
        """Restore the wrapper state captured by :meth:`snapshot_mcts_state`,
        in lockstep with the inner env's restore.

        Returns True when the snapshot carried an obs bundle and it was
        installed — the caller can then skip re-deriving it. False means the
        caller must rebuild the obs/pointer tables itself.
        """
        self._rng.bit_generator.state = copy.deepcopy(snap["rng_state"])
        self._start_route_xy = snap["start_route_xy"]
        cached = snap.get("obs_cache")
        if cached is None:
            return False
        self._last_obs, self._sorted_net_codes, self._cand_mm = cached
        return True

    def action_masks(self) -> np.ndarray:
        """Return the env's action-type mask, ``(NUM_ACTIONS,) bool``."""
        return self.env.action_masks()

    def save_pcb(self, output_path: str) -> str:
        """Persist the underlying engine's current board to ``output_path``.

        Used by parallel rollout dump scripts: the wrapper lives inside a
        :class:`SubprocDecoderVecEnv` worker, so the main process can only
        reach the engine through ``pool.env_method("save_pcb", path,
        indices=[i])``.
        """
        self.env._engine.save(output_path)
        return output_path

    def eval_inline_drc(self, **kwargs) -> dict:
        """Run canonical DRC scoring on the live routed engine.

        Used by serial rollout with ``--inline-drc on``: avoids the
        save-and-reload cost of post-hoc evaluation by reading directly from
        the engine that just finished routing. Thin delegation to
        :func:`eval.metrics.compute_metrics_inline` (the one scoring
        convention both branch wrappers expose over ``env_method`` —
        non-destructive, u_0 from the env's reset-time capture), so the
        caller can ``save_pcb`` either before or after.
        """
        from eval.metrics import compute_metrics_inline
        return compute_metrics_inline(self.env, **kwargs)

    def mode_mask(self) -> np.ndarray:
        """Return ``(3,)`` bool mask over routing modes.

        When ``force_walkaround=True``, only Walkaround (index 2) is
        allowed — the policy's mode logits for the other two modes are
        set to ``-inf`` before sampling. When ``force_walkaround=False``
        (default), all three modes are permitted and no logit masking
        is applied.

        Env-var override: ``CADAGENT_FORCE_ROUTING_MODE`` (0/1/2) wins
        over both flags for ablation experiments.
        """
        return _mask.mode_mask(
            force_routing_mode=_FORCE_ROUTING_MODE,
            force_walkaround=self._force_walkaround,
        )

    def offlayer_pointer_indices(self) -> np.ndarray:
        """Cand-pool indices on a layer OTHER than the router head's — not legal
        ``make_line`` targets.

        ``make_line`` routes on the head's current layer and cannot change it:
        the dispatcher takes only ``x_mm``/``y_mm`` (``_decode_action``) and
        ``fix_route`` is given ``expected_layer=get_current_layer()``, which
        rejects a commit that lands anywhere else as stuck. So a candidate at
        ``(x, y, L != current)`` is not a different target — it routes to the SAME
        ``(x, y)`` on the current layer, i.e. it duplicates the same-layer
        candidate when one exists.

        That duplication defeats the connectivity filter. ``collect_raw_candidates``
        drops a head-connected point by its ``(x, y, layer)`` key, so a thru-hole
        pad (expanded to one candidate per copper layer) loses only the connected
        layer while the same xy on another layer survives — and routing there
        retraces committed copper, which PNS commits as nothing. Measured share of
        PAD candidates coming back ``valid_empty``: make_line **73.8-77.3%** across
        three d3b boards.

        ``make_via`` is deliberately NOT restricted this way: changing layer is its
        purpose, PNS picks the layer pair itself, and its reachable set is every
        layer. Only the make_line row consumes this mask.

        Cannot empty the pool: directional candidates are generated AT the head's
        current layer under the same ``is_routing`` condition, so a routing state
        always has same-layer candidates. Empty ``(0,) int64`` when not routing or
        when nothing is off-layer.
        """
        rh = (self._last_obs or {}).get("router_head") or {}
        if not rh.get("is_routing", False):
            return np.zeros((0,), dtype=np.int64)
        cur = int(rh.get("current_layer", -1))
        if cur < 0 or not self._cand_mm:
            return np.zeros((0,), dtype=np.int64)
        off = [i for i, (_x, _y, lay) in enumerate(self._cand_mm) if int(lay) != cur]
        return np.asarray(off, dtype=np.int64)

    def via_blocked_pointer_indices(self) -> np.ndarray:
        """Cand-pool indices a ``make_via`` may not land on — the engine's own
        ``via_on_thru_pad`` predicate, evaluated WITHOUT stepping.

        A via aimed inside a thru-hole pad is refused by
        ``KiCadEngine.pad_block_reason(for_via=True)`` before the router is
        touched: the pad IS the layer bridge, so PNS drops the pending via while
        still committing the route (a half-executed make_via, which this gate
        prevents). The pool cannot pre-filter it on its own — one shared
        candidate list serves make_line and make_via, and a thru-pad centre is a
        VALID make_line target and an INVALID make_via one, so the distinction
        only exists per action type.

        Why it is worth computing here: MCTS discovers the refusal by stepping,
        which costs a full ``env.step`` per candidate per node. Measured on d3b,
        make_via accounts for 52.7% of popped children — 387 wasted engine steps
        per rollout, 11.3% of all ``env.step`` calls — a cost plain rollouts never
        pay because they never enumerate candidates.

        The test is purely geometric (distance to a thru pad ≤ its radius) and
        independent of the current net, unlike the ``pad_graze`` half of
        ``pad_block_reason``. ``KiCadEngine._thru_pad_geometry`` is cached per
        board, so this is O(cands x thru_pads) of arithmetic — far below one step.

        Returns an empty ``(0,) int64`` array when nothing is blocked.
        """
        eng = getattr(self.env, "_engine", None)
        if eng is None or not self._cand_mm:
            return np.zeros((0,), dtype=np.int64)
        try:
            r = eng.route_item_radius_mm(for_via=True)
            blocked = [
                i for i, (x, y, _l) in enumerate(self._cand_mm)
                if eng.pad_block_reason(x, y, item_radius_mm=r, for_via=True)
                == "via_on_thru_pad"
            ]
        except Exception:                      # engine without the predicate
            return np.zeros((0,), dtype=np.int64)
        return np.asarray(blocked, dtype=np.int64)

    def start_route_pointer_indices(self) -> np.ndarray:
        """Return cand-pool indices matching ``_start_route_xy`` —
        same ``(x, y)`` **and same layer**.

        Layer-aware masking: after ``start_route(x, y, l)`` only the
        exact candidate ``(x, y, l)`` is excluded — the same xy on a
        different layer stays selectable. This keeps a thru-hole pad's
        multi-layer candidates distinct from a stacked front/back SMD pad
        pair (zero-length ratsnest), whose whole candidate pool sits at one
        xy. Cost: same-point re-tries on another layer of a thru-hole pad
        are legal (wasted step, reward-punished, not a crash).

        The trainer stacks the per-env results into a ``(B, K_max)``
        int64 tensor (right-padded with ``-1``) and passes it to
        :meth:`KiCadRLModel.act_and_value` /
        :meth:`KiCadRLModel.evaluate_actions_and_value` as the
        ``pointer_masks`` kwarg. The policy sets every non-(-1) entry
        in each row to ``-inf`` — hard masking, as in MaskablePPO.

        Returns an empty ``(0,) int64`` array when:

        * ``mask_start_point=False`` (feature disabled)
        * No active start point (``_start_route_xy is None``)
        * No cand coordinate matches (rare: routing modes rebuilt the
          cand pool and the start pad is no longer directly listed).
        """
        return _mask.start_route_pointer_indices(
            mask_start_point=self._mask_start_point,
            start_route_xy=self._start_route_xy,
            cand_mm=self._cand_mm,
        )

    # ------------------------------------------------------------------
    # Cache (pointer → mm/net_id mapping)
    # ------------------------------------------------------------------
    def _refresh_cache(self, raw_obs: dict) -> None:
        """Recompute pointer-index → net_code / (x,y,layer) tables.

        Must be called after every reset/step so that subsequent
        ``_decode_action`` calls see the tokenizer-equivalent ordering
        for the *current* observation.
        """
        self._last_obs = raw_obs
        self._sorted_net_codes = _sorted_net_codes_from_obs(raw_obs)
        self._cand_mm = _cand_mm_list_from_obs(raw_obs)
        # Diagnostic guard (not a fallback): an empty candidate pool while a
        # net is selected cannot happen in normal operation — a selectable
        # net always has pads (confirmed by an exhaustive scan of 275 d3b
        # boards), and directional candidates are added by in-progress
        # routing geometry. Fail immediately here (where the obs is still
        # live) with full env context — closer to the root cause than a
        # policy-side guard (net.py) would be, so the offending board/net
        # can be pinpointed.
        rh = raw_obs.get("router_head", {})
        cur_net = rh.get("current_net", -1)
        if cur_net is not None and cur_net > 0 and not self._cand_mm:
            from pcb_world.diag import guard_fail

            guard_fail(
                "empty_cand_pool",
                "empty candidate pool with a net selected — "
                f"net_code={cur_net} is_routing={rh.get('is_routing')} "
                f"step={rh.get('step')} "
                f"board={getattr(self.env, 'board_path', '?')} "
                f"closed_nets={raw_obs.get('closed_nets')}",
                raw_obs=raw_obs,
                sorted_net_codes=self._sorted_net_codes,
                board_path=getattr(self.env, "board_path", None),
            )

    # Exposed for tests / training loops that may want to inspect
    # the tokenizer-compatible orderings without re-computing them.
    @property
    def sorted_net_codes(self) -> list[int]:
        return list(self._sorted_net_codes)

    @property
    def cand_mm_list(self) -> list[tuple[float, float, int]]:
        return list(self._cand_mm)

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack_action(action) -> tuple[int, int, int]:
        """Normalise a policy action to a ``(action_type, ptr, mode)`` tuple.
        Delegates to :func:`methods.rl_agent.models.v1.encoding.unpack_action`."""
        return _ac.unpack_action(action)

    def _decode_action(
        self, action_type: int, pointer_idx: int, routing_mode: int,
    ) -> dict:
        """Build the env action dict from the policy's (3,) output.

        Handles each action_type per ``SLOT_USAGE``. An out-of-range
        pointer (start_route / make_line / make_via, or net_select under
        ``policy_net_select``) decodes to the idle fallback
        (``FALLBACK_ACTION``): engine-safe (idle carries no coords) and routed
        through the env's parse_fail penalty — the SAME invalid-input path the
        LLM branch uses, so neither branch short-circuits the env.
        """
        if action_type in (ACT_START_ROUTE, ACT_MAKE_LINE, ACT_MAKE_VIA) and not (
            0 <= pointer_idx < len(self._cand_mm)
        ):
            return dict(FALLBACK_ACTION)
        if (
            action_type == ACT_NET_SELECT
            and self._policy_net_select
            and not (0 <= pointer_idx < len(self._sorted_net_codes))
        ):
            return dict(FALLBACK_ACTION)

        if action_type == ACT_NET_SELECT:
            if self._policy_net_select:
                # Policy-driven: map pointer → net_id via the tokenizer's
                # sorted net pool (pointer validated above).
                net_id = self._pointer_to_net_id(pointer_idx)
                return {"action_type": ACT_NET_SELECT, "net_id": net_id}
            else:
                # Env-driven: random pick among unrouted nets.
                net_id = self._pick_net_id()
                return {"action_type": ACT_NET_SELECT, "net_id": net_id}

        if action_type == ACT_START_ROUTE:
            x_mm, y_mm, layer = self._pointer_to_cand(pointer_idx)
            return {
                "action_type": ACT_START_ROUTE,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "layer": layer,
            }

        if action_type == ACT_NET_END:
            return {"action_type": ACT_NET_END}

        if action_type == ACT_MAKE_LINE:
            x_mm, y_mm, _layer = self._pointer_to_cand(pointer_idx)
            # NB: no "layer" key — env dispatcher doesn't accept one for
            # make_line and uses the router head's current layer instead.
            return {
                "action_type": ACT_MAKE_LINE,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "routing_mode": self._resolve_mode(routing_mode),
            }

        if action_type == ACT_MAKE_VIA:
            x_mm, y_mm, _layer = self._pointer_to_cand(pointer_idx)
            return {
                "action_type": ACT_MAKE_VIA,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "routing_mode": self._resolve_mode(routing_mode),
            }

        if action_type == ACT_FINISH:
            return {
                "action_type": ACT_FINISH,
                "routing_mode": self._resolve_mode(routing_mode),
            }

        # Unknown action_type — let env surface the error via -1 reward.
        return {"action_type": int(action_type)}

    # ------------------------------------------------------------------
    # External net selection
    # ------------------------------------------------------------------
    def _pick_net_id(self) -> int:
        """Pick a net to route (prefer unrouted nets).

        Uses routing_geometry from the cached observation dict to identify
        nets with remaining ratsnest points.
        """
        if not self._net_ids:
            return 0

        closed = set(self._last_obs.get("closed_nets") or [])
        unrouted_nets = []
        if _is_indexed(self._last_obs):
            rg = self._last_obs["routing_geometry"]
            codes, rat_count = rg["net_code"], rg["rat_count"]
            for nid in self._net_ids:
                p = int(np.searchsorted(codes, nid))
                if (p < len(codes) and codes[p] == nid
                        and rat_count[p] > 0 and nid not in closed):
                    unrouted_nets.append(nid)
        else:
            routing_geom = self._last_obs.get("routing_geometry", {})
            for nid in self._net_ids:
                net_key = f"net_{nid}"
                net_geom = routing_geom.get(net_key)
                if net_geom is not None and net_geom.get("points") and nid not in closed:
                    unrouted_nets.append(nid)
        if unrouted_nets:
            return int(self._rng.choice(unrouted_nets))

        return int(self._rng.choice(self._net_ids))

    def net_valid_mask(self) -> np.ndarray:
        """Return ``(M,) bool`` — True for nets in the current sorted pool
        that still have ratsnest points (i.e. are valid targets for
        ACT_NET_SELECT). Order matches ``self._sorted_net_codes``, which is
        identical to the tokenizer's net pointer pool.

        Fallback: if every net in the pool is fully routed, returns a mask
        that is all-True so the policy still has at least one legal choice
        (the env will reject the selection with -1 reward, consistent with
        the random-pick fallback in :meth:`_pick_net_id`).

        Only meaningful when ``policy_net_select=True``; otherwise callers
        should ignore this and let the env do its own selection.
        """
        return _mask.net_valid_mask(
            sorted_net_codes=self._sorted_net_codes,
            last_obs=self._last_obs,
        )

    # ------------------------------------------------------------------
    # Pointer resolution
    # ------------------------------------------------------------------
    def _pointer_to_net_id(self, pointer_idx: int) -> int:
        """Map a pointer index into the tokenizer net pool → env net_id.

        Out-of-range → ``-1``. In normal use ``_decode_action`` validates the
        pointer first and returns the idle fallback for OOR, so this sentinel
        only matters for direct callers.
        """
        return _ac.pointer_to_net_id(self._sorted_net_codes, pointer_idx)

    def _pointer_to_cand(
        self, pointer_idx: int,
    ) -> tuple[float, float, int]:
        """Map a pointer index into the candidate pool → (x_mm, y_mm, layer).
        Delegates to :func:`methods.rl_agent.models.v1.encoding.pointer_to_cand`."""
        return _ac.pointer_to_cand(self._cand_mm, pointer_idx)

    def _resolve_mode(self, routing_mode: int) -> int:
        """Return the routing_mode that should be sent to the env.

        Env-var override ``CADAGENT_FORCE_ROUTING_MODE`` (0/1/2) wins; then
        ``force_walkaround=True`` returns Walkaround (2); otherwise delegates
        to :meth:`_clamp_mode` which sanitises the policy's output.
        """
        if _FORCE_ROUTING_MODE is not None:
            return _FORCE_ROUTING_MODE
        if self._force_walkaround:
            return MODE_WALKAROUND
        return self._clamp_mode(routing_mode)

    @staticmethod
    def _clamp_mode(routing_mode: int) -> int:
        """Clamp routing_mode to ``[0, 2]``. Delegates to
        :func:`methods.rl_agent.models.v1.encoding.clamp_mode`."""
        return _ac.clamp_mode(routing_mode, NUM_ROUTING_MODES)
