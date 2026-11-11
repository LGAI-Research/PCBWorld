"""The action constraints every agent routes under — one rule set for RL and LLM.

Pure functions of an observation (no engine, no torch): the candidate pool a pointer
policy chooses from and the act-time masks over it (routing mode, same-point,
net-valid, off-layer, off-board). The RL codec (``methods.rl_agent.models.v1.encoding``)
re-exports them; the env applies the same functions, so every caller of
``PCBWorld.step`` meets the same rules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from pcb_world.core.action_schema import (
    ACT_FINISH, ACT_IDLE, ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_NET_SELECT, ACT_START_ROUTE,
    MODE_LETTER_TO_INT, MODE_WALKAROUND,
)
from pcb_world.core.indexed_obs import is_indexed
from pcb_world.vec.candidate_pool import (
    CTYPE_DIRECTIONAL,
    build_directional_candidates,
    collect_raw_candidates,
)

NUM_ROUTING_MODES = len(MODE_LETTER_TO_INT)


# ===========================================================================
# Net order and candidate pool (the pointer index spaces)
# ===========================================================================
def _parse_net_code(net_key: str) -> int:
    return int(net_key.split("_", 1)[1])


def _sorted_net_keys(d: dict) -> list[str]:
    return sorted(d.keys(), key=_parse_net_code)


def sorted_net_codes_from_obs(obs: dict) -> list[int]:
    """Return net codes in the exact order used by StateTokenizer.

    Mirrors ``_sorted_net_keys`` — numeric ascending.
    """
    if is_indexed(obs):
        return sorted(int(c) for c in obs["board_static"]["net_code"])
    nets = (obs.get("board_static") or {}).get("nets") or {}
    codes: list[int] = []
    for key in nets.keys():
        try:
            codes.append(int(key.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    codes.sort()
    return codes


def cand_raw_list_from_obs(obs: dict) -> list[tuple[float, float, int, int]]:
    """Rebuild the raw candidate ``(x_mm, y_mm, layer, ctype)`` list in
    tokenizer order.

    Uses the same shared helpers as ``StateTokenizer._build_candidate_pool``:
    ``collect_raw_candidates`` + ``build_directional_candidates`` (when routing).
    The tokenizer's ``cand_mm_list`` and this function's output must be
    tuple-wise identical for the wrapper's pointer decode to be correct.

    ``obs["_aug"]["directional_candidates"]`` selects the directional mode
    (None = 8-dir / 0.5mm, preset name = 8-dir distance ladder, "grid<N>" =
    1-layer Grid mode) — see ``candidate_pool.parse_directional_mode``.
    """
    rh = obs.get("router_head", {})
    current_net_id = rh.get("current_net", -1)
    if current_net_id is None or current_net_id <= 0:
        current_net_id = None

    extra = None
    if rh.get("is_routing", False):
        head_xy = rh.get("current_xy", [0.0, 0.0])
        aug = obs.get("_aug") or {}
        extra = build_directional_candidates(
            (head_xy[0], head_xy[1]), int(rh.get("current_layer", 1)),
            mode=aug.get("directional_candidates"),
        )

    return collect_raw_candidates(obs, current_net_id, extra)


def cand_mm_list_from_obs(obs: dict) -> list[tuple[float, float, int]]:
    """:func:`cand_raw_list_from_obs` without the ctype column — the wrapper's
    pointer-decode table."""
    return [(x, y, layer) for (x, y, layer, _ctype) in cand_raw_list_from_obs(obs)]


# ===========================================================================
# Masks — what each action row may target
# ===========================================================================
def mode_mask(*, force_routing_mode: int | None, force_walkaround: bool) -> np.ndarray:
    """``(3,)`` bool mask over routing modes: only the forced mode (the
    :func:`forced_routing_mode` rule — ``force_routing_mode``, the env-var override,
    wins; then ``force_walkaround`` = Walkaround), else all three."""
    forced = _forced_mode(force_routing_mode, force_walkaround)
    if forced is None:
        return np.ones(NUM_ROUTING_MODES, dtype=bool)
    mask = np.zeros(NUM_ROUTING_MODES, dtype=bool)
    mask[forced] = True
    return mask


def start_route_pointer_indices(
    *,
    mask_start_point: bool,
    start_route_xy: tuple[float, float, int] | None,
    cand_mm: list[tuple[float, float, int]],
) -> np.ndarray:
    """Cand-pool indices matching ``start_route_xy`` — same ``(x, y)`` AND
    same ``layer``.

    Layer-aware on purpose: the exclusion key must equal the candidate
    identity ``(x, y, layer)``. An xy-only match wipes ALL layer variants
    of the point, and for a net whose every candidate shares one xy
    (stacked front/back SMD pad pair — zero-length ratsnest, e.g. d3b
    0218 net 9) that empties the pool and start_route's pointer row goes
    all-(-inf). Excluding only the started layer keeps the sibling-layer
    candidate selectable.

    Empty ``(0,) int64`` when masking is disabled, no active start point, or
    nothing matches.
    """
    if not mask_start_point or start_route_xy is None:
        return np.zeros((0,), dtype=np.int64)
    sx, sy, sl = start_route_xy
    idxs = [
        i for i, (cx, cy, layer) in enumerate(cand_mm)
        if _same_xy(cx, cy, sx, sy) and layer == sl
    ]
    return np.asarray(idxs, dtype=np.int64)


def net_valid_mask(
    *,
    sorted_net_codes: list[int],
    last_obs: dict,
    net_reselect: bool = False,
) -> np.ndarray:
    """``(M,) bool`` — True for nets that still have ratsnest points AND were
    not consumed by a ``net_end`` this episode (``obs["closed_nets"]``), in
    ``sorted_net_codes`` order — the valid ACT_NET_SELECT targets. By default
    a net_end retires its net for the episode (give-up under permissive
    masking rules); once every routable net is closed the env terminates.

    ``net_reselect`` (the env knob, :attr:`PCBWorld.net_reselect`) drops the
    closed condition: every net with ratsnest points is a target, closed or
    not, and the env reopens a closed net when it is selected. Pass the value
    of the env the observation came from — a mask that keeps closed nets for
    an env that does not reopen them changes the episode semantics.

    Invariant: a valid-net count of 0 occurs only in the state right before
    termination, so all-False cannot appear at a net_select decision point on
    the normal path. If all-False ever reaches the policy, net.py's
    all-(-inf) pointer-row guard fails loudly with context.
    """
    M = len(sorted_net_codes)
    if M == 0:
        return np.zeros((0,), dtype=bool)
    closed = set() if net_reselect else set(last_obs.get("closed_nets") or [])
    mask = np.zeros((M,), dtype=bool)
    if is_indexed(last_obs):
        # Array path: "net has ratsnest points" == rat_count > 0 for the
        # net's row (net_code is ascending — searchsorted lookup).
        rg = last_obs["routing_geometry"]
        codes = rg["net_code"]
        rat_count = rg["rat_count"]
        for i, nid in enumerate(sorted_net_codes):
            p = int(np.searchsorted(codes, nid))
            if (p < len(codes) and codes[p] == nid
                    and rat_count[p] > 0 and nid not in closed):
                mask[i] = True
        return mask
    routing_geom = last_obs.get("routing_geometry") or {}
    for i, nid in enumerate(sorted_net_codes):
        net_geom = routing_geom.get(f"net_{nid}")
        if net_geom is not None and net_geom.get("points") and nid not in closed:
            mask[i] = True
    return mask


def offlayer_pointer_indices(
    *, obs: dict, cand_mm: list[tuple[float, float, int]],
) -> np.ndarray:
    """Cand-pool indices on a layer other than the router head's — blocked on
    the make_line row only (rule and rationale:
    ``KiCadRLWrapper.offlayer_pointer_indices``). Empty ``(0,) int64`` when not
    routing or nothing is off-layer."""
    rh = obs.get("router_head") or {}
    if not rh.get("is_routing", False):
        return np.zeros((0,), dtype=np.int64)
    cur = int(rh.get("current_layer", -1))
    if cur < 0 or not cand_mm:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(
        [i for i, (_x, _y, lay) in enumerate(cand_mm) if int(lay) != cur], dtype=np.int64,
    )


def _board_bbox(obs: dict) -> tuple[float, float, float, float]:
    """The ``board_static`` bbox as ``(x0, y0, x1, y1)``."""
    bs = obs["board_static"]
    x0, y0 = float(bs["bbox_x"]), float(bs["bbox_y"])
    return x0, y0, x0 + float(bs["bbox_w"]), y0 + float(bs["bbox_h"])


def _outside_bbox(bbox: tuple[float, float, float, float], x: float, y: float) -> bool:
    """``(x, y)`` lies outside ``bbox`` (edges inclusive = inside)."""
    x0, y0, x1, y1 = bbox
    return not (x0 <= x <= x1 and y0 <= y <= y1)


def offboard_pointer_indices(
    *,
    offboard_mask: bool,
    obs: dict,
    cand_mm: list[tuple[float, float, int]],
    cand_ctype: list[int],
) -> np.ndarray:
    """Cand-pool indices of DIRECTIONAL candidates outside the ``board_static``
    bbox (edges inclusive) — blocked on every cand row (rule and rationale:
    ``KiCadRLWrapper.offboard_pointer_indices``). Empty ``(0,) int64`` when
    ``offboard_mask`` is off or nothing is off-board."""
    if not offboard_mask or not cand_mm:
        return np.zeros((0,), dtype=np.int64)
    bbox = _board_bbox(obs)
    return np.asarray([
        i for i, ((x, y, _l), ct) in enumerate(zip(cand_mm, cand_ctype))
        if ct == CTYPE_DIRECTIONAL and _outside_bbox(bbox, x, y)
    ], dtype=np.int64)


# Env-var hook for routing-mode ablation experiments. When set to "0", "1",
# or "2" (mark_obstacles / shove / walkaround), every routing_mode the
# policy emits is replaced by this value and the mode_mask exposes only the
# forced mode as valid. Read at module import time so forkserver/spawn
# workers inherit the override from the parent's environment. Unset = no
# override (normal behavior).
FORCE_ROUTING_MODE: int | None = (
    int(os.environ["PCBWORLD_FORCE_ROUTING_MODE"])
    if os.environ.get("PCBWORLD_FORCE_ROUTING_MODE", "").strip() != ""
    else None
)
if FORCE_ROUTING_MODE is not None and FORCE_ROUTING_MODE not in (0, 1, 2):
    raise ValueError(
        f"PCBWORLD_FORCE_ROUTING_MODE must be 0/1/2, got {FORCE_ROUTING_MODE}",
    )


def mask_payload(
    *,
    action_mask: np.ndarray,
    obs: dict,
    cand_mm: list[tuple[float, float, int]],
    cand_ctype: list[int],
    sorted_net_codes: list[int],
    start_route_xy: tuple[float, float, int] | None,
    mask_start_point: bool,
    offboard_mask: bool,
    force_walkaround: bool,
    net_reselect: bool,
) -> dict[str, np.ndarray]:
    """The six act-time mask arrays of one state — the payload
    ``DistInputs.from_payloads`` stacks. One rule set for the wrapper
    (``obs["_masks"]``) and the bare-PCBWorld inference paths (``rl_provider``,
    ``rl_act_server``), so all of them mask the pool they are given as training
    did (the pool itself is :func:`cand_raw_list_from_obs` of that obs)."""
    return {
        "action": np.asarray(action_mask, dtype=bool),
        "pointer": start_route_pointer_indices(
            mask_start_point=mask_start_point, start_route_xy=start_route_xy, cand_mm=cand_mm,
        ),
        "mode": mode_mask(force_routing_mode=FORCE_ROUTING_MODE, force_walkaround=force_walkaround),
        "net_valid": net_valid_mask(
            sorted_net_codes=sorted_net_codes, last_obs=obs, net_reselect=net_reselect,
        ),
        "offlayer": offlayer_pointer_indices(obs=obs, cand_mm=cand_mm),
        "offboard": offboard_pointer_indices(
            offboard_mask=offboard_mask, obs=obs, cand_mm=cand_mm, cand_ctype=cand_ctype,
        ),
    }


# ===========================================================================
# One state's constraints, the start-point rule, the mode rule, the action check
# ===========================================================================
#: How an agent names a target point (EnvConfig ``action_target``): ``candidates`` — a
#: point of the candidate pool, under its masks (the RL action space); ``free`` — any
#: coordinate, under the rules that do not need the pool.
ACTION_TARGETS = ("candidates", "free")


@dataclass(frozen=True)
class ActConstraints:
    """What an agent may do in one state: the candidate pool (the pointer index space,
    tokenizer order), the net order, and the act-time masks over them
    (:func:`mask_payload`). ``cand_aug`` holds the ``_aug`` keys the pool was built
    with — a consumer that rebuilds the pool from the obs (the tokenizer) must see the
    same keys. ``cand_mm`` / ``cand_ctype`` are the two column views of ``cand_raw``."""

    cand_aug: dict
    cand_raw: list
    cand_mm: list
    cand_ctype: list
    sorted_net_codes: list
    masks: dict


def act_constraints(
    obs: dict,
    *,
    action_mask: np.ndarray,
    cand_aug: dict,
    start_route_xy: tuple[float, float, int] | None,
    mask_start_point: bool,
    offboard_mask: bool,
    force_walkaround: bool,
    net_reselect: bool,
) -> ActConstraints:
    """The :class:`ActConstraints` of the state ``obs`` describes."""
    view = {**obs, "_aug": {**(obs.get("_aug") or {}), **cand_aug}}
    raw = cand_raw_list_from_obs(view)
    cand_mm = [(x, y, layer) for (x, y, layer, _c) in raw]
    cand_ctype = [ct for (_x, _y, _l, ct) in raw]
    codes = sorted_net_codes_from_obs(obs)
    masks = mask_payload(
        action_mask=action_mask, obs=view, cand_mm=cand_mm, cand_ctype=cand_ctype,
        sorted_net_codes=codes, start_route_xy=start_route_xy,
        mask_start_point=mask_start_point, offboard_mask=offboard_mask,
        force_walkaround=force_walkaround, net_reselect=net_reselect,
    )
    return ActConstraints(cand_aug=dict(cand_aug), cand_raw=raw, cand_mm=cand_mm,
                          cand_ctype=cand_ctype, sorted_net_codes=codes, masks=masks)


def next_start_route_xy(prev: tuple[float, float, int] | None,
                        action: dict) -> tuple[float, float, int] | None:
    """The same-point state after ``action``: a start_route sets it to its
    ``(x, y, layer)``; make_line / make_via / finish / idle keep it (a failed finish
    must not allow restarting from the same pad); anything else clears it."""
    at = int(action["action_type"])
    if at == ACT_START_ROUTE:
        return float(action["x_mm"]), float(action["y_mm"]), int(action["layer"])
    if at in (ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_FINISH, ACT_IDLE):
        return prev
    return None


def forced_routing_mode(force_walkaround: bool) -> int | None:
    """The one routing mode every routing action uses, or None when the agent picks:
    ``PCBWORLD_FORCE_ROUTING_MODE`` wins, then ``force_walkaround`` (Walkaround)."""
    return _forced_mode(FORCE_ROUTING_MODE, force_walkaround)


def _forced_mode(override: int | None, force_walkaround: bool) -> int | None:
    if override is not None:
        return override
    return MODE_WALKAROUND if force_walkaround else None


def _same_xy(ax: float, ay: float, bx: float, by: float) -> bool:
    """One point to every rule (RL masks and the free-target check): within 0.01 mm."""
    return abs(ax - bx) < 0.01 and abs(ay - by) < 0.01


#: The pool rows each coordinate action reads, by the rule name that blocks them.
_BLOCKING_MASKS = {
    ACT_START_ROUTE: {"same_point": "pointer", "off_board": "offboard"},
    ACT_MAKE_LINE: {"same_point": "pointer", "off_board": "offboard", "off_layer": "offlayer"},
    ACT_MAKE_VIA: {"same_point": "pointer", "off_board": "offboard"},
}


def blocked_candidates(act: ActConstraints, action_type: int) -> dict[str, set[int]]:
    """Pool indices blocked on the ``action_type`` row (start_route / make_line /
    make_via), keyed by the rule that blocks them (the :func:`violation` names)."""
    return {name: set(act.masks[key].tolist())
            for name, key in _BLOCKING_MASKS[action_type].items()}


def allowed_candidates(act: ActConstraints, action_type: int) -> list[int]:
    """Pool indices an ``action_type`` row may target — the complement of
    :func:`blocked_candidates`, in pool order (what an RL pointer may sample)."""
    blocked = set().union(*blocked_candidates(act, action_type).values())
    return [i for i in range(len(act.cand_raw)) if i not in blocked]


def violation(
    action: dict,
    *,
    action_target: str,
    obs: dict,
    act: "ActConstraints | None",
    start_route_xy: tuple[float, float, int] | None,
    mask_start_point: bool,
    offboard_mask: bool,
    net_reselect: bool,
) -> str | None:
    """The name of the rule ``action`` breaks in the state ``obs`` describes, or None.

    * ``net_closed`` — net_select of a net closed by net_end (unless ``net_reselect``).
    * coordinate actions (start_route / make_line / make_via):

      - ``candidates``: the target must be a pool point not blocked on that action's
        row (``act``, required): ``not_a_candidate`` · ``same_point`` · ``off_board`` ·
        ``off_layer`` (make_line only);
      - ``free``: ``same_point`` (the current start point, ``mask_start_point``) and
        ``off_board`` (outside the board bbox, ``offboard_mask``).

    A net that has no ratsnest left, a pad a via cannot land on and the like are the
    dispatcher's and the router's to refuse, as before."""
    at = int(action["action_type"])
    if at == ACT_NET_SELECT:
        if not net_reselect and int(action.get("net_id", -1)) in set(obs.get("closed_nets") or []):
            return "net_closed"
        return None
    if at not in (ACT_START_ROUTE, ACT_MAKE_LINE, ACT_MAKE_VIA):
        return None
    x, y = float(action["x_mm"]), float(action["y_mm"])
    if at == ACT_START_ROUTE:
        layer = int(action["layer"])
    else:
        layer = int((obs.get("router_head") or {}).get("current_layer", -1))
    if action_target == "free":
        if (mask_start_point and start_route_xy is not None
                and _same_xy(x, y, start_route_xy[0], start_route_xy[1])
                and layer == start_route_xy[2]):
            return "same_point"
        if offboard_mask and _outside_bbox(_board_bbox(obs), x, y):
            return "off_board"
        return None
    if action_target != "candidates":
        raise ValueError(f"action_target must be one of {ACTION_TARGETS}, got {action_target!r}")
    if act is None:
        raise ValueError("action_target 'candidates' checks against the state's ActConstraints")
    blocked = blocked_candidates(act, at)
    blocked_any = set().union(*blocked.values())
    # start_route names its layer; make_line / make_via draw from the head's layer, so any
    # pool layer variant of the point is the same target (the RL decode drops the layer).
    hits = [i for i, (cx, cy, cl) in enumerate(act.cand_mm)
            if _same_xy(x, y, cx, cy) and (at != ACT_START_ROUTE or cl == layer)]
    if not hits:
        return "not_a_candidate"
    if any(i not in blocked_any for i in hits):
        return None
    return next(name for name, s in blocked.items() if hits[0] in s)
