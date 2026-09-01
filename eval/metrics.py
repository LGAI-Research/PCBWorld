"""Metric kernel for the central evaluator.

There is one evaluator (:class:`eval.evaluator.Evaluator`); "validation" is just
that evaluator run on a held-out set during training and logged under ``val/*``.
Both the standalone-eval and in-training-validation use cases produce the same
data type, :class:`EvalSummary`, which can then be sent to any sink — a
metric logger (wandb/TensorBoard), CSV/JSON export, or further aggregation.

This module owns two things outright — the per-board metric computation and the
sink-agnostic summary type — so the kernel is self-contained:

  * :func:`compute_metrics` — score a live KiCad engine against design rules
    (engine -> metric dict). Used by the evaluator's disk/inline DRC paths and,
    going forward, by the LLM episode-end scoring path.
  * :class:`EvalSummary` — the canonical, sink-agnostic evaluation summary.
    Holds the aggregated ``overall``/``per_board`` (plus raw ``per_rollout`` for
    export) and projects them to ``<prefix>/<key>`` scalar tags for any
    :class:`~methods._shared.logger.MetricLogger`, or back to an
    :class:`~eval.metrics.EvalResult` for CSV/JSON export.

The single source of *field names* remains
:data:`eval.eval_utils.CADAGENT_VALUE_METRIC_FIELDS`; this module does not
redeclare them. Heavy KiCad imports stay function-local so importing this module
(for projection only) costs nothing.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from eval.aggregation import aggregate_inline
from methods._shared.board_loader import BoardSpec
from configs.loader.schema import DEFAULTS
from eval.eval_utils import (
    CADAGENT_BOOL_VALUE_METRIC_FIELDS,
    CADAGENT_VALUE_METRIC_FIELDS,
)

__all__ = [
    "compute_metrics",
    "evaluate_one",
    "EvalResult",
    "EvalSummary",
    "CADAGENT_VALUE_METRIC_FIELDS",
    "CADAGENT_BOOL_VALUE_METRIC_FIELDS",
]


@dataclass
class EvalResult:
    """Raw output of :func:`eval.rollout.rl.eval_transformer`.

    ``per_rollout`` rows follow :data:`eval.eval_utils.PER_ROLLOUT_FIELDS` (one
    row per board x rollout, including merged DRC-scoring columns when DRC ran).
    ``per_board`` averages shared cadagent value metrics plus, when
    ``selection_mode`` is given, table1-style winner annotation. ``overall`` is a
    flat dict mixing held distribution keys and (optionally) ``selected_*`` keys.
    Wrapped by :class:`EvalSummary` for the sink-facing projection.
    """
    per_rollout: list[dict] = field(default_factory=list)
    per_board: list[dict] = field(default_factory=list)
    overall: dict = field(default_factory=dict)


# ============================================================================
# Per-board metric computation (engine -> metric dict)
# ============================================================================

# Mirror of pcb_world.engine.drc._PROMOTED_ERROR_CODES (canonical) — DRC error codes
# promoted from warning severity.
_PROMOTED_CODES = {12, 13, 37}   # DANGLING_VIA, DANGLING_TRACK, NET_CONFLICT
# Local mirror of pcb_world.engine.drc.DRC_SEVERITY_ERROR/WARNING (canonical). Kept
# local — not imported — so importing eval.metrics stays cheap: this module
# deliberately keeps KiCad imports function-local (see module docstring), and
# importing pcb_world.engine would pull that package in. All three drc mirrors here are
# pinned to the canonical by test_constant_consistency.py
# ::test_eval_metrics_drc_mirrors_match_canonical.
_SEV_ERROR = 0x20
_SEV_WARNING = 0x10
_SEV_NAME = {_SEV_ERROR: "ERROR", _SEV_WARNING: "WARNING"}

# Routing-angle check is implemented as a Python heuristic for both modes.
# We measure the angle between outward direction vectors at every
# same-net same-layer 2-track joint and flag joints whose measured
# angle is not within ``_ANGLE_TOL_DEG`` of any allowed value.
#
# Convention (matches KiCad's DRC_TEST_PROVIDER_TRACK_ANGLE):
#   straight collinear continuation = 180 deg
#   45-degree miter corner          = 135 deg
#   90-degree right-angle corner    =  90 deg
#
# The C++ DRC provider for track_angle is NOT compiled into the rl
# shared library at this time (kicad_rl_router.so is built without
# drc_test_provider_track_angle.cpp), so a .kicad_dru with a
# track_angle constraint silently no-ops. Until that is rebuilt, this
# Python heuristic is the only path that actually flags violations.
_ANGLE_TOL_DEG = 0.5
_ALLOWED_ANGLES = {
    45: (135.0, 180.0),  # octilinear: 45-degree miter or straight only
    90: (90.0, 180.0),   # orthogonal: right-angle or straight only
}


def _violation_to_record(v_dict: dict) -> dict:
    """Plain JSON-safe record for one violation, plus convenience flags."""
    sev = int(v_dict.get("severity", 0))
    code = int(v_dict.get("error_code", 0))
    rec = {
        "severity_label": _SEV_NAME.get(sev, hex(sev)),
        "severity": sev,
        "error_code": code,
        "error_type": v_dict.get("error_type", ""),
        "x_mm": float(v_dict.get("x_mm", 0.0)),
        "y_mm": float(v_dict.get("y_mm", 0.0)),
        "layer": int(v_dict.get("layer", 0)),
        "net_names": list(v_dict.get("net_names", [])),
        "is_error": sev == _SEV_ERROR,
        "is_promoted": (sev == _SEV_ERROR) or (code in _PROMOTED_CODES),
    }
    return rec


def _per_net_summary(eng) -> dict:
    """{net_name: {wirelength_mm, track_count, via_count, unrouted_edges}}"""
    out: dict[str, dict] = {}
    for t in eng.get_tracks():
        d = out.setdefault(t.net_name, {
            "wirelength_mm": 0.0, "track_count": 0, "via_count": 0,
            "unrouted_edges": 0,
        })
        d["wirelength_mm"] += math.hypot(t.x2_mm - t.x1_mm, t.y2_mm - t.y1_mm)
        d["track_count"] += 1
    for v in eng.get_vias():
        d = out.setdefault(v.net_name, {
            "wirelength_mm": 0.0, "track_count": 0, "via_count": 0,
            "unrouted_edges": 0,
        })
        d["via_count"] += 1
    for e in eng.get_ratsnest():
        name = getattr(e, "net_name", None) or f"net_{e.net_code}"
        d = out.setdefault(name, {
            "wirelength_mm": 0.0, "track_count": 0, "via_count": 0,
            "unrouted_edges": 0,
        })
        d["unrouted_edges"] += 1
    return out


def _check_routing_angles(
    eng,
    allowed_angles: tuple[float, ...],
    tol_deg: float = _ANGLE_TOL_DEG,
) -> dict:
    """Heuristic routing-angle check (Python-side, no DRC dependency).

    Walks every same-net same-layer 2-track joint, measures the angle
    between its outward direction vectors, and flags joints whose
    measured angle is not within ``tol_deg`` of any value in
    ``allowed_angles``.

    Joints skipped (not corners in the routing sense, matching KiCad's
    own track_angle provider semantics):
      - !=2 connected segments (pads/vias hub, T-junctions)
      - joint coordinate coincides with a pad or via on the same layer
        (the wire enters/exits a footprint or via stack — the geometry
        there is dictated by the pad placement, not routing choice)

    The single-range KiCad ``track_angle`` constraint cannot express
    the discontinuous allowed sets we need (e.g. {90, 180} for
    orthogonal-only or {135, 180} for 45-only); even if it could, the
    rl shared library does not currently include the C++ provider,
    making this Python path the only working option.
    """
    from collections import defaultdict

    # Build a per-layer exclusion set of (x, y) coordinates that should
    # never be treated as a routing corner: pad anchors on their layer,
    # and via centers on every layer the via spans.
    exclude: set[tuple[int, tuple[float, float]]] = set()
    for pad in eng.get_pads():
        exclude.add(
            (int(pad.layer), (round(pad.x_mm, 6), round(pad.y_mm, 6)))
        )
    for via in eng.get_vias():
        key_xy = (round(via.x_mm, 6), round(via.y_mm, 6))
        top, bot = int(via.top_layer), int(via.bottom_layer)
        lo, hi = min(top, bot), max(top, bot)
        for ly in range(lo, hi + 1):
            exclude.add((ly, key_xy))

    joints: dict = defaultdict(list)
    for t in eng.get_tracks():
        k1 = (t.net_code, t.layer, (round(t.x1_mm, 6), round(t.y1_mm, 6)))
        k2 = (t.net_code, t.layer, (round(t.x2_mm, 6), round(t.y2_mm, 6)))
        joints[k1].append((t.x2_mm, t.y2_mm, t.net_name))
        joints[k2].append((t.x1_mm, t.y1_mm, t.net_name))

    n_checked = 0
    n_skipped_pad_via = 0
    violations: list[dict] = []
    for (net_code, layer, p), neighbors in joints.items():
        if len(neighbors) != 2:
            continue
        if (int(layer), p) in exclude:
            n_skipped_pad_via += 1
            continue
        n_checked += 1
        a, b = neighbors
        v1x, v1y = a[0] - p[0], a[1] - p[1]
        v2x, v2y = b[0] - p[0], b[1] - p[1]
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
        ang = math.degrees(math.acos(cos))
        if any(abs(ang - allowed) <= tol_deg for allowed in allowed_angles):
            continue
        violations.append({
            "net_code": int(net_code),
            "net_name": a[2],
            "layer": int(layer),
            "x_mm": float(p[0]),
            "y_mm": float(p[1]),
            "angle_deg": float(ang),
        })

    return {
        "method": "python_heuristic",
        "allowed_angles_deg": list(allowed_angles),
        "tolerance_deg": float(tol_deg),
        "n_joints_checked": int(n_checked),
        "n_joints_skipped_pad_via": int(n_skipped_pad_via),
        "n_violations": len(violations),
        "violations": violations,
    }


def _delete_all_tracks_vias(eng) -> None:
    """Mimic PCBWorld.reset() track/via deletion to recover unrouted state."""
    remaining = eng.get_track_count()
    while remaining > 0:
        if not eng.delete_track_by_index(0):
            raise RuntimeError("delete_track_by_index(0) failed")
        new_remaining = eng.get_track_count()
        if new_remaining >= remaining:
            raise RuntimeError("track count not decreasing")
        remaining = new_remaining
    remaining = eng.get_via_count()
    while remaining > 0:
        if not eng.delete_via_by_index(0):
            raise RuntimeError("delete_via_by_index(0) failed")
        new_remaining = eng.get_via_count()
        if new_remaining >= remaining:
            raise RuntimeError("via count not decreasing")
        remaining = new_remaining


def _phi_components(pr, state) -> dict[str, float]:
    completion = pr.completion_bonus if state.unconnected == 0 else 0.0
    unconnected = -pr.unconnected_penalty * state.unconnected
    drc = -pr._drc_penalty(state)
    wire = -pr.wirelength_penalty * state.wirelength
    via = -pr.via_penalty * state.via_count
    return {
        "completion_bonus": float(completion),
        "unconnected_penalty": float(unconnected),
        "drc_penalty": float(drc),
        "wirelength_penalty": float(wire),
        "via_penalty": float(via),
        "total": float(completion + unconnected + drc + wire + via),
    }


def compute_metrics(
    eng,
    u_0: int | None,
    *,
    reward_config_name: str,
    check_angle: int,
    initial_pad_groups: dict[int, int] | None = None,
    routed_pcb: str | None = None,
    pro_path: str | None = None,
) -> dict:
    """Shared metric computation for both disk-loaded and live-engine eval.

    When ``u_0`` is ``None`` the engine is destructively reset after
    capturing the routed snapshot so the initial-unconnected count and the
    initial pad grouping can be discovered.  When ``u_0`` is provided the
    engine state is left untouched and ``initial_pad_groups`` must be
    supplied by the caller (captured at episode reset).
    """
    from pcb_world.engine.drc import DRCUtils
    from pcb_world.core.reward import RewardState
    from pcb_world.core.reward_config import get_reward_config

    if check_angle not in _ALLOWED_ANGLES:
        raise ValueError(f"check_angle must be 45 or 90, got {check_angle}")
    if u_0 is not None and initial_pad_groups is None:
        raise ValueError(
            "initial_pad_groups is required when u_0 is supplied (the engine "
            "is not reset, so the initial pad grouping cannot be measured here)"
        )

    # ---- Phase 1: capture routed-state data ----
    routed_snap = eng.get_reward_snapshot(run_drc=True)
    meta = eng.get_board_meta()
    per_net = _per_net_summary(eng)
    # Per-net pad grouping at the routed state: {net_code: pad-group count}.
    # A pad group is a copper cluster holding >=1 pad, so this ignores
    # dangling copper islands entirely (unlike the ratsnest count).
    routed_pad_groups = eng.get_pad_groups()
    routing_session_open = eng.is_routing()
    drc_violations_raw = [
        DRCUtils.drc_violation_to_dict(v) for v in eng.get_drc_violations()
    ]
    drc_violations = [_violation_to_record(v) for v in drc_violations_raw]
    angle_check_result = _check_routing_angles(eng, _ALLOWED_ANGLES[check_angle])

    # Reward is needed for both the final potential and the bare-board initial
    # potential below, so build it once up front. Board-dependent terms are
    # resolved by PotentialReward.bind_board — the training env's own
    # definition — so offline scoring prices the board exactly as training
    # did (static group here; the per-reset group once the baseline pad
    # groups exist below).
    cfg = get_reward_config(reward_config_name)
    pr = cfg.build_reward()
    pr.bind_board(net_count=meta.net_count, bbox_w=meta.bbox_w, bbox_h=meta.bbox_h)

    # ---- Phase 2: discover u_0 if not provided (destructive) ----
    # initial_potential = Phi of the *bare* board (no tracks/vias), captured at
    # the same reset we use to discover u_0. Reported as a baseline so the
    # headline potential can be stated as a (non-negative) gain over the bare
    # board instead of a raw, sometimes-negative absolute value. None on the
    # inline path (u_0 supplied) where the engine is intentionally not reset.
    # run_drc=True so the bare-board Phi uses the SAME DRC convention as the
    # routed final_potential (run_drc=True above); otherwise potential_gain
    # would subtract a DRC-free initial from a DRC-included final and mix two
    # Phi definitions. The training env runs DRC on reset only in per_step
    # mode (RewardTracker.run_drc_on_reset); a terminal-mode env's own
    # info["initial_potential"] is therefore DRC-free and differs from this
    # baseline (tests/test_reward_parity.py compares Φ₀ only where they agree).
    initial_snap = None
    if u_0 is None:
        if eng.is_routing():
            eng.cancel_route()
        _delete_all_tracks_vias(eng)
        eng.build_connectivity()
        initial_snap = eng.get_reward_snapshot(run_drc=True)
        u_0 = int(initial_snap.unrouted_count)
        initial_pad_groups = eng.get_pad_groups()
    # Per-reset reward hook (size-weighted ladder): weights come from the
    # bare-board pad groups — the env's reset-time capture on the inline
    # path, the stripped board here — i.e. the routability baseline itself.
    pr.bind_board(
        pad_groups=initial_pad_groups,
        net_names=eng.get_net_names(),
        routable_nets=eng.get_routable_nets(),
    )
    initial_potential: float | None = (
        float(pr.compute_final(RewardState.from_snapshot(initial_snap)))
        if initial_snap is not None else None
    )

    # ---- Phase 3: derived metrics ----
    u_t = int(routed_snap.unrouted_count)
    # Routability = fraction of the pin-to-pin connections that had to be made
    # which the routing actually made:
    #
    #     routability = sum_i (G_i(0) - G_i(t)) / sum_i (G_i(0) - 1)
    #
    # with G_i(s) = number of pad groups on net i at state s. Joining two pad
    # groups is exactly one connection made, so the initial board scores 0 and
    # a fully connected board scores 1. Dangling copper carries no pad and so
    # forms no pad group — it cannot inflate G_i(t) the way it would inflate a
    # ratsnest-based count (which dangling copper can drive negative).
    #
    # Range: G_i(t) >= 1 always, so routability <= 1 holds algebraically, and
    # == 1 exactly when every net is whole (i.e. iff `success`). The lower end
    # needs G_i(t) <= G_i(0): adding copper can only merge pad groups, so the
    # only way to split one is to delete copper that was joining pads at t=0.
    # On the disk path the baseline is the fully stripped board, where every
    # pad is its own group and a split is impossible; on the inline path it is
    # the episode's starting board. Either way a violation means the metric's
    # premise broke, so fail loudly rather than emit a negative score.
    regressed = {
        nc: (g, routed_pad_groups.get(nc, 0))
        for nc, g in initial_pad_groups.items()
        if routed_pad_groups.get(nc, 0) > g
    }
    if regressed:
        raise RuntimeError(
            f"routed board has more pad groups than the baseline on "
            f"{len(regressed)} net(s) — routing split pads that started out "
            f"joined (pre-existing copper deleted, or a stale baseline). "
            f"net_code: (initial, routed) = "
            f"{dict(sorted(regressed.items())[:8])}"
        )
    required = sum(g - 1 for g in initial_pad_groups.values())
    made = sum(
        g - routed_pad_groups.get(nc, 0) for nc, g in initial_pad_groups.items()
    )
    routability = 1.0 if required == 0 else made / required
    success = all(routed_pad_groups.get(nc, 0) == 1 for nc in initial_pad_groups)

    state = RewardState.from_snapshot(routed_snap)
    final_potential = float(pr.compute_final(state))
    potential_gain = None if initial_potential is None else (final_potential - initial_potential)
    components = _phi_components(pr, state)

    drv_errors_only = [r for r in drc_violations if r["is_error"]]
    drv_errors_and_promoted = [r for r in drc_violations if r["is_promoted"]]

    track_angle_count = int(angle_check_result["n_violations"])
    track_angle_violations = list(angle_check_result["violations"])
    total_drv_count = len(drv_errors_and_promoted) + track_angle_count

    def _by_type(records):
        c = Counter((r["severity_label"], r["error_code"], r["error_type"]) for r in records)
        return [
            {"severity": s, "error_code": c2, "error_type": et, "count": n}
            for (s, c2, et), n in c.most_common()
        ]

    return {
        "board": str(Path(routed_pcb).resolve()) if routed_pcb else None,
        "pro": str(Path(pro_path).resolve()) if pro_path else None,
        "reward_config": cfg.name,
        "success": bool(success),
        "routability": float(routability),
        "track_count": int(routed_snap.track_count),
        "via_count": int(routed_snap.via_count),
        "wirelength_mm": float(routed_snap.total_wirelength),
        "drv_errors_only_count": len(drv_errors_only),
        "drv_errors_and_promoted_count": len(drv_errors_and_promoted),
        "track_angle_drv": {
            "mode": int(check_angle),
            "source": "heuristic",
            "count": int(track_angle_count),
            "violations": track_angle_violations,
        },
        "total_drv_count": int(total_drv_count),
        "clean_pass": bool(success and len(drv_errors_only) == 0),
        "final_potential": final_potential,
        "initial_potential": initial_potential,
        "potential_gain": potential_gain,
        "phi_components": components,
        "phi_weights": {
            "completion_bonus": float(pr.completion_bonus),
            "clean_completion_bonus": float(pr.clean_completion_bonus),
            "net_completion_bonus": float(pr.net_completion_bonus),
            "net_clean_bonus": float(pr.net_clean_bonus),
            "net_bonus_size_log_scale": float(pr.net_bonus_size_log_scale),
            # {net_code: w_i} once bind_board resolved them (size-weighted
            # ladder only); None otherwise.
            "net_size_weights": (
                {int(c): float(w) for c, w in pr.net_size_weights_by_code.items()}
                if pr.net_size_weights_by_code is not None else None
            ),
            "unconnected_penalty": float(pr.unconnected_penalty),
            "drc_penalty": float(pr.drc_penalty),
            "drc_shape": pr.drc_shape,
            "drc_log_scale": float(getattr(pr, "drc_log_scale", 0.0)),
            "drc_log_agg_scale": float(getattr(pr, "drc_log_agg_scale", 0.0)),
            "drc_log_offset": float(getattr(pr, "drc_log_offset", 0.0)),
            "drc_severity_mode": pr.drc_severity_mode,
            "wirelength_penalty": float(pr.wirelength_penalty),
            "via_penalty": float(pr.via_penalty),
            "step_penalty": float(pr.step_penalty),
        },
        "drv_breakdown": {
            "errors_only_by_type": _by_type(drv_errors_only),
            "errors_and_promoted_by_type": _by_type(drv_errors_and_promoted),
        },
        "drv_violations": drc_violations,
        "extras": {
            "initial_unrouted_edges": u_0,
            "unrouted_edges_remaining": u_t,
            "pad_groups_initial": int(sum(initial_pad_groups.values())),
            "pad_groups_remaining": int(
                sum(routed_pad_groups.get(nc, 0) for nc in initial_pad_groups)
            ),
            "pad_connections_required": int(required),
            "drc_error_count": int(routed_snap.drc_error_count),
            "drc_warning_count": int(routed_snap.drc_warning_count),
            "drc_promoted_count": int(routed_snap.drc_promoted_count),
            "per_net": per_net,
            "board_meta": {
                "bbox_x_mm": meta.bbox_x,
                "bbox_y_mm": meta.bbox_y,
                "bbox_w_mm": meta.bbox_w,
                "bbox_h_mm": meta.bbox_h,
                "net_count": meta.net_count,
                "copper_layers": meta.copper_layers,
            },
            "routing_session_open": bool(routing_session_open),
            "angle_check": {
                "mode": int(check_angle),
                "heuristic_result": angle_check_result,
            },
        },
    }


def compute_metrics_inline(
    env,
    *,
    reward_config_name: str = "drc_dense_promoted",
    check_angle: int = 45,
    routed_pcb: str | None = None,
    pro_path: str | None = None,
) -> dict:
    """Canonical DRC scoring of a live, just-routed ``PCBWorld`` env.

    The inline (non-destructive) counterpart of :func:`evaluate_one`: ``u_0``
    and ``initial_pad_groups`` (the routability baseline) come from the env's
    reset-time capture rather than an engine reset, so the just-routed board is
    left untouched. This is the single scoring convention for both branches —
    the RL and LLM worker wrappers expose it as their ``eval_inline_drc``
    ``env_method`` hook (serial rollout ``--inline-drc on`` / LLM manager
    episode scoring).
    """
    return compute_metrics(
        env._engine,
        u_0=int(env._initial_unconnected),
        reward_config_name=reward_config_name,
        check_angle=check_angle,
        initial_pad_groups=dict(env._initial_pad_groups),
        routed_pcb=routed_pcb,
        pro_path=pro_path,
    )


def evaluate_one(
    routed_pcb: str,
    pro_path: str | None,
    reward_config_name: str = DEFAULTS.reward_config,
    check_angle: int = DEFAULTS.check_angle,
) -> dict:
    """Score a saved routed ``.kicad_pcb`` from disk against design rules.

    Loads the board, runs DRC, then destructively resets the engine to discover
    the initial-unconnected count and the initial pad grouping (the routability
    baseline) — the post-hoc / Stage-2 scoring entry point; the live-engine
    inline path is :func:`compute_metrics` with a supplied ``u_0``.
    """
    if check_angle not in _ALLOWED_ANGLES:
        raise ValueError(f"check_angle must be 45 or 90, got {check_angle}")
    from pcb_world.engine.kicad_engine import KiCadEngine

    eng = KiCadEngine(routed_pcb, project_path=(pro_path or None))
    eng.build_connectivity()
    return compute_metrics(
        eng, u_0=None,
        reward_config_name=reward_config_name,
        check_angle=check_angle,
        routed_pcb=routed_pcb,
        pro_path=pro_path,
    )


# ============================================================================
# Evaluation summary: canonical, sink-agnostic data type
# ============================================================================


def _jsonable(value: Any) -> Any:
    """Coerce nested values to a JSON-safe form (non-finite floats -> ``None``).

    The canonical JSON-coercion helper for the eval pipeline (also used by
    :func:`eval.evaluator.emit_csv_artifacts`); kept in this kernel so it
    stays free of a dependency on the CLI orchestration module.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _as_finite_float(value: Any) -> float | None:
    """Coerce a loggable scalar to a finite float, or ``None`` to skip it.

    Mirrors the projection rule used historically by ``emit_tensorboard``:
    booleans become ``0.0``/``1.0``, finite ints/floats pass through, and
    everything else (strings, ``None``, NaN/inf) is dropped. ``bool`` is checked
    first because it is a subclass of ``int``.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        fv = float(value)
        return fv if math.isfinite(fv) else None
    return None


# ``overall`` carries two vocabularies for the same scalars (bare CADAGENT
# value fields + the explicit ``*_mean`` aggregates from
# ``_overall_from_per_board``). Dashboards only need one — skip the bare
# aliases at logger emission. CSV/JSON exports keep the full dict.
# ``fp_overall_max`` is excluded as uninformative for tracking (single best
# (board, rollout) over the whole val set — one lucky sample): CSV/JSON
# still carry it.
_LOGGER_DUPLICATE_KEYS = frozenset({
    "fp_overall_max",     # tracking-only exclusion (not an alias)
    "drc_mean",           # == drc_violations
    "steps",              # == episode_length_mean
    "episode_return",     # == episode_return_mean
    "final_potential",    # == fp_mean_of_means
    "routability",        # == routability_mean
    "success",            # == success_rate
    "track_count",        # == track_count_mean
    "unrouted_count",     # == unrouted_count_mean
    "via_count",          # == via_count_mean
    "wirelength_mm",      # == wirelength_mean
})


@dataclass(frozen=True)
class EvalSummary:
    """Canonical, sink-agnostic evaluation summary.

    Produced by :class:`eval.evaluator.Evaluator` and consumed by any sink. Holds
    the aggregated ``overall`` (flat scalar dict) and ``per_board`` (per-board
    summary dicts) exactly as produced by
    :func:`eval.aggregation.aggregate_inline` / :class:`eval.metrics.EvalResult`,
    plus the raw ``per_rollout`` rows so CSV/JSON export does not need a separate
    handle.

    Sinks:
      * :meth:`to_logger_dict` — flat ``<prefix>/<key>`` scalars for a
        :class:`~methods._shared.logger.MetricLogger` (wandb / TensorBoard).
      * :meth:`as_eval_result` — back to an :class:`EvalResult` for the CSV
        artifact writer (``eval.evaluator.emit_csv_artifacts``).
      * :meth:`to_json_dict` — a JSON-safe nested dict (NaN/inf -> null).
    """

    overall: dict[str, Any]
    per_board: list[dict[str, Any]]
    per_rollout: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_rollout is None:
            object.__setattr__(self, "per_rollout", [])

    @classmethod
    def from_eval_result(cls, result: EvalResult) -> "EvalSummary":
        """Wrap an already-aggregated :class:`EvalResult` (e.g. RL
        ``eval_transformer`` output). If the result was produced with
        ``skip_aggregation`` (empty ``overall``), aggregate its rollout rows
        here instead via :meth:`from_rollouts`."""
        if result.overall:
            return cls(
                overall=dict(result.overall),
                per_board=list(result.per_board),
                per_rollout=list(result.per_rollout),
            )
        boards = _boards_from_rollouts(result.per_rollout)
        return cls.from_rollouts(result.per_rollout, boards)

    @classmethod
    def from_rollouts(
        cls,
        per_rollout: list[dict],
        boards: Sequence[BoardSpec],
        *,
        selection_mode: str | None = None,
        aggregation_mode: str = "mean",
    ) -> "EvalSummary":
        """Aggregate raw per-rollout rows (e.g. the LLM episode path) into an
        evaluation summary, reusing the same aggregation kernel as the
        standalone evaluator."""
        per_board, overall = aggregate_inline(
            per_rollout,
            list(boards),
            aggregation_mode=aggregation_mode,
            selection_mode=selection_mode,
        )
        return cls(overall=overall, per_board=per_board, per_rollout=list(per_rollout))

    def as_eval_result(self) -> EvalResult:
        """Reconstruct an :class:`EvalResult` for the CSV artifact writer."""
        return EvalResult(
            per_rollout=list(self.per_rollout),
            per_board=list(self.per_board),
            overall=dict(self.overall),
        )

    def to_logger_dict(
        self,
        *,
        prefix: str = "val",
        include_per_board: bool = True,
    ) -> dict[str, float]:
        """Project to a flat ``{tag: finite_float}`` dict for a metric logger.

        Overall metrics keep ``<prefix>/<key>`` tags (all finite scalars),
        minus :data:`_LOGGER_DUPLICATE_KEYS` (alias keys the overall dict
        carries twice — dropped from dashboards only, never from CSV/JSON).
        Per-board metrics are restricted to
        :data:`eval.eval_utils.CADAGENT_VALUE_METRIC_FIELDS` and emitted as
        ``<prefix>/per_board/<board_id>/<key>`` — matching the historical
        ``emit_tensorboard`` field selection so dashboards stay stable.
        """
        out: dict[str, float] = {}
        for key, value in self.overall.items():
            if key in _LOGGER_DUPLICATE_KEYS:
                continue
            fv = _as_finite_float(value)
            if fv is not None:
                out[f"{prefix}/{key}"] = fv
        if include_per_board:
            for row in self.per_board:
                board_id = str(
                    row.get("board_id") or row.get("board_index") or "unknown"
                )
                for key in CADAGENT_VALUE_METRIC_FIELDS:
                    fv = _as_finite_float(row.get(key))
                    if fv is not None:
                        out[f"{prefix}/per_board/{board_id}/{key}"] = fv
        return out

    def to_json_dict(self, *, include_per_rollout: bool = False) -> dict[str, Any]:
        """JSON-safe nested dict: ``{overall, per_board[, per_rollout]}`` with
        non-finite floats coerced to ``None``."""
        payload: dict[str, Any] = {
            "overall": _jsonable(self.overall),
            "per_board": _jsonable(self.per_board),
        }
        if include_per_rollout:
            payload["per_rollout"] = _jsonable(self.per_rollout)
        return payload


def _boards_from_rollouts(per_rollout: list[dict]) -> list[BoardSpec]:
    """Reconstruct the board list (for aggregation) from rollout rows, keyed by
    ``board_index``. Order follows first appearance."""
    seen: dict[int, BoardSpec] = {}
    for row in per_rollout:
        try:
            bi = int(row["board_index"])
        except (KeyError, ValueError, TypeError):
            continue
        if bi not in seen:
            seen[bi] = BoardSpec(
                index=bi,
                board_id=str(row.get("board_id", bi)),
                path=str(row.get("board_path", "")),
            )
    return list(seen.values())
