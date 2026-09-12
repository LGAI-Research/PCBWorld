"""Coordinate / net-exchangeability augmentation math for the RL wrapper.

Pure functions extracted from the decoder wrapper. The wrapper still owns the
per-episode aug *state* and the tunable magnitudes (kept as class attributes so
ablations can override them); these functions hold the *math* so the wrapper
stays a thin orchestrator. Behavior is identical to the previous inline code.
See design doc §3.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def collect_fixed_points(bs: dict) -> np.ndarray:
    """``(N, 2)`` centres of every point entity the tokenizer renders at a
    FIXED physical position under bbox-shifted aug — net pads, OBSTACLE
    entities (NPTH holes/slots) and net-less NC pads. Only the board outline
    moves, so these are exactly the points the rejection sampler must keep
    inside it. Handles both obs formats.
    """
    if "pt_xy" in bs:  # indexed_v1 tables — gather rows from the shared pool
        rows = np.concatenate([bs["pad_pt"], bs["obs_pt"], bs["upad_pt"]])
        return np.asarray(bs["pt_xy"], dtype=float)[rows].reshape(-1, 2)
    xy = [
        (p["center"]["xy"][0], p["center"]["xy"][1])
        for net in bs["nets"].values()
        for p in net.get("pads", {}).values()
    ]
    xy += [
        (r["center"]["xy"][0], r["center"]["xy"][1])
        for key in ("obstacles", "unconnected_pads")
        for r in bs.get(key, {}).values()
    ]
    return np.asarray(xy, dtype=float).reshape(-1, 2)


def collect_outline_segments(bs: dict) -> np.ndarray:
    """``(S, 2, 2)`` Edge.Cuts segment endpoints ``[[x1,y1],[x2,y2]]``.

    Arc entries (``outline_obs="arc"`` carries an on-arc ``mid``) are split at
    their midpoint into two chords, so the polygon stays a chord approximation
    of the true boundary — the same approximation the EDGE tokens render.
    Segment ORDER does not matter: the inside test below is a crossing count
    over the whole segment set, which handles multi-loop outlines (cutouts)
    without assembling rings.
    """
    if "pt_xy" in bs:  # indexed_v1 tables
        pt = np.asarray(bs["pt_xy"], dtype=float)
        ep = np.asarray(bs["edge_pt"])
        mid = np.asarray(bs["edge_mid"])
        segs = []
        for i in range(len(ep)):
            a, b = pt[ep[i, 0]], pt[ep[i, 1]]
            if mid[i] >= 0:
                m = pt[mid[i]]
                segs.append((a, m))
                segs.append((m, b))
            else:
                segs.append((a, b))
        return np.asarray(segs, dtype=float).reshape(-1, 2, 2)
    segs = []
    for e in bs.get("boardlines", {}).values():
        a = tuple(e["p1"]["xy"])
        b = tuple(e["p2"]["xy"])
        if "mid" in e:
            m = tuple(e["mid"]["xy"])
            segs.append((a, m))
            segs.append((m, b))
        else:
            segs.append((a, b))
    return np.asarray(segs, dtype=float).reshape(-1, 2, 2)


def _inside_with_clearance(
    pts: np.ndarray, segs: np.ndarray, margin: float,
) -> bool:
    """True when EVERY point in ``pts`` lies inside the polygon bounded by
    ``segs`` and is at least ``margin`` away from its boundary.

    Inside = odd crossing count of a +x ray (works for any closed loop set).
    Clearance = min point-to-segment distance. Both vectorised over
    ``(N, S)``; N and S are tens for the boards this runs on.
    """
    px = pts[:, 0:1]                       # (N,1)
    py = pts[:, 1:2]
    x1, y1 = segs[:, 0, 0][None, :], segs[:, 0, 1][None, :]   # (1,S)
    x2, y2 = segs[:, 1, 0][None, :], segs[:, 1, 1][None, :]

    # --- crossing count (ray toward +x) ---
    straddles = (y1 > py) != (y2 > py)
    dy = y2 - y1
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dy != 0.0, (py - y1) / np.where(dy != 0.0, dy, 1.0), 0.0)
    x_hit = x1 + t * (x2 - x1)
    if not np.all(np.count_nonzero(straddles & (px < x_hit), axis=1) % 2 == 1):
        return False

    # --- clearance to the boundary ---
    ex, ey = x2 - x1, y2 - y1
    len2 = ex * ex + ey * ey
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(
            len2 > 0.0,
            ((px - x1) * ex + (py - y1) * ey) / np.where(len2 > 0.0, len2, 1.0),
            0.0,
        )
    u = np.clip(u, 0.0, 1.0)
    d2 = (px - (x1 + u * ex)) ** 2 + (py - (y1 + u * ey)) ** 2
    return bool(np.all(d2.min(axis=1) >= margin * margin))


def sample_bbox_shifted(
    rng: np.random.Generator,
    raw_obs: dict,
    *,
    range_: float,
    margin: float,
    max_tries: int,
) -> tuple[float, float, float, float]:
    """Rejection-sample ``(scale_x, scale_y, cx, cy)`` so every fixed point
    entity of the current episode stays inside the virtually scaled board
    OUTLINE with ``margin`` clearance. Returns an identity fallback (scale 1,
    bbox centre) when no sample satisfies the constraint within ``max_tries``.

    Under this scheme only the outline moves — ``_norm_pos_edge`` maps an edge
    vertex ``q`` to ``c + s·(q - c)`` while pads/obstacles stay physically
    fixed — so the constraint is evaluated by transforming the outline and
    testing the (unmoved) points against it.

    Two things the axis-aligned-bbox version of this test got wrong (260819
    audit): it accepted samples that push a point outside the real Edge.Cuts
    outline on non-rectangular boards (``d2b_geo`` / ``d2bv_geo``: 3.5-4.6% of
    episodes, up to 11mm out), and it constrained net pads only — OBSTACLE
    entities and net-less NC pads (a further 5.2%) were free to leave the
    board. The scaled bbox is kept only as a cheap NECESSARY pre-filter
    (outline ⊆ bbox, so a bbox reject is an outline reject).
    """
    bs = raw_obs["board_static"]
    xmin = float(bs["bbox_x"])
    ymin = float(bs["bbox_y"])
    xmax = xmin + float(bs["bbox_w"])
    ymax = ymin + float(bs["bbox_h"])
    pts = collect_fixed_points(bs)
    segs = collect_outline_segments(bs)

    lo, hi = 1.0 - range_, 1.0 + range_
    m = margin

    for _ in range(max_tries):
        sx = float(rng.uniform(lo, hi))
        sy = float(rng.uniform(lo, hi))
        cxa = float(rng.uniform(xmin, xmax))
        cya = float(rng.uniform(ymin, ymax))

        nxmin = cxa + sx * (xmin - cxa)
        nxmax = cxa + sx * (xmax - cxa)
        nymin = cya + sy * (ymin - cya)
        nymax = cya + sy * (ymax - cya)
        if nxmin > nxmax:
            nxmin, nxmax = nxmax, nxmin
        if nymin > nymax:
            nymin, nymax = nymax, nymin

        ok = bool(np.all(
            (pts[:, 0] >= nxmin + m) & (pts[:, 0] <= nxmax - m)
            & (pts[:, 1] >= nymin + m) & (pts[:, 1] <= nymax - m)
        )) if len(pts) else True
        if ok and len(segs) and len(pts):
            # Virtually scaled outline — same map _norm_pos_edge applies.
            v = np.empty_like(segs)
            v[..., 0] = cxa + sx * (segs[..., 0] - cxa)
            v[..., 1] = cya + sy * (segs[..., 1] - cya)
            ok = _inside_with_clearance(pts, v, m)
        if ok:
            return sx, sy, cxa, cya

    return 1.0, 1.0, (xmin + xmax) / 2, (ymin + ymax) / 2


def build_aug_dict(
    *,
    bbox_shifted: bool,
    scale_x: float,
    scale_y: float,
    cx: float,
    cy: float,
    axis_swap: bool,
    flip_x: int,
    flip_y: int,
    nn_dx: float,
    nn_dy: float,
    nn_zoom: float,
    slot_perm: list[int] | None,
    directional_candidates: str | None,
    connectivity_filter: bool = True,
    route_start_xy: tuple[float, float, int] | None = None,
    cluster_keys: frozenset[tuple[float, float, int]] | None = None,
    pad_graze_margin_mm: float = 0.0,
) -> dict[str, Any]:
    """Build the ``_aug`` dict the StateTokenizer reads for coordinate
    normalization. Orthogonal-axis keys are always emitted; bbox-shifted keys
    only when that scheme is active (identical to the prior inline behavior).

    The trailing ``connectivity_filter`` / ``route_start_xy`` / ``cluster_keys``
    keys drive the connectivity filter in ``candidate_pool`` (default on; it
    trims existing-copper candidates — pads, vias, track endpoints — that the
    route head is ALREADY connected to). ``cluster_keys`` is the engine-computed
    cluster of the head, resolved by the wrapper (which owns the engine handle)
    so ``collect_raw_candidates`` stays a pure obs function. They are read by
    ``collect_raw_candidates``, never enter the token feature stream.

    ``pad_graze_margin_mm`` (default 0 = off) is the pad-graze guard width, in
    mm, applied to the SYNTHESISED (directional) candidates only — see
    ``candidate_pool._graze_margin_mm``. Same contract: read by the candidate
    builder, never a token feature.
    """
    aug: dict[str, Any] = (
        {"scale_x": scale_x, "scale_y": scale_y, "aug_cx": cx, "aug_cy": cy}
        if bbox_shifted
        else {}
    )
    aug["axis_swap"] = axis_swap
    aug["flip_x"] = flip_x
    aug["flip_y"] = flip_y
    aug["nn_dx"] = nn_dx
    aug["nn_dy"] = nn_dy
    aug["nn_zoom"] = nn_zoom
    aug["slot_perm"] = slot_perm
    aug["directional_candidates"] = directional_candidates
    aug["connectivity_filter"] = connectivity_filter
    aug["route_start_xy"] = route_start_xy
    aug["cluster_keys"] = cluster_keys
    aug["pad_graze_margin_mm"] = pad_graze_margin_mm
    return aug
