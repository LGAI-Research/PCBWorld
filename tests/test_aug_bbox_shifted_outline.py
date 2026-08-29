"""Rejection sampling in the bbox-shifted augmentation anchors on the **real outline**.

``sample_bbox_shifted`` accepts only samples where every point entity stays inside
the board after the board is virtually scaled. Two properties of that acceptance
test are pinned here:

1. It is judged against the actual Edge.Cuts outline, not the **axis-aligned bbox**
   — on a non-rectangular board (``d2b_geo`` / ``d2bv_geo``) bbox ⊋ outline, so a
   pad can sit inside the bbox while sticking out of the board.
2. The constraint set covers OBSTACLE entities (NPTH holes/slots) and netless NC
   pads, not just **net pad centers**.

Verification runs without the engine on a hand-built L-shaped board with a notch.
For an L shape the "inside" test has a closed form
(``not (x > 20 and y > 20)``), giving an implementation-independent ground truth.
"""
from __future__ import annotations

import numpy as np
import pytest

from methods.rl_agent.wrappers.augmentation import sample_bbox_shifted


# L-shaped outline: the top-right quadrant (20..40 × 20..40) is cut out of a 40×40 bbox.
_L_VERTS = [(0, 0), (40, 0), (40, 20), (20, 20), (20, 40), (0, 40)]
RANGE, MARGIN, TRIES = 0.3, 1.0, 100


def _inside_base_L(x: float, y: float) -> bool:
    """Implementation-independent ground truth: is the point inside the base L?"""
    return 0 <= x <= 40 and 0 <= y <= 40 and not (x > 20 and y > 20)


def _inside_scaled_L(px, py, sx, sy, cxa, cya) -> bool:
    """Scaled L = {c + s·(q - c)}. p is inside it iff its preimage q is inside the base L."""
    return _inside_base_L(cxa + (px - cxa) / sx, cya + (py - cya) / sy)


def _obs(fmt: str, pads, obstacles=(), upads=()):
    """Minimal raw_obs for the L-shaped board (both obs formats)."""
    segs = list(zip(_L_VERTS, _L_VERTS[1:] + _L_VERTS[:1]))
    bbox = {"bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 40.0, "bbox_h": 40.0}
    if fmt == "indexed":
        pts = list(pads) + list(obstacles) + list(upads)
        pool = [v for s in segs for v in s] + pts
        n_edge_pts = len(segs) * 2
        base = n_edge_pts
        bs = {
            **bbox,
            "pt_xy": np.asarray(pool, dtype=float),
            "edge_pt": np.arange(n_edge_pts, dtype=np.int64).reshape(-1, 2),
            "edge_mid": np.full(len(segs), -1, dtype=np.int64),
            "pad_pt": np.arange(base, base + len(pads), dtype=np.int64),
            "obs_pt": np.arange(base + len(pads),
                                base + len(pads) + len(obstacles),
                                dtype=np.int64),
            "upad_pt": np.arange(base + len(pads) + len(obstacles),
                                 base + len(pts), dtype=np.int64),
        }
        return {"board_static": bs}
    rect = lambda p: {"center": {"xy": [float(p[0]), float(p[1])]}}
    bs = {
        **bbox,
        "boardlines": {
            f"edge_{i}": {"p1": {"xy": list(map(float, a))},
                          "p2": {"xy": list(map(float, b))}}
            for i, (a, b) in enumerate(segs)
        },
        "nets": {1: {"pads": {f"P{i}": rect(p) for i, p in enumerate(pads)}}},
        "obstacles": {f"O{i}": rect(p) for i, p in enumerate(obstacles)},
        "unconnected_pads": {f"U{i}": rect(p) for i, p in enumerate(upads)},
    }
    return {"board_static": bs}


def _draw(raw_obs, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [
        sample_bbox_shifted(rng, raw_obs, range_=RANGE, margin=MARGIN,
                            max_tries=TRIES)
        for _ in range(n)
    ]


def _assert_all_inside(samples, points):
    """Every point must stay inside the scaled L in every accepted sample."""
    bad = [
        (s, p) for s in samples for p in points
        if not _inside_scaled_L(p[0], p[1], *s)
    ]
    assert not bad, (
        f"{len(bad)}/{len(samples) * len(points)} point-samples left the "
        f"outline; first offender sample={bad[0][0]} point={bad[0][1]}"
    )
    # Guard against passing only because every sample fell back to identity.
    assert any(s[0] != 1.0 or s[1] != 1.0 for s in samples)


@pytest.mark.parametrize("fmt", ["indexed", "json"])
def test_pads_stay_inside_the_real_outline(fmt):
    """Non-rectangular board: no pad leaks into the notch, which is inside the
    bbox but outside the outline."""
    # (19.5, 19.5) sits right next to the notch's inner corner: as the outline
    # shrinks the notch swallows this pad, while it stays inside the scaled bbox.
    pads = [(5.0, 5.0), (34.0, 8.0), (8.0, 34.0), (19.5, 19.5)]
    samples = _draw(_obs(fmt, pads))
    _assert_all_inside(samples, pads)


@pytest.mark.parametrize("fmt", ["indexed", "json"])
def test_obstacles_and_nc_pads_are_constrained(fmt):
    """OBSTACLE (NPTH) entities and netless NC pads belong to the constraint set
    too — the single net pad sits mid-board and constrains nothing on its own."""
    pads = [(20.0, 10.0)]
    obstacles = [(34.0, 8.0), (5.0, 5.0)]
    upads = [(8.0, 34.0)]
    samples = _draw(_obs(fmt, pads, obstacles, upads))
    _assert_all_inside(samples, pads + obstacles + upads)


def test_both_obs_formats_agree():
    """The same board with the same rng yields identical samples in both formats."""
    pads = [(5.0, 5.0), (34.0, 8.0), (8.0, 34.0)]
    assert _draw(_obs("indexed", pads)) == _draw(_obs("json", pads))


def test_identity_fallback_when_unsatisfiable():
    """An unsatisfiable constraint (a point inside the notch) falls back to
    identity once max_tries is exhausted."""
    raw_obs = _obs("indexed", [(30.0, 30.0)])   # outside the L — no sample can pass
    rng = np.random.default_rng(0)
    assert sample_bbox_shifted(
        rng, raw_obs, range_=RANGE, margin=MARGIN, max_tries=TRIES,
    ) == (1.0, 1.0, 20.0, 20.0)
