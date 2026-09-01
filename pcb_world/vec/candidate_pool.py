"""Candidate point builder.

Shared collection logic for both MLP (fixed-size numpy) and Transformer
(variable-length token sequence) policies.

Consumes the observation dict produced by build_json_observation() — the
canonical dict state of the environment. All coordinates in the dict are
in mm; normalization is done by each consumer.

Candidates are collected from the **current net only**, deduplicated by
(x, y, layer) coordinate.
"""

from __future__ import annotations

import numpy as np

MAX_CANDIDATES = 64
NUM_CAND_TYPES = 5  # PAD, TRACK, RATSNEST, VIA, DIRECTIONAL
CAND_FEATURES = 3 + NUM_CAND_TYPES  # x_norm, y_norm, layer_norm, type_onehot(5)

# Candidate type constants
CTYPE_PAD = 0
CTYPE_TRACK = 1
CTYPE_RATSNEST = 2
CTYPE_VIA = 3
CTYPE_DIRECTIONAL = 4

# 8 directions (dx, dy) — N, NE, E, SE, S, SW, W, NW
_DIRECTIONS = [
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
]
_DIR_DISTANCES_MM = [0.5]

# 1-layer Grid mode: 4-direction candidates with per-grid-size step bundles
# so emitted points always land on the underlying grid corners.
# Step values are in *cells*; mm offset = step_cell * grid_spacing where
# grid_spacing = _BOARD_SIZE_MM / grid_size (mirrors the synthetic grid
# generator at tools/datagen/synthetic_generator/generate_grid_boards.py).
_BOARD_SIZE_MM = 100.0
_GRID_4DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
_GRID_STEP_CELLS: dict[int, list[int]] = {
    10: [1],
    30: [1, 5],
    50: [1, 5],
    100: [1, 5],
    200: [1, 5, 25],
    300: [1, 5, 25],
    500: [1, 5, 25],
    1000: [1, 5, 25],
}


# ---------------------------------------------------------------------------
# Connectivity candidate filter (already-connected copper)
# ---------------------------------------------------------------------------
# Gated by ``obs["_aug"]["connectivity_filter"]`` (the wrapper defaults it on;
# ``--no-connectivity-filter`` clears it) and applied to every EXISTING-COPPER
# candidate — pads, vias and track endpoints — while actively routing. One rule:
#
#     drop a candidate iff it is ALREADY electrically connected to the route
#     head, i.e. it belongs to the head's connectivity cluster.
#
# That is precisely the redundant same-net loop we want to suppress: wiring two
# points that the board already joins adds copper and no connectivity.
#
# The cluster comes from the ENGINE (``KiCadEngine.get_connected_points`` →
# ``PNS_RL_ROUTER::getConnectedPoints`` → KiCad ``CONNECTIVITY_DATA``), resolved
# by the wrapper each step and handed over as ``obs["_aug"]["cluster_keys"]`` —
# a frozenset of ``(round(x, 3), round(y, 3), human_layer)``. Keeping the query
# in the wrapper (which owns the engine handle) leaves this module a pure obs
# function, which the tokenizer / MCTS candidate-order invariants require.
#
# Why not ratsnest coordinates: ratsnest edges are a
# Kruskal MST, so they expose ONE representative anchor per cluster. Matching on
# them both over- and under-shoots — an already-connected pad that happens to be
# the representative survived (the loop we meant to block), and a genuinely
# unconnected anchor that was not the representative got dropped. Cluster
# membership has neither failure mode, and it needs no Case A/B split: starting
# from an isolated pad simply yields a singleton cluster.
#
# The match is on (x, y, LAYER), not (x, y): a thru-hole pad or a via reports
# every copper layer it spans (one item, one cluster), so its opposite face is
# dropped by construction instead of by layer-blind coordinate matching, which
# would conflate stacked pads that are NOT connected.
#
# Directional candidates (``extra_candidates``) are NEVER filtered: they are
# generated geometry around the head, not existing copper, and are what keeps
# routing progressing when everything nearby is already connected. They are also
# what guarantees a non-empty pool (the policy's pointer row can never go
# all-(-inf)), since the engine emits them under the same ``is_routing``
# condition that switches this filter on.


def _xy_key(x, y, layer) -> tuple[float, float, int]:
    return (round(float(x), 3), round(float(y), 3), int(layer))


def _graze_margin_mm(obs: dict) -> float:
    """Pad-graze guard width in mm (0 = guard off).

    Directional candidates are SYNTHESISED coordinates (head + a fixed mm
    offset), not board geometry, and nothing else checks them against pads. A
    point landing just outside a same-net pad's copper — inside
    ``(pad_r, pad_r + margin)`` — lets an action place a via / end a track whose
    copper only GRAZES the pad: KiCad's shape-overlap connectivity scores that
    as connected, while its anchor-based dangling test does not (so KiCad's own
    track cleaner deletes such copper). Set the margin to the via radius to make
    that band unaddressable. Judged in mm, deliberately: the tokenizer's
    normalised frame divides by a board-size-derived scale, so a fixed
    normalised threshold would mean a different physical width per board.
    """
    aug = obs.get("_aug") or {}
    try:
        margin = float(aug.get("pad_graze_margin_mm") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return margin if margin > 0.0 else 0.0


def _in_graze_band(x_mm: float, y_mm: float, layer: int,
                   pads: list[tuple[float, float, float, int]],
                   margin: float) -> bool:
    """True when (x, y) sits in the graze annulus of one of ``pads``.

    ``pads`` entries are ``(x, y, radius, pad_layer)`` with pad_layer 0 = thru
    (applies to every copper layer). Points INSIDE the pad copper are kept —
    those are proper on-pad placements, not slivers.
    """
    for px, py, pr, pl in pads:
        if pl != 0 and pl != layer:
            continue
        d = ((x_mm - px) ** 2 + (y_mm - py) ** 2) ** 0.5
        if pr < d < pr + margin:
            return True
    return False


def _cand_filter_ctx(obs: dict):
    """Resolve the candidate filter for this obs.

    Returns the set of ``(x, y, layer)`` keys already connected to the route
    head, or ``None`` when the filter is off / not routing / the engine found no
    copper under the head (nothing to be connected to). The wrapper ships the
    keys as a sorted tuple (JSON-dumpable obs); they are hashed into a set once
    per call here, not per candidate.
    """
    aug = obs.get("_aug") or {}
    if not aug.get("connectivity_filter"):
        return None
    rh = obs.get("router_head") or {}
    if not rh.get("is_routing", False):
        return None
    keys = aug.get("cluster_keys")
    if not keys:
        return None
    return {(round(float(x), 3), round(float(y), 3), int(lay)) for x, y, lay in keys}


# ---------------------------------------------------------------------------
# Shared collection logic
# ---------------------------------------------------------------------------
def collect_raw_candidates(
    obs: dict,
    current_net_id: int | None,
    extra_candidates: list[tuple[float, float, int, int]] | None = None,
) -> list[tuple[float, float, int, int]]:
    """Collect deduplicated raw candidate points from the current net.

    Shared by both ``build_candidates_mlp()`` and Transformer
    ``StateTokenizer._build_candidate_pool()``. Accepts BOTH observation
    formats (legacy nested dict / indexed_v1 tables) — this is the
    single source of the candidate-pool ORDER, which pointer decode and
    the tokenizer must agree on tuple-for-tuple.

    Args:
        obs: Observation from ``build_json_observation()`` or
            ``build_indexed_observation()``.
        current_net_id: Net code of the currently active net, or None.
            When None, returns only extra_candidates (if any).
        extra_candidates: Additional ``(x_mm, y_mm, layer, ctype)`` tuples
            (e.g. directional candidates).

    Returns:
        List of ``(x_mm, y_mm, layer, ctype)`` tuples, deduplicated by
        ``(round(x, 3), round(y, 3), layer)``.
    """
    from pcb_world.core.indexed_obs import is_indexed
    if is_indexed(obs):
        return _collect_raw_candidates_indexed(
            obs, current_net_id, extra_candidates,
        )
    board_static = obs.get("board_static", {})
    max_layer = max(board_static.get("copper_layers", 2), 2)

    raw_cands: list[tuple[float, float, int, int]] = []
    seen: set[tuple[float, float, int]] = set()

    def _add(x_mm: float, y_mm: float, layer: int, ctype: int) -> None:
        key = (round(x_mm, 3), round(y_mm, 3), layer)
        if key not in seen:
            seen.add(key)
            raw_cands.append((x_mm, y_mm, layer, ctype))

    net_key = f"net_{current_net_id}" if current_net_id is not None else None

    # Priority order for dedup: first registered wins.
    # Ratsnest is context-only (a state token), NOT a selectable pointer target:
    # RatsnestEdge carries no per-endpoint layer, so admitting it as a candidate
    # would emit a fake layer=1 signal and occupy the candidate slot of the real
    # pad. Pads/track-endpoints/vias cover all valid target coordinates with
    # accurate layer information.

    net_geom = None
    if net_key is not None:
        net_geom = obs.get("routing_geometry", {}).get(net_key)

    # 1. Pads (from static context) — highest priority since they carry
    #    the authoritative (x, y, layer) for routing targets.
    #    Thru-hole pads carry sentinel layer=0 ("spans every copper layer");
    #    expand them to one candidate per copper layer, mirroring the via
    #    expansion below — otherwise the policy can never select the pad
    #    as a target on any specific routing layer.
    connected = _cand_filter_ctx(obs)

    def _dropped(x_mm: float, y_mm: float, layer: int) -> bool:
        return connected is not None and _xy_key(x_mm, y_mm, layer) in connected

    graze_margin = _graze_margin_mm(obs)
    graze_pads: list[tuple[float, float, float, int]] = []

    if net_key is not None:
        net_ctx = board_static.get("nets", {}).get(net_key)
        if net_ctx is not None:
            for pad in net_ctx.get("pads", {}).values():
                xy = pad["center"]["xy"]
                pl = pad["layer"]
                if graze_margin:
                    graze_pads.append((
                        xy[0], xy[1],
                        max(pad.get("width", 0.0), pad.get("height", 0.0)) / 2.0,
                        pl,
                    ))
                for layer in (
                    range(1, max_layer + 1) if pl == 0 else (pl,)
                ):
                    if not _dropped(xy[0], xy[1], layer):
                        _add(xy[0], xy[1], layer, CTYPE_PAD)

    # 2. Vias
    if net_geom is not None:
        for via in net_geom.get("vias", {}).values():
            xy = via["center"]["xy"]
            l_start = via.get("layer_start", 1)
            l_end = via.get("layer_end", max_layer)
            for layer in range(l_start, l_end + 1):
                if not _dropped(xy[0], xy[1], layer):
                    _add(xy[0], xy[1], layer, CTYPE_VIA)

    # 3. Track endpoints
    if net_geom is not None:
        for track in net_geom.get("tracks", {}).values():
            p1_xy = track["p1"]["xy"]
            p2_xy = track["p2"]["xy"]
            layer = track["layer"]
            if not _dropped(p1_xy[0], p1_xy[1], layer):
                _add(p1_xy[0], p1_xy[1], layer, CTYPE_TRACK)
            if not _dropped(p2_xy[0], p2_xy[1], layer):
                _add(p2_xy[0], p2_xy[1], layer, CTYPE_TRACK)

    # 4. Extra candidates (e.g. directional) — the only synthesised coordinates
    #    in the pool, so the graze guard applies here and nowhere else (a pad
    #    centre or an existing track endpoint is real geometry and stays
    #    selectable).
    if extra_candidates:
        for x_mm, y_mm, layer, ctype in extra_candidates:
            if graze_margin and _in_graze_band(x_mm, y_mm, layer,
                                               graze_pads, graze_margin):
                continue
            _add(x_mm, y_mm, layer, ctype)

    return raw_cands


def _collect_raw_candidates_indexed(
    obs: dict,
    current_net_id: int | None,
    extra_candidates: list[tuple[float, float, int, int]] | None = None,
) -> list[tuple[float, float, int, int]]:
    """indexed_v1 twin of the dict path above. Emission ORDER and dedup
    are byte-identical: pads (thru layer==0 expanded 1..max_layer) →
    vias (expanded layer_start..layer_end) → track p1, p2 → extras;
    dedup key (round(x,3), round(y,3), layer), first registered wins."""
    bs = obs["board_static"]
    max_layer = max(int(bs["copper_layers"]), 2)

    raw_cands: list[tuple[float, float, int, int]] = []
    seen: set[tuple[float, float, int]] = set()

    def _add(x_mm: float, y_mm: float, layer: int, ctype: int) -> None:
        key = (round(x_mm, 3), round(y_mm, 3), layer)
        if key not in seen:
            seen.add(key)
            raw_cands.append((x_mm, y_mm, layer, ctype))

    connected = _cand_filter_ctx(obs)

    def _dropped(x_mm: float, y_mm: float, layer: int) -> bool:
        return connected is not None and _xy_key(x_mm, y_mm, layer) in connected

    graze_margin = _graze_margin_mm(obs)
    graze_pads: list[tuple[float, float, float, int]] = []

    if current_net_id is not None:
        # 1. Pads of the current net (static tables; parser row order).
        hit = np.nonzero(bs["net_code"] == current_net_id)[0]
        if hit.size:
            j = int(hit[0])
            start = int(bs["net_pad_start"][j])
            count = int(bs["net_pad_count"][j])
            pt_xy = bs["pt_xy"]
            pad_pt = bs["pad_pt"]
            pad_layer = bs["pad_layer"]
            pad_wh = bs["pad_wh"]
            for i in range(start, start + count):
                x = float(pt_xy[pad_pt[i], 0])
                y = float(pt_xy[pad_pt[i], 1])
                pl = int(pad_layer[i])
                if graze_margin:
                    graze_pads.append((
                        x, y, float(max(pad_wh[i, 0], pad_wh[i, 1])) / 2.0, pl,
                    ))
                for layer in (range(1, max_layer + 1) if pl == 0 else (pl,)):
                    if not _dropped(x, y, layer):
                        _add(x, y, layer, CTYPE_PAD)

        rg = obs.get("routing_geometry")
        if rg is not None and len(rg["net_code"]):
            p = int(np.searchsorted(rg["net_code"], current_net_id))
            if p < len(rg["net_code"]) and rg["net_code"][p] == current_net_id:
                d_pt = rg["pt_xy"]
                # 2. Vias
                v0 = int(rg["via_start"][p])
                for r in range(v0, v0 + int(rg["via_count"][p])):
                    x = float(d_pt[rg["via_pt"][r], 0])
                    y = float(d_pt[rg["via_pt"][r], 1])
                    for layer in range(int(rg["via_ls"][r]),
                                       int(rg["via_le"][r]) + 1):
                        if not _dropped(x, y, layer):
                            _add(x, y, layer, CTYPE_VIA)
                # 3. Track endpoints
                t0 = int(rg["trk_start"][p])
                for r in range(t0, t0 + int(rg["trk_count"][p])):
                    layer = int(rg["trk_layer"][r])
                    i1, i2 = int(rg["trk_pt"][r, 0]), int(rg["trk_pt"][r, 1])
                    x1, y1 = float(d_pt[i1, 0]), float(d_pt[i1, 1])
                    x2, y2 = float(d_pt[i2, 0]), float(d_pt[i2, 1])
                    if not _dropped(x1, y1, layer):
                        _add(x1, y1, layer, CTYPE_TRACK)
                    if not _dropped(x2, y2, layer):
                        _add(x2, y2, layer, CTYPE_TRACK)

    # 4. Extra candidates (e.g. directional) — graze guard as in the dict path.
    if extra_candidates:
        for x_mm, y_mm, layer, ctype in extra_candidates:
            if graze_margin and _in_graze_band(x_mm, y_mm, layer,
                                               graze_pads, graze_margin):
                continue
            _add(x_mm, y_mm, layer, ctype)

    return raw_cands


# ---------------------------------------------------------------------------
# MLP-specific: sort, truncate, normalize to numpy
# ---------------------------------------------------------------------------
def build_candidates_mlp(
    obs: dict,
    current_net_id: int | None,
    route_head_mm: tuple[float, float],
    max_candidates: int = MAX_CANDIDATES,
    extra_candidates: list[tuple[float, float, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, int]]]:
    """Build candidate target points for the MLP policy.

    Uses ``collect_raw_candidates()`` for collection, then sorts,
    truncates to ``max_candidates``, and normalizes into fixed-size
    numpy arrays.

    Args:
        obs: Observation dict from build_json_observation().
        current_net_id: Net code of the currently active net, or None.
        route_head_mm: Current route head position in mm (x, y).
        max_candidates: Maximum number of candidates to return.
        extra_candidates: Additional (x_mm, y_mm, layer, ctype) tuples.

    Returns:
        cand_features: (max_candidates, CAND_FEATURES) normalized features.
        cand_mask: (max_candidates,) bool — which slots are valid.
        cand_mm: list of (x_mm, y_mm, layer) for action building.
    """
    board_static = obs.get("board_static", {})
    bx = board_static.get("bbox_x", 0.0)
    by = board_static.get("bbox_y", 0.0)
    scale = board_static.get("scale", 1.0)
    max_layer = max(board_static.get("copper_layers", 2), 2)

    raw_cands = collect_raw_candidates(obs, current_net_id, extra_candidates)

    # Sort: PAD > others, then by distance from route head.
    # (Ratsnest is no longer a pointer-selectable candidate; see
    # ``collect_raw_candidates`` for rationale.)
    hx, hy = route_head_mm
    _TYPE_PRIORITY = {CTYPE_PAD: 0}

    def _sort_key(c: tuple) -> tuple[int, float]:
        priority = _TYPE_PRIORITY.get(c[3], 2)
        dist = (c[0] - hx) ** 2 + (c[1] - hy) ** 2
        return (priority, dist)

    raw_cands.sort(key=_sort_key)
    raw_cands = raw_cands[:max_candidates]

    # Build output arrays (normalize for MLP)
    cand_features = np.zeros((max_candidates, CAND_FEATURES), dtype=np.float32)
    cand_mask = np.zeros(max_candidates, dtype=bool)
    cand_mm: list[tuple[float, float, int]] = []

    for i, (x_mm, y_mm, layer, ctype) in enumerate(raw_cands):
        x_n = (x_mm - bx) / scale
        y_n = (y_mm - by) / scale
        cand_features[i, 0] = x_n
        cand_features[i, 1] = y_n
        cand_features[i, 2] = layer / max_layer
        cand_features[i, 3 + ctype] = 1.0  # one-hot type
        cand_mask[i] = True
        cand_mm.append((x_mm, y_mm, layer))

    return cand_features, cand_mask, cand_mm


# Named directional-mode presets for the 8-direction path: distance ladders in
# mm (8 dirs × each distance). Extending behavior = add a VALUE here — the mode
# string is the `directional_candidates` CLI/config value, no new args.
DIRECTIONAL_DISTANCE_PRESETS: dict[str, tuple[float, ...]] = {
    "multi_resolution": (0.2, 1.0, 5.0, 25.0),
    # multi_resolution minus the board-scale 25mm rung — isolates whether the
    # long-jump candidates (which overshoot mid-size boards) drive the mres
    # effect (260813 campaign A6).
    "multi_resolution_no25": (0.2, 1.0, 5.0),
    # Log-scale 1-2-5 ladder; every rung is a multiple of the 0.2 mm generation
    # grid so witness segments decompose exactly (8 dirs x 8 rungs = 64 cands).
    "mres8": (0.2, 0.4, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0),
    # Same 8 directions, one ring, 1.0 mm instead of the 0.5 mm default. The
    # default ring lands INSIDE the pad the route just started from whenever the
    # pad is wider than 1.0 mm, which is most of them: measured over 20 boards
    # each, the share of directional candidates falling in open copper-free space
    # is d3b 25.9% / d2b 23.8% / the d3b-matched synthetic set 0.0% at 0.5 mm,
    # against 86.5% / 100% / 99.9% at 1.0 mm. Past 1.0 mm the gain stops and the
    # candidates start landing on NEIGHBOURING pads instead (d3b: 5.1% at 1.0 mm,
    # 22.5% at 2.0 mm), which the PAD candidates already cover.
    "ring_1mm": (1.0,),
}


def parse_directional_mode(
    mode: str | None,
) -> tuple[int | None, tuple[float, ...] | None]:
    """``directional_candidates`` mode string → ``(grid_size, distances)``.

    * ``None`` → ``(None, None)`` — default 8-dir × 0.5mm ring
    * preset name (``DIRECTIONAL_DISTANCE_PRESETS``) → ``(None, ladder)``
    * ``"grid<N>"`` (N ∈ ``_GRID_STEP_CELLS``) → ``(N, None)`` — 1-layer Grid

    Raises ValueError on anything else, so a config typo fails loudly at
    wrapper construction instead of silently routing with the default ring.
    """
    if mode is None:
        return None, None
    if mode in DIRECTIONAL_DISTANCE_PRESETS:
        return None, DIRECTIONAL_DISTANCE_PRESETS[mode]
    if isinstance(mode, str) and mode.startswith("grid") and mode[4:].isdigit():
        grid_size = int(mode[4:])
        if grid_size in _GRID_STEP_CELLS:
            return grid_size, None
    raise ValueError(
        f"unknown directional_candidates mode {mode!r}; expected a preset in "
        f"{sorted(DIRECTIONAL_DISTANCE_PRESETS)} or 'grid<N>' with N in "
        f"{sorted(_GRID_STEP_CELLS)}"
    )


def build_directional_candidates(
    route_head_mm: tuple[float, float],
    current_layer: int,
    mode: str | None = None,
) -> list[tuple[float, float, int, int]]:
    """Build directional candidates from route head.

    ``mode`` is the ``directional_candidates`` knob (see
    :func:`parse_directional_mode`):

    * ``None`` (default): 8 directions × [0.5mm] = 8 candidates. Used by
      2-layer / real-board configurations and preserves prior behavior.
    * preset name (e.g. ``"multi_resolution"``): 8 directions × the preset's
      distance ladder, emitted distance-major (all 8 dirs at distances[0],
      then [1], …).
    * ``"grid<N>"``: 4 axis-aligned directions × per-grid step bundle.
      Offsets are integer multiples of ``_BOARD_SIZE_MM / N`` so emitted
      points snap to the grid when ``route_head_mm`` is itself grid-aligned.

    Returns raw candidate tuples (x_mm, y_mm, layer, CTYPE_DIRECTIONAL).
    """
    grid_size, distances = parse_directional_mode(mode)
    hx, hy = route_head_mm
    cands: list[tuple[float, float, int, int]] = []

    if grid_size is None:
        for dist in (distances if distances is not None else _DIR_DISTANCES_MM):
            for dx, dy in _DIRECTIONS:
                cands.append(
                    (hx + dx * dist, hy + dy * dist, current_layer, CTYPE_DIRECTIONAL),
                )
        return cands

    spacing = _BOARD_SIZE_MM / grid_size
    for step_cells in _GRID_STEP_CELLS[grid_size]:
        offset = step_cells * spacing
        for dx, dy in _GRID_4DIRS:
            cands.append(
                (hx + dx * offset, hy + dy * offset, current_layer, CTYPE_DIRECTIONAL),
            )
    return cands
