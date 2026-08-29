"""Generate N synthetic KiCad boards — simple_routing_board-style.

Two modes:

1. ``--mode legacy`` (default for backward compatibility)
   - 50 x 30 board, clearance=0.2, trace_width=0.2, continuous pad coords

2. ``--mode grid`` — grid-aligned boards
   - Square board of ``--board-size`` mm (default 100)
   - clearance = trace_width = board_size / 1000 (default 0.1 mm on 100x100)
   - grid_spacing = 2*clearance + trace_width (default 0.3 mm)
   - Pad centers snap to grid cell CENTERS: ``((i+0.5)*g, (j+0.5)*g)``
   - Pin placement restricted to the central 80% of the board.
   - Pairwise min center-to-center distance = ``pad_size + 2*clearance + trace_width``

Each board has 5 nets x 2 pads = 10 SMD pads.  Copper layer count is controlled
by ``--num-layers`` (1 = single-sided F.Cu only, 2 = pads random on F.Cu/B.Cu).
Note: KiCad rejects odd copper counts, so 1-layer still declares F.Cu + B.Cu in
the layer stackup but only uses F.Cu for pads.

Emits ``board_{i:05d}.kicad_pcb`` under ``--out-dir``. Deterministic given seed.

Usage:
    python tools/datagen/synthetic_generator/generate_synthetic_boards.py --n 10000 \
        --mode grid --num-layers 1 \
        --out-dir pcb_dataset_synthetic_5net_2pin_1layer
"""
from __future__ import annotations

import argparse
import math
import random
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import outline_geometry as og

DEFAULT_NUM_NETS = 5
DEFAULT_PADS_PER_NET = 2
CENTRAL_FRAC = 0.8  # default; overridable via --central-frac
PAD_SIZE = 1.0


@dataclass(frozen=True)
class BoardConfig:
    board_w: float
    board_h: float
    clearance: float
    trace_width: float
    grid_spacing: float | None   # None → continuous coords
    grid_count_x: int | None
    grid_count_y: int | None
    pad_size: float
    min_sep: float               # center-to-center
    # Offset added to (i+0.5)*grid_spacing so grid 0 sits far enough from the
    # board edge to satisfy (copper-edge) clearance.  0 for legacy/continuous.
    grid_origin: float = 0.0


ROUNDRECT_RRATIO = 0.25  # matches the (roundrect_rratio ...) emitted by _render


def _compute_min_sep(formula: str, pad_size: float,
                     clearance: float, trace_width: float) -> float:
    if formula == "legacy":
        return pad_size + 2 * clearance + trace_width
    if formula == "four-pitch":
        return pad_size + 4 * (clearance + trace_width)
    raise ValueError(f"unknown min_sep formula {formula!r}")


# Random sequential adsorption of equal disks saturates at ~0.547 area
# fraction, so a min_sep of ``d`` admits at most ``0.547 / (pi*(d/2)^2)`` pad
# centres per unit area. Cost explodes near that limit — ``_place_pads``'s
# 20000-try budget cannot reach it. Boards are therefore sized to stay at
# ``RSA_SAFE_FILL`` of saturation: d2c boards that place cleanly sit around
# 0.36, while a draw demanding 0.99 (131 pads, min_sep 2.524, 1210 mm^2
# usable) cannot be placed at all. The pad-count tail itself is not the
# problem — d3b holds boards up to 145 pads against this generator's 135 —
# the board just has to be sized for the pads it will hold, which is why
# ``generate_one`` samples the net structure BEFORE calling ``cfg_factory``.
RSA_SATURATION = 0.547
RSA_SAFE_FILL = 0.50


def _min_area_for_pads(n_pads: int, min_sep: float,
                       fill: float = RSA_SAFE_FILL) -> float:
    """Usable area needed to place ``n_pads`` at ``min_sep`` without thrashing."""
    per_area = RSA_SATURATION / (math.pi * (min_sep / 2.0) ** 2)
    return n_pads / (per_area * fill)


def _min_sep_for_clearance(pad_size: float, clearance: float,
                           rratio: float = ROUNDRECT_RRATIO) -> float:
    """Smallest EUCLIDEAN center-to-center distance that still guarantees
    ``clearance`` between two SMD roundrect pads.

    ``_place_pads`` rejects candidates on Euclidean distance, but SMD pads are
    square ``roundrect``, not round — two of them at 45 deg come far closer than
    their center distance suggests. With side ``s``, corner radius ``r = rratio*s``
    and both offsets equal to ``d/sqrt(2)``, the nearest features are the facing
    corner arcs, whose centres sit ``s/2 - r`` inside each pad::

        gap = sqrt(2) * (d/sqrt(2) - 2*(s/2 - r)) - 2r  >=  clearance
          =>  d >= sqrt(2)*(s - 2r) + 2r + clearance

    Worked case: s=2.0, r=0.5, min_sep=2.4 puts two pads at offset (1.8, 1.8) —
    Euclidean 2.546, comfortably over 2.4 — yet only 0.1314 mm of copper apart,
    an ERROR against the 0.3 mm netclass clearance on the BARE board. The formula
    gives 2.714 for that geometry. Small pads are unaffected (s=1.2 needs 1.749,
    far under the 3.0 the synthetic datasets use), so the correction only binds
    once pad_size approaches min_sep.
    """
    r = rratio * pad_size
    return math.sqrt(2.0) * (pad_size - 2 * r) + 2 * r + clearance


def _make_config_legacy(pad_size: float, min_sep_override: float | None,
                        min_sep_formula: str = "legacy") -> BoardConfig:
    clearance = 0.2
    trace_width = 0.2
    min_sep = min_sep_override if min_sep_override is not None else (
        _compute_min_sep(min_sep_formula, pad_size, clearance, trace_width)
    )
    return BoardConfig(
        board_w=50.0, board_h=30.0,
        clearance=clearance, trace_width=trace_width,
        grid_spacing=None, grid_count_x=None, grid_count_y=None,
        pad_size=pad_size, min_sep=min_sep,
    )


def _make_config_grid_rect(board_w_target: float, board_h_target: float,
                           pad_size: float, clearance: float, trace_width: float,
                           pitch_formula: str,
                           min_sep_override: float | None,
                           min_sep_formula: str = "legacy") -> BoardConfig:
    # grid_spacing controls the parallel-track pitch:
    #   "c+w"  — tight: adjacent-track edge-to-edge gap = exactly clearance
    #   "2c+w" — loose: adjacent-track edge-to-edge gap = 2 * clearance
    if pitch_formula == "c+w":
        g = clearance + trace_width
    elif pitch_formula == "2c+w":
        g = 2 * clearance + trace_width
    else:
        raise ValueError(f"unknown pitch_formula {pitch_formula!r}")

    # Target an exact nx*ny grid that fits inside the requested board, then
    # *extend* outward by whatever edge-margin is needed so grid 0 and grid
    # n-1 (each axis) clear (clearance + pad_half) from the outline.
    nx = int(round(board_w_target / g))
    ny = int(round(board_h_target / g))
    edge_margin = clearance + pad_size / 2
    needed_shift = max(0.0, edge_margin - g / 2)
    grid_origin = needed_shift
    board_w = nx * g + 2 * needed_shift
    board_h = ny * g + 2 * needed_shift

    min_sep = min_sep_override if min_sep_override is not None else (
        _compute_min_sep(min_sep_formula, pad_size, clearance, trace_width)
    )
    return BoardConfig(
        board_w=board_w, board_h=board_h,
        clearance=clearance, trace_width=trace_width,
        grid_spacing=g, grid_count_x=nx, grid_count_y=ny,
        pad_size=pad_size, min_sep=min_sep,
        grid_origin=grid_origin,
    )


def _make_config_grid(board_size: float, pad_size: float,
                      clearance: float, trace_width: float,
                      pitch_formula: str,
                      min_sep_override: float | None,
                      min_sep_formula: str = "legacy") -> BoardConfig:
    return _make_config_grid_rect(
        board_size, board_size, pad_size, clearance, trace_width,
        pitch_formula, min_sep_override, min_sep_formula,
    )


def _u(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


# ---------------------------------------------------------------------------
# D2-B / D2-B-V real-matched 2L sampling.
#
# Per board: board side ~ lognormal (right-skew like real D3-A/B) — read as
# sqrt(area) once --aspect-sigma stretches the box, net count ~
# Poisson(k * N_ref^2) with N_ref = side / ref_pitch, and per-net fanout from a
# mixture: a 2-pin-dominated bulk plus a rare "rail" (power/GND) heavy tail.
# Geometry rules are either fixed (D2-B) or per-board uniform (D2-B-V).
# ---------------------------------------------------------------------------


def _zipf_weights(kmin: int, kmax: int, s: float,
                  tail_from: int | None = None,
                  tail_mass: float | None = None) -> list[float]:
    """Discrete power-law weights ``P(k) ~ k**-s`` over ``k in [kmin, kmax]``.

    Real boards follow a Zipf body but carry a heavier tail than the pure law
    (power nets: GND/VCC reach 20-42 pads while k**-s puts ~0 there). Measured
    on the d3b 50-board set: MLE ``s=2.955`` with KS distance 0.034 < 0.044
    critical at n=953, but observed P(k>=16)=1.8% against the law's 1.0%.
    ``tail_from``/``tail_mass`` lift that tail to a target mass, spread evenly.
    """
    if kmin < 1 or kmax < kmin:
        raise ValueError(f"invalid zipf range [{kmin}, {kmax}]")
    w = [float(k) ** -s for k in range(kmin, kmax + 1)]
    z = sum(w)
    w = [x / z for x in w]
    if tail_from is not None and tail_mass is not None:
        i0 = tail_from - kmin
        if 0 <= i0 < len(w):
            cur = sum(w[i0:])
            if tail_mass > cur:
                add = (tail_mass - cur) / (len(w) - i0)
                w = [x + (add if j >= i0 else 0.0) for j, x in enumerate(w)]
                z = sum(w)
                w = [x / z for x in w]
    return w


def _order_pads_by_locality(
    pads: list[tuple[float, float, str, bool]],
    pads_per_net: list[int],
    rng: random.Random,
    locality: float,
    decay_to: int | None = None,
) -> list[tuple[float, float, str, bool]]:
    """Permute ``pads`` so :func:`_render`'s in-order slicing yields spatially
    local nets.

    ``_place_pads`` is net-agnostic (uniform/grid sampling) and ``_render``
    assigns net ``i`` the next ``pads_per_net[i]`` entries, so net membership is
    spatially random: a 2-pad net spans ~52% of the board edge, the expected
    distance between two uniform points. Real boards place connected parts
    together — measured on d3b (50 boards, 953 nets), a 2-pad net spans 0.203 of
    the board diagonal against d2b's 0.393.

    ``locality`` picks each next pad from the ``K`` nearest unassigned ones with
    ``K = ceil(len(remaining) ** (1 - locality))``: 0.0 leaves the list untouched
    (K = all, so the emitted board stays bit-identical), 1.0 forces the nearest
    neighbour. 0.7 reproduces d3b's small-net spans (2/3/4-5 pads -> 0.187 /
    0.326 / 0.409 against d3b's 0.203 / 0.321 / 0.440).

    ``decay_to`` fades locality out linearly with net size, reaching 0 at that
    many pads. Real boards are local only for small nets: a d3b 2-pad net spans
    0.203 of the diagonal but a 10+-pad net spans 0.895, i.e. power nets cross
    the whole board. Without a decay a single locality over-localises them
    (0.755 at 10+ pads). ``decay_to=10`` matches both ends.
    """
    if locality <= 0.0:
        return pads
    if not 0.0 <= locality <= 1.0:
        raise ValueError(f"--net-locality must be in [0, 1], got {locality}")
    kmin = min(pads_per_net) if pads_per_net else 2
    rem = list(range(len(pads)))
    order: list[int] = []
    for count in pads_per_net:
        if not rem:
            break
        loc = locality
        if decay_to is not None and decay_to > kmin:
            loc = locality * max(0.0, 1.0 - (count - kmin) / (decay_to - kmin))
        grp = [rem.pop(rng.randrange(len(rem)))]
        if loc > 0.0:
            for _ in range(count - 1):
                if not rem:
                    break
                ax, ay = pads[grp[-1]][0], pads[grp[-1]][1]
                rem.sort(key=lambda i: (pads[i][0] - ax) ** 2 + (pads[i][1] - ay) ** 2)
                k = max(1, math.ceil(len(rem) ** (1.0 - loc)))
                grp.append(rem.pop(rng.randrange(k)))
        else:
            for _ in range(count - 1):
                if not rem:
                    break
                grp.append(rem.pop(rng.randrange(len(rem))))
        order.extend(grp)
    order.extend(rem)
    return [pads[i] for i in order]


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's Poisson sampler using the board RNG (keeps determinism)."""
    if lam <= 0.0:
        return 0
    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= target:
            return k - 1


def _lognormvariate_clip(rng: random.Random, median: float, sigma: float,
                         lo: float, hi: float) -> float:
    """Lognormal with the given median (mu = ln(median)), clipped to [lo, hi]."""
    v = rng.lognormvariate(math.log(median), sigma)
    return min(hi, max(lo, v))


@dataclass(frozen=True)
class D2BParams:
    # board size (lognormal, mm)
    board_median: float
    board_sigma: float
    board_clip_min: float
    board_clip_max: float
    # net count: lam = net_k * (side / ref_pitch) ** 2
    net_k: float
    ref_pitch: float
    min_nets: int
    max_nets: int
    # fanout mixture
    rail_prob: float
    rail_median: float
    rail_sigma: float
    rail_min: int
    rail_max: int
    bulk_base: int
    bulk_lambda: float
    # geometry
    rule_mode: str               # "fixed" | "uniform"
    pitch_formula: str
    min_sep_formula: str
    # fixed-rule values (rule_mode == "fixed")
    clearance: float
    trace_width: float
    pad_size: float
    via_dia: float
    via_drill: float
    # uniform-rule ranges (rule_mode == "uniform")
    uni_clearance_min: float
    uni_clearance_max: float
    uni_clearance_step: float
    uni_width_factor_min: float
    uni_width_factor_max: float
    uni_pad_pitch_mult_min: float
    uni_pad_pitch_mult_max: float
    uni_via_drill_mult_min: float
    uni_via_drill_mult_max: float
    uni_via_dia_mult: float
    # board aspect ratio (0 = square boards).
    # log(long/short) ~ |N(0, aspect_sigma)|; sigma 0.60 reproduces the real
    # d3b aspect quantiles (p50/p75/p90 = 1.49/2.00/2.68 vs 1.49/2.00/2.74).
    aspect_sigma: float = 0.0
    aspect_max: float = 4.0
    aspect_min_short: float = 16.0


def _sample_board_wh(rng: random.Random,
                     pr: D2BParams) -> tuple[float, float, float]:
    """Board box (w, h) plus the size scalar s = sqrt(w*h).

    ``s`` keeps the original lognormal side distribution, so net count (which
    scales with s^2 = area) and pad density are untouched by the aspect draw.
    The ratio is clamped — not rejection-resampled — by ``aspect_max`` and by
    the short-side floor, so the draw stays a single deterministic step.
    With ``aspect_sigma == 0`` no extra rng is consumed and the board is square.
    """
    s = _lognormvariate_clip(rng, pr.board_median, pr.board_sigma,
                             pr.board_clip_min, pr.board_clip_max)
    if pr.aspect_sigma <= 0.0:
        return s, s, s
    r = math.exp(abs(rng.gauss(0.0, pr.aspect_sigma)))
    r = min(r, pr.aspect_max, max(1.0, (s / pr.aspect_min_short) ** 2))
    root = math.sqrt(r)
    w, h = s * root, s / root
    if rng.random() < 0.5:
        w, h = h, w
    return w, h, s


def _sample_uniform_rule(rng: random.Random, pr: D2BParams) -> dict:
    """One uniform design-rule draw (a single net's rules in D2-B-V).

    via clearance = net clearance is implicit (the net_class clearance applies
    to vias too); width >= clearance is enforced.
    """
    steps = []
    c = pr.uni_clearance_min
    while c <= pr.uni_clearance_max + 1e-9:
        steps.append(round(c, 4))
        c += pr.uni_clearance_step
    clearance = rng.choice(steps)
    f = rng.uniform(pr.uni_width_factor_min, pr.uni_width_factor_max)
    step = pr.uni_clearance_step
    width = max(clearance, round(clearance * f / step) * step)
    pitch = clearance + width
    pad_size = pitch * rng.uniform(pr.uni_pad_pitch_mult_min,
                                   pr.uni_pad_pitch_mult_max)
    via_drill = width * rng.uniform(pr.uni_via_drill_mult_min,
                                    pr.uni_via_drill_mult_max)
    via_dia = via_drill * pr.uni_via_dia_mult
    return dict(clearance=clearance, width=width, pad_size=pad_size,
                via_dia=via_dia, via_drill=via_drill)


def _cfg_from_rule(w: float, h: float, rule: dict,
                   pr: D2BParams) -> BoardConfig:
    return _make_config_grid_rect(
        w, h, rule["pad_size"], rule["clearance"], rule["width"],
        pr.pitch_formula, None, min_sep_formula=pr.min_sep_formula,
    )


def _sample_d2b_fanout(rng: random.Random, n_nets: int,
                       pr: D2BParams) -> list[int]:
    """Per-net pad count: bulk (2-pin dominated) + rare rail heavy tail."""
    out = []
    for _ in range(n_nets):
        if rng.random() < pr.rail_prob:
            v = round(_lognormvariate_clip(rng, pr.rail_median, pr.rail_sigma,
                                           pr.rail_min, pr.rail_max))
            out.append(int(v))
        else:
            out.append(pr.bulk_base + _poisson(rng, pr.bulk_lambda))
    return out


def _placement_capacity(cfg: BoardConfig, central_frac: float) -> int:
    """Conservative max pad count that random min-sep placement can fit in the
    central area (so dense small boards don't fail). Derated below the
    disk-packing limit since rejection sampling jams early."""
    area = (cfg.board_w * central_frac) * (cfg.board_h * central_frac)
    return max(4, int(0.40 * area / (cfg.min_sep ** 2)))


def _clip_fanout(fanouts: list[int], cap: int) -> list[int]:
    """Trim total pads down to ``cap``: shrink the largest nets toward 2 first,
    then drop whole nets if still over. Mirrors real boards where small boards
    carry few/no high-fanout rails."""
    fanouts = list(fanouts)
    total = sum(fanouts)
    if total <= cap:
        return fanouts
    for i in sorted(range(len(fanouts)), key=lambda j: -fanouts[j]):
        if total <= cap:
            break
        take = min(fanouts[i] - 2, total - cap)
        if take > 0:
            fanouts[i] -= take
            total -= take
    while total > cap and len(fanouts) > 2:
        total -= fanouts.pop()
    return fanouts


def _geo_spacing_pad(rule_pad: float, board_geo) -> float:
    """Effective pad dimension for grid/min-sep when a THT profile can emit
    pads larger than the rule pad (oval long dim averaged: the worst pair —
    two ovals broadside — is rare, so full-max spacing would over-thin the
    board; routability is re-checked in the pilot)."""
    if board_geo is None or board_geo.tht is None:
        return rule_pad
    t = board_geo.tht
    return max(rule_pad, (t.pad + t.max_dim) / 2.0)


def _census_spacing_pad(rule_pad: float, shape_census: bool,
                        min_sep_formula: str = "four-pitch") -> float:
    """--shape-census SMD-oval spacing widen — LEGACY min-sep formula only.

    Census SMD ovals render up to ``SMD_OVAL_LEN_MULT[1]`` (2.2x) the rule
    pad. Under the "four-pitch" formula (pad + 4(c+w), the shipped d2b_geo
    recipe) the worst broadside oval pair provably still clears — widening
    there would only derate placement capacity board-wide (measured -28%
    pads). The "legacy" formula (pad + 2c + w)
    does leak oval-pair clearance, so only that path gets the THT-style
    averaged widen.
    """
    if not shape_census or min_sep_formula == "four-pitch":
        return rule_pad
    return max(rule_pad, rule_pad * (1.0 + og.SMD_OVAL_LEN_MULT[1]) / 2.0)


def _census_margin_half(rule_pad: float, shape_census: bool) -> float | None:
    """--shape-census worst pad half-extent for edge/cutout/NPTH margins.

    Unlike pair spacing this must cover the FULL oval envelope: a single
    oval near the outline is not a rare pair event, and an under-sized margin
    lets copper cross Edge.Cuts or eat into NPTH clearance.
    None = legacy margin (cfg.pad_size / 2).
    """
    if not shape_census:
        return None
    return rule_pad * og.SMD_OVAL_LEN_MULT[1] / 2.0


def _geo_capacity(cap: int, board_geo) -> int:
    if board_geo is None:
        return cap
    return max(4, int(cap * board_geo.usable_frac()))


def generate_one_d2b(seed: int, num_layers: int, central_frac: float,
                     thru_hole_prob: float, pr: D2BParams,
                     geo: bool = False, shape_census: bool = False) -> str:
    rng = random.Random(seed)
    bw, bh, s = _sample_board_wh(rng, pr)
    n_ref = s / pr.ref_pitch
    lam = pr.net_k * n_ref * n_ref
    n_nets = min(pr.max_nets, max(pr.min_nets, _poisson(rng, lam)))
    pads_per_net = _sample_d2b_fanout(rng, n_nets, pr)
    board_geo = og.sample_geo(rng, bw, bh, shape_census=shape_census) if geo else None
    if board_geo is not None:
        bw, bh = board_geo.outline.w, board_geo.outline.h
    allowed = ("F.Cu", "B.Cu") if num_layers == 2 else ("F.Cu",)

    if pr.rule_mode == "uniform":
        # D2-B-V: each net gets its own design rule (clearance/width/via/pad).
        # The board grid + min-sep use the most conservative (largest) rule so
        # every net's pads fit; per-net clearance is emitted as one net_class
        # per net and resolved by the engine's DRC rule engine at route time.
        net_rules = [_sample_uniform_rule(rng, pr) for _ in range(n_nets)]
        worst = max(net_rules, key=lambda r: r["pad_size"] + 4 * (r["clearance"] + r["width"]))
        spacing = dict(worst, pad_size=_census_spacing_pad(
            _geo_spacing_pad(worst["pad_size"], board_geo), shape_census,
            pr.min_sep_formula))
        cfg = _cfg_from_rule(bw, bh, spacing, pr)
        pads_per_net = _clip_fanout(pads_per_net, _geo_capacity(
            _placement_capacity(cfg, central_frac), board_geo))
        net_rules = net_rules[:len(pads_per_net)]
        pads = _place_pads(rng, cfg, allowed, total_pads=sum(pads_per_net),
                           central_frac=central_frac, thru_hole_prob=thru_hole_prob,
                           max_tries=200000, geo=board_geo,
                           margin_half=_census_margin_half(
                               worst["pad_size"], shape_census))
        return _render(pads, pads_per_net, rng, cfg, net_rules=net_rules,
                       geo=board_geo, shape_census=shape_census)

    # D2-B: a single fixed rule set for the whole board.
    rule = dict(clearance=pr.clearance, width=pr.trace_width, pad_size=pr.pad_size,
                via_dia=pr.via_dia, via_drill=pr.via_drill)
    spacing = dict(rule, pad_size=_census_spacing_pad(
        _geo_spacing_pad(rule["pad_size"], board_geo), shape_census,
        pr.min_sep_formula))
    cfg = _cfg_from_rule(bw, bh, spacing, pr)
    pads_per_net = _clip_fanout(pads_per_net, _geo_capacity(
        _placement_capacity(cfg, central_frac), board_geo))
    pads = _place_pads(rng, cfg, allowed, total_pads=sum(pads_per_net),
                       central_frac=central_frac, thru_hole_prob=thru_hole_prob,
                       max_tries=200000, geo=board_geo,
                       margin_half=_census_margin_half(
                           rule["pad_size"], shape_census))
    cfg_render = replace(cfg, pad_size=pr.pad_size)
    return _render(pads, pads_per_net, rng, cfg_render,
                   via_dia=pr.via_dia, via_drill=pr.via_drill, geo=board_geo,
                   shape_census=shape_census)


def generate_one_paired(seed: int, num_layers: int, central_frac: float,
                        thru_hole_prob: float, pr: D2BParams,
                        geo: bool = False,
                        shape_census: bool = False) -> tuple[str, str]:
    """Generate the SAME board under both rule sets — returns (d2b, d2bv).

    Identical board size, net count, fanout, pad (x,y) positions, thru-hole
    choices and UUIDs; only the design rules differ (D2-B single fixed vs D2-B-V
    per-net). Placement uses the most conservative (coarsest) rule across BOTH
    sets, so the one shared pad layout is valid for either; the capacity clip is
    applied once, so both boards end up with identical net/pad structure (when
    capacity forces clipping, both are clipped the same way). The two renders use
    a fresh RNG seeded identically, so even the UUIDs match — the only byte
    differences between the paired boards are the rule/pad-size lines.
    """
    rng = random.Random(seed)
    bw, bh, s = _sample_board_wh(rng, pr)
    n_ref = s / pr.ref_pitch
    lam = pr.net_k * n_ref * n_ref
    n_nets = min(pr.max_nets, max(pr.min_nets, _poisson(rng, lam)))
    pads_per_net = _sample_d2b_fanout(rng, n_nets, pr)
    board_geo = og.sample_geo(rng, bw, bh, shape_census=shape_census) if geo else None
    if board_geo is not None:
        bw, bh = board_geo.outline.w, board_geo.outline.h
    net_rules = [_sample_uniform_rule(rng, pr) for _ in range(n_nets)]
    fixed_rule = dict(clearance=pr.clearance, width=pr.trace_width,
                      pad_size=pr.pad_size, via_dia=pr.via_dia, via_drill=pr.via_drill)
    worst = max(net_rules + [fixed_rule],
                key=lambda r: r["pad_size"] + 4 * (r["clearance"] + r["width"]))
    spacing = dict(worst, pad_size=_census_spacing_pad(
        _geo_spacing_pad(worst["pad_size"], board_geo), shape_census,
        pr.min_sep_formula))
    cfg = _cfg_from_rule(bw, bh, spacing, pr)
    pads_per_net = _clip_fanout(pads_per_net, _geo_capacity(
        _placement_capacity(cfg, central_frac), board_geo))
    net_rules = net_rules[:len(pads_per_net)]
    allowed = ("F.Cu", "B.Cu") if num_layers == 2 else ("F.Cu",)
    pads = _place_pads(rng, cfg, allowed, total_pads=sum(pads_per_net),
                       central_frac=central_frac, thru_hole_prob=thru_hole_prob,
                       max_tries=200000, geo=board_geo,
                       margin_half=_census_margin_half(
                           worst["pad_size"], shape_census))
    # D2-B render: same conservative grid/positions, but the FIXED rule values.
    cfg_d2b = replace(cfg, clearance=pr.clearance, trace_width=pr.trace_width,
                      pad_size=pr.pad_size)
    d2b = _render(pads, pads_per_net, random.Random(seed), cfg_d2b,
                  via_dia=pr.via_dia, via_drill=pr.via_drill, geo=board_geo,
                  shape_census=shape_census)
    d2bv = _render(pads, pads_per_net, random.Random(seed), cfg,
                   net_rules=net_rules, geo=board_geo,
                   shape_census=shape_census)
    return d2b, d2bv


def _place_pads(
    rng: random.Random,
    cfg: BoardConfig,
    allowed_layers: tuple[str, ...],
    total_pads: int,
    central_frac: float = CENTRAL_FRAC,
    thru_hole_prob: float = 0.0,
    max_tries: int = 20000,
    geo=None,
    margin_half: float | None = None,
) -> list[tuple[float, float, str, bool]]:
    """Place pads with min-sep on (x,y) only — i.e. (x,y) is unique across
    layers (no two pads share the same (x,y) regardless of side). Each pad
    is independently chosen as through-hole with ``thru_hole_prob``; through-
    hole pads are layer-agnostic (rendered with all-copper-layer pad). Non-
    thru pads pick a layer uniformly from ``allowed_layers``.
    """
    mx = cfg.board_w * (1 - central_frac) / 2
    my = cfg.board_h * (1 - central_frac) / 2
    min_sep_sq = cfg.min_sep ** 2
    # Edge/cutout/NPTH margin half-extent; --shape-census passes the worst
    # SMD oval envelope here (legacy None = the square rule pad).
    pad_half = cfg.pad_size / 2.0 if margin_half is None else margin_half
    pads: list[tuple[float, float, str, bool]] = []

    def _accept(x: float, y: float) -> None:
        thru = (thru_hole_prob > 0.0) and (rng.random() < thru_hole_prob)
        layer = "F.Cu" if thru else rng.choice(allowed_layers)
        pads.append((x, y, layer, thru))

    if cfg.grid_spacing is not None:
        g = cfg.grid_spacing
        o = cfg.grid_origin
        # Grid cell (i, j) has center at (o + (i+0.5)*g, o + (j+0.5)*g).
        # Cell center must lie inside [mx, board_w - mx].
        i_lo = max(0, int(math.ceil((mx - o) / g - 0.5)))
        j_lo = max(0, int(math.ceil((my - o) / g - 0.5)))
        i_hi = min(cfg.grid_count_x - 1,
                   int(math.floor((cfg.board_w - mx - o) / g - 0.5)))
        j_hi = min(cfg.grid_count_y - 1,
                   int(math.floor((cfg.board_h - my - o) / g - 0.5)))
        for _ in range(max_tries):
            i = rng.randint(i_lo, i_hi)
            j = rng.randint(j_lo, j_hi)
            x = o + (i + 0.5) * g
            y = o + (j + 0.5) * g
            if geo is not None and not geo.allows_pad(
                    x, y, pad_half, cfg.clearance):
                continue
            if all((x - px) ** 2 + (y - py) ** 2 >= min_sep_sq for px, py, _, _ in pads):
                _accept(x, y)
                if len(pads) == total_pads:
                    return pads
    else:
        x_lo, x_hi = mx, cfg.board_w - mx
        y_lo, y_hi = my, cfg.board_h - my
        for _ in range(max_tries):
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
            if geo is not None and not geo.allows_pad(
                    x, y, pad_half, cfg.clearance):
                continue
            if all((x - px) ** 2 + (y - py) ** 2 >= min_sep_sq for px, py, _, _ in pads):
                _accept(x, y)
                if len(pads) == total_pads:
                    return pads

    raise RuntimeError(
        f"could not place {total_pads} pads with min_sep={cfg.min_sep} "
        f"inside {cfg.board_w}x{cfg.board_h} (central {central_frac*100:.0f}%) "
        f"after {max_tries} tries"
    )


_HEADER = '''(kicad_pcb
  (version 20241229)
  (generator "synthetic_board_generator")
  (generator_version "9.0.5")
  (general
    (thickness 1.6)
    (legacy_teardrops no)
  )
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user "B.Mask")
    (39 "F.Mask" user "F.Mask")
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (44 "Edge.Cuts" user)
  )
  (setup
    (pad_to_mask_clearance 0)
    (allow_soldermask_bridges_in_footprints no)
    (pcbplotparams
      (layerselection 0x00010fc_ffffffff)
      (plot_on_all_layers_selection 0x0000000_00000000)
      (disableapertmacros no)
      (usegerberextensions no)
      (usegerberattributes yes)
      (usegerberadvancedattributes yes)
      (creategerberjobfile yes)
      (dashed_line_dash_ratio 12.000000)
      (dashed_line_gap_ratio 3.000000)
      (svgprecision 4)
      (plotframeref no)
      (viasonmask no)
      (mode 1)
      (useauxorigin no)
      (hpglpennumber 1)
      (hpglpenspeed 20)
      (hpglpendiameter 15.000000)
      (pdf_front_fp_property_popups yes)
      (pdf_back_fp_property_popups yes)
      (dxfpolygonmode yes)
      (dxfimperialunits yes)
      (dxfusepcbnewfont yes)
      (psnegative no)
      (psa4output no)
      (plotreference yes)
      (plotvalue yes)
      (plotfptext yes)
      (plotinvisibletext no)
      (sketchpadsonfab no)
      (subtractmaskfromsilk no)
      (outputformat 1)
      (mirror no)
      (drillshape 1)
      (scaleselection 1)
      (outputdirectory "")
    )
  )
'''


def _render(pads: list[tuple[float, float, str, bool]],
            pads_per_net: list[int],
            rng: random.Random, cfg: BoardConfig,
            via_dia: float = 0.6, via_drill: float = 0.3,
            thru_pad_drill: float | None = None,
            net_rules: list[dict] | None = None,
            geo=None, shape_census: bool = False) -> str:
    # pns_rl_router.cpp promotes the netclass clearance into the router's
    # m_MinClearance/m_CopperEdgeClearance/etc. at init time and attaches a
    # DRC_ENGINE, so .kicad_pcb-only boards route correctly without requiring
    # any additional setup directives.
    #
    # When ``net_rules`` is given (one rule dict per net), each net is placed in
    # its OWN legacy net_class via ``(add_net "NETk")`` — the engine resolves
    # per-net clearance through its DRC rule engine (verified), exactly like the
    # multi-net_class real boards in pcb_dataset/real. Pad size and thru-hole
    # drill then follow each net's own rule.
    num_nets = len(pads_per_net)
    out = [_HEADER, ""]
    out.append('  (net 0 "")')
    for n in range(1, num_nets + 1):
        out.append(f'  (net {n} "NET{n}")')
    out.append("")
    if net_rules is not None:
        base = min(net_rules, key=lambda r: r["clearance"])
        out.append('  (net_class "Default" "Default net class"')
        out.append(f"    (clearance {base['clearance']})")
        out.append(f"    (trace_width {base['width']})")
        out.append(f"    (via_dia {base['via_dia']:.4f})")
        out.append(f"    (via_drill {base['via_drill']:.4f})")
        out.append("    (uvia_dia 0.3)")
        out.append("    (uvia_drill 0.1)")
        out.append("  )")
        for i, r in enumerate(net_rules):
            nid = i + 1
            out.append(f'  (net_class "NC{nid}" ""')
            out.append(f"    (clearance {r['clearance']})")
            out.append(f"    (trace_width {r['width']})")
            out.append(f"    (via_dia {r['via_dia']:.4f})")
            out.append(f"    (via_drill {r['via_drill']:.4f})")
            out.append("    (uvia_dia 0.3)")
            out.append("    (uvia_drill 0.1)")
            out.append(f'    (add_net "NET{nid}")')
            out.append("  )")
    else:
        out.append('  (net_class "Default" "Default net class"')
        out.append(f"    (clearance {cfg.clearance})")
        out.append(f"    (trace_width {cfg.trace_width})")
        out.append(f"    (via_dia {via_dia})")
        out.append(f"    (via_drill {via_drill})")
        out.append("    (uvia_dia 0.3)")
        out.append("    (uvia_drill 0.1)")
        out.append("  )")
    out.append("")

    drill = via_drill if thru_pad_drill is None else thru_pad_drill
    pad_idx = 0
    for net_idx, count in enumerate(pads_per_net):
        net_id = net_idx + 1
        pad_sz = cfg.pad_size if net_rules is None else net_rules[net_idx]["pad_size"]
        pad_drill = drill if net_rules is None else net_rules[net_idx]["via_drill"]
        # Diversified THT profile: one shape draw per net (rng draw happens
        # for every net so the stream stays fixed whether or not the net has
        # thru pads).
        tht_pad = (geo.tht.sample_net_pad(rng)
                   if geo is not None and geo.tht is not None else None)
        # --shape-census: one SMD shape draw per net (fixed 3-draw cost,
        # unconditional — same stream discipline as the THT draw above).
        # Off (default) draws nothing: legacy byte-reproducibility.
        smd_pad = og.sample_smd_pad(rng, pad_sz) if shape_census else None
        for slot in range(count):
            x, y, layer, thru = pads[pad_idx]
            pad_idx += 1
            fp_uuid = _u(rng)
            pad_uuid = _u(rng)
            ref = f"P{net_id}_{slot+1}"
            if thru:
                fp_name = "SamplePad:PTH"
                out.append(f'  (footprint "{fp_name}"')
                out.append('    (layer "F.Cu")')
                out.append(f"    (at {x:.6f} {y:.6f})")
                out.append(f'    (uuid "{fp_uuid}")')
                out.append(f'    (property "Reference" "{ref}"')
                out.append("      (at 0 -1)")
                out.append('      (layer "F.SilkS")')
                out.append("      (effects (font (size 0.6 0.6) (thickness 0.1)))")
                out.append("    )")
                out.append(f'    (property "Value" "Pad{slot+1}"')
                out.append("      (at 0 1)")
                out.append('      (layer "F.Fab")')
                out.append("      (effects (font (size 0.6 0.6) (thickness 0.1)))")
                out.append("    )")
                if tht_pad is not None:
                    t_shape, t_w, t_h, t_drill = tht_pad
                else:
                    t_shape, t_w, t_h, t_drill = "circle", pad_sz, pad_sz, pad_drill
                out.append(f'    (pad "1" thru_hole {t_shape}')
                out.append("      (at 0 0)")
                out.append(f"      (size {t_w} {t_h})")
                out.append(f"      (drill {t_drill})")
                out.append('      (layers "*.Cu" "*.Mask")')
                out.append(f'      (net {net_id} "NET{net_id}")')
                out.append(f'      (uuid "{pad_uuid}")')
                out.append("    )")
                out.append("  )")
                out.append("")
            else:
                pad_layers = (
                    '"F.Cu" "F.Paste" "F.Mask"' if layer == "F.Cu"
                    else '"B.Cu" "B.Paste" "B.Mask"'
                )
                fp_name = "SamplePad:FCu" if layer == "F.Cu" else "SamplePad:BCu"
                out.append(f'  (footprint "{fp_name}"')
                out.append(f'    (layer "{layer}")')
                out.append(f"    (at {x:.6f} {y:.6f})")
                out.append(f'    (uuid "{fp_uuid}")')
                out.append(f'    (property "Reference" "{ref}"')
                out.append("      (at 0 -1)")
                out.append('      (layer "F.SilkS")')
                out.append("      (effects (font (size 0.6 0.6) (thickness 0.1)))")
                out.append("    )")
                out.append(f'    (property "Value" "Pad{slot+1}"')
                out.append("      (at 0 1)")
                out.append('      (layer "F.Fab")')
                out.append("      (effects (font (size 0.6 0.6) (thickness 0.1)))")
                out.append("    )")
                if smd_pad is not None:
                    s_shape, s_w, s_h = smd_pad
                else:
                    s_shape, s_w, s_h = "roundrect", pad_sz, pad_sz
                out.append(f'    (pad "1" smd {s_shape}')
                out.append("      (at 0 0)")
                out.append(f"      (size {s_w} {s_h})")
                out.append(f"      (layers {pad_layers})")
                if s_shape == "roundrect":
                    out.append("      (roundrect_rratio 0.25)")
                out.append(f'      (net {net_id} "NET{net_id}")')
                out.append(f'      (uuid "{pad_uuid}")')
                out.append("    )")
                out.append("  )")
                out.append("")

    if geo is not None:
        geo.emit_hole_footprints(rng, out)
        geo.emit_edge_cuts(rng, out)
    else:
        edge_uuid = _u(rng)
        out.append("  (gr_rect")
        out.append("    (start 0.0 0.0)")
        out.append(f"    (end {cfg.board_w} {cfg.board_h})")
        out.append("    (stroke (width 0.15) (type solid))")
        out.append("    (fill none)")
        out.append('    (layer "Edge.Cuts")')
        out.append(f'    (uuid "{edge_uuid}")')
        out.append("  )")
    out.append(")")
    return "\n".join(out)


def _sample_pads_per_net(rng: random.Random, nets_min: int, nets_max: int,
                         pads_per_net_min: int, pads_per_net_max: int,
                         pads_per_net_weights: list[float] | None) -> list[int]:
    """Per-net pad counts — the same draw the inline path in ``generate_one``
    makes, factored out so the cfg_factory path can run it before sizing the
    board. Consumes RNG identically."""
    n_nets = rng.randint(nets_min, nets_max)
    pad_choices = list(range(pads_per_net_min, pads_per_net_max + 1))
    if pads_per_net_weights is not None:
        if len(pads_per_net_weights) != len(pad_choices):
            raise ValueError(
                f"pads_per_net_weights length {len(pads_per_net_weights)} "
                f"!= range size {len(pad_choices)} "
                f"({pads_per_net_min}..{pads_per_net_max})"
            )
        return rng.choices(pad_choices, weights=pads_per_net_weights, k=n_nets)
    return [rng.randint(pads_per_net_min, pads_per_net_max)
            for _ in range(n_nets)]


def generate_one(seed: int, cfg: BoardConfig | None, num_layers: int,
                 central_frac: float = CENTRAL_FRAC,
                 nets_min: int = DEFAULT_NUM_NETS,
                 nets_max: int = DEFAULT_NUM_NETS,
                 pads_per_net_min: int = DEFAULT_PADS_PER_NET,
                 pads_per_net_max: int = DEFAULT_PADS_PER_NET,
                 pads_per_net_weights: list[float] | None = None,
                 fixed_pads_per_net: list[int] | None = None,
                 via_dia: float = 0.6, via_drill: float = 0.3,
                 thru_hole_prob: float = 0.0,
                 net_locality: float = 0.0,
                 net_locality_decay: int | None = None,
                 size_board_for_pads: bool = False,
                 cfg_factory=None) -> str:
    """Generate one board.

    Either pass a pre-built ``cfg`` (fixed dims) or a ``cfg_factory(rng) -> BoardConfig``
    callable to sample dimensions per board (e.g. random rectangle in [80,120] mm).
    """
    rng = random.Random(seed)
    allowed = ("F.Cu", "B.Cu") if num_layers == 2 else ("F.Cu",)

    if cfg is None:
        if cfg_factory is None:
            raise ValueError("either cfg or cfg_factory must be provided")
        if not size_board_for_pads:
            # LEGACY ORDER — board first, then nets, drawing from the SAME rng.
            # The shipped datasets (d1 grid50, d2a synth_2L_v2, ...) depend on
            # this exact RNG consumption order; reordering it silently
            # regenerates different boards for the same seed (pinned by
            # tests/test_synthetic_dataset_reproduction.py).
            cfg = cfg_factory(rng)
        # Net structure first, board second: the board has to be large enough
        # for the pads it will hold. Drawing its size independently lets a Zipf
        # draw of several large nets demand a placement past the RSA limit,
        # which _place_pads can only report as a failure after burning its whole
        # try budget (see _min_area_for_pads). Opt-in because it changes the RNG
        # order (see above).
        if fixed_pads_per_net is not None:
            pads_per_net = list(fixed_pads_per_net)
        else:
            pads_per_net = _sample_pads_per_net(
                rng, nets_min, nets_max, pads_per_net_min, pads_per_net_max,
                pads_per_net_weights,
            )
        total = sum(pads_per_net)
        if size_board_for_pads:
            cfg = cfg_factory(rng, total_pads=total)
        pads = _place_pads(rng, cfg, allowed, total_pads=total,
                           central_frac=central_frac,
                           thru_hole_prob=thru_hole_prob)
        pads = _order_pads_by_locality(pads, pads_per_net, rng, net_locality,
                                       net_locality_decay)
        return _render(pads, pads_per_net, rng, cfg,
                       via_dia=via_dia, via_drill=via_drill)

    if fixed_pads_per_net is not None:
        pads_per_net = list(fixed_pads_per_net)
        total = sum(pads_per_net)
        pads = _place_pads(rng, cfg, allowed, total_pads=total,
                           central_frac=central_frac,
                           thru_hole_prob=thru_hole_prob)
        pads = _order_pads_by_locality(pads, pads_per_net, rng, net_locality,
                                           net_locality_decay)
        return _render(pads, pads_per_net, rng, cfg,
                       via_dia=via_dia, via_drill=via_drill)
    n_nets = rng.randint(nets_min, nets_max)
    pad_choices = list(range(pads_per_net_min, pads_per_net_max + 1))
    if pads_per_net_weights is not None:
        if len(pads_per_net_weights) != len(pad_choices):
            raise ValueError(
                f"pads_per_net_weights length {len(pads_per_net_weights)} "
                f"!= range size {len(pad_choices)} "
                f"({pads_per_net_min}..{pads_per_net_max})"
            )
        pads_per_net = rng.choices(pad_choices,
                                   weights=pads_per_net_weights,
                                   k=n_nets)
    else:
        pads_per_net = [rng.randint(pads_per_net_min, pads_per_net_max)
                        for _ in range(n_nets)]
    total = sum(pads_per_net)
    pads = _place_pads(rng, cfg, allowed, total_pads=total,
                       central_frac=central_frac,
                       thru_hole_prob=thru_hole_prob)
    pads = _order_pads_by_locality(pads, pads_per_net, rng, net_locality,
                                   net_locality_decay)
    return _render(pads, pads_per_net, rng, cfg,
                   via_dia=via_dia, via_drill=via_drill)


def _run_d2b(args) -> None:
    pr = D2BParams(
        board_median=args.board_lognormal_median,
        board_sigma=args.board_lognormal_sigma,
        board_clip_min=args.board_clip_min,
        board_clip_max=args.board_clip_max,
        aspect_sigma=args.aspect_sigma,
        aspect_max=args.aspect_max,
        aspect_min_short=args.aspect_min_short,
        net_k=args.net_density_k,
        min_nets=args.min_nets,
        ref_pitch=args.net_ref_pitch,
        max_nets=args.max_nets,
        rail_prob=args.rail_prob,
        rail_median=args.rail_median,
        rail_sigma=args.rail_sigma,
        rail_min=args.rail_min,
        rail_max=args.rail_max,
        bulk_base=args.bulk_base,
        bulk_lambda=args.bulk_lambda,
        rule_mode=args.rule_mode,
        pitch_formula=args.pitch_formula,
        min_sep_formula=args.min_sep_formula,
        clearance=args.clearance,
        trace_width=args.trace_width,
        pad_size=args.pad_size,
        via_dia=args.via_dia,
        via_drill=args.via_drill,
        uni_clearance_min=args.uni_clearance_min,
        uni_clearance_max=args.uni_clearance_max,
        uni_clearance_step=args.uni_clearance_step,
        uni_width_factor_min=args.uni_width_factor_min,
        uni_width_factor_max=args.uni_width_factor_max,
        uni_pad_pitch_mult_min=args.uni_pad_pitch_mult_min,
        uni_pad_pitch_mult_max=args.uni_pad_pitch_mult_max,
        uni_via_drill_mult_min=args.uni_via_drill_mult_min,
        uni_via_drill_mult_max=args.uni_via_drill_mult_max,
        uni_via_dia_mult=args.uni_via_dia_mult,
    )
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paired_dir = None
    if pr.rule_mode == "paired":
        if not args.paired_dir:
            raise SystemExit("--rule-mode paired requires --paired-dir (the D2-B-V twin dir)")
        paired_dir = Path(args.paired_dir)
        if not paired_dir.is_absolute():
            paired_dir = repo_root / paired_dir
        paired_dir.mkdir(parents=True, exist_ok=True)

    tag = {"uniform": "D2-B-V (per-net)", "paired": "PAIRED (D2-B + D2-B-V)"}.get(
        pr.rule_mode, "D2-B (fixed)")
    print(f"Generating {args.n} d2b boards [{tag}] into {out_dir}"
          + (f" + {paired_dir}" if paired_dir else ""))
    print(f"  board side ~ lognormal(median={pr.board_median}, sigma={pr.board_sigma}) "
          f"clip[{pr.board_clip_min},{pr.board_clip_max}] mm"
          + (" (= sqrt(area))" if pr.aspect_sigma > 0 else " (square)"))
    if pr.aspect_sigma > 0:
        print(f"  aspect ~ exp|N(0,{pr.aspect_sigma})| clip[1,{pr.aspect_max}] "
              f"short side >= {pr.aspect_min_short} mm, orientation 50/50")
    print(f"  nets ~ Poisson(k={pr.net_k} * (side/{pr.ref_pitch})^2), min 2, max {pr.max_nets}")
    print(f"  fanout: {1-pr.rail_prob:.0%} bulk({pr.bulk_base}+Poisson({pr.bulk_lambda})) "
          f"+ {pr.rail_prob:.0%} rail(lognormal med={pr.rail_median} sigma={pr.rail_sigma} "
          f"clip[{pr.rail_min},{pr.rail_max}])")
    if pr.rule_mode == "uniform":
        print(f"  rules ~ uniform: clearance[{pr.uni_clearance_min},{pr.uni_clearance_max}] "
              f"step {pr.uni_clearance_step}; width=c*[{pr.uni_width_factor_min},"
              f"{pr.uni_width_factor_max}]; pad=pitch*[{pr.uni_pad_pitch_mult_min},"
              f"{pr.uni_pad_pitch_mult_max}]; via_drill=w*[{pr.uni_via_drill_mult_min},"
              f"{pr.uni_via_drill_mult_max}]; via_dia=drill*{pr.uni_via_dia_mult}")
    else:
        print(f"  rules fixed: clearance={pr.clearance} width={pr.trace_width} "
              f"pad={pr.pad_size} via={pr.via_dia}/{pr.via_drill}")
    print(f"  thru-hole prob={args.thru_hole_prob}, layers={args.num_layers}, "
          f"central={args.central_frac*100:.0f}%, min_sep={args.min_sep_formula}, seed={args.seed}")
    if args.geo:
        print("  GEO: real-matched outlines (4-line rect / fillet / rectilinear / "
              "circle) + cutouts + NPTH + slots + THT profile "
              "(rates: outline_geometry.py census constants)")

    width = max(5, len(str(args.start_index + args.n - 1)))
    report_every = max(1, args.n // 20)
    for i in range(args.n):
        idx = args.start_index + i
        seed = (args.seed * 1_000_003 + idx if args.seed_mode == "legacy"
                else args.seed + idx)
        if pr.rule_mode == "paired":
            d2b, d2bv = generate_one_paired(
                seed=seed, num_layers=args.num_layers,
                central_frac=args.central_frac,
                thru_hole_prob=args.thru_hole_prob, pr=pr, geo=args.geo,
                shape_census=args.shape_census)
            (out_dir / f"board_{idx:0{width}d}.kicad_pcb").write_text(d2b)
            (paired_dir / f"board_{idx:0{width}d}.kicad_pcb").write_text(d2bv)
        else:
            text = generate_one_d2b(
                seed=seed, num_layers=args.num_layers,
                central_frac=args.central_frac,
                thru_hole_prob=args.thru_hole_prob, pr=pr, geo=args.geo,
                shape_census=args.shape_census)
            (out_dir / f"board_{idx:0{width}d}.kicad_pcb").write_text(text)
        if (i + 1) % report_every == 0 or i + 1 == args.n:
            print(f"  {i+1}/{args.n}")
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["legacy", "grid", "d2b"], default="legacy")
    p.add_argument("--board-size", type=float, default=100.0,
                   help="square board side length (mm), grid mode only. "
                        "Overridden when --board-size-min/--board-size-max are set.")
    p.add_argument("--board-size-min", type=float, default=None,
                   help="if set together with --board-size-max, board width and "
                        "height are independently sampled uniformly in "
                        "[min, max] mm per board (rectangular boards).")
    p.add_argument("--board-size-max", type=float, default=None,
                   help="see --board-size-min")
    p.add_argument("--pad-size-min", type=float, default=None,
                   help="if set together with --pad-size-max, pad size is sampled "
                        "uniformly in [min, max] mm PER BOARD (real boards differ "
                        "in package pitch: the d3b set's per-board median pad width "
                        "spans 1.00-1.70 mm, p25-p75). Requires --mode grid with "
                        "--board-size-min/max. min_sep is raised per board to the "
                        "shape-aware floor for the sampled size, so a larger pad "
                        "never produces a bare-board clearance ERROR.")
    p.add_argument("--pad-size-max", type=float, default=None,
                   help="see --pad-size-min")
    p.add_argument("--clearance", type=float, default=0.05,
                   help="clearance in mm (grid mode only, default 0.05)")
    p.add_argument("--trace-width", type=float, default=0.05,
                   help="trace width in mm (grid mode only, default 0.05)")
    p.add_argument("--pitch-formula", choices=["c+w", "2c+w"], default="c+w",
                   help="parallel-track pitch formula: 'c+w' (tight, default; "
                        "edge-to-edge gap = clearance) or '2c+w' (loose)")
    p.add_argument("--pad-size", type=float, default=PAD_SIZE)
    p.add_argument("--via-dia", type=float, default=0.6,
                   help="via outer diameter in mm (default 0.6)")
    p.add_argument("--via-drill", type=float, default=0.3,
                   help="via drill diameter in mm (default 0.3)")
    p.add_argument("--min-sep", type=float, default=None,
                   help="override center-to-center min distance (mm)")
    p.add_argument("--min-sep-formula", choices=["legacy", "four-pitch"],
                   default="legacy",
                   help="min_sep formula when --min-sep is not set. "
                        "'legacy' = pad + 2c + w (default, matches 5net_2pin); "
                        "'four-pitch' = pad + 4*(c + w).")
    p.add_argument("--num-layers", type=int, default=2, choices=[1, 2])
    p.add_argument("--thru-hole-prob", type=float, default=0.0,
                   help="per-pad probability of being a through-hole (PTH) pad "
                        "exposed on all copper layers (drill = via_drill). "
                        "Non-thru pads pick a layer uniformly from allowed_layers. "
                        "Default 0.0 (all SMD).")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--seed-mode", choices=["legacy", "linear"], default="linear",
                   help="per-board RNG seed derivation. 'linear' (default): "
                        "seed = base_seed + global_index — decoupled from shard "
                        "layout, so any SHARDS count yields identical boards and "
                        "splits stay disjoint via distinct base offsets "
                        "(e.g. train base 0, val 1_000_000_000, test 2_000_000_000). "
                        "'legacy': seed = base_seed * 1_000_003 + global_index — "
                        "the derivation the existing released datasets were built "
                        "with; pin this to reproduce them bit-for-bit.")
    p.add_argument("--central-frac", type=float, default=CENTRAL_FRAC,
                   help="fraction of board (centered) where pad centers may go "
                        "(default 0.8; use 1.0 for full board coverage)")
    p.add_argument("--nets-min", type=int, default=DEFAULT_NUM_NETS,
                   help=f"min nets per board (default {DEFAULT_NUM_NETS})")
    p.add_argument("--nets-max", type=int, default=DEFAULT_NUM_NETS,
                   help=f"max nets per board (default {DEFAULT_NUM_NETS})")
    p.add_argument("--pads-per-net-min", type=int, default=DEFAULT_PADS_PER_NET,
                   help=f"min pads per net (default {DEFAULT_PADS_PER_NET})")
    p.add_argument("--pads-per-net-max", type=int, default=DEFAULT_PADS_PER_NET,
                   help=f"max pads per net (default {DEFAULT_PADS_PER_NET})")
    p.add_argument("--fixed-pads-per-net", type=str, default=None,
                   help="comma-separated fixed pads-per-net list, e.g. "
                        "'2,2,2,3,4' = 3 nets of 2 pins + 1 net of 3 + 1 net of 4. "
                        "Overrides --nets-*/--pads-per-net-* and forces every "
                        "board to have exactly this net/pin structure.")
    p.add_argument("--pads-per-net-weights", type=str, default=None,
                   help="comma-separated weights for pads_per_net values "
                        "from min..max. e.g. --pads-per-net-min 2 "
                        "--pads-per-net-max 4 --pads-per-net-weights 0.7,0.2,0.1 "
                        "means 2 pins=70%%, 3=20%%, 4=10%%. "
                        "If unset, uniform random over [min, max].")
    p.add_argument("--pads-per-net-zipf", type=float, default=None,
                   help="sample pads-per-net from a discrete Zipf P(k) ~ k**-s "
                        "over [--pads-per-net-min, --pads-per-net-max] instead of "
                        "uniform. Mutually exclusive with --pads-per-net-weights. "
                        "s=2.955 with --pads-per-net-min 2 --pads-per-net-max 42 "
                        "matches the d3b real-PCB set (MLE fit; KS 0.034 < 0.044 "
                        "critical at n=953).")
    p.add_argument("--pads-per-net-zipf-tail", type=str, default=None,
                   help="'FROM:MASS' — lift the Zipf tail P(k >= FROM) to MASS, "
                        "spread evenly (real power nets are heavier than k**-s). "
                        "'16:0.018' matches d3b. Requires --pads-per-net-zipf.")
    p.add_argument("--net-locality", type=float, default=0.0,
                   help="spatial locality of net membership, in [0, 1]. Pads are "
                        "placed net-agnostically, so the default 0.0 (unchanged, "
                        "bit-identical) makes a 2-pad net span ~52%% of the board "
                        "edge. 1.0 forces nearest-neighbour grouping; 0.5 "
                        "reproduces the d3b real-PCB net spans.")
    p.add_argument("--net-locality-decay", type=int, default=None,
                   help="fade --net-locality out linearly with net size, "
                        "reaching 0 at this many pads. Real power nets are "
                        "global, not local (d3b: 2-pad nets span 0.203 of the "
                        "diagonal, 10+-pad nets 0.895). 10 matches both ends.")
    p.add_argument("--size-board-for-pads", action="store_true",
                   help="sample the net structure BEFORE the board and enlarge "
                        "the board to hold those pads (_min_area_for_pads). "
                        "Needed with a heavy-tailed --pads-per-net-zipf: an "
                        "independently drawn board can be asked to place past "
                        "the RSA limit and _place_pads then burns its whole try "
                        "budget before failing (observed at board 17,198/100k: "
                        "131 pads, 99%% of saturation). OFF by "
                        "default because it changes the RNG consumption order, "
                        "which the existing datasets depend on "
                        "(tests/test_synthetic_dataset_reproduction.py).")

    # --- D2-B / D2-B-V (--mode d2b) -------------------------------------
    g = p.add_argument_group("d2b mode (real-D3-matched 2L)")
    g.add_argument("--rule-mode", choices=["fixed", "uniform", "paired"], default="fixed",
                   help="d2b geometry: 'fixed' = D2-B (single rule set); "
                        "'uniform' = D2-B-V (per-net rules); "
                        "'paired' = emit BOTH from one shared layout — D2-B to "
                        "--out-dir, D2-B-V to --paired-dir (identical board/nets/"
                        "pad-positions, only rules differ).")
    g.add_argument("--paired-dir", type=str, default=None,
                   help="(rule-mode paired) output dir for the D2-B-V twin; "
                        "--out-dir receives the D2-B twin.")
    g.add_argument("--geo", action="store_true",
                   help="sample real-matched board geometry per board: outline "
                        "mix (4-line rect / corner-fillet / rectilinear poly / "
                        "circle), internal cutouts, NPTH mounting holes, oval "
                        "slots, and a diversified THT pad profile. Rates and "
                        "ranges live in outline_geometry.py (d3b census). "
                        "Off = plain gr_rect boards.")
    g.add_argument("--shape-census", action="store_true",
                   help="census-matched pad boundary shapes (d3b census): "
                        "SMD pads draw rect/oval/roundrect/circle per "
                        "net (off: always roundrect) and the --geo THT "
                        "profile switches to the census mix (oval-dominant). "
                        "Off = fixed shapes, byte-reproducible with existing "
                        "dataset seeds. Pair with shape_obs training so the "
                        "channel is not synthetic-constant (OOD guard).")
    g.add_argument("--board-lognormal-median", type=float, default=45.0)
    g.add_argument("--board-lognormal-sigma", type=float, default=0.30)
    g.add_argument("--board-clip-min", type=float, default=20.0)
    g.add_argument("--board-clip-max", type=float, default=90.0)
    g.add_argument("--aspect-sigma", type=float, default=0.0,
                   help="board aspect ratio: log(long/short) ~ |N(0, sigma)|, "
                        "long axis 50/50 horizontal/vertical, board area "
                        "unchanged (the lognormal side becomes sqrt(area)). "
                        "0 (default) = square boards + no rng draw, so "
                        "existing dataset seeds stay byte-reproducible. "
                        "0.60 matches the real d3b aspect quantiles.")
    g.add_argument("--aspect-max", type=float, default=4.0,
                   help="aspect clamp (real d3b: p95 ~3.3, p99 7.5 — the "
                        "extreme strips route differently and are excluded).")
    g.add_argument("--aspect-min-short", type=float, default=16.0,
                   help="short-side floor (mm); the aspect is clamped down "
                        "rather than resampled when it would go below.")
    g.add_argument("--net-density-k", type=float, default=0.0010,
                   help="net count lambda = k * (side / ref_pitch)^2")
    g.add_argument("--net-ref-pitch", type=float, default=0.45)
    g.add_argument("--min-nets", type=int, default=2,
                   help="floor on nets/board (Poisson is clamped up to this).")
    g.add_argument("--max-nets", type=int, default=80)
    g.add_argument("--rail-prob", type=float, default=0.10)
    g.add_argument("--rail-median", type=float, default=13.0)
    g.add_argument("--rail-sigma", type=float, default=0.5)
    g.add_argument("--rail-min", type=int, default=8)
    g.add_argument("--rail-max", type=int, default=32)
    g.add_argument("--bulk-base", type=int, default=2)
    g.add_argument("--bulk-lambda", type=float, default=0.5)
    g.add_argument("--uni-clearance-min", type=float, default=0.10)
    g.add_argument("--uni-clearance-max", type=float, default=0.40)
    g.add_argument("--uni-clearance-step", type=float, default=0.05)
    g.add_argument("--uni-width-factor-min", type=float, default=1.0)
    g.add_argument("--uni-width-factor-max", type=float, default=1.6)
    g.add_argument("--uni-pad-pitch-mult-min", type=float, default=2.0)
    g.add_argument("--uni-pad-pitch-mult-max", type=float, default=3.0)
    g.add_argument("--uni-via-drill-mult-min", type=float, default=1.5)
    g.add_argument("--uni-via-drill-mult-max", type=float, default=2.5)
    g.add_argument("--uni-via-dia-mult", type=float, default=1.8)
    args = p.parse_args()

    if args.mode == "d2b":
        _run_d2b(args)
        return

    if args.nets_min > args.nets_max or args.nets_min < 1:
        raise SystemExit(f"invalid --nets-min/--nets-max: "
                         f"{args.nets_min}..{args.nets_max}")
    if args.pads_per_net_min > args.pads_per_net_max or args.pads_per_net_min < 1:
        raise SystemExit(f"invalid --pads-per-net-min/--pads-per-net-max: "
                         f"{args.pads_per_net_min}..{args.pads_per_net_max}")

    fixed_pads_per_net: list[int] | None = None
    if args.fixed_pads_per_net is not None:
        fixed_pads_per_net = [int(x) for x in args.fixed_pads_per_net.split(",")]
        if any(p < 1 for p in fixed_pads_per_net) or len(fixed_pads_per_net) < 1:
            raise SystemExit(f"invalid --fixed-pads-per-net: {fixed_pads_per_net}")

    pads_per_net_weights: list[float] | None = None
    if args.pads_per_net_weights is not None:
        pads_per_net_weights = [float(x) for x in
                                args.pads_per_net_weights.split(",")]
        expected = args.pads_per_net_max - args.pads_per_net_min + 1
        if len(pads_per_net_weights) != expected:
            raise SystemExit(
                f"--pads-per-net-weights has {len(pads_per_net_weights)} "
                f"values but range {args.pads_per_net_min}..{args.pads_per_net_max} "
                f"needs {expected}"
            )
        if any(w < 0 for w in pads_per_net_weights):
            raise SystemExit("--pads-per-net-weights must be non-negative")
        if sum(pads_per_net_weights) <= 0:
            raise SystemExit("--pads-per-net-weights must sum to > 0")

    if args.pads_per_net_zipf is not None:
        if pads_per_net_weights is not None:
            raise SystemExit("--pads-per-net-zipf and --pads-per-net-weights "
                             "are mutually exclusive")
        tail_from = tail_mass = None
        if args.pads_per_net_zipf_tail is not None:
            try:
                _f, _m = args.pads_per_net_zipf_tail.split(":")
                tail_from, tail_mass = int(_f), float(_m)
            except ValueError:
                raise SystemExit("--pads-per-net-zipf-tail must be 'FROM:MASS', "
                                 f"got {args.pads_per_net_zipf_tail!r}")
        pads_per_net_weights = _zipf_weights(
            args.pads_per_net_min, args.pads_per_net_max,
            args.pads_per_net_zipf, tail_from, tail_mass,
        )
    elif args.pads_per_net_zipf_tail is not None:
        raise SystemExit("--pads-per-net-zipf-tail requires --pads-per-net-zipf")
    if not 0.0 <= args.net_locality <= 1.0:
        raise SystemExit(f"--net-locality must be in [0, 1], got {args.net_locality}")

    _eff_sep = (args.min_sep if args.min_sep is not None
                else _compute_min_sep(args.min_sep_formula, args.pad_size,
                                      args.clearance, args.trace_width))
    _need_sep = _min_sep_for_clearance(args.pad_size, args.clearance)
    if _eff_sep < _need_sep - 1e-9:
        print(f"  WARNING: min_sep {_eff_sep:.3f} < {_need_sep:.3f} required for "
              f"pad_size {args.pad_size} at clearance {args.clearance}. SMD pads are "
              f"square roundrect, so a 45-degree pair can sit {_eff_sep:.3f} mm apart "
              f"center-to-center yet violate clearance — the BARE board will fail DRC. "
              f"Raise --min-sep to {_need_sep:.3f}+ or shrink --pad-size.")

    bs_min = args.board_size_min
    bs_max = args.board_size_max
    if (bs_min is None) != (bs_max is None):
        raise SystemExit("--board-size-min and --board-size-max must be set together")
    randomize_size = bs_min is not None
    if randomize_size:
        if bs_min <= 0 or bs_max < bs_min:
            raise SystemExit(f"invalid --board-size-min/max: {bs_min}..{bs_max}")
        if args.mode != "grid":
            raise SystemExit("--board-size-min/max requires --mode grid")

    ps_min, ps_max = args.pad_size_min, args.pad_size_max
    if (ps_min is None) != (ps_max is None):
        raise SystemExit("--pad-size-min and --pad-size-max must be set together")
    randomize_pad = ps_min is not None
    if randomize_pad:
        if ps_min <= 0 or ps_max < ps_min:
            raise SystemExit(f"invalid --pad-size-min/max: {ps_min}..{ps_max}")
        if not randomize_size:
            raise SystemExit("--pad-size-min/max requires --board-size-min/max")
        print(f"  pad size sampled per board in [{ps_min}, {ps_max}] mm; "
              f"min_sep floored per board at "
              f"{_min_sep_for_clearance(ps_min, args.clearance):.3f}"
              f"..{_min_sep_for_clearance(ps_max, args.clearance):.3f}")

    cfg: BoardConfig | None
    cfg_factory = None
    if args.mode == "grid":
        if randomize_size:
            cfg = None
            def _factory(rng: random.Random, *, total_pads: int | None = None,
                         _lo=bs_min, _hi=bs_max,
                         _ps=args.pad_size, _c=args.clearance,
                         _w=args.trace_width, _pf=args.pitch_formula,
                         _ms=args.min_sep, _msf=args.min_sep_formula,
                         _cf=args.central_frac,
                         _pmin=ps_min, _pmax=ps_max) -> BoardConfig:
                bw = rng.uniform(_lo, _hi)
                bh = rng.uniform(_lo, _hi)
                ps, ms = _ps, _ms
                if _pmin is not None:
                    ps = rng.uniform(_pmin, _pmax)
                    # Larger pads need a wider EUCLIDEAN separation to keep the
                    # 45-degree corner-arc gap legal — see _min_sep_for_clearance.
                    floor = _min_sep_for_clearance(ps, _c)
                    ms = floor if ms is None else max(ms, floor)
                if total_pads:
                    eff_ms = ms if ms is not None else _compute_min_sep(
                        _msf, ps, _c, _w)
                    # Pads only go in the central fraction, so that is the area
                    # that has to clear the RSA bound. Grow both sides by the
                    # same factor to keep the sampled aspect ratio.
                    need = _min_area_for_pads(total_pads, eff_ms)
                    have = (bw * _cf) * (bh * _cf)
                    if have < need:
                        k = math.sqrt(need / have)
                        bw *= k
                        bh *= k
                return _make_config_grid_rect(
                    bw, bh, ps, _c, _w, _pf, ms, min_sep_formula=_msf,
                )
            cfg_factory = _factory
        else:
            cfg = _make_config_grid(
                args.board_size, args.pad_size,
                args.clearance, args.trace_width,
                args.pitch_formula, args.min_sep,
                min_sep_formula=args.min_sep_formula,
            )
    else:
        cfg = _make_config_legacy(args.pad_size, args.min_sep,
                                  min_sep_formula=args.min_sep_formula)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[3] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} boards into {out_dir}")
    if randomize_size:
        # Probe-build a representative cfg for the log only.
        probe = _make_config_grid_rect(
            (bs_min + bs_max) / 2, (bs_min + bs_max) / 2,
            args.pad_size, args.clearance, args.trace_width,
            args.pitch_formula, args.min_sep,
            min_sep_formula=args.min_sep_formula,
        )
        print(f"  mode={args.mode}, board=Uniform({bs_min}..{bs_max}) mm "
              f"x Uniform({bs_min}..{bs_max}) mm (independent), "
              f"central {args.central_frac*100:.0f}%")
        print(f"  clearance={probe.clearance}  trace_width={probe.trace_width}  "
              f"pad_size={probe.pad_size}  via={args.via_dia}/{args.via_drill}")
        print(f"  grid spacing={probe.grid_spacing} mm (per-axis cell count varies "
              f"per board)")
        print(f"  min center-to-center = {probe.min_sep:.4f} mm "
              f"(formula={args.min_sep_formula})")
    else:
        assert cfg is not None
        print(f"  mode={args.mode}, board={cfg.board_w} x {cfg.board_h} mm, "
              f"central {args.central_frac*100:.0f}%")
        print(f"  clearance={cfg.clearance}  trace_width={cfg.trace_width}  "
              f"pad_size={cfg.pad_size}  via={args.via_dia}/{args.via_drill}")
        if cfg.grid_spacing is not None:
            print(f"  grid={cfg.grid_count_x} x {cfg.grid_count_y} cells, "
                  f"spacing={cfg.grid_spacing} mm, pads snap to cell centers")
        print(f"  min center-to-center = {cfg.min_sep:.4f} mm "
              f"(formula={args.min_sep_formula})")
    if args.thru_hole_prob > 0.0:
        print(f"  thru-hole pad prob = {args.thru_hole_prob:.2f} per pad "
              f"(drill = via_drill = {args.via_drill} mm)")
    if fixed_pads_per_net is not None:
        print(f"  FIXED structure: {len(fixed_pads_per_net)} nets, "
              f"pads_per_net={fixed_pads_per_net}, "
              f"total pads={sum(fixed_pads_per_net)}")
    elif pads_per_net_weights is not None:
        dist = ", ".join(
            f"{v}:{w/sum(pads_per_net_weights)*100:.0f}%"
            for v, w in zip(range(args.pads_per_net_min,
                                  args.pads_per_net_max + 1),
                            pads_per_net_weights)
        )
        print(f"  nets per board: {args.nets_min}..{args.nets_max}  "
              f"pads per net (weighted): {dist}")
    else:
        print(f"  nets per board: {args.nets_min}..{args.nets_max}  "
              f"pads per net (uniform): {args.pads_per_net_min}..{args.pads_per_net_max}")
    if args.net_locality > 0.0:
        print(f"  net locality = {args.net_locality:.2f} "
              f"(0 = spatially random membership, 1 = nearest-neighbour)"
              + (f", fading to 0 at {args.net_locality_decay} pads"
                 if args.net_locality_decay else ""))
    if args.num_layers == 2:
        print(f"  seed={args.seed}, 2 copper layers, pad layer = random(F.Cu | B.Cu)")
    else:
        print(f"  seed={args.seed}, single-sided (all pads on F.Cu; stackup F.Cu+B.Cu "
              f"since KiCad rejects odd copper counts)")

    width = max(5, len(str(args.start_index + args.n - 1)))
    report_every = max(1, args.n // 20)
    for i in range(args.n):
        idx = args.start_index + i
        board_seed = (args.seed * 1_000_003 + idx if args.seed_mode == "legacy"
                      else args.seed + idx)
        text = generate_one(
            seed=board_seed,
            cfg=cfg,
            num_layers=args.num_layers,
            central_frac=args.central_frac,
            nets_min=args.nets_min,
            nets_max=args.nets_max,
            pads_per_net_min=args.pads_per_net_min,
            pads_per_net_max=args.pads_per_net_max,
            pads_per_net_weights=pads_per_net_weights,
            fixed_pads_per_net=fixed_pads_per_net,
            via_dia=args.via_dia,
            via_drill=args.via_drill,
            thru_hole_prob=args.thru_hole_prob,
            net_locality=args.net_locality,
            net_locality_decay=args.net_locality_decay,
            size_board_for_pads=args.size_board_for_pads,
            cfg_factory=cfg_factory,
        )
        (out_dir / f"board_{idx:0{width}d}.kicad_pcb").write_text(text)
        if (i + 1) % report_every == 0 or i + 1 == args.n:
            print(f"  {i+1}/{args.n}")

    print("Done.")


if __name__ == "__main__":
    main()
