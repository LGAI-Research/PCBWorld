"""Regression test: PNS router honors netclass clearance without a .kicad_pro.

Background
----------
KiCad's PNS router historically read its routing clearance from
``bds.m_MinClearance`` (a board-level design rule normally loaded from the
project file .kicad_pro). When only a .kicad_pcb exists, that field defaults
to 0, causing the router to shove/walkaround obstacles with zero clearance —
producing DRC violations (clearance/short) right next to pads even though the
net class specifies a non-zero clearance.

The patch in ``engine/kicad-patches/rl/pns_rl_router.cpp`` fixes this by:

  1. Promoting the default net class's clearance into the router's Sizes().
  2. Creating and attaching a ``DRC_ENGINE`` to
     ``BOARD_DESIGN_SETTINGS::m_DRCEngine`` at router init, so the PNS rule
     resolver's ``QueryConstraint`` path returns non-zero clearance instead
     of falling through to the null-engine branch.

This test pins that behaviour: a classic X-crossing between two nets must
route cleanly (no clearance/short violations) under SHOVE and WALKAROUND
modes when only the .kicad_pcb declares the netclass clearance.

Test scenario
-------------
Board: 100x100 mm, 2 nets, 4 pads at corners.
  Net 1 pads: (5, 5) and (95, 95)   — diagonal SW↔NE
  Net 2 pads: (5, 95) and (95, 5)   — diagonal NW↔SE
If both routed straight they cross at the center. The router must detour one
of them. With the patch, the detour respects the 0.05 mm netclass clearance.
"""
from __future__ import annotations

import random
import uuid
from pathlib import Path

import pytest

from pcb_world.engine import KiCadEngine
from tests.helpers.pro_sidecar import materialize_pro_pair

BOARD_SIZE = 100.0
PAD_SIZE = 1.0
# Defaults for the single-config scenario; the parametrized PNS/DRC-agreement
# test overrides these per-case.
CLEARANCE = 0.05
TRACE_WIDTH = 0.05

_HEADER = '''(kicad_pcb
  (version 20241229)
  (generator "synthetic_board_generator")
  (generator_version "9.0.5")
  (general (thickness 1.6) (legacy_teardrops no))
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
  (setup (pad_to_mask_clearance 0) (allow_soldermask_bridges_in_footprints no))
'''


def _u(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _render_board(pads: list[tuple[float, float, int]],
                  clearance: float = CLEARANCE,
                  trace_width: float = TRACE_WIDTH) -> str:
    rng = random.Random(0)
    nets = sorted({p[2] for p in pads})
    out = [_HEADER, ""]
    out.append('  (net 0 "")')
    for n in nets:
        out.append(f'  (net {n} "NET{n}")')
    out.append("")
    out.append('  (net_class "Default" "Default net class"')
    out.append(f"    (clearance {clearance})")
    out.append(f"    (trace_width {trace_width})")
    out.append("    (via_dia 0.6) (via_drill 0.3) (uvia_dia 0.3) (uvia_drill 0.1))")
    out.append("")
    for i, (x, y, n) in enumerate(pads, start=1):
        fp_u, pad_u = _u(rng), _u(rng)
        out.append('  (footprint "SamplePad:FCu"')
        out.append('    (layer "F.Cu")')
        out.append(f"    (at {x:.6f} {y:.6f})")
        out.append(f'    (uuid "{fp_u}")')
        out.append(f'    (property "Reference" "P{i}"')
        out.append('      (at 0 -1) (layer "F.SilkS")')
        out.append("      (effects (font (size 0.6 0.6) (thickness 0.1))))")
        out.append(f'    (property "Value" "Pad{i}"')
        out.append('      (at 0 1) (layer "F.Fab")')
        out.append("      (effects (font (size 0.6 0.6) (thickness 0.1))))")
        out.append('    (pad "1" smd roundrect')
        out.append('      (at 0 0)')
        out.append(f"      (size {PAD_SIZE} {PAD_SIZE})")
        out.append('      (layers "F.Cu" "F.Paste" "F.Mask")')
        out.append('      (roundrect_rratio 0.25)')
        out.append(f'      (net {n} "NET{n}")')
        out.append(f'      (uuid "{pad_u}")')
        out.append('    )')
        out.append('  )')
        out.append('')
    out.append('  (gr_rect (start 0.0 0.0) '
               f'(end {BOARD_SIZE} {BOARD_SIZE}) '
               '(stroke (width 0.15) (type solid)) (fill none) (layer "Edge.Cuts") '
               f'(uuid "{_u(rng)}"))')
    out.append(')')
    return "\n".join(out)


def _xcross_board(tmp_path: Path,
                  clearance: float = CLEARANCE,
                  trace_width: float = TRACE_WIDTH,
                  name: str = "xcross") -> str:
    pads = [
        (5.0, 5.0, 1),    (95.0, 95.0, 1),   # Net 1 diagonal SW → NE
        (5.0, 95.0, 2),   (95.0, 5.0, 2),    # Net 2 diagonal NW → SE
    ]
    bp = tmp_path / f"{name}.kicad_pcb"
    bp.write_text(_render_board(pads, clearance=clearance,
                                trace_width=trace_width))
    # The rendered board declares its rules via an in-pcb net_class block —
    # the engine load contract wants them in a companion .kicad_pro, so
    # round-trip once (what the retired upgrade cache used to do).
    return materialize_pro_pair(bp)


def _route_xcross(board_path: str, mode: int) -> dict:
    e = KiCadEngine(board_path)
    try:
        def prep() -> None:
            e.set_routing_mode(mode)
            e.set_track_width(0)
            e.reset_via_mode()
            e.clear_drc_cache()

        prep()
        ok1 = e.start_route(5.0, 5.0, 1)      # PNS layer 1 = F.Cu
        e.move(95.0, 95.0)
        fin1 = e.finish()

        prep()
        ok2 = e.start_route(5.0, 95.0, 1)
        e.move(95.0, 5.0)
        fin2 = e.finish()

        violations = e.run_drc()
        clearance_shorts = [
            v for v in violations
            if "Clearance" in str(v) or "shorting" in str(v)
        ]
        tracks = e.get_tracks()
        return {
            "net1_ok": ok1 and fin1,
            "net2_ok": ok2 and fin2,
            "tracks": len(tracks),
            "track_widths": [t.width_mm for t in tracks],
            "clearance_shorts": len(clearance_shorts),
            "violations": clearance_shorts,
        }
    finally:
        e.close()


def _route_xcross_with_params(board_path: str, mode: int) -> dict:
    """Same as _route_xcross but uses the board's netclass defaults."""
    return _route_xcross(board_path, mode)


@pytest.mark.parametrize(
    "mode_id,mode_name",
    [(1, "SHOVE"), (2, "WALKAROUND")],
)
@pytest.mark.parametrize(
    "clearance,trace_width",
    [
        pytest.param(0.05, 0.05, id="c=0.05_w=0.05"),
        pytest.param(0.10, 0.05, id="c=0.10_w=0.05"),
        pytest.param(0.05, 0.10, id="c=0.05_w=0.10"),
        pytest.param(0.20, 0.20, id="c=0.20_w=0.20"),
    ],
)
def test_pns_and_drc_agree_on_clearance(tmp_path: Path, mode_id: int,
                                        mode_name: str, clearance: float,
                                        trace_width: float) -> None:
    """PNS router and DRC must use the same clearance/width for ALL params.

    Parametrized over {SHOVE, WALKAROUND} x (4 c/w combinations).  For every
    case, after routing the X-crossing:

      (1) run_drc() reports 0 clearance/short violations — DRC agrees the
          routed geometry meets the declared netclass clearance.
      (2) Every placed track's width equals the declared netclass trace_width
          — PNS used the same width DRC will evaluate.

    The goal is to catch any future drift between the router's internal
    clearance (from m_MinClearance / DRC_ENGINE) and the DRC checker's
    clearance (always from netclass/design rules). A mismatch manifests as
    phantom "Clearance violation" errors along walkaround detours.
    """
    board_path = _xcross_board(
        tmp_path, clearance=clearance, trace_width=trace_width,
        name=f"x_c{clearance}_w{trace_width}",
    )
    result = _route_xcross(board_path, mode_id)

    assert result["net1_ok"] and result["net2_ok"], (
        f"{mode_name} c={clearance} w={trace_width}: one or both nets failed "
        "to finish routing"
    )
    assert result["clearance_shorts"] == 0, (
        f"{mode_name} c={clearance} w={trace_width}: DRC reports "
        f"{result['clearance_shorts']} clearance/short violations — PNS and "
        f"DRC disagreed on clearance.\n"
        + "\n".join(f"  {v}" for v in result["violations"])
    )

    # Sanity-check every placed track uses the netclass trace_width (within a
    # tiny tolerance for internal-unit rounding). If PNS was using a different
    # track width than DRC assumes, widths would diverge from the netclass.
    assert result["tracks"] > 0, "no tracks after routing"
    for w in result["track_widths"]:
        assert abs(w - trace_width) < 1e-6, (
            f"track width {w} != netclass trace_width {trace_width}"
        )


@pytest.mark.parametrize(
    "mode_id,mode_name",
    [(1, "SHOVE"), (2, "WALKAROUND")],
)
def test_xcross_routes_drc_clean(tmp_path: Path, mode_id: int,
                                 mode_name: str) -> None:
    """SHOVE/WALKAROUND must route the X-crossing without clearance/short errors.

    Fails on the pre-patch router (which used bds.m_MinClearance = 0) — the
    walkaround/shove path would hug pad edges producing ~3 clearance/short
    violations. Passes after the netclass-clearance + DRC-engine-attach patch
    in ``engine/kicad-patches/rl/pns_rl_router.cpp``.
    """
    board_path = _xcross_board(tmp_path)
    result = _route_xcross(board_path, mode_id)

    assert result["net1_ok"], (
        f"{mode_name}: Net 1 diagonal route did not finish"
    )
    assert result["net2_ok"], (
        f"{mode_name}: Net 2 diagonal route did not finish"
    )
    assert result["clearance_shorts"] == 0, (
        f"{mode_name}: expected 0 clearance/short violations (netclass "
        f"clearance = {CLEARANCE} mm). Got {result['clearance_shorts']}:\n"
        + "\n".join(f"  {v}" for v in result["violations"])
    )
