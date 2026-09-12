"""DRC smoke tests on grid-snapped synthetic boards.

Verifies that KiCad's DRC correctly flags violations we intentionally induce:
  (1) a clean single-trace route between two pads of the same net -> 0 violations
  (2) two parallel traces of different nets spaced *less than* required
      clearance apart -> non-zero violations
  (3) a trace of net A that runs through a pad of net B -> non-zero violations
      (net-short / copper-copper)

Uses the tight-pitch spec (c=w=0.05, grid=0.1 mm) matching
pcb_dataset_synthetic_5net_2pin_1layer/.
"""
from __future__ import annotations

import random
import uuid
from pathlib import Path

import pytest

from pcb_world.engine import KiCadEngine
from tests.helpers.pro_sidecar import materialize_pro_pair

BOARD_SIZE = 100.0
CLEARANCE = 0.05
TRACE_WIDTH = 0.05
GRID_SPACING = CLEARANCE + TRACE_WIDTH   # 0.1 mm
PAD_SIZE = 1.0


def _u(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


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


def _render_board(pads: list[tuple[float, float, int]]) -> str:
    """pads: list of (x_mm, y_mm, net_id).  net_id 1..N; also emits (net ...) headers."""
    rng = random.Random(42)
    net_ids = sorted({p[2] for p in pads})
    out = [_HEADER, ""]
    out.append('  (net 0 "")')
    for n in net_ids:
        out.append(f'  (net {n} "NET{n}")')
    out.append("")
    out.append('  (net_class "Default" "Default net class"')
    out.append(f"    (clearance {CLEARANCE})")
    out.append(f"    (trace_width {TRACE_WIDTH})")
    out.append("    (via_dia 0.6) (via_drill 0.3) (uvia_dia 0.3) (uvia_drill 0.1))")
    out.append("")
    for i, (x, y, net) in enumerate(pads, start=1):
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
        out.append(f'      (net {net} "NET{net}")')
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


def _prep(engine: KiCadEngine) -> None:
    engine.set_routing_mode(2)
    engine.set_track_width(0)
    engine.reset_via_mode()
    engine.clear_drc_cache()


def test_drc_clean_single_route(tmp_path: Path) -> None:
    """Single straight trace between two same-net pads: DRC should be clean."""
    pads = [(10.05, 50.05, 1), (90.05, 50.05, 1)]
    bp = tmp_path / "clean.kicad_pcb"
    bp.write_text(_render_board(pads))
    e = KiCadEngine(materialize_pro_pair(bp))
    try:
        _prep(e)
        assert e.start_route(pads[0][0], pads[0][1], 1)
        e.move(pads[1][0], pads[1][1])
        assert e.finish()
        violations = e.run_drc()
        assert len(violations) == 0, (
            f"expected 0 DRC violations on clean route, got {len(violations)}: "
            f"{violations}"
        )
    finally:
        e.close()


def _render_board_with_tracks(pads: list[tuple[float, float, int]],
                              segments: list[tuple[float, float, float, float, int]]
                              ) -> str:
    """Render a board with pads and literal copper (segment ...) track entries.

    Each segment: (x1, y1, x2, y2, net_id) on F.Cu.
    """
    board = _render_board(pads)
    # Inject segments right before the closing ")" of the kicad_pcb form.
    assert board.endswith(")")
    body = board[:-1].rstrip()
    rng = random.Random(7)
    seg_lines = []
    for (x1, y1, x2, y2, net) in segments:
        seg_lines.append("  (segment")
        seg_lines.append(f"    (start {x1:.6f} {y1:.6f})")
        seg_lines.append(f"    (end {x2:.6f} {y2:.6f})")
        seg_lines.append(f"    (width {TRACE_WIDTH})")
        seg_lines.append('    (layer "F.Cu")')
        seg_lines.append(f'    (net {net})')
        seg_lines.append(f'    (uuid "{_u(rng)}")')
        seg_lines.append("  )")
    return body + "\n" + "\n".join(seg_lines) + "\n)"


def test_drc_detects_too_close_parallel_tracks(tmp_path: Path) -> None:
    """Two parallel traces of different nets spaced < required pitch -> DRC flags it.

    Bypasses the router (which auto-shoves to avoid violations) and injects the
    track segments directly into the .kicad_pcb so DRC sees the raw geometry.
    Tracks run at y=50.05 and y=50.11 — edge-to-edge = 0.01 mm << clearance 0.05.
    """
    pads = [
        (10.05, 50.05, 1), (90.05, 50.05, 1),
        (10.05, 55.05, 2), (90.05, 55.05, 2),
    ]
    segments = [
        (10.05, 50.05, 90.05, 50.05, 1),   # Net 1 straight
        (10.05, 50.11, 90.05, 50.11, 2),   # Net 2 too close
    ]
    bp = tmp_path / "too_close_injected.kicad_pcb"
    bp.write_text(_render_board_with_tracks(pads, segments))
    e = KiCadEngine(materialize_pro_pair(bp))
    try:
        violations = e.run_drc()
        assert len(violations) > 0, (
            f"expected DRC violations for parallel tracks 0.06 mm center-to-center "
            f"(required pitch = {CLEARANCE + TRACE_WIDTH} mm), got 0"
        )
    finally:
        e.close()


def test_drc_detects_pad_overlap_different_nets(tmp_path: Path) -> None:
    """Two pads of different nets overlapping at the same spot -> violations."""
    pads = [
        (50.05, 50.05, 1),
        (50.05, 50.05, 2),  # same location, different net
    ]
    bp = tmp_path / "pad_overlap.kicad_pcb"
    bp.write_text(_render_board(pads))
    e = KiCadEngine(materialize_pro_pair(bp))
    try:
        violations = e.run_drc()
        assert len(violations) > 0, (
            "expected DRC violations for overlapping different-net pads, got 0"
        )
    finally:
        e.close()
