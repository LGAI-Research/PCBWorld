"""Verifies rule-area keepout acts as a hard obstacle in the PNS router.

Board: tests/fixtures/simple_keepout_board.kicad_pcb
  P1(0,0) --- NET1 --- P2(4,0), with a rule-area keepout rect between them
  [1.5,-1]~[2.5,1] (tracks/vias not_allowed).

Keepout is not project-added logic — it's inherited from upstream KiCad
(``syncZone`` -> ``IsKeepout``). No dataset exercises it, so this test locks
it in as a regression guard: the presence of the keepout alone must turn a
straight-through path into a detour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_MODULE_DIR = PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"
BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_keepout_board.kicad_pcb"

sys.path.insert(0, str(RL_MODULE_DIR))

from tests.helpers.geometry_helpers import (  # noqa: E402
    Rect,
    Segment,
    segment_rect_intersect,
)

KEEPOUT_RECT: Rect = ((1.5, -1.0), (2.5, 1.0))
START = (0.0, 0.0)
END = (4.0, 0.0)


def _skip_if_unavailable() -> None:
    if not BOARD_PATH.exists():
        pytest.skip(f"Board not found: {BOARD_PATH}")
    try:
        import kicad_rl_router  # noqa: F401
    except ImportError:
        pytest.skip("kicad_rl_router not available")


def _route(board_path: str) -> list[Segment]:
    """Routes P1->P2 and returns the resulting segment list. RLRouter is only
    kept alive within this function's scope (it's a process-wide singleton,
    so two instances must never coexist)."""
    import kicad_rl_router as krl

    r = krl.RLRouter(board_path)
    r.set_routing_mode(krl.MODE_WALKAROUND)
    assert r.start_route(START[0], START[1], 0), "start_route failed"
    r.move(END[0], END[1])
    assert r.fix_route(END[0], END[1]), "fix_route failed"
    return [((t.x1_mm, t.y1_mm), (t.x2_mm, t.y2_mm)) for t in r.get_tracks()]


def _control_board(tmp_path: Path) -> str:
    """Control board with just the keepout zone block removed (same geometry)."""
    text = BOARD_PATH.read_text()
    i0 = text.index("  (zone")
    i1 = text.index("  (gr_rect")
    ctrl = tmp_path / "control_no_keepout.kicad_pcb"
    ctrl.write_text(text[:i0] + text[i1:])
    return str(ctrl)


def _crossings(segments: list[Segment]) -> list[Segment]:
    return [s for s in segments if segment_rect_intersect(s, KEEPOUT_RECT)]


def test_keepout_forces_detour() -> None:
    """Keepout board: the routed result must not cross the keepout rect."""
    _skip_if_unavailable()
    segs = _route(str(BOARD_PATH))
    assert len(segs) > 0, "no track was created"
    crossings = _crossings(segs)
    assert not crossings, f"track crosses the keepout: {crossings}"


def test_control_board_crosses_without_keepout(tmp_path: Path) -> None:
    """Control board (zone removed): with identical geometry, the straight
    path must cross the rect.

    This is a positive control proving the detour above is caused by the
    keepout."""
    _skip_if_unavailable()
    segs = _route(_control_board(tmp_path))
    assert _crossings(segs), (
        f"control board did not cross the keepout rect (tracks: {segs}). "
        f"positive control failed -> the keepout test would be meaningless"
    )


def test_get_keepouts_returns_polygon() -> None:
    """get_keepouts() must expose the rule-area keepout as a polygon + flags."""
    _skip_if_unavailable()
    import kicad_rl_router as krl

    r = krl.RLRouter(str(BOARD_PATH))
    zones = r.get_keepouts()
    assert len(zones) == 1, f"expected 1 keepout zone, got: {len(zones)}"

    z = zones[0]
    assert z.name == "KEEPOUT1"
    assert z.layer == 0  # F_Cu (board layer)
    assert z.keepout_tracks is True
    assert z.keepout_vias is True
    assert z.keepout_pads is False
    pts = {(round(x, 3), round(y, 3)) for x, y in z.pts}
    assert pts == {(1.5, -1.0), (2.5, -1.0), (2.5, 1.0), (1.5, 1.0)}, (
        f"keepout outline mismatch: {pts}"
    )
