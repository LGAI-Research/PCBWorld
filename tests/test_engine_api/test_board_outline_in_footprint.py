"""Outline shapes owned by a footprint (``fp_line`` on Edge.Cuts) are part of the board outline (needs C++).

KiCad's outline builder, bbox and the PNS world include footprint-owned Edge.Cuts shapes; five PCBench boards
(0107, 0357, 0565, 0621, 0643) draw their whole outline inside a footprint and two more (0301, 0502) draw part of it
there. Before the fix the router's outline getters enumerated ``BOARD::Drawings()`` only, so those boards loaded
with an empty outline and ``parse_pcb_file`` refused them ("no Edge.Cuts outline segments found").
Fixture: corner_edge_board with its 6 Edge.Cuts segments moved into a ``Frame:Outline`` footprint at (10, 5) as
local-coordinate ``fp_line``s — the absolute geometry is unchanged.
"""
import gc

import pytest

krl = pytest.importorskip("kicad_rl_router")

from tests.test_engine_api.conftest import FIXTURES_DIR

BOARD = FIXTURES_DIR / "outline_in_footprint_board.kicad_pcb"
REFERENCE = FIXTURES_DIR / "corner_edge_board.kicad_pcb"


def _edges(path):
    router = krl.RLRouter(str(path))
    edges = sorted((round(e.x1_mm, 4), round(e.y1_mm, 4), round(e.x2_mm, 4), round(e.y2_mm, 4)) for e in router.get_board_outline())
    del router; gc.collect()
    return edges


def test_footprint_owned_edge_cuts_form_the_outline():
    assert BOARD.exists() and REFERENCE.exists()
    edges = _edges(BOARD)
    assert len(edges) == 6, edges
    assert edges == _edges(REFERENCE)          # same absolute geometry as the board-level original


def test_outline_shapes_and_bbox_include_footprint_graphics():
    router = krl.RLRouter(str(BOARD))
    shapes = router.get_board_outline_shapes()
    assert len(shapes) == 6 and all(s.kind == 0 for s in shapes)
    bb = router.get_board_bbox()
    assert (round(bb.width_mm), round(bb.height_mm)) == (40, 40)
    del router; gc.collect()


def test_env_parser_accepts_the_board():
    """The Python load gate used to raise ValueError('no Edge.Cuts outline segments found') on such boards."""
    from pcb_world.engine import KiCadEngine
    from pcb_world.engine.pcb_file_parser import parse_pcb_file
    eng = KiCadEngine(str(BOARD))
    try:
        parsed = parse_pcb_file(str(BOARD), eng)
        assert len(parsed["board_edges"]) == 6
    finally:
        eng.close()
