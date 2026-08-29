"""Engine-level tests for the load-time outline-simplify pass.

Covers the C++ read/replace surface (``get_graphic_shapes`` /
``replace_graphic_shapes``), the ``KiCadEngine(simplify_outline=True)`` flag,
and the save→reload idempotency loop on a micro-segment fixture
(``corner_staircase_edge.kicad_pcb``: 720 Edge.Cuts gr_lines).
"""

import pytest

from tests.test_engine_api.conftest import FIXTURES_DIR

STAIRCASE = FIXTURES_DIR / "corner_staircase_edge.kicad_pcb"


@pytest.fixture
def staircase_path():
    if not STAIRCASE.exists():
        pytest.skip(f"Board not found: {STAIRCASE}")
    return str(STAIRCASE)


def test_graphic_shapes_read_surface(staircase_path):
    krl = pytest.importorskip("kicad_rl_router")
    r = krl.RLRouter(staircase_path)
    shapes = r.get_graphic_shapes(krl.LAYER_EDGE_CUTS)
    assert len(shapes) == 720
    assert all(s.kind == 0 for s in shapes)
    assert all(s.width_nm > 0 for s in shapes)
    # exact-int coordinates, distinct endpoints
    s0 = shapes[0]
    assert isinstance(s0.x1_nm, int)
    assert (s0.x1_nm, s0.y1_nm) != (s0.x2_nm, s0.y2_nm)


def test_replace_bad_index_raises(staircase_path):
    krl = pytest.importorskip("kicad_rl_router")
    r = krl.RLRouter(staircase_path)
    with pytest.raises(RuntimeError, match="out of range"):
        r.replace_graphic_shapes(krl.LAYER_EDGE_CUTS, [10**6], [], [])
    with pytest.raises(RuntimeError, match="target layer"):
        r.replace_graphic_shapes(krl.LAYER_MARGIN,
                                 [r.get_graphic_shapes(krl.LAYER_EDGE_CUTS)[0].index],
                                 [], [])


def test_engine_flag_simplifies_and_roundtrips(staircase_path, tmp_path):
    krl = pytest.importorskip("kicad_rl_router")
    from pcb_world.engine.kicad_engine import KiCadEngine

    eng = KiCadEngine(staircase_path, simplify_outline=True)
    try:
        rep = eng.outline_simplify_report
        assert rep is not None and rep.changed
        assert rep.n_input_segments == 720
        outline = eng.get_board_outline()
        assert len(outline) < 40  # was 720 straight segments
        # engine remains fully usable: connectivity + save
        eng.build_connectivity()
        out = tmp_path / "staircase_simplified.kicad_pcb"
        eng.save(str(out))
        assert out.exists() and out.with_suffix(".kicad_pro").exists()
    finally:
        eng.close()

    # reload the converted board with the pass enabled → no-op (idempotent)
    eng2 = KiCadEngine(str(out), simplify_outline=True)
    try:
        rep2 = eng2.outline_simplify_report
        assert rep2 is not None and not rep2.changed
        # raw binding surface — reaches the router over either transport
        shapes = eng2._r.get_graphic_shapes(krl.LAYER_EDGE_CUTS)
        assert len(shapes) < 40
    finally:
        eng2.close()


def test_engine_flag_off_is_untouched(staircase_path):
    from pcb_world.engine.kicad_engine import KiCadEngine

    eng = KiCadEngine(staircase_path)
    try:
        assert eng.outline_simplify_report is None
        assert len(eng.get_board_outline()) == 720
    finally:
        eng.close()
