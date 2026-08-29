"""``get_board_outline_shapes`` + the 3 ``outline_obs`` modes (needs C++).

The primitive-preserving outline API returns Edge.Cuts entries typed as
segment (kind 0), arc (kind 1: p1/p2 endpoints + on-arc midpoint) or circle
(kind 2: p1 == p2 on the circle, mid = antipode). ``parse_pcb_file`` turns
them into the obs ``board_edges`` per ``outline_mode``:

  tess   — the C++ tessellation path (bit-stable),
  poly16 — Python re-tessellation at a fixed 16 segments per 90°,
  arc    — one entry per arc, mid carried through to obs ``boardlines``.

Fixture: crossover_board (modern pcb+pro pair, upgraded once from
crossover_legacy) has 12 straight Edge.Cuts segments + 4 ninety-degree
fillet arcs (tessellated: 60 segments). The legacy original stays in the
tree solely as the engine load-contract refusal specimen.
"""

import gc
import math

import pytest

krl = pytest.importorskip("kicad_rl_router")

from tests.test_engine_api.conftest import FIXTURES_DIR

BOARD = FIXTURES_DIR / "crossover_board.kicad_pcb"


@pytest.fixture(scope="module")
def outline_data():
    """One router pass: tessellated outline + typed shapes."""
    if not BOARD.exists():
        pytest.skip(f"Board not found: {BOARD}")
    router = krl.RLRouter(str(BOARD))
    tess = [((e.x1_mm, e.y1_mm), (e.x2_mm, e.y2_mm), e.width_mm)
            for e in router.get_board_outline()]
    shapes = [(s.kind, (s.x1_mm, s.y1_mm), (s.x2_mm, s.y2_mm),
               (s.x3_mm, s.y3_mm), s.width_mm)
              for s in router.get_board_outline_shapes()]
    del router
    gc.collect()
    return tess, shapes


class TestOutlineShapesBinding:
    def test_kinds_and_counts(self, outline_data):
        tess, shapes = outline_data
        kinds = [s[0] for s in shapes]
        assert kinds.count(0) == 12
        assert kinds.count(1) == 4
        assert len(tess) == 60  # 12 straight + 4 arcs x 12 segments at 0.005mm

    def test_arc_three_points_cocircular(self, outline_data):
        _, shapes = outline_data
        for kind, p1, p2, mid, _w in shapes:
            if kind != 1:
                continue
            (x1, y1), (x2, y2), (xm, ym) = p1, p2, mid
            d = 2.0 * (x1 * (ym - y2) + xm * (y2 - y1) + x2 * (y1 - ym))
            assert abs(d) > 1e-9  # not collinear
            s1, sm, s2 = (x1**2 + y1**2), (xm**2 + ym**2), (x2**2 + y2**2)
            cx = (s1 * (ym - y2) + sm * (y2 - y1) + s2 * (y1 - ym)) / d
            cy = (s1 * (x2 - xm) + sm * (x1 - x2) + s2 * (xm - x1)) / d
            radii = [math.hypot(x - cx, y - cy) for x, y in (p1, p2, mid)]
            assert max(radii) - min(radii) < 1e-6

    def test_arc_endpoints_match_tessellation(self, outline_data):
        """Each arc's endpoints appear as endpoints in the tessellated run."""
        tess, shapes = outline_data
        tess_pts = {p for seg in tess for p in seg[:2]}

        def _has(pt):
            return any(math.hypot(pt[0] - q[0], pt[1] - q[1]) < 1e-6
                       for q in tess_pts)

        for kind, p1, p2, _mid, _w in shapes:
            if kind == 1:
                assert _has(p1) and _has(p2)


class TestCircleOutline:
    """kind-2 full-circle path: geometry helpers + a real gr_circle board."""

    def test_extreme_points_full_circle(self):
        from pcb_world.engine.pcb_file_parser import _arc_extreme_points

        # p1 == p2 on the circle, mid = antipode (the kind-2 encoding).
        pts = _arc_extreme_points(15.0, 20.0, 5.0, 20.0, 15.0, 20.0)
        assert sorted(pts) == [(5.0, 20.0), (10.0, 15.0), (10.0, 25.0), (15.0, 20.0)]

    def test_poly16_quarter_fillet_keeps_16_segments(self):
        from pcb_world.engine.pcb_file_parser import _tessellate_arc_poly16

        # Exact 90-degree fillets at int-nm-rounded radii: the C++ path emits
        # 16 segments, and the mm-double recomputation must match it — the
        # int() epsilon guard keeps it from truncating to 15.
        for r in (2.54, 3.0, 5.0, 0.9999999):
            s = math.sin(math.pi / 4)
            pts = _tessellate_arc_poly16(r, 0.0, r * s, r * s, 0.0, r)
            assert len(pts) - 1 == 16, f"r={r}: {len(pts) - 1} segments"

    def test_board_edges_from_shapes_circle(self):
        from types import SimpleNamespace

        from pcb_world.engine.pcb_file_parser import _board_edges_from_shapes

        circle = SimpleNamespace(kind=2, x1_mm=15.0, y1_mm=20.0,
                                 x2_mm=15.0, y2_mm=20.0,
                                 x3_mm=5.0, y3_mm=20.0, width_mm=0.1)
        # arc mode: one entry, mid carried, bbox extras span the full circle.
        edges, bbox_pts = _board_edges_from_shapes([circle], "arc")
        assert len(edges) == 1 and edges[0].x3_mm == 5.0
        xs = [p[0] for p in bbox_pts] + [edges[0].x1_mm, edges[0].x2_mm]
        ys = [p[1] for p in bbox_pts] + [edges[0].y1_mm, edges[0].y2_mm]
        assert (min(xs), max(xs), min(ys), max(ys)) == (5.0, 15.0, 15.0, 25.0)
        # poly16 mode: 32-gon.
        edges, _ = _board_edges_from_shapes([circle], "poly16")
        assert len(edges) == 32

    def test_real_gr_circle_board(self, tmp_path):
        """Append a gr_circle to the fixture: kind-2 emission + bbox parity."""
        text = BOARD.read_text()
        gr = '  (gr_circle (center 100 60) (end 105 60) ' \
             '(stroke (width 0.1) (type solid)) (layer "Edge.Cuts"))\n'
        board = tmp_path / "with_circle.kicad_pcb"
        board.write_text(text[: text.rfind(")")] + gr + ")\n")

        from pcb_world.engine.pcb_file_parser import _board_edges_from_shapes

        router = krl.RLRouter(str(board))
        shapes = router.get_board_outline_shapes()
        circles = [s for s in shapes if s.kind == 2]
        assert len(circles) == 1
        c = circles[0]
        assert (c.x1_mm, c.y1_mm) == (c.x2_mm, c.y2_mm)  # p1 == p2
        assert (c.x1_mm + c.x3_mm, c.y1_mm + c.y3_mm) == (200.0, 120.0)  # antipode

        # bbox from arc-mode entries+extras == bbox of the tessellated run.
        tess = router.get_board_outline()
        tx = [e.x1_mm for e in tess] + [e.x2_mm for e in tess]
        ty = [e.y1_mm for e in tess] + [e.y2_mm for e in tess]
        edges, extras = _board_edges_from_shapes(shapes, "arc")
        ax = [e.x1_mm for e in edges] + [e.x2_mm for e in edges] + [p[0] for p in extras]
        ay = [e.y1_mm for e in edges] + [e.y2_mm for e in edges] + [p[1] for p in extras]
        assert min(ax) == pytest.approx(min(tx), abs=1e-3)
        assert max(ax) == pytest.approx(max(tx), abs=1e-3)
        assert min(ay) == pytest.approx(min(ty), abs=1e-3)
        assert max(ay) == pytest.approx(max(ty), abs=1e-3)
        del router
        gc.collect()


class TestOutlineObsModes:
    """parse_pcb_file / PCBWorld end-to-end per mode (sequential envs)."""

    def _reset(self, **kw):
        from pcb_world.core.env import PCBWorld

        env = PCBWorld(board_path=str(BOARD), **kw)
        try:
            obs, _ = env.reset()
        finally:
            env.close()
            del env
            gc.collect()
        return obs

    def test_mode_edge_counts_and_mid(self):
        bl_tess = self._reset(outline_obs="tess")["board_static"]["boardlines"]
        bl_poly = self._reset(outline_obs="poly16")["board_static"]["boardlines"]
        bl_arc = self._reset(outline_obs="arc")["board_static"]["boardlines"]

        assert len(bl_tess) == 60
        assert not any("mid" in e for e in bl_tess.values())
        # poly16: 90-degree arcs -> n = max(2, int(16 * 90 / 90)) = 16 each.
        assert len(bl_poly) == 12 + 4 * 16
        assert not any("mid" in e for e in bl_poly.values())
        # arc: one entry per primitive; exactly the 4 fillets carry mid.
        assert len(bl_arc) == 16
        assert sum(1 for e in bl_arc.values() if "mid" in e) == 4

    def test_arc_mode_indexed_twin_bit_identical(self):
        import numpy as np

        from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer

        obs_json = self._reset(outline_obs="arc", obs_format="json")
        obs_idx = self._reset(outline_obs="arc", obs_format="indexed")

        em = obs_idx["board_static"]["edge_mid"]
        assert (em >= 0).sum() == 4

        tok = BatchedStateTokenizer()
        wd = tok._walk_obs([obs_json])
        wi = tok._walk_obs([obs_idx])
        for f, (a, b) in enumerate(zip(wd["edge"], wi["edge"])):
            assert a.dtype == b.dtype and a.shape == b.shape, f
            np.testing.assert_array_equal(a, b, err_msg=f"edge field {f}")

    def test_default_mode_is_tess(self):
        obs = self._reset()
        assert len(obs["board_static"]["boardlines"]) == 60

    def test_invalid_mode_raises(self):
        from pcb_world.core.env import PCBWorld

        with pytest.raises(ValueError, match="outline_obs"):
            PCBWorld(board_path=str(BOARD), outline_obs="bogus")


class TestBoardBBoxIsOutlineOnly:
    """``get_board_bbox`` = Edge.Cuts-only bbox (decoration/routing-invariant).

    The bbox comes from the board outline alone. Merging every object (as
    ``GetBoundingBox`` does) pulls in silkscreen, annotations and tracks that
    sit outside the outline and inflates both obs normalization (norm_scale)
    and the wirelength scale. Pinned on a copy carrying silkscreen graffiti
    far from the outline: the bbox must stay at the outline size.
    """

    def test_bbox_matches_outline_and_ignores_offboard_silk(self, tmp_path):
        text = BOARD.read_text().rstrip()
        assert text.endswith(")")
        graffiti = (
            '\t(gr_line (start 500 500) (end 520 520) '
            '(stroke (width 0.1) (type solid)) (layer "F.SilkS") '
            '(uuid "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"))\n'
        )
        dest = tmp_path / BOARD.name
        dest.write_text(text[:-1] + graffiti + ")\n")

        router = krl.RLRouter(str(dest))
        try:
            bbox = router.get_board_bbox()
            xs = [c for e in router.get_board_outline()
                  for c in (e.x1_mm, e.x2_mm)]
            ys = [c for e in router.get_board_outline()
                  for c in (e.y1_mm, e.y2_mm)]
        finally:
            del router
            gc.collect()

        ow, oh = max(xs) - min(xs), max(ys) - min(ys)
        # Tolerance is the outline stroke width; merging the graffiti
        # (500..520mm) would move the bbox by hundreds of mm.
        assert abs(bbox.width_mm - ow) < 0.5, (bbox.width_mm, ow)
        assert abs(bbox.height_mm - oh) < 0.5, (bbox.height_mm, oh)
        assert bbox.x_mm + bbox.width_mm < 400   # graffiti excluded
