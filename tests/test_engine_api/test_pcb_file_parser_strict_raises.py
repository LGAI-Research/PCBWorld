"""Reverse tests for pcb_file_parser strict-raise cases.

An unmappable layer or missing outline is a hard ``ValueError`` rather than
a silent fallback (e.g. defaulting to layer=1, max_layer, or a
(0,0,100,100) bbox sentinel). These tests construct minimal mock board
state that would trigger each such case and assert that the parser
refuses it.

Five guarded paths:

1. ``_PadView`` SMD pad with unmappable PCB_LAYER_ID  (parse_pcb_file)
2. ``_TrackView`` with unmappable PCB_LAYER_ID         (direct)
3. ``_ViaView`` top_layer unmappable                   (direct)
4. ``_ViaView`` bottom_layer unmappable                (direct)
5. ``parse_pcb_file`` with empty Edge.Cuts outline     (parse_pcb_file)

The thru_hole pad ``human_layer = 0`` sentinel is intentionally NOT a
fallback (it is a legacy contract: "spans every copper layer") and remains
unguarded — see test_thru_hole_pad_uses_zero_sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from pcb_world.engine.layer_mapping import LayerMapping
from pcb_world.engine.pcb_file_parser import (
    _TrackView,
    _ViaView,
    parse_pcb_file,
)


# ---------------------------------------------------------------------------
# Lightweight mocks that mirror the engine's pad / track / via duck-types
# ---------------------------------------------------------------------------

@dataclass
class _MockPad:
    pad_type: str = "smd"
    layer: int = 0           # PCB_LAYER_ID (F_Cu = 0 by default)
    x_mm: float = 1.0
    y_mm: float = 2.0
    width_mm: float = 0.5
    height_mm: float = 0.5
    net_code: int = 1
    net_name: str = "NET1"
    shape: str = "rect"
    pad_name: str = "1"
    footprint_ref: str = "U1"


@dataclass
class _MockTrack:
    layer: int = 0
    x1_mm: float = 0.0
    y1_mm: float = 0.0
    x2_mm: float = 1.0
    y2_mm: float = 1.0
    width_mm: float = 0.2
    net_code: int = 1
    net_name: str = "NET1"


@dataclass
class _MockVia:
    top_layer: int = 0       # F_Cu
    bottom_layer: int = 2    # B_Cu (valid for 2-layer LayerMapping)
    x_mm: float = 5.0
    y_mm: float = 6.0
    diameter_mm: float = 0.6
    drill_mm: float = 0.3
    net_code: int = 1
    net_name: str = "NET1"


@dataclass
class _MockEdge:
    x1_mm: float = 0.0
    y1_mm: float = 0.0
    x2_mm: float = 10.0
    y2_mm: float = 10.0
    width_mm: float = 0.05


def _make_mock_engine(
    pads=None, tracks=None, vias=None, edges=None,
    copper_layers=2, net_count=1,
):
    """Build a minimal duck-typed engine for ``parse_pcb_file``."""
    layer_map = LayerMapping(copper_layers)
    return SimpleNamespace(
        layer_map=layer_map,
        get_board_meta=lambda: SimpleNamespace(
            net_count=net_count, copper_layers=copper_layers,
        ),
        get_pads=lambda: list(pads or []),
        get_tracks=lambda: list(tracks or []),
        get_vias=lambda: list(vias or []),
        get_ratsnest=lambda: [],
        get_track_count=lambda: len(tracks or []),
        get_unrouted_count=lambda: 0,
        get_board_outline=lambda: list(edges or []),
        get_net_names=lambda: {1: "NET1"},
    )


# ---------------------------------------------------------------------------
# 1. SMD pad on a non-copper PCB_LAYER_ID  (must raise, not default to Top)
# ---------------------------------------------------------------------------

class TestPadStrictRaise:
    """SMD pad whose layer is not in layer_map must raise — never default to Top."""

    @pytest.mark.parametrize("bad_layer", [1, 3, 5, 9, 31])  # F.Mask=1, In3.Cu=8 absent in 2-layer, etc.
    def test_smd_pad_unmappable_layer_raises(self, bad_layer):
        bad_pad = _MockPad(pad_type="smd", layer=bad_layer, net_name="VCC")
        engine = _make_mock_engine(
            pads=[bad_pad],
            edges=[_MockEdge()],   # provide outline so we don't trip rule (5)
        )
        with pytest.raises(ValueError, match="Pad parsing failed"):
            parse_pcb_file("dummy.kicad_pcb", engine)

    def test_error_message_includes_pad_context(self):
        bad_pad = _MockPad(
            pad_type="smd", layer=99, net_name="DDR_CK", x_mm=12.5, y_mm=7.25,
        )
        engine = _make_mock_engine(pads=[bad_pad], edges=[_MockEdge()])
        with pytest.raises(ValueError) as excinfo:
            parse_pcb_file("dummy.kicad_pcb", engine)
        msg = str(excinfo.value)
        assert "99" in msg              # the offending board_layer
        assert "smd" in msg              # pad_type for diagnostics
        assert "DDR_CK" in msg           # net name for diagnostics
        assert "12.5" in msg and "7.25" in msg  # location

    def test_thru_hole_pad_uses_zero_sentinel(self):
        """thru_hole pads bypass the layer lookup — sentinel 0 is not a fallback."""
        thru_pad = _MockPad(pad_type="thru_hole", layer=999)  # bogus layer, ignored
        engine = _make_mock_engine(pads=[thru_pad], edges=[_MockEdge()])
        out = parse_pcb_file("dummy.kicad_pcb", engine)
        assert out["parse_stats"]["thru_hole_pads"] == 1
        assert out["board_snapshot"].pads[0].layer == 0

    def test_smd_pad_on_copper_passes(self):
        """Sanity: a well-formed SMD pad on F_Cu must NOT raise."""
        good_pad = _MockPad(pad_type="smd", layer=0)  # F_Cu
        engine = _make_mock_engine(pads=[good_pad], edges=[_MockEdge()])
        out = parse_pcb_file("dummy.kicad_pcb", engine)
        assert out["board_snapshot"].pads[0].layer == 1  # human Top


# ---------------------------------------------------------------------------
# 2. Track on a non-copper PCB_LAYER_ID  (must raise, not default to Top)
# ---------------------------------------------------------------------------

class TestTrackStrictRaise:

    @pytest.mark.parametrize("bad_layer", [1, 3, 5, 9, 31])
    def test_track_unmappable_layer_raises(self, bad_layer):
        layer_map = LayerMapping(2)
        track = _MockTrack(layer=bad_layer, net_name="GND")
        with pytest.raises(ValueError, match="Track parsing failed"):
            _TrackView(track, layer_map)

    def test_error_message_includes_track_context(self):
        layer_map = LayerMapping(2)
        track = _MockTrack(
            layer=77, net_name="DATA0",
            x1_mm=1.0, y1_mm=2.0, x2_mm=3.0, y2_mm=4.0,
        )
        with pytest.raises(ValueError) as excinfo:
            _TrackView(track, layer_map)
        msg = str(excinfo.value)
        assert "77" in msg
        assert "DATA0" in msg
        assert "1.0" in msg and "4.0" in msg  # endpoints

    def test_track_on_copper_passes(self):
        layer_map = LayerMapping(2)
        track = _MockTrack(layer=2)  # B_Cu → human 2
        wrapped = _TrackView(track, layer_map)
        assert wrapped.layer == 2


# ---------------------------------------------------------------------------
# 3 + 4. Via top / bottom on unmappable layer  (must raise, not default to 1 / max_layer)
# ---------------------------------------------------------------------------

class TestViaStrictRaise:

    @pytest.mark.parametrize("bad_top", [1, 3, 5, 9])
    def test_via_top_layer_unmappable_raises(self, bad_top):
        layer_map = LayerMapping(2)
        via = _MockVia(top_layer=bad_top, bottom_layer=2)
        with pytest.raises(ValueError, match="cannot map top board layer"):
            _ViaView(via, layer_map)

    @pytest.mark.parametrize("bad_bot", [1, 3, 5, 9])
    def test_via_bottom_layer_unmappable_raises(self, bad_bot):
        layer_map = LayerMapping(2)
        via = _MockVia(top_layer=0, bottom_layer=bad_bot)
        with pytest.raises(ValueError, match="cannot map bottom board layer"):
            _ViaView(via, layer_map)

    def test_via_error_message_includes_context(self):
        layer_map = LayerMapping(2)
        via = _MockVia(
            top_layer=88, bottom_layer=2,
            net_name="CLK", x_mm=4.5, y_mm=6.5,
        )
        with pytest.raises(ValueError) as excinfo:
            _ViaView(via, layer_map)
        msg = str(excinfo.value)
        assert "88" in msg
        assert "CLK" in msg
        assert "4.5" in msg and "6.5" in msg

    def test_through_via_passes(self):
        """Sanity: F_Cu→B_Cu via on a 2-layer board must NOT raise."""
        layer_map = LayerMapping(2)
        via = _MockVia(top_layer=0, bottom_layer=2)
        wrapped = _ViaView(via, layer_map)
        assert wrapped.top_layer == 1
        assert wrapped.bottom_layer == 2

    def test_inner_via_on_4layer_passes(self):
        """Sanity: inner-layer via on a 4-layer board (top=In1, bot=In2) passes."""
        layer_map = LayerMapping(4)  # [F_Cu=0, In1=4, In2=6, B_Cu=2]
        via = _MockVia(top_layer=4, bottom_layer=6)
        wrapped = _ViaView(via, layer_map)
        assert wrapped.top_layer == 2  # In1 → human 2
        assert wrapped.bottom_layer == 3  # In2 → human 3


# ---------------------------------------------------------------------------
# 5. Empty Edge.Cuts outline  (must raise, not default to bbox (0,0,100,100))
# ---------------------------------------------------------------------------

class TestEmptyOutlineStrictRaise:

    def test_empty_outline_raises(self):
        engine = _make_mock_engine(pads=[], edges=[])  # no Edge.Cuts segments
        with pytest.raises(ValueError, match="no Edge.Cuts outline"):
            parse_pcb_file("path/to/missing_outline.kicad_pcb", engine)

    def test_outline_error_message_includes_path(self):
        engine = _make_mock_engine(pads=[], edges=[])
        with pytest.raises(ValueError) as excinfo:
            parse_pcb_file("path/to/missing_outline.kicad_pcb", engine)
        assert "missing_outline.kicad_pcb" in str(excinfo.value)

    def test_single_outline_segment_passes(self):
        """Sanity: one Edge.Cuts segment is enough to compute a bbox."""
        engine = _make_mock_engine(
            pads=[],
            edges=[_MockEdge(x1_mm=0, y1_mm=0, x2_mm=20, y2_mm=10)],
        )
        out = parse_pcb_file("dummy.kicad_pcb", engine)
        meta = out["board_meta"]
        assert meta.bbox_x == 0 and meta.bbox_y == 0
        assert meta.bbox_w == 20 and meta.bbox_h == 10
