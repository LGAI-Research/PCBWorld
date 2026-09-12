"""Tests for LayerMapping: human layer <-> board layer conversion.

Unit tests verify the mapping logic for 2, 4, 6, 8 copper layers.
Integration test loads a real 4-layer board through the C++ engine.
"""

from __future__ import annotations

import os
import pytest

from pcb_world.engine.layer_mapping import LayerMapping


# ---------------------------------------------------------------------------
# 1. Board layer order
# ---------------------------------------------------------------------------

class TestLayerMappingBuildOrder:
    """Verify board_layer_order for various copper layer counts."""

    def test_2_layer_order(self):
        m = LayerMapping(2)
        assert m.board_layer_order == [0, 2]  # F_Cu, B_Cu
        assert m.max_layer == 2

    def test_4_layer_order(self):
        m = LayerMapping(4)
        assert m.board_layer_order == [0, 4, 6, 2]  # F_Cu, In1, In2, B_Cu
        assert m.max_layer == 4

    def test_6_layer_order(self):
        m = LayerMapping(6)
        assert m.board_layer_order == [0, 4, 6, 8, 10, 2]
        assert m.max_layer == 6

    def test_8_layer_order(self):
        m = LayerMapping(8)
        assert m.board_layer_order == [0, 4, 6, 8, 10, 12, 14, 2]
        assert m.max_layer == 8


# ---------------------------------------------------------------------------
# 2. Bidirectional (round-trip) conversion
# ---------------------------------------------------------------------------

class TestLayerMappingBidirectional:
    """human_to_board and board_to_human must be inverses."""

    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_round_trip_all_counts(self, n):
        m = LayerMapping(n)
        for h in range(1, n + 1):
            b = m.human_to_board(h)
            assert m.board_to_human(b) == h, f"Round-trip failed for human={h}"
        for b in m.board_layer_order:
            h = m.board_to_human(b)
            assert m.human_to_board(h) == b, f"Round-trip failed for board={b}"

    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_fcu_always_human_1(self, n):
        """F_Cu (board=0) is always human layer 1."""
        m = LayerMapping(n)
        assert m.human_to_board(1) == 0
        assert m.board_to_human(0) == 1

    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_bcu_always_human_last(self, n):
        """B_Cu (board=2) is always the last human layer."""
        m = LayerMapping(n)
        assert m.human_to_board(n) == 2
        assert m.board_to_human(2) == n

    def test_4_layer_inner_mapping(self):
        """4-layer: human 2=In1_Cu(4), human 3=In2_Cu(6)."""
        m = LayerMapping(4)
        assert m.human_to_board(2) == 4   # In1_Cu
        assert m.human_to_board(3) == 6   # In2_Cu
        assert m.board_to_human(4) == 2
        assert m.board_to_human(6) == 3


# ---------------------------------------------------------------------------
# 3. Boundary / error cases
# ---------------------------------------------------------------------------

class TestLayerMappingBoundary:
    """Invalid inputs and edge cases."""

    def test_invalid_human_layer_raises(self):
        m = LayerMapping(2)
        with pytest.raises(KeyError):
            m.human_to_board(3)
        with pytest.raises(KeyError):
            m.human_to_board(0)
        with pytest.raises(KeyError):
            m.human_to_board(-1)

    def test_invalid_board_layer_raises(self):
        m = LayerMapping(2)
        with pytest.raises(KeyError):
            m.board_to_human(4)  # In1_Cu doesn't exist on 2-layer

    def test_min_clamps_to_2(self):
        """LayerMapping(1) should behave like LayerMapping(2)."""
        m1 = LayerMapping(1)
        m2 = LayerMapping(2)
        assert m1.board_layer_order == m2.board_layer_order
        assert m1.max_layer == m2.max_layer


# ---------------------------------------------------------------------------
# 4. Integration: load 4-layer board through C++ engine
# ---------------------------------------------------------------------------

_4LAYER_BOARD = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "build_rl", "kicad_src", "demos",
    "kit-dev-coldfire-xilinx_5213", "kit-dev-coldfire-xilinx_5213.kicad_pcb",
)


def _skip_if_no_kicad():
    try:
        import kicad_rl_router  # noqa: F401
    except ImportError:
        pytest.skip("kicad_rl_router not available")


def _skip_if_no_4layer_board():
    if not os.path.exists(_4LAYER_BOARD):
        pytest.skip(f"4-layer board not found: {_4LAYER_BOARD}")


class TestLayerMappingWithEngine:
    """Integration: verify LayerMapping on a real 4-layer board."""

    def test_4layer_board_engine(self):
        _skip_if_no_kicad()
        _skip_if_no_4layer_board()
        from pcb_world.engine.kicad_engine import KiCadEngine

        engine = KiCadEngine(_4LAYER_BOARD)
        assert engine.layer_map.max_layer == 4
        assert engine.layer_map.board_layer_order == [0, 4, 6, 2]
        # Verify conversions
        assert engine.layer_map.human_to_board(1) == 0   # F_Cu
        assert engine.layer_map.human_to_board(2) == 4   # In1_Cu
        assert engine.layer_map.human_to_board(3) == 6   # In2_Cu
        assert engine.layer_map.human_to_board(4) == 2   # B_Cu
        engine.close()
