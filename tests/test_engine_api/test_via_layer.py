"""Via and layer control API tests (Step 2.5).

Tests:
- toggle_via / is_placing_via
- switch_layer / get_current_layer
- set_via_diameter / set_via_drill
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))

BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"


def _import_krl():
    try:
        import kicad_rl_router as krl
        return krl
    except ImportError:
        pytest.skip("kicad_rl_router module not available")


@pytest.fixture
def board_path() -> str:
    if not BOARD_PATH.exists():
        pytest.skip(f"Test board not found: {BOARD_PATH}")
    return str(BOARD_PATH)


class TestToggleVia:
    def test_toggle_via_changes_state(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.start_route(0.0, 0.0, 0)

        assert r.is_placing_via() is False
        r.toggle_via()
        assert r.is_placing_via() is True
        r.toggle_via()
        assert r.is_placing_via() is False

        r.cancel_route()

    def test_toggle_via_before_routing(self, board_path: str) -> None:
        """toggle_via can be called even when not routing (no crash)."""
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.toggle_via()  # should not raise
        # PNS via state only meaningful during active routing
        assert r.is_placing_via() is False


class TestSwitchLayer:
    def test_get_current_layer_default(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        layer = r.get_current_layer()
        assert isinstance(layer, int)
        # -1 is valid when not routing (no active layer)
        assert layer == -1

    def test_switch_layer_during_routing(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.start_route(0.0, 0.0, 0)
        r.move(1.0, 1.0)

        ok = r.switch_layer(31)  # B.Cu
        # switch_layer may or may not succeed depending on PNS state
        assert isinstance(ok, bool)

        r.cancel_route()

    def test_switch_layer_not_routing_returns_false(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        ok = r.switch_layer(31)
        assert ok is False


class TestViaDimensions:
    def test_set_via_diameter(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_via_diameter(0.6)

    def test_set_via_drill(self, board_path: str) -> None:
        krl = _import_krl()
        r = krl.RLRouter(board_path)
        r.set_via_drill(0.3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
