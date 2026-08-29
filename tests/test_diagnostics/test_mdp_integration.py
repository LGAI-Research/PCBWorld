"""Integration tests for MDP visualization pipeline.

These tests require the C++ kicad_rl_router library.
Run with:
    conda activate cadagent
    export PYTHONPATH='build_rl/pcbnew/python/rl:.'
    pytest tests/test_diagnostics/test_mdp_integration.py -v
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

# Skip entire module if the C++ router build is absent — probed WITHOUT
# importing the GPL module (import-hygiene gate).
from pcb_world.engine import engine_available

HAS_KICAD = engine_available()

pytestmark = pytest.mark.skipif(not HAS_KICAD, reason="kicad_rl_router not available")


@pytest.fixture
def board_path():
    """Path to test fixture board."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "fixtures", "simple_obstacle_board.kicad_pcb")
    if not os.path.exists(path):
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.mark.skipif(not shutil.which("kicad-cli"), reason="kicad-cli not found on PATH")
class TestPCBRenderer:
    """Test PCBRenderer with real KiCad engine."""

    def test_render_returns_array(self, board_path):
        from pcb_world.engine.kicad_engine import KiCadEngine
        from pcb_world.rendering.renderer import PCBRenderer

        engine = KiCadEngine(board_path)
        renderer = PCBRenderer()
        try:
            frame = renderer.render(engine)
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
            assert frame.shape[2] == 3  # RGB
            assert frame.dtype == np.uint8
        finally:
            renderer.close()

    def test_render_to_file(self, board_path, tmp_path):
        from pcb_world.engine.kicad_engine import KiCadEngine
        from pcb_world.rendering.renderer import PCBRenderer

        engine = KiCadEngine(board_path)
        renderer = PCBRenderer()
        try:
            out_png = str(tmp_path / "test_render.png")
            renderer.render_to_file(engine, out_png)
            assert os.path.exists(out_png)
            assert os.path.getsize(out_png) > 0
        finally:
            renderer.close()

    def test_render_with_overlay(self, board_path):
        from pcb_world.engine.kicad_engine import KiCadEngine
        from pcb_world.rendering.renderer import PCBRenderer

        engine = KiCadEngine(board_path)
        renderer = PCBRenderer()
        try:
            frame = renderer.render(engine, step_info={"Step": 1, "Tracks": 0})
            assert isinstance(frame, np.ndarray)
        finally:
            renderer.close()
