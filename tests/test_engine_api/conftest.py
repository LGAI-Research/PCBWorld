"""Shared fixtures for API tests."""

import shutil
from pathlib import Path

import pytest
import yaml

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
# KiCad 9 pcb whose setup block carries only modern tokens: without its .kicad_pro
# the load sets neither provenance flag (the "no rules" state).
MODERN_PCB_FIXTURE = FIXTURES_DIR / "simple_routing_board.kicad_pcb"


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def board_path():
    """Return path to default test board."""
    p = FIXTURES_DIR / "simple_obstacle_board.kicad_pcb"
    if not p.exists():
        pytest.skip(f"Board not found: {p}")
    return str(p)


@pytest.fixture
def sample_board_path():
    """Return path to sample board."""
    p = FIXTURES_DIR / "sample_board.kicad_pcb"
    if not p.exists():
        pytest.skip(f"Board not found: {p}")
    return str(p)


@pytest.fixture
def krl():
    """The in-process ``kicad_rl_router`` module (skips when it is not built)."""
    return pytest.importorskip("kicad_rl_router")


@pytest.fixture
def orphan_modern_pcb(tmp_path: Path) -> Path:
    """The KiCad 9 pcb copied into tmp_path WITHOUT its .kicad_pro sibling: both
    provenance flags come back False, the only state where ``rules_source`` acts."""
    dest = tmp_path / "orphan.kicad_pcb"
    shutil.copy(MODERN_PCB_FIXTURE, dest)
    return dest


@pytest.fixture
def custom_drc_yaml(tmp_path: Path) -> Path:
    """DRC YAML whose minima differ from the repo default and from KiCad's built-in
    rules, so an assertion tells "YAML applied" from "left at engine defaults"."""
    path = tmp_path / "custom_drc.yaml"
    path.write_text(yaml.safe_dump({
        "name": "test_custom",
        "global_minima": {
            "clearance":             0.37,
            "track_width":           0.41,
            "via_diameter":          0.71,
            "through_hole_drill":    0.27,
            "via_annular_width":     0.13,
            "hole_to_hole":          0.29,
            "uvia_diameter":         0.19,
            "uvia_drill":            0.09,
            "copper_edge_clearance": 0.17,
        },
    }))
    return path
