"""Shared fixtures for environment tests."""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


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
