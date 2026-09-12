"""Unit tests for grid-aware build_directional_candidates (1-layer Grid mode).

Verifies that when mode="grid<N>" is set:
  - candidate count matches per-grid step bundle (4 / 8 / 8 / 12 / 12)
  - all emitted points lie on the underlying grid (offset is integer
    multiple of grid_spacing = 100mm / grid_size)
  - only 4 axis-aligned directions are used (no diagonals)
  - layer is preserved
  - an unsupported grid size raises ValueError
  - default (mode=None) path is unchanged: 8 candidates at 0.5mm
"""

import math

import pytest

from pcb_world.vec.candidate_pool import (
    CTYPE_DIRECTIONAL,
    _BOARD_SIZE_MM,
    _GRID_STEP_CELLS,
    build_directional_candidates,
)


GRID_EXPECTED_COUNT = {
    10: 4,
    30: 8,
    50: 8,
    100: 8,
    200: 12,
    300: 12,
    500: 12,
    1000: 12,
}


@pytest.mark.parametrize("grid_size", sorted(_GRID_STEP_CELLS))
def test_grid_count(grid_size):
    cands = build_directional_candidates(
        (50.0, 50.0), current_layer=1, mode=f"grid{grid_size}",
    )
    assert len(cands) == GRID_EXPECTED_COUNT[grid_size]


@pytest.mark.parametrize("grid_size", sorted(_GRID_STEP_CELLS))
def test_grid_points_on_grid(grid_size):
    """Every candidate must sit on a grid corner relative to the head."""
    spacing = _BOARD_SIZE_MM / grid_size
    hx, hy = 12.5, 7.5  # arbitrary head coords; use real numbers
    cands = build_directional_candidates(
        (hx, hy), current_layer=1, mode=f"grid{grid_size}",
    )
    for x, y, layer, ctype in cands:
        assert ctype == CTYPE_DIRECTIONAL
        dx, dy = x - hx, y - hy
        # one of dx/dy must be 0 (axis-aligned)
        assert math.isclose(dx, 0.0, abs_tol=1e-9) or math.isclose(
            dy, 0.0, abs_tol=1e-9,
        ), f"non-axis-aligned: dx={dx} dy={dy}"
        # the non-zero offset must be an integer multiple of spacing
        nonzero = dx if abs(dx) > abs(dy) else dy
        ratio = nonzero / spacing
        assert math.isclose(ratio, round(ratio), abs_tol=1e-6), (
            f"offset {nonzero} not on grid spacing {spacing}"
        )


@pytest.mark.parametrize("grid_size", sorted(_GRID_STEP_CELLS))
def test_grid_step_set_matches_table(grid_size):
    """Distinct |offset|/spacing values match the configured step_cells."""
    spacing = _BOARD_SIZE_MM / grid_size
    cands = build_directional_candidates(
        (0.0, 0.0), current_layer=2, mode=f"grid{grid_size}",
    )
    step_set = sorted(
        {round((abs(x) + abs(y)) / spacing) for x, y, *_ in cands}
    )
    assert step_set == _GRID_STEP_CELLS[grid_size]


@pytest.mark.parametrize("grid_size", sorted(_GRID_STEP_CELLS))
def test_grid_layer_preserved(grid_size):
    cands = build_directional_candidates(
        (0.0, 0.0), current_layer=3, mode=f"grid{grid_size}",
    )
    assert all(c[2] == 3 for c in cands)


def test_unsupported_grid_size_raises():
    with pytest.raises(ValueError):
        build_directional_candidates((0.0, 0.0), 1, mode="grid42")


def test_default_path_unchanged_no_grid_size():
    """mode=None must be byte-equivalent to original 8-dir 0.5mm path."""
    cands = build_directional_candidates((10.0, 20.0), current_layer=1)
    assert len(cands) == 8
    # Each offset magnitude is 0.5 along axis or sqrt(0.5) for diagonals.
    head = (10.0, 20.0)
    for x, y, layer, ctype in cands:
        dx, dy = x - head[0], y - head[1]
        assert ctype == CTYPE_DIRECTIONAL
        assert layer == 1
        assert any(
            math.isclose(abs(dx), v, abs_tol=1e-9) for v in (0.0, 0.5)
        )
        assert any(
            math.isclose(abs(dy), v, abs_tol=1e-9) for v in (0.0, 0.5)
        )
