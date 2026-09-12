"""Tests for the D1 grid-scenario figure generator."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pytest

from experiments.kdd.figure5_d1.plot_grid_scenarios import (
    ConnectorInstance,
    _draw_panel,
    build_partial_routes,
    cell_width_fraction,
    load_connector_instance,
    pad_style_contract,
    route_head_arrow_contract,
    route_head_direction,
    route_fraction_for_grid,
)


def _connector_array_root() -> Path:
    """Fixed Connector-v2 instance arrays: D1_BASELINE_ARRAY_ROOT overrides,
    else the ``d1_connector_arrays`` dataset of configs/paths.yaml (an
    absolute-sub entry outside the data root); the caller skips when the
    resolved directory is absent."""
    env = os.environ.get("D1_BASELINE_ARRAY_ROOT")
    if env:
        return Path(env)
    from configs.loader.paths import resolve_dataset

    try:
        return resolve_dataset("d1_connector_arrays")
    except RuntimeError as e:  # empty data root (release configuration)
        pytest.skip(str(e))


def _write_npz(path: Path, *, grid_size: int = 10) -> None:
    starts = np.asarray([[[0, 0], [2, 2]]], dtype=np.int32)
    targets = np.asarray([[[8, 4], [2, 8]]], dtype=np.int32)
    np.savez(
        path,
        starts=starts,
        targets=targets,
        board_ids=np.asarray(["board_00000"], dtype="U128"),
        grid_size=np.asarray(grid_size),
        num_agents=np.asarray(2),
        connector_version=np.asarray("Connector-v2"),
        reward_family=np.asarray("jumanji_connector"),
    )


def test_load_connector_instance_reads_board_coordinates(tmp_path: Path) -> None:
    npz_path = tmp_path / "test.npz"
    _write_npz(npz_path, grid_size=10)

    instance = load_connector_instance(npz_path)

    assert instance.board_id == "board_00000"
    assert instance.grid_size == 10
    assert instance.starts.tolist() == [[0, 0], [2, 2]]
    assert instance.targets.tolist() == [[8, 4], [2, 8]]


def test_build_partial_routes_create_discrete_cells_that_stop_before_targets() -> None:
    starts = np.asarray([[0, 0], [2, 2]], dtype=np.int32)
    targets = np.asarray([[8, 4], [2, 8]], dtype=np.int32)

    routes = build_partial_routes(starts, targets, fraction=0.5)

    assert [route.cells[0] for route in routes] == [(0, 0), (2, 2)]
    assert [route.head_cell for route in routes] == [route.cells[-1] for route in routes]
    assert [route.target_cell for route in routes] == [(8, 4), (2, 8)]
    assert [route.target_center for route in routes] == [(8.5, 4.5), (2.5, 8.5)]
    for route in routes:
        assert route.head_cell != route.target_cell
        assert route.head_center != route.target_center
        assert all(isinstance(coord, int) for cell in route.cells for coord in cell)


def test_build_partial_routes_avoid_existing_route_occupancy_when_possible() -> None:
    starts = np.asarray([[2, 5], [0, 3]], dtype=np.int32)
    targets = np.asarray([[2, 0], [4, 3]], dtype=np.int32)

    routes = build_partial_routes(starts, targets, fraction=0.55, grid_size=6)

    first_route_cells = set(routes[0].cells[1:])
    second_route_cells = set(routes[1].cells[1:])
    assert not (first_route_cells & second_route_cells)


def test_build_partial_routes_avoid_overlap_on_fixed_connector_instances() -> None:
    array_root = _connector_array_root()
    if not array_root.exists():
        pytest.skip("fixed Connector-v2 arrays are not available "
                    "(set D1_BASELINE_ARRAY_ROOT to their directory)")

    for grid_size in (10, 50):
        instance = load_connector_instance(array_root / f"grid{grid_size}" / "test.npz")
        routes = build_partial_routes(
            instance.starts,
            instance.targets,
            fraction=route_fraction_for_grid(grid_size),
            grid_size=grid_size,
        )
        route_cells: dict[tuple[int, int], int] = {}
        endpoint_cells = {tuple(cell) for cell in instance.starts.tolist() + instance.targets.tolist()}
        for route in routes:
            for cell in route.cells:
                assert cell not in route_cells
                assert cell not in endpoint_cells - {route.cells[0], route.target_cell}
                route_cells[cell] = route.net_index


def test_grid50_routes_are_longer_than_grid10_for_visual_scale_contrast() -> None:
    assert route_fraction_for_grid(50) > route_fraction_for_grid(10)
    assert route_fraction_for_grid(50) < 0.98


def test_pad_style_contract_matches_connector_animation_semantics() -> None:
    style = pad_style_contract()

    assert style["source"]["fills_full_cell"] is True
    assert style["source"]["alpha"] > style["route"]["alpha"]
    assert style["source"]["edgecolor"] == "none"
    assert style["source"]["linewidth"] == 0.0
    assert style["target"]["fills_full_cell"] is False
    assert style["target"]["edgecolor"] == "white"
    assert style["target"]["linewidth"] <= 0.35
    assert style["source"]["inset"] < style["target"]["inset"]
    assert style["route"]["fills_full_cell"] is True


def test_panel_does_not_draw_auxiliary_scale_cue() -> None:
    instance = ConnectorInstance(
        board_id="board_00000",
        grid_size=10,
        starts=np.asarray([[0, 0], [2, 2]], dtype=np.int32),
        targets=np.asarray([[8, 4], [2, 8]], dtype=np.int32),
    )
    fig, ax = plt.subplots()
    try:
        _draw_panel(ax, instance)
        cue_rgb = mcolors.to_rgb("#111827")
        neutral_cue_patches = [
            patch
            for patch in ax.patches
            if tuple(round(channel, 6) for channel in patch.get_facecolor()[:3])
            == tuple(round(channel, 6) for channel in cue_rgb)
            and patch.get_alpha() == pytest.approx(0.46)
        ]
        assert neutral_cue_patches == []
    finally:
        plt.close(fig)


def _direction_name(direction: tuple[int, int]) -> str:
    return {
        (0, -1): "up",
        (0, 1): "down",
        (-1, 0): "left",
        (1, 0): "right",
    }[direction]


def test_route_head_directions_cover_four_neighbor_moves_on_fixed_instances() -> None:
    array_root = _connector_array_root()
    if not array_root.exists():
        pytest.skip("fixed Connector-v2 arrays are not available "
                    "(set D1_BASELINE_ARRAY_ROOT to their directory)")

    expected_examples = {
        (10, 0): "down",
        (10, 1): "right",
        (10, 2): "up",
        (50, 3): "left",
    }
    seen: set[str] = set()
    for grid_size in (10, 50):
        instance = load_connector_instance(array_root / f"grid{grid_size}" / "test.npz")
        routes = build_partial_routes(
            instance.starts,
            instance.targets,
            fraction=route_fraction_for_grid(grid_size),
            grid_size=grid_size,
        )
        for route in routes:
            direction = route_head_direction(route, grid_size)
            name = _direction_name(direction)
            seen.add(name)

            hx, hy = route.head_cell
            tx, ty = route.target_cell
            before = abs(tx - hx) + abs(ty - hy)
            after = abs(tx - (hx + direction[0])) + abs(ty - (hy + direction[1]))
            assert after < before
            if (grid_size, route.net_index) in expected_examples:
                assert name == expected_examples[(grid_size, route.net_index)]

    assert seen == {"down", "left", "right", "up"}


def test_route_head_arrow_contract_scales_with_grid_resolution() -> None:
    grid10 = route_head_arrow_contract(10)
    grid50 = route_head_arrow_contract(50)

    assert grid10["length"] > grid10["gap"] > 0
    assert grid50["length"] > grid50["gap"] > 0
    assert grid50["length"] > grid10["length"]
    assert grid10["linewidth"] >= 1.35
    assert grid50["linewidth"] >= 1.20


def test_panel_draws_route_head_arrows_without_standalone_glyph_or_bitmap_title() -> None:
    instance = ConnectorInstance(
        board_id="board_00000",
        grid_size=10,
        starts=np.asarray([[0, 0], [2, 2]], dtype=np.int32),
        targets=np.asarray([[8, 4], [2, 8]], dtype=np.int32),
    )
    fig, ax = plt.subplots()
    try:
        _draw_panel(ax, instance)
        route_head_arrows = [
            patch
            for patch in ax.patches
            if isinstance(patch, FancyArrowPatch)
            and patch.get_label() == "route-head-action-arrow"
        ]
        standalone_arrows = [patch for patch in ax.patches if patch.get_label() == "four-neighbor-action-arrow"]
        standalone_centers = [patch for patch in ax.patches if patch.get_label() == "four-neighbor-action-center"]

        assert len(route_head_arrows) == len(instance.starts)
        assert standalone_arrows == []
        assert standalone_centers == []
        assert ax.get_title() == ""
    finally:
        plt.close(fig)


def test_cell_width_fraction_shrinks_with_grid_resolution() -> None:
    assert cell_width_fraction(10) == pytest.approx(0.1)
    assert cell_width_fraction(50) == pytest.approx(0.02)
    assert cell_width_fraction(50) < cell_width_fraction(10)
