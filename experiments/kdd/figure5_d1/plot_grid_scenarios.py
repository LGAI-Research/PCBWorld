#!/usr/bin/env python3
"""Render the D1 Connector-v2 grid scenario figure.

The figure uses real fixed Connector-v2 start/target coordinates and overlays
deterministic partial Manhattan routes for explanatory visualization.
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe
import numpy as np

# Direct execution puts this file's directory on sys.path, not the repository
# root, which the ``configs`` import below needs.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from configs.loader.paths import data_root_path

# The packaged benchmark tree is not shipped with the repo: KDD_BENCH_ROOT, or
# $CADAGENT_DATA_ROOT/KDD_benchmark, or "" (checked before use).
DEFAULT_BENCH_ROOT = Path(
    os.environ.get("KDD_BENCH_ROOT") or data_root_path("KDD_benchmark")
)
DEFAULT_EXPR_ROOT = Path(os.environ.get("EXPR_ROOT", str(DEFAULT_BENCH_ROOT / "experimental_results")))
DEFAULT_DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", str(DEFAULT_BENCH_ROOT / "dataset")))
DEFAULT_ARRAY_ROOT = DEFAULT_DATASET_ROOT / "synthetic/connector_v2"
DEFAULT_OUTPUT = Path(os.environ.get("RQ1_SCENARIO_OUTPUT", "figs/rq1_grid_scenarios.png"))
DEFAULT_GRIDS = (10, 50)
DEFAULT_BOARD_INDEX = 0
NET_COLORS = ("#d7191c", "#2c7bb6", "#00a65a", "#f6c700", "#b900d9")


@dataclass(frozen=True)
class ConnectorInstance:
    board_id: str
    grid_size: int
    starts: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class PartialRoute:
    net_index: int
    cells: tuple[tuple[int, int], ...]
    head_cell: tuple[int, int]
    target_cell: tuple[int, int]

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        return tuple(_center(cell) for cell in self.cells)

    @property
    def head_center(self) -> tuple[float, float]:
        return _center(self.head_cell)

    @property
    def target_center(self) -> tuple[float, float]:
        return _center(self.target_cell)


def cell_width_fraction(grid_size: int) -> float:
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    return 1.0 / float(grid_size)


def route_fraction_for_grid(grid_size: int) -> float:
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    return 0.78 if grid_size >= 50 else 0.55


def route_head_arrow_contract(grid_size: int) -> dict[str, float]:
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if grid_size <= 20:
        return {
            "gap": 0.17,
            "length": 0.54,
            "linewidth": 2.05,
            "mutation_scale": 11.8,
            "stroke_width": 3.00,
        }
    return {
        "gap": 0.36,
        "length": 1.34,
        "linewidth": 1.85,
        "mutation_scale": 10.8,
        "stroke_width": 2.75,
    }


def load_connector_instance(npz_path: Path, board_index: int = DEFAULT_BOARD_INDEX) -> ConnectorInstance:
    with np.load(npz_path, allow_pickle=False) as data:
        starts = np.asarray(data["starts"], dtype=np.int32)
        targets = np.asarray(data["targets"], dtype=np.int32)
        board_ids = np.asarray(data["board_ids"]).astype(str)
        grid_size = int(np.asarray(data["grid_size"]).item())

    if board_index < 0 or board_index >= len(board_ids):
        raise IndexError(f"board_index={board_index} outside 0..{len(board_ids) - 1}")
    return ConnectorInstance(
        board_id=str(board_ids[board_index]),
        grid_size=grid_size,
        starts=np.asarray(starts[board_index], dtype=np.int32),
        targets=np.asarray(targets[board_index], dtype=np.int32),
    )


def _center(cell: Sequence[int]) -> tuple[float, float]:
    return float(cell[0]) + 0.5, float(cell[1]) + 0.5


def _as_cell(cell: Sequence[int]) -> tuple[int, int]:
    return int(cell[0]), int(cell[1])


def _walk_discrete_manhattan(
    start_cell: tuple[int, int],
    target_cell: tuple[int, int],
    *,
    x_first: bool,
) -> tuple[tuple[int, int], ...]:
    current = [start_cell[0], start_cell[1]]
    target = [target_cell[0], target_cell[1]]
    path = [tuple(current)]
    axes = (0, 1) if x_first else (1, 0)
    for axis in axes:
        while current[axis] != target[axis]:
            current[axis] += 1 if target[axis] > current[axis] else -1
            path.append(tuple(current))
    return tuple(path)


def _partial_cells(
    start_cell: tuple[int, int],
    target_cell: tuple[int, int],
    *,
    fraction: float,
    x_first: bool,
) -> tuple[tuple[int, int], ...]:
    full_path = _walk_discrete_manhattan(start_cell, target_cell, x_first=x_first)
    if len(full_path) <= 1:
        return full_path
    last_non_target_idx = len(full_path) - 2
    requested_idx = math.ceil((len(full_path) - 1) * min(max(float(fraction), 0.0), 0.98))
    head_idx = max(0, min(last_non_target_idx, requested_idx))
    return full_path[: head_idx + 1]


def _partial_path_cells(
    path: tuple[tuple[int, int], ...],
    *,
    fraction: float,
) -> tuple[tuple[int, int], ...]:
    if len(path) <= 1:
        return path
    last_non_target_idx = len(path) - 2
    requested_idx = math.ceil((len(path) - 1) * min(max(float(fraction), 0.0), 0.98))
    head_idx = max(0, min(last_non_target_idx, requested_idx))
    return path[: head_idx + 1]


def _infer_grid_size(starts: np.ndarray, targets: np.ndarray) -> int:
    max_coord = int(np.max(np.concatenate([starts, targets], axis=0)))
    return max_coord + 1


def _ordered_neighbors(
    cell: tuple[int, int],
    target: tuple[int, int],
    *,
    prefer_x: bool,
) -> tuple[tuple[int, int], ...]:
    x, y = cell
    tx, ty = target
    x_step = 1 if tx > x else -1
    y_step = 1 if ty > y else -1
    x_moves = [(x + x_step, y), (x - x_step, y)]
    y_moves = [(x, y + y_step), (x, y - y_step)]
    ordered = x_moves + y_moves if prefer_x else y_moves + x_moves
    return tuple(dict.fromkeys(ordered))


def _shortest_grid_path(
    start: tuple[int, int],
    target: tuple[int, int],
    *,
    grid_size: int,
    blocked: set[tuple[int, int]],
    prefer_x: bool,
) -> tuple[tuple[int, int], ...] | None:
    blocked = set(blocked) - {start, target}
    queue: list[tuple[int, int, int, tuple[int, int], tuple[tuple[int, int], ...]]] = []
    start_h = abs(target[0] - start[0]) + abs(target[1] - start[1])
    heapq.heappush(queue, (start_h, 0, 0, start, (start,)))
    best_cost = {start: 0}
    order = 0

    while queue:
        _priority, cost, _order, cell, path = heapq.heappop(queue)
        if cell == target:
            return path
        if cost > best_cost[cell]:
            continue
        for neighbor in _ordered_neighbors(cell, target, prefer_x=prefer_x):
            x, y = neighbor
            if x < 0 or y < 0 or x >= grid_size or y >= grid_size:
                continue
            if neighbor in blocked:
                continue
            next_cost = cost + 1
            if next_cost >= best_cost.get(neighbor, 10**9):
                continue
            best_cost[neighbor] = next_cost
            heuristic = abs(target[0] - x) + abs(target[1] - y)
            order += 1
            heapq.heappush(queue, (next_cost + heuristic, next_cost, order, neighbor, path + (neighbor,)))
    return None


def _route_collision_score(
    cells: tuple[tuple[int, int], ...],
    *,
    own_start: tuple[int, int],
    reserved_cells: set[tuple[int, int]],
    occupied_cells: set[tuple[int, int]],
) -> tuple[int, int, int]:
    route_cells = set(cells[1:])
    endpoint_hits = len(route_cells & reserved_cells)
    occupied_hits = len(route_cells & occupied_cells)
    source_revisits = cells.count(own_start) - 1
    return endpoint_hits, occupied_hits, source_revisits


def build_partial_routes(
    starts: np.ndarray,
    targets: np.ndarray,
    *,
    fraction: float = 0.55,
    grid_size: int | None = None,
) -> list[PartialRoute]:
    if starts.shape != targets.shape:
        raise ValueError(f"starts shape {starts.shape} does not match targets shape {targets.shape}")
    if grid_size is None:
        grid_size = _infer_grid_size(starts, targets)
    routes: list[PartialRoute] = []
    endpoint_cells = {_as_cell(cell) for cell in np.concatenate([starts, targets], axis=0)}
    occupied_cells: set[tuple[int, int]] = set()
    for net_index, (start_cell, target_cell) in enumerate(zip(starts, targets, strict=True)):
        start = _as_cell(start_cell)
        target = _as_cell(target_cell)
        reserved_cells = endpoint_cells - {start, target}
        full_path = _shortest_grid_path(
            start,
            target,
            grid_size=grid_size,
            blocked=reserved_cells | occupied_cells,
            prefer_x=(net_index % 2 == 0),
        )
        if full_path is None:
            candidates = []
            for x_first in (net_index % 2 == 0, net_index % 2 != 0):
                cells = _partial_cells(start, target, fraction=fraction, x_first=x_first)
                candidates.append(
                    (
                        _route_collision_score(
                            cells,
                            own_start=start,
                            reserved_cells=reserved_cells,
                            occupied_cells=occupied_cells,
                        ),
                        cells,
                    )
                )
            cells = min(candidates, key=lambda item: item[0])[1]
        else:
            cells = _partial_path_cells(
                full_path,
                fraction=fraction,
            )
        occupied_cells.update(cells)
        routes.append(
            PartialRoute(
                net_index=net_index,
                cells=cells,
                head_cell=cells[-1],
                target_cell=target,
            )
        )
    return routes


def route_head_direction(route: PartialRoute, grid_size: int) -> tuple[int, int]:
    del grid_size
    hx, hy = route.head_cell
    tx, ty = route.target_cell
    dx = tx - hx
    dy = ty - hy
    prefer_x = route.net_index % 2 == 1
    if prefer_x and dx != 0:
        return (1 if dx > 0 else -1, 0)
    if dy != 0:
        return (0, 1 if dy > 0 else -1)
    if dx != 0:
        return (1 if dx > 0 else -1, 0)
    return (0, 0)


def pad_style_contract() -> dict[str, dict[str, float | bool | str]]:
    return {
        "source": {
            "fills_full_cell": True,
            "inset": 0.02,
            "alpha": 0.98,
            "edgecolor": "none",
            "linewidth": 0.0,
        },
        "target": {
            "fills_full_cell": False,
            "inset": 0.25,
            "alpha": 0.98,
            "edgecolor": "white",
            "linewidth": 0.30,
        },
        "route": {
            "fills_full_cell": True,
            "inset": 0.025,
            "alpha": 0.28,
            "edgecolor": "none",
            "linewidth": 0.0,
        },
    }


def _draw_grid(ax: Axes, grid_size: int) -> None:
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    minor_alpha = 0.36 if grid_size <= 20 else 0.10
    major_alpha = 0.78 if grid_size <= 20 else 0.30
    for i in range(grid_size + 1):
        alpha = major_alpha if grid_size <= 20 or i % 5 == 0 else minor_alpha
        lw = 0.80 if grid_size <= 20 or i % 5 == 0 else 0.30
        ax.axhline(i, color="#111827", linewidth=lw, alpha=alpha, zorder=0)
        ax.axvline(i, color="#111827", linewidth=lw, alpha=alpha, zorder=0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.95)
        spine.set_color("#111827")


def _add_cell_rect(
    ax: Axes,
    cell: tuple[int, int],
    *,
    color: str,
    inset: float,
    alpha: float,
    edgecolor: str,
    linewidth: float,
    zorder: int,
) -> None:
    x, y = cell
    ax.add_patch(
        Rectangle(
            (x + inset, y + inset),
            1 - 2 * inset,
            1 - 2 * inset,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    )


def _draw_target(ax: Axes, cell: tuple[int, int], color: str, grid_size: int) -> None:
    style = pad_style_contract()["target"]
    _add_cell_rect(
        ax,
        cell,
        color=color,
        inset=float(style["inset"]),
        alpha=float(style["alpha"]),
        edgecolor=str(style["edgecolor"]),
        linewidth=float(style["linewidth"]),
        zorder=5,
    )


def _draw_source(ax: Axes, cell: tuple[int, int], color: str, grid_size: int) -> None:
    style = pad_style_contract()["source"]
    _add_cell_rect(
        ax,
        cell,
        color=color,
        inset=float(style["inset"]),
        alpha=float(style["alpha"]),
        edgecolor=str(style["edgecolor"]),
        linewidth=float(style["linewidth"]),
        zorder=7,
    )


def _draw_route(ax: Axes, route: PartialRoute, color: str, grid_size: int) -> None:
    style = pad_style_contract()["route"]
    for cell in route.cells:
        _add_cell_rect(
            ax,
            cell,
            color=color,
            inset=float(style["inset"]),
            alpha=float(style["alpha"]),
            edgecolor=str(style["edgecolor"]),
            linewidth=float(style["linewidth"]),
            zorder=2,
        )

    hx, hy = route.head_center
    ax.scatter(
        [hx],
        [hy],
        s=18 if grid_size <= 20 else 9,
        color=color,
        edgecolor="white",
        linewidth=0.45,
        zorder=8,
    )


def _draw_route_head_arrow(ax: Axes, route: PartialRoute, color: str, grid_size: int) -> None:
    direction = route_head_direction(route, grid_size)
    if direction == (0, 0):
        return
    hx, hy = route.head_center
    gap = route_head_arrow_contract(grid_size)["gap"]
    length = route_head_arrow_contract(grid_size)["length"]
    start = (hx + direction[0] * gap, hy + direction[1] * gap)
    end = (hx + direction[0] * (gap + length), hy + direction[1] * (gap + length))
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=route_head_arrow_contract(grid_size)["mutation_scale"],
        linewidth=route_head_arrow_contract(grid_size)["linewidth"],
        color=color,
        alpha=0.86,
        zorder=9,
        shrinkA=0,
        shrinkB=0,
        label="route-head-action-arrow",
    )
    arrow.set_path_effects(
        [
            pe.withStroke(
                linewidth=route_head_arrow_contract(grid_size)["stroke_width"],
                foreground="white",
                alpha=0.86,
            )
        ]
    )
    ax.add_patch(arrow)


def _draw_panel(ax: Axes, instance: ConnectorInstance) -> None:
    grid_size = instance.grid_size
    _draw_grid(ax, grid_size)
    routes = build_partial_routes(
        instance.starts,
        instance.targets,
        fraction=route_fraction_for_grid(grid_size),
        grid_size=grid_size,
    )
    for route in routes:
        color = NET_COLORS[route.net_index % len(NET_COLORS)]
        _draw_route(ax, route, color, grid_size)
        _draw_route_head_arrow(ax, route, color, grid_size)
        _draw_target(ax, route.target_cell, color, grid_size)
        _draw_source(ax, route.cells[0], color, grid_size)

    # Panel titles are typeset in LaTeX so the bitmap only contains geometry.


def render_figure(instances: Sequence[ConnectorInstance], output: Path) -> None:
    if len(instances) != 2:
        raise ValueError("D1 grid scenario figure expects exactly two grid instances")
    fig = plt.figure(figsize=(7.15, 3.55), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, instance in zip(axes, instances, strict=True):
        _draw_panel(ax, instance)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_panel(instance: ConnectorInstance, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 3.45), constrained_layout=True)
    _draw_panel(ax, instance)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_output_path(output: Path, grid_size: int) -> Path:
    return output.with_name(f"{output.stem}_grid{grid_size}{output.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-root", type=Path, default=DEFAULT_ARRAY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--board-index", type=int, default=DEFAULT_BOARD_INDEX)
    parser.add_argument("--grids", type=int, nargs=2, default=DEFAULT_GRIDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_paths = [args.array_root / f"grid{grid}" / "test.npz" for grid in args.grids]
    missing = [p for p in npz_paths if not p.is_file()]
    if missing:
        print("figure5_d1: a required D1 input is absent — nothing was drawn.", file=sys.stderr)
        for path in missing:
            print(f"  missing  Connector-v2 test arrays: {path}", file=sys.stderr)
        print(
            "\nD1 (paper Figure 5) is the synthetic 1-layer grid sweep. Its corpus is\n"
            "NOT distributed with this repository and no generator here reproduces\n"
            "it. To draw this figure, point --array-root (or DATASET_ROOT) at a tree\n"
            "that provides the paths above.\n"
            "Details: experiments/kdd/figure5_d1/README.md",
            file=sys.stderr,
        )
        raise SystemExit(2)
    instances = [
        load_connector_instance(path, board_index=args.board_index)
        for path in npz_paths
    ]
    render_figure(instances, args.output)
    for instance in instances:
        render_panel(instance, panel_output_path(args.output, instance.grid_size))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
