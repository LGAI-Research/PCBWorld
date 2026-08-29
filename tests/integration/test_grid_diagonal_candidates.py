"""Verify 8-direction (axis + 45° diagonal) candidate behavior + actual 45°
KiCad track production under uniform random policy on grid boards.

Counterpart to scripts/verify_grid_directional_random.py:
  - That script wraps with directional_candidates="grid<N>" → 4 axis-aligned candidates
    on grid corners.
  - This script wraps with directional_candidates=None → 8 directions × 0.5mm
    (default mode used by 2-layer / real-board configurations).

For each grid_size:
  1. Generate (or reuse) a tiny 1-layer grid board (3 nets × 2 pins).
  2. Wrap PCBWorld with KiCadRLWrapper(directional_candidates=None).
  3. Run uniform-random policy for several steps × episodes.
  4. While mid-route, assert that the directional candidate set has:
       - exactly 8 candidates,
       - 4 axis-aligned (dx==0 xor dy==0) at distance 0.5 mm,
       - 4 diagonal (|dx|==|dy|==0.5 mm).
  5. After each episode, inspect engine tracks; count segments with
       |dx|>0 AND |dy|>0  → "diagonal" (engine produced 45° miter).
     Aggregate per grid; require at least one diagonal segment across all
     episodes for that grid.

Run on a host with the C++ router built (see the build instructions).
"""
from __future__ import annotations

import argparse
import math
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pcb_world.core.env import PCBWorld  # noqa: E402
from pcb_world.vec.candidate_pool import (  # noqa: E402
    CTYPE_DIRECTIONAL,
    _BOARD_SIZE_MM,
    _DIR_DISTANCES_MM,
    _GRID_STEP_CELLS,
    build_directional_candidates,
    collect_raw_candidates,
)
from methods.rl_agent.wrappers.adapter import (  # noqa: E402
    KiCadRLWrapper,
)

# Default 8-direction mode: 4 axis + 4 diagonals × 1 distance bundle (0.5 mm)
EXPECTED_CAND_COUNT = 8 * len(_DIR_DISTANCES_MM)


def ensure_grid_board(grid_size: int, out_root: Path) -> Path:
    out_dir = out_root / f"grid_{grid_size}"
    pcb = out_dir / "board_00000.kicad_pcb"
    if pcb.is_file():
        return pcb

    out_dir.mkdir(parents=True, exist_ok=True)
    spacing = _BOARD_SIZE_MM / grid_size
    clearance = spacing / 2
    trace_width = spacing / 2
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/datagen/synthetic_generator/generate_synthetic_boards.py"),
        "--n", "1",
        "--seed", str(grid_size),
        "--seed-mode", "legacy",
        "--mode", "grid",
        "--num-layers", "1",
        "--board-size", f"{_BOARD_SIZE_MM:.10g}",
        "--clearance", f"{clearance:.10g}",
        "--trace-width", f"{trace_width:.10g}",
        "--pitch-formula", "c+w",
        "--pad-size", f"{trace_width:.10g}",
        "--min-sep", f"{spacing:.10g}",
        "--fixed-pads-per-net", "2,2,2",
        "--central-frac", "1.0",
        "--via-dia", "0.6",
        "--via-drill", "0.3",
        "--out-dir", str(out_dir),
    ]
    print(f"[gen grid={grid_size}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    if not pcb.is_file():
        raise RuntimeError(f"generator did not produce {pcb}")
    return pcb


def directional_candidates_pre_dedup(obs: dict) -> list[tuple[float, float, int]]:
    rh = obs.get("router_head", {})
    if not rh.get("is_routing", False):
        return []
    head_xy = rh.get("current_xy", [0.0, 0.0])
    layer = int(rh.get("current_layer", 1))
    raw = build_directional_candidates(
        (head_xy[0], head_xy[1]), layer, mode=None,
    )
    return [(x, y, ly) for (x, y, ly, ct) in raw if ct == CTYPE_DIRECTIONAL]


def directional_candidates_post_dedup(obs: dict) -> list[tuple[float, float, int]]:
    rh = obs.get("router_head", {})
    if not rh.get("is_routing", False):
        return []
    head_xy = rh.get("current_xy", [0.0, 0.0])
    layer = int(rh.get("current_layer", 1))
    current_net_id = rh.get("current_net", -1)
    if current_net_id is None or current_net_id <= 0:
        current_net_id = None
    extra = build_directional_candidates(
        (head_xy[0], head_xy[1]), layer, mode=None,
    )
    raw = collect_raw_candidates(obs, current_net_id, extra)
    return [(x, y, ly) for (x, y, ly, ct) in raw if ct == CTYPE_DIRECTIONAL]


def assert_axis_and_diagonal(
    head_xy: tuple[float, float],
    head_layer: int,
    dir_cands: list[tuple[float, float, int]],
    tol: float = 1e-6,
) -> tuple[int, int]:
    """Verify 8-direction structure. Returns (n_axis, n_diag)."""
    if len(dir_cands) != EXPECTED_CAND_COUNT:
        raise AssertionError(
            f"expected {EXPECTED_CAND_COUNT} directional candidates, "
            f"got {len(dir_cands)}"
        )
    hx, hy = head_xy
    n_axis = 0
    n_diag = 0
    valid_dists = set(_DIR_DISTANCES_MM)
    for x, y, layer in dir_cands:
        if layer != head_layer:
            raise AssertionError(
                f"candidate layer {layer} != head layer {head_layer}"
            )
        dx, dy = x - hx, y - hy
        ax = abs(dx) < tol
        ay = abs(dy) < tol
        if ax and ay:
            raise AssertionError(
                f"degenerate candidate (no offset) head=({hx},{hy}) cand=({x},{y})"
            )
        if ax or ay:
            # axis-aligned: nonzero offset must equal one of _DIR_DISTANCES_MM
            offset = abs(dy) if ax else abs(dx)
            if not any(math.isclose(offset, d, abs_tol=1e-6) for d in valid_dists):
                raise AssertionError(
                    f"axis offset {offset} not in {sorted(valid_dists)} "
                    f"head=({hx},{hy}) cand=({x},{y})"
                )
            n_axis += 1
        else:
            # diagonal: |dx|==|dy|, and that magnitude must equal a dist value
            if not math.isclose(abs(dx), abs(dy), abs_tol=1e-6):
                raise AssertionError(
                    f"non-45° diagonal head=({hx},{hy}) cand=({x},{y}) "
                    f"dx={dx} dy={dy}"
                )
            mag = abs(dx)
            if not any(math.isclose(mag, d, abs_tol=1e-6) for d in valid_dists):
                raise AssertionError(
                    f"diagonal |dx|=|dy|={mag} not in {sorted(valid_dists)} "
                    f"head=({hx},{hy}) cand=({x},{y})"
                )
            n_diag += 1
    if n_axis != 4 * len(_DIR_DISTANCES_MM):
        raise AssertionError(f"expected {4*len(_DIR_DISTANCES_MM)} axis-aligned, got {n_axis}")
    if n_diag != 4 * len(_DIR_DISTANCES_MM):
        raise AssertionError(f"expected {4*len(_DIR_DISTANCES_MM)} diagonals, got {n_diag}")
    return n_axis, n_diag


def count_track_segments(env: PCBWorld, tol: float = 1e-6) -> tuple[int, int]:
    """Return (n_axis_seg, n_diag_seg) over engine tracks."""
    engine = getattr(env, "engine", None) or getattr(env, "_engine", None)
    if engine is None:
        return 0, 0
    try:
        tracks = engine.get_tracks()
    except Exception:
        return 0, 0
    n_axis = 0
    n_diag = 0
    for t in tracks:
        dx = t.x2_mm - t.x1_mm
        dy = t.y2_mm - t.y1_mm
        if abs(dx) < tol and abs(dy) < tol:
            continue  # zero-length, skip
        if abs(dx) < tol or abs(dy) < tol:
            n_axis += 1
        else:
            n_diag += 1
    return n_axis, n_diag


def run_one_grid(
    grid_size: int,
    board_path: Path,
    n_episodes: int,
    n_steps: int,
    seed: int,
) -> dict:
    env = PCBWorld(
        board_path=str(board_path),
        max_steps=n_steps,
        masking_rule="default",
        reward_rule="drc_only_dense",
        emit_drc_tokens=False,
    )
    wrapper = KiCadRLWrapper(
        env, seed=seed, directional_candidates=None,  # 8-direction default
    )
    rng = np.random.default_rng(seed)
    n_routing = 0
    n_total_axis = 0
    n_total_diag = 0
    seg_axis_total = 0
    seg_diag_total = 0
    seg_diag_per_ep: list[int] = []

    for ep in range(n_episodes):
        obs, _ = wrapper.reset(seed=seed + ep)
        # plumbing: ensure no grid_size leaked
        assert obs.get("_aug", {}).get("directional_candidates") is None, (
            f"_aug.directional_candidates unexpectedly set: {obs.get('_aug')}"
        )

        for step in range(n_steps):
            mask = wrapper.action_masks().astype(bool)
            valid_action_types = np.flatnonzero(mask)
            if len(valid_action_types) == 0:
                break
            at = int(rng.choice(valid_action_types))
            cand_mm = wrapper._cand_mm
            ptr = int(rng.integers(0, max(len(cand_mm), 1))) if cand_mm else 0
            mode = int(rng.integers(0, 3))
            action = np.array([at, ptr, mode], dtype=np.int64)

            rh = obs.get("router_head", {})
            if rh.get("is_routing", False):
                head_xy = tuple(rh["current_xy"])
                head_layer = int(rh["current_layer"])
                pre = directional_candidates_pre_dedup(obs)
                n_a, n_d = assert_axis_and_diagonal(head_xy, head_layer, pre)
                # post-dedup may drop some candidates colliding with pads/tracks
                post = directional_candidates_post_dedup(obs)
                if len(post) > len(pre):
                    raise AssertionError(
                        f"post-dedup directional count {len(post)} > pre-dedup {len(pre)}"
                    )
                n_routing += 1
                n_total_axis += n_a
                n_total_diag += n_d

            obs, reward, term, trunc, info = wrapper.step(action)
            if term or trunc:
                break

        # End-of-episode: count diagonal segments produced by the engine.
        ep_axis, ep_diag = count_track_segments(env)
        seg_axis_total += ep_axis
        seg_diag_total += ep_diag
        seg_diag_per_ep.append(ep_diag)

    return {
        "grid_size": grid_size,
        "n_routing_steps": n_routing,
        "cand_axis_total": n_total_axis,
        "cand_diag_total": n_total_diag,
        "seg_axis_total": seg_axis_total,
        "seg_diag_total": seg_diag_total,
        "seg_diag_per_ep": seg_diag_per_ep,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-sizes", nargs="+", type=int,
                   default=sorted(_GRID_STEP_CELLS),
                   help="Grid sizes to test. Default: all configured.")
    p.add_argument("--n-episodes", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", default=None)
    p.add_argument("--require-engine-diagonal", action="store_true",
                   help="Fail if engine never produces a 45° track segment "
                        "across all episodes for a grid.")
    args = p.parse_args()

    if args.workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="cadagent_diag_verify_"))
        print(f"[workdir] {workdir} (auto-tempdir)")
    else:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        print(f"[workdir] {workdir}")

    failures: list[str] = []
    reports: list[dict] = []
    for g in args.grid_sizes:
        try:
            board = ensure_grid_board(g, workdir)
            rep = run_one_grid(
                grid_size=g,
                board_path=board,
                n_episodes=args.n_episodes,
                n_steps=args.n_steps,
                seed=args.seed,
            )
            reports.append(rep)
            print(
                f"PASS grid_size={g:>4}: "
                f"routing_steps={rep['n_routing_steps']:>4}  "
                f"cand_axis={rep['cand_axis_total']:>5}  "
                f"cand_diag={rep['cand_diag_total']:>5}  "
                f"seg_axis={rep['seg_axis_total']:>4}  "
                f"seg_diag={rep['seg_diag_total']:>4}",
                flush=True,
            )
            if args.require_engine_diagonal and rep["seg_diag_total"] == 0:
                failures.append(
                    f"grid_size={g}: random policy produced 0 diagonal "
                    f"track segments across {args.n_episodes} episodes"
                )
        except Exception as exc:
            failures.append(f"grid_size={g}: {exc}")
            print(f"FAIL grid_size={g}: {exc}", flush=True)

    print("\n=== Summary ===")
    for r in reports:
        print(r)
    if failures:
        print("\n=== Failures ===")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll grids verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
