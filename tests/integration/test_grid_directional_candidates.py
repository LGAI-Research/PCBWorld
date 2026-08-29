"""Verify grid-size-aware directional candidates under uniform random policy.

For each grid_size in {10, 30, 100, 300, 1000}:
  1. Generate (or reuse) a tiny 1-layer grid dataset via the synthetic
     generator (1 board, --num-layers 1).
  2. Wrap PCBWorld with KiCadRLWrapper(directional_candidates="grid<N>").
  3. Run uniform-random actions for several steps × episodes.
  4. Whenever the env is mid-route (is_routing == True), assert that:
       - the count of CTYPE_DIRECTIONAL candidates equals the configured
         per-grid count (4 / 8 / 8 / 12 / 12),
       - each directional candidate's offset from the route head is an
         integer multiple of grid_spacing = 100 mm / N (within 1e-6 mm),
       - candidates are axis-aligned (one of dx/dy is exactly 0).

Run directly on a host with the C++ router built. Do NOT submit via a job scheduler.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pcb_world.core.env import PCBWorld  # noqa: E402
from pcb_world.vec.candidate_pool import (  # noqa: E402
    CTYPE_DIRECTIONAL,
    _BOARD_SIZE_MM,
    _GRID_STEP_CELLS,
    collect_raw_candidates,
)
from methods.rl_agent.wrappers.adapter import (  # noqa: E402
    KiCadRLWrapper,
)

GRID_EXPECTED_COUNT = {10: 4, 30: 8, 100: 8, 300: 12, 1000: 12}


def ensure_grid_board(grid_size: int, out_root: Path) -> Path:
    """Generate (or reuse) a single 1-layer grid board for the given grid size.

    Returns path to the .kicad_pcb file.
    """
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
        "--fixed-pads-per-net", "2,2,2",  # 3 nets × 2 pins → small board
        "--central-frac", "1.0",
        "--via-dia", "0.6",
        "--via-drill", "0.3",
        "--out-dir", str(out_dir),
    ]
    print(f"[gen grid={grid_size}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    if not pcb.is_file():
        raise RuntimeError(
            f"Expected board at {pcb} but generator did not produce it."
        )
    return pcb


def directional_candidates_pre_dedup(obs: dict) -> list[tuple[float, float, int]]:
    """Build directional candidates from obs WITHOUT going through the
    pad/track dedup step (collect_raw_candidates may drop a directional
    point that coincides with a pad/track endpoint at fine grid spacings,
    e.g. grid_size=1000 where 0.1mm steps frequently land on pad centers).

    For verifying grid alignment we want the raw output of
    ``build_directional_candidates`` itself.
    """
    rh = obs.get("router_head", {})
    if not rh.get("is_routing", False):
        return []
    aug = obs.get("_aug") or {}
    head_xy = rh.get("current_xy", [0.0, 0.0])
    layer = int(rh.get("current_layer", 1))

    from pcb_world.vec.candidate_pool import build_directional_candidates
    raw = build_directional_candidates(
        (head_xy[0], head_xy[1]), layer,
        mode=aug.get("directional_candidates"),
    )
    return [(x, y, ly) for (x, y, ly, ct) in raw if ct == CTYPE_DIRECTIONAL]


def directional_candidates_post_dedup(obs: dict) -> list[tuple[float, float, int]]:
    """Same as pre-dedup but routed through collect_raw_candidates so
    candidates colliding with pads / tracks / vias are dropped. Used to
    sanity-check that post-dedup count is in [1, expected]."""
    rh = obs.get("router_head", {})
    if not rh.get("is_routing", False):
        return []
    aug = obs.get("_aug") or {}
    head_xy = rh.get("current_xy", [0.0, 0.0])
    layer = int(rh.get("current_layer", 1))
    current_net_id = rh.get("current_net", -1)
    if current_net_id is None or current_net_id <= 0:
        current_net_id = None

    from pcb_world.vec.candidate_pool import build_directional_candidates
    extra = build_directional_candidates(
        (head_xy[0], head_xy[1]), layer,
        mode=aug.get("directional_candidates"),
    )
    raw = collect_raw_candidates(obs, current_net_id, extra)
    return [(x, y, ly) for (x, y, ly, ct) in raw if ct == CTYPE_DIRECTIONAL]


def assert_grid_aligned(
    head_xy: tuple[float, float],
    head_layer: int,
    dir_cands: list[tuple[float, float, int]],
    grid_size: int,
    tol: float = 1e-6,
) -> None:
    """Raise AssertionError if directional candidates violate grid invariants."""
    spacing = _BOARD_SIZE_MM / grid_size
    expected = GRID_EXPECTED_COUNT[grid_size]
    if len(dir_cands) != expected:
        raise AssertionError(
            f"grid_size={grid_size}: expected {expected} directional "
            f"candidates, got {len(dir_cands)}"
        )
    hx, hy = head_xy
    for x, y, layer in dir_cands:
        if layer != head_layer:
            raise AssertionError(
                f"grid_size={grid_size}: directional candidate layer "
                f"{layer} != head layer {head_layer}"
            )
        dx, dy = x - hx, y - hy
        # Axis-aligned: one of dx/dy must be ~0
        if not (abs(dx) < tol or abs(dy) < tol):
            raise AssertionError(
                f"grid_size={grid_size}: non-axis-aligned candidate "
                f"head=({hx},{hy}) cand=({x},{y}) dx={dx} dy={dy}"
            )
        nonzero = dx if abs(dx) >= abs(dy) else dy
        ratio = nonzero / spacing
        if not math.isclose(ratio, round(ratio), abs_tol=1e-6):
            raise AssertionError(
                f"grid_size={grid_size}: offset {nonzero:.9f} not on grid "
                f"spacing {spacing:.9f} (ratio={ratio})"
            )


def run_one_grid(
    grid_size: int,
    board_path: Path,
    n_episodes: int,
    n_steps: int,
    seed: int,
) -> dict:
    """Run uniform-random policy and verify grid alignment.

    Returns a small report dict.
    """
    env = PCBWorld(
        board_path=str(board_path),
        max_steps=n_steps,
        masking_rule="default",
        reward_rule="drc_only_dense",
        emit_drc_tokens=False,
    )
    wrapper = KiCadRLWrapper(
        env, seed=seed, directional_candidates=f"grid{grid_size}",
    )
    rng = np.random.default_rng(seed)
    n_routing = 0
    n_total_dir = 0

    from pcb_world.core.masking import NUM_ACTIONS

    for ep in range(n_episodes):
        obs, _ = wrapper.reset(seed=seed + ep)

        # plumbing assertion
        assert obs.get("_aug", {}).get("directional_candidates") == f"grid{grid_size}", (
            f"_aug.directional_candidates not propagated: "
            f"{obs.get('_aug')}"
        )

        for step in range(n_steps):
            mask = wrapper.action_masks().astype(bool)
            valid_action_types = np.flatnonzero(mask)
            if len(valid_action_types) == 0:
                break
            at = int(rng.choice(valid_action_types))
            # Build a (3,) action; choose pointer / mode at random within
            # plausible range. The wrapper rejects out-of-range pointers,
            # so we sample within [0, n_cands) when relevant.
            cand_mm = wrapper._cand_mm  # internal cache, populated in cache refresh
            ptr = int(rng.integers(0, max(len(cand_mm), 1))) if cand_mm else 0
            mode = int(rng.integers(0, 3))
            action = np.array([at, ptr, mode], dtype=np.int64)

            # Pre-step verification: directional candidates emitted in
            # current obs must satisfy invariants.
            rh = obs.get("router_head", {})
            if rh.get("is_routing", False):
                head_xy = tuple(rh["current_xy"])
                head_layer = int(rh["current_layer"])
                pre = directional_candidates_pre_dedup(obs)
                assert_grid_aligned(
                    head_xy, head_layer, pre, grid_size,
                )
                # Post-dedup may drop candidates that coincide with pads/
                # tracks (still valid behavior). Just assert it's a
                # non-strict subset count-wise.
                post = directional_candidates_post_dedup(obs)
                if len(post) > len(pre):
                    raise AssertionError(
                        f"grid_size={grid_size}: post-dedup directional count "
                        f"{len(post)} > pre-dedup {len(pre)}"
                    )
                n_routing += 1
                n_total_dir += len(pre)

            obs, reward, term, trunc, info = wrapper.step(action)
            if term or trunc:
                break

    return {
        "grid_size": grid_size,
        "n_routing_steps": n_routing,
        "n_directional_total": n_total_dir,
        "expected_count_per_step": GRID_EXPECTED_COUNT[grid_size],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-sizes", nargs="+", type=int,
                   default=sorted(_GRID_STEP_CELLS),
                   help="Grid sizes to verify. Default: all configured.")
    p.add_argument("--n-episodes", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", default=None,
                   help="Directory for generated boards (default: tempdir).")
    args = p.parse_args()

    if args.workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="cadagent_grid_verify_"))
        print(f"[workdir] {workdir} (auto-tempdir)")
    else:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        print(f"[workdir] {workdir}")

    failures: list[str] = []
    reports: list[dict] = []
    for g in args.grid_sizes:
        if g not in _GRID_STEP_CELLS:
            failures.append(f"grid_size={g}: not configured")
            continue
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
                f"PASS grid_size={g:>4}: routing_steps={rep['n_routing_steps']:>4}"
                f" total_dir_cands={rep['n_directional_total']:>5}"
                f" (per-step expected {rep['expected_count_per_step']})",
                flush=True,
            )
        except Exception as exc:  # capture and continue per grid
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
