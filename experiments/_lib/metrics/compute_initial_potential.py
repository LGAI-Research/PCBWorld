#!/usr/bin/env python3
"""Compute `initial_potential` (Phi of the *bare* board) ONCE per unique board
and cache it, so `potential_gain = final_potential - initial_potential` can be
reported without re-running the whole dispatch.

initial_potential depends only on the board's bare (no-track) state, so it is
identical across all rollouts, seeds, and *methods* of the same board within a
namespace (= the dir that defines the board set: ``d2a``, ``d3/d3a``, ``d3/d3b``,
``d1/d1_grid<G>``). We therefore compute it once per (namespace, board_id) from
one representative routed board, via the minimal load->reset->Phi path (skips the
expensive routed DRC), in a forkserver pool (KiCad engine is a per-process
singleton; never fork/thread).

Read-only on experiments: only reads ``.kicad_pcb``/``.kicad_pro``; writes the
cache JSON under ``paper_outputs/``.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

REWARD = os.environ.get("REWARD", "drc_dense_promoted")
# namespaces whose tables need potential_gain (Table 3/22/23, Fig 9)
NAMESPACES = ["d2a", "d3/d3a", "d3/d3b",
              "d1/d1_grid10", "d1/d1_grid50", "d1/d1_grid100",
              "d1/d1_grid200", "d1/d1_grid500"]


def _representative_boards(ns: str) -> dict[str, tuple[str, str | None]]:
    """{board_id: (pcb_path, pro_path|None)} — one representative per board_id,
    unioned across all method-cells in the namespace (r00 files only)."""
    out: dict[str, tuple[str, str | None]] = {}
    ns_dir = C.EXP / ns
    if not ns_dir.is_dir():
        return out
    # Prefer native-KiCad cells (transformer/reference/krt) as the representative;
    # freerouting/ortho boards are format-converted and some fail to load with
    # "undefined layer names", so sort those last.
    def _pref(p):
        bad = ("freerouting" in p.name) or ("ortho" in p.name)
        return (bad, p.name)
    for cell_dir in sorted((p for p in ns_dir.iterdir() if (p / "boards").is_dir()), key=_pref):
        cell = cell_dir.name
        for pcb in glob.glob(str(cell_dir / "boards" / "*_s*_r00.kicad_pcb")):
            parsed = C.u.parse_cell_artifact(pcb, cell)
            if parsed is None:
                continue
            bid = parsed[0]
            if bid in out:
                continue
            pro = os.path.join(os.path.dirname(pcb), f"{bid}.kicad_pro")
            out[bid] = (pcb, pro if os.path.exists(pro) else None)
    return out


def _work(arg):
    key, pcb, pro = arg
    try:
        from pcb_world.engine.kicad_engine import KiCadEngine
        from pcb_world.core.reward import RewardState
        from pcb_world.core.reward_config import get_reward_config
        from eval.metrics import _delete_all_tracks_vias
        eng = KiCadEngine(pcb, project_path=(pro or None))
        eng.build_connectivity()
        meta = eng.get_board_meta()
        pr = get_reward_config(REWARD).build_reward()
        # Board-resolution hooks: same definition as PCBWorld / evaluate_one.
        pr.bind_board(net_count=meta.net_count, bbox_w=meta.bbox_w, bbox_h=meta.bbox_h)
        if eng.is_routing():
            eng.cancel_route()
        _delete_all_tracks_vias(eng)
        eng.build_connectivity()
        pr.bind_board(
            pad_groups=eng.get_pad_groups(),
            net_names=eng.get_net_names(),
            routable_nets=eng.get_routable_nets(),
        )
        snap = eng.get_reward_snapshot(run_drc=False)
        return key, float(pr.compute_final(RewardState.from_snapshot(snap))), ""
    except Exception as e:  # noqa: BLE001
        return key, None, f"{type(e).__name__}: {e}"


def main() -> None:
    jobs = []
    for ns in NAMESPACES:
        reps = _representative_boards(ns)
        for bid, (pcb, pro) in reps.items():
            jobs.append((f"{ns}::{bid}", pcb, pro))
        print(f"[init-pot] {ns}: {len(reps)} boards")
    print(f"[init-pot] total {len(jobs)} boards; reward={REWARD}")

    out_path = C.assert_output_path(C.PAPER_OUT / "initial_potential.json")
    cache: dict[str, float] = {}
    if out_path.exists():
        cache = json.loads(out_path.read_text())  # resume / merge
    todo = [j for j in jobs if j[0] not in cache]
    print(f"[init-pot] {len(cache)} cached, {len(todo)} to compute")

    nproc = min(16, max(1, (os.cpu_count() or 4) - 2))
    ctx = get_context("forkserver")
    done = 0
    errs = 0
    with ctx.Pool(nproc) as pool:
        for key, val, err in pool.imap_unordered(_work, todo, chunksize=4):
            done += 1
            if val is None:
                errs += 1
                if errs <= 10:
                    print(f"  [err] {key}: {err}", file=sys.stderr)
            else:
                cache[key] = val
            if done % 100 == 0:
                print(f"  ... {done}/{len(todo)}  (errs={errs})")
                out_path.write_text(json.dumps(cache, indent=0))  # checkpoint
    out_path.write_text(json.dumps(cache, indent=0))
    print(f"[init-pot] wrote {len(cache)} entries to {out_path}  (errs={errs})")


if __name__ == "__main__":
    main()
