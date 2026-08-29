"""Pre-route synth_2L val boards to use as few-shot examples.

The synth_2L_v2 val/test boards ship unrouted, so they can't be used as
``(input PCB, output ROUTING)`` few-shot pairs as-is. This script walks
the C++ PNS router across each val board, drops segments + vias, and
saves the routed copy + ``.kicad_pro`` companion to a cache directory.

Routing strategy (deterministic, per board):
    1. group pads by net_code (skip net 0 = no-net)
    2. for each net's pads, route a chain pad[0] -> pad[1] -> pad[2] -> ...
       using ``KiCadEngine.start_route`` + ``fix_route``.
    3. when consecutive pads are on different layers, drop a via at the
       second pad (``toggle_via`` + ``switch_layer`` mid-route) so the
       ratsnest closes with proper electrical continuity.

Boards that fail mid-routing keep whatever was already laid down — the
PNS-generated segments remain valid even when later nets fail. Quality
isn't critical here; the goal is to demonstrate the *format* of routed
KiCad sexpr (proper segment/via syntax, layer references, net codes)
to the LLM.

Usage:
    python scripts/prepare_synth_fewshot.py \
        $CADAGENT_DATA_ROOT/synthetic/synth_2L_v2/val \
        -o cache/synth_2L_fewshot/ \
        --limit 8

    # Then point the few-shot pool at the cache dir:
    bash scripts/run_cadgen_llm.sh synth_fs        # auto-uses cache
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent.parent.parent.parent  # llm_eval→paper_repro→scripts→repo
_KICAD_RL_DIR = _THIS_DIR / "build_rl" / "pcbnew" / "python" / "rl"
for p in (_THIS_DIR, _KICAD_RL_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _route_net(engine, pads) -> tuple[int, int]:
    """Route a single net's pads as a chain. Returns (segments_added, vias_added)."""
    initial_tracks = len(engine.get_tracks())
    initial_vias = len(engine.get_vias())

    for i in range(len(pads) - 1):
        a, b = pads[i], pads[i + 1]
        a_layer = engine._b2h(a.layer)
        b_layer = engine._b2h(b.layer)
        if a_layer < 1 or b_layer < 1:
            # Multi-layer (THT) pad — accessible from any copper layer; pick L1.
            a_layer = a_layer if a_layer >= 1 else 1
            b_layer = b_layer if b_layer >= 1 else a_layer

        try:
            if not engine.start_route(a.x_mm, a.y_mm, a_layer):
                continue

            if a_layer == b_layer:
                engine.fix_route(b.x_mm, b.y_mm, force_finish=True)
            else:
                # Drive a head to b's xy on the start layer, drop a via,
                # switch, then commit at b on the destination layer. The
                # PNS engine handles the via geometry once toggle_via +
                # switch_layer are called between move and fix.
                engine.move(b.x_mm, b.y_mm)
                engine.toggle_via()
                engine.switch_layer(b_layer)
                engine.fix_route(b.x_mm, b.y_mm, force_finish=True)
        except Exception:
            try:
                engine.cancel_route()
            except Exception:
                pass

    return (
        len(engine.get_tracks()) - initial_tracks,
        len(engine.get_vias()) - initial_vias,
    )


def route_board(src: Path, dst: Path) -> dict:
    """Open ``src`` with KiCadEngine, PNS-route every net, save to ``dst``.

    Returns a dict with per-net stats and the final ratsnest count.
    """
    from pcb_world.engine.kicad_engine import KiCadEngine

    engine = KiCadEngine(str(src))
    try:
        engine.build_connectivity()

        nets: dict[int, list] = defaultdict(list)
        for p in engine.get_pads():
            nets[p.net_code].append(p)

        per_net = {}
        for nc, pads in sorted(nets.items()):
            if nc == 0 or len(pads) < 2:
                continue
            seg, via = _route_net(engine, pads)
            per_net[nc] = {"pads": len(pads), "segments": seg, "vias": via}

        engine.build_connectivity()
        ratsnest = len(engine.get_ratsnest())
        track_count = len(engine.get_tracks())
        via_count = len(engine.get_vias())

        dst.parent.mkdir(parents=True, exist_ok=True)
        engine.save(str(dst))
    finally:
        # Singleton constraint: drop the engine before the next board's
        # KiCadEngine constructor.
        try:
            if engine.is_routing():
                engine.cancel_route()
        except Exception:
            pass

    return {
        "src": str(src),
        "dst": str(dst),
        "per_net": per_net,
        "track_count": track_count,
        "via_count": via_count,
        "ratsnest_remaining": ratsnest,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src_dir", type=Path,
                   help="Directory of unrouted .kicad_pcb (e.g. synth_2L_v2/val).")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="Cache dir for routed copies.")
    p.add_argument("--limit", type=int, default=8,
                   help="Pre-route only the first N boards (default 8).")
    p.add_argument("-f", "--force", action="store_true",
                   help="Overwrite existing routed cache files.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.src_dir.is_dir():
        print(f"[ERROR] not a directory: {args.src_dir}", file=sys.stderr)
        return 2

    boards = sorted(args.src_dir.glob("*.kicad_pcb"))[: args.limit]
    if not boards:
        print(f"[ERROR] no .kicad_pcb under {args.src_dir}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"  source : {args.src_dir}")
    print(f"  out    : {args.output}")
    print(f"  boards : {len(boards)}")
    print()

    n_ok = n_skip = n_fail = 0
    for src in boards:
        dst = args.output / src.name
        dst_pro = dst.with_suffix(".kicad_pro")
        if dst.exists() and dst_pro.exists() and not args.force:
            print(f"  [skip] {dst.name} (use --force to overwrite)")
            n_skip += 1
            continue
        try:
            stats = route_board(src, dst)
        except Exception as exc:
            traceback.print_exc()
            print(f"  [FAIL] {src.name}: {type(exc).__name__}: {exc}")
            n_fail += 1
            continue

        # Copy the source's .kicad_pro across so KiCadEngine.save's auto-pro
        # doesn't drift from the val source's design rules. KiCadEngine.save
        # already emits a .kicad_pro, but we mirror the val pro file so all
        # downstream consumers keep BDS / NetSettings identical.
        src_pro = src.with_suffix(".kicad_pro")
        if src_pro.exists():
            shutil.copyfile(src_pro, dst_pro)

        per_net = stats["per_net"]
        n_segs = sum(d["segments"] for d in per_net.values())
        n_vias = sum(d["vias"] for d in per_net.values())
        print(
            f"  [ok] {src.name}: nets={len(per_net)} "
            f"tracks={stats['track_count']} ({n_segs} new) "
            f"vias={stats['via_count']} ({n_vias} new) "
            f"ratsnest_left={stats['ratsnest_remaining']}"
        )
        if args.verbose:
            for nc, d in per_net.items():
                print(f"    net {nc}: pads={d['pads']} +seg={d['segments']} +via={d['vias']}")
        n_ok += 1

    print()
    print(f"Routed: {n_ok}  Skipped: {n_skip}  Failed: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
