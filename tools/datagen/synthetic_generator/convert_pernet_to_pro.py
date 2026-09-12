"""Convert per-net synthetic boards (embedded legacy net_class blocks) to the
modern KiCad format: strip the legacy blocks from each .kicad_pcb and emit a
sibling .kicad_pro carrying the per-net rules as net_settings.classes +
netclass_patterns.

Unlike ``migrate_dataset_to_pro.py`` (which derives ONE shared rules template
from the default netclass and reuses it for every board — correct only when all
boards share rules), this runs a PER-BOARD engine round-trip:
``KiCadEngine(board).save(board)``. The engine writes a .kicad_pro whose
net_settings.classes/patterns preserve each net's own clearance/width/via
(verified: get_netclass_for_net round-trips per-net). Required for D2-B-V, where
every board — and every net within a board — has different rules.

Usage:
    python tools/datagen/synthetic_generator/convert_pernet_to_pro.py --src DIR [--workers N]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path


def _convert_one(pcb_path_str: str) -> tuple[str, bool, str]:
    # Engine is a per-process C++ singleton, so one board at a time per worker.
    try:
        # Raw-router primitive, NOT KiCadEngine: the wrapper's strict load
        # contract refuses exactly the pro-less pernet boards this tool
        # exists to convert (see load_and_save_via_engine's docstring).
        from pcb_world.engine.utils import load_and_save_via_engine
        load_and_save_via_engine(pcb_path_str, pcb_path_str)
        ok = Path(pcb_path_str).with_suffix(".kicad_pro").is_file()
        return (pcb_path_str, ok, "" if ok else "no .kicad_pro written")
    except Exception as exc:  # noqa: BLE001
        return (pcb_path_str, False, repr(exc))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="dir of board_*.kicad_pcb")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="convert only files whose sorted index ≡ I (mod N) — "
                        "for multi-host splits over one shared dir. The file "
                        "set must be complete before any shard starts.")
    args = p.parse_args()

    src = Path(args.src)
    boards = sorted(str(p) for p in src.glob("board_*.kicad_pcb"))
    if not boards:
        raise SystemExit(f"[error] no board_*.kicad_pcb under {src}")
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"[error] bad --shard {args.shard!r}")
        boards = boards[i::n]
        print(f"[convert] shard {i}/{n}: {len(boards)} boards")
    print(f"[convert] {len(boards)} boards in {src}  ({args.workers} workers, per-board round-trip)")

    fails = []
    with mp.Pool(args.workers) as pool:
        for i, (pcb, ok, err) in enumerate(pool.imap_unordered(_convert_one, boards, chunksize=8)):
            if not ok:
                fails.append((pcb, err))
            if (i + 1) % max(1, len(boards) // 20) == 0 or i + 1 == len(boards):
                print(f"  {i+1}/{len(boards)}  (failures: {len(fails)})")

    # Retry pass. At high worker counts a few boards intermittently come back
    # "no .kicad_pro written" (observed 16/100k at 48 workers; every one of them
    # converts fine on a second, less contended attempt). Retry once with a
    # small pool before failing the run — a 100k pipeline aborting on 0.02%
    # transient write misses costs more than one extra pass. Still exits 1 if
    # anything survives the retry: no silent partial datasets.
    if fails:
        retry = [pcb for pcb, _ in fails]
        print(f"[convert] retrying {len(retry)} failure(s) with 4 workers")
        fails = []
        with mp.Pool(min(4, len(retry))) as pool:
            for pcb, ok, err in pool.imap_unordered(_convert_one, retry):
                if not ok:
                    fails.append((pcb, err))
        print(f"[convert] retry done: {len(retry) - len(fails)} recovered, "
              f"{len(fails)} still failing")

    # Count pros for THIS run's board set only — a directory-wide glob would
    # also count other shards' outputs under --shard.
    npro = sum(1 for b in boards if Path(b).with_suffix(".kicad_pro").is_file())
    print(f"[convert] done: {len(boards)} pcb, {npro} pro, {len(fails)} failures")
    if fails:
        for pcb, err in fails[:5]:
            print(f"  FAIL {pcb}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
