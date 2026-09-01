#!/usr/bin/env python3
"""
Step 0 of the PCBench -> exacad_sorted chain: re-save the PCBench boards in the
KiCad 9 file format.

The public PCBench repository ships ``PCBs/<name>/processed.kicad_pcb`` in the
KiCad 5 format (file version 20171130) and no project file. ``drc_fix_v9.py``
needs the KiCad 9 form — ``processed.kicad_pcb`` + ``processed.kicad_pro``, the
project file that KiCad 9 fills from the legacy ``(setup ...)`` block on load.
Loading each board with the ``pcbnew`` Python module and saving it again performs
exactly that migration (the round trip the KiCad GUI does on "Save").

Every board is converted in its own ``pcbnew`` child process, so a board that
crashes the loader only fails itself. ``metadata.json`` / ``final.json`` are copied
alongside — ``sort_prefix.py`` reads the board statistics from ``final.json``.

KiCad assigns fresh UUIDs to every item during this conversion, so two runs are
not byte-identical; geometry, nets and design settings are.

Environment:
  PCBENCH_PCBS_ROOT  the ``PCBs/`` directory of a PCBench clone (input)
  PCBENCH_V9_ROOT    output root, ``<name>/processed.kicad_pcb`` + ``.kicad_pro``
  PCBNEW_PYTHON      python that imports ``pcbnew`` — resolved by ``kicad_tools``
                     (default: the engine's BUILD_PCBNEW=1 build, else /usr/bin/python3)
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import kicad_tools

CONVERT_TIMEOUT = 300

CONVERT_HELPER_SRC = """
import sys, pcbnew
src, dst = sys.argv[1], sys.argv[2]
board = pcbnew.LoadBoard(src)
pcbnew.SaveBoard(dst, board)
"""


def _env_dir(name: str) -> Path:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"{name} is not set. Point it at the matching directory "
            "(see tools/datagen/pcbench_prep/README.md)."
        )
    return Path(val)


def convert_sample(name: str, pcbs_dir: Path, out_dir: Path, helper: Path) -> dict:
    src_dir = pcbs_dir / name
    pcb_src = src_dir / "processed.kicad_pcb"
    if not pcb_src.exists():
        return {"name": name, "status": "missing"}

    dst = out_dir / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    pcb_dst = dst / "processed.kicad_pcb"
    try:
        proc = subprocess.run(
            [kicad_tools.pcbnew_python(), str(helper), str(pcb_src), str(pcb_dst)],
            capture_output=True, text=True, timeout=CONVERT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(dst, ignore_errors=True)
        return {"name": name, "status": "timeout"}

    if proc.returncode != 0 or not pcb_dst.exists() \
            or not pcb_dst.with_suffix(".kicad_pro").exists():
        shutil.rmtree(dst, ignore_errors=True)
        return {"name": name, "status": "failed",
                "error": (proc.stderr or proc.stdout)[-400:]}

    for extra in ("metadata.json", "final.json"):
        ep = src_dir / extra
        if ep.exists():
            shutil.copy(ep, dst / extra)
    return {"name": name, "status": "ok"}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0,
                   help="0=all, N=only the first N boards (sorted by name)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    pcbs_dir = _env_dir("PCBENCH_PCBS_ROOT")
    out_dir = _env_dir("PCBENCH_V9_ROOT")
    out_dir.mkdir(parents=True, exist_ok=True)
    kicad_tools.announce()

    helper = Path(tempfile.mkdtemp(prefix="convert_v9_")) / "convert_helper.py"
    helper.write_text(CONVERT_HELPER_SRC, encoding="utf-8")

    samples = sorted(e for e in os.listdir(pcbs_dir) if (pcbs_dir / e).is_dir())
    if args.limit:
        samples = samples[:args.limit]
    print(f"Converting {len(samples)} samples with {args.workers} workers...")

    results, ok, done = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_sample, s, pcbs_dir, out_dir, helper): s
                for s in samples}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            if r["status"] == "ok":
                ok += 1
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(samples)}  (ok: {ok})", flush=True)

    shutil.rmtree(helper.parent, ignore_errors=True)
    results.sort(key=lambda r: r["name"])
    failures = [r for r in results if r["status"] != "ok"]
    with open(out_dir / "_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(f"  Total:   {len(results)}")
    print(f"  Success: {ok}")
    print(f"  Failure: {len(failures)}")
    for r in failures[:20]:
        print(f"    {r['status']:8s} {r['name']}")


if __name__ == "__main__":
    main()
