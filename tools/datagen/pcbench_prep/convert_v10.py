#!/usr/bin/env python3
"""
Step 0 of the KiCad 10.0.7 PCBench -> exacad_sorted_v10 chain.

The public PCBench repository ships ``PCBs/<name>/processed.kicad_pcb`` in
KiCad 5 format and no project file. Load/save through the Engine's KiCad 10.0.7
``pcbnew`` build creates ``processed_v10.kicad_pcb`` and its project file in a
separate v10 output tree. Source folder names are preserved through conversion.

Each board is converted in its own child process, so a failing loader only
fails that board. ``metadata.json`` / ``final.json`` are copied alongside for
sorting.

Environment:
  PCBENCH_PCBS_ROOT  PCBench clone's ``PCBs/`` directory (input)
  PCBENCH_V10_ROOT   v10 conversion output root
  PCBNEW_PYTHON      optional interpreter importing the Engine pcbnew build
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
OUTPUT_STEM = "processed_v10"

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
    dst = out_dir / name
    if dst.exists():
        shutil.rmtree(dst)
    pcb_src = src_dir / "processed.kicad_pcb"
    if not pcb_src.is_file():
        return {"name": name, "status": "missing", "returncode": None,
                "error": f"source PCB missing: {pcb_src}"}

    dst.mkdir(parents=True)
    pcb_dst = dst / f"{OUTPUT_STEM}.kicad_pcb"
    try:
        proc = subprocess.run(
            [kicad_tools.pcbnew_python(), str(helper), str(pcb_src), str(pcb_dst)],
            capture_output=True, text=True, timeout=CONVERT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(dst, ignore_errors=True)
        return {"name": name, "status": "timeout", "returncode": None,
                "error": f"pcbnew conversion timed out after {CONVERT_TIMEOUT}s"}
    except OSError as exc:
        shutil.rmtree(dst, ignore_errors=True)
        return {"name": name, "status": "failed", "returncode": None,
                "error": f"could not start pcbnew child: {exc}"}

    missing = [str(path.name) for path in (
        pcb_dst, pcb_dst.with_suffix(".kicad_pro")) if not path.is_file()]
    if proc.returncode != 0 or missing:
        shutil.rmtree(dst, ignore_errors=True)
        detail = (proc.stderr or proc.stdout or "").strip()[-400:]
        if detail:
            error = detail
        elif proc.returncode != 0:
            error = f"pcbnew child exited with returncode {proc.returncode} without diagnostics"
        else:
            error = "pcbnew child exited successfully but omitted: " + ", ".join(missing)
        if missing:
            error += f"; missing output artifact(s): {', '.join(missing)}"
        return {"name": name, "status": "failed", "returncode": proc.returncode,
                "error": error}

    try:
        for extra in ("metadata.json", "final.json"):
            ep = src_dir / extra
            if ep.exists():
                shutil.copy(ep, dst / extra)
    except OSError as exc:
        shutil.rmtree(dst, ignore_errors=True)
        return {"name": name, "status": "failed", "returncode": proc.returncode,
                "error": f"could not copy {extra}: {exc}"}
    return {"name": name, "status": "ok", "returncode": proc.returncode}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0,
                   help="0=all, N=only the first N boards (sorted by source folder)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    pcbs_dir = _env_dir("PCBENCH_PCBS_ROOT")
    out_dir = _env_dir("PCBENCH_V10_ROOT")
    out_dir.mkdir(parents=True, exist_ok=True)
    kicad_tools.assert_all()

    samples = sorted(e for e in os.listdir(pcbs_dir) if (pcbs_dir / e).is_dir())
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise SystemExit(f"no PCBench board folders under {pcbs_dir}")
    sample_set = set(samples)
    for stale in out_dir.iterdir():
        if stale.is_dir() and stale.name not in sample_set:
            shutil.rmtree(stale)

    helper = Path(tempfile.mkdtemp(prefix="convert_v10_")) / "convert_helper.py"
    helper.write_text(CONVERT_HELPER_SRC, encoding="utf-8")
    print(f"Converting {len(samples)} samples with {args.workers} workers...")

    results, ok, done = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_sample, s, pcbs_dir, out_dir, helper): s
                for s in samples}
        for future in as_completed(futs):
            name = futs[future]
            try:
                result = future.result()
            except Exception as exc:
                shutil.rmtree(out_dir / name, ignore_errors=True)
                result = {"name": name, "status": "failed", "returncode": None,
                          "error": f"{type(exc).__name__}: {exc}"[:400]}
            results.append(result)
            if result["status"] == "ok":
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
        print(f"    {r['status']:8s} {r['name']} rc={r.get('returncode')}: {r.get('error', '')}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
