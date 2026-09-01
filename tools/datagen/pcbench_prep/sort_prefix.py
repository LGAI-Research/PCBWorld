#!/usr/bin/env python3
"""
Step 3 of the PCBench -> exacad_sorted chain: difficulty sort + 4-digit prefix
layout.

Reads the board folders produced by steps 1-2, computes the statistics PCBench
reports in its characteristics table, sorts, and copies every folder to
``<out>/<NNNN>_<name>/`` in that order (index from 0001), writing
``pcb_characteristics_exacad_sorted.csv`` (``sample,nets,components,pins,layers``)
alongside — the layout ``experiments/kdd/d3_dataset/build.py`` consumes.

Statistics (verified 1182/1182 against PCBench's own ``pcb_characteristics.csv``):
  nets        number of nets in ``final.json``
  components  number of footprints in ``processed_v9.kicad_pcb``
  pins        total pads listed under the ``final.json`` nets
  layers      ``len(final.json["layers"])``
Sort key: ``(pins, nets, components)`` ascending.

The output directory's ``<NNNN>_<name>/`` entries are owned by this step: a re-run
replaces them and removes the ones an earlier run left (a full run after a
``--limit`` trial of the chain, for instance).

Environment:
  PCBENCH_NEWDRC_OUT  board folders from drc_fix_v9.py (+ make_guide.py), each
                      with ``processed_v9.kicad_pcb`` and ``final.json`` (input)
  PCBENCH_SORTED_OUT  the exacad_sorted directory to write (output)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

CSV_NAME = "pcb_characteristics_exacad_sorted.csv"
FOOTPRINT_RE = re.compile(r"^\t\(footprint ", re.MULTILINE)
PREFIXED_RE = re.compile(r"^\d{4}_")


def _env_dir(name: str) -> Path:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"{name} is not set. Point it at the matching directory "
            "(see tools/datagen/pcbench_prep/README.md)."
        )
    return Path(val)


def board_stats(folder: Path) -> dict | None:
    final = folder / "final.json"
    pcb = folder / "processed_v9.kicad_pcb"
    if not final.exists() or not pcb.exists():
        return None
    data = json.loads(final.read_text(encoding="utf-8"))
    nets = data["nets"]
    return {
        "sample": folder.name,
        "nets": len(nets),
        "components": len(FOOTPRINT_RE.findall(pcb.read_text(encoding="utf-8"))),
        "pins": sum(len(pads) for pads in nets.values()),
        "layers": len(data["layers"]),
    }


def main():
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()

    src_dir = _env_dir("PCBENCH_NEWDRC_OUT")
    out_dir = _env_dir("PCBENCH_SORTED_OUT")
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted(f for f in src_dir.iterdir() if f.is_dir())
    rows, skipped = [], []
    for f in folders:
        s = board_stats(f)
        (rows if s else skipped).append(s or f.name)
    rows.sort(key=lambda r: (r["pins"], r["nets"], r["components"]))
    if skipped:
        print(f"Skipped {len(skipped)} folders without final.json / "
              f"processed_v9.kicad_pcb: {skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
    if not rows:
        # Never touch the output on a degenerate input (e.g. PCBENCH_NEWDRC_OUT pointed
        # at the wrong tree): sweeping stale entries with zero rows would wipe the set.
        raise SystemExit(f"no usable board folders under {src_dir} — output left untouched")

    written = set()
    for i, r in enumerate(rows, start=1):
        dst = out_dir / f"{i:04d}_{r['sample']}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_dir / r["sample"], dst)
        written.add(dst.name)
    stale = [d for d in out_dir.iterdir()
             if d.is_dir() and PREFIXED_RE.match(d.name) and d.name not in written]
    for d in stale:
        shutil.rmtree(d)
    if stale:
        print(f"Removed {len(stale)} <NNNN>_<name>/ folders left by an earlier run")

    with open(out_dir / CSV_NAME, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample", "nets", "components", "pins", "layers"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} boards to {out_dir} (+ {CSV_NAME})")


if __name__ == "__main__":
    main()
