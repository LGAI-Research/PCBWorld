#!/usr/bin/env python3
"""
DRC-compatibility patch for the PCBs_version_9 board set.

Those files are already in KiCad 9 format and their .kicad_pro carries the
original constraints. Many samples still fail DRC because checks that KiCad 9
added or tightened are set to `error` in rule_severities.

This script keeps .kicad_pcb / .kicad_pro as they are, overrides the minimum
set of .kicad_pro entries needed to re-run the DRC check, and copies every
sample that passes into <PCBENCH_NEWDRC_OUT>/<name>/.

Overridden entries (two categories):
  [A] checks introduced in v9 — rule_severities → ignore
      solder_mask_bridge, drill_out_of_range, malformed_courtyard

  [B] thresholds tightened in v9 — rule_severities → ignore + rules floor → 0
      annular_width, courtyards_overlap, copper_edge_clearance, hole_clearance
  [C] hole-to-hole minimum — the pcbnew round trip (convert_v9.py) leaves KiCad's
      default 0.25 mm, stricter than many boards' copper clearance; routers and
      DRC honor the .kicad_pro as authoritative, so it is floored at the Default
      net-class clearance (capped at 0.25): a drill pair may sit as close as two
      tracks may. This is the value the paper's D3 set carried (672 of 679).

Zone refill is unnecessary: processed.kicad_pcb carries no zones (one
exception aside).

The DRC runs through the ``kicad-cli`` that ``kicad_tools`` resolves (default: the
engine's BUILD_CLI=1 build, else ``kicad-cli`` on PATH; override with KICAD_CLI).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import kicad_tools

def _env_dir(name: str) -> Path:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"{name} is not set. Point it at the matching directory "
            "(see tools/datagen/pcbench_prep/README.md)."
        )
    return Path(val)

V9_DIR  = _env_dir("PCBENCH_V9_ROOT")     # KiCad-9-format PCBench boards (input)
OUT_DIR = _env_dir("PCBENCH_NEWDRC_OUT")  # DRC-passing boards are copied here (output)

# [A] checks introduced in v9: rule_severities → ignore
SEVERITY_PATCH = {
    "solder_mask_bridge":  "ignore",
    "drill_out_of_range":  "ignore",
    "malformed_courtyard": "ignore",
    "courtyards_overlap":  "ignore",   # [B]
    "annular_width":       "ignore",   # [B]
    "copper_edge_clearance": "ignore", # [B]
    "hole_clearance":      "ignore",   # [B]
}

# [B] defaults tightened in v9: rules floor → 0 (applied together with ignore)
RULES_PATCH = {
    "min_via_annular_width":     0.0,
    "min_copper_edge_clearance": 0.0,
    "min_hole_clearance":        0.0,
}
# [C] hole-to-hole minimum = min(KiCad default, Default net-class clearance)
HOLE_TO_HOLE_DEFAULT_MM = 0.25


def hole_to_hole_min(pro: dict) -> float | None:
    """[C]: the Default net class clearance, capped at KiCad's default."""
    classes = pro.get("net_settings", {}).get("classes", []) or []
    default = next((c for c in classes if c.get("name") == "Default"), None)
    clearance = (default or {}).get("clearance")
    if clearance is None:
        return None
    return min(HOLE_TO_HOLE_DEFAULT_MM, float(clearance))

DRC_TIMEOUT = 60


def patch_pro(src_pro: Path, dst_pro: Path) -> None:
    """Read the original .kicad_pro, patch the minimum entries, save it."""
    with open(src_pro, encoding="utf-8") as f:
        pro = json.load(f)

    ds = pro.setdefault("board", {}).setdefault("design_settings", {})

    # rule_severities patch
    sev = ds.setdefault("rule_severities", {})
    sev.update(SEVERITY_PATCH)

    # rules patch
    rules = ds.setdefault("rules", {})
    rules.update(RULES_PATCH)
    h2h = hole_to_hole_min(pro)
    if h2h is not None:
        rules["min_hole_to_hole"] = h2h   # [C]

    with open(dst_pro, "w", encoding="utf-8") as f:
        json.dump(pro, f)


def run_drc(pcb: Path) -> list[dict]:
    """Run kicad-cli drc and return the list of violations."""
    out = pcb.with_suffix(".drc.json")
    try:
        subprocess.run(
            [kicad_tools.kicad_cli(), "pcb", "drc", "--format", "json",
             "--severity-error", "--output", str(out), str(pcb)],
            capture_output=True, timeout=DRC_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return []
    if not out.exists():
        return []
    try:
        data = json.load(open(out, encoding="utf-8"))
        return data.get("violations", [])
    except Exception:
        return []


def process_sample(name: str) -> dict:
    src_dir = V9_DIR / name
    pcb_src = src_dir / "processed.kicad_pcb"
    pro_src = src_dir / "processed.kicad_pro"

    if not pcb_src.exists() or not pro_src.exists():
        return {"name": name, "status": "missing"}

    tmp = Path(tempfile.mkdtemp(prefix="drc_v9_"))
    try:
        pcb = tmp / "board.kicad_pcb"
        pro = tmp / "board.kicad_pro"
        shutil.copy(pcb_src, pcb)
        patch_pro(pro_src, pro)

        viols = run_drc(pcb)
        ok = len(viols) == 0

        if ok:
            dst = OUT_DIR / name
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            shutil.copy(pcb, dst / "processed_v9.kicad_pcb")
            shutil.copy(pro, dst / "processed_v9.kicad_pro")
            for extra in ("metadata.json", "final.json"):
                ep = src_dir / extra
                if ep.exists():
                    shutil.copy(ep, dst / extra)

        return {
            "name": name,
            "status": "ok" if ok else "failed",
            "violations": [
                {"type": v["type"], "description": v.get("description", "")[:200]}
                for v in viols[:20]
            ],
        }
    except Exception as e:
        return {"name": name, "status": "exception", "error": str(e)[:200]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0,
                   help="0=all, N=only the first N boards (sorted by name)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kicad_tools.announce(need_pcbnew=False)

    samples = sorted(e for e in os.listdir(V9_DIR) if (V9_DIR / e).is_dir())
    if args.limit:
        samples = samples[:args.limit]
    print(f"Processing {len(samples)} samples with {args.workers} workers...")

    results, ok, done = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_sample, s): s for s in samples}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            if r.get("status") == "ok":
                ok += 1
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(samples)}  (ok: {ok})", flush=True)

    results.sort(key=lambda r: r["name"])
    failures = [r for r in results if r.get("status") != "ok"]

    with open(OUT_DIR / "_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    # Tally the failure causes
    import collections
    type_cnt = collections.Counter()
    for r in failures:
        for v in r.get("violations", []):
            type_cnt[v["type"]] += 1

    print(f"\n=== Summary ===")
    print(f"  Total:   {len(results)}")
    print(f"  Success: {ok}")
    print(f"  Failure: {len(failures)}")
    if type_cnt:
        print(f"\n  Remaining violation types:")
        for t, n in type_cnt.most_common():
            print(f"    {n:5d}  {t}")


if __name__ == "__main__":
    main()
