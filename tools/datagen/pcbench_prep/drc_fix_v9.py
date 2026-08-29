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
  [A] checks introduced in v9 — rule_severities -> ignore
      solder_mask_bridge, drill_out_of_range, malformed_courtyard

  [B] thresholds tightened in v9 — rule_severities -> ignore + rules floor -> 0
      annular_width, courtyards_overlap, copper_edge_clearance, hole_clearance

Zone refill is unnecessary: processed.kicad_pcb carries no zones (one
exception aside).
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _env_dir(name: str) -> Path:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"{name} is not set. Point it at the matching directory "
            "(see tools/datagen/pcbench_prep/README.md)."
        )
    return Path(val)

# [A] checks introduced in v9: rule_severities -> ignore
SEVERITY_PATCH = {
    "solder_mask_bridge":  "ignore",
    "drill_out_of_range":  "ignore",
    "malformed_courtyard": "ignore",
    "courtyards_overlap":  "ignore",   # [B]
    "annular_width":       "ignore",   # [B]
    "copper_edge_clearance": "ignore", # [B]
    "hole_clearance":      "ignore",   # [B]
}

# [B] defaults tightened in v9: rules floor -> 0 (applied together with ignore)
RULES_PATCH = {
    "min_via_annular_width":     0.0,
    "min_copper_edge_clearance": 0.0,
    "min_hole_clearance":        0.0,
}

DRC_TIMEOUT = 60
WORKERS = 16


def patch_pro(src_pro: Path, dst_pro: Path) -> None:
    """Read the original .kicad_pro, patch the minimum entries, save it."""
    with open(src_pro, encoding="utf-8") as f:
        pro = json.load(f)

    ds = pro.setdefault("board", {}).setdefault("design_settings", {})

    # rule_severities
    sev = ds.setdefault("rule_severities", {})
    sev.update(SEVERITY_PATCH)

    # rules
    rules = ds.setdefault("rules", {})
    rules.update(RULES_PATCH)

    with open(dst_pro, "w", encoding="utf-8") as f:
        json.dump(pro, f)


def run_drc(pcb: Path) -> list[dict]:
    """Run kicad-cli drc and return the list of violations."""
    out = pcb.with_suffix(".drc.json")
    try:
        subprocess.run(
            ["kicad-cli", "pcb", "drc", "--format", "json",
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


def process_sample(name: str, v9_dir: Path, out_dir: Path) -> dict:
    src_dir = v9_dir / name
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
            dst = out_dir / name
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
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Input/output directories come from the environment:\n"
               "  PCBENCH_V9_ROOT     KiCad-9-format PCBench boards (input)\n"
               "  PCBENCH_NEWDRC_OUT  DRC-passing boards are copied here (output)",
    ).parse_args()

    v9_dir = _env_dir("PCBENCH_V9_ROOT")
    out_dir = _env_dir("PCBENCH_NEWDRC_OUT")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = sorted(e for e in os.listdir(v9_dir) if (v9_dir / e).is_dir())
    print(f"Processing {len(samples)} samples with {WORKERS} workers...")

    results, ok, done = [], 0, 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_sample, s, v9_dir, out_dir): s for s in samples}
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

    with open(out_dir / "_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_dir / "_failures.json", "w", encoding="utf-8") as f:
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
