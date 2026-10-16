#!/usr/bin/env python3
"""DRC compatibility filter for KiCad 10.0.7-converted PCBench boards.

The KiCad 9-era D3 repair policy is retained unchanged: its severity overrides,
rule floors, and hole-to-hole floor ``min(0.25 mm, Default net-class clearance)``
remain the historical contract. The stage copies only boards with a valid,
zero-violation KiCad 10.0.7 DRC report to ``PCBENCH_NEWDRC_OUT``.

A valid nonempty DRC report is a legitimate ``drc_rejected`` board outcome.
Tool failures (timeout, nonzero exit, missing output, malformed JSON) are
``drc_error`` and never become a zero-violation pass.
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

SEVERITY_PATCH = {
    "solder_mask_bridge": "ignore",
    "drill_out_of_range": "ignore",
    "malformed_courtyard": "ignore",
    "courtyards_overlap": "ignore",
    "annular_width": "ignore",
    "copper_edge_clearance": "ignore",
    "hole_clearance": "ignore",
}

RULES_PATCH = {
    "min_via_annular_width": 0.0,
    "min_copper_edge_clearance": 0.0,
    "min_hole_clearance": 0.0,
}
HOLE_TO_HOLE_DEFAULT_MM = 0.25
DRC_TIMEOUT = 60
V10_DIR: Path | None = None
OUT_DIR: Path | None = None


def _env_dir(name: str) -> Path:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"{name} is not set. Point it at the matching directory "
            "(see tools/datagen/pcbench_prep/README.md)."
        )
    return Path(val)


def hole_to_hole_min(pro: dict) -> float | None:
    """Historical floor: Default net-class clearance capped at 0.25 mm."""
    classes = pro.get("net_settings", {}).get("classes", []) or []
    default = next((c for c in classes if c.get("name") == "Default"), None)
    clearance = (default or {}).get("clearance")
    if clearance is None:
        return None
    return min(HOLE_TO_HOLE_DEFAULT_MM, float(clearance))


def patch_pro(src_pro: Path, dst_pro: Path) -> None:
    """Patch only the historical D3 severity and rule-floor settings."""
    with open(src_pro, encoding="utf-8") as f:
        pro = json.load(f)

    ds = pro.setdefault("board", {}).setdefault("design_settings", {})
    ds.setdefault("rule_severities", {}).update(SEVERITY_PATCH)
    rules = ds.setdefault("rules", {})
    rules.update(RULES_PATCH)
    h2h = hole_to_hole_min(pro)
    if h2h is not None:
        rules["min_hole_to_hole"] = h2h

    with open(dst_pro, "w", encoding="utf-8") as f:
        json.dump(pro, f)


def run_drc(pcb: Path) -> dict:
    """Return a checked DRC report or an explicit stage error."""
    out = pcb.with_suffix(".drc.json")
    try:
        proc = subprocess.run(
            [kicad_tools.kicad_cli(), "pcb", "drc", "--format", "json",
             "--severity-error", "--output", str(out), str(pcb)],
            capture_output=True, text=True, timeout=DRC_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"DRC timed out after {DRC_TIMEOUT}s"}
    except OSError as exc:
        return {"status": "error", "error": f"could not start kicad-cli: {exc}"}

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-400:]
        return {"status": "error", "returncode": proc.returncode,
                "error": f"kicad-cli exited {proc.returncode}" + (f": {detail}" if detail else "")}
    if not out.is_file():
        return {"status": "error", "returncode": proc.returncode,
                "error": "kicad-cli produced no DRC JSON output"}
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "error", "returncode": proc.returncode,
                "error": f"invalid DRC JSON: {exc}"}
    if not isinstance(data, dict) or not isinstance(data.get("violations"), list) \
            or any(not isinstance(v, dict) for v in data["violations"]):
        return {"status": "error", "returncode": proc.returncode,
                "error": "malformed DRC JSON: expected an object with a violations array"}
    return {"status": "ok", "returncode": proc.returncode,
            "violations": data["violations"]}


def process_sample(name: str, src_root: Path | None = None,
                   out_root: Path | None = None) -> dict:
    src_root = src_root or V10_DIR
    out_root = out_root or OUT_DIR
    if src_root is None or out_root is None:
        raise RuntimeError("PCBENCH_V10_ROOT and PCBENCH_NEWDRC_OUT are not initialized")
    src_dir = src_root / name
    dst = out_root / name
    if dst.exists():
        shutil.rmtree(dst)
    pcb_src = src_dir / "processed_v10.kicad_pcb"
    pro_src = src_dir / "processed_v10.kicad_pro"

    if not pcb_src.is_file() or not pro_src.is_file():
        return {"name": name, "status": "input_missing",
                "error": "missing processed_v10.kicad_pcb or processed_v10.kicad_pro"}

    tmp = Path(tempfile.mkdtemp(prefix="drc_v10_"))
    try:
        pcb = tmp / "board.kicad_pcb"
        pro = tmp / "board.kicad_pro"
        shutil.copy(pcb_src, pcb)
        patch_pro(pro_src, pro)

        result = run_drc(pcb)
        if result["status"] == "error":
            return {**result, "name": name, "status": "drc_error"}
        violations = result["violations"]
        counts: dict[str, int] = {}
        for violation in violations:
            violation_type = str(violation.get("type", "unknown"))
            counts[violation_type] = counts.get(violation_type, 0) + 1
        if violations:
            return {"name": name, "status": "drc_rejected",
                    "returncode": result["returncode"], "violation_counts": counts,
                    "violations": [
                        {"type": v.get("type", "unknown"),
                         "description": str(v.get("description", ""))[:200]}
                        for v in violations[:20]
                    ]}

        dst.mkdir(parents=True)
        shutil.copy(pcb, dst / "processed_v10.kicad_pcb")
        shutil.copy(pro, dst / "processed_v10.kicad_pro")
        for extra in ("metadata.json", "final.json"):
            ep = src_dir / extra
            if ep.exists():
                shutil.copy(ep, dst / extra)
        return {"name": name, "status": "ok", "returncode": result["returncode"],
                "violation_counts": {}}
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "status": "stage_error",
                "error": f"{type(exc).__name__}: {exc}"[:400]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _load_conversion_results(root: Path) -> list[dict]:
    path = root / "_results.json"
    if not path.is_file():
        raise SystemExit(f"conversion report missing: {path}")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid conversion report {path}: {exc}") from exc
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("name"), str)
        or not isinstance(row.get("status"), str) for row in rows
    ):
        raise SystemExit(f"malformed conversion report: {path}")
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate source folder IDs in conversion report: {path}")
    if not rows:
        raise SystemExit(f"conversion report has no inputs: {path}")
    return rows


def main():
    global V10_DIR, OUT_DIR
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0,
                   help="0=all, N=only the first N source folders")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    V10_DIR = _env_dir("PCBENCH_V10_ROOT")
    OUT_DIR = _env_dir("PCBENCH_NEWDRC_OUT")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kicad_tools.assert_all()

    converted = _load_conversion_results(V10_DIR)
    if args.limit:
        converted = converted[:args.limit]
    passed_conversion = {row["name"] for row in converted if row["status"] == "ok"}
    for stale in OUT_DIR.iterdir():
        if stale.is_dir() and stale.name not in passed_conversion:
            shutil.rmtree(stale)
    print(f"Processing {len(converted)} converted inputs with {args.workers} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for row in converted:
            name = row["name"]
            if row["status"] != "ok":
                results.append({"name": name, "status": "conversion_failed",
                                "conversion_status": row["status"],
                                "error": row.get("error", "")})
            else:
                futs[ex.submit(process_sample, name, V10_DIR, OUT_DIR)] = name
        for future in as_completed(futs):
            name = futs[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"name": name, "status": "stage_error",
                                "error": f"{type(exc).__name__}: {exc}"[:400]})

    results.sort(key=lambda r: r["name"])
    failures = [r for r in results if r["status"] not in ("ok", "drc_rejected")]
    with open(OUT_DIR / "_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    counts: dict[str, int] = {}
    violation_counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        for kind, count in result.get("violation_counts", {}).items():
            violation_counts[kind] = violation_counts.get(kind, 0) + count
    print("\n=== Summary ===")
    print(f"  Total:         {len(results)}")
    print(f"  DRC pass:      {counts.get('ok', 0)}")
    print(f"  DRC rejected:  {counts.get('drc_rejected', 0)}")
    print(f"  Stage errors:  {len(failures)}")
    for status, count in sorted(counts.items()):
        print(f"  {status:16s} {count}")
    for r in failures[:20]:
        print(f"    {r['status']:16s} {r['name']}: {r.get('error', '')}")
    if violation_counts:
        print("  DRC violation types:")
        for kind, count in sorted(violation_counts.items()):
            print(f"    {count:5d}  {kind}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
