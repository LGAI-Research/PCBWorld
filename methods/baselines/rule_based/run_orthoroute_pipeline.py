#!/usr/bin/env python3
"""End-to-end OrthoRoute pipeline for a directory of KiCad boards.

Given a dataset root whose sub-folders each contain a board, this runs the
full chain per board:

    Stage 1  kicad_pcb (+kicad_pro)  ->  .orp   (pcbnew; via Docker or local)
    Stage 2  .orp                    ->  .ORS   (orthoroute headless, GPU/CPU)
    Stage 3  .ORS  + unrouted pcb    ->  routed .kicad_pcb   (pure text)

Layout assumed per board folder ``<data-root>/<board_id>/``:
    <stem>_unrouted.kicad_pcb     (routing input + stage-3 base)
    <stem>.kicad_pro              (optional; for netclass DRC in the .orp)

Outputs under ``<out-root>/<board_id>/``:
    <board_id>.orp
    <board_id>.ORS
    <board_id>_orthoroute_<gpu|cpu>.kicad_pcb
    <board_id>.routing.log

Stage 1 needs KiCad's ``pcbnew``. When the conda env does not ship it, Stage 1
runs inside the ``pcbnew-repro:9.0`` Docker image (see
``engine/pcbnew_prep/pcbnew_env/``). Alternatively point ``--orp-root`` at
pre-made ``.orp`` files (the pre-converted DSN/ORP mirror of the dataset) to
skip Stage 1 entirely.

Stages 2-3 run in the current Python env (needs ``orthoroute`` importable, and
``cupy`` + a GPU for ``--gpu``).

Examples:
  # full chain, 5 boards, GPU, ORP via Docker
  python methods/baselines/rule_based/run_orthoroute_pipeline.py \
      --data-root $CADAGENT_DATA_ROOT/pcbench/exacad_sorted \
      --out-root methods/baselines/rule_based/_run_outputs/ortho_exacad --limit 5 --gpu

  # skip Stage 1, route from canonical ORPs, CPU (deterministic)
  python methods/baselines/rule_based/run_orthoroute_pipeline.py \
      --data-root $CADAGENT_DATA_ROOT/pcbench/exacad_sorted \
      --orp-root $CADAGENT_DATA_ROOT/pcbench/exacad_sorted_dsn \
      --out-root methods/baselines/rule_based/_run_outputs/ortho_exacad_cpu --cpu

  # only generate ORPs (Stage 1), e.g. on the host that has the Docker image
  python methods/baselines/rule_based/run_orthoroute_pipeline.py --data-root ... --out-root ... --stage orp
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_BASELINES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASELINES_DIR.parents[2]   # rule_based -> baselines -> methods -> repo root
_EXTERNAL = _PROJECT_ROOT / "external"      # vendored rule-based libs (OrthoRoute, freerouting)
sys.path.insert(0, str(_BASELINES_DIR / "_converters"))

DEFAULT_STEM = "processed_v9_guide_v3"
DEFAULT_IMAGE = "pcbnew-repro:9.0"
MAIN_SCRIPT = _EXTERNAL / "OrthoRoute" / "main.py"
# The engine is a separate repository, pinned as the engine/ submodule.
BATCH_ORP = _PROJECT_ROOT / "engine" / "pcbnew_prep" / "batch_kicad_to_orp.py"


def discover_boards(data_root: Path, stem: str, only: list[str] | None):
    """Yield (board_id, unrouted_pcb, pro_or_None) for each board folder."""
    boards = []
    for folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if only and folder.name not in only:
            continue
        pcb = folder / f"{stem}_unrouted.kicad_pcb"
        if not pcb.is_file():
            continue
        # pcbnew converter strips routing anyway; prefer the routed-variant pro
        # (the _unrouted variant often lacks its own .kicad_pro)
        pro = folder / f"{stem}.kicad_pro"
        if not pro.is_file():
            pro = folder / f"{stem}_unrouted.kicad_pro"
        boards.append((folder.name, pcb, pro if pro.is_file() else None))
    return boards


def stage1_orp(boards, out_root: Path, orp_root: Path | None, image: str,
               use_docker: bool):
    """Produce <out>/<board_id>.orp for every board. Returns {board_id: orp_path}."""
    orp_paths = {}
    todo = []
    for board_id, pcb, pro in boards:
        out_dir = out_root / board_id
        out_dir.mkdir(parents=True, exist_ok=True)
        orp = out_dir / f"{board_id}.orp"
        if orp_root is not None:
            # use a pre-made ORP if present
            cand = orp_root / board_id / f"{pcb.name.replace('_unrouted.kicad_pcb', '')}.orp"
            if cand.is_file():
                orp_paths[board_id] = cand
                continue
        if orp.is_file() and orp.stat().st_size > 0:
            orp_paths[board_id] = orp
            continue
        todo.append({"board_id": board_id, "pcb": str(pcb),
                     "orp": str(orp), "pro": str(pro) if pro else None})
        orp_paths[board_id] = orp

    if not todo:
        print(f"[stage1] all {len(orp_paths)} ORPs already present — nothing to convert")
        return orp_paths

    manifest = out_root / "_orp_manifest.json"
    manifest.write_text(json.dumps(todo, indent=2), encoding="utf-8")
    print(f"[stage1] converting {len(todo)} boards -> ORP "
          f"({'docker '+image if use_docker else 'local pcbnew'})")

    if use_docker:
        import os
        uid = os.getuid()
        gid = os.getgid()
        # Mount the two roots that paths reference + the repo, all by absolute path.
        mounts = ["-v", f"{_PROJECT_ROOT}:{_PROJECT_ROOT}:ro",
                  "-v", f"{out_root}:{out_root}"]
        # data roots: mount each board's source tree root once (parents of pcbs)
        src_roots = {str(Path(item["pcb"]).parents[1]) for item in todo}
        for r in sorted(src_roots):
            mounts += ["-v", f"{r}:{r}:ro"]
        cmd = ["docker", "run", "--rm", "--user", f"{uid}:{gid}",
               "-e", "HOME=/tmp", *mounts, image,
               "python3", str(BATCH_ORP), str(manifest)]
    else:
        cmd = ["python3", str(BATCH_ORP), str(manifest)]

    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[stage1] WARNING batch converter exited rc={rc}")
    return orp_paths


def stage2_route(board_id, orp, out_dir, use_gpu, max_iterations, timeout_s):
    """Run orthoroute headless: .orp -> .ORS. Returns (ors_path|None, log_path)."""
    ors = out_dir / f"{board_id}.ORS"
    log = out_dir / f"{board_id}.routing.log"
    cmd = [sys.executable, str(MAIN_SCRIPT), "headless", str(orp),
           "-o", str(ors), "--max-iterations", str(max_iterations),
           "--use-gpu" if use_gpu else "--cpu-only"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        log.write_text((proc.stdout or "") + (proc.stderr or ""))
        if proc.returncode != 0 or not ors.is_file() or ors.stat().st_size == 0:
            return None, log
        return ors, log
    except subprocess.TimeoutExpired:
        log.write_text(f"TIMEOUT after {timeout_s}s\n")
        return None, log


def stage3_pcb(board_id, ors, base_pcb, out_dir, tag):
    """ORS -> routed .kicad_pcb. Returns (pcb_path, stats)."""
    from ors_to_kicadpcb import convert as convert_ors
    out_pcb = out_dir / f"{board_id}_orthoroute_{tag}.kicad_pcb"
    stats = convert_ors(Path(ors), Path(base_pcb), out_pcb)
    return out_pcb, stats


def route_and_convert_one(task: dict) -> dict:
    """Stage 2 + Stage 3 for a single board. Top-level so it is picklable for a
    process pool. Each board is fully independent (its own ORS/PCB), so running
    several in parallel does not change any board's own (deterministic) result —
    it only hides the per-process cupy/CUDA init latency."""
    board_id = task["board_id"]
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    orp = Path(task["orp"])
    tag = task["tag"]
    rec = {"board_id": board_id, "status": "", "tracks": 0, "vias": 0,
           "elapsed_s": 0.0, "routed_pcb": None}
    if not orp.is_file():
        rec["status"] = "no_orp"
        return rec
    t0 = time.time()
    ors, log = stage2_route(board_id, orp, out_dir, task["use_gpu"],
                            task["max_iterations"], task["timeout"])
    if ors is None:
        rec["status"] = "route_fail"
        rec["elapsed_s"] = round(time.time() - t0, 2)
        return rec
    try:
        out_pcb, stats = stage3_pcb(board_id, ors, Path(task["base_pcb"]), out_dir, tag)
        rec.update(status="ok",
                   tracks=stats.get("tracks_written", 0),
                   vias=stats.get("vias_written", 0),
                   routed_pcb=str(out_pcb))
    except Exception as exc:  # noqa: BLE001
        rec["status"] = f"convert_fail: {exc}"
    rec["elapsed_s"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path,
                    help="dir whose sub-folders each hold a board")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--limit", type=int, default=0, help="0 = all boards")
    ap.add_argument("--boards", nargs="*", default=None,
                    help="explicit board_id folder names (overrides --limit)")
    ap.add_argument("--orp-root", type=Path, default=None,
                    help="use pre-made .orp from here (per <board_id>/<stem>.orp); "
                         "boards missing one fall back to Stage 1")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--gpu", dest="gpu", action="store_true")
    mode.add_argument("--cpu", dest="gpu", action="store_false")
    ap.set_defaults(gpu=True)
    ap.add_argument("--max-iterations", type=int, default=250)
    ap.add_argument("--timeout", type=int, default=1800, help="per-board route timeout (s)")
    ap.add_argument("--stage", choices=("all", "orp", "route"), default="all",
                    help="all=full chain; orp=Stage 1 only; route=Stages 2-3 only")
    ap.add_argument("--docker-image", default=DEFAULT_IMAGE)
    ap.add_argument("--local-pcbnew", action="store_true",
                    help="run Stage 1 with the current python's pcbnew (no Docker)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel boards for Stages 2-3 (each board is an "
                         "independent process; hides per-process cupy/CUDA init). "
                         "Boards are tiny so the shared GPU time-slices fine.")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    boards = discover_boards(data_root, args.stem, args.boards)
    if not args.boards and args.limit:
        boards = boards[:args.limit]
    if not boards:
        print(f"no boards with {args.stem}_unrouted.kicad_pcb under {data_root}")
        sys.exit(1)
    print(f"[pipeline] {len(boards)} boards | mode={'GPU' if args.gpu else 'CPU'} "
          f"| stage={args.stage} | out={out_root}")

    # Stage 1
    orp_paths = {}
    if args.stage in ("all", "orp"):
        orp_paths = stage1_orp(boards, out_root, args.orp_root,
                               args.docker_image, use_docker=not args.local_pcbnew)
    else:
        for board_id, pcb, _pro in boards:
            cand = out_root / board_id / f"{board_id}.orp"
            if args.orp_root is not None:
                pre = args.orp_root / board_id / f"{args.stem}.orp"
                if pre.is_file():
                    cand = pre
            orp_paths[board_id] = cand
    if args.stage == "orp":
        print("[pipeline] Stage 1 only — done")
        return

    # Stages 2-3
    tag = "gpu" if args.gpu else "cpu"
    tasks = []
    for board_id, pcb, _pro in boards:
        tasks.append({
            "board_id": board_id, "out_dir": str(out_root / board_id),
            "orp": str(orp_paths.get(board_id, "")), "base_pcb": str(pcb),
            "tag": tag, "use_gpu": args.gpu, "max_iterations": args.max_iterations,
            "timeout": args.timeout,
        })

    results = []
    n = len(tasks)
    done = 0
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(route_and_convert_one, t): t["board_id"] for t in tasks}
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
                done += 1
                print(f"[{done}/{n}] {rec['board_id']}: {rec['status']} "
                      f"tracks={rec['tracks']} vias={rec['vias']} ({rec['elapsed_s']}s)",
                      flush=True)
    else:
        for t in tasks:
            rec = route_and_convert_one(t)
            results.append(rec)
            done += 1
            print(f"[{done}/{n}] {rec['board_id']}: {rec['status']} "
                  f"tracks={rec['tracks']} vias={rec['vias']} ({rec['elapsed_s']}s)",
                  flush=True)
    results.sort(key=lambda r: r["board_id"])

    summary = {
        "data_root": str(data_root), "out_root": str(out_root),
        "stem": args.stem, "mode": tag, "max_iterations": args.max_iterations,
        "n_boards": len(results),
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "results": results,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[pipeline] {summary['n_ok']}/{summary['n_boards']} OK "
          f"-> {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
