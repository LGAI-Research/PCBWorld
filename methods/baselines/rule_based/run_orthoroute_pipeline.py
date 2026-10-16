#!/usr/bin/env python3
"""End-to-end OrthoRoute pipeline for a directory of KiCad boards.

Given a dataset root whose sub-folders each contain a board, this runs the
full chain per board:

    Stage 1  kicad_pcb (+kicad_pro)  ->  .orp   (pcbnew)
    Stage 2  .orp                    ->  .ORS   (orthoroute headless, GPU/CPU)
    Stage 3  .ORS  + unrouted pcb    ->  routed .kicad_pcb   (pure text)

Layout assumed per board folder ``<data-root>/<board_id>/``:
    <stem>_unrouted.kicad_pcb     (routing input + stage-3 base)
    <stem>.kicad_pro              (optional; for netclass DRC in the .orp)

Outputs under ``<out-root>/<board_id>/``:
    <stem>.orp
    <board_id>.ORS
    <board_id>_orthoroute_<gpu|cpu>.kicad_pcb
    <board_id>.routing.log

Stage 1 calls ``kicad_tools.assert_all()`` before work, requiring the pinned
KiCad 10.0.7 Engine and independent stock CLI/Python pair. It then invokes
``make_dsn_orp_v3.py`` with the Engine ``pcbnew`` interpreter. Every input board,
including reused ORPs, is recorded in ``<out-root>/_orp_stage1_report.json``; a
nonzero converter exit, invalid report, or missing/empty ORP aborts Stage 1.
Alternatively point ``--orp-root`` at pre-made ``.orp`` files (the
pre-converted DSN/ORP mirror registered as ``pcbench_exacad_dsn`` in
``configs/paths.yaml``) to skip conversion; preflight and the outcome report
still run.

Stages 2-3 run in the current Python env (needs ``orthoroute`` importable, and
``cupy`` + a GPU for ``--gpu``).

Examples (dataset paths via the resolver CLI — no hardcoded roots):
  # full chain, 5 boards, GPU (ORP via the resolved pcbnew)
  python methods/baselines/rule_based/run_orthoroute_pipeline.py \
      --data-root "$(python -m configs.loader.paths resolve pcbench_exacad)" \
      --out-root methods/baselines/rule_based/_run_outputs/ortho_exacad --limit 5 --gpu

  # skip Stage 1, route from canonical ORPs, CPU (deterministic)
  python methods/baselines/rule_based/run_orthoroute_pipeline.py \
      --data-root "$(python -m configs.loader.paths resolve pcbench_exacad)" \
      --orp-root "$(python -m configs.loader.paths resolve pcbench_exacad_dsn)" \
      --out-root methods/baselines/rule_based/_run_outputs/ortho_exacad_cpu --cpu

  # only generate ORPs (Stage 1)
  python methods/baselines/rule_based/run_orthoroute_pipeline.py --data-root ... --out-root ... --stage orp
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_BASELINES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASELINES_DIR.parents[2]   # rule_based -> baselines -> methods -> repo root
_EXTERNAL = _PROJECT_ROOT / "external"      # vendored rule-based libs (OrthoRoute, freerouting)
sys.path.insert(0, str(_BASELINES_DIR / "_converters"))

DEFAULT_STEM = "processed_v10_guide_v3"
MAIN_SCRIPT = _EXTERNAL / "OrthoRoute" / "main.py"
MAKE_DSN_ORP = _PROJECT_ROOT / "engine" / "pcbnew_prep" / "make_dsn_orp_v3.py"

# Which child interpreter imports pcbnew: the engine's BUILD_PCBNEW=1 build first,
# then /usr/bin/python3 (an apt KiCad); KICAD_CLI / PCBNEW_PYTHON override.
sys.path.insert(0, str(_PROJECT_ROOT / "tools" / "datagen" / "pcbench_prep"))
import kicad_tools  # noqa: E402


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


def stage1_orp(boards, out_root: Path, orp_root: Path | None, stem: str):
    """Create or reuse every ORP, recording and validating all outcomes."""
    out_root.mkdir(parents=True, exist_ok=True)
    outcome_path = out_root / "_orp_stage1_report.json"
    converter_report_path = None
    converter_status = "not_run"
    converter_returncode = None
    converter_error = None
    converter_results = []
    outcomes = {
        board_id: {
            "board_id": board_id, "input_pcb": str(pcb),
            "orp_path": None, "status": "pending",
            "source": None, "converter_status": "not_run", "converter_msg": None,
            "error": None,
        }
        for board_id, pcb, _pro in boards
    }

    def write_report():
        outcome_path.write_text(json.dumps({
            "stem": stem,
            "converter": {
                "script": str(MAKE_DSN_ORP), "status": converter_status,
                "returncode": converter_returncode,
                "report_path": str(converter_report_path) if converter_report_path else None,
                "error": converter_error, "results": converter_results,
            },
            "boards": list(outcomes.values()),
        }, indent=2), encoding="utf-8")

    try:
        kicad_tools.assert_all()
    except SystemExit as exc:
        converter_status = "blocked_by_preflight"
        converter_error = str(exc) or "kicad_tools.assert_all() failed"
        for outcome in outcomes.values():
            outcome.update(status="preflight_failed", error=converter_error)
        write_report()
        raise

    orp_paths = {}
    todo = []
    for board_id, pcb, _pro in boards:
        out_dir = out_root / board_id
        out_dir.mkdir(parents=True, exist_ok=True)
        orp = out_dir / f"{stem}.orp"
        candidates = []
        if orp_root is not None:
            candidates.append((orp_root / board_id / f"{stem}.orp", "orp_root"))
        candidates.append((orp, "out_root"))
        reused = next(
            ((candidate, source) for candidate, source in candidates
             if candidate.is_file() and candidate.stat().st_size > 0),
            None,
        )
        outcome = outcomes[board_id]
        if reused is not None:
            path, source = reused
            outcome.update(status="reused", orp_path=str(path), source=source)
            orp_paths[board_id] = path
        else:
            outcome["orp_path"] = str(orp)
            todo.append(board_id)

    if not todo:
        write_report()
        print(f"[stage1] all {len(orp_paths)} ORPs already present — nothing to convert")
        return orp_paths

    if stem != DEFAULT_STEM:
        converter_status = "blocked"
        converter_error = (
            f"Stage 1 uses make_dsn_orp_v3.py and requires --stem {DEFAULT_STEM!r}; "
            "pass --orp-root to reuse other stems"
        )
        for board_id in todo:
            outcomes[board_id].update(status="failed", source="converter", error=converter_error)
        write_report()
        raise SystemExit(f"{converter_error}; see {outcome_path}")

    pcb_root = boards[0][1].parent.parent
    py = kicad_tools.pcbnew_python()
    print(f"[stage1] converting {len(todo)} boards to v10 DSN/ORP siblings "
          f"(pcbnew: {py})")
    env = os.environ.copy()
    env["PCBENCH_PCB_ROOT"] = str(pcb_root)
    env["PCBENCH_DSN_ROOT"] = str(out_root)
    converter_report_path = out_root / "_orp_converter_report.json"
    converter_report_path.write_text("[]", encoding="utf-8")
    cmd = [
        py, str(MAKE_DSN_ORP), "--src", str(pcb_root), "--dst", str(out_root),
        "--report", str(converter_report_path), "--targets", *todo,
    ]
    try:
        converter_returncode = subprocess.run(cmd, env=env).returncode
    except OSError as exc:
        converter_error = f"{type(exc).__name__}: {exc}"

    try:
        converter_results = json.loads(converter_report_path.read_text(encoding="utf-8"))
        if not isinstance(converter_results, list):
            raise ValueError("report root is not a list")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        converter_error = converter_error or f"invalid converter report: {exc}"
        converter_results = []

    report_by_id = {}
    report_error = None
    for result in converter_results:
        if not isinstance(result, dict) or not isinstance(result.get("folder"), str):
            report_error = "converter report contains a malformed board record"
            break
        board_id = result["folder"]
        if board_id in report_by_id:
            report_error = f"converter report repeats board {board_id!r}"
            break
        report_by_id[board_id] = result
    if report_error is None and set(report_by_id) != set(todo):
        missing = sorted(set(todo) - set(report_by_id))
        unexpected = sorted(set(report_by_id) - set(todo))
        report_error = f"converter report board mismatch (missing={missing}, unexpected={unexpected})"

    failed = []
    for board_id in todo:
        outcome = outcomes[board_id]
        result = report_by_id.get(board_id)
        outcome["source"] = "converter"
        outcome["converter_status"] = result.get("status") if result else "missing_record"
        outcome["converter_msg"] = result.get("msg") if result else None
        orp = Path(outcome["orp_path"])
        errors = []
        if converter_returncode is None:
            errors.append(converter_error or "converter did not return a status")
        elif converter_returncode != 0:
            errors.append(f"converter exited with rc={converter_returncode}")
        if converter_error and converter_returncode is not None:
            errors.append(converter_error)
        if report_error:
            errors.append(report_error)
        if result is None:
            errors.append("missing converter report record")
        elif result.get("status") != "ok":
            errors.append(f"converter status {result.get('status')!r}: {result.get('msg', '')}")
        if not orp.is_file() or orp.stat().st_size == 0:
            errors.append("expected ORP is missing or empty")
        if errors:
            outcome.update(status="failed", error="; ".join(errors))
            failed.append(board_id)
        else:
            outcome.update(status="generated", error=None)
            orp_paths[board_id] = orp

    converter_status = "failed" if failed else "ok"
    write_report()
    if failed:
        shown = ", ".join(failed[:10])
        if len(failed) > 10:
            shown += f", ... ({len(failed) - 10} more)"
        raise SystemExit(f"Stage 1 failed to produce valid ORPs for {shown}; see {outcome_path}")
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
        orp_paths = stage1_orp(boards, out_root, args.orp_root, args.stem)
    else:
        for board_id, pcb, _pro in boards:
            cand = out_root / board_id / f"{args.stem}.orp"
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
