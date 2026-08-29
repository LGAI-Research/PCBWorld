"""Unified runner for the three rule-based routers (freerouting / krt / orthoroute).

Per board:
  1. Locate raw .kicad_pcb + .kicad_pro + sibling .dsn / .orp via _lib.datasets
  2. Drive the chosen baseline -> raw output (.ses / .ORS / .kicad_pcb)
  3. Convert raw -> .kicad_pcb if needed (SES/ORS via _converters)
  4. Run eval.metrics.evaluate_one against the routed .kicad_pcb
  5. Aggregate per-run summary.{json,txt}

Output: <output-root>/<dataset_tag>/<algorithm>/seed<N>/{raw,routed,eval,manifest.json}
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_BASELINES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASELINES_DIR.parents[2]   # rule_based -> baselines -> methods -> repo root
_EXTERNAL = _PROJECT_ROOT / "external"      # vendored rule-based libs (OrthoRoute, freerouting)
for p in (_PROJECT_ROOT, _PROJECT_ROOT / "eval", _PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from methods.baselines.rule_based._lib.datasets import (  # noqa: E402
    BoardPaths, algorithm_tag, dataset_tag, select_boards,
)

log = logging.getLogger("run_rule_based_routers")

DEFAULT_OUTPUT_ROOT = Path("methods/baselines/rule_based") / "_run_outputs"
FREEROUTING_JAR = _EXTERNAL / "freerouting" / "freerouting-2.1.0.jar"

DEFAULT_TIMEOUTS = {"freerouting": 1800, "krt": 1800, "orthoroute": 1800}


@dataclass
class BoardResult:
    board_id: str
    seed: int
    status: str
    elapsed_routing_s: float
    routed_pcb: Optional[str] = None
    raw_path: Optional[str] = None
    routing_log: Optional[str] = None
    eval_log: Optional[str] = None
    error: str = ""
    eval_metrics: dict = field(default_factory=dict)


def _inject_snap_angle_90(src_dsn, dst_dsn):
    text = src_dsn.read_text(encoding="utf-8", errors="replace")
    if "snap_angle" in text:
        dst_dsn.write_text(text, encoding="utf-8")
        return
    text = re.sub(r"(\(structure\s*\n)",
                  r"\1    (snap_angle ninety_degree)\n",
                  text, count=1)
    dst_dsn.write_text(text, encoding="utf-8")


def _route_freerouting(board, seed, raw_dir, routed_dir, algo, timeout_s,
                       via_prohibit, angle_90, max_passes, extra_env):
    from methods.baselines.rule_based._converters import convert_ses
    ses_path = raw_dir / f"{board.board_id}.ses"
    log_path = raw_dir / f"{board.board_id}.routing.log"
    routed_path = routed_dir / f"{board.board_id}_{algo}.kicad_pcb"
    tmp_dir = raw_dir / f"{board.board_id}.freerouting_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    patched_dsn = None
    if angle_90:
        patched_dsn = raw_dir / f"{board.board_id}._angle90.dsn"
        _inject_snap_angle_90(board.dsn, patched_dsn)
        dsn_in = patched_dsn
    else:
        dsn_in = board.dsn
    env = os.environ.copy()
    if via_prohibit:
        env["FREEROUTING__ROUTER__VIAS_ALLOWED"] = "false"
        env["FREEROUTING__ROUTER__SCORING__VIA_COSTS"] = "10000"
    env.update(extra_env)
    cmd = ["java", "-Djava.awt.headless=true",
           f"-Djava.io.tmpdir={tmp_dir}", "-Xmx4g",
           "-jar", str(FREEROUTING_JAR),
           "-de", str(dsn_in), "-do", str(ses_path),
           "-mp", str(max_passes)]
    t0 = time.time()
    log_text = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=env)
        log_text = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        log_text = f"TIMEOUT after {timeout_s}s\n"
        if exc.stdout:
            log_text += exc.stdout.decode("utf-8", errors="replace")
        log_path.write_text(log_text)
        return BoardResult(board.board_id, seed, "timeout",
                           round(time.time()-t0, 3),
                           raw_path=str(ses_path) if ses_path.exists() else None,
                           routing_log=str(log_path))
    except Exception as exc:
        log_path.write_text(f"ERROR: {exc}\n")
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           routing_log=str(log_path), error=str(exc))
    finally:
        log_path.write_text(log_text)
    if not ses_path.exists() or ses_path.stat().st_size == 0:
        return BoardResult(board.board_id, seed, "no_output",
                           round(time.time()-t0, 3),
                           routing_log=str(log_path))
    try:
        convert_ses(ses_path, board.raw_pcb, routed_path)
    except Exception as exc:
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           raw_path=str(ses_path), routing_log=str(log_path),
                           error=f"ses_to_kicadpcb: {exc}")
    finally:
        if patched_dsn and patched_dsn.exists():
            patched_dsn.unlink()
    return BoardResult(board.board_id, seed, "ok",
                       round(time.time()-t0, 3),
                       routed_pcb=str(routed_path), raw_path=str(ses_path),
                       routing_log=str(log_path))


def _parse_dsn_constraints(dsn_path):
    if not dsn_path.is_file():
        return None
    try:
        text = dsn_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    layers = re.findall(r"\(layer\s+([\w.]+)\s*\n\s*\(type\s+signal\)", text)
    if not layers:
        layers = re.findall(r"\(layer\s+(F\.Cu|B\.Cu|In\d+\.Cu)\b", text)
    if not layers:
        layers = ["F.Cu", "B.Cu"]
    via_m = re.search(r'\(via\s+"?Via\[\d+-\d+\]_(\d+):(\d+)_um"?', text)
    via_size_mm = int(via_m.group(1)) / 1000.0 if via_m else 1.2
    via_drill_mm = int(via_m.group(2)) / 1000.0 if via_m else 0.6
    cls_block = re.search(r'\(class\s+kicad_default[\s\S]*?\(rule[\s\S]*?\)\s*\)', text)
    track_w_mm, clearance_mm = 0.3, 0.3
    if cls_block:
        wm = re.search(r"\(width\s+(\d+)\)", cls_block.group(0))
        cm = re.search(r"\(clearance\s+(\d+)\)", cls_block.group(0))
        if wm:
            track_w_mm = int(wm.group(1)) / 1000.0
        if cm:
            clearance_mm = int(cm.group(1)) / 1000.0
    non_default = {}
    for cls_m in re.finditer(
        r'\(class\s+(\S+)([\s\S]*?)\)\s*(?=\(class|\)\s*\(wiring|\Z)', text):
        name = cls_m.group(1)
        body = cls_m.group(2)
        if name == "kicad_default":
            continue
        wm = re.search(r"\(width\s+(\d+)\)", body)
        if not wm:
            continue
        cls_w_mm = int(wm.group(1)) / 1000.0
        if abs(cls_w_mm - track_w_mm) < 1e-9:
            continue
        for net_token in re.findall(
            r'(?<!\w)("(?:[^"\\]|\\.)*"|[\w/+\-]+)(?=\s|\))', body):
            if net_token in ("circuit", "rule", "use_via", "width", "clearance"):
                continue
            n = net_token.strip('"')
            if n and not n.isdigit():
                non_default[n] = cls_w_mm
    return {"layers": layers, "track_width": round(track_w_mm, 4),
            "clearance": round(clearance_mm, 4),
            "via_size": round(via_size_mm, 4),
            "via_drill": round(via_drill_mm, 4),
            "non_default_net_widths": non_default}


def _route_krt(board, seed, raw_dir, routed_dir, algo, timeout_s):
    from krt_runner import run_route
    routed_path = routed_dir / f"{board.board_id}_{algo}.kicad_pcb"
    metrics_path = raw_dir / f"{board.board_id}.metrics.json"
    log_path = raw_dir / f"{board.board_id}.routing.log"
    constraints = _parse_dsn_constraints(board.dsn)
    if constraints is None:
        return BoardResult(board.board_id, seed, "error", 0.0,
                           error="failed to parse DSN constraints")
    args = [str(board.raw_pcb.resolve()), str(routed_path.resolve()),
            "--layers", *constraints["layers"],
            "--track-width", str(constraints["track_width"]),
            "--clearance", str(constraints["clearance"]),
            "--via-size", str(constraints["via_size"]),
            "--via-drill", str(constraints["via_drill"]),
            "--grid-step", "0.1",
            "--max-iterations", "200000",
            "--max-probe-iterations", "5000",
            "--max-ripup", "3",
            "--heuristic-weight", "1.9",
            "--ordering", "mps",
            "--metrics-json", str(metrics_path.resolve())]
    nd = constraints.get("non_default_net_widths", {}) or {}
    # KRT's argparse treats net names beginning with '-' as options when they
    # are passed after --power-nets. Leave those nets at default width rather
    # than failing the whole board.
    nd = {name: width for name, width in nd.items() if not name.startswith("-")}
    if nd:
        names = list(nd.keys()); widths = [str(v) for v in nd.values()]
        args += ["--power-nets", *names, "--power-nets-widths", *widths]
    t0 = time.time()
    log_text = ""
    try:
        proc = run_route(args, capture_output=True, text=True, timeout=timeout_s)
        log_text = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        log_text = f"TIMEOUT after {timeout_s}s\n"
        if exc.stdout:
            log_text += exc.stdout.decode("utf-8", errors="replace")
        log_path.write_text(log_text)
        return BoardResult(board.board_id, seed, "timeout",
                           round(time.time()-t0, 3),
                           routing_log=str(log_path))
    except Exception as exc:
        log_path.write_text(f"ERROR: {exc}\n")
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           routing_log=str(log_path), error=str(exc))
    finally:
        log_path.write_text(log_text)
    if rc != 0 or not routed_path.exists():
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           routing_log=str(log_path),
                           error=f"krt rc={rc}")
    return BoardResult(board.board_id, seed, "ok",
                       round(time.time()-t0, 3),
                       routed_pcb=str(routed_path),
                       routing_log=str(log_path))


def _route_orthoroute(board, seed, raw_dir, routed_dir, algo, timeout_s, use_gpu=True):
    from methods.baselines.rule_based._converters import convert_ors
    main_script = _EXTERNAL / "OrthoRoute" / "main.py"
    ors_path = raw_dir / f"{board.board_id}.ORS"
    log_path = raw_dir / f"{board.board_id}.routing.log"
    routed_path = routed_dir / f"{board.board_id}_{algo}.kicad_pcb"
    if not board.orp.is_file():
        log_path.write_text(f"ERROR: missing .orp at {board.orp}\n")
        return BoardResult(board.board_id, seed, "error", 0.0,
                           routing_log=str(log_path),
                           error=f"no ORP: {board.orp}")
    cmd = [sys.executable, str(main_script),
           "headless", str(board.orp), "-o", str(ors_path)]
    if use_gpu:
        cmd.append("--use-gpu")
    else:
        cmd.append("--cpu-only")
    t0 = time.time()
    log_text = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        log_text = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        log_text = f"TIMEOUT after {timeout_s}s\n"
        if exc.stdout:
            log_text += exc.stdout.decode("utf-8", errors="replace")
        log_path.write_text(log_text)
        return BoardResult(board.board_id, seed, "timeout",
                           round(time.time()-t0, 3),
                           raw_path=str(ors_path) if ors_path.exists() else None,
                           routing_log=str(log_path))
    except Exception as exc:
        log_path.write_text(f"ERROR: {exc}\n")
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           routing_log=str(log_path), error=str(exc))
    finally:
        log_path.write_text(log_text)
    if rc != 0 or not ors_path.exists() or ors_path.stat().st_size == 0:
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           raw_path=str(ors_path) if ors_path.exists() else None,
                           routing_log=str(log_path),
                           error=f"orthoroute headless rc={rc}")
    try:
        convert_ors(ors_path, board.raw_pcb, routed_path)
    except Exception as exc:
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           raw_path=str(ors_path),
                           routing_log=str(log_path),
                           error=f"ors_to_kicadpcb: {exc}")
    return BoardResult(board.board_id, seed, "ok",
                       round(time.time()-t0, 3),
                       routed_pcb=str(routed_path),
                       raw_path=str(ors_path),
                       routing_log=str(log_path))


def _run_eval(board, routed_pcb, eval_logs_dir, reward_config, check_angle, algo):
    from eval import metrics as eval_metrics
    log_path = eval_logs_dir / f"{board.board_id}_{algo}.json"
    m = eval_metrics.evaluate_one(
        str(routed_pcb), str(board.pro),
        reward_config_name=reward_config, check_angle=check_angle)
    log_path.write_text(json.dumps(m, indent=2, default=str))
    return m, log_path


def _process_one_board(args):
    board = args["board"]; seed = args["seed"]; baseline = args["baseline"]
    raw_dir = args["raw_dir"]; routed_dir = args["routed_dir"]
    eval_logs_dir = args["eval_logs_dir"]; algo = args["algo"]
    timeout_s = args["timeout_s"]; do_eval = args["do_eval"]
    reward_config = args["reward_config"]; check_angle = args["check_angle"]
    via_prohibit = args["via_prohibit"]; angle_90 = args["angle_90"]
    max_passes = args["max_passes"]; extra_env = args["extra_env"]
    use_gpu = args["use_gpu"]
    if baseline == "freerouting":
        result = _route_freerouting(board, seed, raw_dir, routed_dir, algo,
                                    timeout_s, via_prohibit, angle_90,
                                    max_passes, extra_env)
    elif baseline == "krt":
        result = _route_krt(board, seed, raw_dir, routed_dir, algo, timeout_s)
    elif baseline == "orthoroute":
        result = _route_orthoroute(board, seed, raw_dir, routed_dir, algo,
                                   timeout_s, use_gpu=use_gpu)
    else:
        result = BoardResult(board.board_id, seed, "error", 0.0,
                             error=f"unknown baseline {baseline!r}")
    if do_eval and result.status == "ok" and result.routed_pcb:
        try:
            m, ev_log = _run_eval(board, Path(result.routed_pcb), eval_logs_dir,
                                  reward_config=reward_config,
                                  check_angle=check_angle, algo=algo)
            result.eval_log = str(ev_log)
            result.eval_metrics = {
                "success": m.get("success"),
                "routability": m.get("routability"),
                "track_count": m.get("track_count"),
                "via_count": m.get("via_count"),
                "wirelength_mm": m.get("wirelength_mm"),
                "drv_errors_only_count": m.get("drv_errors_only_count"),
                "drv_errors_and_promoted_count": m.get("drv_errors_and_promoted_count"),
                "final_potential": m.get("final_potential")}
        except Exception as exc:
            result.status = "eval_error"
            result.error = f"eval: {exc}"
    gc.collect()
    return asdict(result)


def _aggregate_summary(eval_logs_dir, seed_dir, dataset_label, algorithm_label):
    from eval import aggregation
    per_sample = []
    for f in sorted(eval_logs_dir.glob("*.json")):
        try:
            per_sample.append(json.loads(f.read_text()))
        except Exception:
            pass
    if not per_sample:
        log.warning("aggregate: no per-board logs in %s", eval_logs_dir)
        return
    agg = aggregation.aggregate(per_sample)
    (seed_dir / "eval" / "summary.json").write_text(
        json.dumps(agg, indent=2, default=str))
    # NOTE: _format_summary_txt was never part of the eval API (pre-existing
    # broken reference); summary.txt is skipped rather than crashing.


def _emit_progress(r, baseline):
    em = r.get("eval_metrics", {}) or {}
    print(
        f"  {baseline:11s} {r['board_id'][:34]:34s} {r['status']:10s} "
        f"t={r['elapsed_routing_s']:6.1f}s "
        f"r={em.get('routability', '-')!s:6} "
        f"DRV[ep]={em.get('drv_errors_and_promoted_count', '-')!s} "
        f"Phi={em.get('final_potential', '-')!s}")


def run_one_seed(boards, *, baseline, dataset, grid, seed, seed_dir,
                 timeout_s, workers, do_eval, reward_config, check_angle,
                 via_prohibit, angle_90, max_passes, extra_env, use_gpu):
    raw_dir = seed_dir / "raw"
    routed_dir = seed_dir / "routed"
    eval_logs_dir = seed_dir / "eval" / "logs"
    for d in (raw_dir, routed_dir, eval_logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    algo = algorithm_tag(baseline, dataset, grid=grid)
    work = [
        dict(board=b, seed=seed, baseline=baseline, raw_dir=raw_dir,
             routed_dir=routed_dir, eval_logs_dir=eval_logs_dir, algo=algo,
             timeout_s=timeout_s, do_eval=do_eval, reward_config=reward_config,
             check_angle=check_angle, via_prohibit=via_prohibit,
             angle_90=angle_90, max_passes=max_passes, extra_env=extra_env,
             use_gpu=use_gpu)
        for b in boards]
    results = []
    if workers <= 1:
        for w in work:
            r = _process_one_board(w)
            _emit_progress(r, baseline)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process_one_board, w): w["board"].board_id for w in work}
            for fut in as_completed(futs):
                r = fut.result()
                _emit_progress(r, baseline)
                results.append(r)
    if do_eval:
        _aggregate_summary(eval_logs_dir, seed_dir,
                           dataset_label=dataset_tag(dataset, grid=grid),
                           algorithm_label=algo)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True,
                    choices=("freerouting", "krt", "orthoroute"))
    ap.add_argument("--dataset", required=True,
                    choices=("pcbench", "synthetic_1l", "synthetic_2l"))
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", action="append", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=None,
                    help=f"Per-board timeout (s). Defaults: {DEFAULT_TIMEOUTS}.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--reward-config", default="drc_dense_promoted")
    ap.add_argument("--check-angle", type=int, choices=(45, 90), default=45)
    ap.add_argument("--max-passes", type=int, default=10)
    gpu_grp = ap.add_mutually_exclusive_group()
    gpu_grp.add_argument("--use-gpu", dest="use_gpu", action="store_true",
                         help="OrthoRoute: enable GPU. Faster but non-deterministic; "
                              "small per-board DRV wobble (~+1 segment) can occur "
                              "vs CPU/paper-canonical numbers.")
    gpu_grp.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                         default=False, help="OrthoRoute: force CPU-only (default). "
                                             "Bit-perfect reproducible.")
    args = ap.parse_args()
    via_prohibit = False
    angle_90 = False
    if args.dataset == "synthetic_1l":
        if args.baseline != "freerouting":
            ap.error("synthetic_1l only supports --baseline freerouting "
                     "(via prohibited, 90-deg routing only).")
        if args.grid is None:
            ap.error("synthetic_1l requires --grid {10,50,100,200,500}.")
        via_prohibit = True
        angle_90 = True
        if args.check_angle == 45:
            args.check_angle = 90
    timeout_s = args.timeout if args.timeout is not None else DEFAULT_TIMEOUTS[args.baseline]
    boards = select_boards(args.dataset, grid=args.grid, split=args.split,
                           limit=args.limit, sample_substrings=args.sample)
    if not boards:
        ap.error("no boards matched the selection.")
    valid, missing = [], []
    for b in boards:
        need = []
        if not b.raw_pcb.is_file():
            need.append(f"raw_pcb={b.raw_pcb}")
        if not b.pro.is_file():
            need.append(f"pro={b.pro}")
        if args.baseline in ("freerouting", "krt") and not b.dsn.is_file():
            need.append(f"dsn={b.dsn}")
        if args.baseline == "orthoroute" and not b.orp.is_file():
            need.append(f"orp={b.orp}")
        if need:
            missing.append((b.board_id, ", ".join(need)))
        else:
            valid.append(b)
    if missing:
        log.warning("Skipping %d board(s) with missing inputs:\n%s",
                    len(missing),
                    "\n".join(f"  {bid}: {why}" for bid, why in missing))
    if not valid:
        ap.error("no boards remain after dropping incomplete inputs.")
    ds_tag = dataset_tag(args.dataset, grid=args.grid)
    algo_tag = algorithm_tag(args.baseline, args.dataset, grid=args.grid)
    run_root = Path(args.output_root) / ds_tag / algo_tag
    log.info("=== %s / %s ===", ds_tag, algo_tag)
    log.info("boards: %d  workers: %d  timeout: %ds  seeds: %d  gpu: %s",
             len(valid), args.workers, timeout_s, args.seeds, args.use_gpu)
    log.info("output: %s", run_root)
    seed_count = args.seeds if args.baseline == "freerouting" else 1
    if seed_count != args.seeds and args.seeds > 1:
        log.info("--seeds %d collapsed to 1 (%s is deterministic).",
                 args.seeds, args.baseline)
    all_results = {}
    for seed in range(seed_count):
        seed_dir = run_root / f"seed{seed}"
        log.info("--- seed %d -> %s", seed, seed_dir)
        results = run_one_seed(
            valid, baseline=args.baseline, dataset=args.dataset, grid=args.grid,
            seed=seed, seed_dir=seed_dir, timeout_s=timeout_s,
            workers=args.workers, do_eval=not args.no_eval,
            reward_config=args.reward_config, check_angle=args.check_angle,
            via_prohibit=via_prohibit, angle_90=angle_90,
            max_passes=args.max_passes, extra_env={}, use_gpu=args.use_gpu)
        all_results[seed] = results
        (seed_dir / "manifest.json").write_text(json.dumps({
            "dataset": args.dataset, "dataset_tag": ds_tag,
            "baseline": args.baseline, "algorithm_tag": algo_tag,
            "grid": args.grid, "split": args.split, "seed": seed,
            "n_boards": len(valid),
            "boards": [b.board_id for b in valid],
            "timeout_s": timeout_s,
            "via_prohibit": via_prohibit, "angle_90": angle_90,
            "max_passes": args.max_passes, "use_gpu": args.use_gpu,
            "reward_config": args.reward_config,
            "check_angle": args.check_angle, "results": results,
        }, indent=2, default=str))
    ok = sum(1 for rrs in all_results.values() for r in rrs if r["status"] == "ok")
    total = sum(len(r) for r in all_results.values())
    log.info("done: %d/%d ok across %d seed(s).", ok, total, seed_count)
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
