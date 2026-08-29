"""Route the three rule-based routers and emit rollout-stage artifacts only.

conda env: the single ``cadagent`` env covers this too -- ``environment.yml``
carries openjdk 21 (Freerouting) and rust (KRT), and cupy/scipy/shapely are in
the pip lock (externals via ``tools/setup/fetch_baselines.sh``). The legacy
separate ``cadagent-classical`` env still works but is not the standard. E.g.::

    conda activate cadagent
    python eval/rollout/rule_based.py --baseline krt --dataset synthetic_2l --limit 1

A standalone routing+conversion entrypoint built on
``methods/baselines/rule_based/run_rule_based_routers.py``. Per board it:

  1. Locates the raw .kicad_pcb (+ .kicad_pro / .dsn / .orp) via
     ``_lib.datasets``.
  2. Drives the chosen baseline -> raw output (.ses / .ORS / .kicad_pcb),
     reusing the proven ``_route_*`` functions from ``run_rule_based_routers`` verbatim.
  3. Converts the raw output -> .kicad_pcb (SES/ORS via ``_converters``).
  4. Saves the routed .kicad_pcb in the SAME shape/location that ``eval/pipeline.py``
     produces in its rollout stage, so a separate eval pass can consume it.

It deliberately stops there -- NO DRC / reward / aggregation. Score the saved
artifacts with your own eval code. Each ``seed<N>`` directory is a
``eval/pipeline.py``-compatible rollout dir, so you can also run e.g.::

    python -m eval.pipeline --skip-rollout \
        --output-dir <.../seed0> --boards-dir <dataset root>

Output layout (mirrors eval/pipeline.py rollout stage):

    <output-root>/<dataset_tag>/<algorithm_tag>/seed<N>/
      ├── artifacts/board_{board_index:05d}_rollout_0.kicad_pcb
      │            board_{board_index:05d}_rollout_0.kicad_pro   (co-located)
      ├── raw/      (.ses / .ORS + routing logs, for debugging)
      ├── per_rollout.csv   (eval.eval_utils.PER_ROLLOUT_FIELDS; DRC columns blank)
      └── manifest.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent            # <repo root>/eval/rollout
_PROJECT_ROOT = _THIS_DIR.parents[1]                   # <repo root>
_BASELINES_DIR = _PROJECT_ROOT / "methods" / "baselines" / "rule_based"
# _BASELINES_DIR -> `import run_rule_based_routers`; _PROJECT_ROOT -> `methods.baselines.rule_based.*`
# / `eval.*`; _THIS_DIR covers running this file directly as a script.
for _p in (_BASELINES_DIR, _PROJECT_ROOT, _THIS_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import run_rule_based_routers as rb  # noqa: E402  (routing functions reused verbatim)
from methods.baselines.rule_based._lib.datasets import (  # noqa: E402
    algorithm_tag, dataset_tag, select_boards,
)
from eval import eval_utils as u  # noqa: E402  (PER_ROLLOUT_FIELDS, write_csv)

log = logging.getLogger("rollout.rule_based")

# Kept separate from run_rule_based_routers's _run_outputs so the two layouts (this one
# has artifacts/ + per_rollout.csv; run_rule_based_routers has routed/ + eval/) never mix.
DEFAULT_OUTPUT_ROOT = Path("methods/baselines/rule_based") / "_run_outputs_rule_based"
from configs.loader.schema import SCHEMA_VERSION  # noqa: E402 — shared manifest version

# Map a run_rule_based_routers routing status onto the rollout-row status vocabulary that
# eval/pipeline.py's post-hoc DRC understands ("completed" rows with an
# artifact_path get scored; everything else is skipped).
_STATUS_MAP = {
    "ok":        ("completed", dict(terminated=True)),
    "timeout":   ("truncated", dict(truncated=True)),
    "no_output": ("crashed",   dict(crashed=True)),
    "error":     ("crashed",   dict(crashed=True)),
}


def _build_row(board, idx, seed, result, artifacts_dir, deterministic):
    """Turn a run_rule_based_routers ``BoardResult`` into one PER_ROLLOUT_FIELDS row.

    On success the routed .kicad_pcb is renamed to the rollout naming
    convention and its .kicad_pro is co-located next to it (so
    ``eval.eval_utils.resolve_pro_path`` finds the pro by its first rule).
    DRC / reward columns are left blank for the downstream eval to fill.
    """
    row = {field: "" for field in u.PER_ROLLOUT_FIELDS}
    status, flags = _STATUS_MAP.get(result.status, ("crashed", dict(crashed=True)))
    error = result.error
    artifact_path = ""
    if result.status == "ok" and result.routed_pcb:
        src = Path(result.routed_pcb)
        dst = artifacts_dir / f"board_{idx:05d}_rollout_0.kicad_pcb"
        try:
            if src.resolve() != dst.resolve():
                os.replace(src, dst)
            if board.pro.is_file():
                shutil.copy2(board.pro, dst.with_suffix(".kicad_pro"))
            artifact_path = str(dst.resolve())
        except OSError as exc:
            status, flags = "crashed", dict(crashed=True)
            error = f"artifact move/copy: {exc}"
    row.update({
        "board_index": idx,
        "board_id": board.board_id,
        "board_path": str(board.raw_pcb),
        "rollout_idx": 0,
        "seed": seed,
        "deterministic": deterministic,
        "status": status,
        "skip_reason": "" if result.status == "ok" else result.status,
        "terminated": flags.get("terminated", False),
        "truncated": flags.get("truncated", False),
        "crashed": flags.get("crashed", False),
        "engine_crash": False,
        "per_board_rollout_time": result.elapsed_routing_s,
        "rollout_finish_time": time.time(),
        "artifact_path": artifact_path,
        "error": error,
    })
    return row


def _route_and_emit(job):
    board = job["board"]; idx = job["idx"]; seed = job["seed"]
    baseline = job["baseline"]; raw_dir = job["raw_dir"]
    artifacts_dir = job["artifacts_dir"]; algo = job["algo"]
    timeout_s = job["timeout_s"]
    if baseline == "freerouting":
        result = rb._route_freerouting(
            board, seed, raw_dir, artifacts_dir, algo, timeout_s,
            job["via_prohibit"], job["angle_90"], job["max_passes"], {})
    elif baseline == "krt":
        result = rb._route_krt(board, seed, raw_dir, artifacts_dir, algo, timeout_s)
    elif baseline == "orthoroute":
        result = rb._route_orthoroute(
            board, seed, raw_dir, artifacts_dir, algo, timeout_s,
            use_gpu=job["use_gpu"])
    else:
        result = rb.BoardResult(board.board_id, seed, "error", 0.0,
                                error=f"unknown baseline {baseline!r}")
    row = _build_row(board, idx, seed, result, artifacts_dir, job["deterministic"])
    gc.collect()
    return row


def _emit_progress(row, baseline):
    detail = (f"-> {Path(row['artifact_path']).name}"
              if row["artifact_path"] else (row.get("error", "") or "")[:40])
    print(f"  {baseline:11s} {str(row['board_id'])[:34]:34s} "
          f"{row['status']:10s} t={float(row['per_board_rollout_time'] or 0):6.1f}s "
          f"{detail}")


def run_one_seed(boards, *, baseline, dataset, grid, seed, seed_dir, timeout_s,
                 workers, via_prohibit, angle_90, max_passes, use_gpu, deterministic):
    raw_dir = seed_dir / "raw"
    artifacts_dir = seed_dir / "artifacts"
    for d in (raw_dir, artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)
    algo = algorithm_tag(baseline, dataset, grid=grid)
    jobs = [
        dict(board=b, idx=i, seed=seed, baseline=baseline, raw_dir=raw_dir,
             artifacts_dir=artifacts_dir, algo=algo, timeout_s=timeout_s,
             via_prohibit=via_prohibit, angle_90=angle_90, max_passes=max_passes,
             use_gpu=use_gpu, deterministic=deterministic)
        for i, b in enumerate(boards)]
    rows = []
    if workers <= 1:
        for j in jobs:
            r = _route_and_emit(j)
            _emit_progress(r, baseline)
            rows.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_route_and_emit, j): j["idx"] for j in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                _emit_progress(r, baseline)
                rows.append(r)
    rows.sort(key=lambda r: r["board_index"])
    u.write_csv(seed_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    return rows


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
                    help=f"Per-board timeout (s). Defaults: {rb.DEFAULT_TIMEOUTS}.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Write directly into this rollout dir (eval/pipeline.py "
                         "--output-dir style). With multiple seeds, seed<N>/ is "
                         "nested under it. Overrides --output-root.")
    ap.add_argument("--max-passes", type=int, default=10)
    gpu_grp = ap.add_mutually_exclusive_group()
    gpu_grp.add_argument("--use-gpu", dest="use_gpu", action="store_true",
                         help="OrthoRoute: enable GPU (faster, non-deterministic).")
    gpu_grp.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                         default=False, help="OrthoRoute: force CPU-only (default).")
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

    timeout_s = (args.timeout if args.timeout is not None
                 else rb.DEFAULT_TIMEOUTS[args.baseline])
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
    # Freerouting is run across independent seeds (multithreaded -> not
    # bit-reproducible); the others are single-shot deterministic, so seeds
    # collapse to 1. GPU OrthoRoute is the one deterministic-router exception.
    seed_count = args.seeds if args.baseline == "freerouting" else 1
    if seed_count != args.seeds and args.seeds > 1:
        log.info("--seeds %d collapsed to 1 (%s is deterministic).",
                 args.seeds, args.baseline)
    deterministic = {
        "freerouting": seed_count == 1,
        "krt": True,
        "orthoroute": not args.use_gpu,
    }[args.baseline]

    if args.output_dir is not None:
        base = Path(args.output_dir)
        run_label = str(base)
    else:
        base = Path(args.output_root) / ds_tag / algo_tag
        run_label = str(base)
    log.info("=== %s / %s ===", ds_tag, algo_tag)
    log.info("boards: %d  workers: %d  timeout: %ds  seeds: %d  gpu: %s",
             len(valid), args.workers, timeout_s, seed_count, args.use_gpu)
    log.info("output: %s", run_label)

    ok = total = 0
    for seed in range(seed_count):
        if args.output_dir is not None:
            seed_dir = base if seed_count == 1 else base / f"seed{seed}"
        else:
            seed_dir = base / f"seed{seed}"
        log.info("--- seed %d -> %s", seed, seed_dir)
        rows = run_one_seed(
            valid, baseline=args.baseline, dataset=args.dataset, grid=args.grid,
            seed=seed, seed_dir=seed_dir, timeout_s=timeout_s,
            workers=args.workers, via_prohibit=via_prohibit, angle_90=angle_90,
            max_passes=args.max_passes, use_gpu=args.use_gpu,
            deterministic=deterministic)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "env_kwargs": {},
            "args": {
                "generator": "rollout/rule_based.py",
                "stage": "rollout_only",
                "baseline": args.baseline,
                "dataset": args.dataset,
                "dataset_tag": ds_tag,
                "algorithm_tag": algo_tag,
                "grid": args.grid,
                "split": args.split,
                "seed": seed,
                "n_boards": len(valid),
                "boards": [b.board_id for b in valid],
                "timeout_s": timeout_s,
                "via_prohibit": via_prohibit,
                "angle_90": angle_90,
                "max_passes": args.max_passes,
                "use_gpu": args.use_gpu,
                "deterministic": deterministic,
                "artifact_naming": "board_{board_index:05d}_rollout_0.kicad_pcb",
                "note": ("Routed .kicad_pcb artifacts only (no DRC/eval). This "
                         "dir is an eval/pipeline.py-compatible rollout dir: score "
                         "it with your own eval, e.g. `python eval/pipeline.py "
                         "--skip-rollout --output-dir <this dir> --boards-dir "
                         "<dataset root>`. A .kicad_pro is co-located next to "
                         "each artifact for pro resolution."),
            },
        }
        (seed_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str))
        ok += sum(1 for r in rows if r["status"] == "completed")
        total += len(rows)

    log.info("done: %d/%d routed (status=completed) across %d seed(s).",
             ok, total, seed_count)
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
