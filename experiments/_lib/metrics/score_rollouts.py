"""Common (4) eval stage — score every routed ``.kicad_pcb`` under a rollout
root with ``eval.metrics.evaluate_one`` so all methods (PCBWORLD,
CAD-Gen Code-level, API-Seq API-level, RL, search baselines) flow through
the same scorer and compare apples-to-apples.

All quickstart wrappers normalise their dumps to a single canonical
layout under ``--rollout-root`` — this scorer expects only that shape::

    <root>/per_board/<board_id>/sample_NN.kicad_pcb              # direct
    <root>/<scenario>/per_board/<board_id>/sample_NN.kicad_pcb   # multi

The "direct" form fires when the wrapper output itself is the eval-type
root (e.g. ``$OUT/codelevel/`` on its own). The "multi" form fires when
the rollout root holds several eval-types side by side (e.g. ``$OUT/``
with ``codelevel/``, ``apilevel/``, ``pcbworld/`` next to each other).
Both end in the same ``per_board/<id>/sample_NN.kicad_pcb`` shape and
recover the source ``.kicad_pro`` either as a sibling of the sample (the
PCBWORLD wrapper's post-process moves the triple in together) or from
``aggregate.json::source_path`` (CAD-Gen / API-Seq direct dumps).

Each ``evaluate_one`` call returns the same metric dict — ``success``
(connectivity), ``clean_pass`` (success AND total_drv_count == 0),
``routability``, ``drv_errors_{only,and_promoted}_count``,
``track_angle_drv``, ``final_potential`` — so the per-board / per-scenario
aggregates surface P@k (= ``pass_at_k``) and CP@k (= ``clean_pass_at_k``)
columns uniformly across methods.

Outputs (parallel tree under ``--out``, default ``<root>_eval``):

  per_board/<board_id>/<sample_stem>.json    # full evaluate_one result
  per_board/<board_id>/aggregate.json        # board-level pass_at_k, ...
  summary.csv                                 # boards × aggregate columns
  overall.json                                # scenario-level fmean

Re-runs are safe — the original rollout dir is never modified.

Usage:
    python experiments/_lib/metrics/score_rollouts.py \\
        --rollout-root "$EXPR_ROOT/table1/llm/gpt54mini/d2a" \\
        --out          "$EXPR_ROOT/table1/llm/gpt54mini/t2_eval"
        [--check-angle 45] [--reward-config drc_dense_promoted] [--limit 10]
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # llm_eval→paper_repro→scripts→repo
_KICAD_RL_DIR = _PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"
for p in (_PROJECT_ROOT, _KICAD_RL_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_source_pro(sample_pcb: Path, board_dir: Path | None) -> Path | None:
    """Locate the source .kicad_pro for one routed sample.

    Two recovery paths, tried in order:
      1. Sibling of the sample (PCBWORLD wrapper post-process moves the
         PCBWorld-emitted .pcb/.pro/.prl triple into per_board/<id>/).
      2. ``aggregate.json::source_path`` next to the per-board dir — used
         by CAD-Gen / API-Seq which keeps the .kicad_pro at the
         dataset side, not next to the generated sample.

    Returns ``None`` if neither is found; evaluate_one will then fall back
    to whatever the .kicad_pcb's embedded settings provide.
    """
    sibling = sample_pcb.with_suffix(".kicad_pro")
    if sibling.exists():
        return sibling
    if board_dir is None:
        return None
    agg = board_dir / "aggregate.json"
    if not agg.exists():
        return None
    try:
        d = json.loads(agg.read_text())
    except Exception:
        return None
    src = d.get("source_path", "")
    if not src:
        return None
    pro = Path(src).with_suffix(".kicad_pro")
    return pro if pro.exists() else None


def _failure_record(error: str) -> dict:
    """Shape-compatible empty metrics dict for a failed sample."""
    return {
        "success": False,
        "routability": 0.0,
        "track_count": 0,
        "via_count": 0,
        "wirelength_mm": 0.0,
        "drv_errors_only_count": 0,
        "drv_errors_and_promoted_count": 0,
        "drv_count": 0,
        "track_angle_drv": {"mode": 0, "source": "", "count": 0, "violations": []},
        "clean_pass": False,
        "final_potential": 0.0,
        "phi_components": {},
        "extras": {},
        "reeval_ok": False,
        "error": error,
    }


def _evaluate_sample(
    sample_pcb: Path,
    source_pro: Path | None,
    reward_config: str,
    check_angle: int,
) -> dict:
    """Wrap eval.metrics.evaluate_one with our metric shape.

    Maps the new evaluator's keys onto the existing per-sample schema so
    BoardResult.aggregate (downstream) keeps working without changes:
      drv_count := drv_errors_and_promoted_count (the broader of the two
                   counters — same convention as drc_dense_promoted reward
                   used for v54-class training).
    """
    from eval.metrics import evaluate_one as _eval_routed

    try:
        m = _eval_routed(
            str(sample_pcb),
            str(source_pro) if source_pro else None,
            reward_config_name=reward_config,
            check_angle=check_angle,
        )
    except Exception as exc:
        traceback.print_exc()
        return _failure_record(f"{type(exc).__name__}: {exc}")

    track_angle = m.get("track_angle_drv", {}) or {}
    return {
        # Core (BoardResult.aggregate keys)
        "success": bool(m["success"]),
        "routability": float(m["routability"]),
        "track_count": int(m["track_count"]),
        "via_count": int(m["via_count"]),
        "wirelength_mm": float(m["wirelength_mm"]),
        "drv_count": int(m.get("drv_errors_and_promoted_count", 0)),
        # New evaluator's additions, preserved verbatim
        "drv_errors_only_count": int(m.get("drv_errors_only_count", 0)),
        "drv_errors_and_promoted_count": int(m.get("drv_errors_and_promoted_count", 0)),
        "track_angle_drv": track_angle,
        "track_angle_drv_count": int(track_angle.get("count", 0)),
        "clean_pass": bool(m.get("clean_pass", False)),
        "final_potential": float(m.get("final_potential", 0.0)),
        "phi_components": m.get("phi_components", {}),
        "phi_weights": m.get("phi_weights", {}),
        "drc_violations": m.get("drc_violations", []),
        "extras": {
            "board": m.get("board"),
            "pro": m.get("pro"),
            "reward_config": m.get("reward_config"),
        },
        "reeval_ok": True,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Per-board aggregate (mirrors eval_cadgen_llm BoardResult.aggregate but
# adds the new clean_pass / track_angle_drv summaries).
# ---------------------------------------------------------------------------

@dataclass
class BoardResult:
    board_id: str
    source_path: str
    samples: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def k(self) -> int:
        return len(self.samples)

    @property
    def successes(self) -> int:
        return sum(1 for s in self.samples if s.get("success"))

    @property
    def clean_passes(self) -> int:
        return sum(1 for s in self.samples if s.get("clean_pass"))

    def aggregate(self) -> dict:
        k = self.k
        rb = [float(s.get("routability", 0.0)) for s in self.samples]
        cs = [bool(s.get("clean_pass", False)) for s in self.samples]
        fp = [float(s.get("final_potential", 0.0)) for s in self.samples]
        agg = {
            "board_id": self.board_id,
            "source_path": self.source_path,
            "num_samples": k,
            "successes": self.successes,
            "k": k,
            # Connectivity-only pass@k (v1/v3 definition).
            "pass_at_k": int(self.successes > 0),
            "success_at_k": int(self.successes > 0),
            "mean_success_rate": (self.successes / k) if k else 0.0,
            # Clean (= connectivity AND no DRC + no angle DRV) pass@k.
            "clean_pass_at_k": int(self.clean_passes > 0),
            "clean_mean_success_rate": (self.clean_passes / k) if k else 0.0,
            # Routability
            "routability_at_k_best": max(rb) if rb else 0.0,
            "routability_at_k_mean": (sum(rb) / k) if k else 0.0,
            # DRV / angle
            "drv_at_k_min": min((s.get("drv_count", 0) for s in self.samples), default=0),
            "track_angle_drv_at_k_min": min(
                (s.get("track_angle_drv_count", 0) for s in self.samples), default=0,
            ),
            # Reward
            "final_potential_at_k_best": max(fp) if fp else 0.0,
            "final_potential_at_k_mean": (sum(fp) / k) if k else 0.0,
            # Wirelength on success-only samples
            "wirelength_at_k_best": min(
                (s.get("wirelength_mm", 0.0) for s in self.samples
                 if s.get("success")),
                default=0.0,
            ),
            "error": self.error,
        }
        return agg


_SUMMARY_FIELDS = [
    "board_id",
    "source_path",
    "num_samples",
    "successes",
    "k",
    "pass_at_k",
    "mean_success_rate",
    "clean_pass_at_k",
    "clean_mean_success_rate",
    "routability_at_k_best",
    "routability_at_k_mean",
    "drv_at_k_min",
    "track_angle_drv_at_k_min",
    "final_potential_at_k_best",
    "final_potential_at_k_mean",
    "wirelength_at_k_best",
    "error",
]


def _write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _SUMMARY_FIELDS})


def _overall(rows: list[dict]) -> dict:
    if not rows:
        return {"boards_evaluated": 0}

    def _arr(key):
        return [r[key] for r in rows if key in r]

    pass_k = _arr("pass_at_k")
    clean_k = _arr("clean_pass_at_k")
    mean_succ = _arr("mean_success_rate")
    clean_succ = _arr("clean_mean_success_rate")
    rb_best = _arr("routability_at_k_best")
    rb_mean = _arr("routability_at_k_mean")
    fp_best = _arr("final_potential_at_k_best")
    fp_mean = _arr("final_potential_at_k_mean")
    drv_min = _arr("drv_at_k_min")
    angle_min = _arr("track_angle_drv_at_k_min")

    out = {
        "boards_evaluated": len(rows),
        "pass_at_k": statistics.fmean(pass_k),
        "clean_pass_at_k": statistics.fmean(clean_k),
        "mean_success_rate": statistics.fmean(mean_succ),
        "clean_mean_success_rate": statistics.fmean(clean_succ),
        "routability_at_k_best_mean": statistics.fmean(rb_best),
        "routability_at_k_best_std": statistics.pstdev(rb_best) if len(rb_best) > 1 else 0.0,
        "routability_at_k_mean_mean": statistics.fmean(rb_mean),
        "routability_at_k_mean_std": statistics.pstdev(rb_mean) if len(rb_mean) > 1 else 0.0,
        "final_potential_at_k_best_mean": statistics.fmean(fp_best),
        "final_potential_at_k_mean_mean": statistics.fmean(fp_mean),
        "drv_at_k_min_mean": statistics.fmean(drv_min),
        "track_angle_drv_at_k_min_mean": statistics.fmean(angle_min),
    }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _walk_scenarios(input_dir: Path):
    """Yield ``(scenario_dir, board_groups)`` tuples for every per_board tree.

    Both supported shapes end the same way (``per_board/<id>/sample_NN.kicad_pcb``);
    we only differ in whether the rollout root is itself the scenario dir
    or holds several scenario subdirs side by side.

    ``board_groups`` is ``[(board_id, per_board_dir, [sample_pcbs])]``. The
    per-board dir is used downstream to recover the source ``.kicad_pro``
    (either from a sibling that the wrapper moved in, or from
    ``aggregate.json::source_path`` when CAD-Gen / API-Seq produced the dump).
    """
    direct = input_dir / "per_board"
    if direct.is_dir():
        yield input_dir, _per_board_groups(direct)
        return

    scenarios = sorted(
        s for s in input_dir.iterdir()
        if s.is_dir() and (s / "per_board").is_dir()
    )
    for s in scenarios:
        yield s, _per_board_groups(s / "per_board")


def _per_board_groups(per_board_root: Path) -> list[tuple[str, Path, list[Path]]]:
    """[(board_id, per_board_dir, [sample_pcbs])] for the common layout."""
    out: list[tuple[str, Path, list[Path]]] = []
    for board_dir in sorted(p for p in per_board_root.iterdir() if p.is_dir()):
        pcbs = sorted(board_dir.glob("sample_*.kicad_pcb"))
        if pcbs:
            out.append((board_dir.name, board_dir, pcbs))
    return out


def reeval_scenario(
    scenario_in: Path,
    scenario_out: Path,
    board_groups: list[tuple[str, Path | None, list[Path]]],
    reward_config: str,
    check_angle: int,
    limit: int,
) -> dict:
    """Re-evaluate one scenario; return overall stats dict.

    ``board_groups`` is what ``_walk_scenarios`` yields:
        [(board_id, per_board_dir, [sample_pcbs])]
    """
    out_per_board = scenario_out / "per_board"

    boards = list(board_groups)
    if limit > 0:
        boards = boards[:limit]

    print(f"\n{'='*60}")
    print(f"  scenario : {scenario_in.name}")
    print(f"  boards   : {len(boards)}")
    print(f"  out      : {scenario_out}")
    print(f"  angle    : {check_angle}°    reward: {reward_config}")
    print("="*60)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_kw):  # type: ignore[no-redef]
            return it

    rows: list[dict] = []
    n_samples_total = 0
    t0 = time.time()
    for board_id, board_dir, sample_pcbs in tqdm(boards, desc=scenario_in.name, unit="board"):
        out_board_dir = out_per_board / board_id
        out_board_dir.mkdir(parents=True, exist_ok=True)

        # Recover source_path for the aggregate row + a shared source_pro
        # for the per-board scope. PCBWORLD looks up the .kicad_pro
        # per-sample from siblings below.
        # Record source_path (from aggregate.json, when present) so the
        # per-board CSV row points back to the dataset board.
        source_path = ""
        if board_dir is not None:
            agg = board_dir / "aggregate.json"
            if agg.exists():
                try:
                    source_path = json.loads(agg.read_text()).get("source_path", "")
                except Exception:
                    pass

        result = BoardResult(board_id=board_id, source_path=source_path)
        for sp in sample_pcbs:
            source_pro = _resolve_source_pro(sp, board_dir)
            metrics = _evaluate_sample(sp, source_pro, reward_config, check_angle)
            # All wrappers normalise samples to ``sample_NN`` stems, so the
            # trailing integer is the per-board rollout index.
            try:
                metrics["sample_idx"] = int(sp.stem.split("_")[-1])
            except ValueError:
                metrics["sample_idx"] = len(result.samples)
            metrics["sample_pcb"] = str(sp.resolve())
            (out_board_dir / f"{sp.stem}.json").write_text(json.dumps(metrics, indent=2))
            result.samples.append(metrics)
            n_samples_total += 1
            # KiCadEngine has a per-process singleton; nudge the GC so the
            # native BOARD pointer is released before the next iteration.
            gc.collect()

        if not result.samples:
            result.error = "no_samples"
        agg_row = result.aggregate()
        (out_board_dir / "aggregate.json").write_text(json.dumps(agg_row, indent=2))
        rows.append(agg_row)

    summary_path = scenario_out / "summary.csv"
    _write_summary(rows, summary_path)
    overall = _overall(rows)
    overall["scenario"] = scenario_in.name
    overall["reward_config"] = reward_config
    overall["check_angle"] = int(check_angle)
    overall["wall_time_sec"] = time.time() - t0
    overall["samples_evaluated"] = n_samples_total
    (scenario_out / "overall.json").write_text(json.dumps(overall, indent=2))
    return overall


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rollout-root", "--input-dir", dest="input_dir",
        type=Path, required=True,
        help="Rollout root containing per_board/<id>/sample_NN.kicad_pcb "
             "(direct), or <scenario>/per_board/<id>/sample_NN.kicad_pcb "
             "(one or more scenario subdirs side by side).",
    )
    p.add_argument(
        "--out", "--output-dir", dest="output_dir", type=Path, default=None,
        help="Where to write fresh sample JSONs / aggregates / summary.csv "
             "/ overall.json (default: <input-dir>_eval).",
    )
    p.add_argument("--check-angle", type=int, choices=(45, 90), default=45)
    p.add_argument("--reward-config", default="drc_dense_promoted")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, evaluate only the first N boards per scenario.")
    args = p.parse_args()

    args.input_dir = args.input_dir.resolve()
    if not args.input_dir.is_dir():
        print(f"[ERROR] not a directory: {args.input_dir}", file=sys.stderr)
        return 2
    if args.output_dir is None:
        args.output_dir = args.input_dir.parent / f"{args.input_dir.name}_eval"
    args.output_dir = args.output_dir.resolve()

    print(f"input  : {args.input_dir}")
    print(f"output : {args.output_dir}")

    overalls: list[dict] = []
    scenarios = list(_walk_scenarios(args.input_dir))
    if not scenarios:
        print(
            f"[ERROR] no per_board/<id>/sample_*.kicad_pcb tree under {args.input_dir}",
            file=sys.stderr,
        )
        return 2
    for scenario_in, board_groups in scenarios:
        rel = (scenario_in.relative_to(args.input_dir)
               if scenario_in != args.input_dir else Path("."))
        scenario_out = args.output_dir / rel
        overalls.append(reeval_scenario(
            scenario_in, scenario_out, board_groups,
            args.reward_config, args.check_angle, args.limit,
        ))

    print()
    print("=" * 60)
    print("  Summary across scenarios")
    print("=" * 60)
    for o in overalls:
        print(f"  {o['scenario']:<32}  "
              f"pass={o.get('pass_at_k', 0):.3f}  "
              f"clean_pass={o.get('clean_pass_at_k', 0):.3f}  "
              f"rout_best={o.get('routability_at_k_best_mean', 0):.3f}  "
              f"angle_drv_min={o.get('track_angle_drv_at_k_min_mean', 0):.2f}  "
              f"final_phi_best={o.get('final_potential_at_k_best_mean', 0):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
