#!/usr/bin/env python3
"""Shared helpers for the KDD/PCBWorld paper metric-extraction scripts.

This is a *thin* layer over ``eval/eval_utils.py``; it adds no new metric math.
It only (a) loads a cell's ``per_rollout.csv``, (b) groups rows by
``(board_id, seed)`` using the canonical artifact-name contract, (c) reduces to a
single ``final_potential``-winner per (board, seed) and reads every reported
metric off that winner, and (d) aggregates board-means into per-seed scalars so
the across-seed mean / sample-std match the paper's convention.

Locked conventions:

* ``best = final_potential winner`` for *every* metric, including ``clean_pass``.
  The reward potential is constructed so that if any rollout is a clean pass, the
  potential winner is also a clean pass -- so ``winner.clean_pass`` == CP@5.
* WL / Via / DRV are taken from that winner and averaged over *all* boards
  (no routed-only filter).
* ``±`` is the sample std (ddof=1) across the (typically 4) seeds; single-seed
  cells report ``†`` (std undefined).
* ``time`` is read only from ``per_board_rollout_time`` (mean over a board's
  rollouts). LLM and rule-based cells carry no such column; their timing comes
  from the model-level lookups below (:func:`llm_time_for` /
  :func:`rule_based_time_for`).

READ-ONLY GUARANTEE: everything under ``experimental_results/`` is only ever
opened for reading (``open_ro``); all writes go to ``paper_outputs/`` and are
checked by ``assert_output_path``. Scripts must not mutate experiment results.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

# --- repo import path: experiments/_lib/metrics/common.py -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import eval_utils as u  # noqa: E402  (read_csv/parse_metric/sample_std/...)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
EXP = (REPO_ROOT / "var" / "results" / "kdd").resolve()
PAPER_OUT = (REPO_ROOT / "var" / "results" / "kdd" / "paper_outputs").resolve()


# Task IDs are d1/d2a/d3* in code; the read-only var/results/kdd tree stores
# them as t1/t2/t3*. Reads fall back to the t-series spelling; outputs written
# here use the d-names.
_LEGACY_TASK_DIRS = {"d1": "t1", "d2a": "t2",
                     "d3": "t3", "d3a": "t3a", "d3b": "t3b", "d3c": "t3c"}
# LLM cell prefixes are interactive_/plan_only_/engine_free_ in code; on disk
# the same cells carry pcbworld_/apiseq_/cadgen_.
_LEGACY_CELL_PREFIXES = {"interactive_": "pcbworld_", "plan_only_": "apiseq_",
                         "engine_free_": "cadgen_"}
# Exact-name cell aliases: the ``reference`` cell is stored as ``human``.
_LEGACY_CELL_NAMES = {"reference": "human"}


def legacy_rel(rel: str | Path) -> str:
    """Translate a d-series rel path to the t-series spelling used on disk."""
    parts = []
    for tok in Path(rel).parts:
        if tok in _LEGACY_TASK_DIRS:
            tok = _LEGACY_TASK_DIRS[tok]
        elif tok.startswith("d1_grid"):
            tok = "t1_grid" + tok[len("d1_grid"):]
        elif tok in _LEGACY_CELL_NAMES:
            tok = _LEGACY_CELL_NAMES[tok]
        else:
            for new, old in _LEGACY_CELL_PREFIXES.items():
                if tok.startswith(new):
                    tok = old + tok[len(new):]
                    break
        parts.append(tok)
    return str(Path(*parts)) if parts else str(rel)


def canon_rel(rel: str | Path) -> str:
    """Inverse of :func:`legacy_rel` — t-series disk spelling to d-series."""
    inv_task = {v: k for k, v in _LEGACY_TASK_DIRS.items()}
    inv_pre = {v: k for k, v in _LEGACY_CELL_PREFIXES.items()}
    inv_name = {v: k for k, v in _LEGACY_CELL_NAMES.items()}
    parts = []
    for tok in Path(rel).parts:
        if tok in inv_task:
            tok = inv_task[tok]
        elif tok in inv_name:
            tok = inv_name[tok]
        elif tok.startswith("t1_grid"):
            tok = "d1_grid" + tok[len("t1_grid"):]
        else:
            for old, new in inv_pre.items():
                if tok.startswith(old):
                    tok = new + tok[len(old):]
                    break
        parts.append(tok)
    return str(Path(*parts)) if parts else str(rel)


def cell_dir(rel: str) -> Path:
    """Absolute path to a cell from its rel path under ``experimental_results``.

    Falls back to the t-series task dir when the d-series path does not exist
    on disk."""
    p = (EXP / rel).resolve()
    if not p.exists():
        legacy = (EXP / legacy_rel(rel)).resolve()
        if legacy.exists():
            return legacy
    return p


def cell_name(rel_or_dir: str | Path) -> str:
    """Trailing path component (the canonical cell token)."""
    return Path(rel_or_dir).name


def disk_cell_name(rel: str) -> str:
    """Cell token as it appears in on-disk artifact names — the RESOLVED dir's
    basename. On-disk cells carry the pcbworld_/apiseq_/cadgen_ prefixes inside
    artifact filenames, so artifact parsing must use this, not
    :func:`cell_name`."""
    return cell_dir(rel).name


def cell_exists(rel: str) -> bool:
    """A cell is usable if it has a ``per_rollout.csv``."""
    return (cell_dir(rel) / "per_rollout.csv").is_file()


# --------------------------------------------------------------------------- #
# Read-only guards
# --------------------------------------------------------------------------- #
def _under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def open_ro(path: str | Path, binary: bool = False):
    """Open a file for reading only. Refuses if the target is under
    ``experimental_results`` and a write mode was somehow requested. This is the
    single choke-point through which experiment files are touched."""
    p = Path(path)
    mode = "rb" if binary else "r"
    if _under(p, EXP):
        # Defensive: ``open`` here is hard-coded read; this assert documents intent
        assert mode in ("r", "rb"), f"refusing non-read open under EXP: {p}"
    return p.open(mode, encoding=None if binary else "utf-8",
                  newline="" if not binary else None)


def assert_output_path(path: str | Path) -> Path:
    """Guarantee an output path lives under ``paper_outputs`` and *not* under the
    experiment-results tree. Returns the resolved path (parent dirs created)."""
    p = Path(path).resolve()
    # PAPER_OUT lives *under* EXP, so it is the one writable subtree carved out
    # of the otherwise read-only results tree.
    if _under(p, EXP) and not _under(p, PAPER_OUT):
        raise RuntimeError(f"output path is inside experimental_results (read-only!): {p}")
    if not _under(p, PAPER_OUT):
        raise RuntimeError(f"output path must be under {PAPER_OUT}: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Metric vocabulary  (logical name -> per_rollout.csv column)
# --------------------------------------------------------------------------- #
# DRV headline = errors-only (drv_errors_only_count) -- the same severity
# bucket clean_pass uses. (errors+promoted / total_drv are also available
# below for reference.)
METRIC_COL = {
    "clean_pass":          "clean_pass",
    "potential":           "final_potential",      # raw Phi (can be negative)
    "potential_gain":      "potential_gain",        # final - initial (bare board); reported headline
    "initial_potential":   "initial_potential",
    "routability":         "routability",
    "success":             "success",
    "wl":                  "wirelength_mm",
    "via":                 "via_count",
    "drv":                 "drv_errors_only_count",
    "drv_errors_promoted": "drv_errors_and_promoted_count",
    "total_drv":           "total_drv_count",
    "time":                "per_board_rollout_time",
}
WINNER_MODE = "final_potential"


# --------------------------------------------------------------------------- #
# Loading & grouping
# --------------------------------------------------------------------------- #
def load_rollouts(rel: str) -> list[dict[str, str]]:
    """All ``per_rollout.csv`` rows for a cell (read-only).

    A CSV carrying ``clean_success`` is exposed under the paper name
    ``clean_pass`` (var/ is read-only and never rewritten)."""
    path = cell_dir(rel) / "per_rollout.csv"
    with open_ro(path) as fh:
        import csv
        rows = list(csv.DictReader(fh))
    for row in rows:
        if "clean_pass" not in row and "clean_success" in row:
            row["clean_pass"] = row["clean_success"]
    return rows


def group_by_seed_board(rows: Sequence[dict], cell: str) -> dict[tuple[str, int], list[dict]]:
    """Group rows by ``(board_id, seed)`` using the artifact-name contract
    ``<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb`` (same parser as the aggregation
    pipeline). Rows whose artifact does not parse (crashes / no-geometry) are
    skipped, exactly as ``eval.aggregation.aggregate_boards`` does."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        p = u.parse_cell_artifact(r.get("artifact_path", "") or "", cell)
        if p is None:
            continue
        bid, seed, _rollout = p
        groups.setdefault((bid, seed), []).append(r)
    return groups


# A board whose *every* rollout failed to parse/evaluate is a real failure, so
# these success-type metrics are folded to 0 (rather than excluded) for it.
FOLD_ZERO_ON_ALLFAIL = ("clean_pass", "routability", "success")


def is_parse_fail(r: dict) -> bool:
    """A rollout whose generated .kicad_pcb could not be loaded/evaluated."""
    return ((r.get("eval_status") or "").strip() == "error"
            or bool((r.get("eval_error") or "").strip()))


def best_at_k(group: Sequence[dict]) -> dict[str, float | None]:
    """Reduce one (board, seed) group to its ``final_potential`` winner and read
    every reported metric off that single winning rollout. ``time`` is the mean
    of ``per_board_rollout_time`` over the group (not a winner quantity).

    If *all* rollouts in the group are parse-fails (unloadable board), the
    success-type metrics in ``FOLD_ZERO_ON_ALLFAIL`` are reported as ``0.0``
    instead of missing, so such boards count as failures in CP@5 / routability
    rather than being silently dropped. WL/Via/DRV/potential stay missing (a
    quality value is meaningless without a parseable board)."""
    grp = list(group)
    _, w = u.select_best_in_board(grp, WINNER_MODE)
    all_fail = bool(grp) and all(is_parse_fail(r) for r in grp)
    out: dict[str, float | None] = {}
    for name, col in METRIC_COL.items():
        if name == "time":
            tvals = [u.parse_metric(r.get(col)) for r in group]
            tvals = [t for t in tvals if t is not None]
            out["time"] = (sum(tvals) / len(tvals)) if tvals else None
        else:
            v = u.parse_metric((w or {}).get(col))
            if v is None and all_fail and name in FOLD_ZERO_ON_ALLFAIL:
                v = 0.0
            out[name] = v
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean(xs: Iterable[float | None]) -> float | None:
    vs = [x for x in xs if x is not None]
    return (sum(vs) / len(vs)) if vs else None


def board_namespace(rel: str) -> str:
    """Dir that defines a cell's board set (boards are shared across its methods):
    ``d2a/freerouting`` -> ``d2a``; ``d1/d1_grid10/sable`` -> ``d1/d1_grid10``;
    ``d3/d3a/reference`` -> ``d3/d3a``. Cache key prefix for initial_potential."""
    return str(Path(rel).parent)


_INIT_CACHE: dict[str, float] | None = None


def load_initial_cache() -> dict[str, float]:
    """Per-board ``initial_potential`` cache (``<namespace>::<board_id>`` -> Phi)
    produced by ``compute_initial_potential.py``. Empty dict if not built yet."""
    global _INIT_CACHE
    if _INIT_CACHE is None:
        p = PAPER_OUT / "initial_potential.json"
        try:
            with open_ro(p) as fh:
                import json
                _INIT_CACHE = json.load(fh)
        except FileNotFoundError:
            _INIT_CACHE = {}
    return _INIT_CACHE


# --------------------------------------------------------------------------- #
# LLM per-episode time (PCBWorld / API-level / Code-level cells)
# --------------------------------------------------------------------------- #
# The unified LLM cells are boards-only, so per_board_rollout_time is blank. The
# routing latency lives in the LLM eval aggregate csv as `sec/rollout`
# (= the paper's sec/ep), mapped onto each LLM cell by (interface, model, ns).
_LLM_TIME_CSV = (REPO_ROOT / "var" / "results" / "kdd" / "legacy"
                 / "aggregated_metrics" / "llm_eval_metrics.csv")
_LLM_DATASET_NS = {"synth2l": "d2a", "pcbench": "d3/d3a"}  # csv dataset -> namespace
_LLM_TIME: dict[tuple[str, str, str], float] | None = None


def _norm_model(m: str) -> str:
    return m.replace("-think-off", "").replace("-think-on", "").strip()


def load_llm_time() -> dict[tuple[str, str, str], float]:
    """(interface, model, namespace) -> mean sec/rollout, from the LLM eval
    aggregate csv. interface in {interactive, plan_only, engine_free};
    namespace in {d2a, d3/d3a}. Empty if the csv is absent."""
    global _LLM_TIME
    if _LLM_TIME is not None:
        return _LLM_TIME
    _LLM_TIME = {}
    try:
        import csv
        with open_ro(_LLM_TIME_CSV) as fh:
            for r in csv.DictReader(fh):
                model = r.get("model", "")
                t = u.parse_metric(r.get("sec/rollout"))
                ns = _LLM_DATASET_NS.get((r.get("dataset", "") or "").lower())
                if t is None or ns is None or not model:
                    continue
                # the csv names models with the on-disk cell prefixes
                if model.startswith("apiseq_"):
                    iface, mdl = "plan_only", model[len("apiseq_"):]
                elif model.startswith("cadgen_"):
                    iface, mdl = "engine_free", model[len("cadgen_"):]
                else:
                    iface, mdl = "interactive", model
                _LLM_TIME[(iface, _norm_model(mdl), ns)] = t
    except FileNotFoundError:
        pass
    return _LLM_TIME


def llm_time_for(rel: str) -> float | None:
    """sec/rollout for an LLM cell (``<ns>/<interface>_<model>``), else None."""
    leaf = cell_name(rel)
    legacy_iface = {"pcbworld": "interactive", "apiseq": "plan_only",
                    "cadgen": "engine_free"}
    for pre in ("interactive", "plan_only", "engine_free",
                "pcbworld", "apiseq", "cadgen"):
        if leaf.startswith(pre + "_"):
            iface = legacy_iface.get(pre, pre)
            mdl = _norm_model(leaf[len(pre) + 1:])
            ns = canon_rel(board_namespace(rel))
            return load_llm_time().get((iface, mdl, ns))
    return None


# KRT / OrthoRoute cells carry no persisted routing time, so it is filled here
# by lookup, keyed (leaf, namespace). D2 / D3-A use the paper's reported numbers
# (Table 3/22); KRT D3-B is the measured mean elapsed_routing_s over its 10
# baseline boards. Ortho D3-B has no entry -> reported as '--'.
RULE_BASED_TIME_PAPER = {
    ("krt",   "d2a"):    0.82,
    ("krt",   "d3/d3a"): 0.65,
    ("krt",   "d3/d3b"): 3.265,   # measured, 5/28 run
    ("ortho", "d2a"):    2.54,
    ("ortho", "d3/d3a"): 2.20,
}


def rule_based_time_for(rel: str) -> float | None:
    """Paper-reused routing time for a KRT/OrthoRoute cell, else None."""
    return RULE_BASED_TIME_PAPER.get((cell_name(rel), board_namespace(rel)))


def reduce_cell(rel: str, metrics: Sequence[str]) -> dict:
    """Per-cell reduction.

    Returns a dict::

        {
          "n_boards": int,
          "n_seeds":  int,
          "per_seed": {metric: {seed: board_mean}},     # seed-level scalars
          "agg":      {metric: (mean_over_seeds, std_over_seeds, n_seeds)},
          "grand":    {metric: pooled_mean_over_all_board_seed},  # Fig 6c
        }

    ``mean_over_seeds`` is the mean of the per-seed board-means; ``std`` is the
    sample std (ddof=1) across seeds (``None`` if <2 seeds -> rendered ``†``).
    """
    cell = disk_cell_name(rel)
    rows = load_rollouts(rel)
    groups = group_by_seed_board(rows, cell)

    # (board, seed) -> metric dict
    by_bs = {bs: best_at_k(g) for bs, g in groups.items()}

    # potential_gain backfill: when the per_rollout column is absent or blank,
    # derive it from final_potential minus the per-board cached
    # initial_potential (compute_initial_potential.py).
    if "potential_gain" in metrics:
        cache = load_initial_cache()
        ns = board_namespace(rel)
        for (bid, _seed), md in by_bs.items():
            if md.get("potential_gain") is None and md.get("potential") is not None:
                init = cache.get(f"{ns}::{bid}")
                if init is None:  # cache built pre-rename keeps t-series keys
                    init = cache.get(f"{legacy_rel(ns)}::{bid}")
                if init is not None:
                    md["potential_gain"] = md["potential"] - init

    seeds = sorted({seed for (_bid, seed) in by_bs})
    boards = sorted({bid for (bid, _seed) in by_bs})

    per_seed: dict[str, dict[int, float | None]] = {m: {} for m in metrics}
    grand: dict[str, float | None] = {}
    for m in metrics:
        # per-seed board-mean
        for s in seeds:
            vals = [md.get(m) for (bid, seed), md in by_bs.items() if seed == s]
            per_seed[m][s] = _mean(vals)
        # pooled over all (board, seed)
        grand[m] = _mean([md.get(m) for md in by_bs.values()])

    agg: dict[str, tuple[float | None, float | None, int]] = {}
    for m in metrics:
        seed_scalars = [per_seed[m][s] for s in seeds]
        present = [x for x in seed_scalars if x is not None]
        agg[m] = (_mean(seed_scalars), u.sample_std(seed_scalars), len(present))

    # LLM/rule-based cells carry no per_board_rollout_time (boards-only). Fall
    # back to the model-level LLM sec/rollout, then to the paper-reused KRT/Ortho
    # time (no per-seed std for either).
    if "time" in metrics and agg.get("time", (None,))[0] is None:
        t = llm_time_for(rel)
        if t is None:
            t = rule_based_time_for(rel)
        if t is not None:
            agg["time"] = (t, None, 1)
            grand["time"] = t

    return {
        "n_boards": len(boards),
        "n_seeds": len(seeds),
        "per_seed": per_seed,
        "agg": agg,
        "grand": grand,
    }


def parse_fail_stats(rel: str) -> dict:
    """Boards/rollouts whose generated .kicad_pcb could not be parsed/evaluated.

    A rollout is a *parse fail* when the post-hoc DRC worker errored
    (``eval_status == 'error'`` or a non-empty ``eval_error`` -- e.g.
    ``worker_exception`` from an unloadable board). These rows carry blank metric
    columns and are otherwise silently dropped from the means, so they are
    surfaced separately here.

    Returns ``{n_rollouts, n_fail_rollouts, fail_rate, n_boards,
    n_fail_boards_all}`` where ``n_fail_boards_all`` counts boards whose *every*
    rollout failed to parse.
    """
    rows = load_rollouts(rel)
    n = len(rows)
    nf = sum(1 for r in rows if is_parse_fail(r))
    by_board: dict[str, list[bool]] = {}
    for r in rows:
        by_board.setdefault(r.get("board_id", ""), []).append(is_parse_fail(r))
    n_all_fail = sum(1 for v in by_board.values() if v and all(v))
    return {
        "n_rollouts": n,
        "n_fail_rollouts": nf,
        "fail_rate": (nf / n) if n else None,
        "n_boards": len(by_board),
        "n_fail_boards_all": n_all_fail,
    }


def coverage(rel: str, col: str) -> tuple[int, int]:
    """(non-blank, total) count for a raw column -- used to warn on mid-run cells."""
    rows = load_rollouts(rel)
    nb = sum(1 for r in rows if (r.get(col) or "").strip() != "")
    return nb, len(rows)


def warn_if_sparse(rel: str, cols: Sequence[str], thresh: float = 0.9) -> None:
    """Print a stderr warning if any needed column is <``thresh`` populated
    (i.e. the cell's DRC stage is likely still running)."""
    for c in cols:
        nb, n = coverage(rel, c)
        if n and nb / n < thresh:
            print(f"[warn] {rel}: column {c!r} only {nb}/{n} populated "
                  f"(< {thresh:.0%}) -- numbers provisional", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Method registry / cell discovery
# --------------------------------------------------------------------------- #
# Dataset rel-prefix used by Table 3 / 22 / 24-25.
DATASETS = {"D2": "d2a", "D3-A": "d3/d3a", "D3-B": "d3/d3b"}

# Rule-based + RL rows (display name -> cell leaf), resolved per dataset prefix.
RULE_BASED_RL = [
    ("Reference",         "reference"), # D3 only (legacy disk cell name: human)
    ("Freerouting",       "freerouting"),
    ("OrthoRoute",        "ortho"),
    ("KiCadRoutingTools", "krt"),
    ("PPO (per-step)",    "transformer_pcbworld"),
    ("PPO (terminal)",    "transformer_pcbworld_episodic"),
    ("GRPO",              "transformer_pcbworld_grpo"),
]
# LLM backbones reported in Table 3 (interactive agent).
LLM_BACKBONES = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "qwen3.5-397b"]


def rows_for_dataset(ds_key: str) -> list[tuple[str, str]]:
    """Ordered (display_name, rel_path) for the Table-3 row set on a dataset,
    keeping only cells that exist on disk."""
    pre = DATASETS[ds_key]
    out: list[tuple[str, str]] = []
    # rule-based + RL, with LLM block inserted before the RL block
    for disp, leaf in RULE_BASED_RL:
        if disp == "PPO (per-step)":  # LLM backbones come right before the RL block
            for mdl in LLM_BACKBONES:
                rel = f"{pre}/interactive_{mdl}"
                if cell_exists(rel):
                    out.append((f"PCBWorld ({mdl})", rel))
        rel = f"{pre}/{leaf}"
        if cell_exists(rel):
            out.append((disp, rel))
    return out


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def fmt(x: float | None, nd: int = 2, pct: bool = False) -> str:
    """Rounded display string for LaTeX/paper *tables* (default 2 decimals)."""
    if x is None:
        return "--"
    if pct:
        return f"{100 * x:.1f}"
    return f"{x:.{nd}f}"


def fmt_raw(x: float | None, pct: bool = False) -> str:
    """Full-precision string for *figure* data dumps -- no rounding.

    Figures must reflect the actual values (graphs are plotted from the raw
    floats); only the LaTeX tables round, via ``fmt``/``fmt_pm``. ``pct`` scales
    by 100 to keep a column's percentage semantics but applies no rounding."""
    if x is None:
        return "--"
    return repr(100 * x if pct else x)


def fmt_pm(mean: float | None, std: float | None, nd: int = 2, pct: bool = False) -> str:
    """``mean ± std``; ``†`` when std is undefined (single seed); ``--`` if no value."""
    if mean is None:
        return "--"
    m = fmt(mean, nd, pct)
    if std is None:
        return f"{m}†"
    return f"{m} ± {fmt(std, nd, pct)}"


# --------------------------------------------------------------------------- #
# Matplotlib style (shared figure conventions)
# --------------------------------------------------------------------------- #
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    })
    return plt


PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Draft theme (mirrors the paper's Fig 5/6 routability plot):
# our method = blue, SABLE = green, Jumanji A2C = orange.
DRAFT_BLUE, DRAFT_ORANGE, DRAFT_GREEN = "#2f6df6", "#ef7d1a", "#1f9d55"
GRID_GREY = "#d8dee8"


def clean_axes(ax) -> None:
    """Drop the top/right spines and apply the light draft grid."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, color=GRID_GREY, linewidth=0.55, alpha=0.8)


def savefig(fig, stem: str) -> list[Path]:
    """Save a figure as both PDF and PNG under ``paper_outputs`` (guarded)."""
    outs = []
    for ext in ("pdf", "png"):
        p = assert_output_path(PAPER_OUT / f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        outs.append(p)
    return outs


def write_table(stem: str, header: Sequence[str], rows: Sequence[Sequence[str]],
                title: str = "") -> tuple[Path, Path]:
    """Write a CSV and a GitHub-flavoured Markdown table under ``paper_outputs``."""
    import csv
    csv_p = assert_output_path(PAPER_OUT / f"{stem}.csv")
    with csv_p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md_p = assert_output_path(PAPER_OUT / f"{stem}.md")
    lines = []
    if title:
        lines += [f"# {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    md_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_p, md_p
