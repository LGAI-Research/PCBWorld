#!/usr/bin/env python3
"""Regenerate reward-weight factorial figures for the CADAgent paper.

The intended source is W&B training history. In practice, this script also reads
the local TensorBoard event logs written by the same training jobs, which keeps
the figure reproducible when the W&B cloud project only contains post-hoc eval
runs or when the machine is offline.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from configs.loader.paths import data_root_path


WIRE_VALUES = (0.0, 0.001, 0.002)
VIA_VALUES = (0.0, 0.05, 0.1)
SEEDS = (42, 43, 44, 45)

TARGET_RUN_TEMPLATES = (
    "v56_2L_dense_cross_wire0.001_via0.1_perstep_seed{seed}",
    "v56_2L_dense_cross_wire0.002_via0.05_perstep_seed{seed}",
    "v56_2L_dense_default_wire0.001_via0.05_cell_wire0.001_via0_perstep_seed{seed}",
    "v56_2L_dense_default_wire0.001_via0.05_cell_wire0_via0.05_perstep_seed{seed}",
    "v56_2L_dense_default_wire0.001_via0.05_cell_wire0.001_via0.05_perstep_seed{seed}",
    "v56_2L_dense_default_wire0.002_via0.1_cell_wire0.002_via0_perstep_seed{seed}",
    "v56_2L_dense_default_wire0.002_via0.1_cell_wire0_via0.1_perstep_seed{seed}",
    "v56_2L_dense_default_wire0.001_via0.05_cell_wire0_via0_perstep_seed{seed}",
    "v56_2L_dense_default_wire0.002_via0.1_cell_wire0.002_via0.1_perstep_seed{seed}",
)
TARGET_RUN_NAMES = {template.format(seed=seed) for template in TARGET_RUN_TEMPLATES for seed in SEEDS}

# The packaged benchmark tree is not shipped with the repo: EXPR_ROOT, or
# $CADAGENT_DATA_ROOT/KDD_benchmark/experimental_results, or "" (checked before use).
DEFAULT_EXPR_ROOT = Path(
    os.environ.get("EXPR_ROOT")
    or data_root_path("KDD_benchmark", "experimental_results")
)
DEFAULT_TB_ROOTS = (
    str(DEFAULT_EXPR_ROOT / "training_logs/reward_ablation/tensorboard_logs/v56_dense_scale_wirevia_20260429_155706"),
    str(DEFAULT_EXPR_ROOT / "training_logs/reward_ablation/tensorboard_logs/v56_dense_cross_wirevia_20260502"),
)

WIRE_KEYS = ("eval/wirelength_mean", "common/eval/wirelength_mean")
VIA_KEYS = ("eval/via_count_mean", "common/eval/via_count_mean")

DEFAULT_RE = re.compile(
    r"^v56_2L_dense_default_wire(?P<base_wire>[0-9.]+)_via(?P<base_via>[0-9.]+)"
    r"_cell_wire(?P<wire>[0-9.]+)_via(?P<via>[0-9.]+)_perstep_seed(?P<seed>\d+)$"
)
CROSS_RE = re.compile(
    r"^v56_2L_dense_cross_wire(?P<wire>[0-9.]+)_via(?P<via>[0-9.]+)_perstep_seed(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class Candidate:
    wire: float
    via: float
    seed: int
    run_name: str
    source: str
    source_path: str
    selected_step: int
    wirelength_mean: float
    via_count_mean: float
    source_mtime: float


def _float_label(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:g}"


def _parse_run_name(name: str) -> tuple[float, float, int] | None:
    """Return actual reward weights and seed from a V56 dense sweep run name.

    Default sweep runs encode the actual ablation cell in ``cell_wire/cell_via``;
    cross runs encode it directly in ``wire/via``.
    """

    if name not in TARGET_RUN_NAMES:
        return None
    match = DEFAULT_RE.match(name) or CROSS_RE.match(name)
    if not match:
        return None
    wire = float(match.group("wire"))
    via = float(match.group("via"))
    seed = int(match.group("seed"))
    return wire, via, seed


def _targeted_scalar(events: list, target_step: int) -> tuple[int, float] | None:
    usable = [event for event in events if event.step <= target_step]
    if not usable:
        return None
    exact = [event for event in usable if event.step == target_step]
    event = exact[-1] if exact else usable[-1]
    return int(event.step), float(event.value)


def _read_tb_candidates(tb_roots: Iterable[Path], target_step: int) -> list[Candidate]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    candidates: list[Candidate] = []
    for root in tb_roots:
        if not root.exists():
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            parsed = _parse_run_name(run_dir.name)
            if parsed is None:
                continue
            wire, via, seed = parsed
            if wire not in WIRE_VALUES or via not in VIA_VALUES or seed not in SEEDS:
                continue
            event_files = list(run_dir.glob("events.out.tfevents*"))
            if not event_files:
                continue
            accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = set(accumulator.Tags().get("scalars", []))

            wire_pair = None
            for key in WIRE_KEYS:
                if key in tags:
                    wire_pair = _targeted_scalar(accumulator.Scalars(key), target_step)
                    if wire_pair is not None:
                        break

            via_pair = None
            for key in VIA_KEYS:
                if key in tags:
                    via_pair = _targeted_scalar(accumulator.Scalars(key), target_step)
                    if via_pair is not None:
                        break

            if wire_pair is None or via_pair is None:
                continue
            selected_step = min(wire_pair[0], via_pair[0])
            source_mtime = max(path.stat().st_mtime for path in event_files)
            candidates.append(
                Candidate(
                    wire=wire,
                    via=via,
                    seed=seed,
                    run_name=run_dir.name,
                    source="tensorboard",
                    source_path=str(run_dir),
                    selected_step=selected_step,
                    wirelength_mean=wire_pair[1],
                    via_count_mean=via_pair[1],
                    source_mtime=source_mtime,
                )
            )
    return candidates


def _safe_import_wandb():
    """Import real wandb even when repo-local ``wandb/`` shadows the package."""

    original_path = list(sys.path)
    cwd = str(Path.cwd().resolve())
    repo = str(Path(__file__).resolve().parents[2])
    try:
        sys.path = [
            item
            for item in sys.path
            if item
            and str(Path(item).resolve()) not in {cwd, repo}
            and not str(Path(item).resolve()).startswith(f"{repo}{os.sep}")
        ]
        import wandb  # type: ignore

        if not hasattr(wandb, "Api"):
            return None
        return wandb
    except Exception:
        return None
    finally:
        sys.path = original_path


def _wandb_created_at_timestamp(run) -> float:
    created_at = getattr(run, "created_at", None)
    if created_at is None:
        return 0.0
    if hasattr(created_at, "timestamp"):
        return float(created_at.timestamp())
    if isinstance(created_at, str):
        try:
            return datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _read_wandb_candidates(
    *,
    entity: str,
    project: str,
    target_step: int,
    timeout: int,
) -> list[Candidate]:
    wandb = _safe_import_wandb()
    if wandb is None:
        return []
    try:
        api = wandb.Api(timeout=timeout)
        runs = api.runs(f"{entity}/{project}", per_page=1000)
    except Exception:
        return []

    candidates: list[Candidate] = []
    try:
        for run in iter(runs):
            parsed = _parse_run_name(run.name or "")
            if parsed is None:
                continue
            wire, via, seed = parsed
            if wire not in WIRE_VALUES or via not in VIA_VALUES or seed not in SEEDS:
                continue
            try:
                selected_row: tuple[int, float, float] | None = None
                for wire_key, via_key in (
                    ("eval/wirelength_mean", "eval/via_count_mean"),
                    ("common/eval/wirelength_mean", "common/eval/via_count_mean"),
                ):
                    rows = run.scan_history(keys=["_step", wire_key, via_key], page_size=1000)
                    usable: list[tuple[int, float, float]] = []
                    for row in rows:
                        step = row.get("_step")
                        wire_value = row.get(wire_key)
                        via_value = row.get(via_key)
                        if step is None or wire_value is None or via_value is None:
                            continue
                        step_i = int(step)
                        if step_i <= target_step:
                            usable.append((step_i, float(wire_value), float(via_value)))
                    if usable:
                        exact = [row for row in usable if row[0] == target_step]
                        selected_row = exact[-1] if exact else usable[-1]
                        break
            except Exception:
                continue
            if selected_row is None:
                continue
            step, wire_value, via_value = selected_row
            candidates.append(
                Candidate(
                    wire=wire,
                    via=via,
                    seed=seed,
                    run_name=run.name,
                    source="wandb",
                    source_path=f"{entity}/{project}/{run.id}",
                    selected_step=step,
                    wirelength_mean=wire_value,
                    via_count_mean=via_value,
                    source_mtime=_wandb_created_at_timestamp(run),
                )
            )
    except Exception:
        return []
    return candidates


def _dedupe_candidates(candidates: list[Candidate]) -> tuple[list[Candidate], list[dict[str, str]]]:
    by_key: dict[tuple[float, float, int], list[Candidate]] = {}
    for candidate in candidates:
        by_key.setdefault((candidate.wire, candidate.via, candidate.seed), []).append(candidate)

    selected: list[Candidate] = []
    duplicate_rows: list[dict[str, str]] = []
    for key, group in sorted(by_key.items()):
        group_sorted = sorted(
            group,
            key=lambda item: (
                item.source == "wandb",
                item.selected_step,
                item.source_mtime,
                item.run_name,
            ),
            reverse=True,
        )
        keep = group_sorted[0]
        selected.append(keep)
        for item in group_sorted[1:]:
            duplicate_rows.append(
                {
                    "wire": _float_label(item.wire),
                    "via": _float_label(item.via),
                    "seed": str(item.seed),
                    "kept_run_name": keep.run_name,
                    "dropped_run_name": item.run_name,
                    "kept_source": keep.source,
                    "dropped_source": item.source,
                    "reason": "same_wire_via_seed_latest_source_selected",
                }
            )
    return selected, duplicate_rows


def _validate_selected(selected: list[Candidate]) -> None:
    errors: list[str] = []
    by_cell: dict[tuple[float, float], set[int]] = {}
    for candidate in selected:
        by_cell.setdefault((candidate.wire, candidate.via), set()).add(candidate.seed)
    for wire in WIRE_VALUES:
        for via in VIA_VALUES:
            seeds = by_cell.get((wire, via), set())
            missing = sorted(set(SEEDS) - seeds)
            if missing:
                errors.append(f"missing seeds for wire={wire:g}, via={via:g}: {missing}")
    if len(selected) != len(WIRE_VALUES) * len(VIA_VALUES) * len(SEEDS):
        errors.append(f"expected 36 selected rows, got {len(selected)}")
    if errors:
        raise RuntimeError("Incomplete reward-ablation factorial data:\n" + "\n".join(errors))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(selected: list[Candidate]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for wire in WIRE_VALUES:
        for via in VIA_VALUES:
            group = [item for item in selected if item.wire == wire and item.via == via]
            rows.append(
                {
                    "level": "cell",
                    "wire": _float_label(wire),
                    "via": _float_label(via),
                    "n": len(group),
                    "wirelength_mean": float(np.mean([item.wirelength_mean for item in group])),
                    "wirelength_std": float(np.std([item.wirelength_mean for item in group])),
                    "via_count_mean": float(np.mean([item.via_count_mean for item in group])),
                    "via_count_std": float(np.std([item.via_count_mean for item in group])),
                }
            )
    for wire in WIRE_VALUES:
        group = [item for item in selected if item.wire == wire]
        seed_wire_means = [
            float(np.mean([item.wirelength_mean for item in group if item.seed == seed]))
            for seed in SEEDS
        ]
        seed_via_means = [
            float(np.mean([item.via_count_mean for item in group if item.seed == seed]))
            for seed in SEEDS
        ]
        rows.append(
            {
                "level": "marginal_wire",
                "wire": _float_label(wire),
                "via": "ALL",
                "n": len(group),
                "wirelength_mean": float(np.mean([item.wirelength_mean for item in group])),
                "wirelength_std": float(np.std(seed_wire_means)),
                "via_count_mean": float(np.mean([item.via_count_mean for item in group])),
                "via_count_std": float(np.std(seed_via_means)),
            }
        )
    for via in VIA_VALUES:
        group = [item for item in selected if item.via == via]
        seed_wire_means = [
            float(np.mean([item.wirelength_mean for item in group if item.seed == seed]))
            for seed in SEEDS
        ]
        seed_via_means = [
            float(np.mean([item.via_count_mean for item in group if item.seed == seed]))
            for seed in SEEDS
        ]
        rows.append(
            {
                "level": "marginal_via",
                "wire": "ALL",
                "via": _float_label(via),
                "n": len(group),
                "wirelength_mean": float(np.mean([item.wirelength_mean for item in group])),
                "wirelength_std": float(np.std(seed_wire_means)),
                "via_count_mean": float(np.mean([item.via_count_mean for item in group])),
                "via_count_std": float(np.std(seed_via_means)),
            }
        )
    return rows


def _backup_existing(fig_dir: Path) -> None:
    existing = [
        fig_dir / "rq4_factorial_wl.pdf",
        fig_dir / "rq4_factorial_wl.png",
        fig_dir / "rq4_factorial_via.pdf",
        fig_dir / "rq4_factorial_via.png",
        fig_dir / "rq4_factorial.pdf",
        fig_dir / "rq4_factorial.png",
    ]
    if not any(path.exists() for path in existing):
        return
    archive = fig_dir / "archive" / f"rq4_factorial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive.mkdir(parents=True, exist_ok=True)
    for path in existing:
        if path.exists():
            shutil.copy2(path, archive / path.name)


def _plot_pair(
    *,
    x_values: tuple[float, ...],
    primary: list[float],
    primary_err: list[float],
    secondary: list[float],
    secondary_err: list[float],
    primary_ylabel: str,
    secondary_ylabel: str,
    xlabel: str,
    title: str,
    output_stem: Path,
) -> None:
    orange = "#e68632"
    teal = "#1b9aaa"
    error_kw = {"ecolor": "#333333", "elinewidth": 0.75, "capsize": 2.4, "capthick": 0.75}
    x = np.arange(len(x_values))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(3.6, 2.5))
    ax2 = ax1.twinx()
    ax1.bar(
        x - width / 2,
        primary,
        yerr=primary_err,
        width=width,
        color=orange,
        label=primary_ylabel,
        error_kw=error_kw,
    )
    ax2.bar(
        x + width / 2,
        secondary,
        yerr=secondary_err,
        width=width,
        color=teal,
        label=secondary_ylabel,
        error_kw=error_kw,
    )

    def padded_limits(values: list[float], errors: list[float], *, min_value: float | None = None) -> tuple[float, float]:
        lower = min(value - err for value, err in zip(values, errors))
        upper = max(value + err for value, err in zip(values, errors))
        span = max(upper - lower, 1.0)
        pad = span * 0.16
        lower -= pad
        upper += pad
        if min_value is not None:
            lower = max(min_value, lower)
        return lower, upper

    ax1.set_ylim(*padded_limits(primary, primary_err))
    ax2.set_ylim(*padded_limits(secondary, secondary_err, min_value=0.0))

    for xpos, value, err in zip(x - width / 2, primary, primary_err):
        ax1.annotate(
            f"{value:.1f}",
            xy=(xpos, value + err),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color=orange,
        )
    for xpos, value, err in zip(x + width / 2, secondary, secondary_err):
        ax2.annotate(
            f"{value:.2f}",
            xy=(xpos, value + err),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color=teal,
        )

    if xlabel == "$w_p$":
        tick_labels = [rf"$w_p={_float_label(value)}$" for value in x_values]
        plot_title = r"Wirelength penalty $w_p$ (avg. over $v_p$)"
    else:
        tick_labels = [rf"$v_p={_float_label(value)}$" for value in x_values]
        plot_title = r"Via penalty $v_p$ (avg. over $w_p$)"
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels, fontsize=9)
    ax1.set_title(plot_title, fontsize=10.5, fontweight="bold", pad=6)
    ax1.set_ylabel(f"{primary_ylabel} (mean)", color=orange, fontsize=10, fontweight="bold", labelpad=4)
    ax2.set_ylabel(f"{secondary_ylabel} (mean)", color=teal, fontsize=10, fontweight="bold", labelpad=4)
    ax1.tick_params(axis="y", labelcolor=orange, labelsize=9)
    ax2.tick_params(axis="y", labelcolor=teal, labelsize=9)
    ax1.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.65)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9,
        handlelength=1.3,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), pad=0.4)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_combined(summary: list[dict[str, object]], output_stem: Path) -> None:
    marg_w = [row for row in summary if row["level"] == "marginal_wire"]
    marg_v = [row for row in summary if row["level"] == "marginal_via"]
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.0))

    for ax, rows, xlabel, title in [
        (axes[0], marg_w, "$w_p$", "Sweeping $w_p$"),
        (axes[1], marg_v, "$v_p$", "Sweeping $v_p$"),
    ]:
        ax2 = ax.twinx()
        x = np.arange(len(rows))
        width = 0.36
        wl = [float(row["wirelength_mean"]) for row in rows]
        wl_err = [float(row["wirelength_std"]) for row in rows]
        vc = [float(row["via_count_mean"]) for row in rows]
        vc_err = [float(row["via_count_std"]) for row in rows]
        labels = [str(row["wire"] if xlabel == "$w_p$" else row["via"]) for row in rows]
        error_kw = {"ecolor": "#333333", "elinewidth": 0.75, "capsize": 2.4, "capthick": 0.75}
        ax.bar(x - width / 2, wl, yerr=wl_err, width=width, color="#e68632", label="wirelength", error_kw=error_kw)
        ax2.bar(x + width / 2, vc, yerr=vc_err, width=width, color="#1b9aaa", label="via count", error_kw=error_kw)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_ylabel("Wirelength", color="#e68632")
        ax2.set_ylabel("Via count", color="#1b9aaa")
        ax.tick_params(axis="y", labelcolor="#e68632")
        ax2.tick_params(axis="y", labelcolor="#1b9aaa")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overleaf-root", type=Path, default=Path(os.environ.get("OVERLEAF_ROOT", "var/results/kdd/paper_outputs")))
    parser.add_argument("--target-step", type=int, default=300)
    # No baked-in entity: it identifies the W&B account that owns the runs.
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "pcbworld"))
    parser.add_argument("--wandb-timeout", type=int, default=45)
    parser.add_argument("--tb-root", action="append", default=[])
    parser.add_argument("--source", choices=("auto", "wandb", "tensorboard"), default="auto")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    tb_roots = [repo_root / root for root in (args.tb_root or DEFAULT_TB_ROOTS)]

    candidates: list[Candidate] = []
    if args.source in {"auto", "wandb"}:
        if not args.wandb_entity:
            # Explicit --source wandb must fail; "auto" may fall back to
            # TensorBoard below, but says so rather than skipping silently.
            if args.source == "wandb":
                raise SystemExit(
                    "--source wandb needs an entity — pass --wandb-entity or "
                    "export WANDB_ENTITY."
                )
            print(
                "[figure6] WANDB_ENTITY unset — skipping W&B, "
                "reading TensorBoard instead.",
                file=sys.stderr,
            )
        else:
            candidates = _read_wandb_candidates(
                entity=args.wandb_entity,
                project=args.wandb_project,
                target_step=args.target_step,
                timeout=args.wandb_timeout,
            )
    if args.source == "wandb" and not candidates:
        raise RuntimeError("No matching W&B training runs were found.")
    if args.source in {"auto", "tensorboard"} and not candidates:
        candidates = _read_tb_candidates(tb_roots, args.target_step)

    selected, duplicates = _dedupe_candidates(candidates)
    _validate_selected(selected)

    fig_dir = args.overleaf_root / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_backup:
        _backup_existing(fig_dir)

    source_rows = [
        {
            "wire": _float_label(item.wire),
            "via": _float_label(item.via),
            "seed": item.seed,
            "run_name": item.run_name,
            "source": item.source,
            "source_path": item.source_path,
            "selected_step": item.selected_step,
            "wirelength_mean": item.wirelength_mean,
            "via_count_mean": item.via_count_mean,
        }
        for item in sorted(selected, key=lambda row: (row.wire, row.via, row.seed))
    ]
    _write_csv(
        fig_dir / "rq4_factorial_source.csv",
        source_rows,
        [
            "wire",
            "via",
            "seed",
            "run_name",
            "source",
            "source_path",
            "selected_step",
            "wirelength_mean",
            "via_count_mean",
        ],
    )
    _write_csv(
        fig_dir / "rq4_factorial_duplicates.csv",
        duplicates,
        ["wire", "via", "seed", "kept_run_name", "dropped_run_name", "kept_source", "dropped_source", "reason"],
    )

    summary = _summary_rows(selected)
    _write_csv(
        fig_dir / "rq4_factorial_summary.csv",
        summary,
        ["level", "wire", "via", "n", "wirelength_mean", "wirelength_std", "via_count_mean", "via_count_std"],
    )

    marg_w = [row for row in summary if row["level"] == "marginal_wire"]
    marg_v = [row for row in summary if row["level"] == "marginal_via"]
    _plot_pair(
        x_values=WIRE_VALUES,
        primary=[float(row["wirelength_mean"]) for row in marg_w],
        primary_err=[float(row["wirelength_std"]) for row in marg_w],
        secondary=[float(row["via_count_mean"]) for row in marg_w],
        secondary_err=[float(row["via_count_std"]) for row in marg_w],
        primary_ylabel="Wirelength",
        secondary_ylabel="Via count",
        xlabel="$w_p$",
        title="Averaged over $v_p$",
        output_stem=fig_dir / "rq4_factorial_wl",
    )
    _plot_pair(
        x_values=VIA_VALUES,
        primary=[float(row["wirelength_mean"]) for row in marg_v],
        primary_err=[float(row["wirelength_std"]) for row in marg_v],
        secondary=[float(row["via_count_mean"]) for row in marg_v],
        secondary_err=[float(row["via_count_std"]) for row in marg_v],
        primary_ylabel="Wirelength",
        secondary_ylabel="Via count",
        xlabel="$v_p$",
        title="Averaged over $w_p$",
        output_stem=fig_dir / "rq4_factorial_via",
    )
    _plot_combined(summary, fig_dir / "rq4_factorial")

    print(f"selected_rows={len(selected)}")
    print(f"duplicates={len(duplicates)}")
    for row in marg_w:
        print(f"marginal_wire wire={row['wire']} wirelength={float(row['wirelength_mean']):.3f} via={float(row['via_count_mean']):.3f}")
    for row in marg_v:
        print(f"marginal_via via={row['via']} wirelength={float(row['wirelength_mean']):.3f} via={float(row['via_count_mean']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
