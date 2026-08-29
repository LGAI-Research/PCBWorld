#!/usr/bin/env python3
"""Build appendix W&B validation-curve figures for the CADAgent paper.

The paper tables report held-out rollout/post-hoc evaluation.  This script
instead visualizes training-time validation diagnostics from the exact W&B runs
used to train those checkpoints, and writes source-audit CSV/JSON files outside
Overleaf.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.lines import Line2D

from configs.loader.paths import data_root_path


# Two of the audited W&B runs were named after the internal account that
# launched them, so their raw ids are not carried in this public source.
# Whoever holds the runs exports the id via the named variable to refetch them.
WITHHELD_RUN_ID = "WITHHELD"


def internal_run_id(env_var: str) -> str:
    return os.environ.get(env_var, WITHHELD_RUN_ID)


def wandb_entity() -> str:
    """W&B entity that owns these runs — from $WANDB_ENTITY, no baked-in default.

    Fails loudly rather than guessing: a wrong entity yields a confusing 404
    from the W&B API instead of an actionable message.
    """
    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        raise SystemExit(
            "WANDB_ENTITY is not set — export it to the W&B entity that owns "
            "these training runs before regenerating this figure."
        )
    return entity


# W&B projects holding the audited runs. Like the entity and the run ids, the
# project names identify the account that owns them, so they are supplied via
# the named variables rather than baked in.
T1_TRANSFORMER_PROJECT = os.environ.get("WANDB_PROJECT_T1_TRANSFORMER", "pcbworld")
T1_BASELINE_PROJECT = os.environ.get("WANDB_PROJECT_T1_BASELINE", "pcbworld")
POLICY_PROJECT = os.environ.get("WANDB_PROJECT_POLICY", "pcbworld")
T1_GRIDS = (10, 50, 100, 200, 500)
T1_TOTAL_NETS = 5.0
POLICY_TOTAL_NETS = 10.0
# Section labels used in the audit sub-directory names and the emitted figure names.
T1_AUDIT_SECTION = "rq1"
POLICY_AUDIT_SECTION = "rq5"

METHOD_COLORS = {
    "kicad_ppo": "#2F6FDF",
    "jumanji_a2c": "#E57D16",
    "sable": "#1F9D55",
    "ppo_per_step": "#2F6FDF",
    "grpo_sparse": "#E57D16",
    "ppo_sparse": "#C026D3",
}
GRID_COLORS = {
    10: "#2563EB",
    50: "#7C3AED",
    100: "#F97316",
    200: "#16A34A",
    500: "#475569",
}
GRID_STYLES = {
    10: {"linestyle": "-", "marker": "o"},
    50: {"linestyle": "-", "marker": "D"},
    100: {"linestyle": "-", "marker": "s"},
    200: {"linestyle": "-", "marker": "^"},
    500: {"linestyle": "-", "marker": "P"},
}


@dataclass(frozen=True)
class RunSpec:
    section: str
    method: str
    method_label: str
    project: str
    run_id: str
    seed: int | None
    grid: int | None
    native_unit: str
    x_aliases: tuple[str, ...]
    max_env_steps: int | None = None
    display_x_scale: float = 1.0
    display_x_max: float | None = None
    status: str = "ok"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    aliases: tuple[str, ...]
    transform: Callable[[pd.Series, RunSpec, str], pd.Series]
    ylabel: str
    lower_is_better: bool = False


def _identity(series: pd.Series, _spec: RunSpec, _source: str) -> pd.Series:
    return series.astype(float)


def _routability(series: pd.Series, spec: RunSpec, source: str) -> pd.Series:
    value = series.astype(float)
    if "routed_nets" in source or source.endswith("num_connections"):
        total = T1_TOTAL_NETS if spec.section == T1_AUDIT_SECTION else POLICY_TOTAL_NETS
        value = value / total
    return value.clip(lower=0.0, upper=1.0)


def _wirelength_mm(series: pd.Series, spec: RunSpec, _source: str) -> pd.Series:
    value = series.astype(float)
    if spec.section == T1_AUDIT_SECTION and spec.method in {"jumanji_a2c", "sable"}:
        if spec.grid is None:
            raise ValueError("D1 grid-action wirelength conversion requires grid")
        value = value * (100.0 / float(spec.grid))
    return value


T1_METRICS = (
    MetricSpec(
        key="routability",
        label="Routability",
        aliases=(
            "common/eval/routability",
            "common/eval/routed_nets_mean",
            "eval/routability_mean",
            "eval/ratio_connections",
            "common/eval/routed_ratio",
        ),
        transform=_routability,
        ylabel="Rout.",
    ),
    MetricSpec(
        key="wirelength_mm",
        label="Wirelength",
        aliases=(
            "common/eval/wirelength_mean",
            "eval/wirelength_mean",
            "eval/total_path_length",
            "common/absolute/wirelength_mean",
        ),
        transform=_wirelength_mm,
        ylabel="WL (mm)",
        lower_is_better=True,
    ),
    MetricSpec(
        key="steps_per_ep",
        label="Steps / episode",
        aliases=("common/eval/episode_length_mean", "eval/episode_length"),
        transform=_identity,
        ylabel="Steps/ep.",
    ),
)

POLICY_METRICS = (
    MetricSpec(
        key="routability",
        label="Routability",
        aliases=(
            "common/eval/routability",
            "eval/routability_mean",
            "rollout/routability_mean",
            "common/eval/routed_nets_mean",
            "eval/routed_nets_mean",
        ),
        transform=_routability,
        ylabel="Rout.",
    ),
    MetricSpec(
        key="drv",
        label="DRV",
        aliases=(
            "common/eval/drv_mean",
            "common/eval/drc_violations_mean",
            "eval/drc_violations_mean",
            "rollout/drc_violations_mean",
        ),
        transform=_identity,
        ylabel="DRV",
        lower_is_better=True,
    ),
    MetricSpec(
        key="wirelength",
        label="Wirelength",
        aliases=(
            "common/eval/wirelength_mean",
            "eval/wirelength_mean",
            "rollout/final_wirelength_mean",
        ),
        transform=_identity,
        ylabel="WL",
        lower_is_better=True,
    ),
    MetricSpec(
        key="vias",
        label="Vias",
        aliases=(
            "common/eval/via_count_mean",
            "eval/via_count_mean",
            "rollout/final_via_count_mean",
        ),
        transform=_identity,
        ylabel="Vias",
        lower_is_better=True,
    ),
)


def t1_run_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for grid in (10, 100, 200, 500):
        for seed in (42, 43, 44, 45):
            rid = (
                "v56_l1_transformer_grid10_100_200_500_iter300_eval20_"
                f"20260505_122254_v56_l1_transformer_grid{grid}_seed{seed}"
            )
            specs.append(
                RunSpec(
                    section=T1_AUDIT_SECTION,
                    method="kicad_ppo",
                    method_label="PPO",
                    project=T1_TRANSFORMER_PROJECT,
                    run_id=rid,
                    seed=seed,
                    grid=grid,
                    native_unit="PPO iteration",
                    x_aliases=("iteration", "_step"),
                    display_x_max=300.0,
                )
            )
    for seed in (42, 43, 44, 45):
        specs.append(
            RunSpec(
                section=T1_AUDIT_SECTION,
                method="kicad_ppo",
                method_label="PPO",
                project=T1_TRANSFORMER_PROJECT,
                run_id=(
                    "v56_l1_transformer_grid50_iter300_eval20_metrics_"
                    f"20260505_174844_v56_l1_transformer_grid50_seed{seed}"
                ),
                seed=seed,
                grid=50,
                native_unit="PPO iteration",
                x_aliases=("iteration", "_step"),
                display_x_max=300.0,
            )
        )

    jumanji_ids = {
        (10, 42): "c6l4jzof",
        (10, 43): "rqs515qq",
        (50, 42): "ny9zjtmq",
        (50, 43): "4wzlup67",
        (100, 42): "v56_l1_jumanji_a2c_grid100_seed42_origcfg_b16_lr1p25e5_e2400_v56_l1_context_half_jumanji_g100_long24h_20260504_1",
        (100, 43): "v56_l1_jumanji_a2c_grid100_seed43_origcfg_b16_lr1p25e5_e2400_v56_l1_context_half_jumanji_g100_long24h_20260504_1",
        (200, 42): "v56_l1_jumanji_a2c_grid200_seed42_origcfg_b8_lr6p25e6_e2200_24hfix_v56_l1_context_half_jumanji_g200_b8_e2200_24hfix_20260504",
        (200, 43): internal_run_id("WANDB_RUN_ID_JUMANJI_G200_S43"),
    }
    for (grid, seed), rid in jumanji_ids.items():
        specs.append(
            RunSpec(
                section=T1_AUDIT_SECTION,
                method="jumanji_a2c",
                method_label="Jumanji A2C",
                project=T1_BASELINE_PROJECT,
                run_id=rid,
                seed=seed,
                grid=grid,
                native_unit="A2C epoch",
                x_aliases=("progress/epoch", "epoch", "env_steps", "_step"),
                display_x_max=3500.0,
            )
        )
    specs.append(
        RunSpec(
            section=T1_AUDIT_SECTION,
            method="jumanji_a2c",
            method_label="Jumanji A2C",
            project=T1_BASELINE_PROJECT,
            run_id="OOM",
            seed=None,
            grid=500,
            native_unit="A2C epoch",
            x_aliases=("progress/epoch", "epoch"),
            display_x_max=3500.0,
            status="oom",
        )
    )

    sable_ids = {
        (10, 42): "v56_l1_sable_mava_grid10_seed42_fresh22h_u1580k_v56_l1_sable_grid10_fresh22h_u1580k_20260505_191339",
        (10, 43): "v56_l1_sable_mava_grid10_seed43_fresh22h_u1580k_v56_l1_sable_grid10_fresh22h_u1580k_20260505_191339",
        (50, 42): "v56_l1_sable_mava_grid50_seed42_ctxhalf_fixedfov_fov12_24h_v56_l1_sable_fovfix_g50_g100_20260505_0140",
        (50, 43): "v56_l1_sable_mava_grid50_seed43_ctxhalf_fixedfov_fov12_24h_v56_l1_sable_fovfix_g50_g100_20260505_0140",
        (100, 42): "v56_l1_sable_mava_grid100_seed42_ctxhalf_fixedfov_fov25_24h_v56_l1_sable_fovfix_g50_g100_20260505_0140",
        (100, 43): "v56_l1_sable_mava_grid100_seed43_ctxhalf_fixedfov_fov25_24h_v56_l1_sable_fovfix_g50_g100_20260505_0140",
        (200, 42): "v56_l1_sable_mava_grid200_seed42_ctxhalf_ub2_lr2p5e4_24hfix_slurm_v56_l1_context_half_sable_heavy_24hfix_slurm_20260504",
        (200, 43): "v56_l1_sable_mava_grid200_seed43_ctxhalf_ub2_lr2p5e4_24hfix_slurm_v56_l1_context_half_sable_heavy_24hfix_slurm_20260504",
        (500, 42): "v56_l1_sable_mava_grid500_seed42_ctxhalf_ub2_lr2p5e4_24hfix_slurm_v56_l1_context_half_sable_heavy_24hfix_slurm_20260504",
        (500, 43): "v56_l1_sable_mava_grid500_seed43_ctxhalf_ub2_lr2p5e4_24hfix_slurm_v56_l1_context_half_sable_heavy_24hfix_slurm_20260504",
    }
    for (grid, seed), rid in sable_ids.items():
        specs.append(
            RunSpec(
                section=T1_AUDIT_SECTION,
                method="sable",
                method_label="SABLE",
                project=T1_BASELINE_PROJECT,
                run_id=rid,
                seed=seed,
                grid=grid,
                native_unit="SABLE update",
                x_aliases=("progress/update", "eval_step", "env_steps", "_step"),
                display_x_scale=0.01,
                display_x_max=18_000.0,
            )
        )
    return specs


def policy_run_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for seed in (42, 43, 44, 45):
        specs.append(
            RunSpec(
                section=POLICY_AUDIT_SECTION,
                method="ppo_per_step",
                method_label="PPO",
                project=POLICY_PROJECT,
                run_id=(
                    "v56_dense_scale_wirevia_20260429_155706_"
                    "v56_2L_dense_default_wire0_002_via0_1_cell_wire0_002_via0_1_"
                    f"perstep_seed{seed}"
                ),
                seed=seed,
                grid=None,
                native_unit="training step",
                x_aliases=("iteration", "_step"),
            )
        )
        specs.append(
            RunSpec(
                section=POLICY_AUDIT_SECTION,
                method="grpo_sparse",
                method_label="GRPO",
                project=POLICY_PROJECT,
                run_id=(
                    "v56_grpo_iter1800_routability_20260504_001_"
                    f"v56_grpo_2L_w2e-3_v1e-1_iter1800_seed{seed}"
                ),
                seed=seed,
                grid=None,
                native_unit="training step",
                x_aliases=("iteration", "_step"),
            )
        )
        sparse_id = (
            "v56_sparse_method300_devmerge_no_pack_retry_20260501_145537_"
            "v56_ppo_sparse_2L_w2e-3_v1e-1"
            if seed == 42
            else internal_run_id(f"WANDB_RUN_ID_PPO_SPARSE_S{seed}")
        )
        specs.append(
            RunSpec(
                section=POLICY_AUDIT_SECTION,
                method="ppo_sparse",
                method_label="PPO (terminal)",
                project=POLICY_PROJECT,
                run_id=sparse_id,
                seed=seed,
                grid=None,
                native_unit="training step",
                x_aliases=("iteration", "_step"),
            )
        )
    return specs


def first_existing(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    for key in aliases:
        if key in df.columns and df[key].notna().any():
            return key
    return None


def resolve_x_values(df: pd.DataFrame, spec: RunSpec) -> tuple[pd.Series, str]:
    x_key = first_existing(df, spec.x_aliases)
    if x_key is None:
        return pd.Series(np.arange(len(df), dtype=float), index=df.index), "row_index"
    return pd.to_numeric(df[x_key], errors="coerce"), x_key


def load_run_history(api: wandb.Api, spec: RunSpec, samples: int) -> tuple[pd.DataFrame, dict[str, object]]:
    if spec.status == "oom":
        return pd.DataFrame(), {
            "section": spec.section,
            "method": spec.method,
            "method_label": spec.method_label,
            "project": spec.project,
            "run_id": spec.run_id,
            "seed": spec.seed,
            "grid": spec.grid,
            "state": "oom",
            "history_rows": 0,
            "display_x_scale": spec.display_x_scale,
            "display_x_max": spec.display_x_max,
        }
    if spec.run_id == WITHHELD_RUN_ID:
        raise SystemExit(
            f"run id for {spec.method} seed={spec.seed} grid={spec.grid} is not "
            "carried in this public source (it embedded an internal account "
            "handle). Export the matching WANDB_RUN_ID_* variable to refetch it."
        )
    run = api.run(f"{wandb_entity()}/{spec.project}/{spec.run_id}")
    df = run.history(samples=samples, pandas=True)
    if spec.max_env_steps is not None and "env_steps" in df.columns:
        df = df[df["env_steps"].fillna(0) <= spec.max_env_steps].copy()
    audit = {
        "section": spec.section,
        "method": spec.method,
        "method_label": spec.method_label,
        "project": spec.project,
        "run_id": spec.run_id,
        "run_name": run.name,
        "run_group": run.group,
        "run_state": run.state,
        "created_at": run.created_at,
        "seed": spec.seed,
        "grid": spec.grid,
        "history_rows": int(len(df)),
        "max_env_steps": spec.max_env_steps,
        "display_x_scale": spec.display_x_scale,
        "display_x_max": spec.display_x_max,
    }
    return df, audit


def extract_metric_points(
    spec: RunSpec,
    df: pd.DataFrame,
    metric: MetricSpec,
    source_audit: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    if df.empty:
        return pd.DataFrame(), None
    raw_x, x_key = resolve_x_values(df, spec)
    metric_key = first_existing(df, metric.aliases)
    if metric_key is None:
        missing = {
            **source_audit,
            "metric": metric.key,
            "metric_label": metric.label,
            "x_key": x_key,
            "metric_key": metric_key,
            "aliases": list(metric.aliases),
            "x_aliases": list(spec.x_aliases),
        }
        return pd.DataFrame(), missing
    values = metric.transform(df[metric_key], spec, metric_key)
    display_x = raw_x * spec.display_x_scale
    out = pd.DataFrame(
        {
            "section": spec.section,
            "method": spec.method,
            "method_label": spec.method_label,
            "seed": spec.seed,
            "grid": spec.grid,
            "native_unit": spec.native_unit,
            "x": display_x,
            "raw_x": raw_x,
            "display_x_scale": spec.display_x_scale,
            "display_x_max": spec.display_x_max,
            "metric": metric.key,
            "metric_label": metric.label,
            "value": pd.to_numeric(values, errors="coerce"),
            "x_source": x_key,
            "metric_source": metric_key,
            "run_id": spec.run_id,
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "value"])
    if spec.display_x_max is not None:
        out = out[out["x"] <= spec.display_x_max].copy()
    out = out.sort_values("x")
    out = out.drop_duplicates(subset=["x"], keep="last")
    return out, None


def collect_points(specs: list[RunSpec], metrics: tuple[MetricSpec, ...], samples: int, audit_root: Path) -> pd.DataFrame:
    api = wandb.Api(timeout=90)
    point_frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for spec in specs:
        df, source_audit = load_run_history(api, spec, samples)
        sources.append(source_audit)
        if spec.status == "oom":
            continue
        for metric in metrics:
            frame, metric_missing = extract_metric_points(spec, df, metric, source_audit)
            if metric_missing is not None:
                missing.append(metric_missing)
            if not frame.empty:
                point_frames.append(frame)
    audit_root.mkdir(parents=True, exist_ok=True)
    write_json(audit_root / "source_runs.json", sources)
    write_json(audit_root / "missing_metrics.json", missing)
    if point_frames:
        points = pd.concat(point_frames, ignore_index=True)
    else:
        points = pd.DataFrame()
    points.to_csv(audit_root / "curve_points.csv", index=False)
    write_sources_csv(audit_root / "source_runs.csv", sources)
    write_missing_csv(audit_root / "missing_metrics.csv", missing)
    return points


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_sources_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted(set().union(*(row.keys() for row in rows)) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_missing_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted(set().union(*(row.keys() for row in rows)) if rows else ["metric"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def interp_seed_curves(rows: pd.DataFrame, n_points: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
    for _seed, seed_df in rows.groupby("seed", dropna=False):
        seed_df = seed_df.sort_values("x")
        xs = seed_df["x"].to_numpy(dtype=float)
        ys = seed_df["value"].to_numpy(dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) == 0:
            continue
        if len(xs) == 1:
            seed_curves.append((xs, ys))
            continue
        uniq_x, uniq_idx = np.unique(xs, return_index=True)
        seed_curves.append((uniq_x, ys[uniq_idx]))
    if not seed_curves:
        return np.array([]), np.array([]), np.array([])
    x_min = min(float(xs.min()) for xs, _ in seed_curves)
    x_max = max(float(xs.max()) for xs, _ in seed_curves)
    if math.isclose(x_min, x_max):
        grid = np.array([x_max], dtype=float)
    else:
        grid = np.linspace(x_min, x_max, n_points)
    values = []
    for xs, ys in seed_curves:
        if len(xs) == 1:
            interp = np.full_like(grid, ys[0], dtype=float)
        else:
            interp = np.interp(grid, xs, ys, left=np.nan, right=np.nan)
        values.append(interp)
    arr = np.vstack(values)
    return grid, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def plot_seed_traces(ax: plt.Axes, rows: pd.DataFrame, color: str, linestyle: str) -> None:
    for _seed, seed_df in rows.groupby("seed", dropna=False):
        seed_df = seed_df.sort_values("x")
        ax.plot(
            seed_df["x"].to_numpy(dtype=float),
            seed_df["value"].to_numpy(dtype=float),
            color=color,
            linewidth=0.75,
            linestyle=linestyle,
            alpha=0.16,
            zorder=1,
        )


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#D9DEE7", linewidth=0.55, alpha=0.9)
    ax.tick_params(labelsize=7, colors="#111827", width=0.65, length=2.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.65)


def plot_t1(points: pd.DataFrame, output_path: Path) -> None:
    method_order = ("kicad_ppo", "jumanji_a2c", "sable")
    method_titles = {
        "kicad_ppo": "PPO",
        "jumanji_a2c": "Jumanji A2C",
        "sable": "SABLE",
    }
    fig, axes = plt.subplots(3, 3, figsize=(7.15, 5.75), sharex="col", constrained_layout=False)
    for row_idx, metric in enumerate(T1_METRICS):
        for col_idx, method in enumerate(method_order):
            ax = axes[row_idx, col_idx]
            style_axes(ax)
            if row_idx == 0:
                ax.set_title(method_titles[method], fontsize=9, fontweight="bold", pad=5)
            if col_idx == 0:
                ax.set_ylabel(metric.ylabel, fontsize=8, color="#111827")
            subset = points[
                (points["section"] == T1_AUDIT_SECTION)
                & (points["method"] == method)
                & (points["metric"] == metric.key)
            ]
            display_max = subset["display_x_max"].dropna().max() if "display_x_max" in subset else np.nan
            if np.isfinite(display_max):
                ax.set_xlim(0.0, float(display_max))
            for grid in T1_GRIDS:
                grid_subset = subset[subset["grid"] == grid]
                if grid_subset.empty:
                    if method == "jumanji_a2c" and grid == 500 and row_idx == 0:
                        ax.text(0.56, 0.12, "G500 OOM", transform=ax.transAxes, fontsize=7, color=GRID_COLORS[500])
                    continue
                xs, mean, std = interp_seed_curves(grid_subset)
                if len(xs) == 0:
                    continue
                color = GRID_COLORS[grid]
                style = GRID_STYLES[grid]
                plot_seed_traces(ax, grid_subset, color, style["linestyle"])
                markevery = max(1, len(xs) // 8)
                ax.plot(
                    xs,
                    mean,
                    color=color,
                    linewidth=2.05,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=2.2,
                    markevery=markevery,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    solid_capstyle="round",
                    zorder=3,
                )
                ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.24, linewidth=0, zorder=2)
            if metric.key == "routability":
                ax.set_ylim(-0.03, 1.03)
            if row_idx == 2:
                unit = subset["native_unit"].dropna().iloc[0] if not subset.empty else "native step"
                ax.set_xlabel(unit, fontsize=8, color="#111827")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=GRID_COLORS[grid],
            linewidth=2.7,
            linestyle=GRID_STYLES[grid]["linestyle"],
            marker=GRID_STYLES[grid]["marker"],
            markersize=4.0,
            markerfacecolor=GRID_COLORS[grid],
            markeredgecolor="white",
            markeredgewidth=0.3,
            label=f"G{grid}",
        )
        for grid in T1_GRIDS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=len(T1_GRIDS),
        fontsize=7.5,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.3,
        columnspacing=0.95,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.075, wspace=0.28, hspace=0.36)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_policy(points: pd.DataFrame, output_path: Path) -> None:
    method_order = ("ppo_per_step", "grpo_sparse", "ppo_sparse")
    method_titles = {
        "ppo_per_step": "PPO (per-step)",
        "grpo_sparse": "GRPO",
        "ppo_sparse": "PPO (terminal)",
    }
    fig, axes = plt.subplots(len(POLICY_METRICS), 3, figsize=(7.15, 5.7), sharex="col", constrained_layout=False)
    for row_idx, metric in enumerate(POLICY_METRICS):
        for col_idx, method in enumerate(method_order):
            ax = axes[row_idx, col_idx]
            style_axes(ax)
            if row_idx == 0:
                ax.set_title(method_titles[method], fontsize=9, fontweight="bold", pad=5)
            if col_idx == 0:
                ax.set_ylabel(metric.ylabel, fontsize=8, color="#111827")
            subset = points[
                (points["section"] == POLICY_AUDIT_SECTION)
                & (points["method"] == method)
                & (points["metric"] == metric.key)
            ]
            if subset.empty:
                ax.text(0.5, 0.5, "not logged", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="#64748B")
                continue
            xs, mean, std = interp_seed_curves(subset)
            if len(xs):
                color = METHOD_COLORS[method]
                markevery = max(1, len(xs) // 9)
                plot_seed_traces(ax, subset, color, "-")
                ax.plot(
                    xs,
                    mean,
                    color=color,
                    linewidth=2.05,
                    marker="o",
                    markersize=2.2,
                    markevery=markevery,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    solid_capstyle="round",
                    zorder=3,
                )
                ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.24, linewidth=0, zorder=2)
            if metric.key == "routability":
                ax.set_ylim(-0.03, 1.03)
            if row_idx == len(POLICY_METRICS) - 1:
                ax.set_xlabel("logged step", fontsize=8, color="#111827")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.95, bottom=0.07, wspace=0.28, hspace=0.36)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def validate_counts(points: pd.DataFrame, source_runs: list[dict[str, object]], audit_root: Path) -> None:
    ok_sources = [row for row in source_runs if row.get("state") != "oom"]
    counts = {
        "t1_transformer_sources": sum(
            1 for r in ok_sources if r["section"] == T1_AUDIT_SECTION and r["method"] == "kicad_ppo"
        ),
        "t1_jumanji_sources": sum(
            1 for r in ok_sources if r["section"] == T1_AUDIT_SECTION and r["method"] == "jumanji_a2c"
        ),
        "t1_sable_sources": sum(
            1 for r in ok_sources if r["section"] == T1_AUDIT_SECTION and r["method"] == "sable"
        ),
        "policy_sources": sum(1 for r in ok_sources if r["section"] == POLICY_AUDIT_SECTION),
        "curve_points": int(len(points)),
    }
    expected = {
        "t1_transformer_sources": 20,
        "t1_jumanji_sources": 8,
        "t1_sable_sources": 10,
        "policy_sources": 12,
    }
    errors = {key: (counts[key], value) for key, value in expected.items() if counts[key] != value}
    write_json(audit_root / "validation_counts.json", {"counts": counts, "expected": expected, "errors": errors})
    if errors:
        raise SystemExit(f"unexpected W&B source counts: {errors}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overleaf-root", type=Path, default=Path(os.environ.get("OVERLEAF_ROOT", "var/results/kdd/paper_outputs")))
    parser.add_argument("--audit-root", type=Path,
                        default=Path(os.environ.get("WANDB_AUDIT_ROOT")
                                     or data_root_path("paper_wandb_curve_audit_20260507")))
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse audit curve_points.csv instead of querying W&B.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "axes.edgecolor": "#222222",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    audit_root = args.audit_root
    if args.skip_fetch:
        cached = audit_root / "curve_points_all.csv"
        points = pd.read_csv(cached if cached.exists() else audit_root / "curve_points.csv")
    else:
        t1_points = collect_points(t1_run_specs(), T1_METRICS, args.samples, audit_root / T1_AUDIT_SECTION)
        policy_points = collect_points(
            policy_run_specs(), POLICY_METRICS, args.samples, audit_root / POLICY_AUDIT_SECTION
        )
        points = pd.concat([t1_points, policy_points], ignore_index=True)
        audit_root.mkdir(parents=True, exist_ok=True)
        points.to_csv(audit_root / "curve_points_all.csv", index=False)
        combined_sources = []
        for section in (T1_AUDIT_SECTION, POLICY_AUDIT_SECTION):
            combined_sources.extend(json.loads((audit_root / section / "source_runs.json").read_text(encoding="utf-8")))
        write_json(audit_root / "source_runs_all.json", combined_sources)
        write_json(
            audit_root / "missing_metrics_all.json",
            {
                section: json.loads((audit_root / section / "missing_metrics.json").read_text(encoding="utf-8"))
                for section in (T1_AUDIT_SECTION, POLICY_AUDIT_SECTION)
            },
        )
    # Validate section-specific source counts.
    if not args.skip_fetch:
        validate_counts(points, combined_sources, audit_root)
    figs = args.overleaf_root / "figs"
    plot_t1(points, figs / "rq1_wandb_validation_curves.pdf")
    plot_policy(points, figs / "rq5_wandb_validation_curves.pdf")
    print(f"wrote {figs / 'rq1_wandb_validation_curves.pdf'}")
    print(f"wrote {figs / 'rq5_wandb_validation_curves.pdf'}")
    print(f"audit root: {audit_root}")


if __name__ == "__main__":
    main()
