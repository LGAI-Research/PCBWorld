#!/usr/bin/env python3
"""Render final-board visualizations for the best V56 rollout rows.

This is intentionally a post-processing tool.  The full rollout evaluator
should stay focused on saving routed KiCad boards and metrics quickly; this
script reads ``rollouts.csv`` afterwards and renders only the best final PCB per
dataset/run/checkpoint group.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    value = value.strip("._")
    return value or "item"


def _float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("run_name", "")),
        str(row.get("checkpoint_label", "")),
        str(row.get("checkpoint_name", "")),
    )


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, int, int, str]:
    """Sort key where the first row is the best.

    Priority:
    1. routability descending
    2. final_potential descending
    3. drc_violations ascending
    4. wirelength ascending
    5. via_count ascending
    6. deterministic board/rollout/output tie-breaks
    """
    return (
        -_float(row.get("routability"), -math.inf),
        -_float(row.get("final_potential"), -math.inf),
        _float(row.get("drc_violations"), math.inf),
        _float(row.get("wirelength"), math.inf),
        _float(row.get("via_count"), math.inf),
        _int(row.get("board_index"), 0),
        _int(row.get("rollout_idx"), 0),
        str(row.get("output_pcb", "")),
    )


def load_rollout_rows(csv_paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in csv_paths:
        with Path(path).open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def select_best_rows(
    rows: Iterable[dict[str, Any]],
    *,
    require_existing_output: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        output_pcb = row.get("output_pcb")
        if not output_pcb:
            continue
        if require_existing_output and not Path(str(output_pcb)).exists():
            continue
        grouped.setdefault(_group_key(row), []).append(dict(row))

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        selected.append(sorted(grouped[key], key=_rank_key)[0])
    return selected


def _output_stem(row: dict[str, Any]) -> str:
    return _safe_name(
        "__".join(
            [
                str(row.get("dataset", "")),
                str(row.get("run_name", "")),
                str(row.get("checkpoint_label", "")),
                f"board_{_int(row.get('board_index'), 0):05d}",
                f"r{_int(row.get('rollout_idx'), 0):02d}",
            ]
        )
    )


def _manifest_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    for key in ("svg_path", "render_error"):
        if key not in seen:
            seen.append(key)
    return seen


def render_best_rows(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    renderer_cls: type | None = None,
    write_svg: bool = True,
) -> Path:
    if renderer_cls is None:
        from pcb_world.rendering.renderer import PCBRenderer

        renderer_cls = PCBRenderer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = renderer_cls()
    rendered_rows: list[dict[str, Any]] = []

    for row in rows:
        rendered = dict(row)
        stem = _output_stem(row)
        output_pcb = str(row.get("output_pcb", ""))
        errors: list[str] = []

        if write_svg:
            svg_path = output_dir / f"{stem}.svg"
            try:
                renderer.export_svg(output_pcb, str(svg_path))
                rendered["svg_path"] = str(svg_path)
            except Exception as exc:  # noqa: BLE001
                rendered["svg_path"] = ""
                errors.append(f"svg: {exc!r}")

        rendered["render_error"] = "; ".join(errors)
        rendered_rows.append(rendered)

    manifest_path = output_dir / "best_visualizations.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=_manifest_fieldnames(rendered_rows),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rendered_rows)
    return manifest_path


def _rollouts_for_manifest_dir(manifest_dir: Path) -> Path:
    return Path(manifest_dir) / "rollouts.csv"


def _collect_inputs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    inputs: list[tuple[Path, Path]] = []
    for manifest_dir_text in args.manifest_dir or []:
        manifest_dir = Path(manifest_dir_text).expanduser()
        inputs.append((_rollouts_for_manifest_dir(manifest_dir), manifest_dir))
    for csv_text in args.rollouts_csv or []:
        csv_path = Path(csv_text).expanduser()
        inputs.append((csv_path, csv_path.parent))
    if not inputs:
        raise SystemExit("Provide at least one --manifest-dir or --rollouts-csv")
    return inputs


def _output_dir_for(base: Path, manifest_dir: Path, count: int) -> Path:
    if count == 1:
        return base
    return base / _safe_name(manifest_dir.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", action="append", default=None)
    parser.add_argument("--rollouts-csv", action="append", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-svg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = _collect_inputs(args)
    summaries: list[dict[str, Any]] = []
    for csv_path, manifest_dir in inputs:
        rows = load_rollout_rows([csv_path])
        selected = select_best_rows(rows)
        base_output = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else manifest_dir / "best_visualizations"
        )
        output_dir = _output_dir_for(base_output, manifest_dir, len(inputs))
        manifest_path = render_best_rows(
            selected,
            output_dir,
            write_svg=not args.no_svg,
        )
        summaries.append({
            "rollouts_csv": str(csv_path),
            "output_dir": str(output_dir),
            "best_rows": len(selected),
            "manifest": str(manifest_path),
        })
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
