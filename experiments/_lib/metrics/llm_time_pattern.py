#!/usr/bin/env python3
"""Reusable *pattern* for back-filling LLM per-episode routing time into a cell's
``per_rollout.csv`` (column ``per_board_rollout_time``).

WHY THIS EXISTS
---------------
The unified eval cells re-run only DRC on pre-routed LLM boards, so an LLM cell's
``per_rollout.csv`` has ``per_board_rollout_time`` blank (``eval_time_sec`` is the
*DRC* time, not the agent's routing latency). The real per-episode latency
(the paper's sec/ep, Tables 15/16) is recorded by the *original* agentic rollout
under a separate LLM-rollout root, e.g. ``score_rollouts.py`` /
``aggregate_p_cp_at_k.py`` trees (``.../per_board/<board_id>/*.json``,
``summary.csv``, ``overall.json`` carrying ``wall_time_sec``).

This module does **not** mutate experiments by default. It is the *pattern* a
later, dedicated back-fill agent reuses:

  * ``latency_index(llm_root, time_keys)`` -> ``{board_id: seconds}`` by scanning
    an LLM-rollout root for a time-like field (tolerant to field/layout naming).
  * ``plan_fill(cell_rel, llm_root)`` -> rows that *would* be written (dry-run).
  * CLI default is ``--dry-run`` (read-only, prints the plan). ``--write`` exists
    for the separate agent and is intentionally never invoked by the table
    scripts; it prints a loud banner before touching the file.

The paper table scripts (``table3.py`` etc.) do NOT import this module; they only
read whatever is already in ``per_board_rollout_time``. Once the back-fill agent
has populated the LLM cells, those scripts pick up the times automatically.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# Field names that have meant "per-episode routing latency, seconds" across the
# various LLM rollout loggers. The back-fill agent should confirm which one its
# root actually uses; first match wins.
DEFAULT_TIME_KEYS = (
    "per_board_rollout_time", "routing_time_sec", "rollout_time_sec",
    "latency_sec", "wall_time_sec", "sec_per_ep", "elapsed_sec", "elapsed",
)


def _first_num(d: dict, keys) -> float | None:
    for k in keys:
        if k in d:
            v = C.u.parse_metric(d[k])
            if v is not None:
                return v
    return None


def latency_index(llm_root: Path, time_keys=DEFAULT_TIME_KEYS) -> dict[str, float]:
    """Scan an LLM-rollout root for per-board latency, keyed by ``board_id``.

    Tolerant to two common layouts:
      * ``**/per_board/<board_id>/aggregate.json`` (or ``*.json``) with a time key
      * ``**/summary.csv`` rows carrying ``board_id`` + a time column

    All reads go through ``common.open_ro`` (read-only).
    """
    root = Path(llm_root)
    idx: dict[str, float] = {}

    # (a) per_board/<id>/*.json
    for j in root.rglob("per_board/*/aggregate.json"):
        bid = j.parent.name
        try:
            with C.open_ro(j) as fh:
                data = json.load(fh)
        except Exception:
            continue
        t = _first_num(data, time_keys)
        if t is not None:
            idx.setdefault(bid, t)

    # (b) summary.csv with board_id + a time column
    for s in root.rglob("summary.csv"):
        try:
            with C.open_ro(s) as fh:
                for row in csv.DictReader(fh):
                    bid = row.get("board_id")
                    if not bid or bid in idx:
                        continue
                    t = _first_num(row, time_keys)
                    if t is not None:
                        idx[bid] = t
        except Exception:
            continue
    return idx


def plan_fill(cell_rel: str, llm_root: Path) -> list[dict]:
    """Return the rows of ``cell_rel`` that *would* gain a
    ``per_board_rollout_time`` from ``latency_index``, as a list of
    ``{board_id, artifact, old, new}`` dicts. Pure read; writes nothing."""
    idx = latency_index(llm_root)
    rows = C.load_rollouts(cell_rel)
    cell = C.disk_cell_name(cell_rel)
    plan = []
    for r in rows:
        p = C.u.parse_cell_artifact(r.get("artifact_path", "") or "", cell)
        if p is None:
            continue
        bid = p[0]
        new = idx.get(bid)
        if new is None:
            continue
        plan.append({"board_id": bid,
                     "artifact": Path(r.get("artifact_path", "")).name,
                     "old": r.get("per_board_rollout_time", ""),
                     "new": new})
    return plan


def _write_fill(cell_rel: str, llm_root: Path) -> int:
    """DESTRUCTIVE: rewrite the cell's per_rollout.csv with filled times.
    Intended ONLY for the dedicated back-fill agent. Not used by table scripts."""
    print("=" * 70, file=sys.stderr)
    print(f"!! MUTATING EXPERIMENT FILE: {C.cell_dir(cell_rel)/'per_rollout.csv'}",
          file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    idx = latency_index(llm_root)
    cell = C.disk_cell_name(cell_rel)
    path = C.cell_dir(cell_rel) / "per_rollout.csv"
    # NOTE: bypasses common.open_ro intentionally; only this function may write.
    with path.open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        fields = rdr.fieldnames or []
        rows = list(rdr)
    n = 0
    for r in rows:
        p = C.u.parse_cell_artifact(r.get("artifact_path", "") or "", cell)
        if p is None:
            continue
        t = idx.get(p[0])
        if t is not None:
            r["per_board_rollout_time"] = f"{t}"
            n += 1
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, help="cell rel path, e.g. d2a/pcbworld_gpt-5.4")
    ap.add_argument("--llm-root", required=True, type=Path,
                    help="root of the original LLM-rollout logs (per_board/summary)")
    ap.add_argument("--write", action="store_true",
                    help="DESTRUCTIVE back-fill (separate agent only); default is dry-run")
    args = ap.parse_args()

    if args.write:
        n = _write_fill(args.cell, args.llm_root)
        print(f"[write] filled per_board_rollout_time for {n} rows in {args.cell}")
        return

    plan = plan_fill(args.cell, args.llm_root)
    print(f"[dry-run] {args.cell}: {len(plan)} rows would be filled (no write).")
    for p in plan[:10]:
        print(f"  {p['board_id']:24s} {p['artifact']}  {p['old'] or '∅'} -> {p['new']}")
    if len(plan) > 10:
        print(f"  ... (+{len(plan) - 10} more)")


if __name__ == "__main__":
    main()
