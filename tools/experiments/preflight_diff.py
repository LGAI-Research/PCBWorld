#!/usr/bin/env python3
"""Batch preflight: diff resolved configs across cases before launch.

Compares the resolved values of N run records (``run_record.yaml``, written by
``train_ppo --dump-config-only`` or at the start of any run; see
``configs/loader/record.py``). Keys identical across every
case are folded to a count; keys whose values differ are printed as a
key x case matrix. ``--expect`` declares the keys *intended* to differ (the
experiment axes) — a difference in any OTHER key aborts with exit 1 so the
human stops the batch launch (e.g. the 260704 batch where --time-feature was
bundled into only one gamma case, confounding the axis). Only the config values
are compared — where they came from is in each record (``python -m configs.loader.explain``).

Usage (dump one config per case first, then diff)::

    for c in $(bash -c 'source sandbox/d2b_midboard/260706_cases.sh; d2b_all_cases'); do
      DRY_RUN=0 bash .../train_one.sh --case $c ... --dump-config-only  # concept example
    done
    python tools/experiments/preflight_diff.py 'var/.../*/run_record.yaml' \
        --expect gamma,masking_rule,truncation_bootstrap

Exit codes: 0 = all differences are declared axes; 1 = unexpected difference
(or bad input). Run in the ``pcbworld`` env (needs PyYAML).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path


_ABSENT = "<absent>"


def expand_paths(patterns: list[str]) -> list[str]:
    """Expand globs, keep order, dedupe; error out on a pattern with no match."""
    paths: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            sys.exit(f"error: no file matches {pat!r}")
        for p in hits:
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                paths.append(ap)
    return paths


def case_labels(paths: list[str]) -> list[str]:
    """Shortest distinguishing labels: paths relative to their common prefix;
    when all basenames are identical (the usual run_record.yaml), the
    per-case directory alone."""
    common = os.path.commonpath(paths)
    labels = [os.path.relpath(p, common) for p in paths]
    if len({os.path.basename(p) for p in paths}) == 1:
        labels = [os.path.dirname(lb) or lb for lb in labels]
    return labels


def diff_configs(cases: list[tuple[str, dict]]) -> tuple[list[str], dict[str, list]]:
    """(keys identical across all cases, {differing key: per-case values}).

    A key missing from some case counts as a difference (value ``<absent>``) —
    that is exactly a code-version/flag-set mismatch worth stopping for.
    """
    keys = sorted({k for _, cfg in cases for k in cfg})
    same: list[str] = []
    diff: dict[str, list] = {}
    for k in keys:
        vals = [cfg.get(k, _ABSENT) for _, cfg in cases]
        if all(v == vals[0] for v in vals[1:]):
            same.append(k)
        else:
            diff[k] = vals
    return same, diff


def format_matrix(labels: list[str], diff: dict[str, list]) -> str:
    header = ["key"] + labels
    rows = [
        [key] + [_ABSENT if v is _ABSENT else repr(v) for v in vals]
        for key, vals in sorted(diff.items())
    ]
    widths = [
        max(len(row[i]) for row in [header] + rows) for i in range(len(header))
    ]
    lines = [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip()
        for row in [header] + rows
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    repo_dir = str(Path(__file__).resolve().parents[2])  # configs.loader when run as a script
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from configs.loader.cli import StrictArgumentParser
    ap = StrictArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("configs", nargs="+",
                    help="run_record.yaml paths (globs allowed)")
    ap.add_argument("--expect", action="append", default=[], metavar="KEY[,KEY...]",
                    help="comma-separated keys intended to differ across cases "
                         "(the experiment axes); repeatable")
    args = ap.parse_args(argv)

    paths = expand_paths(args.configs)
    if len(paths) < 2:
        sys.exit("error: need >= 2 config files to diff")
    from configs.loader.explain import load_provenance
    cases = []
    for label, path in zip(case_labels(paths), paths):
        try:
            provenance = load_provenance(path)
        except ValueError as e:
            sys.exit(f"error: {e}")
        cases.append((label, {k: e["value"] for k, e in provenance.items()}))

    same, diff = diff_configs(cases)
    expected = {
        key.strip().replace("-", "_")
        for group in args.expect for key in group.split(",") if key.strip()
    }

    print(f"[preflight] {len(paths)} configs; "
          f"{len(same)} keys identical across all cases")
    if diff:
        print(f"[preflight] {len(diff)} keys differ:")
        print(format_matrix([label for label, _ in cases], diff))

    silent_axes = expected - set(diff)
    if silent_axes:
        print("[preflight] WARNING: --expect keys with NO difference "
              f"(axis not actually varying?): {', '.join(sorted(silent_axes))}")

    unexpected = set(diff) - expected
    if unexpected:
        print("[preflight] FAIL: unexpected differences in: "
              f"{', '.join(sorted(unexpected))}")
        print("            declare intended axes via --expect, "
              "or fix the launch flags before the batch.")
        return 1
    print("[preflight] OK: all differences are declared experiment axes"
          if diff else "[preflight] OK: configs are identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
