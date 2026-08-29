#!/usr/bin/env python3
"""Write a ``_run_meta.json`` provenance sidecar into an output cell dir.

Records the resolved input path + git commit + producing command,
so any produced cell is traceable to its source
even though inputs and outputs live under different roots. Hand-invoked by rollout
wrappers, e.g.::

    python tools/write_run_meta.py "$celldir" --dataset synth_2L_v2 --cmd "$0 $*"

Run in the ``cadagent`` env when ``--dataset`` is given (needs the resolver / PyYAML).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def write_run_meta(out_dir, dataset=None, cmd=None, extra=None) -> Path:
    """Write ``<out_dir>/_run_meta.json`` and return its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "command": cmd,
        "cwd": str(Path.cwd()),
    }
    if dataset:
        sys.path.insert(0, str(_REPO))
        from configs.loader.paths import resolve_dataset

        meta["dataset"] = dataset
        meta["dataset_path"] = str(resolve_dataset(dataset))
    if extra:
        meta.update(extra)
    path = out / "_run_meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("out_dir", help="cell dir to write _run_meta.json into")
    ap.add_argument("--dataset", default=None,
                    help="logical dataset name (resolved + recorded)")
    ap.add_argument("--cmd", default=None,
                    help="command line that produced this cell")
    args = ap.parse_args()
    print(write_run_meta(args.out_dir, dataset=args.dataset, cmd=args.cmd))


if __name__ == "__main__":
    main()
