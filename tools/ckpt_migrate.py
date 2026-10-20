#!/usr/bin/env python3
"""Write a checkpoint's resolved config into the file.

A checkpoint stores its training ``args``; knobs added later are missing from old files and
are resolved at load time by ``configs.loader.ckpt.load_ckpt_config`` (pre-knob values from
``configs/ckpt_migrations.yaml``, weight-shaped keys derived from the state_dict, renamed keys
mapped). This tool writes that same result into the file, so it is self-contained (e.g. before
publishing it). It exists to process old checkpoints, so it never runs in strict mode.
Idempotent: a checkpoint whose args already hold every resolved key is left untouched.

    python tools/ckpt_migrate.py <ckpt.pt> [...]          # rewrite in place (a .bak copy is kept)
    python tools/ckpt_migrate.py --dry-run <ckpt.pt> ...  # only report the keys that would be added
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    from configs.loader.cli import StrictArgumentParser
    ap = StrictArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write")
    a = ap.parse_args()
    import torch
    from configs.loader.ckpt import legacy_args, load_ckpt_config

    for path in a.ckpts:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        values, sources = load_ckpt_config(payload, strict=False)
        filled = {k: f"{values[k]!r} ({s})" for k, s in sources.items() if s != "ckpt"}
        if not filled:
            print(f"{path}: complete (nothing to add)")
            continue
        print(f"{path}: {'would add' if a.dry_run else 'adding'} {filled}")
        if a.dry_run:
            continue
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        payload["args"] = legacy_args(values)
        torch.save(payload, path)


if __name__ == "__main__":
    main()
