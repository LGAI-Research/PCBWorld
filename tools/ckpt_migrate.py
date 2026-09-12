#!/usr/bin/env python3
"""Write the paper-era values of pre-knob settings into old checkpoints.

A checkpoint stores the training ``args``; knobs added later are missing from old files and
are filled at load time from ``experiments/kdd/configs/legacy_ckpt_defaults.yaml``. This tool
writes those same values into the file, so it is self-contained (e.g. before publishing it).
Idempotent: a checkpoint that already carries every key is left untouched.

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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write")
    a = ap.parse_args()
    import torch
    from configs.loader.schema import with_legacy_ckpt_defaults

    for path in a.ckpts:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        args = payload.get("args")
        if not isinstance(args, dict):
            print(f"{path}: no 'args' dict in the payload — skipped")
            continue
        filled = with_legacy_ckpt_defaults(args)
        added = {k: filled[k] for k in filled if k not in args}
        if not added:
            print(f"{path}: complete (nothing to add)")
            continue
        print(f"{path}: {'would add' if a.dry_run else 'adding'} {added}")
        if a.dry_run:
            continue
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        payload["args"] = filled
        torch.save(payload, path)


if __name__ == "__main__":
    main()
