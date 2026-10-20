#!/usr/bin/env python3
"""Print the D3-A board files (configs/datasets/d3.json easy.test) under a corpus root.

    python d3a_boards.py <corpus_dir> <board_filename> <limit>

One path per line, in d3.json order: ``<corpus_dir>/<board_id>/<board_filename>``.
``board_filename`` may be a glob; the first match (sorted) in each board folder is used.
``limit`` 0, or at least the D3-A size, means the whole D3-A set, and then any
missing board is an error (exit 1, the missing ids on stderr) so a partial corpus
never passes for D3-A; a smaller ``limit`` skips missing boards with a warning and
prints the first ``limit`` present ones. A wrong argument count or a ``limit`` that
is not a non-negative integer is a usage error (exit 2).
"""
import glob
import json
import os
import sys
from pathlib import Path

D3_JSON = Path(__file__).resolve().parents[4] / "configs" / "datasets" / "d3.json"


def main(argv: list[str]) -> int:
    try:
        corpus, fname, limit_arg = argv
        limit = int(limit_arg)
    except ValueError:  # wrong argument count or a non-integer limit
        sys.stderr.write(__doc__)
        return 2
    if limit < 0:
        sys.stderr.write(__doc__)
        return 2
    bids = json.loads(D3_JSON.read_text())["easy"]["test"]
    out, missing = [], []
    for bid in bids:
        matches = sorted(glob.glob(os.path.join(glob.escape(os.path.join(corpus, bid)), fname)))
        if matches:
            out.append(matches[0])
        else:
            missing.append(bid)
    whole_set = limit == 0 or limit >= len(bids)
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        msg = (f"{len(missing)} of {len(bids)} D3-A boards have no {fname} "
               f"under {corpus}: {shown}\n")
        if whole_set:
            sys.stderr.write(f"[ERROR] {msg}")
            return 1
        sys.stderr.write(f"[WARN] {msg}")
    if not whole_set:
        out = out[:limit]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
