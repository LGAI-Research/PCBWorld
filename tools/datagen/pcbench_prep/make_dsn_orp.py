#!/usr/bin/env python3
"""
Step 4 of the PCBench -> exacad_sorted chain: the ``.dsn`` / ``.orp`` siblings the
rule-based routers (``methods/baselines/rule_based/``) read, one ``<NNNN>_<name>/``
folder per sorted board.

Runs the engine's converter ``engine/pcbnew_prep/make_dsn_orp_v3.py`` (pcbnew's
ExportSpecctraDSN, then each board's ``.kicad_pro`` net-class rules written exactly)
in the pcbnew interpreter ``kicad_tools`` resolves, so pcbnew never loads in this
process. Extra arguments go to the converter (``--limit``, ``--targets``). Its per-board
results land in ``PCBENCH_DSN_ROOT/_results.json``, rewritten by every run.

Exit status: 0 when every board converted; 3 when the converter finished and some
boards are ``fail`` rows (each reason on its row); anything else when it did not
finish (no ``_results.json``).

Environment:
  PCBENCH_SORTED_OUT  the exacad_sorted directory from sort_prefix.py (input)
  PCBENCH_DSN_ROOT    the DSN/ORP mirror to write (output; ``pcbench_exacad_dsn``
                      in configs/paths.yaml)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import kicad_tools

CONVERTER = kicad_tools.REPO_ROOT / "engine" / "pcbnew_prep" / "make_dsn_orp_v3.py"
REPORT = "_results.json"
FAILED_BOARDS = 3


def main(argv: list[str]) -> int:
    for name in ("PCBENCH_SORTED_OUT", "PCBENCH_DSN_ROOT"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} is not set. Point it at the matching directory "
                             "(see tools/datagen/pcbench_prep/README.md).")
    kicad_tools.announce()   # exits when no interpreter imports pcbnew
    report = Path(os.environ["PCBENCH_DSN_ROOT"]) / REPORT
    report.unlink(missing_ok=True)   # an earlier run's report must not pass for this one
    child = dict(os.environ, PCBENCH_PCB_ROOT=os.environ["PCBENCH_SORTED_OUT"])
    rc = subprocess.run([kicad_tools.pcbnew_python(), str(CONVERTER), *argv, "--report", str(report)],
                        env=child).returncode
    if not report.is_file():   # the converter writes it after its last board
        return rc if rc not in (0, FAILED_BOARDS) else 1
    return 0 if rc == 0 else FAILED_BOARDS


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
