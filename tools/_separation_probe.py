#!/usr/bin/env python3
"""Route a board for real, then read both processes' memory maps.

Run by ``tools/check_separation.py --runtime``; also runnable on its own.

While the environment holds a live engine session — so the engine is
definitely doing work — this dumps:

  * this process's ``/proc/self/maps``       (the environment)
  * the engine server child's ``/proc/<pid>/maps``

and asserts that the engine's shared library is mapped in the child and not
in the parent. Prints ``SEPARATION_PROOF: OK`` when that holds.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LIB_RE = re.compile(r"kicad_rl_router.*\.so")
BOARD = REPO / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"


def maps_of(pid: str | int) -> list[str]:
    return Path(f"/proc/{pid}/maps").read_text().splitlines()


def engine_lib_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if LIB_RE.search(ln)]


def child_pids(pid: int) -> list[int]:
    out = []
    for task in Path(f"/proc/{pid}/task").iterdir():
        children = (task / "children")
        if children.exists():
            out += [int(x) for x in children.read_text().split()]
    return out


def main() -> int:
    from pcb_world.engine import engine_available
    if not engine_available():
        print("engine build not found — build it first "
              "(BUILD_DIR=\"$PWD/build_rl\" bash engine/build_rl_router.sh)",
              file=sys.stderr)
        return 2
    if not BOARD.exists():
        print(f"missing fixture board {BOARD}", file=sys.stderr)
        return 2

    from pcb_world.engine.kicad_engine import KiCadEngine
    from pcb_world.engine.router_client import RouterProxy

    eng = KiCadEngine(str(BOARD))
    try:
        if not isinstance(eng._r, RouterProxy):
            print("engine is running in-process (KICAD_ENGINE_IPC=0) — this "
                  "probe measures the IPC boundary", file=sys.stderr)
            return 2
        # Do real engine work so the binding is unambiguously in use.
        eng.build_connectivity()
        pads = eng.get_pads()
        eng.run_drc()
        server_pid = eng._r._conn.pid

        me = os.getpid()
        parent_maps = maps_of("self")
        server_maps = maps_of(server_pid)

        print(f"environment process : pid {me}")
        print(f"engine server child : pid {server_pid}")
        print(f"board               : {BOARD.relative_to(REPO)} "
              f"({len(pads)} pads)")
        print(f"child of {me}?       : "
              f"{server_pid in child_pids(me)}")

        parent_hits = engine_lib_lines(parent_maps)
        server_hits = engine_lib_lines(server_maps)

        print(f"\n--- /proc/{me}/maps  (environment) — "
              f"{len(parent_maps)} mappings, "
              f"{len(parent_hits)} matching kicad_rl_router*.so ---")
        for ln in parent_hits:
            print("   ", ln)
        if not parent_hits:
            print("    (none)")

        print(f"\n--- /proc/{server_pid}/maps  (engine server) — "
              f"{len(server_maps)} mappings, "
              f"{len(server_hits)} matching kicad_rl_router*.so ---")
        for ln in server_hits:
            print("   ", ln)

        ok = not parent_hits and bool(server_hits)
        print("\nSEPARATION_PROOF:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    finally:
        eng.close()


if __name__ == "__main__":
    raise SystemExit(main())
