"""Tiny launcher that pre-loads ``grid_router`` from the user-owned cache
before KRT's ``startup_checks.check_rust_library`` gets a chance to find
the (potentially GLIBC-incompatible) pre-built shared library that ships
with the KRT install tree.

Invoked by ``krt_runner.run_route``; do not call directly.

The pre-built ``grid_router.so`` under ``<KRT>/rust_router/`` was compiled
against a newer glibc than some Ubuntu 20.04-class hosts ship. KRT's
``check_rust_library`` does ``sys.path.insert(0, '<KRT>/rust_router')``
and then ``import grid_router`` — so once the incompatible .so is on
``sys.path`` ahead of ours, Python finds it first, fails to dlopen, and
KRT prints "Rust router module not found".

We win the race by stuffing the locally-built ``grid_router`` into
``sys.modules`` before KRT's startup logic runs. The cached import wins,
``check_rust_library`` returns happily, and the real route.py body runs.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    cache_dir = os.environ.get("KRT_RUST_SO_DIR")
    if not cache_dir:
        cache_dir = str(Path.home() / ".cache" / "krt_runner")
    if cache_dir not in sys.path:
        sys.path.insert(0, cache_dir)

    # Pre-load grid_router so KRT's check_rust_library sees a cached
    # module and never tries to import the upstream broken .so.
    import grid_router  # noqa: F401

    route_py = os.environ["KRT_ROUTE_PY"]
    sys.argv = [route_py, *sys.argv[1:]]
    runpy.run_path(route_py, run_name="__main__")


if __name__ == "__main__":
    main()
