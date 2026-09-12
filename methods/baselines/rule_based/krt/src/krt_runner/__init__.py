"""Pip-installable thin wrapper around KiCadRoutingTools/route.py.

KRT itself is a flat-imported tool tree (101 modules expecting to live at
the top of ``sys.path``) and is not re-packaged here; this wrapper just
records the install location and provides a ``run_route()`` helper /
``krt-route`` CLI that subprocess-invokes ``route.py`` with the correct
``cwd`` so KRT's own imports resolve.

Install location resolution (first match wins):
    1. env var ``KRT_ROOT``
    2. default ``external/KiCadRoutingTools`` in this repo (the pinned
       checkout made by ``tools/setup/fetch_baselines.sh``). The repo-side
       runner falls back to the ``krt_tool`` entry of ``configs/paths.yaml``
       when that checkout is absent (this package stays repo-import-free).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# This file lives at methods/baselines/rule_based/krt/src/krt_runner/ (the
# editable install keeps the real path), so the repo root is parents[6].
_DEFAULT_KRT_ROOT = Path(__file__).resolve().parents[6] / "external" / "KiCadRoutingTools"


def get_krt_root() -> Path:
    """Return the absolute path to the KRT install tree."""
    env = os.environ.get("KRT_ROOT")
    return Path(env).resolve() if env else _DEFAULT_KRT_ROOT


def _resolve_route_py() -> Path:
    root = get_krt_root()
    route_py = root / "route.py"
    if not route_py.is_file():
        raise FileNotFoundError(
            f"KRT route.py not found at {route_py}. "
            f"Set KRT_ROOT to the directory containing route.py."
        )
    return route_py


def _build_grid_router_so(target_so: Path) -> None:
    """Build ``grid_router.so`` from KRT's rust_router source.

    The upstream pre-built ``libgrid_router.so`` is linked against
    GLIBC newer than what some hosts ship (e.g. Ubuntu 20.04 has
    glibc 2.31, the wheel needs 2.33+). We rebuild from source into
    a user-owned target dir and stage the result at ``target_so``.

    Requires ``cargo`` on PATH (install via ``conda install -c
    conda-forge rust``).
    """
    import shutil

    if shutil.which("cargo") is None:
        raise RuntimeError(
            "cargo not on PATH; install Rust to rebuild grid_router.so "
            "(conda install -c conda-forge rust)"
        )

    manifest = get_krt_root() / "rust_router" / "Cargo.toml"
    if not manifest.is_file():
        raise FileNotFoundError(f"Cargo.toml not found at {manifest}")

    build_dir = target_so.parent / "rust_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "cargo", "build", "--release",
            "--manifest-path", str(manifest),
            "--target-dir", str(build_dir),
        ],
        check=True,
    )

    built = build_dir / "release" / "libgrid_router.so"
    if not built.is_file():
        raise RuntimeError(f"cargo build succeeded but {built} is missing")
    if target_so.exists() or target_so.is_symlink():
        target_so.unlink()
    shutil.copyfile(built, target_so)
    target_so.chmod(0o755)


def _so_is_loadable(path: Path) -> bool:
    """True if ``ctypes`` can dlopen ``path`` in this process.

    Used to detect GLIBC / arch mismatches before subprocess invocation.
    """
    import ctypes
    try:
        ctypes.CDLL(str(path))
        return True
    except OSError:
        return False


def _ensure_grid_router_so() -> Path:
    """Materialize a usable ``grid_router.so`` and return its parent dir.

    KRT's ``startup_checks`` looks for ``grid_router.so`` on
    ``sys.path``. The shared KRT install ships only
    ``rust_router/target/release/libgrid_router.so`` and (a) is
    typically not writable by the user, and (b) may be built against
    a newer glibc than the host. We stage a per-user copy:

    Resolution (first match wins):
      1. env var ``KRT_RUST_SO_DIR`` — directory containing
         ``grid_router.so`` (skip creation if present)
      2. ``~/.cache/krt_runner/grid_router.so``

    If the upstream pre-built .so cannot load on this host, ``cargo``
    is invoked to rebuild from source into the cache dir (one-time).
    """
    explicit = os.environ.get("KRT_RUST_SO_DIR")
    d = Path(explicit).resolve() if explicit else (
        Path.home() / ".cache" / "krt_runner"
    )
    d.mkdir(parents=True, exist_ok=True)
    target = d / "grid_router.so"

    if target.exists() and _so_is_loadable(target):
        return d

    upstream = (
        get_krt_root() / "rust_router" / "target" / "release"
        / "libgrid_router.so"
    )
    if upstream.is_file():
        # Try the upstream pre-built first — if loadable, mirror it.
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(upstream)
        except OSError:
            import shutil
            shutil.copyfile(upstream, target)
        if _so_is_loadable(target):
            return d
        # Pre-built doesn't load (typically GLIBC mismatch) — fall through.
        target.unlink(missing_ok=True)

    # Rebuild from source.
    _build_grid_router_so(target)
    if not _so_is_loadable(target):
        raise RuntimeError(
            f"Rebuilt {target} still fails to load; check Rust toolchain "
            f"+ glibc compatibility."
        )
    return d


def run_route(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Invoke KRT's ``route.py`` via a launcher that pre-loads our
    locally-built ``grid_router``.

    Why the launcher and not ``python route.py``: ``route.py`` calls
    ``startup_checks.run_all_checks`` at module top-level, which does
    ``sys.path.insert(0, '<KRT>/rust_router')`` and then
    ``import grid_router``. The KRT install ships a pre-built
    ``grid_router.so`` linked against a newer glibc than some Ubuntu
    20.04 hosts have, so that import fails and KRT prints "Rust router
    module not found" before any of our PYTHONPATH manipulation can
    help. ``_launcher.py`` pre-imports our cached ``grid_router`` so
    ``sys.modules`` has the working module by the time KRT's check
    runs, and ``check_rust_library``'s ``import grid_router`` short-
    circuits on the cached entry. The route.py body then runs normally.

    ``kwargs`` are forwarded to ``subprocess.run`` (timeout, capture_
    output, check, ...). Returns the ``CompletedProcess``.
    """
    route_py = _resolve_route_py()

    rust_so_dir = _ensure_grid_router_so()
    env = dict(kwargs.pop("env", os.environ.copy()))
    env["KRT_ROUTE_PY"] = str(route_py)
    env["KRT_RUST_SO_DIR"] = str(rust_so_dir)
    env["PYTHONPATH"] = (
        str(rust_so_dir) + os.pathsep + env.get("PYTHONPATH", "")
    )

    launcher = Path(__file__).parent / "_launcher.py"
    cmd = [sys.executable, str(launcher), *args]
    kwargs.setdefault("cwd", str(route_py.parent))
    return subprocess.run(cmd, env=env, **kwargs)


__all__ = ["get_krt_root", "run_route"]

