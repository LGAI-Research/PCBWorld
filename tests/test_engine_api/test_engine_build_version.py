"""Build provenance: the loaded ``kicad_rl_router.so`` must match the source engine
version, so a stale build (source patched + bumped, but not rebuilt) is caught early
instead of silently producing wrong results in every downstream engine test.

Mechanism:
  - ``engine/kicad-patches/ENGINE_VERSION`` holds the engine version (MAJOR.MINOR,
    optionally ``+g<hash>`` — a C++ content marker), written on every version bump.
    String equality is the contract: any difference means the built
    router no longer matches the source engine identity.
  - ``engine/build_rl_router.sh`` copies that file next to the built ``.so`` as a
    stamp (a marker only — no runtime code reads it).
  - This test compares the two. A mismatch (usually source > stamp: bumped but not yet
    rebuilt) fails with a rebuild instruction.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pcb_world.engine import engine_home

# The engine is a separate repository, pinned as the engine/ submodule.
SRC_VERSION_FILE = Path(engine_home()) / "kicad-patches" / "ENGINE_VERSION"
REBUILD_HINT = "rebuild the router: `bash engine/build_rl_router.sh`"


def _read_version(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _module_dir() -> Path:
    """Directory of the actually-loaded router .so (conftest puts build_rl on the path)."""
    import kicad_rl_router

    return Path(kicad_rl_router.__file__).resolve().parent


def test_engine_build_matches_source_version() -> None:
    assert SRC_VERSION_FILE.exists(), (
        f"missing source engine-version file {SRC_VERSION_FILE} — "
        f"recreate it with the current MAJOR.MINOR."
    )
    source = _read_version(SRC_VERSION_FILE)

    stamp_file = _module_dir() / "ENGINE_VERSION"
    if not stamp_file.exists():
        pytest.fail(
            f"the built router at {stamp_file.parent} carries no ENGINE_VERSION stamp — "
            f"it predates the build-versioning mechanism (source is {source!r}). {REBUILD_HINT}."
        )
    stamp = _read_version(stamp_file)

    assert stamp == source, (
        f"engine build is stale: source ENGINE_VERSION is {source!r} but the built .so "
        f"was stamped {stamp!r}"
        + (" (source is ahead — a minor/major bump landed without a rebuild)"
           if source > stamp else " (unexpected: build is ahead of source)")
        + f". {REBUILD_HINT}."
    )
