"""Router provenance guard: never run a router built from other C++ than this tree's.

The build stamps ``ENGINE_CPP_HASH`` (the content hash of ``engine/kicad-patches/``, from
``engine/tools/cpp_content_hash.sh``) next to ``kicad_rl_router.so``. Right before a router
is loaded — the IPC server spawn (default) or the in-process import — the same script hashes
this tree's sources and the two must agree. A mismatch means "the C++ was edited (or this tree
is on other C++ than the build) and nobody rebuilt": every result would be attributed to
sources that did not produce it, so it is an error, not a warning.

    PCBWORLD_ENGINE_ALLOW_MISMATCH=1   downgrade to a warning (deliberately running an old router)

A router built before the stamp existed carries none: it is allowed with one warning per
process (rebuild once and the guard becomes strict). The check is content-based, so a tree that
points at a build made elsewhere (a worktree using the main clone's build or a snapshot) is
judged correctly; mtimes could not do that.

Finding the build (``resolve_build_dir``): a tree with no ``build_rl`` of its own — a git
worktree — looks for a build of ITS C++ among the known places (``build_rl`` and ``var/builds/*``
of this tree and of the main clone) and takes the first whose stamp equals the source hash. Only
an exact match is taken; a stampless build is used only when no stamped one matches (legacy,
warned); none at all leaves the default path so the failure is the ordinary "no router" one,
while "builds exist but every one is of other C++" is raised with that reason by the loader
(``pcb_world.engine.ensure_router_provenance``). An explicit ``PCBWORLD_KICAD_RL_BUILD_DIR`` or
``PCBWORLD_KICAD_RL_MODULE_DIR`` is never second-guessed: it is used, and guarded.

The guard is between the router and THE TREE A PROCESS RUNS FROM, never the tree that built
the router: a run pinned to a snapshot via ``PCBWORLD_KICAD_RL_BUILD_DIR`` is still checked
against ``engine/kicad-patches/`` of the checkout it imports from. Editing that C++ under a
running snapshot therefore makes every NEW worker process refuse — by design: a new worker
also imports the edited Python, so nothing about it is the pinned run any more. Long runs run
from a worktree pinned to a commit.

Hashing is loud: a bundle without the script disables the guard with one warning per process
(an older engine bundle), a script that FAILS is a ``RouterProvenanceError`` carrying its
stderr, never a silent pass. The hash is computed once per process per engine root: the tree a
running process would see is the one it started from, and a refused guard re-raises on retry
without re-hashing.
"""
from __future__ import annotations

import glob
import os
import subprocess
import warnings

STAMP_NAME = "ENGINE_CPP_HASH"
ALLOW_ENV = "PCBWORLD_ENGINE_ALLOW_MISMATCH"
REBUILD_HINT = "rebuild it (`bash engine/build_rl_router.sh`) or point PCBWORLD_KICAD_RL_BUILD_DIR at a build of this C++"

_checked: dict[tuple[str, str], bool] = {}     # (lib_dir, engine_root) → already verified in this process
_source_hashes: dict[str, str | None] = {}     # engine_root → hash (None: bundle without the script, warned)
_main_roots: dict[str, str] = {}               # project_root → main clone root (one git call per process)


class RouterProvenanceError(RuntimeError):
    """The router about to be loaded was built from other C++ sources than this tree's."""


def source_hash(engine_root: str) -> str | None:
    """This tree's C++ content hash via the bundle's script, computed once per process per
    *engine_root*. None — with one warning — when the bundle has no script (older engine bundle:
    the guard cannot run); a script that fails or prints nothing raises RouterProvenanceError with
    its stderr (not memoised, so a fixed script is picked up on the next call)."""
    if engine_root in _source_hashes:
        return _source_hashes[engine_root]
    tool = os.path.join(engine_root, "tools", "cpp_content_hash.sh")
    if not os.path.isfile(tool):
        warnings.warn(f"engine bundle at {engine_root} has no tools/cpp_content_hash.sh: the router "
                      f"provenance guard cannot run (an older engine bundle) — update the engine checkout",
                      RuntimeWarning, stacklevel=2)
        _source_hashes[engine_root] = None
        return None
    out = subprocess.run(["bash", tool], capture_output=True, text=True)
    value = out.stdout.strip()
    if out.returncode != 0 or not value:
        raise RouterProvenanceError(
            f"{tool} failed (exit {out.returncode}): engine/kicad-patches/ cannot be hashed, so the "
            f"router cannot be verified against it\n--- script stderr ---\n{out.stderr.strip()}")
    _source_hashes[engine_root] = value
    return value


def lib_dir_of(build_dir: str) -> str:
    return os.path.join(build_dir, "pcbnew", "python", "rl")


def _read_lib_stamp(lib_dir: str) -> str | None:
    try:
        with open(os.path.join(lib_dir, STAMP_NAME), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def read_stamp(build_dir: str) -> str | None:
    return _read_lib_stamp(lib_dir_of(build_dir))


def router_cpp_hash() -> str | None:
    """The ``ENGINE_CPP_HASH`` stamp of the router this process loads (the IPC server and the
    in-process import resolve the same ``pcb_world.engine.router_lib_dir``), after the provenance
    guard has passed — the C++ identity available for the run record (in place of the C++
    constants). Under ``PCBWORLD_ENGINE_ALLOW_MISMATCH=1`` it is the loaded router's stamp, not
    this tree's hash. None only for a router built before the stamp existed (the guard warned)."""
    from pcb_world.engine import ensure_router_provenance, router_lib_dir

    ensure_router_provenance()
    return _read_lib_stamp(router_lib_dir())


def main_root(project_root: str) -> str:
    """The main clone's root when *project_root* is a git worktree (the common git dir is always
    ``<main>/.git``); *project_root* itself otherwise or when git is unavailable. One git call per
    process per root."""
    if project_root in _main_roots:
        return _main_roots[project_root]
    out = subprocess.run(["git", "-C", project_root, "rev-parse", "--git-common-dir"],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        root = project_root
    else:
        common = out.stdout.strip()
        if not os.path.isabs(common):
            common = os.path.join(project_root, common)
        root = os.path.dirname(os.path.normpath(common))
    _main_roots[project_root] = root
    return root


def candidate_build_dirs(project_root: str) -> list[str]:
    """Build dirs that hold a router, in preference order: this tree's build_rl, the main
    clone's build_rl, then every ``var/builds/*`` (snapshots and content-named builds) of both."""
    roots = [project_root]
    main = main_root(project_root)
    if os.path.realpath(main) != os.path.realpath(project_root):
        roots.append(main)
    cands = [os.path.join(r, "build_rl") for r in roots]
    for r in roots:
        cands += sorted(glob.glob(os.path.join(r, "var", "builds", "*")))
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        real = os.path.realpath(c)
        if real in seen or not glob.glob(os.path.join(lib_dir_of(c), "kicad_rl_router*")):
            continue
        seen.add(real)
        out.append(c)
    return out


def resolve_build_dir(project_root: str, engine_root: str) -> tuple[str | None, str | None]:
    """→ (build dir of THIS tree's C++, None), or (None, problem) when builds exist but none is
    usable, or (None, None) when nothing is built anywhere (the ordinary "no router" failure).

    Exact stamp match first; a stampless (pre-stamp) build only when nothing matches; a
    candidate stamped with OTHER sources is never chosen — the guard would refuse it anyway.
    With no hash script in the bundle nothing can be verified: the first candidate is used, as
    the guard then also lets any router through (``source_hash`` warned once).
    """
    present = candidate_build_dirs(project_root)
    if not present:
        return None, None
    source = source_hash(engine_root)
    if source is None:
        return present[0], None
    stamps = {c: read_stamp(c) for c in present}
    for c, stamp in stamps.items():
        if stamp == source:
            return c, None
    for c, stamp in stamps.items():
        if stamp is None:
            return c, None                       # unverified: the guard warns once about it
    listed = ", ".join(f"{os.path.relpath(c, project_root)}={stamp}" for c, stamp in stamps.items())
    return None, (f"every known build was made from other C++ (this tree hashes to {source}; "
                  f"builds: {listed}) — {REBUILD_HINT}")


def check_router_matches_sources(lib_dir: str, engine_root: str) -> None:
    """Raise RouterProvenanceError when the router in *lib_dir* was not built from the C++ under
    *engine_root* (stamp ≠ source hash); warn once for a stampless (pre-stamp) router; no-op when
    the bundle has no hash script (warned once by ``source_hash``) or no router is present.
    Verified once per process per pair."""
    key = (lib_dir, engine_root)
    if _checked.get(key):
        return
    if not os.path.isdir(lib_dir):
        return                                   # no router here: the import itself will say so
    source = source_hash(engine_root)
    if source is None:
        return                                   # older engine bundle without the script
    stamped = _read_lib_stamp(lib_dir)
    if stamped is None:
        warnings.warn(f"router at {lib_dir} carries no {STAMP_NAME} (built before the stamp existed): "
                      f"cannot verify it matches engine/kicad-patches/ — {REBUILD_HINT}",
                      RuntimeWarning, stacklevel=2)
        _checked[key] = True
        return
    if stamped != source:
        msg = (f"router at {lib_dir} was built from other C++ than this tree: engine/kicad-patches/ "
               f"hashes to {source}, the router is stamped {stamped}. {REBUILD_HINT}"
               f" (or {ALLOW_ENV}=1 to run it anyway).")
        if os.environ.get(ALLOW_ENV) == "1":
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        else:
            raise RouterProvenanceError(msg)
    _checked[key] = True
