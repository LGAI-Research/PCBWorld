"""Model/backend-agnostic shared foundation (candidate_pool, slots, vec backend)."""

from __future__ import annotations

import multiprocessing as mp

# Modules imported once in the forkserver process so its children inherit them
# instead of importing them after fork. On macOS a forked (not exec'd) child can
# fail numpy's first import at random — ``RuntimeError: Could not parse python
# long as longdouble: -113 (Invalid argument)`` from getlimits' strtold — and the
# worker dies before it builds its env. Only pure-Python/BSD modules belong here:
# the forkserver must never load the GPL router or any KiCad C++ state.
FORKSERVER_PRELOAD = ("numpy",)


def _merged_forkserver_preload(extra: tuple[str, ...]) -> list[str]:
    """``extra`` added to what another caller already registered.

    The list is process-global and set by assignment, so a plain assignment
    would drop someone else's entry (the speed profiler registers its worker
    shim this way). CPython seeds the list with ``__main__``, which is not a
    caller's request: preloading it would run the entry script's module body
    inside the forkserver and break the "no GPL router, no KiCad state" rule
    above, so it is never carried over.
    """
    try:
        from multiprocessing import forkserver
        current = tuple(forkserver._forkserver._preload_modules)
    except Exception:       # private attribute moved: start from nothing
        current = ()
    keep = tuple(m for m in current if m != "__main__")
    return list(dict.fromkeys(keep + extra))


def add_forkserver_preload(*modules: str) -> None:
    """Register ``modules`` for the forkserver, keeping what is already there.

    The merging counterpart of ``mp.set_forkserver_preload`` — callers that need
    a module in every forkserver child (the speed profiler's worker shim) use
    this so they neither drop nor get dropped by ``get_mp_context``.
    """
    mp.set_forkserver_preload(_merged_forkserver_preload(tuple(modules)))


def get_mp_context(start_method: str) -> mp.context.BaseContext:
    """``mp.get_context`` that also preloads ``FORKSERVER_PRELOAD`` for forkserver.

    The preload list only takes effect before the (process-wide) forkserver
    starts; setting it again later is a harmless no-op.
    """
    ctx = mp.get_context(start_method)
    if start_method == "forkserver":
        ctx.set_forkserver_preload(_merged_forkserver_preload(FORKSERVER_PRELOAD))
    return ctx
