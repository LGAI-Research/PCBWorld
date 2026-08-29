"""Utilities for reading / normalizing KiCad PCB files.

Extracted from ``pcb_file_parser`` so external scripts (e.g. the datagen
converters) can reuse the engine-based normalization without pulling in
the rest of the engine package.

Public surface:
    - ``load_and_save_via_engine(load_path, output_path)`` — raw round-trip
      primitive: load any ``.kicad_pcb`` (legacy versions included) with the
      native router and save it through the current build. Raises on failure.
      This is the ONE-SHOT upgrade tool for legacy boards: the save emits a
      modern pcb + companion ``.kicad_pro`` pair, which is what
      ``KiCadEngine`` requires (it loads sources directly and refuses
      legacy/pro-less files — see its load contract).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

__all__ = [
    "apply_thread_pool_cap",
    "load_and_save_via_engine",
]


# ---------------------------------------------------------------------------
# KiCad thread pool cap
# ---------------------------------------------------------------------------

# Once-per-process guard: the pool is process-global and each reset
# destroys/respawns its threads, so re-applying on every engine
# construction (board reloads recreate the engine) would be pure churn.
_THREAD_POOL_CAP_APPLIED = False


def apply_thread_pool_cap(krl) -> None:
    """Cap the process-global KiCad thread pool (DRC / build_connectivity).

    The native pool defaults to ``hardware_concurrency()`` threads *per
    process*; with N parallel env workers that means N × nproc threads
    fighting over the same cores. Policy via ``KICAD_ENGINE_THREADS``:

        unset / "1"   → 1 thread (default: each worker stays single-threaded)
        "<int>"       → that many threads
        "physical"    → the host's physical core count

    Called with the imported ``kicad_rl_router`` module, once per process,
    before the first ``RLRouter`` construction. No-op (with a warning) on
    router builds that predate ``set_thread_pool_size``.

    Why a pybind API instead of KiCad's ``kicad_advanced`` config file
    (``MaximumThreads`` + ``KICAD_CONFIG_HOME``): headless bindings never
    read that file — ``ADVANCED_CFG::loadFromConfigFile()`` returns early
    when ``!wxTheApp``, so the config route is silently a no-op here.
    """
    global _THREAD_POOL_CAP_APPLIED
    if _THREAD_POOL_CAP_APPLIED:
        return
    _THREAD_POOL_CAP_APPLIED = True

    if not hasattr(krl, "set_thread_pool_size"):
        warnings.warn(
            "kicad_rl_router build predates set_thread_pool_size(); KiCad "
            "thread pool left at hardware_concurrency(). Rebuild via "
            "engine/build_rl_router.sh to enable the cap.",
            RuntimeWarning,
        )
        return

    raw = os.environ.get("KICAD_ENGINE_THREADS", "1").strip().lower()
    if raw == "physical":
        import psutil

        n = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    else:
        try:
            n = int(raw)
        except ValueError:
            raise ValueError(
                f"KICAD_ENGINE_THREADS must be a positive integer or "
                f"'physical', got {raw!r}"
            ) from None
        if n < 1:
            raise ValueError(f"KICAD_ENGINE_THREADS must be >= 1, got {n}")
    krl.set_thread_pool_size(n)


# ---------------------------------------------------------------------------
# Core primitive
# ---------------------------------------------------------------------------

def _drain(count_fn, delete_fn, *, desc: str, verbose: bool) -> None:
    """Delete-by-index in a loop, optionally rendering a tqdm progress bar.

    ``count_fn`` returns the current element count (queried once upfront
    for the total, then each iteration to terminate the loop). ``delete_fn``
    takes an index and removes that element; we always pass ``0`` so the
    router collapses its list from the front. ``leave=False`` is used for
    the tqdm bar so it disappears cleanly when nested under an outer
    progress bar (e.g. ``cad_file_patcher``).
    """
    total = count_fn()
    if total == 0:
        return
    if verbose:
        try:
            from tqdm import tqdm
        except ImportError:
            verbose = False
    if verbose:
        with tqdm(total=total, desc=desc, unit="ea",
                  leave=False) as pbar:
            while count_fn() > 0:
                delete_fn(0)
                pbar.update(1)
    else:
        while count_fn() > 0:
            delete_fn(0)


def load_and_save_via_engine(load_path: str | Path,
                             output_path: str | Path,
                             *,
                             unroute: bool = False,
                             simplify_outline: bool = False,
                             verbose: bool = False) -> None:
    """Round-trip a ``.kicad_pcb`` through a throwaway ``RLRouter``.

    Loads ``load_path`` with the C++ router (whose loader handles any
    historical KiCad file version) and immediately writes the in-memory
    board out to ``output_path`` via the same build's saver. The result
    is guaranteed readable by any router instance produced by the same
    build — making this a drop-in normalizer for format / version drift.

    ``unroute=True`` deletes every existing track and via (matching
    ``PCBWorld.reset()``) before saving, producing a clean "bare board"
    snapshot suitable for routing from scratch.

    ``simplify_outline=True`` runs the load-time outline-simplify pass
    (:func:`pcb_world.engine.outline_simplify.apply_graphics_simplify`)
    before saving, so the output file carries native arcs instead of
    tessellated micro-segment chains. The router is seeded in this mode so
    the freshly minted arc/line UUIDs — and therefore the output bytes —
    are deterministic per input.

    ``verbose=True`` renders a tqdm progress bar for each delete loop
    (tracks, vias) while ``unroute`` is in effect so callers can confirm
    the router is actually draining its element count. No-op when
    ``unroute=False`` or tqdm is not installed.

    Uses ``krl.RLRouter`` directly rather than the ``KiCadEngine`` wrapper:
    the wrapper's load contract refuses exactly the legacy/pro-less files
    this primitive exists to convert.

    Raises whatever the native router raises on load/save failure;
    callers that want a "best-effort" fallback should wrap this in their
    own try/except.
    """
    # Lazy import: keeps this module importable in pure-Python contexts
    # (e.g. lightweight analysis scripts) that never need the round-trip.
    import kicad_rl_router as krl

    apply_thread_pool_cap(krl)

    r = None
    try:
        if simplify_outline:
            r = krl.RLRouter(str(load_path), "", 77)  # seeded → deterministic UUIDs
            from pcb_world.engine.outline_simplify import apply_graphics_simplify
            apply_graphics_simplify(r)
        else:
            r = krl.RLRouter(str(load_path))
        r.build_connectivity()
        if unroute:
            # Mirrors the track/via purge in ``PCBWorld.reset`` so this
            # stays aligned with what the env considers a fresh episode.
            if r.is_routing():
                r.cancel_route()
            if r.is_dragging():
                r.cancel_drag()
            _drain(r.get_track_count, r.delete_track_by_index,
                   desc="tracks", verbose=verbose)
            _drain(r.get_via_count, r.delete_via_by_index,
                   desc="vias", verbose=verbose)
            r.build_connectivity()
        # The native project save also emits a ``.kicad_prl`` local-settings
        # sidecar (GUI view state — nothing downstream reads it); drop it
        # unless one already existed at the path (same policy as
        # ``KiCadEngine.save``).
        prl = Path(output_path).with_suffix(".kicad_prl")
        had_prl = prl.exists()
        r.save(str(output_path))
        if not had_prl:
            try:
                prl.unlink()
            except FileNotFoundError:
                pass
    finally:
        # Drop the native BOARD/VIA pointer before the next RLRouter
        # construction — see ``KiCadEngine.close`` for the rationale.
        # Rebinding ``r = None`` inside finally matters when this function
        # exits via exception: the propagating traceback pins this frame's
        # locals, so without the explicit rebind the RLRouter would stay
        # alive until the caller's except block releases the traceback.
        if r is not None:
            try:
                if r.is_routing():
                    r.cancel_route()
                if r.is_dragging():
                    r.cancel_drag()
            except Exception:  # noqa: BLE001
                pass
            r = None
