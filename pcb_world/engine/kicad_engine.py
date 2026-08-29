"""KiCadEngine: Python wrapper around kicad_rl_router.RLRouter.

This is the ONLY file that imports kicad_rl_router.
All other modules access the C++ router through this class.
"""

from __future__ import annotations

import contextlib
import gc
import math
import os
import traceback
import warnings
import weakref

from pcb_world.engine.containers import (
    BoardMeta,
    BoardSnapshot,
    CleanupResult,
    DRCResult,
    RewardSnapshot,
    RoutingSessionState,
)

from pcb_world.engine.drc import DRCUtils, DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED
from pcb_world.engine.layer_mapping import LayerMapping
from pcb_world.engine.router_client import RouterProxy, acquire_router, ipc_enabled



# KiCAD RPT_SEVERITY_* enum values (include/widgets/report_severity.h)
_SEVERITY_LABELS = {
    0x01: "undefined",
    0x02: "info",
    0x04: "exclusion",
    0x08: "action",
    0x10: "warning",
    0x20: "error",
    0x40: "ignore",
    0x80: "debug",
}


# --- Track-cleaner presets (KiCadEngine.cleanup_tracks) -----------------------
#
# Splat these into the call: engine.cleanup_tracks(dry_run=False, **PRESET).
#
# The only combination that leaves connectivity — and with it Φ, the ratsnest and
# every net's reachability — unchanged. Safe to run inside an episode, which is
# what a future net_end hook would use.
CLEANUP_TOPOLOGY_PRESERVING = {
    "merge_segments": True,
}

# Full tidy for a FINISHED board (export / scoring). The dangling passes delete
# in-progress routing (an unfinished route is dangling by definition), so this
# must never run mid-episode. remove_shorts is deliberately absent: PNS does not
# create shorts, and the pass deletes rather than reports — run it explicitly as
# a dry run when auditing an imported board instead.
CLEANUP_FINALIZE = {
    "merge_segments": True,
    "clean_vias": True,
    "tracks_in_pads": True,
    "dangling_tracks": True,
    "dangling_vias": True,
}


# --- One-live-RLRouter-per-process enforcement ---------------------------------
#
# Two live RLRouters in one process share KiCad global state (the PNS::ROUTER
# singleton, BOARD/VIA aliasing, the KIID generator): the stale one's late
# destruction nulls the live router's singleton and crashes it mid-routing.
# Enforced loudly at construction: an engine dropped
# without close() is either reclaimed HERE (before the new router exists — the
# only safe order) with a loud warning naming the leak, or, if still strongly
# referenced, construction refuses with the offender's creation stack.
#
# INTENDED coexistence (in-process multi-env: the trainer's --no-vecenv list
# mode, side-by-side comparison tests) is legitimate and must opt in EXPLICITLY
# at the construction site via ``allow_router_coexistence(reason)`` — visible in
# code, never silent. The destruction-order crash class inside such a scope is
# covered by the C++ guards (ROUTER dtor singleton check + tracer loud-abort).
_LIVE_ENGINES: "weakref.WeakSet[KiCadEngine]" = weakref.WeakSet()
_COEXISTENCE_DEPTH = 0


@contextlib.contextmanager
def allow_router_coexistence(reason: str):
    """Explicitly permit >1 live RLRouter in this process inside the scope.

    ``reason`` is required documentation at the call site (why coexistence is
    intended here); it is not interpreted. Nesting is allowed.
    """
    if not reason or not reason.strip():
        raise ValueError("allow_router_coexistence requires a non-empty reason")
    global _COEXISTENCE_DEPTH
    _COEXISTENCE_DEPTH += 1
    try:
        yield
    finally:
        _COEXISTENCE_DEPTH -= 1


def _assert_no_live_router() -> None:
    """Refuse to construct a second live RLRouter in this process.

    A close()-less engine stuck in a GC cycle is collected now — destruction
    then precedes the new router's construction, which is safe — but never
    silently: a RuntimeWarning names its creation site. An engine that is
    still strongly referenced cannot be reclaimed; that is the corruption
    scenario, so raise with the offender's creation stack.
    """
    if _COEXISTENCE_DEPTH > 0:   # explicit opt-in scope (see allow_router_coexistence)
        return
    stray = [(weakref.ref(e), e._creation_stack)
             for e in _LIVE_ENGINES if getattr(e, "_r", None) is not None]
    if not stray:
        return

    refs = [r for r, _ in stray]
    stacks = [s for _, s in stray]
    del stray  # drop strong refs so the cycle collector can actually reclaim them

    gc.collect()

    alive = [r() for r in refs]
    still = [e for e in alive if e is not None and getattr(e, "_r", None) is not None]
    if still:
        offender_stacks = "\n--- live engine created at ---\n".join(
            e._creation_stack for e in still)
        raise RuntimeError(
            f"KiCadEngine: {len(still)} live RLRouter(s) already exist in this "
            "process — one live router per process. "
            "close() the existing engine before creating a new one.\n"
            "--- live engine created at ---\n" + offender_stacks)

    warnings.warn(
        "KiCadEngine: an engine dropped without close() still held a live "
        "RLRouter and was garbage-collected just before constructing this one. "
        "Fix the leak — call close(). Leaked engine created at:\n"
        + "\n--- leaked engine created at ---\n".join(stacks),
        RuntimeWarning, stacklevel=3)


def severity_label(severity: int) -> str:
    """Convert a severity integer into a human-readable string.

    Per the KiCAD RPT_SEVERITY_* enum:
      0x10 (16) -> "warning"
      0x20 (32) -> "error"
      0x40 (64) -> "ignore"

    Args:
        severity: DRCViolation.severity integer value.

    Returns:
        One of "error" / "warning" / "ignore" / "info" / ...
        Unknown values return "unknown(N)".
    """
    return _SEVERITY_LABELS.get(severity, f"unknown({severity})")


class KiCadEngine:
    """Wrapper around the C++ RLRouter.

    Methods are grouped into three categories:

    **Actions** — mutate board or router state
        Configuration, routing, dragging, board manipulation, DRC execution,
        and file I/O.

    **Diagnostics** — evaluate routing quality
        DRC violation queries and reward snapshot (unrouted count, wirelength,
        DRC violations).

    **Queries** — read current state without side effects
        Board metadata, board element lists, routing session state, and
        high-level snapshot aggregations.

    The underlying ``kicad_rl_router`` module is imported lazily so that
    pure-Python modules can reference this class without requiring the
    C++ shared library at import time.
    """

    def __init__(
        self, board_path: str, project_path: str | None = None,
        engine_seed: int | None = 77,
        shove_iter_limit: int = 250,
        followbranch_iter_limit: int = 1_000_000,
        reject_if_stuck: bool = True,
        simplify_outline: bool = False,
        allow_default_rules: bool = False,
    ) -> None:
        """``engine_seed`` (default 77): seed KiCad's (process-global) KIID/UUID
        generator at construction for reproducible routing + UUID-keyed DRC across
        runs/processes (decided once here, never re-seeded — re-seeding mid-run would
        replay the same UUID stream and risk collisions). ``None`` = default entropy
        seeding (non-reproducible). The generator is global, so use one engine per
        process for clean determinism.

        ``shove_iter_limit`` (250) / ``followbranch_iter_limit`` (1,000,000) bound the
        PNS shove loop and ``TOPOLOGY::followBranch`` DFS. They are iteration counts,
        not wallclock timeouts, so a given board truncates at the same point on every
        run (a wallclock cutoff would land differently each time).

        ``simplify_outline`` (default False): after loading, rewrite tessellated
        micro-segment outline chains (Edge.Cuts/Margin) into native arcs/merged
        lines in the loaded board — see ``pcb_world.engine.outline_simplify``.
        Removes the PNS walkaround cluster blowup on baked-curve boards; the
        result is what routing, obs, DRC, and ``save`` all consume.

        ``allow_default_rules`` (default False): the load contract refuses a
        board whose design rules did not come from a project file (they would
        be KiCad compile-time defaults). ``True`` declares the caller will
        substitute its own rules right after construction (the env's
        ``use_yaml_drc_fallback`` path) — the pro-less load is then allowed.
        Legacy (KiCad 5 era, rules embedded in the pcb) boards are refused
        regardless.
        """
        # One live RLRouter per process — refuse (or GC-reclaim, loudly).
        # Kept in IPC mode too: the guard preserves the one-engine-at-a-time
        # contract (and the explicit coexistence opt-in) regardless of
        # transport.
        _assert_no_live_router()
        self._creation_stack = "".join(traceback.format_stack()[-12:])

        # Strict load contract: the engine opens the SOURCE file directly —
        # there is no normalize-and-cache layer in between. The source must
        # therefore be self-sufficient (verified right after the load below):
        #   * design rules must come from a project file — an explicit
        #     ``project_path`` or the ``<stem>.kicad_pro`` sibling. A board
        #     without one would silently route under KiCad DEFAULT rules,
        #     which is refused unless the caller declared it will substitute
        #     its own rules (``allow_default_rules=True`` — the env sets it
        #     for its ``use_yaml_drc_fallback`` opt-in, which then pushes the
        #     YAML rules via ``apply_default_drc_if_fallback``);
        #   * legacy boards that embed their rules in the .kicad_pcb body
        #     (KiCad 5 era) are refused unconditionally — convert once via
        #     ``pcb_world.engine.utils.load_and_save_via_engine``, whose
        #     save emits the modern pcb + .kicad_pro pair.
        board_path = str(board_path)

        # The loader always attaches a PROJECT. Resolution order:
        #   explicit project_path > <board_stem>.kicad_pro sibling >
        #   blank in-memory fallback
        # Regardless of which branch wins, the attached PROJECT is what
        # makes ``save`` always emit a valid ``.kicad_pro`` so subsequent
        # reloads carry design rules. Use ``was_project_loaded_from_file()``
        # / ``was_legacy_design_settings_loaded()`` to tell whether the
        # currently exposed design rules originated on disk or are the
        # KiCad default fallback.
        # None or a negative value → -1 (entropy seeding, non-reproducible);
        # a non-negative value seeds the KIID generator deterministically.
        seed = -1 if engine_seed is None or int(engine_seed) < 0 else (int(engine_seed) & 0x7FFFFFFF)

        if ipc_enabled():
            # GPL/NC boundary (default): the GPL shared library loads only in
            # the engine-server child; this process talks plain data over a
            # unix socket. See router_client / engine/engine_server. The
            # server opens the source file under the same strict load contract
            # (no upgrade/normalize layer); the post-load checks below run
            # here through ordinary proxy calls.
            self._r, board_path = acquire_router(
                str(board_path), project_path or "", seed,
                int(shove_iter_limit), int(followbranch_iter_limit))
        else:
            # Explicit in-process escape hatch (KICAD_ENGINE_IPC=0):
            # debugging/benchmark only — this loads the GPL .so into the
            # current process.
            import kicad_rl_router as krl
            from pcb_world.engine.utils import apply_thread_pool_cap

            # Cap the process-global KiCad thread pool (DRC/build_connectivity)
            # before the first native call — default 1, KICAD_ENGINE_THREADS to
            # widen (see ``apply_thread_pool_cap``). IPC mode: the server does
            # this at startup.
            apply_thread_pool_cap(krl)
            self._r = krl.RLRouter(str(board_path), project_path or "", seed,
                                   int(shove_iter_limit),
                                   int(followbranch_iter_limit))
        # Register immediately after the native router exists, so even an engine
        # whose remaining __init__ fails is visible to the liveness guard.
        _LIVE_ENGINES.add(self)

        # Post-load contract checks (see the strict load contract above).
        # ``close()`` before raising so the refused board does not leave a
        # live router behind for the liveness guard to reclaim later.
        if self._r.was_legacy_design_settings_loaded():
            self.close()
            raise RuntimeError(
                f"{board_path}: legacy board — its design rules are embedded "
                f"in the .kicad_pcb body (KiCad 5 era), a layout this engine "
                f"no longer consumes directly. Convert once (CLI: "
                f"tools/cad_file_patcher.py; library: pcb_world.engine.utils."
                f"load_and_save_via_engine) and use the emitted pcb+pro pair."
            )
        if not self._r.was_project_loaded_from_file() and not allow_default_rules:
            self.close()
            raise RuntimeError(
                f"{board_path}: project file was not loaded from disk, so "
                f"design rules would be KiCad defaults — add the .kicad_pro "
                f"next to the board (or pass project_path), or construct "
                f"with allow_default_rules=True if the caller substitutes "
                f"its own rules (env: use_yaml_drc_fallback)."
            )
        # Default stuck-rejection policy for fix_route (make_line). When True, a
        # route that can't reach its target is aborted rather than committing a
        # partial dangling stub. Overridable per-call via fix_route(reject_if_stuck=).
        self._reject_if_stuck = bool(reject_if_stuck)
        # Load-time outline simplification — must run before anything reads the
        # board (obs parse, routing, DRC) and before router configuration: the
        # native replace re-inits the PNS world and ROUTING_SETTINGS.
        self.outline_simplify_report = None
        if simplify_outline:
            from pcb_world.engine.outline_simplify import apply_graphics_simplify
            # IPC mode: pass the layer ids from the handshake constants —
            # letting apply_graphics_simplify default them would import
            # kicad_rl_router into this (NC) process.
            layers = None
            consts = getattr(self._r, "constants", None)
            if consts is not None:
                layers = (consts["LAYER_EDGE_CUTS"], consts["LAYER_MARGIN"])
            self.outline_simplify_report = apply_graphics_simplify(
                self._r, layers=layers)
        self._r.build_connectivity()
        self.drc_helper = DRCUtils()
        self.layer_map = LayerMapping(self._r.get_copper_layer_count())
        # Net-subset (partial routing): when set (via set_target_nets), the
        # unrouted/connectivity count is scoped to these net codes so "board
        # fully routed" means "every TARGET net connected" — see
        # get_unrouted_count. None = whole-board count.
        self._target_nets: frozenset[int] | None = None
        # Lazy net universe for per-net connectivity (ladder reward): distinct
        # pad net codes. Pads never change within an engine lifetime (reloads
        # rebuild the engine), so computed once on first use.
        self._pad_net_codes: frozenset[int] | None = None
        # Pads that bridge every copper layer (thru-hole), as (x, y, r, net).
        # Built on first use — pads are fixed for the life of the engine (reset
        # only strips copper). Read by pad_block_reason().
        self._thru_pads: list[tuple[float, float, float, int]] | None = None

    # --- Pad-adjacent placement guard ---

    def _thru_pad_geometry(self) -> list[tuple[float, float, float, int]]:
        if self._thru_pads is None:
            self._thru_pads = [
                (p.x_mm, p.y_mm, max(p.width_mm, p.height_mm) / 2.0, p.net_code)
                for p in self.get_pads()
                if p.pad_type in ("thru_hole", "np_thru_hole")
            ]
        return self._thru_pads

    def pad_block_reason(
        self, x_mm: float, y_mm: float, *, item_radius_mm: float, for_via: bool,
    ) -> str | None:
        """Why a route may not terminate at (x, y), or None when it may.

        Two placements PNS accepts without being able to honour them, both
        against pads that already bridge every copper layer:

        ``via_on_thru_pad`` (via only) — the pad IS the layer bridge, so PNS
        drops the pending via. It still commits the route, which makes make_via
        a half-executed action.

        ``pad_graze`` — the endpoint lands in the annulus just outside a
        same-net thru-hole pad, where the item's copper overlaps the pad
        without anchoring to it. KiCad's shape-overlap connectivity scores that
        as connected while its anchor-based dangling test does not, so the
        connection is a sliver that KiCad's own track cleaner deletes (measured:
        8 of 6691 thru-hole pads across 199 routed boards are held that way).

        ``item_radius_mm`` is the copper half-width of what would land there
        (via radius / half track width). The test is in mm on purpose: the
        tokenizer's normalised frame divides by a board-size-derived scale, so a
        normalised threshold would mean a different physical width per board.
        """
        net = self.get_current_net_code()
        for px, py, pr, pnet in self._thru_pad_geometry():
            d = math.hypot(x_mm - px, y_mm - py)
            if for_via and d <= pr:
                return "via_on_thru_pad"
            if pnet == net and pr < d < pr + item_radius_mm:
                return "pad_graze"
        return None

    def route_item_radius_mm(self, *, for_via: bool) -> float:
        """Copper half-width of the item the next fix_route would land.

        Netclass default for the net being routed; falls back to KiCad's own
        defaults when the class leaves the field unset (reported as -1).
        """
        nc = self.get_netclass_for_net(self.get_current_net_code())
        raw = (getattr(nc, "via_diameter_mm", -1.0) if for_via
               else getattr(nc, "track_width_mm", -1.0))
        if raw is None or raw <= 0.0:
            raw = 0.6 if for_via else 0.2
        return float(raw) / 2.0

    def set_target_nets(self, target_nets) -> None:
        """Scope the unrouted/connectivity count to ``target_nets`` (net codes).

        The C++ board is untouched (non-target nets stay physically present as
        obstacles); only :meth:`get_unrouted_count` — and therefore the reward
        completion indicator and every consumer of it — is restricted to the
        given nets. ``None`` restores the whole-board count.
        """
        self._target_nets = (
            frozenset(int(n) for n in target_nets)
            if target_nets is not None else None
        )

    def close(self) -> None:
        """Cancel any active routing/dragging session and release the C++ router.

        Dropping ``self._r`` here is required for correctness on board
        reload: the native ``RLRouter`` holds BOARD/VIA pointers that alias
        into KiCad global state, and constructing a new router while the
        old one is still alive corrupts that state (observed as segfaults
        in ``start_route``).
        """
        if self._r is not None:
            try:
                if self._r.is_routing():
                    self._r.cancel_route()
                if self._r.is_dragging():
                    self._r.cancel_drag()
            except Exception:
                # A crashed engine server already surfaced loudly at the
                # call site that hit it; close() must still tear down.
                pass
            release = getattr(self._r, "release", None)
            if release is not None:      # RouterProxy: park the server
                release()
            self._r = None

    def __del__(self) -> None:
        """Cancel any active routing/dragging before the C++ router is destroyed."""
        try:
            self.close()
        except Exception:
            pass

    # ==================================================================
    # Actions — mutate board or router state
    # ==================================================================

    # --- Low-level access ---

    def get_native(self):
        """Return the underlying C++ RLRouter for low-level access (e.g. rendering).

        In-process mode only. Under engine IPC (the default) there is no
        native object in this process — callers that genuinely need one are
        GPL-side (engine-direct tests) and must run with KICAD_ENGINE_IPC=0.
        """
        if isinstance(self._r, RouterProxy):
            raise RuntimeError(
                "get_native() is unavailable in engine-IPC mode: the native "
                "RLRouter lives in the engine-server process. Run with "
                "KICAD_ENGINE_IPC=0 for in-process native access.")
        return self._r

    def _prewarm(self, calls) -> None:
        """Seed the IPC query cache with one batched roundtrip (no-op in-process).

        ``calls`` = [(getter_name, args_tuple), ...] on the raw router — used
        by the fixed getter sequences (board snapshot / session state / reward
        snapshot) so each costs one socket roundtrip instead of N.
        """
        prewarm = getattr(self._r, "batch_prewarm", None)
        if prewarm is not None:
            prewarm(calls)

    # --- Configuration ---

    def set_routing_mode(self, mode: int) -> None:
        """Routing strategy: 0=MarkObstacles, 1=Shove, 2=Walkaround."""
        self._r.set_routing_mode(mode)

    def set_corner_mode(self, mode: int) -> None:
        """Corner mode: 0=MITERED_45 (default), 2=MITERED_90 (no diagonals)."""
        self._r.set_corner_mode(mode)

    def set_track_width(self, width_mm: float) -> None:
        """Track width in millimetres (0 = use design rules)."""
        self._r.set_track_width(width_mm)

    def set_via_diameter(self, diameter_mm: float) -> None:
        """Via outer diameter in millimetres."""
        self._r.set_via_diameter(diameter_mm)

    def set_via_drill(self, drill_mm: float) -> None:
        """Via drill diameter in millimetres."""
        self._r.set_via_drill(drill_mm)

    def reset_via_mode(self) -> None:
        self._r.reset_via_mode()

    # --- Routing ---

    def start_route(self, x_mm: float, y_mm: float, layer: int) -> bool:
        """Start routing. layer is human layer (1=Top, N=Bottom).

        Returns False (no-op) when the router is already active or the
        layer is out of range, preventing a C++ segfault.
        """
        if self.is_routing() or self.is_dragging():
            return False
        if layer < 1 or layer > self.layer_map.max_layer:
            return False
        return self._r.start_route(x_mm, y_mm, self.layer_map.human_to_board(layer))

    def move(self, x_mm: float, y_mm: float) -> None:
        self._r.move(x_mm, y_mm)

    def fix_route(
        self, x_mm: float, y_mm: float, force_finish: bool = True,
        reject_if_stuck: bool | None = None,
        expected_layer: int | None = None,
        arrive_tol_mm: float = 0.0,
        require_via: bool = False,
    ) -> bool:
        """Route to (x_mm, y_mm) and fix the result.

        ``reject_if_stuck`` aborts without committing (returns False) when the
        walkaround could not reach (x_mm, y_mm) — the head got stuck at the board
        edge or existing copper — so no partial dangling stub is drawn. ``None``
        (the default) uses the engine-level policy set at construction
        (``reject_if_stuck=``, default True); pass True/False to override per call.

        ``arrive_tol_mm`` (0 = exact coordinate match) widens the reject_if_stuck
        arrival test to a radius. Exact match is stricter than PNS's own rule —
        LINE_PLACER counts a point within head width / 2 as reached, because the
        placed item's copper covers it — so an exact compare rejects commits the
        router considers arrivals. Pass the item radius (see
        :meth:`route_item_radius_mm`) to follow that convention.

        ``require_via``: refuse to commit unless the head still ends with a via
        (make_via's all-or-nothing contract). Gated on the CALLER's intent rather
        than the router's via mode, which survives a failed make_via until the
        next episode reset — reading the mode would make a later make_line fail
        spuriously whenever that leak is live.

        ``expected_layer`` (human layer): when given, a commit whose head lands
        on any other layer is rejected as stuck — a guard against a non-via
        action (make_line) committing an unintended layer switch left over
        from residual via-placement state. ``None`` skips the check (make_via
        is exempt since a layer switch is its intended outcome).
        """
        if reject_if_stuck is None:
            reject_if_stuck = self._reject_if_stuck
        board_layer = (
            -1 if expected_layer is None
            else self.layer_map.human_to_board(expected_layer)
        )
        return self._r.fix_route(x_mm, y_mm, force_finish, reject_if_stuck,
                                 board_layer, float(arrive_tol_mm),
                                 bool(require_via))

    def cancel_route(self) -> None:
        self._r.cancel_route()

    def finish(self, max_attempts: int = 5) -> bool:
        return self._r.finish(max_attempts)

    def undo_last_segment(self) -> bool:
        return self._r.undo_last_segment()

    def flip_posture(self) -> None:
        self._r.flip_posture()

    def toggle_via(self) -> None:
        self._r.toggle_via()

    def switch_layer(self, layer: int) -> bool:
        """Switch routing layer. layer is human layer (1=Top, N=Bottom)."""
        return self._r.switch_layer(self.layer_map.human_to_board(layer))

    # --- Dragging ---

    def start_drag(
        self, x_mm: float, y_mm: float, layer: int, drag_mode: int = 0x17
    ) -> bool:
        """Start dragging. layer is human layer (1=Top, N=Bottom).

        Returns False when the router is already active or the layer
        is out of range.
        """
        if self.is_routing() or self.is_dragging():
            return False
        if layer < 1 or layer > self.layer_map.max_layer:
            return False
        return self._r.start_drag(x_mm, y_mm, self.layer_map.human_to_board(layer), drag_mode)

    def fix_drag(self, force_commit: bool = True) -> bool:
        return self._r.fix_drag(force_commit)

    def cancel_drag(self) -> None:
        self._r.cancel_drag()

    # --- Board manipulation ---

    def delete_track_by_index(self, index: int) -> bool:
        return self._r.delete_track_by_index(index)

    def delete_track_near(
        self,
        x1_mm: float,
        y1_mm: float,
        x2_mm: float,
        y2_mm: float,
        human_layer: int,
        net_code: int,
        tol_mm: float = 0.1,
    ) -> bool:
        """Delete only the segment whose layer and net both match (coordinates
        alone can misfire on a mirrored F/B route or a different net within tol)."""
        board_layer = self.layer_map.human_to_board(human_layer)
        return self._r.delete_track_near(
            x1_mm, y1_mm, x2_mm, y2_mm, board_layer, net_code, tol_mm)

    def delete_via_by_index(self, index: int) -> bool:
        return self._r.delete_via_by_index(index)

    def delete_via_near(
        self,
        x_mm: float,
        y_mm: float,
        net_code: int,
        tol_mm: float = 0.1,
    ) -> bool:
        """Delete only the via whose net matches."""
        return self._r.delete_via_near(x_mm, y_mm, net_code, tol_mm)

    def get_via_count(self) -> int:
        return self._r.get_via_count()

    def lock_net(self, net_code: int, locked: bool = True) -> int:
        """Lock/unlock a net's tracks/vias/arcs so shove treats them as
        immovable (BOARD lock flag → PNS ``MK_LOCKED``).

        A locked net's copper is walked around, never pushed, by the shove
        engine — used to fix an already-routed net while routing others
        (net-subset / staged routing). Only meaningful in Shove mode
        (``set_routing_mode(1)``); Walkaround never moves existing copper.
        Resyncs the PNS world and cancels any active routing/dragging session.
        Returns the number of items whose lock flag changed.
        """
        return self._r.lock_net(int(net_code), bool(locked))

    def delete_routing_of_nets(self, net_codes) -> int:
        """Remove tracks/vias/arcs of ``net_codes`` only; keep all other routing.

        Net-aware strip for the env's reset: only the nets being re-routed (the
        target subset) are wiped; pre-routed nets outside the set are kept
        **regardless of lock** (keeping a net's copper is independent of fixing
        it — lock only governs shove movability). Resyncs the world + ratsnest.
        Returns the number of items removed.
        """
        return self._r.delete_routing_of_nets([int(n) for n in net_codes])

    # --- Track cleanup (KiCad track cleaner, RL fork) ---

    def cleanup_tracks(
        self,
        *,
        dry_run: bool = True,
        merge_segments: bool = False,
        clean_vias: bool = False,
        remove_shorts: bool = False,
        tracks_in_pads: bool = False,
        dangling_tracks: bool = False,
        dangling_vias: bool = False,
        net_codes=None,
    ) -> CleanupResult:
        """Run KiCad's track cleaner over the board.

        QUIESCENT ONLY — the call is REJECTED (``ran=False``, ``reject_reason``)
        while a routing or drag session is open, because the cleaner mutates the
        board behind the router's back. Finish or cancel the route first.

        ``dry_run=True`` (the default) reports what would be cleaned without
        touching board geometry. A live run resyncs the PNS world + ratsnest and
        rebuilds connectivity here, so no follow-up :meth:`build_connectivity` is
        needed. It is undoable through checkpoint/restore: no pass mints a new
        UUID, so a checkpoint taken beforehand restores the pre-cleanup board.

        Pass selection (see :data:`CLEANUP_TOPOLOGY_PRESERVING` /
        :data:`CLEANUP_FINALIZE` for the two vetted combinations):

        - ``merge_segments`` — collinear merge + duplicate + zero-length tracks.
          Topology-preserving: the only pass safe to run mid-episode.
        - ``clean_vias`` — superimposed vias, and vias on an all-layer THT pad.
        - ``remove_shorts`` — segments joining two different nets.
        - ``tracks_in_pads`` — tracks fully buried inside a pad.
        - ``dangling_tracks`` / ``dangling_vias`` — items not connected at both
          ends / on fewer than two layers. **Never mid-episode**: an in-progress
          route is dangling by definition.

        ``net_codes`` limits every pass to those nets (None/empty = all nets).
        """
        result = self._r.cleanup_tracks(
            dry_run=bool(dry_run),
            merge_segments=bool(merge_segments),
            clean_vias=bool(clean_vias),
            remove_shorts=bool(remove_shorts),
            tracks_in_pads=bool(tracks_in_pads),
            dangling_tracks=bool(dangling_tracks),
            dangling_vias=bool(dangling_vias),
            net_codes=[int(n) for n in (net_codes or [])],
        )
        out = CleanupResult(
            ran=result.ran,
            reject_reason=result.reject_reason,
            items=list(result.items),
            removed=list(result.removed),
            modified=list(result.modified),
        )
        if out.changed:
            # Track geometry changed: refresh the connectivity the ratsnest and
            # connected-cluster queries read.
            self.build_connectivity()
        return out

    def build_connectivity(self) -> None:
        self._r.build_connectivity()

    # --- DRC execution ---

    def run_drc(self, rules_path: str = "") -> list:
        self.drc_helper.clear()
        violations = self._r.run_drc(rules_path)
        self.drc_helper.update(violations)
        return violations

    def run_drc_incremental(self, rules_path: str = "") -> list:
        """Same result as run_drc() but rechecks only clearance for tracks/vias
        changed since the last DRC (full fallback on first call / zoned boards)."""
        self.drc_helper.clear()
        violations = self._r.run_drc_incremental(rules_path)
        self.drc_helper.update(violations)
        return violations

    def clear_drc_cache(self) -> None:
        # Reset BOTH the C++ incremental-DRC state (m_drcViolations +
        # m_drcItemSig) AND the Python-side violation cache. Without the
        # drc_helper.clear() the Python cache keeps the previous episode's
        # violations until the next run_drc()/run_drc_incremental() overwrites
        # it — a stale read hazard on env.reset() (the sole caller).
        self._r.clear_drc_cache()
        self.drc_helper.clear()

    # --- Design rules ---

    def get_design_rules(self):
        """Snapshot BDS + NetSettings as a kicad_rl_router.DesignRules object.

        Field summary:
          - Global minima (writable): min_clearance_mm, min_track_width_mm,
            min_via_diameter_mm, min_through_hole_mm, min_via_annular_width_mm,
            min_hole_to_hole_mm, min_uvia_diameter_mm, min_uvia_drill_mm,
            copper_edge_clearance_mm.
          - Presets (read-only): track_width_presets_mm, via_presets_mm.
          - Netclasses (read-only): default_netclass (NetClassInfo) + netclasses
            (list[NetClassInfo]). NetClassInfo fields that are unset in KiCad
            are reported as -1.0.
        """
        return self._r.get_design_rules()

    def set_design_rules(self, rules) -> None:
        """Apply the global minima fields of a DesignRules object.

        Preset lists and netclass entries are ignored (read-only at this layer).
        Negative field values are treated as 'leave unchanged', so partial
        updates are easy::

            rules = engine.get_design_rules()
            rules.min_clearance_mm = 0.25
            engine.set_design_rules(rules)
        """
        self._r.set_design_rules(rules)

    def get_netclass_for_net(self, net_code: int):
        """Return the NetClassInfo KiCad assigns to ``net_code``.

        Mirrors ``BOARD::FindNet(code)->GetNetClass()`` on the C++ side:
        nets with no explicit class come back as the Default netclass.
        Unknown ``net_code`` values yield a struct whose ``name`` is empty
        — callers should treat that as "fall back to ``default_netclass``".

        Field-level inheritance still applies: non-Default classes that
        leave a field unset report it as ``-1.0``, matching
        :meth:`get_design_rules`.
        """
        return self._r.get_netclass_for_net(int(net_code))

    # --- Project / design-rule provenance ---

    def get_project_path(self) -> str:
        """Absolute path of the .kicad_pro associated with the loaded board.

        Always non-empty (auto-derived as <stem>.kicad_pro when no explicit
        path was given). The file at this path may or may not exist — use
        ``was_project_loaded_from_file()`` to tell.
        """
        return self._r.get_project_path()

    def was_project_loaded_from_file(self) -> bool:
        """True when the .kicad_pro was read from disk, False when we fell
        back to a blank in-memory project."""
        return self._r.was_project_loaded_from_file()

    def was_legacy_design_settings_loaded(self) -> bool:
        """True when the .kicad_pcb contained legacy (pre-KiCad 6) setup
        tokens that populated BDS/NetSettings during parsing."""
        return self._r.was_legacy_design_settings_loaded()

    # --- I/O ---

    def save(self, output_path: str, project_output_path: str | None = None) -> None:
        """Save the board and its companion .kicad_pro.

        ``project_output_path=None`` (default) auto-derives
        ``<output_stem>.kicad_pro`` next to the pcb; supply an explicit path
        to place the pro file elsewhere. The pro file is always written —
        without it the modern .kicad_pcb format silently drops BDS +
        NetSettings on save. GUARANTEED even when KiCad opened the source
        project read-only (its lock file held by another process / left
        stale by a crash): the C++ side clears the flag scoped to this
        save-as — the engine never writes the SOURCE project — and raises
        RuntimeError if the sidecar still could not be written.

        KiCad's project save also emits a ``.kicad_prl`` local-settings
        sidecar (GUI view state — nothing the engine or scoring reads);
        it is removed after the save unless one already existed there.
        """
        pro = project_output_path or (os.path.splitext(output_path)[0] + ".kicad_pro")
        prl = os.path.splitext(pro)[0] + ".kicad_prl"
        had_prl = os.path.exists(prl)
        self._r.save(output_path, project_output_path or "")
        if not had_prl:
            try:
                os.unlink(prl)
            except FileNotFoundError:
                pass

    # --- Checkpoint / Restore (MCTS tree search) ---

    def checkpoint(self) -> int:
        """Capture board + engine config + routing session into the C++ store
        and return an opaque integer handle.

        The handle is worker/router-local and becomes invalid once this engine
        is closed (board reload destroys the router and all its checkpoints).
        """
        return self._r.checkpoint()

    def restore(self, handle: int) -> bool:
        """Restore the state captured by ``handle``, including its DRC state.

        The checkpoint stores the DRC violations + per-track signature snapshot
        (C++ side), so they come back with the board: the Python cache is
        repopulated from the restored C++ state rather than dropped.

        Returns ``True`` if restored, ``False`` if ``handle`` is invalid (unknown /
        released / cleared by :meth:`reset_checkpoints`), in which case the board is
        left unchanged. Check the return (or :meth:`has_checkpoint` beforehand) in
        long-lived loops so a stale handle is never silently acted on.
        """
        ok = self._r.restore(handle)
        if ok:
            self.drc_helper.clear()
            self.drc_helper.update(self._r.get_drc_violations())
        return ok

    def restore_incremental(self, handle: int) -> bool:
        """Incremental restore (diff-at-restore): updates only the changed tracks
        in the PNS world. Same board result as :meth:`restore`, much faster on
        large boards. The checkpoint's DRC state is restored too.

        Returns ``True`` if restored, ``False`` if ``handle`` is invalid (see
        :meth:`restore`); the board is left unchanged on ``False``.
        """
        ok = self._r.restore_incremental(handle)
        if ok:
            self.drc_helper.clear()
            self.drc_helper.update(self._r.get_drc_violations())
        return ok

    def has_checkpoint(self, handle: int) -> bool:
        """True if ``handle`` refers to a live checkpoint.

        Validate a handle before :meth:`restore`. Reliable even across
        :meth:`reset_checkpoints` / engine instances because handles are globally
        unique (epoch+sequence) — a stale handle never aliases a live one.
        """
        return self._r.has_checkpoint(handle)

    def release_checkpoint(self, handle: int) -> None:
        """Release a checkpoint handle and free its cloned items. Idempotent."""
        self._r.release_checkpoint(handle)

    def reset_checkpoints(self) -> None:
        """Release ALL checkpoints at once (frees every clone + DRC state).

        Use at episode boundaries to bound memory: each checkpoint clones the whole
        board (~0.1 MB on medium boards, ~3.5 MB on large ones), so a long-lived
        store grows unboundedly otherwise. Re-seeds the handle epoch, so every
        handle from before the reset becomes permanently invalid (:meth:`restore`
        returns ``False`` / :meth:`has_checkpoint` returns ``False``).
        """
        self._r.reset_checkpoints()

    def checkpoint_count(self) -> int:
        """Number of live checkpoints held by the C++ router (diagnostic)."""
        return self._r.get_checkpoint_count()

    def rewind_kiid_to_episode_start(self) -> None:
        """Rewind the process-global KIID/UUID generator to the position captured at
        construction (post board-load, pre-routing).

        Called as the LAST engine step in :meth:`PCBWorld.reset` so every episode
        mints the SAME UUID stream — making per-episode routing and UUID-keyed DRC
        reproducible instead of drifting as the global generator advances across
        episodes (the generator is seeded once at construction and never re-seeded;
        see ``engine_seed``). No-op under entropy seeding (``engine_seed=None`` / < 0).
        Collision-safe the same way :meth:`restore` is: pre-load items keep their file
        UUIDs (drawn before the captured position) and the previous episode's routed
        tracks are deleted before this call, so the re-issued stream never aliases a
        live item."""
        self._r.rewind_kiid_to_episode_start()

    # ==================================================================
    # Diagnostics — evaluate routing quality
    # ==================================================================

    def get_drc_violation_count(self) -> int:
        return self.drc_helper.get_violation_count()

    def get_drc_violations(self):
        """Return cached DRC results from the last run_drc() call."""
        return self.drc_helper.get_violations()

    def get_drc_result(self) -> DRCResult:
        """Return cached DRC results as a DRCResult instance."""
        return DRCResult(
            violation_count=self.drc_helper.get_violation_count(),
            violations=self.drc_helper.get_violations(),
        )

    def get_reward_snapshot(
        self, run_drc: bool = False, rules_path: str = "",
        incremental: bool = False,
    ) -> RewardSnapshot:
        """Capture minimal state for reward computation.

        ``rules_path`` is forwarded to ``run_drc`` (a path to a
        ``.kicad_dru`` file whose rules are layered on top of the
        board's own design rules); ignored when ``run_drc`` is False.

        ``incremental``: when True, use :meth:`run_drc_incremental` instead of the
        full :meth:`run_drc` for the per-step reward DRC. Bit-exact with the full
        DRC on a forward rollout (diff is taken against the previous step's
        snapshot, which the engine maintains internally and through checkpoints),
        and much faster — this is the per_step-mode PPO fast path.
        """
        drc_count = 0
        drc_per_net: dict[str, int] = {}
        drc_err = 0
        drc_warn = 0
        drc_err_per_net: dict[str, int] = {}
        drc_warn_per_net: dict[str, int] = {}
        drc_prom = 0
        drc_prom_per_net: dict[str, int] = {}
        if run_drc:
            if incremental:
                self.run_drc_incremental(rules_path)
            else:
                self.run_drc(rules_path)
            drc_count = self.get_drc_violation_count()
            drc_per_net = self.drc_helper.get_violation_counts_by_net()
            drc_err = self.drc_helper.get_error_count()
            drc_warn = self.drc_helper.get_warning_count()
            drc_err_per_net = self.drc_helper.get_error_counts_by_net()
            drc_warn_per_net = self.drc_helper.get_warning_counts_by_net()
            drc_prom = self.drc_helper.get_count_by_severity_mode(
                DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED,
            )
            drc_prom_per_net = (
                self.drc_helper.get_counts_by_net_by_severity_mode(
                    DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED,
                )
            )

        self._prewarm([
            ("get_tracks", ()), ("get_track_count", ()), ("get_via_count", ()),
            ("get_unrouted_count", ()), ("get_ratsnest", ()),
        ])
        tracks = self.get_tracks()
        wirelength = sum(
            math.hypot(t.x2_mm - t.x1_mm, t.y2_mm - t.y1_mm) for t in tracks
        )

        # Per-net connectivity for ladder reward bonuses: a net is connected
        # iff no ratsnest edge of it remains. Universe = target nets under
        # net-subset routing, else every net with a pad (whole-board; padless
        # net codes carry no ratsnest edges, and 1-pad nets count as
        # trivially connected — a per-board constant, harmless for Φ deltas).
        if self._target_nets is not None:
            nets = self._target_nets
        else:
            if self._pad_net_codes is None:
                self._pad_net_codes = frozenset(
                    p.net_code for p in self._r.get_pads() if p.net_code > 0
                )
            nets = self._pad_net_codes
        unconnected_nets = {
            e.net_code for e in self._r.get_ratsnest()
            if e.net_code in nets
        }
        target_net_count = len(nets)
        connected_net_count = target_net_count - len(unconnected_nets)

        return RewardSnapshot(
            unrouted_count=self.get_unrouted_count(),
            track_count=self.get_track_count(),
            via_count=self.get_via_count(),
            total_wirelength=wirelength,
            drc_violation_count=drc_count,
            drc_violations_per_net=drc_per_net,
            drc_error_count=drc_err,
            drc_warning_count=drc_warn,
            drc_errors_per_net=drc_err_per_net,
            drc_warnings_per_net=drc_warn_per_net,
            drc_promoted_count=drc_prom,
            drc_promoted_per_net=drc_prom_per_net,
            connected_net_count=connected_net_count,
            target_net_count=target_net_count,
            unconnected_net_codes=frozenset(unconnected_nets),
        )

    # ==================================================================
    # Queries — read current state without side effects
    # ==================================================================

    # --- Board metadata ---

    def get_board_bbox(self):
        return self._r.get_board_bbox()

    def get_board_net_count(self) -> int:
        return self._r.get_board_net_count()

    def get_copper_layer_count(self) -> int:
        return self._r.get_copper_layer_count()

    # --- Board elements ---

    def get_tracks(self):
        return self._r.get_tracks()

    def get_vias(self):
        return self._r.get_vias()

    def get_pads(self):
        return self._r.get_pads()

    def get_keepouts(self):
        """Rule-area keepout zones as ZoneInfo objects (one per zone per copper
        layer). Each carries the outline ``pts`` (list of (x_mm, y_mm)), the
        board ``layer``, and per-item ``keepout_tracks/vias/pads`` flags."""
        return self._r.get_keepouts()

    def get_footprints(self):
        """Components as FootprintInfo objects.

        Each carries ``ref`` / ``value`` / ``fpid``, the origin ``x_mm``,
        ``y_mm``, ``orientation_deg``, ``flipped``, ``layer``, and
        ``courtyard`` — a list of closed contours, each a list of (x_mm, y_mm)
        already in board coordinates. ``courtyard`` is empty for footprints
        that declare none, which is common in older libraries.

        Static board geometry: unaffected by routing, so no liveness concern.
        """
        return self._r.get_footprints()

    def _b2h(self, board_layer: int) -> int:
        """Board layer → human layer (safe: returns -1 for unknown)."""
        try:
            return self.layer_map.board_to_human(board_layer)
        except KeyError:
            return -1

    def get_points(self) -> list[dict]:
        """Collect all key coordinates on the board.

        Collected:
          - pad centers (type="pad")
          - via centers (type="via")
          - track endpoints (type="track_start", "track_end")

        Returns:
            list of dicts, each with keys:
              x, y, type, layer (human), net_code, net_name.
        """
        points = []
        for p in self.get_pads():
            points.append({
                "x": p.x_mm, "y": p.y_mm,
                "type": "pad",
                "layer": self._b2h(p.layer),
                "net_code": p.net_code,
                "net_name": p.net_name,
            })
        for v in self.get_vias():
            points.append({
                "x": v.x_mm, "y": v.y_mm,
                "type": "via",
                "layer": self._b2h(v.top_layer),
                "net_code": v.net_code,
                "net_name": v.net_name,
            })
        for t in self.get_tracks():
            points.append({
                "x": t.x1_mm, "y": t.y1_mm,
                "type": "track_start",
                "layer": self._b2h(t.layer),
                "net_code": t.net_code,
                "net_name": t.net_name,
            })
            points.append({
                "x": t.x2_mm, "y": t.y2_mm,
                "type": "track_end",
                "layer": self._b2h(t.layer),
                "net_code": t.net_code,
                "net_name": t.net_name,
            })
        return points

    def get_ratsnest(self):
        return self._r.get_ratsnest()

    def get_pad_groups(self) -> dict[int, int]:
        """{net_code: number of distinct pad groups on that net}.

        A pad group is a connectivity cluster holding at least one pad;
        pad-free copper islands are excluded.  ``1`` means every pad on the
        net is electrically joined.  This is the quantity the routability
        metric is defined on — unlike the ratsnest count, it cannot be
        inflated by dangling copper.

        Reads COMMITTED board connectivity, so call ``build_connectivity()``
        after the last mutation first (same contract as ``get_ratsnest()``).
        """
        groups: dict[int, int] = {}
        for net_code, _pad_count in self._r.get_pad_clusters():
            groups[net_code] = groups.get(net_code, 0) + 1
        return groups

    def get_routable_nets(self) -> frozenset[int]:
        """Net codes that need routing work: >=2 pads (NPTH holes excluded),
        scoped to the target nets under net-subset routing.

        The one definition of "routable net" — the env's net-selection /
        all-nets-closed universe and the universe the size-weighted ladder
        weights (``PotentialReward.bind_board``) are resolved over; the
        offline scorer (``eval.metrics``) reads the same set so both paths
        weight the same nets. Single-pad nets carry no ratsnest edge and are
        excluded. Pads never change within an engine lifetime, but the
        target scope can, so this is not cached.
        """
        counts: dict[int, int] = {}
        for p in self._r.get_pads():
            if p.net_code > 0 and p.pad_type != "np_thru_hole":
                counts[p.net_code] = counts.get(p.net_code, 0) + 1
        return frozenset(
            nc for nc, n in counts.items()
            if n >= 2 and (self._target_nets is None or nc in self._target_nets)
        )

    def get_connected_points(
        self, x_mm: float, y_mm: float, layer: int,
    ) -> list[tuple[float, float, int]]:
        """Anchor points already electrically connected to the copper at
        ``(x_mm, y_mm, layer)`` — the whole connectivity cluster of the item
        under that point, itself included.

        Returns ``(x_mm, y_mm, human_layer)`` tuples: pad / via centres and
        track endpoints, one per copper layer the item occupies (so a thru-hole
        pad reports both faces). Empty when no copper sits at the query point.

        This is the exact "what am I already joined to?" query the candidate
        filter needs — a superset check on ratsnest coordinates only
        approximates it (ratsnest edges are an MST, so they expose one
        representative anchor per cluster).

        Reads COMMITTED board connectivity, so call ``build_connectivity()``
        after the last mutation first (same contract as ``get_ratsnest()``).
        """
        board_layer = self.layer_map.human_to_board(layer)
        out: list[tuple[float, float, int]] = []
        for p in self._r.get_connected_points(x_mm, y_mm, board_layer):
            human = self._b2h(p.layer)
            if human >= 0:  # skip non-copper / unmapped layers
                out.append((p.x_mm, p.y_mm, human))
        return out

    def get_board_outline(self):
        return self._r.get_board_outline()

    def get_board_outline_shapes(self):
        return self._r.get_board_outline_shapes()

    def get_net_names(self) -> dict[int, str]:
        names: dict[int, str] = {}
        for p in self._r.get_pads():
            if p.net_code > 0 and p.net_code not in names:
                names[p.net_code] = p.net_name
        for t in self._r.get_tracks():
            if t.net_code > 0 and t.net_code not in names:
                names[t.net_code] = t.net_name
        for v in self._r.get_vias():
            if v.net_code > 0 and v.net_code not in names:
                names[v.net_code] = v.net_name
        return names

    def get_track_count(self) -> int:
        return self._r.get_track_count()

    def get_unrouted_count(self) -> int:
        """Number of unconnected (ratsnest) connections still to route.

        Whole-board (``GetUnconnectedCount``) by default. Under net-subset
        routing (:meth:`set_target_nets`), counts only the ratsnest edges of
        the target nets, so the count is 0 exactly when every target net is
        connected — matching env's per-net ratsnest logic
        (``is_current_net_connected``) and the reward completion indicator.
        """
        if self._target_nets is None:
            return self._r.get_unrouted_count()
        return sum(
            1 for e in self._r.get_ratsnest()
            if e.net_code in self._target_nets
        )

    # --- Routing session ---

    def get_router_state_code(self) -> int:
        
        return self._r.get_router_state()

    def is_routing(self) -> bool:
        return self._r.is_routing()

    def is_dragging(self) -> bool:
        return self._r.is_dragging()

    def is_placing_via(self) -> bool:
        return self._r.is_placing_via()

    def get_current_layer(self) -> int:
        """Current routing layer as human layer (1=Top, N=Bottom). -1 if not routing."""
        board_layer = self._r.get_current_layer()
        if board_layer < 0:
            return -1
        return self.layer_map.board_to_human(board_layer)

    def get_route_head(self):
        """Route head (x_mm, y_mm, human_layer). Returns (0,0,-1) if not routing."""
        raw = self._r.get_route_head()
        board_layer = int(raw[2])
        human_layer = self.layer_map.board_to_human(board_layer) if board_layer >= 0 else -1.0
        return (raw[0], raw[1], float(human_layer))

    def get_current_net_code(self) -> int:
        return self._r.get_current_net_code()

    def get_routing_target(self):
        """Routing target (x_mm, y_mm, human_layer). Returns (0,0,-1) if unavailable.

        human_layer is **the target anchor's own layer**. 0 = the target's
        parent is multi-layer (thru — the pad_layer convention), -1 = no
        target. C++ encoding: >=0 board id / -2 thru / -1 none.
        """
        raw = self._r.get_routing_target()
        board_layer = int(raw[2])
        if board_layer >= 0:
            human_layer = self.layer_map.board_to_human(board_layer)
        elif board_layer == -2:
            human_layer = 0.0
        else:
            human_layer = -1.0
        return (raw[0], raw[1], float(human_layer))

    # --- High-level snapshots ---

    def get_board_meta(self) -> BoardMeta:
        """Extract board metadata (call once at reset)."""
        bbox = self.get_board_bbox()
        return BoardMeta(
            bbox_x=bbox.x_mm,
            bbox_y=bbox.y_mm,
            bbox_w=max(bbox.width_mm, 1e-6),
            bbox_h=max(bbox.height_mm, 1e-6),
            net_count=max(self.get_board_net_count(), 1),
            copper_layers=max(self.get_copper_layer_count(), 2),
        )

    def get_board_snapshot(self) -> BoardSnapshot:
        """Capture current board state (call every step)."""
        self._prewarm([
            ("get_tracks", ()), ("get_vias", ()), ("get_pads", ()),
            ("get_ratsnest", ()), ("get_track_count", ()),
            ("get_unrouted_count", ()),
        ])
        return BoardSnapshot(
            tracks=self.get_tracks(),
            vias=self.get_vias(),
            pads=self.get_pads(),
            ratsnest=self.get_ratsnest(),
            track_count=self.get_track_count(),
            unrouted_count=self.get_unrouted_count(),
        )

    def get_routing_session_state(self) -> RoutingSessionState:
        """Query routing session state."""
        self._prewarm([
            ("get_router_state", ()), ("is_routing", ()), ("is_dragging", ()),
            ("is_placing_via", ()), ("get_current_layer", ()),
            ("get_route_head", ()), ("get_current_net_code", ()),
            ("get_routing_target", ()),
        ])
        head_raw = self.get_route_head()
        target_raw = self.get_routing_target()

        return RoutingSessionState(
            state_code=self.get_router_state_code(),
            is_routing=self.is_routing(),
            is_dragging=self.is_dragging(),
            is_placing_via=self.is_placing_via(),
            current_layer=self.get_current_layer(),
            route_head=(head_raw[0], head_raw[1], head_raw[2]),
            current_net_code=self.get_current_net_code(),
            routing_target=(target_raw[0], target_raw[1], target_raw[2]),
        )
