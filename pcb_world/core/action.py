"""Routing action functions and table-driven dispatcher.

Combines stateless composite action functions (net_select, start_route, etc.)
with the ActionDispatcher that maps action IDs to handler functions.

The dispatcher directly tracks ``current_net_id`` and ``routing_mode``
without a separate phase state machine — masking is handled by masking.py
based on observable engine state.

HL Action -> Low-Level engine composition:
    0: net_select    -> (conceptual, no C++ call)
    1: start_route   -> engine.start_route(x, y, layer)
    2: net_end       -> engine.build_connectivity() + ratsnest check
    3: make_line     -> engine.set_routing_mode() + engine.fix_route(x, y, force_finish=True, reject_if_stuck=True)
    4: make_via      -> engine.set_routing_mode() + engine.toggle_via() + engine.fix_route(x, y)
    5: finish        -> engine.set_routing_mode() + engine.finish(max_attempts)
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

from pcb_world.engine.kicad_engine import KiCadEngine
from pcb_world.core.masking import (
    ACT_FINISH,
    ACT_IDLE,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACTION_NAMES,
    ACTION_REGISTRY,
    NUM_ACTIONS,
)


# ---------------------------------------------------------------------------
# Stateless action functions
# ---------------------------------------------------------------------------

def net_select(engine: KiCadEngine, net_id: int) -> tuple[bool, dict[str, Any]]:
    """Select a net for routing.

    Succeeds only when ``net_id`` corresponds to an unconnected net — i.e.
    the engine's current ratsnest contains at least one edge with this
    net_code. Rejects already-routed nets (no remaining ratsnest edges)
    and net_ids not present on the board (incl. 0 / negatives).
    """
    unconnected = {int(e.net_code) for e in engine.get_ratsnest()}
    if net_id not in unconnected:
        return False, {
            "error": "net not selectable",
            "net_id": net_id,
            "reason": "no ratsnest edges for this net (already routed or invalid)",
        }
    return True, {"net_id": net_id}


def start_route(
    engine: KiCadEngine, x: float, y: float, layer: int
) -> tuple[bool, dict[str, Any]]:
    """Start routing from a pad point."""
    success = engine.start_route(x, y, layer)
    return success, {"x": x, "y": y, "layer": layer}


def net_end(
    engine: KiCadEngine, current_net_id: int
) -> tuple[bool, dict[str, Any]]:
    """Finalize the current net selection (deselect; enables net_select).

    The "net must be fully connected" precondition is NOT checked here — it
    is an availability rule owned by the masking rule (default rules require
    ``net_fully_connected``; ``env.step`` enforces the mask before dispatch
    for every caller). A masking rule that exposes net_end early therefore
    lets the agent give up / skip an unfinished net; it stays re-selectable
    since its ratsnest edges remain. ``remaining`` is reported for logging.
    """
    engine.build_connectivity()
    ratsnest = engine.get_ratsnest()
    remaining = sum(1 for e in ratsnest if e.net_code == current_net_id)
    return True, {"net_id": current_net_id, "remaining": remaining}


def make_line(
    engine: KiCadEngine, x: float, y: float, routing_mode: int = 2
) -> tuple[bool, dict[str, Any]]:
    """Route a line segment to the given point.

    Uses the engine's ``reject_if_stuck`` policy (default True): if the walkaround
    cannot reach (x, y) — the head jams against the board edge or existing copper —
    nothing is committed and success is False, instead of drawing a partial
    dangling stub. Matches ``finish``, which likewise commits only on arrival.
    Configure via ``EnvConfig.reject_if_stuck`` / ``KiCadEngine(reject_if_stuck=)``.
    """
    blocked = engine.pad_block_reason(
        x, y, item_radius_mm=engine.route_item_radius_mm(for_via=False),
        for_via=False)
    if blocked:
        return False, {"x": x, "y": y, "routing_mode": routing_mode,
                       "rejected": blocked}

    engine.set_routing_mode(routing_mode)
    # make_line must not change layer — declaring the current layer as the
    # intent lets fix_route reject an unintended layer switch (caused by
    # residual via-placement state leaking from a prior failed make_via) as
    # stuck.
    success = engine.fix_route(
        x, y, force_finish=True, expected_layer=engine.get_current_layer())
    return success, {"x": x, "y": y, "routing_mode": routing_mode}


def make_via(
    engine: KiCadEngine, x: float, y: float, routing_mode: int = 2
) -> tuple[bool, dict[str, Any]]:
    """Route a line segment to (x, y) and place a via at that point.

    All-or-nothing on BOTH sides of the router. Before: ``pad_block_reason``
    refuses a target the via provably cannot land on. During: ``require_via``
    makes ``fix_route`` verify the head still ends with a via before committing,
    so a drop cause the pre-check does not model no longer leaves a committed
    route behind an action reported as failed (v0.30; measured on d3b at 3.5%
    (Lizard) to 30.6% (ATtiny461) of make_via candidate evaluations, 73-99% of
    which had changed the board). The via_count check below is a safety net now,
    not the mechanism.
    """
    blocked = engine.pad_block_reason(
        x, y, item_radius_mm=engine.route_item_radius_mm(for_via=True),
        for_via=True)
    if blocked:
        return False, {"x": x, "y": y, "routing_mode": routing_mode,
                       "routed": False, "via_placed": False, "rejected": blocked}

    engine.set_routing_mode(routing_mode)
    vias_before = engine.get_via_count()
    engine.toggle_via()
    # Arrival tolerance = the via's own radius: a head that stopped within it
    # puts the via's copper over the requested point, which is what PNS's own
    # reached-test means. Exact-match rejected 71% of make_via calls as stuck
    # (measured on d3b), the single largest sink of the action budget.
    routed = engine.fix_route(
        x, y, force_finish=True,
        arrive_tol_mm=engine.route_item_radius_mm(for_via=True),
        require_via=True)
    via_placed = engine.get_via_count() > vias_before
    return routed and via_placed, {
        "x": x, "y": y, "routing_mode": routing_mode,
        "routed": routed, "via_placed": via_placed,
    }


def finish(
    engine: KiCadEngine, routing_mode: int = 2, max_attempts: int = 5
) -> tuple[bool, dict[str, Any]]:
    """Finish routing to the closest unconnected pad.

    Trusts ``ROUTER::Finish()``'s internal Move-convergence loop; ``max_attempts``
    bounds how many times the engine retries it. Only shove makes incremental
    progress across retries (see ``PNS_RL_ROUTER::finish``); the old wrapper-level
    ``get_routing_target`` + ``move`` retry used an inconsistent target oracle and
    is gone.
    """
    engine.set_routing_mode(routing_mode)
    success = engine.finish(max_attempts)
    return success, {"routing_mode": routing_mode, "max_attempts": max_attempts}


# ---------------------------------------------------------------------------
# Action dispatch interface
# ---------------------------------------------------------------------------

class ActionResult(NamedTuple):
    """Result of an action dispatch."""
    success: bool
    info: dict[str, Any]


@runtime_checkable
class ActionInterface(Protocol):
    """Common interface for action dispatch at any abstraction level."""

    def dispatch(
        self, engine: Any, action_id: int, params: dict[str, Any],
    ) -> ActionResult:
        ...

    @property
    def num_actions(self) -> int:
        ...

    def action_name(self, action_id: int) -> str:
        ...


# ---------------------------------------------------------------------------
# Table-driven dispatcher
# ---------------------------------------------------------------------------

class ActionDispatcher:
    """Table-driven dispatcher for the 6 routing actions.

    Directly owns ``current_net_id`` and ``routing_mode`` state.
    No PhaseTracker — masking.py derives valid actions from engine state.

    Parameter filtering is driven by ACTION_REGISTRY: only params listed
    in the ActionDef for each action are forwarded to its handler.
    """

    def __init__(self) -> None:
        self.current_net_id: int | None = None
        self.routing_mode: int = 2

        self._dispatch_table: dict[int, tuple] = {
            ACT_NET_SELECT: (self._do_net_select,),
            ACT_START_ROUTE: (self._do_start_route,),
            ACT_NET_END: (self._do_net_end,),
            ACT_MAKE_LINE: (self._do_make_line,),
            ACT_MAKE_VIA: (self._do_make_via,),
            ACT_FINISH: (self._do_finish,),
            ACT_IDLE: (self._do_idle,),
        }

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def action_name(self, action_id: int) -> str:
        return ACTION_NAMES[action_id]

    def reset(self) -> None:
        """Reset to initial state (episode boundary)."""
        self.current_net_id = None
        self.routing_mode = 2

    def dispatch(
        self, engine: Any, action_id: int, params: dict[str, Any],
    ) -> ActionResult:
        """Dispatch to the appropriate action handler.

        Only the parameters listed in ACTION_REGISTRY for this action
        are forwarded to the handler.
        """
        handler = self._dispatch_table.get(action_id)
        if handler is None:
            return ActionResult(False, {"error": f"unknown action_id {action_id}"})

        allowed = ACTION_REGISTRY[action_id].params
        filtered = {k: v for k, v in params.items() if k in allowed}

        return handler[0](engine, filtered)

    # --- Per-action handlers ---

    def _do_net_select(self, engine: Any, params: dict) -> ActionResult:
        net_id = int(params.get("net_id", 0))
        success, info = net_select(engine, net_id)
        if success:
            self.current_net_id = net_id
        return ActionResult(success, info)

    def _do_start_route(self, engine: Any, params: dict) -> ActionResult:
        x_mm = float(params.get("x_mm", 0.0))
        y_mm = float(params.get("y_mm", 0.0))
        layer = int(params.get("layer", 1))
        success, info = start_route(engine, x_mm, y_mm, layer)
        return ActionResult(success, info)

    def _do_net_end(self, engine: Any, _params: dict) -> ActionResult:
        if self.current_net_id is None:
            return ActionResult(False, {"error": "no active net"})
        success, info = net_end(engine, self.current_net_id)
        if success:
            self.current_net_id = None
        return ActionResult(success, info)

    def _do_make_line(self, engine: Any, params: dict) -> ActionResult:
        x_mm = float(params.get("x_mm", 0.0))
        y_mm = float(params.get("y_mm", 0.0))
        routing_mode = int(params.get("routing_mode", 2))
        self.routing_mode = routing_mode
        success, info = make_line(engine, x_mm, y_mm, routing_mode)
        # On fail the engine may leave the routing session dangling
        # (success ends it via force_finish=True). Drop it so the next
        # step sees is_routing()=False and start_route becomes available
        # again — mirrors the _do_finish recovery path.
        if not success and engine.is_routing():
            engine.cancel_route()
        return ActionResult(success, info)

    def _do_make_via(self, engine: Any, params: dict) -> ActionResult:
        x_mm = float(params.get("x_mm", 0.0))
        y_mm = float(params.get("y_mm", 0.0))
        routing_mode = int(params.get("routing_mode", 2))
        self.routing_mode = routing_mode
        success, info = make_via(engine, x_mm, y_mm, routing_mode)
        if not success and engine.is_routing():
            engine.cancel_route()
        return ActionResult(success, info)

    def _do_finish(self, engine: Any, params: dict) -> ActionResult:
        routing_mode = int(params.get("routing_mode", 2))
        self.routing_mode = routing_mode
        success, info = finish(engine, routing_mode)
        # FINISH must end the routing session regardless of outcome: if
        # engine.finish() exhausts its attempts without closing the route,
        # drop the active session so the next step sees is_routing()=False
        # (phase reverts to START_ROUTE).
        if engine.is_routing():
            engine.cancel_route()
        return ActionResult(success, info)

    def _do_idle(self, engine: Any, _params: dict) -> ActionResult:
        """No-op action. Does not touch engine state."""
        return ActionResult(False, {"idle": True})
