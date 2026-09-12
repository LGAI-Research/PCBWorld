"""State capture utility for before/after API testing.

Captures comprehensive router state snapshots and computes diffs
between states to verify API side effects.
"""

from typing import Optional

import numpy as np


def capture_state(router) -> dict:
    """Capture comprehensive snapshot of router state.

    Returns dict with:
        track_count: int -- number of track segments
        tracks: list[dict] -- each track's (x1,y1,x2,y2,width,layer,net_code,net_name)
        pad_count: int -- number of pads
        pads: list[dict] -- each pad's (x,y,width,height,layer,net_code,net_name)
        ratsnest_count: int -- number of unrouted edges
        ratsnest: list[dict] -- each edge's (x1,y1,x2,y2,net_code)
        unrouted_count: int -- from get_unrouted_count()
        is_routing: bool
        is_dragging: bool
        router_state: int -- get_router_state()
        current_layer: int -- get_current_layer()
        is_placing_via: bool
        net_count: int -- get_net_count()
        board_net_count: int -- get_board_net_count()
    """
    tracks_raw = router.get_tracks()
    tracks = [
        {
            "x1": t.x1_mm,
            "y1": t.y1_mm,
            "x2": t.x2_mm,
            "y2": t.y2_mm,
            "width": t.width_mm,
            "layer": t.layer,
            "net_code": t.net_code,
            "net_name": t.net_name,
        }
        for t in tracks_raw
    ]

    pads_raw = router.get_pads()
    pads = [
        {
            "x": p.x_mm,
            "y": p.y_mm,
            "width": p.width_mm,
            "height": p.height_mm,
            "layer": p.layer,
            "net_code": p.net_code,
            "net_name": p.net_name,
        }
        for p in pads_raw
    ]

    ratsnest_raw = router.get_ratsnest()
    ratsnest = [
        {
            "x1": r.x1_mm,
            "y1": r.y1_mm,
            "x2": r.x2_mm,
            "y2": r.y2_mm,
            "net_code": r.net_code,
        }
        for r in ratsnest_raw
    ]

    return {
        "track_count": router.get_track_count(),
        "tracks": tracks,
        "pad_count": len(pads),
        "pads": pads,
        "ratsnest_count": len(ratsnest),
        "ratsnest": ratsnest,
        "unrouted_count": router.get_unrouted_count(),
        "is_routing": router.is_routing(),
        "is_dragging": router.is_dragging(),
        "router_state": router.get_router_state(),
        "current_layer": router.get_current_layer(),
        "is_placing_via": router.is_placing_via(),
        "net_count": router.get_net_count(),
        "board_net_count": router.get_board_net_count(),
    }


def _track_to_tuple(t: dict) -> tuple:
    return (
        t["x1"], t["y1"], t["x2"], t["y2"],
        t["width"], t["layer"], t["net_code"], t["net_name"],
    )


def _pad_to_tuple(p: dict) -> tuple:
    return (
        p["x"], p["y"], p["width"], p["height"],
        p["layer"], p["net_code"], p["net_name"],
    )


def _ratsnest_to_tuple(r: dict) -> tuple:
    return (r["x1"], r["y1"], r["x2"], r["y2"], r["net_code"])


_LIST_FIELDS = {
    "tracks": _track_to_tuple,
    "pads": _pad_to_tuple,
    "ratsnest": _ratsnest_to_tuple,
}

_COUNT_FIELDS = {
    "tracks": "track_count",
    "pads": "pad_count",
    "ratsnest": "ratsnest_count",
}


def compare_states(before: dict, after: dict) -> dict:
    """Compare two state snapshots and return differences.

    Returns dict with only the keys that changed, each containing:
        {key: {"before": val_before, "after": val_after}}

    For list fields (tracks, pads, ratsnest), reports:
        - added: list of new items
        - removed: list of removed items
        - count_delta: int

    For scalar fields, reports before/after values.
    """
    diff: dict = {}

    all_keys = set(before.keys()) | set(after.keys())

    for key in sorted(all_keys):
        val_before = before.get(key)
        val_after = after.get(key)

        if key in _LIST_FIELDS:
            to_tuple = _LIST_FIELDS[key]
            set_before = set(to_tuple(item) for item in (val_before or []))
            set_after = set(to_tuple(item) for item in (val_after or []))

            added_tuples = set_after - set_before
            removed_tuples = set_before - set_after

            if added_tuples or removed_tuples:
                # Reconstruct dicts for readability
                added = [item for item in (val_after or []) if to_tuple(item) in added_tuples]
                removed = [item for item in (val_before or []) if to_tuple(item) in removed_tuples]
                diff[key] = {
                    "added": added,
                    "removed": removed,
                    "count_delta": len(added) - len(removed),
                }
        else:
            if val_before != val_after:
                diff[key] = {"before": val_before, "after": val_after}

    return diff


def assert_state_unchanged(before: dict, after: dict, exclude: Optional[list] = None):
    """Assert that state hasn't changed, optionally excluding certain keys.

    Raises AssertionError with detailed diff if state changed unexpectedly.
    """
    diff = compare_states(before, after)

    if exclude:
        for key in exclude:
            diff.pop(key, None)
            # Also remove the paired count field if a list field is excluded
            count_key = _COUNT_FIELDS.get(key)
            if count_key:
                diff.pop(count_key, None)

    if diff:
        lines = ["State changed unexpectedly:"]
        for key, change in diff.items():
            if "added" in change or "removed" in change:
                lines.append(
                    f"  {key}: +{len(change.get('added', []))} added,"
                    f" -{len(change.get('removed', []))} removed"
                    f" (delta={change.get('count_delta', 0)})"
                )
            else:
                lines.append(f"  {key}: {change['before']!r} -> {change['after']!r}")
        raise AssertionError("\n".join(lines))


def assert_tracks_changed(before: dict, after: dict, expected_delta: Optional[int] = None):
    """Assert that track count changed, optionally by expected amount."""
    count_before = before.get("track_count", 0)
    count_after = after.get("track_count", 0)
    delta = count_after - count_before

    if delta == 0:
        raise AssertionError(
            f"Expected track count to change, but it stayed at {count_before}"
        )

    if expected_delta is not None and delta != expected_delta:
        raise AssertionError(
            f"Expected track count delta={expected_delta}, got delta={delta}"
            f" (before={count_before}, after={count_after})"
        )


def assert_routing_active(state: dict):
    """Assert router is in routing state."""
    assert state["is_routing"], f"Expected routing active, got state={state['router_state']}"


def assert_idle(state: dict):
    """Assert router is idle."""
    assert not state["is_routing"] and not state["is_dragging"], \
        f"Expected idle, got routing={state['is_routing']}, dragging={state['is_dragging']}"
