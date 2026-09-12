"""Shared mock observation builder used by tokenizer and policy tests.

No C++ dependency — uses pure-Python dicts that match the
``build_json_observation()`` output format. Originally lived in
``tests/test_state_tokenizer.py``; extracted here so the module can be
deleted alongside the Decoder v1 removal without breaking importers
(``test_board_static_sharing.py``, ``test_decoder_only_policy.py``,
``test_state_tokenizer_batched.py``).
"""

from __future__ import annotations


def _make_pad(x, y, layer=1, width=1.0, height=1.0):
    return {
        "id": "D0",
        "center": {"id": "P0", "xy": [x, y]},
        "width": width,
        "height": height,
        "layer": layer,
    }


def _make_track(x1, y1, x2, y2, width=0.25, layer=1):
    return {
        "id": "T0",
        "p1": {"id": "P0", "xy": [x1, y1]},
        "p2": {"id": "P1", "xy": [x2, y2]},
        "width": width,
        "layer": layer,
    }


def _make_via(x, y, layer_start=1, layer_end=2, via_width=0.6):
    return {
        "id": "V0",
        "center": {"id": "P0", "xy": [x, y]},
        "layer_start": layer_start,
        "layer_end": layer_end,
        "via_width": via_width,
    }


def _make_ratsnest(x, y, layer=1):
    return {"id": "Q0", "xy": [x, y], "layer": layer}


def make_mock_obs(
    n_nets=2,
    pads_per_net=2,
    n_edges=4,
    n_arc_edges=0,
    n_tracks=0,
    n_vias=0,
    n_ratsnest_per_net=2,
    is_routing=False,
    current_layer=1,
    current_net_phase=0,
    bbox=(100.0, 50.0, 60.0, 40.0),
    copper_layers=2,
    drc_violations=None,
    step_ratio=0.0,
    steps_remaining=100,
) -> dict:
    """Build a synthetic observation matching build_json_observation() format."""
    bbox_x, bbox_y, bbox_w, bbox_h = bbox

    # Board edges (rectangle)
    edges = {}
    corners = [
        (bbox_x, bbox_y, bbox_x + bbox_w, bbox_y),
        (bbox_x + bbox_w, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h),
        (bbox_x + bbox_w, bbox_y + bbox_h, bbox_x, bbox_y + bbox_h),
        (bbox_x, bbox_y + bbox_h, bbox_x, bbox_y),
    ]
    for i in range(min(n_edges, 4)):
        x1, y1, x2, y2 = corners[i]
        edges[f"edge_{i}"] = {
            "id": f"E{i}",
            "p1": {"id": f"P{2*i}", "xy": [x1, y1]},
            "p2": {"id": f"P{2*i+1}", "xy": [x2, y2]},
            "width": 0.1,
        }
    # Arc outline entries (outline_obs="arc"): p1 -> mid -> p2, mid off-chord.
    for j in range(n_arc_edges):
        i = len(edges)
        x0 = bbox_x + 5.0 + 7.0 * j
        edges[f"edge_{i}"] = {
            "id": f"E{i}",
            "p1": {"id": f"P{2*i}", "xy": [x0, bbox_y]},
            "p2": {"id": f"P{2*i+1}", "xy": [x0 + 4.0, bbox_y]},
            "width": 0.1,
            "mid": {"id": f"PM{i}", "xy": [x0 + 2.0, bbox_y - 2.0]},
        }

    # Nets with pads
    nets = {}
    cx = bbox_x + bbox_w / 2
    cy = bbox_y + bbox_h / 2
    for ni in range(1, n_nets + 1):
        pads = {}
        for pi in range(pads_per_net):
            px = cx + (ni * 5 + pi * 3)
            py = cy + (ni * 3 + pi * 2)
            pad_layer = 1 if pi % 2 == 0 else min(2, copper_layers)
            pads[f"pad_{pi}"] = _make_pad(px, py, layer=pad_layer)
        nets[f"net_{ni}"] = {
            "net_name": f"NET{ni}",
            "pads": pads,
            "constraints": {
                "track_width": 0.25,
                "clearance": 0.2,
                "via_diameter": 0.6,
                "via_drill": 0.3,
            },
        }

    board_static = {
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "scale": max(bbox_w, bbox_h),
        "net_count": n_nets,
        "copper_layers": copper_layers,
        "boardlines": edges,
        "nets": nets,
        "obstacles": {},
        # Shape matches ``compute_hardest_per_netclass`` output
        # (strictest-per-netclass dict) for parity with production envs.
        "board_constraints": {
            "clearance_mm":     0.20,
            "track_width_mm":   0.25,
            "via_diameter_mm":  0.60,
            "via_drill_mm":     0.30,
            "uvia_diameter_mm": -1.0,
            "uvia_drill_mm":    -1.0,
        },
    }

    # Routing geometry
    routing_geometry = {}
    for ni in range(1, n_nets + 1):
        geom: dict = {}
        # Tracks
        if n_tracks > 0:
            tracks = {}
            for ti in range(n_tracks):
                tracks[f"track_{ti}"] = _make_track(
                    cx + ti, cy + ti, cx + ti + 5, cy + ti + 5,
                    layer=1,
                )
            geom["tracks"] = tracks
        # Vias
        if n_vias > 0:
            vias = {}
            for vi in range(n_vias):
                vias[f"via_{vi}"] = _make_via(cx + vi * 2, cy + vi * 2)
            geom["vias"] = vias
        # Ratsnest
        if n_ratsnest_per_net > 0:
            points = []
            for ri in range(n_ratsnest_per_net):
                points.append(_make_ratsnest(
                    cx + ni * 5 + ri * 3, cy + ni * 3 + ri * 2,
                    layer=1,
                ))
            geom["points"] = points
        if geom:
            routing_geometry[f"net_{ni}"] = geom

    # Router head
    router_head = {
        "current_xy": [cx, cy],
        "current_layer": current_layer if not (current_net_phase == 0) else -1,
        "current_net": 1 if current_net_phase > 0 else 0,
        "router_state_code": 0,
        "via_toggle": False,
        "is_routing": is_routing,
        "is_dragging": False,
        "routing_mode": 2,
        "current_net_phase": current_net_phase,
        "step": 0,
        "step_ratio": step_ratio,
        "steps_remaining": steps_remaining,
    }

    return {
        "board_static": board_static,
        "routing_geometry": routing_geometry,
        "router_head": router_head,
        "drc_violations": list(drc_violations) if drc_violations else [],
    }
