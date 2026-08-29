"""Indexed Obs (indexed_v1) — builder + lossless round-trip tests.

Pure Python (no C++ router): drives the PRODUCTION dict builders
(``BoardStatic.from_board`` + ``build_net_geometry`` +
``build_json_observation``) with duck-typed parser/snapshot views, then
asserts

  arrays_to_dict(dict_to_arrays(obs)) == obs   (byte-identical JSON)

which pins pool dedup, string-id reconstruction, key order and nesting
— the contract every cold consumer (LLM serializer, viz/debug, JSON
dumps) relies on. Also pins the two dynamic-table front-ends
(IR vs dict) to each other.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from pcb_world.core.observation import (
    BoardStatic,
    build_json_observation,
    build_net_geometry,
)
from pcb_world.core.indexed_obs import (
    OBS_FMT,
    arrays_to_dict,
    dict_to_arrays,
    dynamic_tables_from_dict,
    dynamic_tables_from_ir,
    is_indexed,
    make_empty_indexed_obs,
)
from pcb_world.engine.containers import BoardMeta, RoutingSessionState
from tests._mock_obs import make_mock_obs


# ---------------------------------------------------------------------------
# Canonical fixture: production builders on duck-typed engine views
# ---------------------------------------------------------------------------

def _edge(x1, y1, x2, y2, w=0.1):
    return NS(x1_mm=x1, y1_mm=y1, x2_mm=x2, y2_mm=y2, width_mm=w)


def _pad(x, y, net, layer=1, w=1.0, h=1.2, shape="rect"):
    return NS(x_mm=x, y_mm=y, width_mm=w, height_mm=h,
              net_code=net, layer=layer, shape=shape)


def _track(net, x1, y1, x2, y2, w=0.25, layer=1):
    return NS(net_code=net, x1_mm=x1, y1_mm=y1, x2_mm=x2, y2_mm=y2,
              width_mm=w, layer=layer)


def _via(net, x, y, top=1, bot=2, dia=0.6):
    return NS(net_code=net, x_mm=x, y_mm=y, top_layer=top,
              bottom_layer=bot, diameter_mm=dia)


def _rat(net, x1, y1, x2, y2):
    return NS(net_code=net, x1_mm=x1, y1_mm=y1, x2_mm=x2, y2_mm=y2)


def make_canonical_obs() -> tuple[dict, dict, BoardStatic]:
    """(legacy obs dict, net_geometry IR, board_info) from the REAL
    builders. Exercises: global static dedup (pad center == edge
    endpoint), per-net dynamic dedup (shared track endpoints, via on a
    track endpoint), duplicate Q ids (shared ratsnest endpoints),
    unsorted net input (build sorts), net_code<=0 filtering
    (unconnected pad + dropped track), obstacles, thru-hole sentinel
    layer=0.
    """
    meta = BoardMeta(bbox_x=10.0, bbox_y=20.0, bbox_w=60.0, bbox_h=40.0,
                     net_count=3, copper_layers=2)
    edges = [
        _edge(10.0, 20.0, 70.0, 20.0),
        _edge(70.0, 20.0, 70.0, 60.0),
        _edge(70.0, 60.0, 10.0, 60.0),
        _edge(10.0, 60.0, 10.0, 20.0),
    ]
    pads = [
        # net 2 first: nets dict keeps INSERTION (parser) order, unsorted.
        _pad(30.0, 30.0, net=2),
        # pad center == edge corner -> shares the global "P" id with edge.
        _pad(10.0, 20.0, net=2, shape="circle"),
        _pad(50.0, 30.0, net=1, layer=0, shape="oval"),  # thru sentinel
        _pad(50.0, 50.0, net=1),
        _pad(33.3, 44.4, net=0),          # net<=0 -> unconnected_pads
    ]
    obstacles = [_pad(60.0, 55.0, net=0, shape="roundrect")]
    net_names = {1: "GND", 2: "SIG_A"}
    board_constraints = {
        "clearance_mm": 0.2, "track_width_mm": 0.25,
        "via_diameter_mm": 0.6, "via_drill_mm": 0.3,
        "uvia_diameter_mm": -1.0, "uvia_drill_mm": -1.0,
    }
    board_info = BoardStatic.from_board(
        meta, pads, edges, net_names, board_constraints,
        obstacles=obstacles,
    )

    snapshot = NS(
        tracks=[
            # net 2 shares endpoint (40,40) between two tracks -> dedup.
            _track(2, 30.0, 30.0, 40.0, 40.0),
            _track(2, 40.0, 40.0, 45.0, 35.0, layer=2),
            _track(1, 50.0, 30.0, 50.0, 40.0),
            _track(0, 0.0, 0.0, 1.0, 1.0),      # net<=0 -> dropped
        ],
        vias=[
            _via(2, 40.0, 40.0),                # via ON track endpoint -> shared P
            _via(1, 50.0, 40.0),
        ],
        ratsnest=[
            # shared endpoint (50,40) across two edges -> duplicate Q id.
            _rat(1, 50.0, 40.0, 50.0, 50.0),
            _rat(1, 50.0, 40.0, 50.0, 30.0),
        ],
    )
    net_geometry = build_net_geometry(snapshot, board_info, layer_map=None)

    router_state = RoutingSessionState(
        state_code=3, is_routing=True, current_layer=1,
        route_head=(40.0, 40.0, 1.0), current_net_code=2,
    )
    obs = build_json_observation(
        snapshot, router_state, board_info,
        step_count=7, max_steps=100, current_net_id=2, routing_mode=2,
        net_geometry=net_geometry,
        drc_violations=[{
            "x_mm": 40.0, "y_mm": 40.0, "layer": 1,
            "error_type": "clearance", "type_id": 0, "severity": 32,
            "net_names": ["SIG_A", "GND"],
        }],
        action_history=[
            {
                "action_type": 3, "pointer_xy": [40.0, 40.0],
                "pointer_layer": 1, "routing_mode": 2,
                "has_pointer": True, "success": True, "net_id": 2,
            },
            {
                "action_type": 0, "pointer_xy": [0.0, 0.0],
                "pointer_layer": 0, "routing_mode": -1,
                "has_pointer": False, "success": True, "net_id": 2,
            },
        ],
        closed_nets=[3],
    )
    return obs, net_geometry, board_info


# ---------------------------------------------------------------------------
# Round-trip (the lossless contract)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_canonical_round_trip_byte_identical(self):
        obs, _, _ = make_canonical_obs()
        rt = arrays_to_dict(dict_to_arrays(obs))
        assert json.dumps(rt) == json.dumps(obs)

    def test_round_trip_preserves_aug_sidecar(self):
        obs, _, _ = make_canonical_obs()
        obs["_aug"] = {"axis_swap": True, "flip_x": -1, "flip_y": 1,
                       "nn_dx": 0.1, "nn_dy": -0.05, "slot_perm": None,
                       "directional_candidates": None}
        rt = arrays_to_dict(dict_to_arrays(obs))
        assert json.dumps(rt) == json.dumps(obs)

    def test_round_trip_empty_dynamic(self):
        obs, _, board_info = make_canonical_obs()
        obs_empty = build_json_observation(
            NS(tracks=[], vias=[], ratsnest=[]),
            RoutingSessionState(), board_info,
            net_geometry={},
        )
        rt = arrays_to_dict(dict_to_arrays(obs_empty))
        assert json.dumps(rt) == json.dumps(obs_empty)

    def test_passthrough_fields_are_same_objects(self):
        """router_head/drc/action_history/closed_nets ride by reference —
        identity round-trip, zero copy."""
        obs, _, _ = make_canonical_obs()
        iobs = dict_to_arrays(obs)
        assert iobs["router_head"] is obs["router_head"]
        assert iobs["drc_violations"] is obs["drc_violations"]
        assert iobs["action_history"] is obs["action_history"]
        assert iobs["closed_nets"] is obs["closed_nets"]

    def test_arrays_to_dict_is_identity_on_legacy(self):
        obs, _, _ = make_canonical_obs()
        assert arrays_to_dict(obs) is obs


# ---------------------------------------------------------------------------
# Table semantics
# ---------------------------------------------------------------------------

class TestTables:
    def test_format_marker(self):
        obs, _, _ = make_canonical_obs()
        iobs = dict_to_arrays(obs)
        assert is_indexed(iobs) and iobs["_fmt"] == OBS_FMT
        assert not is_indexed(obs)

    def test_static_pool_dedup_shared_pad_edge_point(self):
        """Pad center at (10,20) == edge corner -> ONE pool row, and the
        pad's center id equals the edge endpoint id in the round-trip."""
        obs, _, _ = make_canonical_obs()
        iobs = dict_to_arrays(obs)
        bs = iobs["board_static"]
        keys = {(round(float(x), 3), round(float(y), 3))
                for x, y in bs["pt_xy"]}
        assert len(keys) == len(bs["pt_xy"])  # pool rows are unique
        # dict says pad "circle" shares the corner point id:
        net2 = obs["board_static"]["nets"]["net_2"]
        corner_pad = net2["pads"]["pad_1"]
        assert corner_pad["center"]["id"] == \
            obs["board_static"]["boardlines"]["edge_0"]["p1"]["id"]

    def test_static_coords_float64(self):
        obs, _, _ = make_canonical_obs()
        iobs = dict_to_arrays(obs)
        assert iobs["board_static"]["pt_xy"].dtype == np.float64
        assert iobs["routing_geometry"]["pt_xy"].dtype == np.float64
        assert iobs["routing_geometry"]["rat_xy"].dtype == np.float64

    def test_dynamic_net_order_ascending_and_csr(self):
        obs, _, _ = make_canonical_obs()
        rg = dict_to_arrays(obs)["routing_geometry"]
        assert list(rg["net_code"]) == [1, 2]
        assert list(rg["trk_count"]) == [1, 2]
        assert list(rg["via_count"]) == [1, 1]
        assert list(rg["rat_count"]) == [4, 0]  # 2 edges x (p1, p2)

    def test_dynamic_dedup_and_duplicate_q(self):
        obs, _, _ = make_canonical_obs()
        rg = dict_to_arrays(obs)["routing_geometry"]
        # net 2 (row 1): tracks (30,30)-(40,40), (40,40)-(45,35) + via
        # (40,40) -> pool = 3 unique points, via shares row.
        assert int(rg["net_pt_count"][1]) == 3
        t0 = int(rg["trk_start"][1])
        assert int(rg["trk_pt"][t0, 1]) == int(rg["trk_pt"][t0 + 1, 0])
        assert int(rg["via_pt"][int(rg["via_start"][1])]) \
            == int(rg["trk_pt"][t0, 1])
        # net 1 rat: shared endpoint (50,40) -> same Q id at rows 0 and 2.
        q = rg["rat_q"]
        assert q[0] == q[2] and q[1] != q[0] and q[3] != q[0]

    def test_ir_and_dict_dynamic_builders_agree(self):
        obs, net_geometry, _ = make_canonical_obs()
        from_ir = dynamic_tables_from_ir(net_geometry)
        from_dict = dynamic_tables_from_dict(obs["routing_geometry"])
        assert set(from_ir) == set(from_dict)
        for k in from_ir:
            assert np.array_equal(from_ir[k], from_dict[k]), k

    def test_mock_obs_converts(self):
        """Test mocks (cosmetic ids) must convert without error; token
        content fidelity is covered by the tokenizer parity tests."""
        obs = make_mock_obs(n_nets=3, pads_per_net=2, n_tracks=2,
                            n_vias=1, n_ratsnest_per_net=2,
                            is_routing=True, current_net_phase=2)
        iobs = dict_to_arrays(obs)
        bs = iobs["board_static"]
        assert len(bs["net_code"]) == 3
        assert int(bs["net_pad_count"].sum()) == 6
        rg = iobs["routing_geometry"]
        assert int(rg["trk_count"].sum()) == 6
        assert int(rg["rat_count"].sum()) == 6

    def test_empty_indexed_obs_valid(self):
        empty = make_empty_indexed_obs()
        assert is_indexed(empty)
        legacy = arrays_to_dict(empty)
        assert legacy["board_static"]["nets"] == {}
        assert legacy["routing_geometry"] == {}
        assert legacy["closed_nets"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
