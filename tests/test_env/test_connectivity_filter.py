"""Unit tests for the connectivity candidate filter (no C++ dep).

The filter lives in ``pcb_world/vec/candidate_pool.py``, is gated by
``obs["_aug"]["connectivity_filter"]``, and drops every EXISTING-COPPER
candidate (pad / via / track endpoint) that is ALREADY connected to the route
head. The head's cluster is resolved by the wrapper through the engine and
handed over as ``obs["_aug"]["cluster_keys"]`` — a set of
``(x, y, human_layer)`` keys — so this module stays a pure obs function. These
tests inject the key set directly; the engine side that produces it is covered
by tests/test_engine_api/test_connected_points.py.

Covered here:

* off / not-routing / empty cluster → every candidate kept;
* a candidate whose (x, y, layer) is in the cluster is dropped, whatever its
  type (pad / via / track endpoint) — no type re-entry;
* the match is LAYER-AWARE: the same xy on a layer that is NOT in the cluster
  stays selectable (stacked pads that are not connected), while a thru-hole
  reports both faces and loses both;
* directional candidates are never filtered (they keep the pool non-empty);
* the dict and indexed obs paths agree tuple-for-tuple (twin equality).
"""

from pcb_world.core.indexed_obs import dict_to_arrays
from pcb_world.vec.candidate_pool import (
    CTYPE_DIRECTIONAL,
    CTYPE_PAD,
    CTYPE_TRACK,
    CTYPE_VIA,
    collect_raw_candidates,
)

NET = 1
# Four collinear pads a-b-c-d; a may be made thru-hole per test.
A = (0.0, 0.0, 1)
B = (10.0, 0.0, 1)
C = (20.0, 0.0, 1)
D = (30.0, 0.0, 1)
# Copper that is not a pad: a track endpoint and a via centre.
E = (5.0, 0.0)   # track a—e endpoint
F = (15.0, 0.0)  # via centre

_TRACKS = [(A[0], A[1], E[0], E[1], 1)]
_VIAS = [(F[0], F[1], 1, 2)]


def _full_obs(pads, ratsnest=None, *, copper_layers=2, aug=None,
              is_routing=False, tracks=(), vias=()):
    """Build a fully-formed dict obs (all fields dict_to_arrays needs).

    ``tracks`` = [(x1, y1, x2, y2, layer)], ``vias`` = [(x, y, l_start, l_end)].
    """
    net_key = f"net_{NET}"
    board_static = {
        "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 100.0, "bbox_h": 100.0,
        "scale": 100.0, "copper_layers": copper_layers, "net_count": 1,
        "boardlines": {}, "obstacles": {}, "unconnected_pads": {},
        "nets": {net_key: {"net_name": net_key, "pads": {
            f"pad_{i}": {
                "center": {"xy": [x, y]}, "width": 1.0, "height": 1.0,
                "layer": l, "shape": "circle",
            }
            for i, (x, y, l) in enumerate(pads)
        }}},
    }
    net_geom = {
        "tracks": {
            f"track_{i}": {
                "p1": {"xy": [x1, y1]}, "p2": {"xy": [x2, y2]},
                "width": 0.25, "layer": layer,
            }
            for i, (x1, y1, x2, y2, layer) in enumerate(tracks)
        },
        "vias": {
            f"via_{i}": {
                "center": {"xy": [x, y]}, "layer_start": ls, "layer_end": le,
                "via_width": 0.6,
            }
            for i, (x, y, ls, le) in enumerate(vias)
        },
        "points": [{"xy": [x, y], "layer": 1} for (x, y) in (ratsnest or [])],
    }
    obs = {
        "board_static": board_static,
        "routing_geometry": {net_key: net_geom},
        "router_head": {
            "is_routing": is_routing, "current_xy": [0.0, 0.0],
            "current_layer": 1, "current_net": NET,
        },
    }
    if aug is not None:
        obs["_aug"] = aug
    return obs


def _aug(*, on=True, start_xy=None, cluster=()):
    return {"connectivity_filter": on, "route_start_xy": start_xy,
            "cluster_keys": frozenset(cluster)}


def _key(pt, layer=1):
    return (round(pt[0], 3), round(pt[1], 3), layer)


def _xy(t):
    return (round(t[0], 3), round(t[1], 3))


def _xy_of_type(obs, ctype, extra=None):
    """Set of (x, y) among candidates of ``ctype``."""
    raw = collect_raw_candidates(obs, NET, extra)
    return {(round(x, 3), round(y, 3)) for (x, y, _l, ct) in raw if ct == ctype}


def _pad_xy(obs):
    return _xy_of_type(obs, CTYPE_PAD)


def _wired_obs(**aug_kw):
    """Pads a-b-c-d, a track a—e and a via at f, actively routing."""
    return _full_obs([A, B, C, D], ratsnest=[_xy(D)],
                     tracks=_TRACKS, vias=_VIAS,
                     aug=_aug(**aug_kw), is_routing=True)


# ---------------------------------------------------------------------------
# Off / gated
# ---------------------------------------------------------------------------

class TestFilterInactive:
    def test_no_aug_keeps_all_pads(self):
        obs = _full_obs([A, B, C, D], ratsnest=[_xy(D)])
        assert _pad_xy(obs) == {_xy(A), _xy(B), _xy(C), _xy(D)}

    def test_flag_off_keeps_all_pads(self):
        obs = _wired_obs(on=False, cluster=[_key(A), _key(B)])
        assert _pad_xy(obs) == {_xy(A), _xy(B), _xy(C), _xy(D)}

    def test_not_routing_keeps_all_pads(self):
        # The filter engages only while actively routing: outside a route there
        # is no head, so "already connected to the head" is undefined.
        obs = _full_obs([A, B, C, D], ratsnest=[_xy(D)],
                        aug=_aug(cluster=[_key(A)]), is_routing=False)
        assert _pad_xy(obs) == {_xy(A), _xy(B), _xy(C), _xy(D)}

    def test_empty_cluster_keeps_all_pads(self):
        # Engine found no copper under the head → nothing to be connected to.
        obs = _wired_obs(cluster=[])
        assert _pad_xy(obs) == {_xy(A), _xy(B), _xy(C), _xy(D)}


# ---------------------------------------------------------------------------
# Already-connected copper is dropped, whatever its candidate type
# ---------------------------------------------------------------------------

class TestConnectedDropped:
    def test_connected_pads_dropped(self):
        # a and b are wired to the head's cluster; c, d are not.
        obs = _wired_obs(cluster=[_key(A), _key(B)])
        assert _pad_xy(obs) == {_xy(C), _xy(D)}

    def test_connected_track_endpoint_dropped(self):
        obs = _wired_obs(cluster=[_key(E)])
        assert _xy_of_type(obs, CTYPE_TRACK) == set()

    def test_connected_via_dropped(self):
        # The via spans layers 1-2; only layer 1 is in the cluster, so its
        # layer-2 entry survives (a via IS one item, but the key set is what
        # the engine reported — it reports every layer of the item it found).
        obs = _wired_obs(cluster=[_key(F, 1), _key(F, 2)])
        assert _xy_of_type(obs, CTYPE_VIA) == set()

    def test_unconnected_copper_kept(self):
        obs = _wired_obs(cluster=[_key(A)])
        assert _xy_of_type(obs, CTYPE_TRACK) == {_xy(E)}
        assert _xy_of_type(obs, CTYPE_VIA) == {_xy(F)}

    def test_no_type_reentry(self):
        # Pad a and the track endpoint that lands on it share (x, y, layer):
        # dropping the pad must not let the track emitter re-admit the point.
        obs = _wired_obs(cluster=[_key(A)])
        raw = collect_raw_candidates(obs, NET, None)
        assert not [t for t in raw if _xy((t[0], t[1])) == _xy(A)]


# ---------------------------------------------------------------------------
# Layer-aware matching
# ---------------------------------------------------------------------------

class TestLayerAware:
    def test_stacked_pad_on_other_layer_survives(self):
        # Two pads at the same xy on different layers, NOT connected to each
        # other. Only the layer-1 one is in the cluster, so the layer-2 pad
        # stays selectable — an xy-only match would wipe both and could empty
        # the pool.
        stacked = [(0.0, 0.0, 1), (0.0, 0.0, 2), B]
        obs = _full_obs(stacked, ratsnest=[_xy(B)],
                        aug=_aug(cluster=[_key(A, 1)]), is_routing=True)
        raw = collect_raw_candidates(obs, NET, None)
        layers = sorted(l for (x, y, l, ct) in raw
                        if ct == CTYPE_PAD and _xy((x, y)) == _xy(A))
        assert layers == [2]

    def test_thru_hole_loses_both_faces(self):
        # A thru-hole pad is ONE item spanning both copper layers, so the
        # engine reports both faces and both go — no layer-blind matching
        # needed to block the a1→a2 self-loop.
        thru_a = (0.0, 0.0, 0)
        obs = _full_obs([thru_a, B, C, D], ratsnest=[_xy(D)], copper_layers=2,
                        aug=_aug(cluster=[_key(A, 1), _key(A, 2)]),
                        is_routing=True)
        raw = collect_raw_candidates(obs, NET, None)
        assert not [t for t in raw if _xy((t[0], t[1])) == _xy(A)]

    def test_thru_hole_face_not_in_cluster_survives(self):
        thru_a = (0.0, 0.0, 0)
        obs = _full_obs([thru_a, B, C, D], ratsnest=[_xy(D)], copper_layers=2,
                        aug=_aug(cluster=[_key(A, 1)]), is_routing=True)
        raw = collect_raw_candidates(obs, NET, None)
        layers = sorted(l for (x, y, l, ct) in raw
                        if ct == CTYPE_PAD and _xy((x, y)) == _xy(A))
        assert layers == [2]


# ---------------------------------------------------------------------------
# Directional candidates are exempt
# ---------------------------------------------------------------------------

class TestDirectionalNeverFiltered:
    def test_extras_survive_even_when_in_cluster(self):
        # Generated geometry, not existing copper. Exempting them is what keeps
        # the pool non-empty when everything nearby is already connected.
        # (5.0, 0.0) deliberately coincides with track endpoint e, which the
        # cluster drops: the directional entry then claims that dedup key and
        # the point comes back as CTYPE_DIRECTIONAL. Both extras survive.
        extra = [(5.0, 0.0, 1, CTYPE_DIRECTIONAL), (5.5, 0.0, 1, CTYPE_DIRECTIONAL)]
        obs = _wired_obs(cluster=[_key(A), _key(B), _key(C), _key(D),
                                  _key(E), _key(F, 1), _key(F, 2)])
        assert _xy_of_type(obs, CTYPE_DIRECTIONAL, extra) == {(5.0, 0.0), (5.5, 0.0)}
        # Everything that came from existing copper is gone.
        assert not [
            t for t in collect_raw_candidates(obs, NET, extra)
            if t[3] != CTYPE_DIRECTIONAL
        ]


# ---------------------------------------------------------------------------
# Twin equality — dict and indexed obs paths agree tuple-for-tuple
# ---------------------------------------------------------------------------

class TestTwinEquality:
    def _both(self, obs):
        dict_raw = collect_raw_candidates(obs, NET, None)
        idx_raw = collect_raw_candidates(dict_to_arrays(obs), NET, None)
        return dict_raw, idx_raw

    def test_filtered(self):
        d, i = self._both(_wired_obs(cluster=[_key(A), _key(E), _key(F, 1)]))
        assert d == i

    def test_thru_hole(self):
        obs = _full_obs([(0.0, 0.0, 0), B, C, D], ratsnest=[_xy(D)],
                        copper_layers=2, tracks=_TRACKS, vias=_VIAS,
                        aug=_aug(cluster=[_key(A, 1), _key(A, 2)]),
                        is_routing=True)
        d, i = self._both(obs)
        assert d == i

    def test_filter_off(self):
        d, i = self._both(_wired_obs(on=False))
        assert d == i
