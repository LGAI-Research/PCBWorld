"""Unit tests for candidate_builder (no C++ dependency).

Tests:
1. build_candidates_mlp: basic extraction, dedup, sorting, one-hot encoding
2. build_directional_candidates: 8-dir generation + directional modes
   (multi_resolution preset / parse_directional_mode)
3. Via multi-layer expansion
4. extra_candidates (directional) merge
5. Sort priority: PAD > others (ratsnest excluded from pointer pool)
"""

import math

import numpy as np
import pytest

from pcb_world.vec.candidate_pool import (
    CAND_FEATURES,
    CTYPE_DIRECTIONAL,
    CTYPE_PAD,
    CTYPE_RATSNEST,
    CTYPE_TRACK,
    CTYPE_VIA,
    DIRECTIONAL_DISTANCE_PRESETS,
    MAX_CANDIDATES,
    NUM_CAND_TYPES,
    build_candidates_mlp,
    build_directional_candidates,
    parse_directional_mode,
)


# ---------------------------------------------------------------------------
# Helpers — minimal obs dict builders
# ---------------------------------------------------------------------------

def _make_obs(
    pads=None,
    tracks=None,
    ratsnest=None,
    vias=None,
    bbox_x=0.0,
    bbox_y=0.0,
    scale=100.0,
    net_id=1,
):
    """Build a minimal obs dict for testing."""
    net_key = f"net_{net_id}"

    board_static = {
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "scale": scale,
        "nets": {},
    }
    if pads is not None:
        board_static["nets"][net_key] = {
            "pads": {f"pad_{i}": p for i, p in enumerate(pads)},
        }

    routing_geometry = {}
    net_geom = {}
    if tracks is not None:
        net_geom["tracks"] = {f"track_{i}": t for i, t in enumerate(tracks)}
    if ratsnest is not None:
        net_geom["points"] = ratsnest
    if vias is not None:
        net_geom["vias"] = {f"via_{i}": v for i, v in enumerate(vias)}
    if net_geom:
        routing_geometry[net_key] = net_geom

    return {
        "board_static": board_static,
        "routing_geometry": routing_geometry,
    }


def _pad(x, y, layer=1):
    return {"center": {"xy": [x, y]}, "layer": layer}


def _track(x1, y1, x2, y2, layer=1):
    return {
        "p1": {"xy": [x1, y1]},
        "p2": {"xy": [x2, y2]},
        "layer": layer,
    }


def _ratsnest(x, y, layer=1):
    return {"xy": [x, y], "layer": layer}


def _via(x, y, layer_start=1, layer_end=2):
    return {
        "center": {"xy": [x, y]},
        "layer_start": layer_start,
        "layer_end": layer_end,
    }


# ---------------------------------------------------------------------------
# 1. Feature shape & one-hot encoding
# ---------------------------------------------------------------------------

class TestFeatureShape:
    def test_cand_features_dim(self):
        assert CAND_FEATURES == 3 + NUM_CAND_TYPES  # 8

    def test_output_shapes(self):
        obs = _make_obs(pads=[_pad(10, 20)])
        feat, mask, mm = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(10, 20))
        assert feat.shape == (MAX_CANDIDATES, CAND_FEATURES)
        assert mask.shape == (MAX_CANDIDATES,)
        assert len(mm) == 1

    def test_one_hot_pad(self):
        obs = _make_obs(pads=[_pad(10, 20)])
        feat, _, _ = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        # one-hot at index 3 + CTYPE_PAD = 3
        assert feat[0, 3 + CTYPE_PAD] == 1.0
        assert feat[0, 3 + CTYPE_TRACK] == 0.0
        assert feat[0, 3 + CTYPE_RATSNEST] == 0.0
        assert feat[0, 3 + CTYPE_VIA] == 0.0
        assert feat[0, 3 + CTYPE_DIRECTIONAL] == 0.0

    def test_one_hot_track(self):
        obs = _make_obs(tracks=[_track(5, 5, 15, 15)])
        feat, _, _ = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        assert feat[0, 3 + CTYPE_TRACK] == 1.0
        # other type bits should be 0
        for t in [CTYPE_PAD, CTYPE_RATSNEST, CTYPE_VIA, CTYPE_DIRECTIONAL]:
            assert feat[0, 3 + t] == 0.0

    def test_ratsnest_excluded_from_pool(self):
        # Ratsnest is a context-only state token: it is NOT registered as
        # a selectable pointer candidate (its upstream layer is unreliable).
        obs = _make_obs(ratsnest=[_ratsnest(10, 10)])
        _, mask, mm = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=(0, 0),
        )
        assert mask.sum() == 0
        assert mm == []

    def test_one_hot_directional(self):
        obs = _make_obs()
        dir_cands = build_directional_candidates((50, 50), current_layer=1)
        feat, _, _ = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=(50, 50), extra_candidates=dir_cands,
        )
        assert feat[0, 3 + CTYPE_DIRECTIONAL] == 1.0


# ---------------------------------------------------------------------------
# 2. build_directional_candidates
# ---------------------------------------------------------------------------

class TestDirectionalCandidates:
    def test_count(self):
        cands = build_directional_candidates((50, 50), current_layer=1)
        assert len(cands) == 8  # 8 dirs × 1 distance (0.5mm)

    def test_all_have_correct_type(self):
        cands = build_directional_candidates((50, 50), current_layer=1)
        for _, _, _, ctype in cands:
            assert ctype == CTYPE_DIRECTIONAL

    def test_layer_passed_through(self):
        cands = build_directional_candidates((50, 50), current_layer=2)
        for _, _, layer, _ in cands:
            assert layer == 2

    def test_distances(self):
        hx, hy = 50.0, 50.0
        cands = build_directional_candidates((hx, hy), current_layer=1)
        dists = sorted(set(
            round(math.sqrt((x - hx) ** 2 + (y - hy) ** 2), 3)
            for x, y, _, _ in cands
        ))
        # 0.5mm: orthogonal -> 0.5, diagonal -> ~0.707
        assert len(dists) == 2
        assert pytest.approx(dists[0], abs=0.01) == 0.5
        assert pytest.approx(dists[1], abs=0.01) == 0.5 * math.sqrt(2)

    def test_8_directions_per_distance(self):
        hx, hy = 10.0, 10.0
        cands = build_directional_candidates((hx, hy), current_layer=1)
        # Group by distance bucket (close vs far)
        close = [(x, y) for x, y, _, _ in cands if abs(x - hx) <= 1 and abs(y - hy) <= 1]
        assert len(close) == 8  # all 8 directions at 0.5mm


# ---------------------------------------------------------------------------
# 2b. Directional modes (directional_candidates knob)
# ---------------------------------------------------------------------------

class TestDirectionalModes:
    def test_multi_resolution_ladder(self):
        hx, hy = 50.0, 50.0
        cands = build_directional_candidates(
            (hx, hy), current_layer=1, mode="multi_resolution",
        )
        assert len(cands) == 8 * 4  # 8 dirs × [0.2, 1.0, 5.0, 25.0]
        # Axis-aligned candidates sit exactly at each ladder distance.
        axis_dists = sorted(set(
            round(math.sqrt((x - hx) ** 2 + (y - hy) ** 2), 6)
            for x, y, _, _ in cands
            if math.isclose(x, hx, abs_tol=1e-9) or math.isclose(y, hy, abs_tol=1e-9)
        ))
        assert axis_dists == [0.2, 1.0, 5.0, 25.0]
        for _, _, layer, ctype in cands:
            assert layer == 1
            assert ctype == CTYPE_DIRECTIONAL

    def test_multi_resolution_distance_major_order(self):
        # Emission is distance-major: all 8 dirs at distances[0] first.
        preset = DIRECTIONAL_DISTANCE_PRESETS["multi_resolution"]
        cands = build_directional_candidates(
            (0.0, 0.0), current_layer=1, mode="multi_resolution",
        )
        for i, (x, y, _, _) in enumerate(cands):
            expected = preset[i // 8]
            assert max(abs(x), abs(y)) == pytest.approx(expected)

    def test_multi_resolution_no25_drops_only_the_25mm_rung(self):
        hx, hy = 50.0, 50.0
        cands = build_directional_candidates(
            (hx, hy), current_layer=1, mode="multi_resolution_no25",
        )
        assert len(cands) == 8 * 3  # 8 dirs × [0.2, 1.0, 5.0]
        axis_dists = sorted(set(
            round(math.sqrt((x - hx) ** 2 + (y - hy) ** 2), 6)
            for x, y, _, _ in cands
            if math.isclose(x, hx, abs_tol=1e-9) or math.isclose(y, hy, abs_tol=1e-9)
        ))
        assert axis_dists == [0.2, 1.0, 5.0]
        # Prefix property: identical to multi_resolution below 25mm.
        full = build_directional_candidates(
            (hx, hy), current_layer=1, mode="multi_resolution",
        )
        assert cands == full[: len(cands)]

    def test_mres8_ladder(self):
        # Log-scale 1-2-5 ladder, every rung a multiple of the 0.2 mm
        # generation grid; 8 dirs x 8 rungs = 64 candidates.
        ladder = DIRECTIONAL_DISTANCE_PRESETS["mres8"]
        assert ladder == (0.2, 0.4, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
        assert parse_directional_mode("mres8") == (None, ladder)
        hx, hy = 50.0, 50.0
        cands = build_directional_candidates(
            (hx, hy), current_layer=1, mode="mres8",
        )
        assert len(cands) == 8 * 8
        for i, (x, y, layer, ctype) in enumerate(cands):
            assert layer == 1 and ctype == CTYPE_DIRECTIONAL
            # distance-major emission: all 8 dirs at ladder[0] first
            assert max(abs(x - hx), abs(y - hy)) == pytest.approx(ladder[i // 8])
            # diagonals are (d, d) offsets, so each axis offset is on the grid
            for off in (x - hx, y - hy):
                assert abs(off / 0.2 - round(off / 0.2)) < 1e-6

    def test_mode_none_is_default_ring(self):
        assert build_directional_candidates((10.0, 20.0), 1, mode=None) == (
            build_directional_candidates((10.0, 20.0), 1)
        )

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            build_directional_candidates((0.0, 0.0), 1, mode="warp_speed")

    def test_parse_directional_mode(self):
        assert parse_directional_mode(None) == (None, None)
        assert parse_directional_mode("multi_resolution") == (
            None, DIRECTIONAL_DISTANCE_PRESETS["multi_resolution"],
        )
        assert parse_directional_mode("grid200") == (200, None)
        with pytest.raises(ValueError):
            parse_directional_mode("grid42")  # not in _GRID_STEP_CELLS
        with pytest.raises(ValueError):
            parse_directional_mode("gridfoo")


# ---------------------------------------------------------------------------
# 3. Via multi-layer expansion
# ---------------------------------------------------------------------------

class TestViaMultiLayer:
    def test_via_expands_to_all_layers(self):
        obs = _make_obs(vias=[_via(30, 30, layer_start=1, layer_end=4)])
        feat, mask, mm = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(30, 30))
        # Should produce 4 candidates (layers 1, 2, 3, 4)
        assert mask.sum() == 4
        layers = [l for _, _, l in mm]
        assert sorted(layers) == [1, 2, 3, 4]

    def test_via_dedup_same_layer(self):
        # Two vias at same position, overlapping layers
        obs = _make_obs(vias=[
            _via(30, 30, layer_start=1, layer_end=2),
            _via(30, 30, layer_start=2, layer_end=3),
        ])
        feat, mask, mm = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(30, 30))
        # layers 1, 2, 3 — layer 2 deduped
        assert mask.sum() == 3

    def test_via_one_hot(self):
        obs = _make_obs(vias=[_via(30, 30, layer_start=1, layer_end=1)])
        feat, _, _ = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(30, 30))
        assert feat[0, 3 + CTYPE_VIA] == 1.0


# ---------------------------------------------------------------------------
# 4. Sort priority: PAD > others (ratsnest is NOT a pointer candidate)
# ---------------------------------------------------------------------------

class TestSortPriority:
    def test_ratsnest_does_not_enter_pool(self):
        obs = _make_obs(
            pads=[_pad(10, 10)],
            ratsnest=[_ratsnest(90, 90)],
        )
        feat, mask, mm = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=(10, 10),
        )
        # Only the pad should be present. Ratsnest is context-only.
        assert mask.sum() == 1
        assert feat[0, 3 + CTYPE_PAD] == 1.0
        assert feat[0, 3 + CTYPE_RATSNEST] == 0.0

    def test_pad_before_track(self):
        obs = _make_obs(
            pads=[_pad(90, 90)],
            tracks=[_track(10, 10, 11, 11)],
        )
        # route head at (10,10) — track is closer, but pad should come first
        feat, mask, mm = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(10, 10))
        assert feat[0, 3 + CTYPE_PAD] == 1.0
        assert feat[1, 3 + CTYPE_TRACK] == 1.0

    def test_within_same_priority_sorted_by_distance(self):
        obs = _make_obs(
            pads=[_pad(50, 50), _pad(10, 10)],
        )
        feat, mask, mm = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        # pad at (10,10) is closer to (0,0) → should come first
        assert mm[0] == (10, 10, 1)
        assert mm[1] == (50, 50, 1)

    def test_full_priority_order(self):
        """PAD > {TRACK, VIA, DIRECTIONAL} by distance. Ratsnest is excluded
        from the pointer pool (still appears as a state token upstream).
        """
        obs = _make_obs(
            pads=[_pad(10, 0)],
            tracks=[_track(0, 10, 1, 11)],
            ratsnest=[_ratsnest(-10, 0)],
            vias=[_via(0, -10, layer_start=1, layer_end=1)],
        )
        dir_cands = [(20, 0, 0, CTYPE_DIRECTIONAL)]
        feat, mask, mm = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=(0, 0), extra_candidates=dir_cands,
        )
        types = []
        for i in range(int(mask.sum())):
            for t in range(NUM_CAND_TYPES):
                if feat[i, 3 + t] == 1.0:
                    types.append(t)
                    break
        # PAD first (highest priority), ratsnest must not appear.
        assert types[0] == CTYPE_PAD
        assert CTYPE_RATSNEST not in types


# ---------------------------------------------------------------------------
# 5. Deduplication
# ---------------------------------------------------------------------------

class TestDedup:
    def test_same_coordinate_deduped(self):
        obs = _make_obs(
            pads=[_pad(10, 10), _pad(10, 10)],
        )
        _, mask, _ = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        assert mask.sum() == 1

    def test_same_xy_different_layer_not_deduped(self):
        obs = _make_obs(
            pads=[_pad(10, 10, layer=1), _pad(10, 10, layer=2)],
        )
        _, mask, _ = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        assert mask.sum() == 2


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_net_returns_empty(self):
        obs = _make_obs(pads=[_pad(10, 10)])
        feat, mask, mm = build_candidates_mlp(obs, current_net_id=None, route_head_mm=(0, 0))
        assert mask.sum() == 0
        assert len(mm) == 0

    def test_max_candidates_truncation(self):
        pads = [_pad(float(i), float(i)) for i in range(100)]
        obs = _make_obs(pads=pads)
        feat, mask, mm = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        assert mask.sum() == MAX_CANDIDATES
        assert len(mm) == MAX_CANDIDATES

    def test_empty_obs(self):
        obs = {"board_static": {}, "routing_geometry": {}}
        feat, mask, mm = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        assert mask.sum() == 0

    def test_coordinate_normalization(self):
        obs = _make_obs(pads=[_pad(50, 50)], bbox_x=0, bbox_y=0, scale=100)
        feat, _, _ = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        assert pytest.approx(feat[0, 0]) == 0.5  # x_norm = 50/100
        assert pytest.approx(feat[0, 1]) == 0.5  # y_norm = 50/100

    def test_layer_normalization(self):
        obs = _make_obs(pads=[_pad(10, 10, layer=2)])
        feat, _, _ = build_candidates_mlp(obs, current_net_id=1, route_head_mm=(0, 0))
        assert pytest.approx(feat[0, 2]) == 1.0  # 2/2 (max_layer defaults to 2)


# ---------------------------------------------------------------------------
# 7. Thru-hole pad sentinel layer expansion
# ---------------------------------------------------------------------------
#
# Regression: thru_hole pads carry parser sentinel ``layer == 0`` ("spans
# every copper layer"). Without expansion, the candidate pool emits a
# single (x, y, layer=0) entry that never matches any real routing layer
# — so the policy can never select the pad as a target, ``finish`` is
# unreachable, episodes drag to ``max_steps``, and iter-1 reward / entropy
# collapse on every real board with thru-hole pads. The fix mirrors the
# via expansion: one candidate per copper layer in [1, max_layer].

class TestThruHolePadExpansion:
    def test_thru_hole_expands_to_all_layers_2L(self):
        obs = _make_obs(pads=[_pad(30, 30, layer=0)])
        obs["board_static"]["copper_layers"] = 2
        _, mask, mm = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=(30, 30),
        )
        assert mask.sum() == 2
        assert sorted(l for _, _, l in mm) == [1, 2]

    def test_thru_hole_expands_to_all_layers_4L(self):
        obs = _make_obs(pads=[_pad(30, 30, layer=0)])
        obs["board_static"]["copper_layers"] = 4
        _, mask, mm = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=(30, 30),
        )
        assert mask.sum() == 4
        assert sorted(l for _, _, l in mm) == [1, 2, 3, 4]

    def test_sentinel_layer_zero_never_leaks(self):
        # Regardless of board layer count, no candidate should carry the
        # sentinel value 0 — that's an invalid routing layer.
        for n_copper in (2, 4, 6, 8):
            obs = _make_obs(pads=[_pad(30, 30, layer=0)])
            obs["board_static"]["copper_layers"] = n_copper
            _, _, mm = build_candidates_mlp(
                obs, current_net_id=1, route_head_mm=(30, 30),
            )
            for x, y, layer in mm:
                assert layer != 0, (
                    f"Sentinel layer=0 leaked into candidate pool "
                    f"(n_copper={n_copper}, mm={mm})"
                )

    def test_smd_pads_unaffected(self):
        # Non-zero pad layers (SMD / connect) must still emit one candidate
        # at exactly that layer.
        obs = _make_obs(pads=[
            _pad(10, 10, layer=1),
            _pad(20, 20, layer=2),
        ])
        obs["board_static"]["copper_layers"] = 2
        _, mask, mm = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=(0, 0),
        )
        assert mask.sum() == 2
        assert sorted(l for _, _, l in mm) == [1, 2]

    def test_thru_hole_emits_pad_type_one_hot(self):
        # Each expanded candidate is still PAD-typed (not VIA or
        # DIRECTIONAL).
        obs = _make_obs(pads=[_pad(30, 30, layer=0)])
        obs["board_static"]["copper_layers"] = 2
        feat, mask, _ = build_candidates_mlp(
            obs, current_net_id=1, route_head_mm=(30, 30),
        )
        for i in range(int(mask.sum())):
            assert feat[i, 3 + CTYPE_PAD] == 1.0
            assert feat[i, 3 + CTYPE_VIA] == 0.0


# ---------------------------------------------------------------------------
# 8. End-to-end with a real KiCad board (engine-backed)
# ---------------------------------------------------------------------------
#
# The mock-based tests above cover the candidate_builder in isolation. This
# integration test guards the full wire from parser → BoardStatic →
# build_json_observation → collect_raw_candidates: a regression in any of
# those (e.g. the sentinel encoding leaking back, or thru_hole pads being
# silently filtered) is also caught here.
#
# crossover_board.kicad_pcb has 12 thru_hole-typed pads on a 2-layer board.

class TestThruHoleRealBoard:
    @pytest.fixture
    def obs_with_thru_hole(self):
        import os
        board = os.path.join(
            os.path.dirname(__file__), "..", "fixtures",
            "crossover_board.kicad_pcb",
        )
        if not os.path.exists(board):
            pytest.skip(f"Board not found: {board}")
        try:
            from pcb_world.engine.kicad_engine import KiCadEngine
            from pcb_world.engine.pcb_file_parser import parse_pcb_file
            from pcb_world.core.observation import (
                BoardStatic,
                _build_board_static,
            )
        except Exception as e:
            pytest.skip(f"engine/state imports unavailable: {e}")

        engine = KiCadEngine(board)
        parsed = parse_pcb_file(board, engine)
        meta = engine.get_board_meta()
        board_info = BoardStatic.from_board(
            meta=meta,
            pads=parsed["board_snapshot"].pads,
            board_edges=parsed["board_edges"],
            net_names=parsed["net_names"],
            board_constraints={},
            obstacles=parsed.get("obstacles", []),
        )
        # Pick a net that owns at least one thru_hole pad.
        thru_pads = [p for p in parsed["board_snapshot"].pads if p.layer == 0]
        assert thru_pads, "fixture must expose thru_hole pads (layer=0)"
        target_nc = next(
            (p.net_code for p in thru_pads if p.net_code > 0), None,
        )
        assert target_nc is not None, "fixture must have a net-attached thru_hole pad"
        obs = {
            "board_static": _build_board_static(board_info),
            "routing_geometry": {},
        }
        return obs, target_nc, meta.copper_layers

    def test_thru_hole_pad_yields_one_candidate_per_copper_layer(
        self, obs_with_thru_hole,
    ):
        obs, net_id, n_copper = obs_with_thru_hole
        _, mask, mm = build_candidates_mlp(
            obs, current_net_id=net_id, route_head_mm=(0, 0),
        )
        assert mask.sum() > 0
        layers_seen = sorted({l for _, _, l in mm})
        # Every copper layer in [1, n_copper] must appear at least once.
        assert layers_seen == list(range(1, n_copper + 1)), (
            f"thru_hole pad expansion produced layers {layers_seen}, "
            f"expected 1..{n_copper}"
        )
        # And the sentinel 0 must NEVER show up.
        assert 0 not in (l for _, _, l in mm)
