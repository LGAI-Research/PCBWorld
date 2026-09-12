"""Pad-graze guard over synthesised (directional) candidates.

A directional candidate is a coordinate we invent at ``head + offset``; nothing
else checks it against pad geometry. One landing in the annulus just outside a
same-net pad's copper lets an action place a via / end a track whose copper only
GRAZES the pad — KiCad's shape-overlap connectivity scores that as connected
while its anchor-based dangling test does not, so KiCad's own track cleaner
deletes such copper (measured: 8 of 6691 THT pads across 199 routed boards).

``_aug["pad_graze_margin_mm"]`` (0 = off, the default) drops exactly those
synthesised points. Real geometry — pad centres, existing vias, track endpoints
— is never filtered: aiming at a pad centre is the CORRECT action.

Both observation formats share one collector, so the guard is asserted on the
dict and the indexed path alike.
"""

from __future__ import annotations

import pytest

from pcb_world.core.indexed_obs import dict_to_arrays
from pcb_world.vec.candidate_pool import (
    CTYPE_DIRECTIONAL,
    CTYPE_PAD,
    collect_raw_candidates,
)
from tests._mock_obs import make_mock_obs

# Mock pads are 1.0 x 1.0 -> radius 0.5. A 0.6mm via (radius 0.3) placed between
# 0.5 and 0.8 from the centre overlaps the pad without its centre being on it.
PAD_R = 0.5
VIA_R = 0.3


def _obs_and_pad(**kwargs):
    obs = make_mock_obs(n_nets=1, pads_per_net=2, is_routing=True, **kwargs)
    pad = next(iter(obs["board_static"]["nets"]["net_1"]["pads"].values()))
    px, py = pad["center"]["xy"]
    return obs, float(px), float(py)


def _extras(px: float, py: float) -> list[tuple[float, float, int, int]]:
    """One directional candidate per band: inside the pad, grazing, clear."""
    return [
        (px + 0.20, py, 1, CTYPE_DIRECTIONAL),   # inside the pad copper
        (px + 0.65, py, 1, CTYPE_DIRECTIONAL),   # graze annulus (0.5 < d < 0.8)
        (px + 1.50, py, 1, CTYPE_DIRECTIONAL),   # well clear
    ]


def _dirs(pool):
    return [(round(x, 3), round(y, 3)) for x, y, _l, ct in pool
            if ct == CTYPE_DIRECTIONAL]


class TestGuardOff:
    def test_default_keeps_every_directional_candidate(self):
        obs, px, py = _obs_and_pad()
        obs["_aug"] = {}

        pool = collect_raw_candidates(obs, 1, _extras(px, py))

        assert len(_dirs(pool)) == 3

    def test_zero_margin_is_off(self):
        obs, px, py = _obs_and_pad()
        obs["_aug"] = {"pad_graze_margin_mm": 0.0}

        assert len(_dirs(collect_raw_candidates(obs, 1, _extras(px, py)))) == 3


class TestGuardOn:
    def test_drops_only_the_grazing_candidate(self):
        obs, px, py = _obs_and_pad()
        obs["_aug"] = {"pad_graze_margin_mm": VIA_R}

        pool = collect_raw_candidates(obs, 1, _extras(px, py))

        dirs = _dirs(pool)
        assert (round(px + 0.65, 3), round(py, 3)) not in dirs   # grazing: gone
        assert (round(px + 0.20, 3), round(py, 3)) in dirs       # on-pad: kept
        assert (round(px + 1.50, 3), round(py, 3)) in dirs       # clear: kept

    def test_real_geometry_is_never_filtered(self):
        """The pad centre itself stays selectable — it is the correct target."""
        obs, px, py = _obs_and_pad()
        obs["_aug"] = {"pad_graze_margin_mm": VIA_R}

        pool = collect_raw_candidates(obs, 1, _extras(px, py))

        pads = [(round(x, 3), round(y, 3)) for x, y, _l, ct in pool
                if ct == CTYPE_PAD]
        assert (round(px, 3), round(py, 3)) in pads

    def test_margin_width_is_honoured(self):
        """A candidate at 0.65 is outside a 0.10mm band, inside a 0.30mm one."""
        obs, px, py = _obs_and_pad()
        extras = [(px + 0.65, py, 1, CTYPE_DIRECTIONAL)]

        obs["_aug"] = {"pad_graze_margin_mm": 0.10}
        assert len(_dirs(collect_raw_candidates(obs, 1, extras))) == 1

        obs["_aug"] = {"pad_graze_margin_mm": 0.30}
        assert len(_dirs(collect_raw_candidates(obs, 1, extras))) == 0

    def test_layer_scoped_for_single_layer_pads(self):
        """A pad on layer 1 must not shadow a candidate on layer 2."""
        obs, px, py = _obs_and_pad()
        obs["_aug"] = {"pad_graze_margin_mm": VIA_R}
        pad = next(iter(obs["board_static"]["nets"]["net_1"]["pads"].values()))
        assert pad["layer"] == 1                      # not a thru sentinel

        same = collect_raw_candidates(obs, 1, [(px + 0.65, py, 1, CTYPE_DIRECTIONAL)])
        other = collect_raw_candidates(obs, 1, [(px + 0.65, py, 2, CTYPE_DIRECTIONAL)])

        assert len(_dirs(same)) == 0
        assert len(_dirs(other)) == 1


class TestIndexedParity:
    @pytest.mark.parametrize("margin", [0.0, VIA_R])
    def test_dict_and_indexed_agree(self, margin: float):
        obs, px, py = _obs_and_pad()
        obs["_aug"] = {"pad_graze_margin_mm": margin}
        extras = _extras(px, py)

        from_dict = collect_raw_candidates(obs, 1, extras)
        from_indexed = collect_raw_candidates(dict_to_arrays(obs), 1, extras)

        assert from_dict == from_indexed
