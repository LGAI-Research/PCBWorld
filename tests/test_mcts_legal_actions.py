"""Unit tests for ``RLSearchEnv.legal_actions`` mask equivalence (no engine).

MCTS enumerates the factored ``(action_type, pointer, mode)`` space. Its legal
set must equal what the policy can sample with non-zero prob — i.e. compose the
SAME four masks the model hard-applies:

* ``action_masks``   — action type;
* ``net_valid_mask`` — the NET pointer (net_select only);
* ``mode_mask``      — routing mode (make_line/make_via/finish);
* pointer mask (``start_route_pointer_indices``) — the CAND pointer shared by
  start_route/make_line/make_via (the same-point start origin, forced to -inf).

The prior bug omitted the pointer mask for the cand-pointer actions, re-admitting
the zero-length same-point move (invisible under a trained prior≈0, but explored
under a uniform prior). These tests pin the composition with a mask-only fake
wrapper.
"""

import numpy as np
import pytest

from pcb_world.core.action_schema import (
    ACT_FINISH,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    NUM_ACTIONS,
)
from methods.rl_agent.policy.mcts_env import RLSearchEnv


class _FakeEnv:
    _drc_active = False


class _FakeWrapper:
    """Minimal wrapper exposing only what legal_actions() reads."""

    def __init__(self, *, allowed_types, n_cand, ptr_masked=(),
                 net_valid=(True,), valid_modes=(0,), n_modes=3):
        self.env = _FakeEnv()
        self._last_obs = {}
        self._at = np.zeros(NUM_ACTIONS, dtype=bool)
        for at in allowed_types:
            self._at[at] = True
        self._mode = np.zeros(n_modes, dtype=bool)
        for m in valid_modes:
            self._mode[m] = True
        self._nv = np.asarray(net_valid, dtype=bool)
        self._ptr = np.asarray(ptr_masked, dtype=np.int64)
        self.cand_mm_list = [(0.0, 0.0, 1)] * n_cand

    def action_masks(self):
        return self._at

    def mode_mask(self):
        return self._mode

    def net_valid_mask(self):
        return self._nv

    def start_route_pointer_indices(self):
        return self._ptr


def _legal(**kw):
    return list(RLSearchEnv(_FakeWrapper(**kw)).legal_actions())


# ---------------------------------------------------------------------------
# Pointer mask on the cand-pointer actions
# ---------------------------------------------------------------------------

class TestPointerMask:
    def test_start_route_excludes_masked_cand(self):
        acts = _legal(allowed_types=[ACT_START_ROUTE], n_cand=4, ptr_masked=[1])
        ptrs = sorted(p for (at, p, m) in acts if at == ACT_START_ROUTE)
        assert ptrs == [0, 2, 3]  # cand 1 (same-point origin) dropped
        assert all(m == -1 for (_at, _p, m) in acts)

    def test_make_line_via_exclude_masked_cand(self):
        acts = _legal(allowed_types=[ACT_MAKE_LINE, ACT_MAKE_VIA], n_cand=3,
                      ptr_masked=[0], valid_modes=[0, 2])
        for at in (ACT_MAKE_LINE, ACT_MAKE_VIA):
            ptrs = sorted({p for (a, p, m) in acts if a == at})
            assert ptrs == [1, 2]  # cand 0 dropped for BOTH pointer-mode acts
            modes = sorted({m for (a, p, m) in acts if a == at})
            assert modes == [0, 2]
        # (2 cands kept) × (2 modes) × (2 action types) = 8
        assert len(acts) == 8

    def test_empty_ptr_mask_full_enumeration(self):
        # Legacy behavior when no start origin is masked.
        acts = _legal(allowed_types=[ACT_MAKE_LINE], n_cand=3, ptr_masked=[],
                      valid_modes=[0, 1])
        assert len(acts) == 3 * 2
        assert sorted({p for (_a, p, _m) in acts}) == [0, 1, 2]

    def test_multiple_masked_cands(self):
        acts = _legal(allowed_types=[ACT_MAKE_LINE], n_cand=5, ptr_masked=[0, 3],
                      valid_modes=[0])
        assert sorted(p for (_a, p, _m) in acts) == [1, 2, 4]


# ---------------------------------------------------------------------------
# net_select uses the NET pointer, not the cand pointer mask
# ---------------------------------------------------------------------------

class TestNetSelectUnaffected:
    def test_net_valid_drives_net_select(self):
        # ptr_masked names cand index 0 — must NOT touch net_select's net pointer.
        acts = _legal(allowed_types=[ACT_NET_SELECT], n_cand=4, ptr_masked=[0],
                      net_valid=[True, False, True])
        ns = sorted(p for (at, p, m) in acts if at == ACT_NET_SELECT)
        assert ns == [0, 2]  # net pointer 0 stays (cand mask is irrelevant here)
        assert all(m == -1 for (_a, _p, m) in acts)


# ---------------------------------------------------------------------------
# Bare / mode-only action types
# ---------------------------------------------------------------------------

class TestBareActions:
    def test_finish_is_mode_only(self):
        acts = _legal(allowed_types=[ACT_FINISH], n_cand=3, valid_modes=[0, 2])
        assert sorted(acts) == [(ACT_FINISH, -1, 0), (ACT_FINISH, -1, 2)]

    def test_net_end_is_bare(self):
        acts = _legal(allowed_types=[ACT_NET_END], n_cand=3, ptr_masked=[0])
        assert acts == [(ACT_NET_END, -1, -1)]

    def test_masked_out_type_absent(self):
        acts = _legal(allowed_types=[ACT_FINISH], n_cand=3)
        assert all(at == ACT_FINISH for (at, _p, _m) in acts)


# ---------------------------------------------------------------------------
# Composition equals the policy's joint legal set
# ---------------------------------------------------------------------------

class TestJointComposition:
    def test_all_masks_compose(self):
        acts = set(_legal(
            allowed_types=[ACT_NET_SELECT, ACT_START_ROUTE, ACT_MAKE_LINE,
                           ACT_FINISH, ACT_NET_END],
            n_cand=3, ptr_masked=[2], net_valid=[True, False],
            valid_modes=[1],
        ))
        expected = {
            (ACT_NET_SELECT, 0, -1),               # net 1 invalid
            (ACT_START_ROUTE, 0, -1), (ACT_START_ROUTE, 1, -1),  # cand 2 masked
            (ACT_MAKE_LINE, 0, 1), (ACT_MAKE_LINE, 1, 1),        # cand 2 masked
            (ACT_FINISH, -1, 1),
            (ACT_NET_END, -1, -1),
        }
        assert acts == expected
