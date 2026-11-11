"""The constraint surface of ``PCBWorld`` for stub envs: the knobs, the same-point state and
``act_constraints`` built by the shared rules (``pcb_world.core.constraints``) from the stub's
current obs — so a wrapper under test decodes and masks exactly as over the real env."""
from __future__ import annotations

import numpy as np

from pcb_world.core import constraints


class ConstraintEnvMixin:
    """Mix into a stub env that keeps its observation in ``_obs`` and its action mask in
    ``action_masks()``. Knob values = the RL wrapper's former own defaults (8-direction 0.5 mm
    ring, no off-board mask); set them per instance as the test needs."""

    net_reselect = False
    directional_candidates = None
    connectivity_filter = True
    pad_graze_margin_mm = 0.0
    offboard_mask = False
    mask_start_point = True
    force_walkaround = False
    _start_route_xy: tuple[float, float, int] | None = None

    @property
    def _last_obs(self) -> dict:
        """The act-time obs, as ``PCBWorld._last_obs``."""
        return self._obs

    def act_constraints(self) -> constraints.ActConstraints:
        return constraints.act_constraints(
            self._obs, action_mask=np.asarray(self.action_masks(), dtype=bool),
            cand_aug={"directional_candidates": self.directional_candidates,
                      "connectivity_filter": self.connectivity_filter,
                      "route_start_xy": self._start_route_xy, "cluster_keys": None,
                      "pad_graze_margin_mm": self.pad_graze_margin_mm},
            start_route_xy=self._start_route_xy, mask_start_point=self.mask_start_point,
            offboard_mask=self.offboard_mask, force_walkaround=self.force_walkaround,
            net_reselect=self.net_reselect,
        )

    def track_start_point(self, action: dict) -> None:
        """Advance the same-point state as ``PCBWorld.step`` does."""
        self._start_route_xy = constraints.next_start_route_xy(self._start_route_xy, action)


class ObsEnv(ConstraintEnvMixin):
    """A minimal env for a wrapper built with ``object.__new__`` + ``_refresh_cache``: holds one
    obs (``set_obs``), every action allowed, ``board_path`` for guard messages."""

    def __init__(self, obs: dict | None = None, board_path: str = "mock.kicad_pcb", **knobs) -> None:
        self._obs = obs or {}
        self.board_path = board_path
        for k, v in knobs.items():
            setattr(self, k, v)

    def set_obs(self, obs: dict) -> None:
        self._obs = obs

    def action_masks(self) -> np.ndarray:
        return np.ones(6, dtype=bool)
