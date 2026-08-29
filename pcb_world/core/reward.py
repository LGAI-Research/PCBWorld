"""Reward function for KiCad RL environment.

No C++ imports — operates on RewardSnapshot and RoutingSessionState from engine_api.

Unified potential-based reward:
  Φ(s) = completion_bonus·I(unconnected==0)
         - unconnected_penalty·unconnected
         - wirelength_penalty·wirelength
         - drc_penalty·drc_violations

Modes:
  - terminal: -step_penalty per step, Φ(s_final) at episode end.
  - per_step: Φ(s') - Φ(s) - step_penalty per step, no final reward.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from pcb_world.engine.containers import RewardSnapshot, RoutingSessionState
from pcb_world.engine.drc import (
    DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED,
    DRC_SEVERITY_MODE_ERRORS_AND_WARNINGS,
    DRC_SEVERITY_MODE_ERRORS_ONLY,
    DRC_SEVERITY_MODES,
    ORPHAN_NET_KEY,
    VIOLATION_KEYWORDS,
)


@dataclass
class RewardState:
    """Snapshot of board state for reward computation."""

    unconnected: int
    drc_violations: int
    wirelength: float
    track_count: int
    via_count: int = 0
    drc_violations_per_net: dict[str, int] = field(default_factory=dict)
    # Severity-split: error-only counts. Sum (error + warning) stays in the
    # fields above so legacy paths are unchanged.
    drc_errors: int = 0
    drc_errors_per_net: dict[str, int] = field(default_factory=dict)
    # Errors + promoted warnings (track/via_dangling, net_conflict). See
    # pcb_world.engine.drc.DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED.
    drc_promoted: int = 0
    drc_promoted_per_net: dict[str, int] = field(default_factory=dict)
    # Per-net connectivity for ladder bonuses; -1 = unavailable (snapshot
    # taken without net-subset routing, or legacy construction paths).
    connected_net_count: int = -1
    target_net_count: int = -1
    # Net codes with >=1 open ratsnest edge (identity behind
    # ``connected_net_count``); None = unavailable (legacy paths) — the
    # size-weighted ladder bonuses fail loudly then.
    unconnected_net_codes: frozenset[int] | None = None

    @classmethod
    def from_snapshot(cls, snap: RewardSnapshot) -> RewardState:
        """Create RewardState from a RewardSnapshot.

        Args:
            snap: RewardSnapshot from engine_api.

        Returns:
            RewardState for reward computation.
        """
        return cls(
            unconnected=snap.unrouted_count,
            drc_violations=snap.drc_violation_count,
            wirelength=snap.total_wirelength,
            track_count=snap.track_count,
            via_count=snap.via_count,
            drc_violations_per_net=dict(snap.drc_violations_per_net),
            drc_errors=snap.drc_error_count,
            drc_errors_per_net=dict(snap.drc_errors_per_net),
            drc_promoted=snap.drc_promoted_count,
            drc_promoted_per_net=dict(snap.drc_promoted_per_net),
            connected_net_count=getattr(snap, "connected_net_count", -1),
            target_net_count=getattr(snap, "target_net_count", -1),
            unconnected_net_codes=getattr(snap, "unconnected_net_codes", None),
        )


# ---------------------------------------------------------------------------
# Unified potential-based reward
# ---------------------------------------------------------------------------


class PotentialReward:
    """Unified reward based on a single potential function Φ(s).

    Potential:
        Φ(s) = completion_bonus · I(unconnected == 0)
               - unconnected_penalty · unconnected
               - wirelength_penalty · wirelength
               - drc_penalty · drc_violations

    Dense reward per step::

        r_t = Φ(s') - Φ(s) - step_penalty

    Sparse reward at episode end::

        r_T = Φ(s_final)

    Args:
        completion_bonus: Reward when all nets are connected.
        unconnected_penalty: Penalty per remaining unconnected ratsnest edge.
        wirelength_penalty: Penalty per mm of total wirelength. When
            ``wirelength_bbox_normalize`` is set, the unit is instead one
            board long edge (see below).
        wirelength_bbox_normalize: If True, :meth:`bind_board` divides
            ``wirelength_penalty`` by max(bbox_w, bbox_h) (same
            board-resolution hook as ``completion_bonus_log_scale``), so
            the knob prices wire in board-long-edge units and the term is
            invariant to physical board scale.
        drc_penalty: Penalty per DRC violation (set to 0 to disable).
        net_completion_bonus: Bonus per fully-connected target net (its
            ratsnest edges all closed). Requires per-net connectivity in
            RewardState (net-subset routing) — raises if unavailable.
        net_clean_bonus: Bonus per target net with zero violations under
            the resolved severity mode. NOTE: open ratsnest edges are
            themselves counted as "unconnected" violations by the engine
            DRC, so at reset every unrouted net is dirty — this bonus is
            *earned* when a net is fully routed AND violation-free, and
            retracted only when a finished-clean net is disturbed. Orphan
            violations don't dirty any net here.
        clean_completion_bonus: Bonus when the board's total violation
            count (severity-mode set, orphans included) is zero — the
            clean-axis twin of ``completion_bonus``.
        clean_completion_bonus_log_scale: When set, :meth:`bind_board`
            resolves ``clean_completion_bonus = scale * ln(1 + net_count)``
            (same board-resolution hook as ``completion_bonus_log_scale``).
        net_bonus_size_log_scale: When > 0, the per-net ladder bonuses
            (``net_completion_bonus`` / ``net_clean_bonus``) are weighted by
            net size instead of counting nets flat: net i contributes
            ``w_i = scale * ln(1 + pad_groups_i)`` (bare-board pad groups,
            the routability baseline), scaled by the respective bonus
            coefficient. :meth:`bind_board` resolves the weights at every
            reset (via :meth:`set_net_size_weights`); computing a weighted
            potential before they are resolved fails loudly. Weighted net_clean sums
            only weighted (universe) nets — dirty nets outside the universe
            don't subtract (the flat path subtracts any dirty non-orphan).
        step_penalty: Flat cost subtracted every step (positive value).
        truncation_mode: How to reward truncated episodes
            (``"none"`` | ``"penalty_only"`` | ``"full"``).
        completion_bonus_log_scale: When set, :meth:`bind_board` resolves
            ``completion_bonus = scale * ln(1 + net_count)``.

    Board-dependent terms (the four ``*_log_scale`` / ``*_normalize`` hooks
    above) are resolved by :meth:`bind_board` — the ONE definition shared by
    the training env and the offline scorer. Callers (``PCBWorld``,
    ``eval.metrics``) only supply the board facts; they never restate the
    formulas.
    """

    VALID_TRUNCATION_MODES = ("none", "penalty_only", "full")
    VALID_DRC_SHAPES = ("linear", "saturating", "log_per_net")
    VALID_WIRE_VIA_EMISSIONS = ("per_step", "on_net_end")

    def __init__(
        self,
        completion_bonus: float = 5.0,
        unconnected_penalty: float = 1.0,
        wirelength_penalty: float = 0.0,
        wirelength_bbox_normalize: bool = False,
        drc_penalty: float = 0.5,
        net_completion_bonus: float = 0.0,
        net_clean_bonus: float = 0.0,
        clean_completion_bonus: float = 0.0,
        clean_completion_bonus_log_scale: float | None = None,
        net_bonus_size_log_scale: float = 0.0,
        step_penalty: float = 0.01,
        truncation_mode: str = "none",
        drc_shape: str = "linear",
        drc_aggregate_scale: float = 5.0,
        drc_per_net_scale: float = 1.0,
        drc_saturation_offset: float = 2.0,
        drc_log_scale: float = 1.0,
        drc_log_offset: float = 2.0,
        drc_log_agg_scale: float = 0.0,
        completion_bonus_log_scale: float | None = None,
        drc_penalty_include_warning: bool | None = None,
        drc_severity_mode: str | None = None,
        via_penalty: float = 0.0,
        wire_via_emission: str = "per_step",
        jumanji_connector_dense: bool = False,
        jumanji_connector_dense_flat: bool = False,
        jumanji_connector_wirelength_dense: bool = False,
    ) -> None:
        if truncation_mode not in self.VALID_TRUNCATION_MODES:
            raise ValueError(
                f"Invalid truncation_mode={truncation_mode!r}. "
                f"Must be one of {self.VALID_TRUNCATION_MODES}."
            )
        if drc_shape not in self.VALID_DRC_SHAPES:
            raise ValueError(
                f"Invalid drc_shape={drc_shape!r}. "
                f"Must be one of {self.VALID_DRC_SHAPES}."
            )
        if wire_via_emission not in self.VALID_WIRE_VIA_EMISSIONS:
            raise ValueError(
                f"Invalid wire_via_emission={wire_via_emission!r}. "
                f"Must be one of {self.VALID_WIRE_VIA_EMISSIONS}."
            )
        if drc_saturation_offset <= 0:
            raise ValueError(
                f"drc_saturation_offset must be positive, got {drc_saturation_offset}."
            )
        if drc_log_offset <= 0:
            raise ValueError(
                f"drc_log_offset must be positive, got {drc_log_offset}."
            )
        if net_bonus_size_log_scale < 0:
            raise ValueError(
                f"net_bonus_size_log_scale must be >= 0, "
                f"got {net_bonus_size_log_scale}."
            )
        self.completion_bonus = completion_bonus
        self.unconnected_penalty = unconnected_penalty
        self.wirelength_penalty = wirelength_penalty
        # Config-level (pre-normalization) value; bind_board derives the
        # effective wirelength_penalty from it, so rebinding is idempotent.
        self._wirelength_penalty_cfg = wirelength_penalty
        self.wirelength_bbox_normalize = wirelength_bbox_normalize
        self.drc_penalty = drc_penalty
        self.net_completion_bonus = net_completion_bonus
        self.net_clean_bonus = net_clean_bonus
        self.clean_completion_bonus = clean_completion_bonus
        self.clean_completion_bonus_log_scale = clean_completion_bonus_log_scale
        self.net_bonus_size_log_scale = net_bonus_size_log_scale
        # Per-net size weights, resolved by the env at reset (board-dependent)
        # via set_net_size_weights(). Both stay None until then; the weighted
        # potential path fails loudly if used unresolved.
        self.net_size_weights_by_code: dict[int, float] | None = None
        self.net_size_weights_by_name: dict[str, float] | None = None
        self.step_penalty = step_penalty
        self.truncation_mode = truncation_mode
        self.drc_shape = drc_shape
        self.drc_aggregate_scale = drc_aggregate_scale
        self.drc_per_net_scale = drc_per_net_scale
        self.drc_saturation_offset = drc_saturation_offset
        self.drc_log_scale = drc_log_scale
        self.drc_log_offset = drc_log_offset
        self.drc_log_agg_scale = drc_log_agg_scale
        self.completion_bonus_log_scale = completion_bonus_log_scale
        # Resolve the unified DRC severity mode:
        #   - explicit ``drc_severity_mode`` wins when provided,
        #   - else fall back to the legacy ``drc_penalty_include_warning`` bool
        #     (True → errors_and_warnings, False → errors_only),
        #   - else default to errors_only.
        # The resolved value is both the reward penalty gate AND the filter
        # the env applies when emitting DRC state tokens, so the agent never
        # observes violations the reward is ignoring (or vice versa).
        if drc_severity_mode is not None:
            if drc_severity_mode not in DRC_SEVERITY_MODES:
                raise ValueError(
                    f"Invalid drc_severity_mode={drc_severity_mode!r}. "
                    f"Must be one of {DRC_SEVERITY_MODES}."
                )
            if drc_penalty_include_warning is not None:
                warnings.warn(
                    "drc_penalty_include_warning is ignored when "
                    "drc_severity_mode is set.",
                    stacklevel=2,
                )
            resolved_mode = drc_severity_mode
        elif drc_penalty_include_warning is True:
            resolved_mode = DRC_SEVERITY_MODE_ERRORS_AND_WARNINGS
        else:
            resolved_mode = DRC_SEVERITY_MODE_ERRORS_ONLY
        self.drc_severity_mode = resolved_mode
        # Keep the legacy attribute readable so downstream code that only
        # toggles between the two legacy modes still works. For the new
        # ``errors_and_promoted`` mode this flag is False (it is neither
        # "errors only" nor the full "errors + warnings" set).
        self.drc_penalty_include_warning = (
            resolved_mode == DRC_SEVERITY_MODE_ERRORS_AND_WARNINGS
        )
        self.via_penalty = via_penalty
        self.wire_via_emission = wire_via_emission
        self.jumanji_connector_dense = bool(jumanji_connector_dense)
        self.jumanji_connector_dense_flat = bool(jumanji_connector_dense_flat)
        self.jumanji_connector_wirelength_dense = bool(jumanji_connector_wirelength_dense)
        jumanji_variants = sum(
            bool(x)
            for x in (
                self.jumanji_connector_dense,
                self.jumanji_connector_dense_flat,
                self.jumanji_connector_wirelength_dense,
            )
        )
        if jumanji_variants > 1:
            raise ValueError(
                "At most one Jumanji Connector dense reward variant can be enabled."
            )

    def _severity_counts(self, state: RewardState) -> tuple[int, dict[str, int]]:
        """(total count, per-net dict) for the resolved severity mode."""
        mode = self.drc_severity_mode
        if mode == DRC_SEVERITY_MODE_ERRORS_AND_WARNINGS:
            return state.drc_violations, state.drc_violations_per_net
        if mode == DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED:
            return state.drc_promoted, state.drc_promoted_per_net
        return state.drc_errors, state.drc_errors_per_net

    def bind_board(
        self,
        *,
        net_count: int | None = None,
        bbox_w: float | None = None,
        bbox_h: float | None = None,
        pad_groups: Mapping[int, int] | None = None,
        net_names: Mapping[int, str] | None = None,
        routable_nets: Iterable[int] | None = None,
    ) -> None:
        """Resolve the board-dependent reward terms on this board.

        The single definition of every "board-resolution" hook, shared by the
        training env (``PCBWorld``) and the offline scorer
        (``eval.metrics.compute_metrics``) so both price a board identically.
        Two independent phases, selected by which keyword group is passed
        (each group all-or-nothing; at least one group required):

        * **static** — ``net_count``, ``bbox_w``, ``bbox_h``::

              completion_bonus       = completion_bonus_log_scale       * ln(1 + net_count)
              clean_completion_bonus = clean_completion_bonus_log_scale * ln(1 + net_count)
              wirelength_penalty     = <config wirelength_penalty> / max(bbox_w, bbox_h)

          each only when its knob is set (``*_log_scale`` not None /
          ``wirelength_bbox_normalize`` True). Derived from the config values,
          so rebinding is idempotent. Call AFTER any ``with_overrides`` — a
          CLI ``--wirelength-penalty`` is what gets normalized.
        * **per-reset** — ``pad_groups``, ``net_names``, ``routable_nets``:
          size-weighted ladder weights ``w_i = net_bonus_size_log_scale *
          ln(1 + pad_groups[i])`` for every routable net (bare-board pad
          groups: the routability baseline), keyed by code and by name
          (:meth:`set_net_size_weights`). Bare-board pad groups are
          episode-dependent under keep-routing augmentation, so the env
          rebinds at every reset. No-op unless ``net_bonus_size_log_scale > 0``.

        Args:
            net_count: Board net count (``BoardMeta.net_count``).
            bbox_w: Board bbox width, mm (``BoardMeta.bbox_w``).
            bbox_h: Board bbox height, mm (``BoardMeta.bbox_h``).
            pad_groups: ``{net_code: pad-group count}`` of the bare board
                (``KiCadEngine.get_pad_groups()`` after the reset strip).
            net_names: ``{net_code: net_name}`` covering every routable net
                (DRC reports per-net counts by name).
            routable_nets: Net codes that need routing work (>=2 pads —
                ``KiCadEngine.get_routable_nets()``).
        """
        static_given = sum(v is not None for v in (net_count, bbox_w, bbox_h))
        reset_given = sum(
            v is not None for v in (pad_groups, net_names, routable_nets)
        )
        if static_given not in (0, 3) or reset_given not in (0, 3):
            raise ValueError(
                "bind_board: pass the whole static group (net_count, bbox_w, "
                "bbox_h) and/or the whole per-reset group (pad_groups, "
                "net_names, routable_nets) — partial groups are not allowed."
            )
        if not static_given and not reset_given:
            raise ValueError("bind_board: nothing to bind (no keyword group given).")

        if static_given:
            if self.completion_bonus_log_scale is not None:
                self.completion_bonus = (
                    self.completion_bonus_log_scale * math.log1p(net_count)
                )
            if self.clean_completion_bonus_log_scale is not None:
                self.clean_completion_bonus = (
                    self.clean_completion_bonus_log_scale * math.log1p(net_count)
                )
            if self.wirelength_bbox_normalize:
                bbox_long_edge = max(bbox_w, bbox_h)
                if bbox_long_edge <= 0:
                    raise ValueError(
                        f"wirelength_bbox_normalize requires a positive board "
                        f"bbox, got bbox_w={bbox_w}, bbox_h={bbox_h}."
                    )
                self.wirelength_penalty = (
                    self._wirelength_penalty_cfg / bbox_long_edge
                )

        if reset_given and self.net_bonus_size_log_scale > 0:
            scale = self.net_bonus_size_log_scale
            nets = sorted(int(c) for c in routable_nets)
            missing = [c for c in nets if c not in pad_groups]
            if missing:
                raise RuntimeError(
                    f"net_bonus_size_log_scale: routable net(s) {missing} have "
                    f"no pad groups — cannot resolve size weights (stale "
                    f"connectivity? build_connectivity() must precede the bind)."
                )
            unnamed = [c for c in nets if c not in net_names]
            if unnamed:
                raise RuntimeError(
                    f"net_bonus_size_log_scale: routable net(s) {unnamed} have "
                    f"no net name — cannot key the DRC-side weights."
                )
            by_code = {c: scale * math.log1p(pad_groups[c]) for c in nets}
            by_name = {net_names[c]: by_code[c] for c in nets}
            self.set_net_size_weights(by_code, by_name)

    def set_net_size_weights(
        self,
        by_code: dict[int, float],
        by_name: dict[str, float],
    ) -> None:
        """Install the per-net size weights for the weighted ladder bonuses.

        Called by :meth:`bind_board` at every env reset (weights are board-
        and, under keep-routing augmentation, episode-dependent). ``by_code`` keys the
        connectivity side (ratsnest net codes), ``by_name`` the DRC side
        (per-net violation dict keys); both must describe the same nets.
        """
        if len(by_code) != len(by_name):
            raise ValueError(
                f"net size weight maps disagree: {len(by_code)} codes vs "
                f"{len(by_name)} names (duplicate or missing net names?)."
            )
        for w in by_code.values():
            if not (w > 0 and math.isfinite(w)):
                raise ValueError(
                    f"net size weights must be positive finite, got {w}."
                )
        self.net_size_weights_by_code = dict(by_code)
        self.net_size_weights_by_name = dict(by_name)

    def _drc_penalty(self, state: RewardState) -> float:
        """Return the DRC penalty magnitude (non-negative) for *state*.

        Saturating shape uses:
          - x = number of offending nets (len of per-net dict; orphan violations
                are phantom-net entries contributed by DRCUtils)
          - x_i = per-net violation count
        so aggregate saturates at drc_aggregate_scale regardless of total
        violation count. Linear shape keeps its original semantics.

        log_per_net shape uses:
          Φ_drc = -drc_log_agg_scale · ln(1 + x / drc_log_offset)  [if agg_scale > 0]
                  -Σ_i drc_log_scale · ln(1 + x_i / drc_log_offset)
        Per-net log term penalizes depth (violations within a net); optional
        aggregate-log term (drc_log_agg_scale > 0) adds an orthogonal breadth
        signal (how many nets are dirtied). Both share drc_log_offset as the
        log-curve knee. Both terms are concave/unbounded.
        """
        count, per_net = self._severity_counts(state)
        if self.drc_shape == "linear":
            return self.drc_penalty * count
        if self.drc_shape == "log_per_net":
            o = self.drc_log_offset
            s_pn = self.drc_log_scale
            s_agg = self.drc_log_agg_scale
            pen = 0.0
            x = float(len(per_net))
            if s_agg > 0 and x > 0:
                pen += s_agg * math.log1p(x / o)
            for x_i in per_net.values():
                xi = float(x_i)
                if xi > 0:
                    pen += s_pn * math.log1p(xi / o)
            return pen
        # saturating
        o = self.drc_saturation_offset
        x = float(len(per_net))
        pen = self.drc_aggregate_scale * x / (x + o) if x > 0 else 0.0
        for x_i in per_net.values():
            xi = float(x_i)
            if xi > 0:
                pen += self.drc_per_net_scale * xi / (xi + o)
        return pen

    def _phi_main(self, state: RewardState) -> float:
        """Main potential piece: completion + unconnected + DRC.

        Split out from the full potential so the ``on_net_end`` wire/via
        emission mode (see :meth:`compute_dense_netend`) can emit this
        every step while holding wire/via until a net completes.
        """
        phi = -self.unconnected_penalty * state.unconnected
        if state.unconnected == 0:
            phi += self.completion_bonus
        phi -= self._drc_penalty(state)
        if (
            self.net_completion_bonus
            or self.net_clean_bonus
            or self.clean_completion_bonus
        ):
            count, per_net = self._severity_counts(state)
            weighted = self.net_bonus_size_log_scale > 0
            if weighted and self.net_size_weights_by_code is None:
                raise ValueError(
                    "net_bonus_size_log_scale > 0 but per-net size weights are "
                    "unresolved — PCBWorld resolves them at reset; standalone "
                    "use must call set_net_size_weights() first."
                )
            if self.net_completion_bonus:
                if weighted:
                    if state.unconnected_net_codes is None:
                        raise ValueError(
                            "size-weighted net_completion_bonus requires "
                            "unconnected_net_codes in RewardState "
                            "(RewardSnapshot from a legacy construction path)."
                        )
                    uc = state.unconnected_net_codes
                    phi += self.net_completion_bonus * sum(
                        w for c, w in self.net_size_weights_by_code.items()
                        if c not in uc
                    )
                else:
                    if state.connected_net_count < 0:
                        raise ValueError(
                            "net_completion_bonus requires per-net connectivity "
                            "(RewardSnapshot taken without net-subset routing)."
                        )
                    phi += self.net_completion_bonus * state.connected_net_count
            if self.net_clean_bonus:
                if weighted:
                    w_by_name = self.net_size_weights_by_name
                    dirty_w = sum(
                        w_by_name[net] for net, x in per_net.items()
                        if x > 0 and net in w_by_name
                    )
                    phi += self.net_clean_bonus * (
                        sum(w_by_name.values()) - dirty_w
                    )
                else:
                    if state.target_net_count < 0:
                        raise ValueError(
                            "net_clean_bonus requires target_net_count "
                            "(RewardSnapshot taken without net-subset routing)."
                        )
                    dirty = sum(
                        1 for net, x in per_net.items()
                        if x > 0 and net != ORPHAN_NET_KEY
                    )
                    phi += self.net_clean_bonus * (state.target_net_count - dirty)
            if self.clean_completion_bonus and count == 0:
                phi += self.clean_completion_bonus
        return phi

    def _phi_wire_via(self, state: RewardState) -> float:
        """Wire + via potential piece."""
        return -(
            self.wirelength_penalty * state.wirelength
            + self.via_penalty * state.via_count
        )

    def potential(self, state: RewardState) -> float:
        """Evaluate the potential Φ(s).

        Args:
            state: Current board state.

        Returns:
            Scalar potential value.
        """
        return self._phi_main(state) + self._phi_wire_via(state)

    def compute_dense(
        self,
        before: RewardState,
        after: RewardState,
    ) -> float:
        """Dense per-step reward: Φ(after) - Φ(before) - step_penalty.

        Args:
            before: Board state before action.
            after: Board state after action.

        Returns:
            Scalar reward value.
        """
        if (
            self.jumanji_connector_dense
            or self.jumanji_connector_dense_flat
            or self.jumanji_connector_wirelength_dense
        ):
            connected_delta = before.unconnected - after.unconnected
            if self.jumanji_connector_wirelength_dense:
                wire_delta = after.wirelength - before.wirelength
                return connected_delta - self.wirelength_penalty * wire_delta
            if self.jumanji_connector_dense_flat:
                return connected_delta - self.step_penalty
            return connected_delta - self.step_penalty * before.unconnected
        return self.potential(after) - self.potential(before) - self.step_penalty

    def compute_dense_netend(
        self,
        before: RewardState,
        after: RewardState,
        wire_via_ref_state: RewardState,
        flush_wire_via: bool,
    ) -> tuple[float, RewardState]:
        """Dense reward where wire/via penalty is held until ``flush_wire_via``.

        Active when ``wire_via_emission == "on_net_end"`` — wire + via
        contribution is zero on most steps; when the caller reports a
        flush event (i.e. the step's action was ``net_end`` or the
        episode just completed), the wire/via Φ delta between
        ``wire_via_ref_state`` and ``after`` is emitted as a bump. The
        main potential piece (completion + unconnected + DRC) still
        contributes every step, and ``step_penalty`` is subtracted
        every step, matching :meth:`compute_dense` semantics.

        Args:
            before: Board state before action.
            after: Board state after action.
            wire_via_ref_state: Baseline state for wire/via accumulation
                — typically the board state at the previous flush event
                (or at episode reset).
            flush_wire_via: If True, emit the accumulated wire/via delta
                on this step.

        Returns:
            Tuple ``(reward, next_wire_via_ref_state)``. The second
            element is ``after`` when flushed (so the next accumulation
            starts from the current state) or ``wire_via_ref_state``
            unchanged otherwise.
        """
        if (
            self.jumanji_connector_dense
            or self.jumanji_connector_dense_flat
            or self.jumanji_connector_wirelength_dense
        ):
            return self.compute_dense(before, after), wire_via_ref_state
        reward = (
            self._phi_main(after)
            - self._phi_main(before)
            - self.step_penalty
        )
        next_ref = wire_via_ref_state
        if flush_wire_via:
            reward += (
                self._phi_wire_via(after)
                - self._phi_wire_via(wire_via_ref_state)
            )
            next_ref = after
        return reward, next_ref

    def compute_final(self, state: RewardState) -> float:
        """Sparse episode-end reward at true termination: Φ(s_final).

        Args:
            state: Board state at episode end (with DRC info).

        Returns:
            Scalar reward.
        """
        return self.potential(state)

    def compute_truncation(self, state: RewardState) -> float:
        """Reward at truncation (time-limit reached).

        Behaviour depends on ``truncation_mode``:
          - ``"none"``:  0.0 — rely on critic's V(s_next) bootstrap.
          - ``"penalty_only"``: only unconnected + DRC penalties (no bonus).
          - ``"full"``: identical to ``compute_final`` (legacy).

        Args:
            state: Board state at truncation (with DRC info).

        Returns:
            Scalar reward.
        """
        if self.truncation_mode == "none":
            return 0.0
        if self.truncation_mode == "penalty_only":
            reward = -self.unconnected_penalty * state.unconnected
            reward -= self._drc_penalty(state)
            return reward
        # "full" — legacy behaviour
        return self.compute_final(state)


# ---------------------------------------------------------------------------
# Deprecated aliases — kept for backward compatibility
# ---------------------------------------------------------------------------


class RewardFunction:
    """Deprecated: use PotentialReward instead.

    Thin wrapper that translates the legacy RewardFunction API to
    PotentialReward for code that still imports this class.
    """

    def __init__(
        self,
        completion_weight: float = 10.0,
        drc_weight: float = -2.0,
        wirelength_weight: float = -0.01,
        step_cost: float = -0.1,
        completion_bonus: float = 50.0,
        distance_shaping_weight: float = 1.0,
    ) -> None:
        warnings.warn(
            "RewardFunction is deprecated, use PotentialReward instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.completion_weight = completion_weight
        self.drc_weight = drc_weight
        self.wirelength_weight = wirelength_weight
        self.step_cost = step_cost
        self.completion_bonus = completion_bonus
        self.distance_shaping_weight = distance_shaping_weight

    def compute(
        self,
        before: RewardState,
        after: RewardState,
        *,
        prev_distance: float | None = None,
        curr_distance: float | None = None,
    ) -> float:
        """Legacy compute — reproduces old RewardFunction behaviour."""
        unconnected_delta = before.unconnected - after.unconnected
        r = self.completion_weight * unconnected_delta

        drc_delta = after.drc_violations - before.drc_violations
        r += self.drc_weight * drc_delta

        wl_delta = after.wirelength - before.wirelength
        r += self.wirelength_weight * wl_delta

        if (
            prev_distance is not None
            and curr_distance is not None
            and self.distance_shaping_weight != 0.0
        ):
            r += self.distance_shaping_weight * (prev_distance - curr_distance)

        r += self.step_cost

        if after.unconnected == 0:
            r += self.completion_bonus

        return r

    @staticmethod
    def compute_head_target_distance(
        router_state: RoutingSessionState,
    ) -> float | None:
        """Compute Euclidean distance from route head to routing target."""
        head = router_state.route_head
        target = router_state.routing_target

        if head[2] < 0 or target[2] < 0:
            return None

        return math.hypot(head[0] - target[0], head[1] - target[1])


class FinalOnlyReward:
    """Deprecated: use PotentialReward instead.

    Thin wrapper that translates the legacy FinalOnlyReward API to
    PotentialReward for code that still imports this class.
    """

    VALID_TRUNCATION_MODES = ("none", "penalty_only", "full")

    def __init__(
        self,
        completion_bonus: float = 5.0,
        unconnected_penalty: float = 1.0,
        wirelength_penalty: float = 0.01,
        drc_penalty: float = 0.5,
        truncation_mode: str = "none",
    ) -> None:
        warnings.warn(
            "FinalOnlyReward is deprecated, use PotentialReward instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._pr = PotentialReward(
            completion_bonus=completion_bonus,
            unconnected_penalty=unconnected_penalty,
            wirelength_penalty=wirelength_penalty,
            drc_penalty=drc_penalty,
            truncation_mode=truncation_mode,
        )
        self.completion_bonus = completion_bonus
        self.unconnected_penalty = unconnected_penalty
        self.wirelength_penalty = wirelength_penalty
        self.drc_penalty = drc_penalty
        self.truncation_mode = truncation_mode

    def compute_final(
        self,
        unconnected: int,
        drc_violations: int,
        wirelength: float = 0.0,
    ) -> float:
        """Legacy compute_final — keyword-arg interface."""
        state = RewardState(
            unconnected=unconnected,
            drc_violations=drc_violations,
            wirelength=wirelength,
            track_count=0,
        )
        return self._pr.compute_final(state)

    def compute_truncation(
        self,
        unconnected: int,
        drc_violations: int,
        wirelength: float = 0.0,
    ) -> float:
        """Legacy compute_truncation — keyword-arg interface."""
        state = RewardState(
            unconnected=unconnected,
            drc_violations=drc_violations,
            wirelength=wirelength,
            track_count=0,
        )
        return self._pr.compute_truncation(state)


# Keyword fragments mapped to weight-category keys. Retained for the upcoming
# DRC state-token work, which will reuse the same taxonomy. The keyword
# *sequence* is owned by pcb_world.engine.drc.VIOLATION_KEYWORDS (imported above); we
# attach each keyword's own weight-category key here, so the two lists can't
# drift — a keyword added to the owner without a category below raises KeyError
# at import.
_KEYWORD_TO_CATEGORY: dict[str, str] = {
    "clearance": "CLEARANCE",
    "track width": "TRACK_WIDTH",
    "unconnected": "UNCONNECTED",
    "short": "SHORT",
    "via": "VIA",
    "copper edge": "COPPER_EDGE_CLEARANCE",
}
_VIOLATION_KEYWORD_MAP: list[tuple[str, str]] = [
    (kw, _KEYWORD_TO_CATEGORY[kw]) for kw in VIOLATION_KEYWORDS
]
