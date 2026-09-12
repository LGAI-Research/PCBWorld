"""Reward module tests.

Tests:
- PotentialReward: potential(), compute_dense(), compute_final(), compute_truncation()
- Telescope property: sum of dense rewards = Φ(s_final) - Φ(s_0)
- Legacy backward compatibility
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))
sys.path.insert(0, str(PROJECT_ROOT))

BOARD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "simple_obstacle_board.kicad_pcb"


class TestPotentialReward:
    def test_potential_all_routed(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(completion_bonus=5.0, unconnected_penalty=1.0)
        state = RewardState(unconnected=0, drc_violations=0, wirelength=0.0, track_count=0)
        assert pr.potential(state) == pytest.approx(5.0)

    def test_potential_unconnected(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(completion_bonus=5.0, unconnected_penalty=1.0)
        state = RewardState(unconnected=3, drc_violations=0, wirelength=0.0, track_count=0)
        # No bonus, -1.0 * 3
        assert pr.potential(state) == pytest.approx(-3.0)

    def test_potential_with_drc(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(drc_penalty=0.5, unconnected_penalty=0.0)
        # Default drc_penalty_include_warning=False → error-only path;
        # mirror the legacy count into drc_errors so the test still checks
        # the same behavior.
        state = RewardState(
            unconnected=1, drc_violations=4, drc_errors=4,
            wirelength=0.0, track_count=0,
        )
        # No bonus (unconnected>0), -0.5*4 = -2.0
        assert pr.potential(state) == pytest.approx(-2.0)

    def test_dense_positive_on_completion(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            completion_bonus=5.0, unconnected_penalty=1.0, step_penalty=0.0,
        )
        before = RewardState(unconnected=3, drc_violations=0, wirelength=0.0, track_count=0)
        after = RewardState(unconnected=2, drc_violations=0, wirelength=0.0, track_count=1)

        reward = pr.compute_dense(before, after)
        # Φ(after) - Φ(before) = (-2) - (-3) = +1.0
        assert reward == pytest.approx(1.0)

    def test_dense_no_progress_with_step_penalty(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            completion_bonus=5.0, unconnected_penalty=1.0, step_penalty=0.01,
        )
        before = RewardState(unconnected=3, drc_violations=0, wirelength=0.0, track_count=0)
        after = RewardState(unconnected=3, drc_violations=0, wirelength=0.0, track_count=0)

        reward = pr.compute_dense(before, after)
        assert reward == pytest.approx(-0.01)

    def test_dense_completion_bonus(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            completion_bonus=5.0, unconnected_penalty=1.0, step_penalty=0.0,
        )
        before = RewardState(unconnected=1, drc_violations=0, wirelength=0.0, track_count=5)
        after = RewardState(unconnected=0, drc_violations=0, wirelength=5.0, track_count=6)

        reward = pr.compute_dense(before, after)
        # Φ(after) - Φ(before) = (5.0 + 0) - (-1.0) = 6.0
        assert reward == pytest.approx(6.0)

    def test_dense_wirelength_penalty(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            unconnected_penalty=0.0, wirelength_penalty=1.0, step_penalty=0.0,
        )
        before = RewardState(unconnected=1, drc_violations=0, wirelength=0.0, track_count=0)
        after = RewardState(unconnected=1, drc_violations=0, wirelength=10.0, track_count=5)

        reward = pr.compute_dense(before, after)
        # Φ(after) - Φ(before) = (-10.0) - (0.0) = -10.0
        assert reward == pytest.approx(-10.0)

    def test_dense_drc_penalty(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            unconnected_penalty=0.0, drc_penalty=2.0, step_penalty=0.0,
        )
        before = RewardState(unconnected=1, drc_violations=0, wirelength=0.0, track_count=0)
        after = RewardState(
            unconnected=1, drc_violations=3, drc_errors=3,
            wirelength=0.0, track_count=1,
        )

        reward = pr.compute_dense(before, after)
        # Φ(after) - Φ(before) = (-6.0) - (0.0) = -6.0
        assert reward == pytest.approx(-6.0)

    def test_telescope_property(self) -> None:
        """Sum of dense rewards = Φ(s_final) - Φ(s_0)."""
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            completion_bonus=5.0, unconnected_penalty=1.0,
            wirelength_penalty=0.01, step_penalty=0.0,
        )

        states = [
            RewardState(unconnected=3, drc_violations=0, wirelength=0.0, track_count=0),
            RewardState(unconnected=2, drc_violations=0, wirelength=2.0, track_count=2),
            RewardState(unconnected=1, drc_violations=0, wirelength=5.0, track_count=5),
            RewardState(unconnected=0, drc_violations=0, wirelength=8.0, track_count=8),
        ]

        total_dense = sum(
            pr.compute_dense(states[i], states[i + 1])
            for i in range(len(states) - 1)
        )

        expected = pr.potential(states[-1]) - pr.potential(states[0])
        assert total_dense == pytest.approx(expected)

    def test_compute_final(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(completion_bonus=5.0, unconnected_penalty=1.0)
        state = RewardState(unconnected=0, drc_violations=0, wirelength=0.0, track_count=10)
        assert pr.compute_final(state) == pytest.approx(5.0)

    def test_compute_truncation_none(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(truncation_mode="none")
        state = RewardState(unconnected=2, drc_violations=1, wirelength=5.0, track_count=5)
        assert pr.compute_truncation(state) == pytest.approx(0.0)

    def test_compute_truncation_penalty_only(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            unconnected_penalty=1.0, drc_penalty=0.5,
            truncation_mode="penalty_only",
        )
        state = RewardState(
            unconnected=2, drc_violations=4, drc_errors=4,
            wirelength=5.0, track_count=5,
        )
        # -1.0*2 - 0.5*4 = -4.0
        assert pr.compute_truncation(state) == pytest.approx(-4.0)

    def test_compute_truncation_full(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            completion_bonus=5.0, unconnected_penalty=1.0, truncation_mode="full",
        )
        state = RewardState(unconnected=2, drc_violations=0, wirelength=0.0, track_count=5)
        # Same as compute_final: no bonus, -1.0*2 = -2.0
        assert pr.compute_truncation(state) == pytest.approx(-2.0)


class TestDrcSeveritySplit:
    """Check the drc_penalty_include_warning flag toggles between
    error-only (default) and error+warning penalty paths."""

    def test_default_error_only_ignores_warnings(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            drc_penalty=1.0, unconnected_penalty=0.0,
            # drc_penalty_include_warning defaults to False.
        )
        # 2 errors + 3 warnings (sum=5). Default: penalize errors only.
        state = RewardState(
            unconnected=0, drc_violations=5, drc_errors=2,
            wirelength=0.0, track_count=0,
        )
        # -1.0 * 2 errors = -2.0 (bonus wouldn't fire without completion)
        # completion_bonus fires because unconnected==0 → +5 - 2 = 3
        assert pr.potential(state) == pytest.approx(5.0 - 2.0)

    def test_include_warning_sums_both(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            drc_penalty=1.0, unconnected_penalty=0.0,
            drc_penalty_include_warning=True,
        )
        state = RewardState(
            unconnected=0, drc_violations=5, drc_errors=2,
            wirelength=0.0, track_count=0,
        )
        # -1.0 * 5 = -5.0, bonus +5 → 0.0
        assert pr.potential(state) == pytest.approx(0.0)

    def test_saturating_shape_uses_error_per_net(self) -> None:
        from pcb_world.core.reward import PotentialReward, RewardState

        pr = PotentialReward(
            drc_penalty=1.0, unconnected_penalty=0.0,
            drc_shape="saturating",
            drc_aggregate_scale=10.0, drc_per_net_scale=1.0,
            drc_saturation_offset=2.0,
            # error-only (default)
        )
        state = RewardState(
            unconnected=1,  # avoid completion bonus
            drc_violations=10,
            drc_violations_per_net={"A": 5, "B": 5},
            drc_errors=2,
            drc_errors_per_net={"A": 2},
            wirelength=0.0, track_count=0,
        )
        # Uses drc_errors_per_net (1 net dirty, 2 errors on "A")
        # aggregate: 10 * 1/(1+2) = 3.333…; per-net: 1 * 2/(2+2)=0.5
        expected_pen = 10.0 * 1.0 / (1.0 + 2.0) + 1.0 * 2.0 / (2.0 + 2.0)
        assert pr.potential(state) == pytest.approx(-expected_pen)


class TestDrcSeverityMode:
    """The unified ``drc_severity_mode`` knob gates which DRC count feeds
    ``_drc_penalty`` — reward and state must stay in sync via a single
    source of truth."""

    def _state(self):
        from pcb_world.core.reward import RewardState

        # 2 errors, 5 error+promoted, 10 errors+warnings (sum).
        return RewardState(
            unconnected=0,
            drc_violations=10, drc_violations_per_net={"A": 10},
            drc_errors=2, drc_errors_per_net={"A": 2},
            drc_promoted=5, drc_promoted_per_net={"A": 5},
            wirelength=0.0, track_count=0,
        )

    def test_errors_only_default(self) -> None:
        from pcb_world.core.reward import PotentialReward

        pr = PotentialReward(
            drc_penalty=1.0, unconnected_penalty=0.0, completion_bonus=0.0,
        )
        assert pr.drc_severity_mode == "errors_only"
        assert pr.potential(self._state()) == pytest.approx(-2.0)

    def test_errors_and_promoted_penalty(self) -> None:
        from pcb_world.core.reward import PotentialReward

        pr = PotentialReward(
            drc_penalty=1.0, unconnected_penalty=0.0, completion_bonus=0.0,
            drc_severity_mode="errors_and_promoted",
        )
        assert pr.drc_severity_mode == "errors_and_promoted"
        assert pr.potential(self._state()) == pytest.approx(-5.0)

    def test_errors_and_warnings_mode(self) -> None:
        from pcb_world.core.reward import PotentialReward

        pr = PotentialReward(
            drc_penalty=1.0, unconnected_penalty=0.0, completion_bonus=0.0,
            drc_severity_mode="errors_and_warnings",
        )
        assert pr.potential(self._state()) == pytest.approx(-10.0)

    def test_legacy_include_warning_still_works(self) -> None:
        from pcb_world.core.reward import PotentialReward

        pr_true = PotentialReward(
            drc_penalty=1.0, unconnected_penalty=0.0, completion_bonus=0.0,
            drc_penalty_include_warning=True,
        )
        assert pr_true.drc_severity_mode == "errors_and_warnings"
        assert pr_true.potential(self._state()) == pytest.approx(-10.0)

        pr_false = PotentialReward(
            drc_penalty=1.0, unconnected_penalty=0.0, completion_bonus=0.0,
            drc_penalty_include_warning=False,
        )
        assert pr_false.drc_severity_mode == "errors_only"

    def test_invalid_mode_rejected(self) -> None:
        from pcb_world.core.reward import PotentialReward

        with pytest.raises(ValueError):
            PotentialReward(drc_penalty=1.0, drc_severity_mode="bogus")


class TestRewardStateFromRouter:
    def test_captures_state(self) -> None:
        try:
            from pcb_world.engine import KiCadEngine
        except ImportError:
            pytest.skip("kicad_rl_router module not available")

        from pcb_world.core.reward import RewardState

        board_path = str(BOARD_PATH)
        if not BOARD_PATH.exists():
            pytest.skip(f"Test board not found: {BOARD_PATH}")

        r = KiCadEngine(board_path)
        snap = r.get_reward_snapshot(run_drc=False)
        state = RewardState.from_snapshot(snap)

        assert state.unconnected >= 0
        assert state.track_count >= 0
        assert state.wirelength >= 0.0
        assert state.drc_violations == 0  # Not running DRC


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
