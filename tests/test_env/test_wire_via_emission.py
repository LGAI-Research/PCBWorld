"""Tests for the ``wire_via_emission`` reward mode (Phase C.4).

Two hypotheses under test:

1. **Per-step mode** is the legacy behaviour — every step emits the full
   wire + via delta.
2. **On-net-end mode** holds wire/via back and emits the accumulated delta
   on ``ACT_NET_END`` / episode-completion steps. Crucially, the
   *total* reward summed across a trajectory matches the per-step total
   (conservation), because both modes are discounted telescoping sums of
   the same potential.

These tests use the pure-Python :class:`PotentialReward` directly — no
C++ engine, no env wrapper. Trajectories are synthesized as lists of
``RewardState`` snapshots so we can compare the two modes on the exact
same state sequence.
"""

from __future__ import annotations

import pytest

from pcb_world.core.reward import PotentialReward, RewardState


def _mk_state(
    *,
    unconnected: int,
    wirelength: float = 0.0,
    via_count: int = 0,
    drc_violations: int = 0,
) -> RewardState:
    """Compact ``RewardState`` constructor for synthetic trajectories."""
    return RewardState(
        unconnected=unconnected,
        drc_violations=drc_violations,
        wirelength=wirelength,
        track_count=0,
        via_count=via_count,
        drc_errors=drc_violations,
    )


@pytest.fixture
def reward_kwargs() -> dict:
    """Non-zero weights on every term so the mode matters."""
    return dict(
        completion_bonus=5.0,
        unconnected_penalty=1.0,
        wirelength_penalty=0.01,
        via_penalty=0.5,
        drc_penalty=0.0,  # keep DRC out of this test
        step_penalty=0.05,
        truncation_mode="none",
    )


def _trajectory() -> list[tuple[bool, RewardState]]:
    """A 2-net episode. Each row = (is_net_end, after_state).

    Timeline:
      step 0: initial → start_route net A (no wire yet)
      step 1: make_line on A (+5 mm wire)
      step 2: make_via on A (+1 via)
      step 3: make_line on A (+3 mm wire)
      step 4: net_end A  -> 1 net done, unconnected drops from 2 to 1
      step 5: start_route net B
      step 6: make_line on B (+4 mm wire)
      step 7: net_end B  -> unconnected == 0 (all routed)
    """
    return [
        # (is_net_end, state)
        (False, _mk_state(unconnected=2, wirelength=0.0, via_count=0)),
        (False, _mk_state(unconnected=2, wirelength=5.0, via_count=0)),
        (False, _mk_state(unconnected=2, wirelength=5.0, via_count=1)),
        (False, _mk_state(unconnected=2, wirelength=8.0, via_count=1)),
        (True,  _mk_state(unconnected=1, wirelength=8.0, via_count=1)),
        (False, _mk_state(unconnected=1, wirelength=8.0, via_count=1)),
        (False, _mk_state(unconnected=1, wirelength=12.0, via_count=1)),
        (True,  _mk_state(unconnected=0, wirelength=12.0, via_count=1)),
    ]


def _simulate_per_step(pr: PotentialReward, traj) -> list[float]:
    """Run the per-step compute_dense over the trajectory, return rewards."""
    rewards: list[float] = []
    prev = traj[0][1]
    for _, curr in traj[1:]:
        rewards.append(pr.compute_dense(prev, curr))
        prev = curr
    return rewards


def _simulate_on_net_end(pr: PotentialReward, traj) -> list[float]:
    """Run the on_net_end variant over the trajectory, return rewards."""
    rewards: list[float] = []
    prev = traj[0][1]
    ref = traj[0][1]  # seed ref at initial state
    for is_net_end, curr in traj[1:]:
        flush = is_net_end or curr.unconnected == 0
        r, ref = pr.compute_dense_netend(prev, curr, ref, flush_wire_via=flush)
        rewards.append(r)
        prev = curr
    return rewards


def test_per_step_is_the_legacy_reward(reward_kwargs):
    """per_step mode (the default) must match the plain compute_dense sum."""
    pr_a = PotentialReward(wire_via_emission="per_step", **reward_kwargs)
    pr_b = PotentialReward(**reward_kwargs)  # default is per_step
    traj = _trajectory()
    assert _simulate_per_step(pr_a, traj) == _simulate_per_step(pr_b, traj)


def test_on_net_end_total_matches_per_step_total(reward_kwargs):
    """Conservation: summing rewards across the episode is mode-invariant.

    Both modes are discounted telescoping sums of the same Φ; with no
    discount (undiscounted sum), they must agree exactly.
    """
    pr = PotentialReward(**reward_kwargs)
    traj = _trajectory()
    per_step_total = sum(_simulate_per_step(pr, traj))
    net_end_total = sum(_simulate_on_net_end(pr, traj))
    assert per_step_total == pytest.approx(net_end_total, abs=1e-9)


def test_on_net_end_zero_wire_via_on_non_flush_steps(reward_kwargs):
    """Non-flush steps in on_net_end mode must NOT contain wire/via deltas.

    Because Φ_wire_via is held back, the reward delta between two
    consecutive non-flush steps should equal the ``_phi_main`` delta
    minus step_penalty (no wire/via contribution).
    """
    pr = PotentialReward(**reward_kwargs)
    traj = _trajectory()
    on_rewards = _simulate_on_net_end(pr, traj)

    # Steps 1..3 (indexes 0..2 in on_rewards) are all non-flush.
    # Their rewards should equal _phi_main delta - step_penalty only.
    prev = traj[0][1]
    for i in range(3):
        curr = traj[i + 1][1]
        expected_main_delta = pr._phi_main(curr) - pr._phi_main(prev)  # noqa: SLF001
        expected = expected_main_delta - pr.step_penalty
        assert on_rewards[i] == pytest.approx(expected, abs=1e-9), (
            f"Step {i}: expected {expected}, got {on_rewards[i]}"
        )
        prev = curr


def test_invalid_wire_via_emission_raises():
    with pytest.raises(ValueError, match="wire_via_emission"):
        PotentialReward(wire_via_emission="bogus")


def test_cli_override_maps_to_attribute():
    """Constructor-level acceptance of the new field."""
    pr = PotentialReward(wire_via_emission="on_net_end")
    assert pr.wire_via_emission == "on_net_end"
    pr2 = PotentialReward()  # default
    assert pr2.wire_via_emission == "per_step"
