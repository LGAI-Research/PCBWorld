"""Compare dense vs sparse reward on a full 3-net routing scenario.

Routes NET1, NET2, NET3 (via) to completion with DRC=0 on the live
``PCBWorld`` (dict actions), collecting per-step rewards in both modes.
Verifies that:
  - final_potential is identical in both modes (Φ-identical rule pair
    ``drc_only_sparse`` / ``drc_only_dense``)
  - dense episode return = Phi(s_final) - Phi(s_0) - N * step_penalty
  - sparse mode pays Phi + step_penalty only on the last step
"""

import gc
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))
sys.path.insert(0, str(PROJECT_ROOT))

from pcb_world.core.action_schema import (  # noqa: E402
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACTION_NAMES,
)

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"

SPARSE_RULE = "drc_only_sparse"
DENSE_RULE = "drc_only_dense"


def _net_at(env, x, y):
    """Resolve the net code of the pad at (x, y) on the fresh board."""
    for p in env._engine.get_pads():
        if abs(p.x_mm - x) < 0.6 and abs(p.y_mm - y) < 0.6 and p.net_code > 0:
            return int(p.net_code)
    raise AssertionError(f"no pad near ({x}, {y})")


def _route_all_nets(env):
    """Route NET1, NET2, NET3 (via) — returns list of (reward, terminated, truncated, info).

    Same trajectory as tests/test_env_incremental_restore.py::_trajectory:
    lines for the two horizontal nets, then a Top→Bottom→Top via detour for
    the crossing net. Every scripted action must be accepted (not masked).
    """
    steps = []

    def do(action):
        obs, reward, terminated, truncated, info = env.step(action)
        assert info.get("action_class") not in ("mask_reject", "parse_fail"), (
            f"scripted action rejected: {action} -> {info.get('action_class')}"
        )
        steps.append((reward, terminated, truncated, info))
        return obs, reward, terminated, truncated, info

    n1, n2, n3 = _net_at(env, 10, 10), _net_at(env, 10, 20), _net_at(env, 25, 5)
    env._engine.set_via_diameter(0.6)
    env._engine.set_via_drill(0.3)

    # NET1: make_line (10,10) -> (40,10)
    do({"action_type": ACT_NET_SELECT, "net_id": n1})
    do({"action_type": ACT_START_ROUTE, "x_mm": 10.0, "y_mm": 10.0, "layer": 1})
    do({"action_type": ACT_MAKE_LINE, "x_mm": 40.0, "y_mm": 10.0, "routing_mode": 2})
    do({"action_type": ACT_NET_END})

    # NET3: via strategy (25,5) Top -> (25,5.5) Bottom -> (25,25) Top
    do({"action_type": ACT_NET_SELECT, "net_id": n3})
    do({"action_type": ACT_START_ROUTE, "x_mm": 25.0, "y_mm": 5.0, "layer": 1})
    do({"action_type": ACT_MAKE_VIA, "x_mm": 25.0, "y_mm": 5.5, "routing_mode": 2})
    do({"action_type": ACT_START_ROUTE, "x_mm": 25.0, "y_mm": 5.5, "layer": 2})
    do({"action_type": ACT_MAKE_VIA, "x_mm": 25.0, "y_mm": 25.0, "routing_mode": 2})
    do({"action_type": ACT_NET_END})

    # NET2: make_line (10,20) -> (40,20) — completes the board
    do({"action_type": ACT_NET_SELECT, "net_id": n2})
    do({"action_type": ACT_START_ROUTE, "x_mm": 10.0, "y_mm": 20.0, "layer": 1})
    do({"action_type": ACT_MAKE_LINE, "x_mm": 40.0, "y_mm": 20.0, "routing_mode": 2})

    return steps


def _run_episode(reward_rule: str):
    """Run full episode, return (steps, initial_potential).

    initial_potential is Φ(s_0) computed from the env's own RewardTracker,
    which matches how the env computes the first step's before_state.
    """
    from pcb_world.core.env import PCBWorld

    env = PCBWorld(
        board_path=BOARD, max_steps=200,
        masking_rule="default_no_finish",
        reward_rule=reward_rule,
    )
    env.reset()
    # Read the env's own cached initial state (DRC included iff mode=='per_step')
    initial_potential = env._potential_reward.potential(env._reward.prev_state)
    steps = _route_all_nets(env)
    # Fully release the native RLRouter before the next _run_episode constructs
    # one: KiCadEngine is a per-process singleton, so two live RLRouters corrupt
    # KiCad global state and segfault (close + del + gc.collect, mirroring the
    # production reload_board contract).
    env.close()
    del env
    gc.collect()
    return steps, initial_potential


class TestRewardModeComparison:

    def test_dense_vs_sparse_final_potential(self):
        """final_potential should be identical regardless of mode."""
        sparse_steps, _ = _run_episode(SPARSE_RULE)
        dense_steps, _ = _run_episode(DENSE_RULE)

        # Both should terminate
        assert sparse_steps[-1][1], "Sparse should terminate"
        assert dense_steps[-1][1], "Dense should terminate"

        sparse_pot = sparse_steps[-1][3].get("final_potential")
        dense_pot = dense_steps[-1][3].get("final_potential")

        assert sparse_pot is not None, "Sparse missing final_potential"
        assert dense_pot is not None, "Dense missing final_potential"
        assert sparse_pot == pytest.approx(dense_pot, abs=0.01), (
            f"final_potential mismatch: sparse={sparse_pot}, dense={dense_pot}"
        )

    def test_telescope_property(self):
        """Dense sum should equal Phi(s_final) - Phi(s_0) - N * step_penalty."""
        from pcb_world.core.reward_config import get_reward_config

        dense_steps, phi_s0 = _run_episode(DENSE_RULE)
        cfg = get_reward_config(DENSE_RULE)
        pr = cfg.build_reward()

        dense_total = sum(r for r, _, _, _ in dense_steps)
        final_pot = dense_steps[-1][3]["final_potential"]
        n_steps = len(dense_steps)

        expected = final_pot - phi_s0 - n_steps * pr.step_penalty

        assert dense_total == pytest.approx(expected, abs=0.05), (
            f"Telescope failed: dense_total={dense_total:.4f}, "
            f"expected={expected:.4f}, Phi(s_0)={phi_s0:.2f}"
        )

    def test_telescope_property_with_drc(self):
        """Telescope should hold with DRC-inclusive dense config (drc_linear)."""
        from pcb_world.core.reward_config import get_reward_config

        dense_steps, phi_s0 = _run_episode("drc_linear")
        cfg = get_reward_config("drc_linear")
        pr = cfg.build_reward()

        dense_total = sum(r for r, _, _, _ in dense_steps)
        final_pot = dense_steps[-1][3]["final_potential"]
        n_steps = len(dense_steps)

        expected = final_pot - phi_s0 - n_steps * pr.step_penalty

        assert dense_total == pytest.approx(expected, abs=0.05), (
            f"Telescope (DRC) failed: dense_total={dense_total:.4f}, "
            f"expected={expected:.4f}, Phi(s_0)={phi_s0:.2f}"
        )

    def test_final_only_step_penalty_on_last_step(self):
        """Final-only (sparse) mode: last step should include step_penalty."""
        from pcb_world.core.reward_config import get_reward_config

        steps, _ = _run_episode(SPARSE_RULE)
        cfg = get_reward_config(SPARSE_RULE)
        pr = cfg.build_reward()

        assert steps[-1][1], "Should terminate"
        final_pot = steps[-1][3]["final_potential"]
        last_reward = steps[-1][0]

        # Last step reward = Phi(s_final) - step_penalty
        assert last_reward == pytest.approx(
            final_pot - pr.step_penalty, abs=0.01
        ), (
            f"Last step reward={last_reward:.4f}, "
            f"expected Phi(s_final)-step_penalty={final_pot - pr.step_penalty:.4f}"
        )


def main():
    """Print detailed reward comparison for both modes."""
    print("=" * 70)
    print("REWARD MODE COMPARISON: Dense vs Sparse")
    print("Board: simple_routing_board (3 nets, 6 ratsnest edges)")
    print("=" * 70)

    for mode_name, rule in [("SPARSE", SPARSE_RULE),
                            ("DENSE", DENSE_RULE)]:
        steps, initial_potential = _run_episode(rule)

        print(f"\n--- {mode_name} (reward_rule={rule}) ---")
        print(f"{'Step':>4}  {'Action':>12}  {'Reward':>10}  {'Term':>5}  {'Info'}")
        print("-" * 65)

        cumulative = 0.0
        for i, (reward, terminated, truncated, info) in enumerate(steps):
            act_type = info.get("action_type", -1)
            act_name = ACTION_NAMES[act_type] if 0 <= act_type < len(ACTION_NAMES) else "?"
            cumulative += reward
            extra = ""
            if terminated or truncated:
                pot = info.get("final_potential", "N/A")
                drc = info.get("drc_violations", "N/A")
                extra = f"  final_pot={pot}, drc={drc}"
            print(f"{i+1:>4}  {act_name:>12}  {reward:>10.4f}  {str(terminated):>5}{extra}")

        print("-" * 65)
        print(f"Episode return: {cumulative:.4f}")
        print(f"N steps: {len(steps)}")
        final_info = steps[-1][3]
        print(f"Final potential: {final_info.get('final_potential', 'N/A')}")
        print(f"DRC violations: {final_info.get('drc_violations', 'N/A')}")
        print(f"Terminated: {steps[-1][1]}")

    # Summary comparison
    sparse_steps, sparse_phi_s0 = _run_episode(SPARSE_RULE)
    dense_steps, dense_phi_s0 = _run_episode(DENSE_RULE)
    sparse_total = sum(r for r, _, _, _ in sparse_steps)
    dense_total = sum(r for r, _, _, _ in dense_steps)
    sparse_pot = sparse_steps[-1][3].get("final_potential", 0)
    dense_pot = dense_steps[-1][3].get("final_potential", 0)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'':>20}  {'Sparse':>12}  {'Dense':>12}")
    print(f"{'Episode return':>20}  {sparse_total:>12.4f}  {dense_total:>12.4f}")
    print(f"{'Final potential':>20}  {sparse_pot:>12.4f}  {dense_pot:>12.4f}")
    print(f"{'N steps':>20}  {len(sparse_steps):>12}  {len(dense_steps):>12}")
    print(f"{'Phi(s_0)':>20}  {sparse_phi_s0:>12.4f}")
    print(f"{'Difference':>20}  {dense_total - sparse_total:>12.4f}  (dense - sparse)")


if __name__ == "__main__":
    main()
