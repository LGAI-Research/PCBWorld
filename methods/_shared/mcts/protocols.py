"""Contracts the MCTS core depends on — branch-agnostic.

The core never imports an env, a wrapper, or a policy: it talks only to these
Protocols. The RL and LLM branches each provide a ``SearchEnv`` (wrapping their
env + L2 snapshot) and a ``PolicyValueFn`` (their policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass
class NodeState:
    """Checkpoint bundle for one MCTS node.

    ``l1`` is the env checkpoint (board + config + session + episode scalars —
    ``PCBWorld.Checkpoint``). ``l2`` is the branch-specific path-dependent
    state that L1 does not cover (LLM memory / RL rng+start), or ``None`` for a
    branch with no L2. The two are captured and restored in lockstep.
    """

    l1: Any
    l2: Any = None


@dataclass
class StepResult:
    """Outcome of advancing the env by one action.

    ``reward`` is the value the search accumulates (for ``return_bootstrap`` this
    should be the potential delta ΔΦ, NOT the raw env reward — so it telescopes
    onto Φ and excludes per-step / invalid penalties). ``invalid`` marks a
    meaningless step (the branch's action failed / changed nothing): the core
    treats such a child as a penalized dead-end leaf and does not expand it.
    """

    reward: float
    done: bool
    info: dict
    invalid: bool = False


@runtime_checkable
class SearchEnv(Protocol):
    """An env the MCTS core can plan over. Wraps a branch's env + L2 snapshot."""

    def checkpoint(self) -> NodeState:
        """Capture L1 + L2 for the current state (lockstep)."""

    def restore(self, state: NodeState) -> None:
        """Restore L1 + L2 from ``state`` (lockstep). Should use the fast
        incremental L1 path."""

    def release(self, state: NodeState) -> None:
        """Free the L1 handle held by ``state`` (memory reaping)."""

    def step(self, action: Any) -> StepResult:
        """Advance the env by ``action``."""

    def legal_actions(self) -> Sequence[Any]:
        """Actions allowed at the current state (from the action mask)."""

    def potential(self) -> float:
        """Board-derived value Φ(board) at the current state (§7)."""

    def observe(self) -> Any:
        """Current observation, passed to the policy/value function."""

    # Optional — consulted ONLY by the per-decision search telemetry (the
    # ``env._search_diag`` SOLVED counter). An env that omits it simply reports 0
    # completions reached; it has no effect on the search result.
    def solved(self) -> bool:
        """True when the board is fully routed (unconnected == 0)."""


@runtime_checkable
class PolicyValueFn(Protocol):
    """Branch policy: priors over legal actions (+ optional value estimate).

    Returns ``(priors, value)`` where ``priors`` maps each legal action to a
    probability (need not be normalized — the core normalizes), and ``value`` is
    a critic estimate or ``None`` (then the core uses ``SearchEnv.potential``).
    """

    def __call__(
        self, obs: Any, legal_actions: Sequence[Any],
    ) -> tuple[dict[Any, float], float | None]:
        ...


