"""Branch-agnostic MCTS core.

Public API::

    from methods._shared.mcts import run_search, MctsConfig, SearchEnv, PolicyValueFn, NodeState

A branch provides a ``SearchEnv`` (env + L2 snapshot) and a ``PolicyValueFn``
(its policy); the core does the tree search via per-node checkpoint/restore.

Instead of hand-wiring ``MctsConfig`` flags, pick a named algorithm profile —
each bundles the coherent module set of a published implementation (see
``algorithms.py`` for the axis-by-axis comparison)::

    from methods._shared.mcts import AlphaZero, MuZero, GumbelMuZero, make_algorithm
    cfg = GumbelMuZero(n_simulations=100, seed=0).config()

The RL branch adapter lives in ``methods.rl_agent.policy.mcts_env`` and is
imported separately (it pulls in torch / kicad, so it is kept out of this core
package).
"""

from methods._shared.mcts.algorithms import (
    ALGORITHMS,
    AlphaZero,
    GumbelMuZero,
    MuZero,
    SearchAlgorithm,
    make_algorithm,
)
from methods._shared.mcts.config import DEFAULT_SEARCH_GAMMA, MctsConfig
from methods._shared.mcts.node import Node
from methods._shared.mcts.protocols import NodeState, PolicyValueFn, SearchEnv, StepResult
from methods._shared.mcts.search import run_search, search_iter

__all__ = [
    "ALGORITHMS",
    "AlphaZero",
    "DEFAULT_SEARCH_GAMMA",
    "GumbelMuZero",
    "MctsConfig",
    "MuZero",
    "Node",
    "NodeState",
    "PolicyValueFn",
    "SearchAlgorithm",
    "SearchEnv",
    "StepResult",
    "make_algorithm",
    "run_search",
    "search_iter",
]
