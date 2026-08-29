"""MCTS tree node."""

from __future__ import annotations

from typing import Any

from methods._shared.mcts.protocols import NodeState


class Node:
    """One state in the search tree.

    A node is *realized* once it has a checkpoint (``state``) — i.e. it has been
    stepped into. Children are created (with priors) when their parent is
    expanded, but stay unrealized (``state is None``) until first visited; their
    checkpoint is then created by restoring the parent and stepping the edge
    action. This is what keeps restore-to-node to one restore + one step per
    simulation.
    """

    __slots__ = (
        "action", "parent", "prior", "children", "state",
        "N", "W", "terminal", "terminal_value", "reward", "raw_value",
    )

    def __init__(self, action: Any = None, parent: "Node | None" = None,
                 prior: float = 0.0) -> None:
        self.action = action          # incoming edge action (None at root)
        self.parent = parent
        self.prior = prior            # P(action | parent)
        self.children: dict[Any, Node] = {}
        self.state: NodeState | None = None   # checkpoint once realized
        self.N = 0                    # visit count
        self.W = 0.0                  # total value
        self.terminal = False
        self.terminal_value = 0.0
        self.reward = 0.0             # reward of the incoming edge
        # The node's OWN value estimate at expansion time, in the same units as
        # the values backed up into its children's Q — mctx ``raw_values[node]``.
        # Consumed by the Gumbel completed-Q mixed value; None when the node was
        # never value-evaluated (horizon interiors, terminal shortcuts).
        self.raw_value: float | None = None

    @property
    def realized(self) -> bool:
        return self.state is not None

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0

    def expand(self, legal_actions, priors: dict[Any, float]) -> None:
        """Create a child per legal action, carrying its prior."""
        for a in legal_actions:
            self.children[a] = Node(action=a, parent=self,
                                    prior=float(priors.get(a, 0.0)))
