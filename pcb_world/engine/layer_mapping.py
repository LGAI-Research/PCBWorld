"""Human-friendly layer mapping for KiCad PCB boards.

Converts between human layer IDs (1-indexed, top-down) and KiCad board
layer IDs (PCB_LAYER_ID enum values).

Human layer convention (matches PCB industry standard):
    Layer 1 = Top copper (F.Cu)
    Layer 2 = First inner copper (or Bottom if 2-layer)
    ...
    Layer N = Bottom copper (B.Cu)

KiCad PCB_LAYER_ID values:
    F_Cu = 0, B_Cu = 2, In1_Cu = 4, In2_Cu = 6, ...

Usage:
    mapping = LayerMapping(copper_layer_count=4)
    board_id = mapping.human_to_board(1)   # → 0 (F_Cu)
    board_id = mapping.human_to_board(4)   # → 2 (B_Cu)
    human_id = mapping.board_to_human(2)   # → 4 (B_Cu → Layer 4)
"""

from __future__ import annotations

# KiCad PCB_LAYER_ID constants
_F_CU = 0
_B_CU = 2
_IN1_CU = 4  # Inner layers start at 4, step by 2


def _build_board_layer_order(copper_layer_count: int) -> list[int]:
    """Build ordered list of board layer IDs from top to bottom.

    Args:
        copper_layer_count: Number of copper layers (2, 4, 6, ...).

    Returns:
        List of KiCad PCB_LAYER_ID values, ordered top to bottom.
        Example for 4-layer: [0, 4, 6, 2]  (F_Cu, In1_Cu, In2_Cu, B_Cu)
    """
    if copper_layer_count < 2:
        copper_layer_count = 2

    layers = [_F_CU]

    for i in range(copper_layer_count - 2):
        layers.append(_IN1_CU + i * 2)

    layers.append(_B_CU)

    return layers


class LayerMapping:
    """Bidirectional mapping between human layer IDs and KiCad board layer IDs.

    Human layers are 1-indexed (1 = Top, N = Bottom).
    Board layers use KiCad's PCB_LAYER_ID enum values.

    Args:
        copper_layer_count: Number of copper layers on the board (2, 4, 6, ...).
    """

    def __init__(self, copper_layer_count: int) -> None:
        self._board_order = _build_board_layer_order(copper_layer_count)
        self._human_to_board: dict[int, int] = {}
        self._board_to_human: dict[int, int] = {}

        for i, board_id in enumerate(self._board_order):
            human_id = i + 1  # 1-indexed
            self._human_to_board[human_id] = board_id
            self._board_to_human[board_id] = human_id

    def human_to_board(self, human_layer: int) -> int:
        """Convert human layer (1-indexed) to KiCad board layer ID.

        Raises:
            KeyError: If human_layer is out of range.
        """
        if human_layer not in self._human_to_board:
            raise KeyError(
                f"Human layer {human_layer} out of range [1, {self.max_layer}]"
            )
        return self._human_to_board[human_layer]

    def board_to_human(self, board_layer: int) -> int:
        """Convert KiCad board layer ID to human layer (1-indexed).

        Raises:
            KeyError: If board_layer is not a copper layer on this board.
        """
        if board_layer not in self._board_to_human:
            raise KeyError(
                f"Board layer {board_layer} is not a copper layer on this board. "
                f"Valid: {self._board_order}"
            )
        return self._board_to_human[board_layer]

    @property
    def max_layer(self) -> int:
        """Number of copper layers (= max human layer ID)."""
        return len(self._board_order)

    @property
    def board_layer_order(self) -> list[int]:
        """Board layer IDs ordered from top to bottom."""
        return list(self._board_order)

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{h}→{b}" for h, b in sorted(self._human_to_board.items())
        )
        return f"LayerMapping({pairs})"
