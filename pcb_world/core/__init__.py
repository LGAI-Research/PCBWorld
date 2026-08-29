"""KiCad Gymnasium Environment package."""

# --- New modules (canonical imports) ---
from pcb_world.core.env import PCBWorld
from pcb_world.core.masking import (
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACT_NET_END,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_FINISH,
    ACT_IDLE,
    NUM_ACTIONS,
    ACTION_NAMES,
    ACTION_REGISTRY,
    ActionDef,
    MaskContext,
    MaskingRule,
    build_action_mask,
)
from pcb_world.trajectory import TrajectoryLogger

__all__ = [
    # New
    "PCBWorld",
    "ACT_NET_SELECT",
    "ACT_START_ROUTE",
    "ACT_NET_END",
    "ACT_MAKE_LINE",
    "ACT_MAKE_VIA",
    "ACT_FINISH",
    "ACT_IDLE",
    "NUM_ACTIONS",
    "ACTION_NAMES",
    "ACTION_REGISTRY",
    "ActionDef",
    "MaskContext",
    "MaskingRule",
    "build_action_mask",
    "TrajectoryLogger",
]
