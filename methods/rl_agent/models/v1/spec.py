"""Shape & action-space contract for the v1 model variant.

Single source of truth for the dimensions / enums shared by the codec
(``encoding``), the network (``net``), and the rollout buffer, so shape drift
across those three cannot happen silently.

C1-models step 1: these definitions were *relocated* here from
``policy/token_vocabulary.py`` (vocab enums + counts) and
``policy/decoder_only_policy.py`` (action constants). Those modules now
re-export from here, so existing imports keep working unchanged (behavior 0).
"""

from __future__ import annotations

from enum import IntEnum

import torch

# Single source of truth in pcb_world/vec (shared with the slot_perm augmentation).
from pcb_world.vec.slots import N_MAX_SLOTS  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# Geometry / normalization
# ---------------------------------------------------------------------------
MAX_COPPER = 32

# Fixed normalization denominator for the action-history age scalar
# (age / MAX_HISTORY, like layer / MAX_COPPER above): a frozen codec constant,
# NOT the tunable window K (action_history_len). Dividing by a constant keeps
# "k steps ago" encoded identically for every K <= MAX_HISTORY, so K changes
# stay checkpoint-compatible (the age path has no K-shaped parameters).
MAX_HISTORY = 32


# ---------------------------------------------------------------------------
# Structural / control tokens
# ---------------------------------------------------------------------------
class StructuralToken(IntEnum):
    """Sequence delimiters / control tokens."""

    VAL = 0       # value-prediction token (critic reads from here)
    SOD = 1       # start-of-decode (action generation begins)
    SEQ_PAD = 2   # padding for batch alignment


NUM_STRUCTURAL_TOKENS = len(StructuralToken)


# ---------------------------------------------------------------------------
# Per-entity token identity
# ---------------------------------------------------------------------------
class EntityType(IntEnum):
    """Identity of the single-token-per-entity classes."""

    BOARD = 0
    EDGE = 1
    NET = 2
    PAD = 3
    TRACK = 4
    VIA = 5
    RAT = 6
    HEAD = 7
    CAND_PAD = 8
    CAND_TRACK_END = 9
    CAND_VIA = 10
    CAND_RAT = 11
    CAND_DIR = 12
    DRC_VIOLATION = 13
    # Netless immovable blockers (NPTH mounting holes / slots + net-less NC
    # pads). Appended LAST: earlier ids are baked into checkpoints. Emitted
    # only when the obstacle_obs knob is on; pre-knob checkpoints carry a
    # 14-row entity table that the loader pads (row never indexed knob-off).
    OBSTACLE = 14


NUM_ENTITY_TYPES = len(EntityType)

# Taxonomy buckets for DRC violation error types (see pcb_world/engine drc.py:
# classify_violation_type_id). Kept here so the embedding table size stays in
# sync with the taxonomy; pinned to pcb_world.engine.drc.DRC_NUM_TYPE_BUCKETS by
# tests/test_constant_consistency.py::test_drc_type_bucket_count_matches_engine.
NUM_DRC_TYPES = 7

# Pad/obstacle boundary-shape buckets for the shape_obs channel. Bucket ids are
# a checkpoint contract (shape_embed rows); string→id mapping lives in
# encoding.shape_bucket_id. "other" folds the rare families (trapezoid,
# chamfered_rect, custom — <1% of real pads); "unknown" is the "" default.
NUM_SHAPE_BUCKETS = 6


class CandidateType(IntEnum):
    PAD_POINT = 0
    TRACK_ENDPOINT = 1
    VIA_CENTER = 2
    RATSNEST = 3
    DIRECTIONAL = 4


NUM_CAND_TYPES = len(CandidateType)


_CAND_TO_ENTITY = {
    CandidateType.PAD_POINT.value: EntityType.CAND_PAD,
    CandidateType.TRACK_ENDPOINT.value: EntityType.CAND_TRACK_END,
    CandidateType.VIA_CENTER.value: EntityType.CAND_VIA,
    CandidateType.RATSNEST.value: EntityType.CAND_RAT,
    CandidateType.DIRECTIONAL.value: EntityType.CAND_DIR,
}


def cand_type_to_entity(cand_type: int) -> int:
    """Map a :class:`CandidateType` integer to its :class:`EntityType` int."""
    return int(_CAND_TO_ENTITY[int(cand_type)])


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
# Single source of truth = the env's ACTION_REGISTRY (registry order → index).
# Re-exported here (not re-numbered) so the policy/buffer view cannot drift from
# the env's action space.
from pcb_world.core.action_schema import (  # noqa: E402,F401
    ACTION_REGISTRY,
    MODE_LETTER_TO_INT,
    NUM_ACTIONS,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACT_NET_END,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_FINISH,
    ACT_IDLE,
)

# Policy/buffer naming for the same count (registry length).
NUM_ACTION_TYPES = NUM_ACTIONS

# Width of the policy action triple: (action_type, pointer_idx, routing_mode).
ACTION_ARITY = 3

# Number of routing modes = size of the routing-mode encoding — single source
# is pcb_world/core/action_schema.MODE_LETTER_TO_INT (m/p/w → 0/1/2).
NUM_ROUTING_MODES = len(MODE_LETTER_TO_INT)

# Slot usage per action_type: [needs_pointer, needs_mode], DERIVED from
# ACTION_REGISTRY params (single source — #2 resolved) so it cannot drift from
# the action space. A slot is used iff the action's params reference it, in
# registry order (== action index):
#   needs_pointer ⟺ "net_id" (net pointer) or "x_mm" (cand pointer) in params
#     → action consumes a pointer (net or cand) in slot[1]
#   needs_mode    ⟺ "routing_mode" in params
#     → action consumes a routing_mode in slot[2]
SLOT_USAGE = torch.tensor(
    [
        ["net_id" in a.params or "x_mm" in a.params, "routing_mode" in a.params]
        for a in ACTION_REGISTRY
    ],
    dtype=torch.bool,
)  # (NUM_ACTIONS, 2)
