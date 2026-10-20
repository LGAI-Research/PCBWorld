"""Shape & action-space contract for the v1 model variant.

Single source of truth for the dimensions / enums shared by the codec
(``encoding``), the network (``net``), and the rollout buffer, so shape drift
across those three cannot happen silently — including the row bundles
:class:`DistInputs` (everything that shapes the action distribution besides
the obs) and :class:`Transition` (one buffer row), which carry those inputs
from the act-time gather through the buffer to the update in one piece.

C1-models step 1: these definitions were *relocated* here from
``policy/token_vocabulary.py`` (vocab enums + counts) and
``policy/decoder_only_policy.py`` (action constants). Those modules now
re-export from here, so existing imports keep working unchanged (behavior 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import IntEnum
from typing import Any

import numpy as np
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


# ---------------------------------------------------------------------------
# Row bundles — distribution inputs and buffer rows
# ---------------------------------------------------------------------------
# A field holds a row-aligned array: numpy in the buffer, torch in a forward.
Array = Any

# Cand-column index lists are stored int32 (half the buffer and DDP pickle of
# int64); torch's scatter / advanced indexing take int32 indices as they are.
_CAND_IDX = np.int32


def _col(pad: Any = None) -> Any:
    """A bundle field. ``pad`` is the fill for a ragged trailing axis (rows of
    different width are right-padded with it on :meth:`_RowBundle.concat`);
    ``None`` = fixed width or a nested bundle. No default: every field is
    required at construction."""
    return field(metadata={"pad": pad})


def _pad_rows(rows: list, fill: Any, dtype: Any) -> np.ndarray:
    """Right-pad 1-D rows into one ``(len(rows), max_len)`` array of ``fill``."""
    out = np.full((len(rows), max((len(r) for r in rows), default=0)), fill, dtype=dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


class _RowBundle:
    """Row ops over a frozen dataclass of row-aligned fields, driven by
    ``dataclasses.fields()`` — a field added to the dataclass rides every op
    without another edit. A field may be ``None`` (absent for the whole
    bundle) or a nested bundle."""

    def select(self, idx: Any) -> Any:
        """Rows ``idx`` (index list / tensor; keeps the row axis)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = None if v is None else (
                v.select(idx) if isinstance(v, _RowBundle) else v[idx])
        return type(self)(**out)

    def to(self, device: Any) -> Any:
        """Every array as a torch tensor on ``device`` (dtype kept)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = None if v is None else (
                v.to(device) if isinstance(v, _RowBundle) else torch.as_tensor(v, device=device))
        return type(self)(**out)

    @classmethod
    def concat(cls, parts: list) -> Any:
        """Stack numpy bundles row-wise in order; ragged fields are padded with
        their ``pad`` fill. A field must be ``None`` in all parts or in none."""
        if not parts:
            raise ValueError(f"{cls.__name__}.concat needs at least one part")
        out = {}
        for f in fields(cls):
            vals = [getattr(p, f.name) for p in parts]
            if all(v is None for v in vals):
                out[f.name] = None
            elif any(v is None for v in vals):
                raise ValueError(f"{cls.__name__}.concat: field {f.name!r} is None in some parts only")
            elif isinstance(vals[0], _RowBundle):
                out[f.name] = type(vals[0]).concat(vals)
            elif f.metadata["pad"] is None:
                out[f.name] = np.concatenate(vals)
            else:
                out[f.name] = _pad_rows([r for v in vals for r in v], f.metadata["pad"], vals[0].dtype)
        return cls(**out)


@dataclass(frozen=True, eq=False)
class DistInputs(_RowBundle):
    """Everything besides the obs that shapes the action distribution, one row
    per batch row. The act-time gather builds it, the buffer stores it, and the
    update hands the stored bundle back to the model — so the rollout and the
    update score the same distribution by construction."""

    action: Array = _col()              # (B, NUM_ACTIONS) bool, True = legal action type
    pointer: Array = _col(pad=-1)       # (B, K) int32 cand columns blocked on every row:
                                        #   same-point ++ off-board, right-padded -1
    offlayer: Array = _col(pad=-1)      # (B, O) int32 cand columns blocked on the make_line row
    mode: Array = _col()                # (B, NUM_ROUTING_MODES) bool, True = legal mode
    net_valid: Array | None = _col(pad=False)   # (B, M) bool; None iff the policy does
                                                #   not select nets (policy_net_select off)

    @classmethod
    def from_payloads(cls, payloads: list[dict], *, policy_net_select: bool) -> "DistInputs":
        """Stack per-env mask payloads (the six-key dict ``KiCadRLWrapper``
        attaches as ``obs["_masks"]``). A missing key raises ``KeyError``."""
        return cls(
            action=np.stack([p["action"] for p in payloads]).astype(bool),
            pointer=_pad_rows(
                [np.concatenate([p["pointer"], p["offboard"]]) for p in payloads], -1, _CAND_IDX,
            ),
            offlayer=_pad_rows([p["offlayer"] for p in payloads], -1, _CAND_IDX),
            mode=np.stack([p["mode"] for p in payloads]).astype(bool),
            net_valid=(
                _pad_rows([p["net_valid"] for p in payloads], False, bool)
                if policy_net_select else None
            ),
        )


@dataclass(frozen=True, eq=False)
class Transition(_RowBundle):
    """One flat replay-buffer row set: the distribution inputs plus what the
    update re-scores against (PPO and GRPO share it; advantages/returns are
    derived later and live beside it)."""

    dist: DistInputs = _col()
    actions: Array = _col()             # (B, ACTION_ARITY) int64
    old_log_probs: Array = _col()       # (B,) float32, act-time joint log-prob
    old_values: Array = _col()          # (B,) float32, act-time critic value
    kl_mask: Array | None = _col()      # (B,) float32 reference-KL weight; None unless
                                        #   fine-tuning masks the KL (loop.py)
