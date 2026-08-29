"""Cross-module constant consistency guards — the single home for
"these constants, defined in different modules, MUST agree".

This is the automated backstop for the repo's single-source-of-truth convention
(definitions live with their owner: ``pcb_world/core/action_schema`` for the action set
+ routing-mode encoding, ``pcb_world/engine/drc`` for DRC severity/type taxonomy,
``models/v1/spec`` for model-shape contracts, ``pcb_world/vec/slots`` for slot counts).

Prefer to **reference** a canonical directly. Only keep a local mirror when an
import boundary / cheap-import design / C++-vs-Python split forbids importing it
— and when you do, add the parity assertion HERE so drift is caught by the suite.

All imports below are pure Python (no C++ router): ``pcb_world.engine.drc`` imports the
router lazily, so its module-level constants are import-safe without it.
"""
from __future__ import annotations


def test_drc_type_bucket_count_matches_engine() -> None:
    """``models/v1/spec.NUM_DRC_TYPES`` (DRC-type embedding table size) must equal
    the engine's DRC type-bucket count. If the engine taxonomy grows a bucket but
    this copy lags, the DRC-type embedding is silently under-sized."""
    from pcb_world.engine.drc import DRC_NUM_TYPE_BUCKETS
    from methods.rl_agent.models.v1.spec import NUM_DRC_TYPES

    assert NUM_DRC_TYPES == DRC_NUM_TYPE_BUCKETS


def test_eval_metrics_drc_mirrors_match_canonical() -> None:
    """``eval/metrics.py`` keeps local DRC severity/promoted mirrors on purpose —
    it keeps KiCad imports function-local so importing the module stays cheap
    (see its module docstring). Pin those mirrors to ``pcb_world/engine/drc`` here."""
    from pcb_world.engine.drc import (
        DRC_SEVERITY_ERROR,
        DRC_SEVERITY_WARNING,
        _PROMOTED_ERROR_CODES,
    )
    from eval.metrics import _PROMOTED_CODES, _SEV_ERROR, _SEV_WARNING

    assert _SEV_ERROR == DRC_SEVERITY_ERROR
    assert _SEV_WARNING == DRC_SEVERITY_WARNING
    assert _PROMOTED_CODES == set(_PROMOTED_ERROR_CODES)


def test_codec_drc_severity_matches_canonical() -> None:
    """The codec (``models/v1/tokenizer``) imports the canonical directly; assert
    the alias still resolves to the same value (guards an accidental re-copy)."""
    from pcb_world.engine.drc import DRC_SEVERITY_ERROR
    from methods.rl_agent.models.v1.tokenizer import _DRC_SEVERITY_ERROR

    assert _DRC_SEVERITY_ERROR == DRC_SEVERITY_ERROR


def test_reward_keyword_categories_cover_exactly_the_canonical_keywords() -> None:
    """``reward._KEYWORD_TO_CATEGORY`` must stay in 1:1 sync with the owner
    ``drc.VIOLATION_KEYWORDS``. Addition drift already KeyErrors at import (the
    map derivation looks each keyword up); this pins the OTHER direction — a
    keyword *removed* from the owner would otherwise leave a silent stale entry."""
    from pcb_world.core.reward import _KEYWORD_TO_CATEGORY
    from pcb_world.engine.drc import VIOLATION_KEYWORDS

    assert set(_KEYWORD_TO_CATEGORY) == set(VIOLATION_KEYWORDS)


def test_selectable_action_names_derivation_assumes_idle_last() -> None:
    """The selectable-action lists (``training/loop._ACTION_NAMES``,
    ``rollout/transformer.TRACE_ACTION_NAMES``, experiments' plan-only prep) derive
    as ``ACTION_NAMES[:ACT_IDLE]`` — which is only "all selectable actions" while
    ``idle`` is the LAST registry entry. If an action were ever appended after
    idle, the slice would silently drop it; pin the structural assumption here."""
    from pcb_world.core.action_schema import ACT_IDLE, ACTION_NAMES, NUM_ACTIONS

    assert ACT_IDLE == NUM_ACTIONS - 1, "idle must stay last in ACTION_REGISTRY"
    assert ACTION_NAMES[:ACT_IDLE] == [n for n in ACTION_NAMES if n != "idle"]
