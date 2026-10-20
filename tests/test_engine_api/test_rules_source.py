"""Verify ``rules_source`` (``EnvConfig.rules_source``) on apply_default_drc_if_fallback
and the env.

It decides what happens when the board triggers the "real
default" fallback (both ``was_project_loaded_from_file`` and
``was_legacy_design_settings_loaded`` return False):

  - ``board_or_yaml``: push the YAML's global minima into BDS so DRC
    runs against a meaningful floor (with a ``UserWarning`` because
    every dataset we ship should already carry rules), and return the
    substitution record.
  - ``board``: raise ``ValueError``. Strict mode — refuses to
    silently route on KiCad compile-time defaults.

Regardless of the value, boards that already carry authoritative rules
(a pro companion on disk OR legacy KiCad 5 setup tokens) must be left
untouched (no warning, no error).

Tests use ``kicad_rl_router.RLRouter`` directly: the ``KiCadEngine``
load contract refuses the orphan/legacy fixtures these tests construct
on purpose to reach each provenance state.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from configs.loader.schema import RULES_SOURCES, EnvConfig
from eval.eval_utils import runtime_metrics_from_info
from pcb_world.engine.drc_config import (
    DEFAULT_DRC_CONFIG_PATH, apply_default_drc_if_fallback, design_rules_digest, load_drc_config,
)
from tests.helpers.engine import open_engine
from tests.test_engine_api.conftest import FIXTURES_DIR, MODERN_PCB_FIXTURE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODERN_PRO_FIXTURE = MODERN_PCB_FIXTURE.with_suffix(".kicad_pro")

# Legacy KiCad 5 pcb with (setup ...) + (net_class ...) tokens — triggers
# the ``was_legacy_design_settings_loaded==True`` path at load.
LEGACY_PCB_FIXTURE = FIXTURES_DIR / "crossover_legacy.kicad_pcb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mirrors the dict form RLRouter.DesignRules exposes so snapshots can
# be compared field-by-field without pybind objects in the mix.
_GLOBAL_MINIMA_FIELDS = (
    "min_clearance_mm",
    "min_track_width_mm",
    "min_via_diameter_mm",
    "min_through_hole_mm",
    "min_via_annular_width_mm",
    "min_hole_to_hole_mm",
    "min_uvia_diameter_mm",
    "min_uvia_drill_mm",
    "copper_edge_clearance_mm",
)


def _global_minima_snapshot(rules) -> dict[str, float]:
    return {f: float(getattr(rules, f)) for f in _GLOBAL_MINIMA_FIELDS}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paired_modern_pcb(tmp_path: Path) -> Path:
    """Copy the KiCad 9 pcb AND its pro companion into tmp_path.

    The loader finds the pro → ``was_project_loaded_from_file==True`` →
    no fallback branch runs regardless of ``rules_source``.
    """
    if not MODERN_PCB_FIXTURE.exists() or not MODERN_PRO_FIXTURE.exists():
        pytest.skip(f"Fixture pair not found: {MODERN_PCB_FIXTURE} / .kicad_pro")
    pcb = tmp_path / MODERN_PCB_FIXTURE.name
    pro = tmp_path / MODERN_PRO_FIXTURE.name
    shutil.copy(MODERN_PCB_FIXTURE, pcb)
    shutil.copy(MODERN_PRO_FIXTURE, pro)
    return pcb


@pytest.fixture
def legacy_pcb(tmp_path: Path) -> Path:
    """Copy the KiCad 5 crossover fixture into tmp_path alone.

    Legacy setup tokens → ``was_legacy_design_settings_loaded==True`` →
    no fallback branch runs regardless of ``rules_source``.
    """
    if not LEGACY_PCB_FIXTURE.exists():
        pytest.skip(f"Fixture not found: {LEGACY_PCB_FIXTURE}")
    dest = tmp_path / LEGACY_PCB_FIXTURE.name
    shutil.copy(LEGACY_PCB_FIXTURE, dest)
    return dest


# ---------------------------------------------------------------------------
# rules_source on the fallback path
# ---------------------------------------------------------------------------

def test_board_or_yaml_mutates_bds_on_fallback(
    krl, orphan_modern_pcb: Path, custom_drc_yaml: Path
) -> None:
    """Opt-in: YAML minima land in BDS when the board is a real-default
    fallback. Emits a UserWarning naming the raised Default class sizes, and
    returns the substitution record."""
    router = krl.RLRouter(str(orphan_modern_pcb))
    assert not router.was_project_loaded_from_file()
    assert not router.was_legacy_design_settings_loaded()

    builtin = _default_class_sizes(router.get_design_rules())
    with pytest.warns(UserWarning, match="Default net class sizes raised"):
        record = apply_default_drc_if_fallback(
            router, config_path=custom_drc_yaml, rules_source="board_or_yaml",
        )
    assert record["drc_config_path"] == str(custom_drc_yaml)
    assert record["minima"]["min_clearance_mm"] == pytest.approx(0.37)
    # KiCad's built-in Default class (0.2 / 0.2 / 0.6 / 0.3) sits below the YAML
    # minima except the drill (0.27): those sizes are raised, recorded as [old, new].
    raised = record["default_class_raised"]
    assert sorted(raised) == ["clearance_mm", "track_width_mm", "via_diameter_mm"]
    for field, key, new in [("track_width_mm", "track_width", 0.41),
                            ("clearance_mm", "clearance", 0.37),
                            ("via_diameter_mm", "via_diameter", 0.71)]:
        assert raised[field] == pytest.approx([builtin[key], new], abs=2e-6)

    got = router.get_design_rules()
    for field, value in load_drc_config(custom_drc_yaml).items():
        assert getattr(got, field) == pytest.approx(value, abs=2e-6), field


def test_an_unknown_rules_source_is_refused_before_the_engine_is_read(
    custom_drc_yaml: Path,
) -> None:
    with pytest.raises(ValueError, match="rules_source must be one of"):
        apply_default_drc_if_fallback(None, custom_drc_yaml, rules_source="yaml")


def test_board_raises_on_fallback(
    krl, orphan_modern_pcb: Path, custom_drc_yaml: Path
) -> None:
    """Strict mode: ``rules_source='board'`` on a real-default fallback raises
    ``ValueError`` and does not mutate BDS."""
    router = krl.RLRouter(str(orphan_modern_pcb))
    assert not router.was_project_loaded_from_file()
    assert not router.was_legacy_design_settings_loaded()

    before = _global_minima_snapshot(router.get_design_rules())

    with pytest.raises(ValueError, match="DRC fallback triggered"):
        apply_default_drc_if_fallback(
            router, config_path=custom_drc_yaml, rules_source="board",
        )

    after = _global_minima_snapshot(router.get_design_rules())
    assert after == before, (
        "raise path must leave BDS exactly as the loader left it; "
        f"diff: {[k for k in after if after[k] != before[k]]}"
    )


def test_board_or_yaml_without_config_path_raises(
    krl, orphan_modern_pcb: Path
) -> None:
    """``rules_source='board_or_yaml'`` but ``config_path=None`` raises: there is no
    implicit default YAML, so the fallback must name its substitute
    explicitly. BDS is left untouched."""
    router = krl.RLRouter(str(orphan_modern_pcb))
    assert not router.was_project_loaded_from_file()
    assert not router.was_legacy_design_settings_loaded()

    before = _global_minima_snapshot(router.get_design_rules())

    with pytest.raises(ValueError, match="no implicit default YAML"):
        apply_default_drc_if_fallback(router, config_path=None, rules_source="board_or_yaml")

    after = _global_minima_snapshot(router.get_design_rules())
    assert after == before, (
        "missing-path raise must leave BDS exactly as the loader left it; "
        f"diff: {[k for k in after if after[k] != before[k]]}"
    )


# ---------------------------------------------------------------------------
# Non-fallback boards: rules_source is irrelevant
# ---------------------------------------------------------------------------

def test_rules_source_irrelevant_when_project_loaded(
    krl, paired_modern_pcb: Path, custom_drc_yaml: Path
) -> None:
    """When the pro companion is on disk, ``was_project_loaded_from_file``
    is True → the fallback branch never runs, so both values must
    behave identically (no-op) and must not clobber the pro-sourced BDS.
    """
    router = krl.RLRouter(str(paired_modern_pcb))
    assert router.was_project_loaded_from_file(), (
        "sibling .kicad_pro must be picked up by the loader"
    )

    baseline = _global_minima_snapshot(router.get_design_rules())

    for source in RULES_SOURCES:
        assert apply_default_drc_if_fallback(
            router, config_path=custom_drc_yaml, rules_source=source,
        ) is None, f"expected no-op on loaded-from-file board (rules_source={source})"
        after = _global_minima_snapshot(router.get_design_rules())
        assert after == baseline, (
            f"BDS must not change for rules_source={source}; "
            f"diff: {[k for k in after if after[k] != baseline[k]]}"
        )


def test_rules_source_irrelevant_when_legacy_loaded(
    krl, legacy_pcb: Path, custom_drc_yaml: Path
) -> None:
    """Legacy KiCad 5 setup tokens populate BDS via the parser → the
    fallback branch must not run regardless of ``rules_source``."""
    router = krl.RLRouter(str(legacy_pcb))
    assert router.was_legacy_design_settings_loaded(), (
        "legacy crossover fixture must set the legacy flag at parse time"
    )

    baseline = _global_minima_snapshot(router.get_design_rules())

    for source in RULES_SOURCES:
        assert apply_default_drc_if_fallback(
            router, config_path=custom_drc_yaml, rules_source=source,
        ) is None, f"expected no-op on legacy-loaded board (rules_source={source})"
        after = _global_minima_snapshot(router.get_design_rules())
        assert after == baseline, (
            f"BDS must not change for rules_source={source}; "
            f"diff: {[k for k in after if after[k] != baseline[k]]}"
        )


# ---------------------------------------------------------------------------
# The fallback writes the YAML minima into the Default net class (pro-less board)
# ---------------------------------------------------------------------------

STRICT_YAML = PROJECT_ROOT / "experiments" / "kdd" / "configs" / "drc" / "synth_2L_v2.yaml"
STRICT = {"track_width": 0.3, "clearance": 0.3, "via_diameter": 1.2, "via_drill": 0.6}
NET1_PAD, NET1_TARGET = (10.0, 10.0), (40.0, 10.0)     # simple_routing_board NET1 run


def _default_class_sizes(rules) -> dict[str, float]:
    d = rules.default_netclass
    return {"track_width": d.track_width_mm, "clearance": d.clearance_mm,
            "via_diameter": d.via_diameter_mm, "via_drill": d.via_drill_mm}


def test_strict_yaml_fallback_lifts_the_default_class(orphan_modern_pcb: Path) -> None:
    """A pro-less board (KiCad's built-in Default class 0.2/0.2/0.6/0.3) under the strict
    synth_2L_v2 YAML loads: the Default class, the router's sizes and the routed track all
    carry the YAML values (0.3/0.3/1.2/0.6), with no exemption from verify_netclass_rules."""
    from pcb_world.core.env import PCBWorld

    with pytest.warns(UserWarning, match="DRC fallback triggered"):
        env = PCBWorld(
            str(orphan_modern_pcb),
            EnvConfig.with_overrides(max_steps=8, rules_source="board_or_yaml", drc_config_path=str(STRICT_YAML)),
        )
    try:
        env.reset()
        eng = env._engine
        assert _default_class_sizes(eng.get_design_rules()) == pytest.approx(STRICT)
        assert eng.start_route(*NET1_PAD, 1)
        assert eng.get_route_sizes() == pytest.approx(STRICT)
        assert eng.fix_route(*NET1_TARGET, force_finish=True)
        widths = {round(t.width_mm, 6) for t in eng.get_tracks() if t.net_code == 1}
        assert widths == {0.3}
    finally:
        env.close()


def test_default_class_change_reaches_the_router(orphan_modern_pcb: Path) -> None:
    """A Default class value ABOVE the board minimum (so the minimum floor cannot be what
    sizes the route): start_route imports it and the committed track has it — the router's
    net class rules are rebuilt after the change."""

    eng = open_engine(
        str(orphan_modern_pcb),
        allow_default_rules=True,
    )
    try:
        rules = eng.get_design_rules()
        rules.default_netclass.track_width_mm = 0.45
        rules.default_netclass.via_diameter_mm = 1.0
        rules.default_netclass.via_drill_mm = 0.5
        eng.set_design_rules(rules)
        got = eng.get_design_rules()
        assert got.default_netclass.track_width_mm == pytest.approx(0.45)
        assert got.min_track_width_mm < 0.45
        eng.build_connectivity()
        assert eng.start_route(*NET1_PAD, 1)
        sizes = eng.get_route_sizes()
        assert sizes["track_width"] == pytest.approx(0.45)
        assert sizes["via_diameter"] == pytest.approx(1.0)
        assert sizes["via_drill"] == pytest.approx(0.5)
        assert eng.fix_route(*NET1_TARGET, force_finish=True)
        assert {round(t.width_mm, 6) for t in eng.get_tracks() if t.net_code == 1} == {0.45}
    finally:
        eng.close()


def test_design_rules_round_trip_leaves_the_default_class(krl, orphan_modern_pcb: Path) -> None:
    """get -> set with the class's own values (and a fresh struct, all -1) changes nothing."""
    router = krl.RLRouter(str(orphan_modern_pcb))
    before = _default_class_sizes(router.get_design_rules())
    router.set_design_rules(router.get_design_rules())
    fresh = router.get_design_rules()
    fresh.default_netclass = krl.NetClassInfo()       # every field -1 = leave unchanged
    router.set_design_rules(fresh)
    assert _default_class_sizes(router.get_design_rules()) == pytest.approx(before)


# ---------------------------------------------------------------------------
# PCBWorld: rules_source end to end, the substitution record and the rules digest
# ---------------------------------------------------------------------------

def _world(pcb: Path, **kw):
    from pcb_world.core.env import PCBWorld
    return PCBWorld(str(pcb), EnvConfig.with_overrides(**{"max_steps": 4, **kw}))


def test_board_refuses_a_board_without_rules(orphan_modern_pcb: Path) -> None:
    """``rules_source='board'`` (the default): a board with no rules of its own does not
    load — no route under KiCad's compile-time defaults."""
    with pytest.raises(RuntimeError, match="rules_source=board_or_yaml"):
        _world(orphan_modern_pcb, rules_source="board")


def test_board_or_yaml_substitutes_exactly_the_yaml_minima(orphan_modern_pcb: Path) -> None:
    """``board_or_yaml`` on a board without rules = what the old
    ``use_yaml_drc_fallback=True`` did: the board's own rules with the YAML minima
    written over, and each Default class size below its minimum raised to it — nothing
    else. Recorded in ``rules_substitution``; ``rules_digest`` is the effective rules'."""
    from pcb_world.engine.drc_config import NETCLASS_RULE_FIELDS, load_drc_config

    eng = open_engine(str(orphan_modern_pcb), allow_default_rules=True)
    try:
        expected = eng.get_design_rules()           # the board as loaded, before the YAML
    finally:
        eng.close()
    minima = load_drc_config(DEFAULT_DRC_CONFIG_PATH)
    for field, value in minima.items():
        setattr(expected, field, value)
    for nc_attr, floor_attr, _ in NETCLASS_RULE_FIELDS:
        setattr(expected.default_netclass, nc_attr,
                max(getattr(expected.default_netclass, nc_attr), getattr(expected, floor_attr)))

    with pytest.warns(UserWarning, match="DRC fallback triggered"):
        env = _world(orphan_modern_pcb, rules_source="board_or_yaml",
                     drc_config_path=str(DEFAULT_DRC_CONFIG_PATH), max_steps=1)
    try:
        got = env._engine.get_design_rules()
        assert design_rules_digest(got) == design_rules_digest(expected)
        assert env.rules_digest == design_rules_digest(got)
        assert env.rules_substitution["drc_config_path"] == str(DEFAULT_DRC_CONFIG_PATH)
        assert env.rules_substitution["minima"] == minima
        # the episode end carries the record onto the per-rollout row (as JSON)
        reset_info, info = _one_step_episode(env)
        assert reset_info["rules_substitution"] == info["rules_substitution"] == env.rules_substitution
        row = runtime_metrics_from_info(info, terminated=False, truncated=True)
        assert json.loads(row["rules_substitution"]) == json.loads(json.dumps(env.rules_substitution))
    finally:
        env.close()


def test_rules_source_is_a_no_op_on_a_board_with_rules(paired_modern_pcb: Path) -> None:
    """A board with its own .kicad_pro routes under the same rules either way."""
    digests = []
    for source in RULES_SOURCES:
        env = _world(paired_modern_pcb, rules_source=source,
                     drc_config_path=str(DEFAULT_DRC_CONFIG_PATH))
        try:
            assert env.rules_substitution is None
            digests.append(env.rules_digest)
        finally:
            env.close()
    assert digests[0] == digests[1]


def _one_step_episode(env) -> tuple[dict, dict]:
    """Reset ``env`` (max_steps=1) and return the reset and truncated step's infos."""
    from pcb_world.core.masking import ACT_NET_SELECT

    _, reset_info = env.reset()
    _, _, _, truncated, info = env.step({"action_type": ACT_NET_SELECT, "net_id": 1})
    assert truncated
    return reset_info, info


def test_reset_and_terminal_info_carry_the_rules_digest(paired_modern_pcb: Path) -> None:
    """Every reset and episode end carries the digest, so per-rollout rows (eval
    ``runtime_metrics_from_info``; LLM rows from the reset info) record the rules each
    board was routed under; the board's own rules leave ``rules_substitution`` blank."""
    env = _world(paired_modern_pcb, max_steps=1)
    try:
        reset_info, info = _one_step_episode(env)
        assert reset_info["rules_digest"] == info["rules_digest"] == env.rules_digest
        assert reset_info["rules_substitution"] is info["rules_substitution"] is None
        row = runtime_metrics_from_info(info, terminated=False, truncated=True)
        assert row["rules_digest"] == env.rules_digest
        assert row["rules_substitution"] == ""
    finally:
        env.close()


def test_rl_eval_rollout_rows_carry_the_rules_digest(tmp_path: Path) -> None:
    """Integration: the RL eval rollout (env pool -> transformer rollout -> per_rollout.csv)
    writes the digest of the rules the board was loaded with."""
    import torch

    from configs.loader.schema import DRC_OFF, RLEnvConfig
    from eval import eval_utils as u
    from eval.rollout.rl import eval_transformer
    from methods._shared.board_loader import BoardSpec
    from methods.rl_agent.models.v1.net import KiCadRLModel
    from methods.rl_agent.policy.agent import KiCadRLAgent

    eng = open_engine(str(MODERN_PCB_FIXTURE))
    try:
        expected = design_rules_digest(eng.get_design_rules())
    finally:
        eng.close()
    torch.manual_seed(0)
    agent = KiCadRLAgent(KiCadRLModel(disable_slot_emb=False, policy_net_select=False, d_model=32, n_heads=2, n_layers=2, d_ff=64),
                         device=torch.device("cpu"))
    csv_path = tmp_path / "per_rollout.csv"
    eval_transformer(agent, torch.device("cpu"), [BoardSpec(0, "b0", str(MODERN_PCB_FIXTURE))],
                     env_config=RLEnvConfig().with_flat({"max_steps": 8, **DRC_OFF}), n_rollouts=1,
                     n_envs=1, base_seed=0, max_steps=8, deterministic=True,
                     final_drc=False, per_rollout_flush_path=str(csv_path))
    (row,) = u.read_csv(csv_path)
    assert row["rules_digest"] == expected
    assert row["rules_substitution"] == ""
