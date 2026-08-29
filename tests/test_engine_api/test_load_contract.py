"""Engine strict load contract (no normalize-and-cache layer).

``KiCadEngine`` opens the source ``.kicad_pcb`` directly, so the source
must be self-sufficient:

  * modern-format pcb + a project file carrying the design rules — the
    ``<stem>.kicad_pro`` sibling (or an explicit ``project_path``);
  * a pro-less board is refused unless the caller passes
    ``allow_default_rules=True`` (the env sets it for its
    ``use_yaml_drc_fallback`` opt-in, which substitutes YAML rules);
  * legacy boards (KiCad 5 era, rules embedded in the pcb body) are
    refused unconditionally — ``load_and_save_via_engine`` is the
    one-shot conversion tool (its own round-trip contract is pinned by
    ``test_drc_version_upgrade``).

These tests pin the refusal/acceptance matrix at both the engine and the
env entry points.
"""

from __future__ import annotations

import gc
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"))

pytest.importorskip("kicad_rl_router")

from pcb_world.engine.kicad_engine import KiCadEngine  # noqa: E402

FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
MODERN_PAIR = FIXTURES / "simple_routing_board.kicad_pcb"
LEGACY_PCB = FIXTURES / "crossover_legacy.kicad_pcb"


@pytest.fixture
def orphan_pcb(tmp_path: Path) -> Path:
    """Modern pcb copied WITHOUT its .kicad_pro sibling."""
    dest = tmp_path / MODERN_PAIR.name
    shutil.copy(MODERN_PAIR, dest)
    return dest


def _close(engine: KiCadEngine | None) -> None:
    if engine is not None:
        engine.close()
    gc.collect()


def test_pro_pair_loads_with_project_rules():
    engine = None
    try:
        engine = KiCadEngine(str(MODERN_PAIR))
        assert engine.was_project_loaded_from_file()
        assert not engine.was_legacy_design_settings_loaded()
    finally:
        _close(engine)


def test_missing_pro_is_refused(orphan_pcb: Path):
    with pytest.raises(RuntimeError, match="project file was not loaded"):
        KiCadEngine(str(orphan_pcb))
    gc.collect()


def test_missing_pro_allowed_with_default_rules_optin(orphan_pcb: Path):
    engine = None
    try:
        engine = KiCadEngine(str(orphan_pcb), allow_default_rules=True)
        assert not engine.was_project_loaded_from_file()
    finally:
        _close(engine)


def test_legacy_board_is_refused_even_with_optin():
    with pytest.raises(RuntimeError, match="legacy board"):
        KiCadEngine(str(LEGACY_PCB))
    gc.collect()
    # ``allow_default_rules`` must NOT widen the legacy refusal.
    with pytest.raises(RuntimeError, match="legacy board"):
        KiCadEngine(str(LEGACY_PCB), allow_default_rules=True)
    gc.collect()


def test_refused_board_leaves_no_live_router(orphan_pcb: Path):
    """A refused construction must not trip the one-live-router guard for
    the next (valid) construction — close() runs before the raise."""
    with pytest.raises(RuntimeError):
        KiCadEngine(str(orphan_pcb))
    engine = None
    try:
        engine = KiCadEngine(str(MODERN_PAIR))
        assert engine.was_project_loaded_from_file()
    finally:
        _close(engine)


def test_env_orphan_board_strict_and_optin(orphan_pcb: Path):
    """PCBWorld inherits the contract: strict raise by default, loads under
    the use_yaml_drc_fallback opt-in (which substitutes the YAML rules)."""
    from pcb_world.core.env import PCBWorld

    with pytest.raises(RuntimeError, match="project file was not loaded"):
        PCBWorld(board_path=str(orphan_pcb), max_steps=1)
    gc.collect()

    env = None
    try:
        with pytest.warns(UserWarning, match="DRC fallback"):
            env = PCBWorld(
                board_path=str(orphan_pcb), max_steps=1,
                use_yaml_drc_fallback=True,
                drc_config_path=str(PROJECT_ROOT / "configs/drc/default.yaml"),
            )
        env.reset()
    finally:
        if env is not None:
            env.close()
        gc.collect()


def test_save_emits_pro_even_when_source_project_lock_held(tmp_path: Path):
    """A held/stale KiCad lock file must not silently drop the ``.kicad_pro``.

    ``SETTINGS_MANAGER::LoadProject`` opens the project READ-ONLY when the
    project's lock file (``~<stem>.kicad_pro.lck``) already exists — another
    process holding the same source concurrently, or a stale lock left by a
    crash. ``SaveProjectAs`` copies that flag onto the output project file,
    whose ``SaveToFile`` then silently no-ops: a routed ``.kicad_pcb`` lands
    WITHOUT its rules sidecar (a parallel-eval hazard). The
    engine's ``save()`` clears the flag scoped to the save-as (this engine
    never writes the SOURCE project) and fails loudly if the sidecar still
    did not land.
    """
    src = tmp_path / MODERN_PAIR.name
    shutil.copy(MODERN_PAIR, src)
    shutil.copy(MODERN_PAIR.with_suffix(".kicad_pro"), src.with_suffix(".kicad_pro"))
    # Plant a foreign lock next to the source project (KiCad lock naming:
    # LockFilePrefix "~" + name, extension + ".lck").
    lck = tmp_path / f"~{src.stem}.kicad_pro.lck"
    lck.write_text('{"username": "someone-else", "hostname": "other-host"}')

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = out_dir / "routed.kicad_pcb"
    eng = KiCadEngine(str(src))
    try:
        # Read-only open still reads the on-disk rules — the strict contract holds.
        assert eng.was_project_loaded_from_file()
        eng.save(str(out))
    finally:
        eng.close()
        del eng
        gc.collect()
    pro = out.with_suffix(".kicad_pro")
    assert out.is_file()
    assert pro.is_file() and pro.stat().st_size > 0, (
        "engine.save() dropped the .kicad_pro sidecar under a held project lock"
    )


def test_save_leaves_no_prl_sidecar(tmp_path: Path):
    """``save()`` must not litter a ``.kicad_prl`` next to its outputs.

    KiCad's project save emits a local-settings sidecar (``.kicad_prl`` —
    GUI view state; nothing the engine or scoring reads) as a side effect.
    ``save()`` removes a prl it created and preserves one that already
    existed at the path.
    """
    src = tmp_path / MODERN_PAIR.name
    shutil.copy(MODERN_PAIR, src)
    shutil.copy(MODERN_PAIR.with_suffix(".kicad_pro"), src.with_suffix(".kicad_pro"))
    out = tmp_path / "routed.kicad_pcb"
    kept = tmp_path / "kept.kicad_pcb"
    pre_existing = kept.with_suffix(".kicad_prl")
    pre_existing.write_text("{}")

    eng = KiCadEngine(str(src))
    try:
        eng.save(str(out))
        eng.save(str(kept))  # target with a pre-existing prl
    finally:
        eng.close()
        del eng
        gc.collect()
    assert out.with_suffix(".kicad_pro").is_file()
    assert not out.with_suffix(".kicad_prl").exists(), (
        "engine.save() leaked a .kicad_prl sidecar"
    )
    assert pre_existing.exists(), "save() deleted a pre-existing .kicad_prl"


def test_convert_pernet_tool_converts_proless_board(tmp_path: Path):
    """The dataset conversion tool must keep working under the strict contract.

    Its input is by definition pro-less (freshly generated pernet boards), so
    it must go through the raw-router primitive — opening via ``KiCadEngine``
    would refuse every input. Also pins that the conversion leaves no
    ``.kicad_prl`` litter.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "convert_pernet_to_pro",
        PROJECT_ROOT / "tools" / "datagen" / "synthetic_generator"
        / "convert_pernet_to_pro.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dest = tmp_path / MODERN_PAIR.name
    shutil.copy(MODERN_PAIR, dest)          # pcb only — NO .kicad_pro sibling
    _, ok, err = mod._convert_one(str(dest))
    assert ok, f"_convert_one failed on a pro-less board: {err}"
    assert dest.with_suffix(".kicad_pro").is_file()
    assert not dest.with_suffix(".kicad_prl").exists(), (
        "conversion leaked a .kicad_prl sidecar"
    )
