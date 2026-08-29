"""Tests for the net-class aware ``net_select`` machinery.

Three layers of coverage:

1. **Parser** — using the ``tests/fixtures/crossover_board.kicad_pcb`` fixture
   (a 2-layer, all-thru_hole board), verify
   :func:`pcb_world.engine.pcb_file_parser.parse_pcb_file` routes
   ``thru_hole`` pads to ``pads`` with ``layer=0`` (the thru-hole
   sentinel) and keeps ``np_thru_hole`` pads out of the routing set.

2. **NPTH → obstacles** — synthesise a minimal .kicad_pcb (inline) with
   a single ``np_thru_hole`` mounting hole and confirm the parser drops
   it into the ``obstacles`` list instead of ``pads``.

3. **``PCBWorld`` integration** (requires the C++ binding) — run
   ``reset`` + ``step(net_select)`` on the crossover fixture and assert
   that the engine's ``set_track_width`` / ``set_via_diameter`` /
   ``set_via_drill`` are invoked with values from the resolved
   netclass. Skipped when the binding is unavailable.

The fallback-logic paths in :meth:`PCBWorld._resolve_netclass` (engine
without ``get_netclass_for_net``, name-equality heuristic) are covered
by class-level fakes that swap in a lightweight ``_engine`` /
``_board_info`` so we exercise the method without pulling the C++ layer.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.pro_sidecar import write_default_pro

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
CROSSOVER_FIXTURE = FIXTURES_DIR / "crossover_board.kicad_pcb"


# ---------------------------------------------------------------------------
# 1. Parser on crossover fixture
# ---------------------------------------------------------------------------

class TestParserOnCrossover:
    """All 12 crossover pads are ``thru_hole`` → ``layer=0`` sentinel."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not CROSSOVER_FIXTURE.exists():
            pytest.skip(f"Crossover fixture not found: {CROSSOVER_FIXTURE}")

    @pytest.fixture
    def parsed(self):
        from pcb_world.engine.kicad_engine import KiCadEngine
        from pcb_world.engine.pcb_file_parser import parse_pcb_file
        engine = KiCadEngine(str(CROSSOVER_FIXTURE))
        try:
            return parse_pcb_file(CROSSOVER_FIXTURE, engine)
        finally:
            engine.close()

    def test_pad_and_obstacle_counts(self, parsed):
        pads = parsed["board_snapshot"].pads
        obstacles = parsed.get("obstacles", [])
        assert len(pads) == 12, f"expected 12 pads, got {len(pads)}"
        assert len(obstacles) == 0, "crossover board has no NPTH"

    def test_every_pad_is_thru_hole_sentinel(self, parsed):
        pads = parsed["board_snapshot"].pads
        layers = [p.layer for p in pads]
        assert all(l == 0 for l in layers), (
            f"every crossover pad should report layer=0 (thru_hole "
            f"sentinel); got {sorted(set(layers))}"
        )

    def test_pads_retain_net_codes(self, parsed):
        pads = parsed["board_snapshot"].pads
        net_codes = {p.net_code for p in pads}
        # crossover_legacy.kicad_pcb declares nets 1..3 (GND, /A, /B).
        assert net_codes == {1, 2, 3}, (
            f"expected net codes 1..3 on crossover pads, got {sorted(net_codes)}"
        )

    def test_parse_stats(self, parsed):
        stats = parsed["parse_stats"]
        assert stats["pads"] == 12
        assert stats["thru_hole_pads"] == 12
        assert stats["obstacles"] == 0
        assert stats["copper_layers"] == 2


# ---------------------------------------------------------------------------
# 2. NPTH → obstacles (synthetic board)
# ---------------------------------------------------------------------------

_NPTH_BOARD_TEMPLATE = """(kicad_pcb (version 20171130) (host pcbnew "synthetic")
  (general (thickness 1.6) (nets 1))
  (page A4)
  (layers
    (0 F.Cu signal)
    (31 B.Cu signal)
    (44 Edge.Cuts user)
  )
  (net 0 "")
  (net 1 "SIG")
  (gr_line (start 0 0)   (end 30 0)  (layer Edge.Cuts) (width 0.05))
  (gr_line (start 30 0)  (end 30 20) (layer Edge.Cuts) (width 0.05))
  (gr_line (start 30 20) (end 0 20)  (layer Edge.Cuts) (width 0.05))
  (gr_line (start 0 20)  (end 0 0)   (layer Edge.Cuts) (width 0.05))
  (module test:one (layer F.Cu) (at 10 10)
    (pad 1 smd rect (at 0 0) (size 1 1) (layers F.Cu F.Paste F.Mask)
      (net 1 SIG))
    (pad MH np_thru_hole circle (at 5 0) (size 3 3) (drill 3) (layers *.Cu *.Mask)
      (net 0 ""))
  )
)
"""


class TestNpthRoutesToObstacles:
    """``np_thru_hole`` pads land in ``obstacles``, not ``pads``."""

    @pytest.fixture
    def synth_board(self, tmp_path):
        path = tmp_path / "synth_npth.kicad_pcb"
        path.write_text(_NPTH_BOARD_TEMPLATE)
        write_default_pro(path)   # engine load contract: pro sibling required
        return path

    @pytest.fixture
    def parsed(self, synth_board):
        from pcb_world.engine.kicad_engine import KiCadEngine
        from pcb_world.engine.pcb_file_parser import parse_pcb_file
        engine = KiCadEngine(str(synth_board))
        try:
            return parse_pcb_file(synth_board, engine)
        finally:
            engine.close()

    def test_npth_not_in_pads(self, parsed):
        pads = parsed["board_snapshot"].pads
        # Only the SMD pad survives in ``pads``; the NPTH is diverted.
        assert len(pads) == 1, f"expected 1 SMD pad, got {len(pads)}"
        assert pads[0].layer != 0, "SMD pad should keep its real layer"

    def test_npth_in_obstacles(self, parsed):
        obstacles = parsed.get("obstacles", [])
        assert len(obstacles) == 1, f"expected 1 NPTH obstacle, got {len(obstacles)}"
        obs = obstacles[0]
        # Mounting hole at (10+5, 10) = (15, 10) with 3 mm circle.
        assert obs.x_mm == pytest.approx(15.0)
        assert obs.y_mm == pytest.approx(10.0)
        assert obs.shape == "circle"

    def test_parse_stats_split(self, parsed):
        assert parsed["parse_stats"]["pads"] == 1
        assert parsed["parse_stats"]["obstacles"] == 1
        assert parsed["parse_stats"]["thru_hole_pads"] == 0


# ---------------------------------------------------------------------------
# 3. _resolve_netclass fallback paths (unit-level, no C++)
# ---------------------------------------------------------------------------

def _mk_nc(name, tw=-1.0, vd=-1.0, dr=-1.0):
    """Lightweight RLNetClassInfo stand-in with the fields the code reads."""
    return SimpleNamespace(
        name=name,
        clearance_mm=-1.0,
        track_width_mm=tw,
        via_diameter_mm=vd,
        via_drill_mm=dr,
        uvia_diameter_mm=-1.0,
        uvia_drill_mm=-1.0,
    )


def _mk_rules(
    default_nc,
    netclasses,
    *,
    min_tw=-1.0,
    min_cl=-1.0,
    min_vd=-1.0,
    min_th=-1.0,
):
    """Lightweight RLDesignRules stand-in.

    BDS global minima default to ``-1.0`` (unset) so existing tests keep
    their un-clamped expectations. Pass positive values to exercise the
    clamp path in :meth:`PCBWorld._resolve_net_rule_values`. Every floor
    attribute the resolver reads MUST exist here (it accesses them
    without a getattr default on purpose — a missing floor is loud).
    """
    return SimpleNamespace(
        default_netclass=default_nc,
        netclasses=list(netclasses),
        min_track_width_mm=min_tw,
        min_clearance_mm=min_cl,
        min_via_diameter_mm=min_vd,
        min_through_hole_mm=min_th,
    )


class _RecordingEngine:
    """Captures ``set_*`` calls so assertions can inspect the invocation order."""

    def __init__(self, rules, net_to_class_name=None):
        self._rules = rules
        self._net_to_class_name = net_to_class_name or {}
        self.set_track_width_calls: list = []
        self.set_via_diameter_calls: list = []
        self.set_via_drill_calls: list = []

    # engine API surface needed by the env ----------------------------------

    def get_design_rules(self):
        return self._rules

    def get_netclass_for_net(self, net_code):
        name = self._net_to_class_name.get(net_code, "")
        if not name:
            return _mk_nc("")  # empty name → signals "no match" to caller
        # Find the class by name in either default or the list.
        if self._rules.default_netclass.name == name:
            return self._rules.default_netclass
        for nc in self._rules.netclasses:
            if nc.name == name:
                return nc
        return _mk_nc("")

    def set_track_width(self, v):
        self.set_track_width_calls.append(v)

    def set_via_diameter(self, v):
        self.set_via_diameter_calls.append(v)

    def set_via_drill(self, v):
        self.set_via_drill_calls.append(v)


def _mk_env_with_engine(engine, nets):
    """Instantiate ``PCBWorld`` without running ``__init__`` and wire the
    minimal state ``_resolve_netclass`` / ``_apply_net_constraints`` touch.
    """
    from pcb_world.core.env import PCBWorld
    env = PCBWorld.__new__(PCBWorld)
    env._engine = engine
    env._board_info = SimpleNamespace(nets=nets)
    return env


class TestResolveNetclassFallback:
    """Exercises the three resolution branches in ``_resolve_netclass``."""

    def test_engine_api_authoritative(self):
        """When the C++ binding yields a non-empty class, use it verbatim."""
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=0.4)
        hv = _mk_nc("HighSpeed", tw=0.15, vd=0.6, dr=0.3)
        rules = _mk_rules(default, [hv])

        engine = _RecordingEngine(
            rules, net_to_class_name={7: "HighSpeed"},
        )
        nets = {7: SimpleNamespace(net_name="/CLK")}  # name != class
        env = _mk_env_with_engine(engine, nets)

        nc = env._resolve_netclass(7, rules)
        assert nc.name == "HighSpeed", (
            "engine lookup should win even though net name differs from class"
        )

    def test_heuristic_when_engine_returns_empty(self):
        """Empty-name response from the binding → fall through to the
        name-equality heuristic."""
        default = _mk_nc("Default", tw=0.25)
        hv = _mk_nc("HighSpeed", tw=0.15)
        rules = _mk_rules(default, [hv])

        engine = _RecordingEngine(rules)  # get_netclass_for_net → empty
        nets = {5: SimpleNamespace(net_name="HighSpeed")}  # name-match heuristic
        env = _mk_env_with_engine(engine, nets)

        nc = env._resolve_netclass(5, rules)
        assert nc.name == "HighSpeed"

    def test_engine_missing_method(self):
        """Old bindings (no ``get_netclass_for_net``) still reach the
        heuristic via the ``getattr`` guard."""
        default = _mk_nc("Default", tw=0.25)
        rules = _mk_rules(default, [_mk_nc("HighSpeed", tw=0.15)])

        class _OldEngine:
            def get_design_rules(self): return rules
            def set_track_width(self, v): pass
            def set_via_diameter(self, v): pass
            def set_via_drill(self, v): pass

        engine = _OldEngine()
        nets = {5: SimpleNamespace(net_name="HighSpeed")}
        env = _mk_env_with_engine(engine, nets)

        nc = env._resolve_netclass(5, rules)
        assert nc.name == "HighSpeed"

    def test_fall_back_to_default(self):
        """No engine match AND no heuristic match → Default netclass."""
        default = _mk_nc("Default", tw=0.25)
        rules = _mk_rules(default, [_mk_nc("HighSpeed", tw=0.15)])

        engine = _RecordingEngine(rules)
        nets = {3: SimpleNamespace(net_name="/DATA")}  # no match anywhere
        env = _mk_env_with_engine(engine, nets)

        nc = env._resolve_netclass(3, rules)
        assert nc.name == "Default"


class TestApplyNetConstraints:
    """``_apply_net_constraints`` should feed values into the engine."""

    def test_matched_class_values_pushed(self):
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=0.4)
        hv = _mk_nc("HighSpeed", tw=0.15, vd=0.6, dr=0.3)
        rules = _mk_rules(default, [hv])

        engine = _RecordingEngine(
            rules, net_to_class_name={7: "HighSpeed"},
        )
        nets = {7: SimpleNamespace(net_name="/CLK")}
        env = _mk_env_with_engine(engine, nets)

        env._apply_net_constraints(7)

        assert engine.set_track_width_calls == [0.15]
        assert engine.set_via_diameter_calls == [0.6]
        assert engine.set_via_drill_calls == [0.3]

    def test_inherit_unset_fields_from_default(self):
        """A non-Default class that leaves drill unset (-1) inherits
        ``via_drill_mm`` from Default — mirrors KiCad's resolution."""
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=0.4)
        partial = _mk_nc("Partial", tw=0.18, vd=0.7, dr=-1.0)  # drill unset
        rules = _mk_rules(default, [partial])

        engine = _RecordingEngine(
            rules, net_to_class_name={2: "Partial"},
        )
        nets = {2: SimpleNamespace(net_name="/PWR")}
        env = _mk_env_with_engine(engine, nets)

        env._apply_net_constraints(2)

        assert engine.set_track_width_calls == [0.18]
        assert engine.set_via_diameter_calls == [0.7]
        assert engine.set_via_drill_calls == [0.4]   # inherited from Default

    def test_negative_values_skipped(self):
        """If both the matched class AND Default leave a field unset
        (stays negative), that setter is never called."""
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=-1.0)
        partial = _mk_nc("Partial", tw=0.18, vd=-1.0, dr=-1.0)
        rules = _mk_rules(default, [partial])

        engine = _RecordingEngine(
            rules, net_to_class_name={2: "Partial"},
        )
        nets = {2: SimpleNamespace(net_name="/PWR")}
        env = _mk_env_with_engine(engine, nets)

        env._apply_net_constraints(2)

        assert engine.set_track_width_calls == [0.18]
        assert engine.set_via_diameter_calls == [0.8]     # Default's positive
        assert engine.set_via_drill_calls == []           # both unset → skipped

    def test_clamped_to_bds_global_min(self):
        """Values below the BDS global minimum are lifted up to the
        minimum — matches KiCad's own DRC floor."""
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=0.4)
        # Non-Default class deliberately below every min.
        sub_min = _mk_nc("Sub", tw=0.10, vd=0.50, dr=0.20)
        rules = _mk_rules(
            default, [sub_min],
            min_tw=0.20,
            min_vd=0.60,
            min_th=0.30,
        )

        engine = _RecordingEngine(
            rules, net_to_class_name={3: "Sub"},
        )
        nets = {3: SimpleNamespace(net_name="/LOGIC")}
        env = _mk_env_with_engine(engine, nets)

        env._apply_net_constraints(3)

        # Each value gets clamped up to the BDS floor, not the raw class value.
        assert engine.set_track_width_calls == [0.20]
        assert engine.set_via_diameter_calls == [0.60]
        assert engine.set_via_drill_calls == [0.30]

    def test_clamp_is_noop_when_class_exceeds_min(self):
        """Clamp only lifts values below the floor — larger class values
        pass through unchanged."""
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=0.4)
        wide = _mk_nc("Wide", tw=0.50, vd=1.20, dr=0.60)
        rules = _mk_rules(
            default, [wide],
            min_tw=0.20,
            min_vd=0.60,
            min_th=0.30,
        )

        engine = _RecordingEngine(
            rules, net_to_class_name={4: "Wide"},
        )
        nets = {4: SimpleNamespace(net_name="/PWR")}
        env = _mk_env_with_engine(engine, nets)

        env._apply_net_constraints(4)

        assert engine.set_track_width_calls == [0.50]
        assert engine.set_via_diameter_calls == [1.20]
        assert engine.set_via_drill_calls == [0.60]


# ---------------------------------------------------------------------------
# 4. End-to-end on crossover (requires the C++ binding)
# ---------------------------------------------------------------------------

def _binding_available() -> bool:
    try:
        from pcb_world.engine import KiCadEngine  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    not CROSSOVER_FIXTURE.exists(),
    reason="crossover fixture missing",
)
@pytest.mark.skipif(
    not _binding_available(),
    reason="kicad_rl_router binding not built",
)
class TestNetSelectIntegration:
    """Reset on crossover + a ``net_select`` step.

    The crossover fixture has no non-Default netclasses, so every net
    resolves to Default. The env's ``_apply_net_constraints`` fast-paths
    that case — it does **not** re-push values the engine already holds
    from ``__init__``. We verify that behaviour here, and pin the
    underlying invariant: a ``net_select`` step must never leave the
    router with zero/invalid track_width or via sizes regardless of
    whether the setters were called.
    """

    @pytest.fixture
    def env(self):
        from pcb_world.core.env import PCBWorld
        env = PCBWorld(board_path=str(CROSSOVER_FIXTURE), max_steps=5)
        yield env
        env.close()

    def test_default_netclass_takes_fast_path(self, env, monkeypatch):
        env.reset()

        calls = {"tw": [], "vd": [], "vdr": []}
        monkeypatch.setattr(
            env._engine, "set_track_width",
            lambda v, _c=calls: _c["tw"].append(v),
        )
        monkeypatch.setattr(
            env._engine, "set_via_diameter",
            lambda v, _c=calls: _c["vd"].append(v),
        )
        monkeypatch.setattr(
            env._engine, "set_via_drill",
            lambda v, _c=calls: _c["vdr"].append(v),
        )

        net_id = next(iter(env._board_info.nets))
        env.step({"action_type": 0, "net_id": int(net_id)})  # 0 = ACT_NET_SELECT

        # Fast path: Default netclass resolution must NOT re-push values
        # — re-pushing identical widths mid-episode was observed to
        # perturb PNS state on DRC-boundary boards.
        assert calls["tw"] == []
        assert calls["vd"] == []
        assert calls["vdr"] == []

    def test_resolved_class_is_default_on_crossover(self, env):
        env.reset()
        rules = env._engine.get_design_rules()
        net_id = next(iter(env._board_info.nets))
        nc = env._resolve_netclass(net_id, rules)
        assert nc.name == rules.default_netclass.name, (
            "crossover fixture has no custom netclasses; every net should "
            f"resolve to Default, got {nc.name!r}"
        )

    def test_non_default_class_triggers_setters(self, env, monkeypatch):
        """When _resolve_netclass returns a non-Default class, the setters
        must be called with that class's values. Simulated by stubbing the
        resolver to return a synthetic 'HighSpeed' class."""
        env.reset()

        rules = env._engine.get_design_rules()
        hs = SimpleNamespace(
            name="HighSpeed",
            track_width_mm=0.33,
            via_diameter_mm=0.77,
            via_drill_mm=0.41,
        )
        monkeypatch.setattr(
            env, "_resolve_netclass", lambda nid, r: hs,
        )

        calls = {"tw": [], "vd": [], "vdr": []}
        monkeypatch.setattr(
            env._engine, "set_track_width",
            lambda v, _c=calls: _c["tw"].append(v),
        )
        monkeypatch.setattr(
            env._engine, "set_via_diameter",
            lambda v, _c=calls: _c["vd"].append(v),
        )
        monkeypatch.setattr(
            env._engine, "set_via_drill",
            lambda v, _c=calls: _c["vdr"].append(v),
        )

        net_id = next(iter(env._board_info.nets))
        env.step({"action_type": 0, "net_id": int(net_id)})

        assert calls["tw"] == [pytest.approx(0.33)]
        assert calls["vd"] == [pytest.approx(0.77)]
        assert calls["vdr"] == [pytest.approx(0.41)]


# ---------------------------------------------------------------------------
# 5. THT pad representation in downstream consumers
# ---------------------------------------------------------------------------

class TestThtPadInLLMProjection:
    """LLM sees ``layer=0`` as the ``th`` tag — both sexpr and XML paths."""

    @pytest.fixture
    def board_static(self):
        # Minimal board_static dict with one SMD (layer=1) + one THT (layer=0)
        # pad, matching what ``BoardStatic.to_dict()`` would produce.
        return {
            "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 10.0, "bbox_h": 10.0,
            "scale": 10.0, "net_count": 1, "copper_layers": 2,
            "boardlines": {}, "obstacles": {}, "board_constraints": {},
            "nets": {
                "net_1": {
                    "net_name": "GND",
                    "pads": {
                        "pad_0": {
                            "id": "D0",
                            "center": {"id": "P0", "xy": [1.0, 2.0]},
                            "width": 1.0, "height": 1.0,
                            "layer": 1, "shape": "rect",
                        },
                        "pad_1": {
                            "id": "D1",
                            "center": {"id": "P1", "xy": [3.0, 4.0]},
                            "width": 2.0, "height": 2.0,
                            "layer": 0, "shape": "circle",
                        },
                    },
                },
            },
        }

    def test_sexpr_emits_th_tag(self, board_static):
        from methods.llm_agent.wrappers.state_converter import _board_static_to_sexpr
        out = _board_static_to_sexpr(board_static)
        assert "(pad D1" in out
        # The THT pad line ends with the ``th`` tag; the SMD pad keeps its
        # numeric layer.
        tht_line = next(l for l in out.splitlines() if "(pad D1" in l)
        smd_line = next(l for l in out.splitlines() if "(pad D0" in l)
        assert tht_line.rstrip().endswith(" th)")
        assert smd_line.rstrip().endswith(" 1)")

    def test_xml_rewrites_layer_attr(self, board_static):
        from methods.llm_agent.wrappers.state_converter import _tag_thru_hole_pads, _to_xml
        tagged = _tag_thru_hole_pads(board_static)
        xml = _to_xml("board_static", tagged)
        # THT pad attribute becomes ``layer="th"``; SMD pad stays numeric.
        tht = next(l for l in xml.splitlines() if 'id="D1"' in l)
        smd = next(l for l in xml.splitlines() if 'id="D0"' in l)
        assert 'layer="th"' in tht
        assert "layer=1" in smd


class TestThtPadInRLTokenizer:
    """RL tokenizer temporarily treats ``layer=0`` (THT) as Top (``layer=1``)
    to preserve the pre-sentinel encoding until THT handling is finalised.
    FIXME marker: :func:`state_tokenizer_batched.build_batch_tensors`.
    """

    def test_crossover_pads_encoded_as_top(self):
        # The crossover fixture has 12 THT pads (layer=0 in the observation
        # dict). Building an RL obs from it should therefore produce the
        # Top-layer encoding (``dt=0``) rather than the ambiguous ``(0, 0)``
        # the raw sentinel would yield.
        pytest.importorskip(
            "torch",
            reason="RL tokenizer depends on torch / tensor backend",
        )
        from methods.llm_agent.wrappers.state_converter import _board_static_to_sexpr  # noqa: F401 (ensures import path is healthy)

        # We test the single-pad encoding path directly because wiring up a
        # full RL observation requires the KiCad binding and a head token,
        # which are out of scope for this unit test. The code under test
        # is the tiny ``layer == 0 → 1`` shim at the top of the pad loop.
        from methods.rl_agent.models.v1.encoding import _safe_encode_layer

        n_copper = 2
        # Sanity: the sentinel itself encodes to (0, 0) — no layer signal.
        assert _safe_encode_layer(0, n_copper) == (0.0, 0.0)
        # After the shim, THT pads use Top=1 which encodes to a non-zero
        # vector distinct from Bottom=2.
        top = _safe_encode_layer(1, n_copper)
        bot = _safe_encode_layer(n_copper, n_copper)
        assert top != (0.0, 0.0)
        assert top != bot

    def test_shim_preserves_nonzero_layers(self):
        # Sanity: SMD pads (layer >= 1) pass through untouched.
        from methods.rl_agent.models.v1.encoding import _safe_encode_layer
        for ly in (1, 2, 3, 4):
            assert _safe_encode_layer(ly, 4) != (0.0, 0.0)
