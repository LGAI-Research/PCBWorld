"""Per-net DRC constraint observation (the ``net_constraint_obs`` env knob).

Contract under test: with the knob ON every ``board_static`` net carries a
``constraints`` dict of resolved netclass values (track_width / clearance /
via_diameter / via_drill; KiCad inherit → Default fallback, BDS global-min
clamp) **identical to what ``net_select`` pushes into the engine** — the
policy observes the exact widths/clearances it will route with. With the
knob OFF (default) the observation stays byte-identical to the legacy
schema (no ``constraints`` key anywhere).

Layers:

1. **Unit** (no C++) — ``PCBWorld._resolve_net_rule_values`` raw/clamped
   resolution and ``_fill_net_constraint_obs`` population + loud failure,
   on the lightweight engine stand-ins from ``test_net_select_netclass``.
2. **Containers** (no C++) — ``NetContext.to_dict`` emission contract, the
   indexed_v1 ``net_constraints`` passthrough round-trip, and the ckpt
   round-trip (``RLEnvConfig.from_checkpoint`` carrying a stored True —
   the official eval path; the False-fallback legs live in
   tests/test_env_config.py).
3. **Integration** (needs the C++ binding) — a modern-format jumanji
   fixture copied to tmp with a rewritten ``.kicad_pro`` carrying a second
   netclass (pattern-assigned to NET2) + explicit BDS minima. Legacy-format
   fixtures (crossover / simple_routing) ignore a sibling pro's netclasses,
   so the modern board is required to exercise the per-net path. Also pins
   the tokenizer NET-channel normalization (tw/cl/vd ==
   clamped/norm_scale·100) and knob-on ≡ knob-off behavioural equivalence
   (obs minus ``constraints``, rewards, termination, masks) over a
   scripted routing episode.
4. **Eval drift guard** (no C++) — ``eval.rollout.rl.eval_transformer``
   refuses a ckpt↔env ``net_constraint_obs`` mismatch (the stamp comes from
   ``methods.rl_agent.models.loader._load_policy``); matching / unstamped
   policies pass through.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_env.test_net_select_netclass import (
    _RecordingEngine,
    _mk_env_with_engine,
    _mk_nc,
    _mk_rules,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
JUMANJI_DIR = FIXTURES_DIR / "jumanji" / "synth_1L_grid10_5net_v15" / "test"

CONSTRAINT_KEYS = {"track_width", "clearance", "via_diameter", "via_drill"}


# ---------------------------------------------------------------------------
# 1. Unit: _resolve_net_rule_values / _fill_net_constraint_obs
# ---------------------------------------------------------------------------

class TestResolveNetRuleValues:

    def test_inherit_and_clamp(self):
        """Unset (-1) class fields inherit from Default; BDS floors lift
        raw values in ``clamped`` only."""
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=0.4)
        default.clearance_mm = 0.2
        fast = _mk_nc("Fast", tw=0.15, vd=-1.0, dr=0.3)  # vd inherits
        fast.clearance_mm = -1.0                          # cl inherits
        rules = _mk_rules(default, [fast], min_tw=0.2, min_cl=0.25)

        engine = _RecordingEngine(rules, net_to_class_name={7: "Fast"})
        env = _mk_env_with_engine(engine, {7: SimpleNamespace(net_name="/CLK")})

        raw, clamped, is_default = env._resolve_net_rule_values(7, rules)

        assert not is_default
        assert raw == {
            "track_width": 0.15,   # class value
            "clearance": 0.2,      # inherited from Default
            "via_diameter": 0.8,   # inherited from Default
            "via_drill": 0.3,      # class value
        }
        # track_width sits below its floor (0.15 < min_tw 0.2) and the
        # inherited clearance below min_cl (0.2 < 0.25) — both lifted;
        # the other fields have no floor declared and pass through.
        assert clamped == {**raw, "track_width": 0.2, "clearance": 0.25}

    def test_default_class_identity(self):
        """A net resolving to Default: is_default=True, raw == clamped ==
        the Default values when no floor is declared."""
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=0.4)
        default.clearance_mm = 0.2
        rules = _mk_rules(default, [])

        engine = _RecordingEngine(rules)  # no match anywhere → Default
        env = _mk_env_with_engine(engine, {3: SimpleNamespace(net_name="/D")})

        raw, clamped, is_default = env._resolve_net_rule_values(3, rules)

        assert is_default
        assert raw == clamped == {
            "track_width": 0.25, "clearance": 0.2,
            "via_diameter": 0.8, "via_drill": 0.4,
        }


class TestFillNetConstraintObs:

    def test_populates_every_net_with_clamped_values(self):
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=0.4)
        default.clearance_mm = 0.2
        fast = _mk_nc("Fast", tw=0.35, vd=0.9, dr=0.45)
        fast.clearance_mm = 0.3
        rules = _mk_rules(default, [fast])

        engine = _RecordingEngine(rules, net_to_class_name={2: "Fast"})
        nets = {
            1: SimpleNamespace(net_name="/A", constraints=None),
            2: SimpleNamespace(net_name="/B", constraints=None),
        }
        env = _mk_env_with_engine(engine, nets)

        env._fill_net_constraint_obs()

        assert nets[1].constraints == {
            "track_width": 0.25, "clearance": 0.2,
            "via_diameter": 0.8, "via_drill": 0.4,
        }
        assert nets[2].constraints == {
            "track_width": 0.35, "clearance": 0.3,
            "via_diameter": 0.9, "via_drill": 0.45,
        }

    def test_loud_failure_on_unresolvable_field(self):
        """A field with no netclass value, no Default value and no BDS
        floor must raise — never a silent 0 observation. The message
        labels the -1-through-the-chain case as "unset"."""
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=-1.0)  # drill unset
        default.clearance_mm = 0.2
        rules = _mk_rules(default, [])

        engine = _RecordingEngine(rules)
        nets = {5: SimpleNamespace(net_name="/PWR", constraints=None)}
        env = _mk_env_with_engine(engine, nets)

        with pytest.raises(RuntimeError, match=r"/PWR.*via_drill.*unset"):
            env._fill_net_constraint_obs()

    def test_loud_failure_reports_explicit_zero_distinctly(self):
        """An explicit 0.0 field (legal in KiCad, distinct from unset -1)
        also fails loud, and the message shows the actual 0.0 rather than
        claiming the field was unset."""
        default = _mk_nc("Default", tw=0.25, vd=0.8, dr=0.4)
        default.clearance_mm = 0.0  # explicit zero, not inherit
        rules = _mk_rules(default, [])

        engine = _RecordingEngine(rules)
        nets = {5: SimpleNamespace(net_name="/PWR", constraints=None)}
        env = _mk_env_with_engine(engine, nets)

        with pytest.raises(RuntimeError, match=r"clearance.*0\.0"):
            env._fill_net_constraint_obs()


# ---------------------------------------------------------------------------
# 2. Containers: NetContext emission + indexed passthrough
# ---------------------------------------------------------------------------

class TestNetContextEmission:

    def test_no_key_when_unpopulated(self):
        from pcb_world.core.observation import NetContext
        ctx = NetContext(net_code=1, net_name="N")
        assert "constraints" not in ctx.to_dict()

    def test_key_emitted_as_copy_when_populated(self):
        from pcb_world.core.observation import NetContext
        vals = {"track_width": 0.2, "clearance": 0.1,
                "via_diameter": 0.6, "via_drill": 0.3}
        ctx = NetContext(net_code=1, net_name="N", constraints=vals)
        out = ctx.to_dict()
        assert out["constraints"] == vals
        out["constraints"]["track_width"] = 99.0  # mutating the dict copy…
        assert ctx.constraints["track_width"] == 0.2  # …never leaks back


class TestIndexedPassthrough:
    """indexed_v1 carries ``constraints`` losslessly (dict → arrays → dict)."""

    @pytest.fixture
    def board_static(self):
        pad = {
            "id": "D0", "center": {"id": "P0", "xy": [1.0, 2.0]},
            "width": 1.0, "height": 1.0, "layer": 1, "shape": "rect",
        }
        vals = {"track_width": 0.2, "clearance": 0.1,
                "via_diameter": 0.6, "via_drill": 0.3}
        return {
            "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 10.0, "bbox_h": 10.0,
            "scale": 10.0, "net_count": 2, "copper_layers": 2,
            "boardlines": {}, "obstacles": {}, "unconnected_pads": {},
            "board_constraints": {},
            "nets": {
                "net_1": {"net_name": "A", "pads": {"pad_0": pad},
                          "constraints": vals},
                "net_2": {"net_name": "B", "pads": {}},  # knob-off style net
            },
        }

    def test_round_trip_preserves_constraints(self, board_static):
        from pcb_world.core.indexed_obs import (
            arrays_to_dict, dict_to_arrays,
        )
        obs = {
            "board_static": board_static, "routing_geometry": {},
            "router_head": {"current_xy": [0.0, 0.0]},
            "drc_violations": [], "action_history": [], "closed_nets": [],
        }
        back = arrays_to_dict(dict_to_arrays(obs))
        nets = back["board_static"]["nets"]
        assert nets["net_1"]["constraints"] == board_static["nets"]["net_1"]["constraints"]
        assert "constraints" not in nets["net_2"]


class TestCheckpointRoundTrip:
    """ckpt args (saved ``vars(args)``) → ``RLEnvConfig.from_checkpoint``:
    the True leg of the knob round-trip. Bypassing this mapper (hand-built
    ``make_decoder_env`` kwargs) silently evals knob-off, so the official
    eval path must carry a knob-trained checkpoint's True through to the
    env kwargs. The False-fallback legs (knob absent in pre-knob ckpts)
    are pinned by tests/test_env_config.py — placed here rather than there
    because that file's collected-test count is frozen (see its trailing
    NOTE on the native double-init landmine)."""

    def test_true_round_trips_into_env_kwargs(self):
        from configs.loader.schema import RLEnvConfig

        got = RLEnvConfig.from_checkpoint(
            {"net_constraint_obs": True}, max_steps=64,
        ).to_pool_kwargs()
        assert got["net_constraint_obs"] is True


# ---------------------------------------------------------------------------
# 3. Integration: PCBWorld on a modern multi-netclass board
# ---------------------------------------------------------------------------

def _binding_available() -> bool:
    try:
        from pcb_world.engine import KiCadEngine  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# Expected values declared by the rewritten .kicad_pro below. BDS
# min_via_diameter (0.7) and min_clearance (0.22) deliberately sit ABOVE
# the Default class values (0.6 / 0.2) to pin both clamp paths against
# the real engine; the Fast class clears every floor and passes through.
_DEFAULT_VALS = {"track_width": 0.3,
                 "clearance": 0.22,    # class 0.2 clamped up to BDS min
                 "via_diameter": 0.7,  # class 0.6 clamped up to BDS min
                 "via_drill": 0.3}
_FAST_VALS = {"track_width": 0.35, "clearance": 0.25,
              "via_diameter": 0.8, "via_drill": 0.4}


@pytest.fixture(scope="module")
def multiclass_board(tmp_path_factory):
    """Jumanji board + rewritten pro: Default/Fast classes, NET2 → Fast."""
    if not (JUMANJI_DIR / "board_00000.kicad_pcb").exists():
        pytest.skip(f"jumanji fixture missing: {JUMANJI_DIR}")
    tmp = tmp_path_factory.mktemp("netconstraint_board")
    pcb = tmp / "b.kicad_pcb"
    shutil.copy(JUMANJI_DIR / "board_00000.kicad_pcb", pcb)

    pro = json.loads((JUMANJI_DIR / "board_00000.kicad_pro").read_text())
    base = pro["net_settings"]["classes"][0]
    pro["net_settings"]["classes"] = [
        dict(base, name="Default", clearance=0.2, track_width=0.3,
             via_diameter=0.6, via_drill=0.3),
        dict(base, name="Fast", clearance=0.25, track_width=0.35,
             via_diameter=0.8, via_drill=0.4),
    ]
    pro["net_settings"]["netclass_patterns"] = [
        {"netclass": "Fast", "pattern": "NET2"},
    ]
    pro["board"]["design_settings"]["rules"].update({
        "min_clearance": 0.22,          # > Default's 0.2 → clamp case
        "min_track_width": 0.1,
        "min_via_diameter": 0.7,        # > Default's 0.6 → clamp case
        "min_through_hole_diameter": 0.2,
    })
    (tmp / "b.kicad_pro").write_text(json.dumps(pro, indent=2))
    return pcb


@pytest.mark.skipif(not _binding_available(),
                    reason="kicad_rl_router binding not built")
class TestPCBWorldIntegration:

    @pytest.fixture
    def env_on(self, multiclass_board):
        from pcb_world.core.env import PCBWorld
        env = PCBWorld(board_path=str(multiclass_board), max_steps=5,
                       net_constraint_obs=True)
        yield env
        env.close()

    @pytest.fixture
    def env_off(self, multiclass_board):
        from pcb_world.core.env import PCBWorld
        env = PCBWorld(board_path=str(multiclass_board), max_steps=5)
        yield env
        env.close()

    def test_knob_off_keeps_legacy_obs(self, env_off):
        """Default off: no ``constraints`` key anywhere, even on a board
        that DOES declare custom netclasses (byte-identical legacy obs)."""
        obs, _ = env_off.reset()
        for net in obs["board_static"]["nets"].values():
            assert "constraints" not in net

    def test_knob_on_per_net_values(self, env_on):
        """NET2 carries the Fast class values; every other net carries the
        Default values with the BDS via clamp applied."""
        obs, _ = env_on.reset()
        nets = obs["board_static"]["nets"]
        assert set(nets["net_2"]["constraints"]) == CONSTRAINT_KEYS
        assert nets["net_2"]["constraints"] == pytest.approx(_FAST_VALS)
        for key, net in nets.items():
            if key == "net_2":
                continue
            assert net["constraints"] == pytest.approx(_DEFAULT_VALS), key

    def test_obs_matches_engine_push(self, env_on, monkeypatch):
        """The observed values equal what net_select pushes into the
        engine — the policy sees the widths it will actually route with."""
        env_on.reset()
        pushed = {}
        monkeypatch.setattr(env_on._engine, "set_track_width",
                            lambda v: pushed.__setitem__("track_width", v))
        monkeypatch.setattr(env_on._engine, "set_via_diameter",
                            lambda v: pushed.__setitem__("via_diameter", v))
        monkeypatch.setattr(env_on._engine, "set_via_drill",
                            lambda v: pushed.__setitem__("via_drill", v))

        env_on.step({"action_type": 0, "net_id": 2})  # 0 = ACT_NET_SELECT

        expected = env_on.board_static["nets"]["net_2"]["constraints"]
        for k in ("track_width", "via_diameter", "via_drill"):
            assert pushed[k] == pytest.approx(expected[k]), k

    def test_set_target_nets_refills_constraints(self, env_on):
        """The set_target_nets rebuild path re-populates constraints for
        the surviving routable nets."""
        env_on.reset()
        env_on.set_target_nets({1, 2})
        nets = env_on.board_static["nets"]
        assert set(nets) == {"net_1", "net_2"}
        assert nets["net_2"]["constraints"] == pytest.approx(_FAST_VALS)
        assert nets["net_1"]["constraints"] == pytest.approx(_DEFAULT_VALS)

    def test_knob_off_indexed_all_none(self, multiclass_board):
        """Knob off + indexed format: the net_constraints column stays the
        legacy all-None passthrough."""
        from pcb_world.core.env import PCBWorld
        env = PCBWorld(board_path=str(multiclass_board), max_steps=5,
                       obs_format="indexed")
        try:
            obs, _ = env.reset()
            assert all(c is None
                       for c in obs["board_static"]["net_constraints"])
        finally:
            env.close()

    def test_indexed_format_carries_constraints(self, multiclass_board):
        from pcb_world.core.env import PCBWorld
        env = PCBWorld(board_path=str(multiclass_board), max_steps=5,
                       net_constraint_obs=True, obs_format="indexed")
        try:
            obs, _ = env.reset()
            bs = obs["board_static"]
            by_code = dict(zip([int(c) for c in bs["net_code"]],
                               bs["net_constraints"]))
            assert by_code[2] == pytest.approx(_FAST_VALS)
            for code, vals in by_code.items():
                if code == 2:
                    continue
                assert vals == pytest.approx(_DEFAULT_VALS), code
        finally:
            env.close()

    def test_tokenizer_net_channels_pin_board_scale_norm(self, env_on):
        """The NET token's tw/cl/vd channels are exactly
        ``clamped / norm_scale * 100`` — norm_scale (the exact bbox
        half-extent) recomputed here from the board bbox, independently of
        the tokenizer's own NormContext, so a board-scale normalization
        regression breaks numerically rather than cancelling out."""
        from methods.rl_agent.models.v1.encoding import _sorted_net_keys
        from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer

        obs, _ = env_on.reset()
        bs = obs["board_static"]
        norm_scale = max(bs["bbox_w"], bs["bbox_h"]) / 2

        walk = BatchedStateTokenizer(
            d_model=32, n_freq=4, mlp_hidden=16)._walk_obs([obs])
        net_tw, net_cl, net_vd = walk["net"][0], walk["net"][1], walk["net"][2]

        net_keys = _sorted_net_keys(bs["nets"])
        assert net_tw.shape[0] == len(net_keys) > 0
        for row, nk in enumerate(net_keys):
            vals = _FAST_VALS if nk == "net_2" else _DEFAULT_VALS
            assert net_tw[row, 0] == pytest.approx(
                vals["track_width"] / norm_scale * 100), nk
            assert net_cl[row, 0] == pytest.approx(
                vals["clearance"] / norm_scale * 100), nk
            assert net_vd[row, 0] == pytest.approx(
                vals["via_diameter"] / norm_scale * 100), nk

    def _scripted_episode(self, board_path, **env_kwargs):
        """Reset + fixed 3-action routing script on net_1 (net_select →
        start_route at pad A → make_line to pad B). Returns the reset obs
        and a per-step trace (mask before any action, then reward /
        terminated / truncated / action_success / mask per step). Envs run
        SEQUENTIALLY on purpose: each construction re-seeds the process-
        global KIID stream (engine_seed default), whereas coexisting envs
        would split one stream and could flip PNS tie-breaks (the
        nondeterminism test_reset_kiid_determinism.py pins down)."""
        from pcb_world.core.action_schema import (
            ACT_MAKE_LINE,
            ACT_NET_SELECT,
            ACT_START_ROUTE,
        )
        from pcb_world.core.env import PCBWorld

        env = PCBWorld(board_path=str(board_path), max_steps=5, **env_kwargs)
        try:
            obs0, _ = env.reset()
            pads = obs0["board_static"]["nets"]["net_1"]["pads"]
            assert len(pads) >= 2, "script needs two net_1 pads"
            (ax, ay), (bx, by) = [
                p["center"]["xy"] for p in list(pads.values())[:2]
            ]
            script = [
                {"action_type": ACT_NET_SELECT, "net_id": 1},
                {"action_type": ACT_START_ROUTE, "x_mm": ax, "y_mm": ay,
                 "layer": 1},
                {"action_type": ACT_MAKE_LINE, "x_mm": bx, "y_mm": by,
                 "layer": 1, "routing_mode": 2},
            ]
            trace = [{"mask": env.action_masks().tolist()}]
            for act in script:
                _, reward, terminated, truncated, info = env.step(act)
                trace.append({
                    "reward": reward, "terminated": terminated,
                    "truncated": truncated,
                    "action_success": info["action_success"],
                    "mask": env.action_masks().tolist(),
                })
            return obs0, trace
        finally:
            env.close()

    def test_knob_on_off_behaviour_equivalence(self, multiclass_board):
        """Same board + seed + action script: the knob changes ONLY the
        ``constraints`` obs field — reset obs (minus that key), rewards,
        termination and action masks stay identical (regression guard for
        the audited "observation-only" contract)."""
        obs_off, trace_off = self._scripted_episode(multiclass_board)
        obs_on, trace_on = self._scripted_episode(
            multiclass_board, net_constraint_obs=True)

        stripped = copy.deepcopy(obs_on)
        for net in stripped["board_static"]["nets"].values():
            net.pop("constraints")  # KeyError = knob-on obs lost the key
        assert stripped == obs_off

        # The script must actually route — an all-fail trace would make
        # the equivalence below vacuous.
        assert all(s["action_success"] for s in trace_off[1:])
        assert trace_on == trace_off


# ---------------------------------------------------------------------------
# 4. Eval drift guard: ckpt-trained knob vs eval env (eval_transformer)
# ---------------------------------------------------------------------------

class _StubPolicy:
    """Minimal ``eval_transformer`` policy stand-in (no torch module)."""
    training = False

    def __init__(self, net_constraint_obs=None):
        # Unstamped (None) mimics a live training policy — the stamp only
        # exists on checkpoint-loaded policies (loader._load_policy).
        if net_constraint_obs is not None:
            self.net_constraint_obs = net_constraint_obs

    def eval(self):
        return self

    def train(self):
        return self


class TestEvalTransformerDriftGuard:
    """``eval_transformer`` must refuse an env whose ``net_constraint_obs``
    disagrees with the loaded checkpoint's stamp — both directions are
    silent obs drift (resolved NET channels vs constant 0)."""

    def _run(self, policy, env_kwargs, monkeypatch):
        import torch

        import eval.rollout.rl as rl_mod
        from methods._shared.board_loader import BoardSpec
        from methods.rl_agent.wrappers import factory

        class _Pool:
            def close(self):
                pass

        # Router-free: on match the driver reaches pool construction + the
        # batch loop, both stubbed out; on mismatch it raises before either.
        monkeypatch.setattr(factory, "make_decoder_env_pool",
                            lambda *a, **k: _Pool())
        monkeypatch.setattr(rl_mod, "_run_one_batch", lambda **kw: [])
        boards = [BoardSpec(index=0, board_id="b0", path="missing.kicad_pcb")]
        return rl_mod.eval_transformer(
            policy, torch.device("cpu"), boards,
            env_kwargs=env_kwargs, n_rollouts=1, n_envs=1,
            base_seed=0, max_steps=4, skip_aggregation=True,
        )

    @pytest.mark.parametrize("ckpt_on", [True, False])
    def test_mismatch_raises_both_directions(self, ckpt_on, monkeypatch):
        policy = _StubPolicy(net_constraint_obs=ckpt_on)
        with pytest.raises(RuntimeError, match="net_constraint_obs mismatch"):
            self._run(policy, {"net_constraint_obs": not ckpt_on}, monkeypatch)

    def test_missing_env_key_counts_as_off(self, monkeypatch):
        """Hand-built env_kwargs without the key = factory default (off) —
        a knob-trained ckpt must still fail loud."""
        policy = _StubPolicy(net_constraint_obs=True)
        with pytest.raises(RuntimeError, match="net_constraint_obs mismatch"):
            self._run(policy, {}, monkeypatch)

    @pytest.mark.parametrize("on", [True, False])
    def test_match_passes(self, on, monkeypatch):
        result = self._run(_StubPolicy(net_constraint_obs=on),
                           {"net_constraint_obs": on}, monkeypatch)
        assert result.per_rollout == []

    def test_unstamped_policy_skips_guard(self, monkeypatch):
        """A live training policy (no stamp — trainer inline val) is exempt:
        its env_kwargs come from the same training args by construction."""
        result = self._run(_StubPolicy(), {"net_constraint_obs": True},
                           monkeypatch)
        assert result.per_rollout == []
