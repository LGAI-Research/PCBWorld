"""OBSTACLE token (obstacle_obs) + boundary-shape channel (shape_obs).

Layers:
  1. shape-bucket codec (pure function).
  2. Walk-level emission: knob off = zero tokens (byte-identical stream),
     knob on = NPTH/NC-pad tokens with rule-area keepouts excluded, and the
     pad layer-span primitive reused (layer 0 -> (1, n_copper)).
  3. dict-walk vs indexed-walk bit-identity with the knobs on.
  4. Batched vs frozen per-obs reference parity with the knobs on.
  5. Additive shape channel semantics (token = base + shape_embed[bucket]).
  6. Checkpoint compat: knob-off adds no weights; 14-row entity-table
     checkpoints pad-load; shape_obs presence detection from the state_dict.
"""

from __future__ import annotations

import copy

import pytest
import torch

from methods.rl_agent.models.loader import (
    _policy_args_for_checkpoint,
    pad_legacy_entity_type_rows,
    pad_legacy_optimizer_state,
)
from methods.rl_agent.models.v1.embedding import TokenVocabulary
from methods.rl_agent.models.v1.encoding import shape_bucket_id
from methods.rl_agent.models.v1.net import KiCadRLModel
from methods.rl_agent.models.v1.spec import (
    EntityType,
    NUM_ENTITY_TYPES,
    NUM_SHAPE_BUCKETS,
)
from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer
from pcb_world.core.indexed_obs import dict_to_arrays
from tests.helpers.reference_tokenizer import StateTokenizer
from tests.test_indexed_obs import make_canonical_obs
from tests.test_indexed_tokenizer import assert_bit_identical

_SMALL = dict(d_model=64, n_freq=8)
_MODEL_SMALL = dict(d_model=32, n_freq=4, n_layers=1, d_ff=64, n_heads=2)


def _tok(seed: int = 0, **kwargs) -> BatchedStateTokenizer:
    torch.manual_seed(seed)
    tok = BatchedStateTokenizer(**_SMALL, **kwargs)
    tok.eval()
    return tok


def _obs_with_keepout_and_npth() -> dict:
    """Canonical obs (1 roundrect obstacle + 1 NC pad) + an injected NPTH
    hole (layer-0 circle) and a rule-area keepout polygon entry."""
    obs, _, _ = make_canonical_obs()
    obstacles = obs["board_static"]["obstacles"]
    obstacles["obs_npth"] = {
        "id": "O_npth",
        "center": {"id": "P_npth", "xy": [55.0, 25.0], "layer": 0},
        "width": 3.2, "height": 3.2, "layer": 0, "shape": "circle",
    }
    obstacles["obs_keepout"] = {
        "id": "O_ko",
        "center": {"id": "P_ko", "xy": [20.0, 50.0], "layer": 1},
        "width": 8.0, "height": 6.0, "layer": 1, "shape": "polygon",
    }
    return obs


# ===================================================================
# 1. Shape bucket codec
# ===================================================================
class TestShapeBuckets:
    def test_known_strings(self):
        assert shape_bucket_id("rect") == 0
        assert shape_bucket_id("roundrect") == 1
        assert shape_bucket_id("circle") == 2
        assert shape_bucket_id("oval") == 3
        for s in ("trapezoid", "chamfered_rect", "custom", "polygon"):
            assert shape_bucket_id(s) == 4, s
        assert shape_bucket_id("") == 5

    def test_unseen_string_maps_to_unknown(self):
        assert shape_bucket_id("hexagon??") == 5

    def test_bucket_range(self):
        ids = {shape_bucket_id(s) for s in (
            "rect", "roundrect", "circle", "oval", "trapezoid", "", "x",
        )}
        assert ids <= set(range(NUM_SHAPE_BUCKETS))


# ===================================================================
# 2. Walk-level emission
# ===================================================================
class TestObstacleEmission:
    def test_knob_off_emits_nothing(self):
        obs = _obs_with_keepout_and_npth()
        walk = _tok()._walk_obs([obs])
        assert len(walk["obstacle"][0]) == 0
        # pad tuple carries the always-on shape column (fixed arity).
        assert len(walk["pad"]) == 10

    def test_knob_off_stream_unchanged(self):
        """seq_lens with the knob off must not count obstacle entries."""
        obs = _obs_with_keepout_and_npth()
        off = _tok()._walk_obs([obs])
        on = _tok(obstacle_obs=True)._walk_obs([obs])
        n_obst = len(on["obstacle"][0])
        assert n_obst == 3  # roundrect + NPTH + NC pad; keepout excluded
        assert on["seq_lens"][0] == off["seq_lens"][0] + n_obst

    def test_keepout_polygon_excluded(self):
        obs = _obs_with_keepout_and_npth()
        walk = _tok(obstacle_obs=True)._walk_obs([obs])
        shape_ids = walk["obstacle"][8].tolist()
        # polygon bucket (other=4) never appears: the keepout row is skipped
        # BEFORE bucketing, and no other fixture entry is in "other".
        assert 4 not in shape_ids
        assert shape_ids == [
            shape_bucket_id("roundrect"),  # canonical obstacle
            shape_bucket_id("circle"),     # injected NPTH
            shape_bucket_id("rect"),       # NC pad (fixture default shape)
        ]

    def test_npth_layer_span_matches_thru_pad(self):
        """Layer-0 obstacle reuses the pad thru primitive (1, n_copper)."""
        obs = _obs_with_keepout_and_npth()
        walk = _tok(obstacle_obs=True)._walk_obs([obs])
        o_ls, o_le = walk["obstacle"][2], walk["obstacle"][3]
        shape_ids = walk["obstacle"][8].tolist()
        npth_row = shape_ids.index(shape_bucket_id("circle"))
        # The canonical fixture has a thru-sentinel pad (layer=0, oval):
        # its span encoding must equal the NPTH obstacle's.
        p_ls, p_le = walk["pad"][2], walk["pad"][3]
        p_shape = walk["pad"][9].tolist()
        thru_pad_row = p_shape.index(shape_bucket_id("oval"))
        assert (o_ls[npth_row] == p_ls[thru_pad_row]).all()
        assert (o_le[npth_row] == p_le[thru_pad_row]).all()

    def test_obstacle_tokens_have_no_slot(self):
        obs = _obs_with_keepout_and_npth()
        tok = _tok(obstacle_obs=True)
        walk = tok._walk_obs([obs])
        out = tok([obs])
        n_obst = len(walk["obstacle"][0])
        positions = walk["obstacle"][7].tolist()
        for pos in positions:
            assert out.slot_ids[0, pos].item() == -1
        assert n_obst == 3


# ===================================================================
# 3. dict vs indexed bit-identity (knobs on)
# ===================================================================
class TestBitIdentity:
    def test_obstacle_on(self):
        assert_bit_identical(
            _tok(obstacle_obs=True), [_obs_with_keepout_and_npth()],
        )

    def test_obstacle_and_shape_on(self):
        assert_bit_identical(
            _tok(obstacle_obs=True, shape_obs=True),
            [_obs_with_keepout_and_npth()],
        )

    def test_shape_only(self):
        assert_bit_identical(
            _tok(shape_obs=True), [_obs_with_keepout_and_npth()],
        )

    def test_mixed_batch(self):
        obs_a = _obs_with_keepout_and_npth()
        obs_b, _, _ = make_canonical_obs()
        assert_bit_identical(
            _tok(obstacle_obs=True, shape_obs=True), [obs_a, obs_b],
        )


# ===================================================================
# 4. Batched vs frozen per-obs reference parity (knobs on)
# ===================================================================
class TestReferenceParity:
    @pytest.mark.parametrize("knobs", [
        dict(obstacle_obs=True),
        dict(shape_obs=True),
        dict(obstacle_obs=True, shape_obs=True),
    ])
    def test_parity(self, knobs):
        torch.manual_seed(0)
        ref = StateTokenizer(**_SMALL, **knobs)
        torch.manual_seed(0)
        bat = BatchedStateTokenizer(**_SMALL, **knobs)
        bat.vocab.load_state_dict(ref.vocab.state_dict())
        ref.eval(); bat.eval()
        obs = _obs_with_keepout_and_npth()
        # Production always threads the policy's action_type_head weights;
        # the weightless path is a known ref/batched divergence on HIST
        # tokens unrelated to these knobs.
        atw = torch.randn(20, _SMALL["d_model"])
        with torch.no_grad():
            r = ref([obs], action_type_weight=atw)
            b = bat([obs], action_type_weight=atw)
        assert torch.equal(r.seq_lens, b.seq_lens)
        diff = (r.token_embeddings - b.token_embeddings).abs().max().item()
        assert diff < 1e-4, f"max diff {diff:.3e}"
        # slot_ids: the frozen reference returns None (pre-slot_ids API);
        # slot embedding equality is already covered by the embedding diff,
        # and slot_ids content by the dict/indexed bit-identity tests.


# ===================================================================
# 5. Additive shape channel semantics
# ===================================================================
class TestShapeChannel:
    def test_shape_is_additive(self):
        torch.manual_seed(0)
        v = TokenVocabulary(**_SMALL, shape_obs=True)
        v.eval()
        K = 3
        xy = torch.randn(K, 2); wh = torch.rand(K, 2)
        ls = torch.rand(K, 2); le = torch.rand(K, 2)
        ids = torch.tensor([0, 3, 5])
        with torch.no_grad():
            with_shape = v.encode_pad(xy, wh, ls, le, None, shape_id=ids)
            v.shape_obs = False       # consult-at-call-time gate
            base = v.encode_pad(xy, wh, ls, le, None)
            v.shape_obs = True
            delta = v.shape_embed(ids)
        assert torch.allclose(with_shape, base + delta, atol=1e-6)

    def test_shape_on_requires_ids(self):
        v = TokenVocabulary(**_SMALL, shape_obs=True)
        with pytest.raises(ValueError, match="shape_id"):
            v.encode_pad(torch.randn(2, 2), torch.rand(2, 2),
                         torch.rand(2, 2), torch.rand(2, 2), None)

    def test_obstacle_type_row_differs_from_pad(self):
        """Same geometry -> tokens differ exactly by the type-row delta."""
        torch.manual_seed(0)
        v = TokenVocabulary(**_SMALL)
        v.eval()
        xy = torch.randn(2, 2); wh = torch.rand(2, 2)
        ls = torch.rand(2, 2); le = torch.rand(2, 2)
        with torch.no_grad():
            pad_tok = v.encode_pad(xy, wh, ls, le, None)
            obst_tok = v.encode_obstacle(xy, wh, ls, le, None)
            type_delta = (
                v.entity_type_embed.weight[int(EntityType.OBSTACLE)]
                - v.entity_type_embed.weight[int(EntityType.PAD)]
            )
        assert torch.allclose(obst_tok - pad_tok,
                              type_delta.expand_as(pad_tok), atol=1e-6)


# ===================================================================
# 6. Checkpoint compat
# ===================================================================
class TestCheckpointCompat:
    def test_knob_off_adds_no_weights(self):
        torch.manual_seed(0)
        base = KiCadRLModel(**_MODEL_SMALL)
        torch.manual_seed(0)
        obst = KiCadRLModel(**_MODEL_SMALL, obstacle_obs=True)
        assert set(base.state_dict()) == set(obst.state_dict())

    def test_shape_on_adds_only_shape_embed(self):
        base = KiCadRLModel(**_MODEL_SMALL)
        on = KiCadRLModel(**_MODEL_SMALL, shape_obs=True)
        extra = set(on.state_dict()) - set(base.state_dict())
        assert extra == {"tokenizer.vocab.shape_embed.weight"}

    def test_legacy_14_row_table_pads_and_loads(self):
        torch.manual_seed(0)
        model = KiCadRLModel(**_MODEL_SMALL)
        sd = copy.deepcopy(model.state_dict())
        key = "tokenizer.vocab.entity_type_embed.weight"
        legacy_rows = sd[key][: NUM_ENTITY_TYPES - 1].clone()
        sd[key] = legacy_rows.clone()

        torch.manual_seed(1)
        target = KiCadRLModel(**_MODEL_SMALL)
        pad_legacy_entity_type_rows(sd, target)
        assert sd[key].shape[0] == NUM_ENTITY_TYPES
        target.load_state_dict(sd, strict=True)
        got = target.state_dict()[key]
        assert torch.equal(got[: NUM_ENTITY_TYPES - 1], legacy_rows)

    def test_current_table_untouched(self):
        model = KiCadRLModel(**_MODEL_SMALL)
        sd = copy.deepcopy(model.state_dict())
        before = sd["tokenizer.vocab.entity_type_embed.weight"].clone()
        pad_legacy_entity_type_rows(sd, model)
        assert torch.equal(
            sd["tokenizer.vocab.entity_type_embed.weight"], before,
        )

    def test_newer_table_refuses_downgrade(self):
        model = KiCadRLModel(**_MODEL_SMALL)
        sd = copy.deepcopy(model.state_dict())
        key = "tokenizer.vocab.entity_type_embed.weight"
        sd[key] = torch.cat([sd[key], sd[key][:1]], dim=0)  # 16 rows
        with pytest.raises(RuntimeError, match="NEWER"):
            pad_legacy_entity_type_rows(sd, model)

    def test_loader_detects_shape_obs_presence(self):
        args = dict(_MODEL_SMALL)
        on_sd = KiCadRLModel(**_MODEL_SMALL, shape_obs=True).state_dict()
        off_sd = KiCadRLModel(**_MODEL_SMALL).state_dict()
        assert _policy_args_for_checkpoint(args, on_sd)["shape_obs"] is True
        assert _policy_args_for_checkpoint(args, off_sd)["shape_obs"] is False

    def test_legacy_optimizer_moments_pad_and_step(self):
        """Resume twin of the 14-row table pad: Adam moments must be padded
        too, or the FIRST optimizer.step() after resume crashes (torch does
        no shape validation at optimizer load)."""
        key = "tokenizer.vocab.entity_type_embed.weight"
        torch.manual_seed(0)
        model = KiCadRLModel(**_MODEL_SMALL)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss = model.state_dict()[key].sum() * 0  # build moments via a real step
        for p in model.parameters():
            loss = loss + p.sum()
        loss.backward()
        opt.step()
        opt_sd = opt.state_dict()

        # Simulate a pre-OBSTACLE checkpoint: slice the entity moments to 14.
        idx = next(i for i, (n, _) in enumerate(model.named_parameters())
                   if n.endswith("entity_type_embed.weight"))
        for mkey in ("exp_avg", "exp_avg_sq"):
            opt_sd["state"][idx][mkey] = (
                opt_sd["state"][idx][mkey][: NUM_ENTITY_TYPES - 1].clone()
            )

        torch.manual_seed(1)
        target = KiCadRLModel(**_MODEL_SMALL)
        opt2 = torch.optim.AdamW(target.parameters(), lr=1e-4)
        pad_legacy_optimizer_state(opt_sd, target)
        assert opt_sd["state"][idx]["exp_avg"].shape[0] == NUM_ENTITY_TYPES
        # Padded rows are zeros (the moments a fresh row would have).
        assert opt_sd["state"][idx]["exp_avg"][-1].abs().sum() == 0
        opt2.load_state_dict(opt_sd)
        loss2 = sum(p.sum() for p in target.parameters())
        loss2.backward()
        opt2.step()  # would raise RuntimeError without the padding

    def test_current_optimizer_state_untouched(self):
        torch.manual_seed(0)
        model = KiCadRLModel(**_MODEL_SMALL)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss = sum(p.sum() for p in model.parameters())
        loss.backward()
        opt.step()
        opt_sd = opt.state_dict()
        idx = next(i for i, (n, _) in enumerate(model.named_parameters())
                   if n.endswith("entity_type_embed.weight"))
        before = opt_sd["state"][idx]["exp_avg"].clone()
        pad_legacy_optimizer_state(opt_sd, model)
        assert torch.equal(opt_sd["state"][idx]["exp_avg"], before)


class TestPolygonUpadSymmetry:
    """Indexed walk must apply the polygon filter to unconnected_pads too
    (bit-identity by construction — real pads can never carry "polygon",
    but the two walks must agree on injected/synthetic data)."""

    def _obs_with_polygon_upad(self) -> dict:
        obs, _, _ = make_canonical_obs()
        obs["board_static"]["unconnected_pads"]["upad_ko"] = {
            "id": "U_ko",
            "center": {"id": "P_uko", "xy": [22.0, 33.0], "layer": 1},
            "width": 5.0, "height": 4.0, "layer": 1, "shape": "polygon",
        }
        return obs

    def test_polygon_upad_excluded_and_bit_identical(self):
        obs = self._obs_with_polygon_upad()
        tok = _tok(obstacle_obs=True)
        walk = tok._walk_obs([obs])
        # canonical: 1 obstacle (roundrect) + 1 NC pad; polygon upad excluded
        assert len(walk["obstacle"][0]) == 2
        assert_bit_identical(tok, [obs])


def _binding_available() -> bool:
    from pcb_world.engine import engine_available
    return engine_available()   # probe only — no GPL import (import-hygiene)


@pytest.mark.skipif(not _binding_available(), reason="C++ binding not built")
class TestEndToEndObstacleTokens:
    """Board file → engine parser → env obs → OBSTACLE tokens.

    Covers the full source decomposition on one board: NPTH mounting hole
    (np_thru_hole circle) + oval slot (np_thru_hole oval) + net-less NC pad
    — token count must equal obstacles(non-polygon) + unconnected_pads, with
    per-source shape buckets.
    """

    _BOARD = """(kicad_pcb (version 20171130) (host pcbnew "synthetic")
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
    (pad 2 smd rect (at 0 5) (size 1 1) (layers F.Cu F.Paste F.Mask)
      (net 1 SIG))
    (pad NC smd rect (at -5 0) (size 1 1) (layers F.Cu F.Paste F.Mask)
      (net 0 ""))
    (pad MH np_thru_hole circle (at 5 0) (size 3 3) (drill 3)
      (layers *.Cu *.Mask) (net 0 ""))
    (pad SL np_thru_hole oval (at 5 5) (size 1 2.4) (drill oval 0.8 2.2)
      (layers *.Cu *.Mask) (net 0 ""))
  )
)
"""

    @pytest.fixture
    def env_obs(self, tmp_path):
        from pcb_world.core.env import PCBWorld
        path = tmp_path / "npth_slot_nc.kicad_pcb"
        path.write_text(self._BOARD)
        env = PCBWorld(
            str(path), use_yaml_drc_fallback=True,
            drc_config_path="configs/drc/synth_2L_v2.yaml",
        )
        try:
            obs, _ = env.reset(seed=0)
            yield obs
        finally:
            env.close()

    def test_source_decomposition(self, env_obs):
        bs = env_obs["board_static"]
        n_obst = sum(1 for o in bs["obstacles"].values()
                     if o.get("shape") != "polygon")
        n_upad = len(bs.get("unconnected_pads", {}))
        assert n_obst == 2   # NPTH circle + oval slot
        assert n_upad == 1   # net-0 NC pad
        walk = _tok(obstacle_obs=True)._walk_obs([env_obs])
        assert len(walk["obstacle"][0]) == n_obst + n_upad == 3
        shapes = sorted(walk["obstacle"][8].tolist())
        assert shapes == sorted([
            shape_bucket_id("circle"),  # mounting hole
            shape_bucket_id("oval"),    # slot
            shape_bucket_id("rect"),    # NC pad
        ])

    def test_knob_off_emits_nothing(self, env_obs):
        walk = _tok()._walk_obs([env_obs])
        assert len(walk["obstacle"][0]) == 0
