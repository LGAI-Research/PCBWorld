"""Obs-semantics self-probe (ckpt-embedded walk digest) — obs_probe.py +
loader guard (_check_obs_probe) + args/weights contradiction refusal.

Pure CPU, no C++ router needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from methods.rl_agent.models.loader import (
    _check_obs_probe,
    _policy_args_for_checkpoint,
)
from methods.rl_agent.models.v1.obs_probe import build_probe_obs, probe_digest
from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer


def _tok(**kw) -> BatchedStateTokenizer:
    torch.manual_seed(kw.pop("seed", 0))
    kw.setdefault("obstacle_obs", True)
    return BatchedStateTokenizer(d_model=32, n_freq=4, **kw)


# ===================================================================
# Digest properties
# ===================================================================
class TestProbeDigest:
    def test_deterministic_and_weight_independent(self):
        """Same config → same digest, regardless of weight init (the digest
        hashes the Phase-1 CPU walk, which never touches weights)."""
        d1 = probe_digest(_tok(seed=0))
        d2 = probe_digest(_tok(seed=123))
        assert d1 == probe_digest(_tok(seed=0))
        assert d1 == d2

    def test_config_sensitive(self):
        """A tokenizer config change (different consumed obs surface) is a
        different digest — save/load rebuild the tokenizer from the SAME ckpt
        args, so this never fires for a faithful load."""
        assert probe_digest(_tok()) != probe_digest(
            _tok(obstacle_obs=False)
        )

    def test_semantics_change_detected(self, monkeypatch):
        """The nice_scale→exact-scale class: same shapes, different values.
        Simulated by flipping the x sign inside the norm context."""
        import methods.rl_agent.models.v1.tokenizer as tok_mod

        base = probe_digest(_tok())
        orig = tok_mod._compute_norm_ctx

        def flipped(bs, aug=None):
            ctx = orig(bs, aug)
            object.__setattr__(ctx, "flip_x", -ctx.flip_x)
            return ctx

        monkeypatch.setattr(tok_mod, "_compute_norm_ctx", flipped)
        assert probe_digest(_tok()) != base

    def test_json_roundtrip_stable(self):
        """The stored probe obs survives serialization without changing the
        digest (ckpts persist it as a plain dict)."""
        t = _tok()
        probe = build_probe_obs()
        assert probe_digest(t, probe) == probe_digest(
            t, json.loads(json.dumps(probe))
        )

    def test_probe_covers_every_token_type(self):
        """The probe must exercise every walk path — and the polygon keep-out
        must be FILTERED (3 obstacle-dict entries → 2 tokenized)."""
        walk = _tok()._walk_obs([build_probe_obs()])
        expected_min = {
            "board": 1, "edge": 5, "net": 2, "pad": 5, "obstacle": 2,
            "track": 1, "via": 1, "rat": 1, "drc": 2, "head": 1,
            "action_history": 1,
        }
        for key, n_min in expected_min.items():
            n = len(walk[key][0])
            assert n >= n_min, f"probe covers no/few {key}: {n} < {n_min}"
        assert len(walk[key][0]) is not None  # sanity: tuple-of-buffers shape
        assert walk["obstacle"][0].shape[0] == 2  # polygon filtered
        assert len(walk["cand_mm_list"][0]) > 0   # live-head candidates


# ===================================================================
# Loader guard
# ===================================================================
def _stub_policy() -> SimpleNamespace:
    return SimpleNamespace(tokenizer=_tok())


def _probe_record() -> dict:
    probe = build_probe_obs()
    return {"obs": probe, "digest": probe_digest(_tok(), probe)}


class TestCheckObsProbe:
    def test_matching_digest_passes_silently(self):
        policy = _stub_policy()
        _check_obs_probe({"obs_probe": _probe_record()}, policy, Path("x.pt"))
        assert policy.obs_schema_mismatch is None

    def test_missing_probe_skips_with_note(self, capsys):
        policy = _stub_policy()
        _check_obs_probe({}, policy, Path("legacy.pt"))
        assert policy.obs_schema_mismatch is None
        assert "no obs probe" in capsys.readouterr().out

    def test_mismatch_hard_errors_by_default(self, monkeypatch):
        monkeypatch.delenv("CADAGENT_ALLOW_OBS_MISMATCH", raising=False)
        rec = _probe_record()
        rec["digest"] = "0" * 64
        with pytest.raises(RuntimeError, match="Obs-semantics mismatch"):
            _check_obs_probe({"obs_probe": rec}, _stub_policy(), Path("x.pt"))

    def test_escape_hatch_warns_and_stamps(self, monkeypatch, capsys):
        monkeypatch.setenv("CADAGENT_ALLOW_OBS_MISMATCH", "1")
        rec = _probe_record()
        rec["digest"] = "0" * 64
        policy = _stub_policy()
        _check_obs_probe({"obs_probe": rec}, policy, Path("x.pt"))
        assert policy.obs_schema_mismatch is not None
        assert "WARNING" in capsys.readouterr().out

    def test_unencodable_probe_is_a_mismatch(self, monkeypatch):
        """New code that cannot even encode the stored probe = incompatible."""
        monkeypatch.delenv("CADAGENT_ALLOW_OBS_MISMATCH", raising=False)
        rec = _probe_record()
        del rec["obs"]["router_head"]  # required by the walk → encode raises
        with pytest.raises(RuntimeError, match="no longer encodes"):
            _check_obs_probe({"obs_probe": rec}, _stub_policy(), Path("x.pt"))


# ===================================================================
# args ↔ weights contradiction (shape_obs)
# ===================================================================
class TestShapeObsContradiction:
    _SHAPE_KEY = "tokenizer.vocab.shape_embed.weight"

    def test_contradiction_refused(self):
        with pytest.raises(RuntimeError, match="contradiction"):
            _policy_args_for_checkpoint({"shape_obs": True}, {})
        with pytest.raises(RuntimeError, match="contradiction"):
            _policy_args_for_checkpoint(
                {"shape_obs": False}, {self._SHAPE_KEY: object()},
            )

    def test_consistent_and_legacy_pass(self):
        assert _policy_args_for_checkpoint(
            {"shape_obs": True}, {self._SHAPE_KEY: object()},
        )["shape_obs"] is True
        assert _policy_args_for_checkpoint(
            {"shape_obs": False}, {},
        )["shape_obs"] is False
        # Absent key (pre-knob ckpt) → weights verdict, no error.
        assert _policy_args_for_checkpoint(
            {}, {self._SHAPE_KEY: object()},
        )["shape_obs"] is True
