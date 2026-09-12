"""Incremental action decode (state-K/V cache) — machinery equivalence + ckpt.

The 2-zone prefix-LM mask blocks state→action attention, so the state pass'
per-layer K/V can be cached (``KiCadRLModel._run_transformer(...,
return_cache=True)``) and the 1-2 appended action tokens decoded incrementally
(``_decode_appended``) instead of re-running the full stack per pass —
rollout 3→1 full passes, update 2→1 (backward shrinks with the graph).

The model-level multipass reference path was REMOVED (2026-07-16) after the
curve-A/B + equivalence sign-off; incremental decode is the only model path.
The oracle survives at the machinery level: ``_run_transformer`` on the
concatenated ``[state, action-tokens]`` sequence is the exact full-rerun
reference that ``_decode_appended`` must match, forward AND gradients, in
float64 — including its block-structured ``appended_mask`` form, which packs
every independent action branch of ``factored_action_logits`` into one decode.
Model-level guards: ckpt round-trip (old multipass-era checkpoints load and
run identically — the switch was never a state_dict entry) + an idle-batch
smoke over the degenerate cand-pool branch.

No C++ dependency — pure PyTorch. CPU always runs; CUDA runs too when present.
"""

from __future__ import annotations

import pytest
import torch

from methods.rl_agent.models.v1.net import (
    ACT_FINISH,
    ACT_MAKE_LINE,
    ACT_MAKE_VIA,
    ACT_NET_SELECT,
    NUM_ACTION_TYPES,
    SLOT_USAGE,
    KiCadRLModel,
)
from tests._mock_obs import make_mock_obs


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _opened_model(device: str, same_net_bias: bool = True) -> KiCadRLModel:
    """Small model with ReZero gates OPENED.

    ReZero residual gates init to 0, which gates the attention sublayer out of
    the output entirely — an ungated model would make every comparison here
    vacuous (identity passthrough). Same pattern as test_bias_absorption.
    """
    torch.manual_seed(0)
    m = KiCadRLModel(
        d_model=32, n_heads=4, n_layers=3, d_ff=64, max_seq_len=2000,
        n_freq=4, use_critic=True, same_net_bias=same_net_bias,
    ).to(device)
    with torch.no_grad():
        if same_net_bias:
            m.same_net_bias.alpha.copy_(torch.tensor([-0.3, 0.7, -0.5, 1.1]))
        for layer in m.layers:
            layer.res_attn.alpha.fill_(0.7)
            layer.res_ff.alpha.fill_(0.5)
    return m


def _varied_batch() -> list[dict]:
    # Different net counts + routing states → ragged seq_lens (padding rows)
    # and a non-trivial same-net slot structure.
    return [
        make_mock_obs(
            n_nets=nn, pads_per_net=2, n_ratsnest_per_net=2,
            is_routing=(i % 2 == 0), current_net_phase=1, current_layer=1,
            n_tracks=i, n_vias=i // 2,
        )
        for i, nn in enumerate([2, 4, 6, 3])
    ]


# ---------------------------------------------------------------------------
# Pass machinery — float64, strict
# ---------------------------------------------------------------------------
class TestPassMachineryFP64:
    """``_decode_appended`` is an exact rewrite of the full rerun."""

    def _inputs(self, device: str, seed: int = 1):
        torch.manual_seed(seed)
        B, L, d = 3, 17, 32
        embs = torch.randn(B, L, d, dtype=torch.float64, device=device)
        kpm = torch.zeros(B, L, dtype=torch.bool, device=device)
        kpm[0, 15:] = True  # ragged real lengths; row 2 stays full
        kpm[1, 12:] = True
        slot_ids = torch.randint(-1, 4, (B, L), device=device)
        e1 = torch.randn(B, 1, d, dtype=torch.float64, device=device)
        e2 = torch.randn(B, 1, d, dtype=torch.float64, device=device)
        return B, L, embs, kpm, slot_ids, e1, e2

    @pytest.mark.parametrize("device", _devices())
    def test_forward_matches_full_rerun(self, device):
        m = _opened_model(device).double()
        B, L, embs, kpm, slot_ids, e1, e2 = self._inputs(device)

        # Reference: one full pass over [state, e1, e2] with the 2-zone mask.
        full = torch.cat([embs, e1, e2], dim=1)
        kpm_full = torch.cat(
            [kpm, torch.zeros(B, 2, dtype=torch.bool, device=device)], dim=1,
        )
        H_full = m._run_transformer(full, L, kpm_full, slot_ids=slot_ids)

        # Incremental: state pass + decode (both sequential and fused).
        H_state, cache = m._run_transformer(
            embs, L, kpm, slot_ids=slot_ids, return_cache=True,
        )
        # Zone property + cache validity: state rows are unchanged by the
        # appended tokens.
        assert torch.allclose(H_state, H_full[:, :L], atol=1e-12)

        h1, cache1 = m._decode_appended(cache, e1)
        h2, _ = m._decode_appended(cache1, e2)
        assert torch.allclose(h1[:, 0], H_full[:, L], atol=1e-12)
        assert torch.allclose(h2[:, 0], H_full[:, L + 1], atol=1e-12)

        h12, _ = m._decode_appended(cache, torch.cat([e1, e2], dim=1))
        assert torch.allclose(h12[:, 0], H_full[:, L], atol=1e-12)
        assert torch.allclose(h12[:, 1], H_full[:, L + 1], atol=1e-12)

    @pytest.mark.parametrize("device", _devices())
    def test_branch_mask_matches_per_branch_full_rerun(self, device):
        """A block-structured ``appended_mask`` packs INDEPENDENT branches into
        one decode: every appended hidden must equal the per-branch full rerun.

        This is what ``factored_action_logits`` relies on — n_at root tokens and
        one point token per (root, candidate) pair, all sharing the state K/V
        prefix, each blind to the other branches.
        """
        m = _opened_model(device).double()
        B, L, embs, kpm, slot_ids, _, _ = self._inputs(device, seed=3)
        d = embs.size(-1)
        n_at, n_cand = 2, 3
        torch.manual_seed(4)
        at = torch.randn(B, n_at, d, dtype=torch.float64, device=device)
        pt = torch.randn(B, n_cand, d, dtype=torch.float64, device=device)

        # Fused: [at(0..n_at) | point(k) for each (a, k), a-outer].
        n_new = n_at + n_at * n_cand
        new = torch.cat(
            [at, pt.unsqueeze(1).expand(B, n_at, n_cand, d).reshape(B, -1, d)], dim=1,
        )
        app = torch.full((n_new, n_new), float("-inf"), device=device,
                         dtype=torch.float64)
        app.fill_diagonal_(0.0)
        app[
            torch.arange(n_at, n_new, device=device),
            torch.arange(n_at, device=device).repeat_interleave(n_cand),
        ] = 0.0
        _, cache = m._run_transformer(
            embs, L, kpm, slot_ids=slot_ids, return_cache=True,
        )
        h_new, _ = m._decode_appended(cache, new, appended_mask=app)

        # Reference: one full rerun per branch.
        def full(extra: torch.Tensor) -> torch.Tensor:      # extra: (B, n, d)
            seq = torch.cat([embs, extra], dim=1)
            kpm_e = torch.cat(
                [kpm, torch.zeros(B, extra.size(1), dtype=torch.bool, device=device)],
                dim=1,
            )
            return m._run_transformer(seq, L, kpm_e, slot_ids=slot_ids)

        for a in range(n_at):
            assert torch.allclose(h_new[:, a], full(at[:, a:a + 1])[:, L], atol=1e-12)
            for k in range(n_cand):
                ref = full(torch.cat([at[:, a:a + 1], pt[:, k:k + 1]], dim=1))
                assert torch.allclose(
                    h_new[:, n_at + a * n_cand + k], ref[:, L + 1], atol=1e-12,
                ), (a, k)

    @pytest.mark.parametrize("device", _devices())
    def test_extend_cache_false_matches_and_returns_no_cache(self, device):
        """``extend_cache=False`` changes only whether the prefix cache is
        rebuilt — the hidden states are identical. It exists because the
        one-shot decodes discard the cache, and building it allocates two
        ``(B, H, L + n_new, d_head)`` tensors per layer."""
        m = _opened_model(device).double()
        B, L, embs, kpm, slot_ids, _, _ = self._inputs(device, seed=5)
        d = embs.size(-1)
        torch.manual_seed(6)
        new = torch.randn(B, 2, d, dtype=torch.float64, device=device)
        _, cache = m._run_transformer(
            embs, L, kpm, slot_ids=slot_ids, return_cache=True,
        )
        h_keep, cache2 = m._decode_appended(cache, new)
        h_drop, none_cache = m._decode_appended(cache, new, extend_cache=False)
        assert none_cache is None
        assert cache2 is not None
        assert torch.allclose(h_keep, h_drop, atol=1e-12)

    @pytest.mark.parametrize("device", _devices())
    def test_grads_match_multipass_graph(self, device):
        """Gradient equivalence against the exact multipass graph shape.

        Mirrors evaluate_actions_and_value's structure: multipass consumes
        pass-1 state rows (values / point_tok) and recomputes the state inside
        pass 2 (weights contribute through BOTH passes); incremental shares
        one state pass. Total parameter gradients must be identical
        (linearity of differentiation over the duplicated subgraph).
        """
        results = {}
        for mode in ("multipass", "incremental"):
            m = _opened_model(device).double()
            m.zero_grad(set_to_none=True)
            B, L, embs_base, kpm, slot_ids, e1, _ = self._inputs(device, seed=2)
            # Make the state embeddings depend on a leaf so the tokenizer-side
            # grad path (embedding consumed by every pass) is exercised too.
            leaf = embs_base.clone().requires_grad_(True)
            embs = leaf * 1.0

            if mode == "incremental":
                H1, cache = m._run_transformer(
                    embs, L, kpm, slot_ids=slot_ids, return_cache=True,
                )
            else:
                H1 = m._run_transformer(embs, L, kpm, slot_ids=slot_ids)
            # "values" read + "point_tok" gather from the pass-1 output,
            # exactly like evaluate_actions_and_value.
            values_like = H1[:, 2]                      # (B, d)
            point_tok = H1[:, 5] + 0.3                  # (B, d)
            at_tok = e1[:, 0] * 0.7                     # (B, d)
            new = torch.stack([at_tok, point_tok], dim=1)

            if mode == "incremental":
                h_new, _ = m._decode_appended(cache, new)
                h_sod = H1[:, L - 1]
                h_at, h_pt = h_new[:, 0], h_new[:, 1]
            else:
                full = torch.cat([embs, new], dim=1)
                kpm_full = torch.cat(
                    [kpm, torch.zeros(B, 2, dtype=torch.bool, device=device)],
                    dim=1,
                )
                H_full = m._run_transformer(
                    full, L, kpm_full, slot_ids=slot_ids,
                )
                h_sod = H_full[:, L - 1]
                h_at, h_pt = H_full[:, L], H_full[:, L + 1]

            loss = (
                values_like.pow(2).sum()
                + h_sod.sum()
                + h_at.pow(2).sum()
                + (h_pt * 0.5).sum()
            )
            loss.backward()
            grads = {
                n: p.grad.detach().clone()
                for n, p in m.named_parameters()
                if p.grad is not None
            }
            results[mode] = (loss.detach(), grads, leaf.grad.detach().clone())

        loss_m, grads_m, leaf_m = results["multipass"]
        loss_i, grads_i, leaf_i = results["incremental"]
        assert torch.allclose(loss_m, loss_i, atol=1e-10)
        assert torch.allclose(leaf_m, leaf_i, atol=1e-10)
        assert grads_m.keys() == grads_i.keys()
        for name in grads_m:
            assert torch.allclose(grads_m[name], grads_i[name], atol=1e-10), name
        # Guard against a vacuous test: the backbone must actually get grads.
        assert grads_m["layers.0.attn.qkv_proj.weight"].abs().max() > 1e-6


# ---------------------------------------------------------------------------
# Full model — ckpt compat + degenerate-branch smoke (single path)
# ---------------------------------------------------------------------------
class TestModel:
    @pytest.mark.parametrize("device", _devices())
    def test_idle_batch(self, device):
        # No ratsnest/tracks/vias and not routing → smallest cand/state pools;
        # exercises the dummy point_tok branch (empty cand pool).
        m = _opened_model(device)
        obs = [
            make_mock_obs(
                n_nets=2, pads_per_net=2, n_ratsnest_per_net=0,
                n_tracks=0, n_vias=0, is_routing=False, current_net_phase=0,
            )
            for _ in range(3)
        ]
        # Idle states only admit net_select (cand-based actions would hit an
        # empty/degenerate pointer pool — the env masks them in production).
        am = torch.zeros(3, NUM_ACTION_TYPES, dtype=torch.bool, device=device)
        am[:, ACT_NET_SELECT] = True
        a, lp, v = m.act_and_value(obs, deterministic=True, action_masks=am)
        assert a.shape == (3, 3) and (a[:, 0] == ACT_NET_SELECT).all()
        assert torch.isfinite(lp).all() and torch.isfinite(v).all()

    def test_ckpt_roundtrip(self):
        # ckpt hard requirement: the (removed) decode-mode switch was never a
        # parameter/buffer, so multipass-era checkpoints are byte-identical —
        # they load strict and reproduce identical outputs on the incremental
        # path.
        src = _opened_model("cpu")
        assert not any("incremental_decode" in k for k in src.state_dict())

        obs = _varied_batch()
        acts, _ = src.act(obs, deterministic=True)
        dst = KiCadRLModel(
            d_model=32, n_heads=4, n_layers=3, d_ff=64, max_seq_len=2000,
            n_freq=4, use_critic=True, same_net_bias=True,
        )
        missing, unexpected = dst.load_state_dict(src.state_dict(), strict=True)
        assert not missing and not unexpected

        with torch.no_grad():
            lp_s, _, v_s = src.evaluate_actions_and_value(obs, acts)
            lp_d, _, v_d = dst.evaluate_actions_and_value(obs, acts)
        assert torch.allclose(lp_s, lp_d, atol=1e-5)
        assert torch.allclose(v_s, v_d, atol=1e-5)


# ---------------------------------------------------------------------------
# Factored prior EXACT post-pointer mode (factored_action_logits Pass-2)
# ---------------------------------------------------------------------------
class TestFactoredExactMode:
    """``factored_action_logits`` must reconstruct the SAME joint log-prob as
    ``evaluate_actions_and_value`` — the mode factor for make_line/make_via reads
    the post-pointer hidden (``mode_pt_logits``, Pass-2), so the pre-pointer
    approximation (``mode_at_logits``) is never used for them."""

    def _obs(self):
        return [make_mock_obs(
            n_nets=4, pads_per_net=2, n_ratsnest_per_net=2, is_routing=True,
            current_net_phase=1, current_layer=1, n_tracks=3, n_vias=1,
        )]

    @staticmethod
    def _factored_joint(out, acts, *, use_pt):
        """P(at)+P(ptr|at)+P(mode|·) per action from the factored logits.

        ``use_pt`` selects the post-pointer mode (mode_pt_logits) for pointer+mode
        types; False forces the pre-pointer approximation (mode_at_logits)."""
        at_lp = torch.log_softmax(out["at_logits"][0], dim=-1)
        ptr_lp = torch.log_softmax(out["ptr_logits"][0], dim=-1)
        mode_at_lp = torch.log_softmax(out["mode_at_logits"][0], dim=-1)
        mode_pt_lp = (
            torch.log_softmax(out["mode_pt_logits"][0], dim=-1)
            if out.get("mode_pt_logits") is not None else None
        )
        vals = []
        for at, ptr, mode in acts:
            lp = at_lp[at]
            if bool(SLOT_USAGE[at, 0]):
                lp = lp + ptr_lp[at, ptr]
            if bool(SLOT_USAGE[at, 1]):
                if use_pt and bool(SLOT_USAGE[at, 0]) and mode_pt_lp is not None:
                    lp = lp + mode_pt_lp[at, ptr, mode]
                else:
                    lp = lp + mode_at_lp[at, mode]
            vals.append(lp)
        return torch.stack(vals)

    @pytest.mark.parametrize("device", _devices())
    def test_factored_joint_matches_evaluate(self, device):
        m = _opened_model(device)     # float32 (real-obs tokenizer emits float32)
        obs = self._obs()
        out = m.factored_action_logits(obs)
        assert out["mode_pt_logits"] is not None
        K = out["ptr_logits"].shape[-1]
        assert K >= 2, "mock obs must expose candidate pointers"

        acts = []
        for at in (ACT_MAKE_LINE, ACT_MAKE_VIA):
            for ptr in range(min(K, 3)):
                for mode in range(3):
                    acts.append((at, ptr, mode))
        for mode in range(3):                       # finish: pointerless, mode only
            acts.append((ACT_FINISH, 0, mode))

        fac = self._factored_joint(out, acts, use_pt=True)
        actions_t = torch.tensor(
            [[at, (ptr if bool(SLOT_USAGE[at, 0]) else -1), mode]
             for at, ptr, mode in acts],
            dtype=torch.int64, device=device,
        )
        with torch.no_grad():
            ev, _, _ = m.evaluate_actions_and_value(obs * len(acts), actions_t)
        # Exact factored joint == the autoregressive reference (float32: the two
        # paths differ only by full-rerun vs incremental-decode round-off).
        assert torch.allclose(fac, ev, atol=1e-4), (fac - ev).abs().max()

    @pytest.mark.parametrize("device", _devices())
    def test_matches_per_branch_full_rerun(self, device):
        """Model-level guard for the fused branch decode.

        ``factored_action_logits`` packs every (action-type, candidate) branch
        into ONE incremental decode over the cached state prefix. The reference
        here is the literal construction it replaced: one full stack pass per
        branch on ``[state | at_tok(t) | point_tok(k)]``.
        """
        m = _opened_model(device)
        obs = self._obs()
        out = m.factored_action_logits(obs)

        with torch.no_grad():
            enc = m._encode_state(obs, return_cache=False)
            embs = enc.tok_out.token_embeddings
            kpm = enc.tok_out.key_padding_mask
            slot_ids = enc.tok_out.slot_ids
            cand = enc.tok_out.cand_indices
            L, d, T = enc.n_state_max, embs.size(-1), NUM_ACTION_TYPES
            K = cand.size(1)
            assert embs.size(0) == 1 and K >= 2

            def rerun(*extra: torch.Tensor) -> torch.Tensor:
                seq = torch.cat([embs] + [e.view(1, 1, d) for e in extra], dim=1)
                kpm_e = torch.cat(
                    [kpm,
                     torch.zeros(1, len(extra), dtype=torch.bool, device=device)],
                    dim=1,
                )
                return m._run_transformer(seq, L, kpm_e, slot_ids=slot_ids)

            at_toks = m._at_token(torch.arange(T, device=device))          # (T, d)
            point_toks = torch.gather(
                enc.H_state, 1, cand.clamp(min=0).unsqueeze(-1).expand(1, K, d),
            )[0] + m.action_pos_emb[1]                                     # (K, d)

            h_at = torch.stack([rerun(at_toks[t])[0, L] for t in range(T)])
            pt_types = [t for t in range(T)
                        if bool(SLOT_USAGE[t, 0]) and bool(SLOT_USAGE[t, 1])]
            h_pt = torch.stack([
                torch.stack([rerun(at_toks[t], point_toks[k])[0, L + 1]
                             for k in range(K)])
                for t in pt_types
            ])                                                             # (P, K, d)

            exp_mode_at = m._mode_logits(h_at, None)                       # (T, 3)
            exp_mode_pt = m._mode_logits(h_pt.reshape(-1, d), None).reshape(
                len(pt_types), K, -1)
            exp_ptr = m._combined_ptr_logits(
                h_at,
                enc.H_state.expand(T, -1, -1),
                enc.tok_out.net_indices.expand(T, -1),
                cand.expand(T, -1),
                torch.arange(T, device=device) == ACT_NET_SELECT,
            )

        assert torch.allclose(out["mode_at_logits"][0], exp_mode_at, atol=1e-5)
        assert torch.allclose(out["ptr_logits"][0], exp_ptr, atol=1e-5)
        for i, t in enumerate(pt_types):
            assert torch.allclose(
                out["mode_pt_logits"][0, t], exp_mode_pt[i], atol=1e-5,
            ), t

    @pytest.mark.parametrize("device", _devices())
    def test_pruned_branches_leave_reachable_actions_exact(self, device):
        """Branch pruning must not move any REACHABLE action's log-prob.

        ``factored_action_logits`` decodes a (type, candidate) branch only to
        fill ``mode_pt_logits[type, cand]``. Types masked out of ``at_logits``
        and candidates masked out of ``ptr_logits`` are -inf there, so their
        joint is 0 and their branch is skipped — the reference is
        ``evaluate_actions_and_value`` under the SAME masks, which never took
        that shortcut.
        """
        m = _opened_model(device)
        obs = self._obs()
        T = NUM_ACTION_TYPES
        am = torch.ones(1, T, dtype=torch.bool, device=device)
        am[0, ACT_MAKE_VIA] = False           # prunes the whole make_via branch
        out = m.factored_action_logits(obs, action_masks=am)
        K = out["ptr_logits"].shape[-1]
        assert K >= 2

        acts = [(ACT_MAKE_LINE, ptr, mode)
                for ptr in range(min(K, 3)) for mode in range(3)]
        fac = self._factored_joint(out, acts, use_pt=True)
        actions_t = torch.tensor(
            [[at, ptr, mode] for at, ptr, mode in acts],
            dtype=torch.int64, device=device,
        )
        with torch.no_grad():
            ev, _, _ = m.evaluate_actions_and_value(
                obs * len(acts), actions_t,
                action_masks=am.expand(len(acts), T),
            )
        assert torch.allclose(fac, ev, atol=1e-5), (fac - ev).abs().max()
        # the pruned type falls back to the mode_at broadcast, and is -inf in
        # at_logits, so nothing reachable can read it
        assert torch.isinf(out["at_logits"][0, ACT_MAKE_VIA])

    @pytest.mark.parametrize("device", _devices())
    def test_blocked_pointer_columns_are_not_decoded(self, device):
        """A candidate blocked for every batch row is -inf in ``ptr_logits``,
        so its branch is skipped and its ``mode_pt_logits`` slot holds the
        ``mode_at`` fallback — while unblocked columns stay exact."""
        m = _opened_model(device)
        obs = self._obs()
        full = m.factored_action_logits(obs)
        K = full["ptr_logits"].shape[-1]
        assert K >= 2
        blocked = torch.tensor([[0]], dtype=torch.int64, device=device)
        out = m.factored_action_logits(obs, pointer_masks=blocked)

        assert torch.isinf(out["ptr_logits"][0, ACT_MAKE_LINE, 0])
        # blocked column -> mode_at broadcast
        assert torch.allclose(
            out["mode_pt_logits"][0, ACT_MAKE_LINE, 0],
            out["mode_at_logits"][0, ACT_MAKE_LINE], atol=1e-6,
        )
        # unblocked columns unchanged by the pruning
        assert torch.allclose(
            out["mode_pt_logits"][0, ACT_MAKE_LINE, 1:],
            full["mode_pt_logits"][0, ACT_MAKE_LINE, 1:], atol=1e-6,
        )

    @pytest.mark.parametrize("device", _devices())
    def test_approx_mode_is_genuinely_different(self, device):
        """The pre-pointer approximation must actually differ from the reference
        for make_line/make_via — proving Pass-2 (post-pointer mode) is load-bearing,
        not a no-op."""
        m = _opened_model(device)     # float32
        obs = self._obs()
        out = m.factored_action_logits(obs)
        K = out["ptr_logits"].shape[-1]
        acts = [(ACT_MAKE_VIA, ptr, mode)
                for ptr in range(min(K, 3)) for mode in range(3)]
        approx = self._factored_joint(out, acts, use_pt=False)
        actions_t = torch.tensor(
            [[at, ptr, mode] for at, ptr, mode in acts],
            dtype=torch.int64, device=device,
        )
        with torch.no_grad():
            ev, _, _ = m.evaluate_actions_and_value(obs * len(acts), actions_t)
        # h_at ≠ h_pt on an ungated model ⇒ the approximation is measurably off.
        assert (approx - ev).abs().max() > 1e-3
