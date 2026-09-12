"""End-to-end Indexed Obs equivalence on a REAL board (needs the C++ router).

Two ``make_decoder_env`` wrappers on the same fixture board — one
``obs_format="json"``, one ``obs_format="indexed"`` — driven through the
same seeded scripted action sequence. Asserts, at every step:

  * ``arrays_to_dict(indexed obs)`` is byte-identical JSON to the json
    env's obs (lossless producer contract on real geometry),
  * rewards / terminations / all three mask families match exactly,
  * tokenizer outputs on the collected trajectories are ``torch.equal``.

Also pins the env's IR-based table builder to the dict converter on
real-board data (same tables bit-for-bit).
"""

from __future__ import annotations

import gc
import json
import os

import numpy as np
import pytest
import torch

from methods.rl_agent.wrappers.factory import make_decoder_env
from methods.rl_agent.models.v1.tokenizer import BatchedStateTokenizer
from pcb_world.core.action_schema import ACT_NET_SELECT
from pcb_world.core.indexed_obs import arrays_to_dict, dict_to_arrays, is_indexed

BOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "simple_obstacle_board.kicad_pcb"
)

N_STEPS = 30


def _scripted_rollout(obs_format: str):
    """Deterministic rollout; returns per-step records + raw obs list."""
    # The factory requires the whole env-contract surface (no signature
    # defaults) — build it from the schema and override what this test varies.
    from configs.loader.schema import RLEnvConfig

    env = make_decoder_env(
        BOARD_PATH,
        **{**RLEnvConfig().to_pool_kwargs(), "max_steps": 50, "seed": 123,
           "policy_net_select": True, "obs_format": obs_format},
    )
    rng = np.random.default_rng(7)
    obs, _ = env.reset()
    obs_json, rewards, masks, raw_obs = [], [], [], []
    for _ in range(N_STEPS):
        am = env.action_masks()
        nvm = env.net_valid_mask()
        sp = env.start_route_pointer_indices()
        raw_obs.append(obs)
        # Digest = env-core obs only. The RL wrapper always attaches the
        # "_masks" annotation (ndarrays — not JSON-safe, and arrays_to_dict's
        # fixed key set drops it on the indexed side); masks are compared
        # explicitly below via am/nvm/sp, so strip it from the JSON digest.
        d = arrays_to_dict(obs)
        obs_json.append(json.dumps({k: v for k, v in d.items() if k != "_masks"}))
        masks.append((am.tolist(), nvm.tolist(), sp.tolist()))

        valid = np.flatnonzero(am)
        at = int(rng.choice(valid))
        if at == ACT_NET_SELECT:
            valid_nets = np.flatnonzero(nvm)
            ptr = int(rng.choice(valid_nets)) if len(valid_nets) else 0
        else:
            n_cand = max(len(env._cand_mm), 1)
            ptr = int(rng.integers(0, n_cand))
        obs, r, term, trunc, _ = env.step(np.array([at, ptr, 2], dtype=np.int64))
        rewards.append(float(r))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    del env
    gc.collect()
    return obs_json, rewards, masks, raw_obs


@pytest.fixture(scope="module")
def rollout_pair():
    json_run = _scripted_rollout("json")
    idx_run = _scripted_rollout("indexed")
    return json_run, idx_run


class TestIndexedObsEndToEnd:
    def test_formats_actually_differ(self, rollout_pair):
        (_, _, _, raw_j), (_, _, _, raw_i) = rollout_pair
        assert not is_indexed(raw_j[0])
        assert all(is_indexed(o) for o in raw_i)

    def test_obs_byte_identical_every_step(self, rollout_pair):
        (js_j, _, _, _), (js_i, _, _, _) = rollout_pair
        for t, (a, b) in enumerate(zip(js_j, js_i)):
            assert a == b, f"obs JSON diverged at step {t}"

    def test_rewards_and_masks_identical(self, rollout_pair):
        (_, rew_j, m_j, _), (_, rew_i, m_i, _) = rollout_pair
        assert rew_j == rew_i
        assert m_j == m_i

    def test_tokenizer_bit_identical_on_trajectory(self, rollout_pair):
        (_, _, _, raw_j), (_, _, _, raw_i) = rollout_pair
        torch.manual_seed(0)
        tok = BatchedStateTokenizer(d_model=64, n_freq=8)
        tok.eval()
        w = torch.randn(16, 64)
        with torch.no_grad():
            r = tok(raw_j[:8], action_type_weight=w)
            i = tok(raw_i[:8], action_type_weight=w)
        assert torch.equal(r.token_embeddings, i.token_embeddings)
        assert torch.equal(r.seq_lens, i.seq_lens)
        assert torch.equal(r.net_indices, i.net_indices)
        assert torch.equal(r.cand_indices, i.cand_indices)
        assert torch.equal(r.slot_ids, i.slot_ids)
        assert r.cand_mm_list == i.cand_mm_list

    def test_env_tables_equal_dict_converter(self, rollout_pair):
        """Env's IR-built tables == dict_to_arrays(json obs) tables, per key."""
        (_, _, _, raw_j), (_, _, _, raw_i) = rollout_pair
        for t in (0, 5, N_STEPS - 1):
            ref = dict_to_arrays(raw_j[t])
            got = raw_i[t]
            for group in ("board_static", "routing_geometry"):
                ref_g, got_g = ref[group], got[group]
                # "_"-prefixed keys are runtime memos (e.g. the tokenizer's
                # "_walk_cache"), not part of the table contract.
                assert ({k for k in ref_g if not k.startswith("_")}
                        == {k for k in got_g if not k.startswith("_")}), group
                for k, v in ref_g.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, np.ndarray):
                        assert np.array_equal(v, got_g[k]), f"{group}.{k}@{t}"
                    else:
                        assert v == got_g[k], f"{group}.{k}@{t}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
