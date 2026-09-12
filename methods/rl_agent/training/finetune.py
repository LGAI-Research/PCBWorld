"""BC-initialised PPO fine-tuning helpers: reference-KL estimator and its state mask.

The trainer (``loop.py``) fine-tunes a behaviour-cloned policy with three guards
that are the standard recipe for RL after supervised initialisation:

* critic warm-up — iterations ``1..N`` update only the critic head (actor and
  backbone frozen), so the first policy gradients are not driven by a random
  value function;
* a KL(pi || pi_ref) penalty toward the frozen initial policy (AlphaStar /
  RLHF style), estimated per sample on the rollout actions with Schulman's
  ``k3`` form ``exp(r) - r - 1``, ``r = log pi_ref(a|s) - log pi(a|s)`` — an
  unbiased, non-negative, low-variance estimator that needs only the two
  log-probs the update already computes;
* a state mask on that penalty: BC demos are DRC-clean, so their observations
  carry only connectivity-type DRC tokens ("missing connection" etc.). On a
  state whose DRC tokens include a type the reference never saw (clearance,
  short, ...) the reference's output is uninformed, and pinning the policy to
  it would freeze exactly the reaction RL is meant to learn — those samples
  get KL weight 0.

Pure numpy/torch helpers; no engine, no trainer state — unit-tested in
``tests/test_bc_finetune_ppo.py``.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["KL_LOG_RATIO_CLAMP", "kl_k3", "kl_mask_from_obs", "parse_type_ids"]

#: ``log pi_ref - log pi`` is clamped to +-this before ``exp``: a sample the
#: reference finds far likelier than the policy would otherwise dominate the
#: minibatch (exp(8) ~ 3e3 is already a hard pull; beyond it the gradient is
#: noise, not signal).
KL_LOG_RATIO_CLAMP = 8.0


def parse_type_ids(spec: str | None) -> frozenset[int]:
    """``"2,4,6"`` -> ``{2, 4, 6}`` (empty/None -> empty set)."""
    if not spec:
        return frozenset()
    return frozenset(int(tok) for tok in str(spec).split(",") if tok.strip())


def kl_mask_from_obs(obs_list: list[dict], seen_type_ids: frozenset[int]) -> np.ndarray:
    """Per-sample KL weight ``(N,) float32``: 1 when every DRC token's ``type_id``
    is in ``seen_type_ids`` (or the observation has none), else 0."""
    out = np.ones(len(obs_list), dtype=np.float32)
    for i, obs in enumerate(obs_list):
        for vio in obs.get("drc_violations") or ():
            try:
                tid = int(vio.get("type_id", -1)) if isinstance(vio, dict) else -1
            except (TypeError, ValueError):
                tid = -1
            if tid not in seen_type_ids:
                out[i] = 0.0
                break
    return out


def kl_k3(ref_log_prob: torch.Tensor, new_log_prob: torch.Tensor,
          clamp: float = KL_LOG_RATIO_CLAMP) -> torch.Tensor:
    """Per-sample ``k3`` estimate of KL(pi || pi_ref) at the taken actions.

    ``r = log pi_ref(a|s) - log pi(a|s)``; ``exp(r) - r - 1`` is >= 0 with
    equality iff the two log-probs agree, and its expectation under ``pi`` is
    the KL. Gradient flows through ``new_log_prob`` only (``ref_log_prob`` is
    computed under ``no_grad`` by the caller).
    """
    log_ratio = (ref_log_prob - new_log_prob).clamp(-clamp, clamp)
    return torch.exp(log_ratio) - log_ratio - 1.0
