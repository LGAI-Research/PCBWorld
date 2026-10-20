"""Test factories for :class:`~methods.rl_agent.models.v1.spec.DistInputs`,
the mask payload it is stacked from, and a flat update buffer built on it.

Model forwards take ``dist=`` as a required argument; a test that does not
exercise masking builds the bundle here instead of inventing its own default.
Override single fields with ``dataclasses.replace(dist, pointer=...)``.
"""
from __future__ import annotations

import numpy as np

from methods.rl_agent.models.v1.encoding import sorted_net_codes_from_obs
from methods.rl_agent.models.v1.spec import (
    NUM_ACTIONS, NUM_ROUTING_MODES, DistInputs, Transition,
)


def payload(**overrides) -> dict[str, np.ndarray]:
    """A six-key act-time mask payload (``KiCadRLWrapper.mask_payload``) that
    restricts nothing; ``overrides`` replace single keys."""
    out = {
        "action": np.ones(NUM_ACTIONS, dtype=bool),
        "pointer": np.zeros(0, dtype=np.int64),
        "mode": np.ones(NUM_ROUTING_MODES, dtype=bool),
        "net_valid": np.zeros(0, dtype=bool),
        "offlayer": np.zeros(0, dtype=np.int64),
        "offboard": np.zeros(0, dtype=np.int64),
    }
    unknown = set(overrides) - set(out)
    if unknown:
        raise KeyError(f"not a mask payload key: {sorted(unknown)}")
    out.update(overrides)
    return out


def permissive_dist_np(n: int, net_width: int | None = None) -> DistInputs:
    """A numpy bundle of ``n`` rows that restricts nothing: every action type
    and mode legal, no blocked candidate, and — given ``net_width`` (a model
    that selects nets) — ``net_width`` valid nets per row."""
    return DistInputs.from_payloads(
        [payload(net_valid=np.ones(net_width or 0, dtype=bool))] * n,
        policy_net_select=net_width is not None,
    )


def permissive_dist(model, obs_list: list[dict]) -> DistInputs:
    """:func:`permissive_dist_np` for ``obs_list`` on ``model``'s device; when
    the model selects nets, every net in each obs's pool is valid."""
    net_width = (
        max(len(sorted_net_codes_from_obs(o)) for o in obs_list)
        if model.policy_net_select else None
    )
    return permissive_dist_np(len(obs_list), net_width).to(
        next(model.parameters()).device,
    )


def permissive_buffer(obs_list: list[dict], actions, old_log_probs, *,
                      returns: bool = True, dist: DistInputs | None = None) -> dict:
    """A flat update buffer for ``obs_list`` from the act-time ``actions`` /
    ``old_log_probs`` (torch tensors): zero old values, seeded standard-normal
    advantages (then returns, when ``returns``). ``dist`` defaults to
    :func:`permissive_dist_np`."""
    n = len(obs_list)
    rng = np.random.default_rng(0)
    buf = {
        "obs_list": obs_list,
        "transition": Transition(
            dist=permissive_dist_np(n) if dist is None else dist,
            actions=actions.cpu().numpy(),
            old_log_probs=old_log_probs.cpu().numpy().astype(np.float32),
            old_values=np.zeros(n, dtype=np.float32),
            kl_mask=None,
        ),
        "advantages": rng.standard_normal(n).astype(np.float32),
    }
    if returns:
        buf["returns"] = rng.standard_normal(n).astype(np.float32)
    return buf
