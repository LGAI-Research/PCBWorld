"""KiCadRLAgent — the RL inference facade (Agent layer).

Taxonomy (see :mod:`methods._shared.agent`): :class:`~methods.rl_agent.models.v1.net.
KiCadRLModel` is the Model (weights/forward/tokenizer); this class is the
Agent — it performs model-based inference, one decision step from raw env
obs dicts (+ masks) to action tensors. Episode orchestration stays with the
callers (training collectors, eval rollout, serving provider), which all
converge on this facade instead of poking the nn.Module directly.

Three call surfaces:

  * :meth:`act` — eval-path step: masks already tensorised, ``torch.no_grad``.
  * :meth:`act_and_value` — training-collection step: pure delegation
    (no grad-mode change), returns ``(actions, log_probs, values)``.
  * :meth:`act_from_pool` — eval convenience: gathers action/pointer/mode/
    net-valid masks from a vec backend's built-ins, tensorises, then acts.

The mask gather + tensorize glue itself is the module-level pair
:func:`gather_mask_arrays` / :func:`mask_arrays_to_tensors` — the single
implementation shared by the rollout primitive
(``methods/rl_agent/rollout/primitive.py``), :meth:`act_from_pool`, and the
MCTS critic probe (``policy/mcts_env.py``).

Unknown attributes delegate to the wrapped model (``__getattr__``), so the
agent is a drop-in wherever a bare ``KiCadRLModel`` is expected
(``tokenizer`` / ``policy_net_select`` / ``use_critic`` / ``eval()`` ...).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

__all__ = ["KiCadRLAgent", "gather_mask_arrays", "mask_arrays_to_tensors"]


# ---------------------------------------------------------------------------
# Mask gather + tensorize — THE shared glue (§3.4 ③: mask application is the
# agent's job). Single implementation under the rollout primitive
# (methods/rl_agent/rollout/primitive.py), act_from_pool (one-step eval), and
# the MCTS critic probe (policy/mcts_env.py).
# ---------------------------------------------------------------------------


def gather_mask_arrays(
    envs: Any,
    indices: list[int],
    *,
    policy_net_select: bool,
    obs_list: "list[dict] | None" = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Gather act-time mask arrays for ``indices``.

    ``envs`` is anything exposing ``env_method`` (``VecBackend`` pool, Ray
    backend, or the MCTS ``_SingleEnvPool`` shim) — gathered via the
    ``stack_*`` codec helpers — or a plain ``list[KiCadRLWrapper]``
    (sequential per-env queries, same action -> pointer -> mode -> nvm
    order). Returns ``(action_masks, pointer_masks, mode_masks,
    net_valid_masks|None, offlayer_masks)`` as numpy arrays;
    ``net_valid_masks`` only when ``policy_net_select``. ``offlayer_masks``
    lists the cand columns on a layer other than the router head's — consumed
    under a per-row gate on the make_line row only (see
    ``KiCadRLWrapper.offlayer_pointer_indices``).

    ``obs_list`` (aligned with ``indices``): when every obs carries the
    wrapper-embedded ``"_masks"`` payload (``KiCadRLWrapper`` attaches it on
    every step/reset), the masks are stacked locally from it — bit-identical
    to the query paths (same source arrays, same padding) with zero IPC.
    Falls through to the query paths otherwise (stubs, diagnostics).
    """
    import numpy as np

    from methods.rl_agent.models.v1.encoding import (
        stack_action_masks,
        stack_mode_masks,
        stack_net_valid_masks,
        stack_offlayer_masks,
        stack_pointer_masks,
    )

    if obs_list is not None and obs_list and all("_masks" in o for o in obs_list):
        per = [o["_masks"] for o in obs_list]
        masks = np.stack([m["action"] for m in per], axis=0)
        # Variable-K pointer indices, right-padded with -1 ((B, 0) when no
        # match) — same padding as stack_pointer_masks / the list branch.
        per_ptr = [m["pointer"] for m in per]
        K_max = max((p.shape[0] for p in per_ptr), default=0)
        ptr_masks = np.full((len(per), K_max), -1, dtype=np.int64)
        for k, p in enumerate(per_ptr):
            if p.shape[0]:
                ptr_masks[k, : p.shape[0]] = p
        mode_masks = np.stack([m["mode"] for m in per], axis=0)
        nvm = None
        if policy_net_select:
            per_nvm = [m["net_valid"] for m in per]
            M_max = max((m.shape[0] for m in per_nvm), default=0)
            nvm = np.zeros((len(per), M_max), dtype=bool)
            for k, m in enumerate(per_nvm):
                nvm[k, : m.shape[0]] = m
        # Same right-padding as the pointer block above. Older payloads without
        # the key degrade to "nothing off-layer", i.e. the pre-change behaviour.
        per_off = [m.get("offlayer") for m in per]
        if any(o is None for o in per_off):
            off_masks = np.full((len(per), 0), -1, dtype=np.int64)
        else:
            O_max = max((o.shape[0] for o in per_off), default=0)
            off_masks = np.full((len(per), O_max), -1, dtype=np.int64)
            for k, o in enumerate(per_off):
                if o.shape[0]:
                    off_masks[k, : o.shape[0]] = o
        return masks, ptr_masks, mode_masks, nvm, off_masks

    if hasattr(envs, "env_method"):
        masks = stack_action_masks(envs, indices=indices)
        ptr_masks = stack_pointer_masks(envs, indices=indices)
        mode_masks = stack_mode_masks(envs, indices=indices)
        nvm = (
            stack_net_valid_masks(envs, indices=indices)
            if policy_net_select else None
        )
        off_masks = stack_offlayer_masks(envs, indices=indices)
        return masks, ptr_masks, mode_masks, nvm, off_masks

    masks = np.stack([envs[i].action_masks() for i in indices], axis=0)
    # Variable-K pointer masks, right-padded with -1 ((B, 0) when no match).
    per_ptr = [envs[i].start_route_pointer_indices() for i in indices]
    K_max = max((p.shape[0] for p in per_ptr), default=0)
    ptr_masks = np.full((len(indices), K_max), -1, dtype=np.int64)
    for k, p in enumerate(per_ptr):
        if p.shape[0]:
            ptr_masks[k, : p.shape[0]] = p
    mode_masks = np.stack([envs[i].mode_mask() for i in indices], axis=0)
    nvm = None
    if policy_net_select:
        per_nvm = [envs[i].net_valid_mask() for i in indices]
        M_max = max((m.shape[0] for m in per_nvm), default=0)
        nvm = np.zeros((len(indices), M_max), dtype=bool)
        for k, m in enumerate(per_nvm):
            nvm[k, : m.shape[0]] = m
    # Minimal test doubles / stub envs need not implement it — the rule then
    # simply stays inactive (same convention as the connectivity filter).
    per_off = [
        (fn() if (fn := getattr(envs[i], "offlayer_pointer_indices", None))
         is not None else np.zeros((0,), dtype=np.int64))
        for i in indices
    ]
    O_max = max((o.shape[0] for o in per_off), default=0)
    off_masks = np.full((len(indices), O_max), -1, dtype=np.int64)
    for k, o in enumerate(per_off):
        if o.shape[0]:
            off_masks[k, : o.shape[0]] = o
    return masks, ptr_masks, mode_masks, nvm, off_masks


def mask_arrays_to_tensors(
    masks: Any,
    ptr_masks: Any,
    mode_masks: Any,
    nvm: Any,
    device: Any,
    *,
    mode_none_if_all_true: bool,
    off_masks: Any = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Tensorize gathered mask arrays into ``act``/``act_and_value`` kwargs.

    Returns ``(action_mask_t, pointer_mask_t, mode_mask_t|None, kwargs)``. The
    trailing dict carries the non-positional model kwargs — historically only
    the net-valid pair (hence the ``nvm_kwargs`` name at the call sites), now
    also ``offlayer_masks``, the cand columns the make_line row must not point
    at. Anything that RESLICES this dict per sub-batch must slice every array in
    it (see ``rollout/primitive.py``).

    ``mode_none_if_all_true`` picks the mode-mask convention — this is
    semantics, not style (an all-True *tensor* still changes the log-prob
    composition in ``models/v1/net.py``): the training collectors pass
    ``True`` (None when nothing is restricted; mirrored at update time by
    ``algorithms/_common.py``), the eval/one-step paths pass ``False``
    (always a tensor; harmless because they discard log-probs).
    """
    import numpy as np
    import torch

    mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
    ptr_t = torch.as_tensor(ptr_masks, dtype=torch.long, device=device)
    mode_arr = np.asarray(mode_masks) if mode_masks is not None else None
    if mode_arr is None or mode_arr.size == 0:
        mode_t = None
    elif mode_none_if_all_true and mode_arr.all():
        mode_t = None
    else:
        mode_t = torch.as_tensor(mode_arr, dtype=torch.bool, device=device)
    nvm_kwargs: dict[str, Any] = {}
    if nvm is not None:
        nvm_kwargs = {
            "net_valid_mask": torch.as_tensor(
                nvm, dtype=torch.bool, device=device,
            ),
            "allow_net_select_lp": True,
        }
    if off_masks is not None:
        nvm_kwargs["offlayer_masks"] = torch.as_tensor(
            off_masks, dtype=torch.long, device=device,
        )
    return mask_t, ptr_t, mode_t, nvm_kwargs


class KiCadRLAgent:
    """Batched RL inference over a :class:`KiCadRLModel`."""

    def __init__(
        self,
        model: Any,
        device: Any = None,
        *,
        deterministic: bool = False,
    ) -> None:
        import torch

        self.model = model
        self.device = (
            device if device is not None
            else next(model.parameters()).device
        )
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.deterministic = bool(deterministic)
        # Stamped by from_checkpoint(); -1 / {} when constructed directly.
        self.train_iteration: int = -1
        self.ckpt_args: dict = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Any = "auto",
        *,
        deterministic: bool = False,
    ) -> "KiCadRLAgent":
        """Build the agent from a training checkpoint.

        Single load path: delegates to ``methods.rl_agent.models.loader._load_policy``
        (legacy-arg inference, strict-ish state_dict check, ``.eval()``).
        Lazy import — acyclic at runtime; policy stays import-light.
        """
        from methods.rl_agent.models.loader import _load_policy
        from methods.rl_agent.training.utils import auto_device

        dev = auto_device(device) if isinstance(device, str) else device
        model, ckpt_args, iteration = _load_policy(Path(checkpoint_path), dev)
        agent = cls(model, device=dev, deterministic=deterministic)
        agent.ckpt_args = dict(ckpt_args)
        agent.train_iteration = int(iteration)
        return agent

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(
        self,
        obs_list: list[dict],
        *,
        action_masks: Any = None,
        pointer_masks: Any = None,
        offlayer_masks: Any = None,
        mode_mask: Any = None,
        net_valid_mask: Any = None,
        allow_net_select_lp: bool = False,
        deterministic: Optional[bool] = None,
        walked: Optional[dict] = None,
    ):
        """One no-grad decision step -> ``(actions, log_probs)`` tensors."""
        import torch

        with torch.no_grad():
            return self.model.act(
                obs_list,
                action_masks=action_masks,
                offlayer_masks=offlayer_masks,
                deterministic=(
                    self.deterministic if deterministic is None
                    else bool(deterministic)
                ),
                pointer_masks=pointer_masks,
                mode_mask=mode_mask,
                net_valid_mask=net_valid_mask,
                allow_net_select_lp=allow_net_select_lp,
                walked=walked,
            )

    def act_and_value(self, *args: Any, **kwargs: Any):
        """Training-collection step — pure delegation to the model.

        Grad mode is intentionally untouched (exact semantics of the
        pre-facade collectors); callers wanting no-grad wrap it themselves.
        """
        return self.model.act_and_value(*args, **kwargs)

    def act_from_pool(
        self,
        pool: Any,
        obs_list: list[dict],
        live_slots: list[int],
        deterministic: Optional[bool] = None,
    ):
        """Gather masks from a vec backend, act, return ``(B, 3)`` int64 np.

        Thin composition of :func:`gather_mask_arrays` +
        :func:`mask_arrays_to_tensors` (eval convention) + :meth:`act`, so
        this works over anything exposing ``env_method`` (subprocess pool,
        Ray backend, MCTS single-env shim); ``net_valid_mask`` only when the
        model was trained with policy-driven net selection. When ``obs_list``
        carries the wrapper-embedded ``"_masks"`` payload the gather is
        IPC-free (obs-first, query fallback).
        """
        masks, ptr_masks, mode_masks, nvm, off_masks = gather_mask_arrays(
            pool, live_slots,
            policy_net_select=getattr(self.model, "policy_net_select", False),
            obs_list=obs_list,
        )
        mask_t, ptr_t, mode_t, nvm_kwargs = mask_arrays_to_tensors(
            masks, ptr_masks, mode_masks, nvm, self.device, off_masks=off_masks,
            mode_none_if_all_true=False,
        )
        acts, _logp = self.act(
            obs_list,
            action_masks=mask_t,
            pointer_masks=ptr_t,
            mode_mask=mode_t,
            deterministic=deterministic,
            **nvm_kwargs,
        )
        return acts.cpu().numpy()

    # ------------------------------------------------------------------
    # Model passthrough
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only consulted when normal lookup fails — delegates the model's
        # surface (tokenizer / policy_net_select / use_critic / eval / ...)
        # so the agent is a drop-in for a bare model.
        return getattr(self.model, name)
