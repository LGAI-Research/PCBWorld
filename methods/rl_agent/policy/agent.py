"""KiCadRLAgent — the RL inference facade (Agent layer).

Taxonomy (see :mod:`methods._shared.agent`): :class:`~methods.rl_agent.models.v1.net.
KiCadRLModel` is the Model (weights/forward/tokenizer); this class is the
Agent — it performs model-based inference, one decision step from raw env
obs dicts (+ masks) to action tensors. Episode orchestration stays with the
callers (training collectors, eval rollout, serving provider), which all
converge on this facade instead of poking the nn.Module directly.

Three call surfaces, each taking the batch's :class:`~methods.rl_agent.models.
v1.spec.DistInputs` (``dist=``, required):

  * :meth:`act` — eval-path step, ``torch.no_grad``.
  * :meth:`act_and_value` — training-collection step: pure delegation
    (no grad-mode change), returns ``(actions, log_probs, values)``.
  * :meth:`act_from_pool` — eval convenience: gathers the bundle from a vec
    backend, then acts.

The gather itself is the module-level :func:`gather_dist` — the single
implementation shared by the rollout primitive
(``methods/rl_agent/rollout/primitive.py``), the collectors' bootstrap
forwards, :meth:`act_from_pool`, BC batching and MCTS.

Unknown attributes delegate to the wrapped model (``__getattr__``), so the
agent is a drop-in wherever a bare ``KiCadRLModel`` is expected
(``tokenizer`` / ``policy_net_select`` / ``use_critic`` / ``eval()`` ...).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from methods.rl_agent.models.v1.spec import DistInputs

__all__ = ["KiCadRLAgent", "gather_dist"]


def gather_dist(
    envs: Any,
    indices: list[int],
    *,
    policy_net_select: bool,
    obs_list: "list[dict] | None" = None,
) -> "DistInputs":
    """Gather the act-time :class:`DistInputs` (numpy) for ``indices``.

    One payload per env — the six-key dict ``KiCadRLWrapper.mask_payload``
    returns — stacked by :meth:`DistInputs.from_payloads` (``net_valid`` only
    when ``policy_net_select``). Payload source, in order: the
    wrapper-embedded ``obs["_masks"]`` when every obs in ``obs_list`` carries
    it (zero IPC; the wrapper attaches it on every step/reset), else
    ``envs.env_method("mask_payload")`` (``VecBackend`` pool, Ray backend, the
    MCTS ``_SingleEnvPool`` shim), else a plain ``list[KiCadRLWrapper]``. A
    payload missing a key raises ``KeyError`` — never a silently empty mask.
    """
    from methods.rl_agent.models.v1.spec import DistInputs

    if obs_list and all("_masks" in o for o in obs_list):
        payloads = [o["_masks"] for o in obs_list]
    elif hasattr(envs, "env_method"):
        payloads = envs.env_method("mask_payload", indices=indices)
    else:
        payloads = [envs[i].mask_payload() for i in indices]
    return DistInputs.from_payloads(payloads, policy_net_select=policy_net_select)


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
        self.ckpt_args: Mapping = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Any = "auto",
        *,
        strict: bool,
        deterministic: bool = False,
    ) -> "KiCadRLAgent":
        """Build the agent from a training checkpoint.

        Single load path: delegates to ``methods.rl_agent.models.loader._load_policy``
        (legacy-arg inference, strict-ish state_dict check, ``.eval()``).
        Lazy import — acyclic at runtime; policy stays import-light.
        ``strict`` = ``--strict-ckpt`` (``configs.loader.ckpt``); ``ckpt_args``
        is the loader's resolved, read-only config.
        """
        from methods.rl_agent.models.loader import _load_policy
        from methods.rl_agent.training.utils import auto_device

        dev = auto_device(device) if isinstance(device, str) else device
        model, ckpt_args, iteration = _load_policy(Path(checkpoint_path), dev, strict=strict)
        agent = cls(model, device=dev, deterministic=deterministic)
        agent.ckpt_args = ckpt_args
        agent.train_iteration = int(iteration)
        return agent

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(
        self,
        obs_list: list[dict],
        *,
        dist: "DistInputs",
        deterministic: Optional[bool] = None,
        walked: Optional[dict] = None,
    ):
        """One no-grad decision step -> ``(actions, log_probs)`` tensors."""
        import torch

        with torch.no_grad():
            return self.model.act(
                obs_list,
                dist=dist,
                deterministic=(
                    self.deterministic if deterministic is None
                    else bool(deterministic)
                ),
                walked=walked,
            )

    def act_and_value(
        self,
        obs_list: list[dict],
        *,
        dist: "DistInputs",
        deterministic: bool = False,
        walked: Optional[dict] = None,
    ):
        """Training-collection step — pure delegation to the model.

        Grad mode is intentionally untouched (exact semantics of the
        pre-facade collectors); callers wanting no-grad wrap it themselves.
        """
        return self.model.act_and_value(
            obs_list, dist=dist, deterministic=deterministic, walked=walked,
        )

    def act_from_pool(
        self,
        pool: Any,
        obs_list: list[dict],
        live_slots: list[int],
        deterministic: Optional[bool] = None,
    ):
        """Gather the bundle from a vec backend, act, return ``(B, 3)`` int64 np.

        Thin composition of :func:`gather_dist` + :meth:`act`, so this
        works over anything exposing ``env_method`` (subprocess pool, Ray
        backend, MCTS single-env shim). When ``obs_list`` carries the
        wrapper-embedded ``"_masks"`` payload the gather is IPC-free.
        """
        dist = gather_dist(
            pool, live_slots,
            policy_net_select=self.model.policy_net_select,
            obs_list=obs_list,
        )
        acts, _logp = self.act(
            obs_list, dist=dist.to(self.device), deterministic=deterministic,
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
