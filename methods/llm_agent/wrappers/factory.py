"""LLM env factory + vectorised env binding.

Defines the canonical LLM env surface — a Ray-vectorised pool of KiCad routing
workers (:class:`KiCadLLMVecEnv`), each serialising dict observations to text for
an LLM agent — and the ``build_cadagent_envs`` constructor.

Mirrors :mod:`methods.rl_agent.wrappers.factory` (``make_decoder_env_pool`` →
``SubprocDecoderVecEnv``). Unlike the RL subproc pool — which the factory wires
via ``env_fns`` closures, so it needs no binding subclass — the Ray backend
takes an injected ``worker_cls``, so the LLM side needs this thin
:class:`KiCadLLMVecEnv` subclass to bind :class:`KiCadLLMWrapper` and map
``env_kwargs`` → worker ctor kwargs. It imports only *down* (the common Ray
backend) and *sideways within wrappers* (the LLM adapter) — never up into
``training``; training owns the keyword config it passes in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pcb_world.vec.backends.ray import RayVecBackend, _normalize_board_paths
from methods.llm_agent.wrappers.adapter import KiCadLLMWrapper


class KiCadLLMVecEnv(RayVecBackend):
    """Ray-based vectorised KiCad routing environment for the LLM agent.

    Manages ``env_num * group_n`` parallel KiCadLLMWrapper actors. All
    ``group_n`` workers in a GiGPO group share the same board — required
    for group-relative advantage to be meaningful.

    Args:
        board_path:   Path to a single .kicad_pcb. Shorthand for
                      ``board_paths=[board_path] * env_num``. Mutually
                      exclusive with ``board_paths``.
        board_paths:  Per-group board paths; length must equal
                      ``env_num``. Worker ``i`` is bound to
                      ``board_paths[i // group_n]``.
        seed:                  Base random seed (each group_idx gets seed + group_idx).
        env_num:               Number of GiGPO groups (= logical batch size).
        group_n:               Group size for GiGPO rollouts.
        resources_per_worker:  Ray resource dict (e.g. ``{"num_cpus": 1}``).
        is_train:              Unused; kept for API parity.
        env_kwargs:            PCBWorld overrides (``max_steps`` etc.).
    """

    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: Dict[str, Any],
        board_path: Optional[str] = None,
        board_paths: Optional[List[str]] = None,
        is_train: bool = True,
        env_kwargs: Dict[str, Any] = {},
    ) -> None:
        board_paths = _normalize_board_paths(board_path, board_paths, env_num)

        state_format = env_kwargs.get("state_format", "sexpr")
        worker_ctor_kwargs: Dict[str, Any] = {
            "max_steps": env_kwargs.get("max_steps", 200),
            "masking_rule": env_kwargs.get("masking_rule", "strict"),
            "reward_rule": env_kwargs.get("reward_rule", "default"),
            "state_format": state_format,
            "corner_mode": env_kwargs.get("corner_mode", 0),
            "via_penalty": env_kwargs.get("via_penalty", None),
            "reward_noise_std": env_kwargs.get("reward_noise_std", 0.0),
            "emit_drc_tokens": env_kwargs.get("emit_drc_tokens", True),
            "use_yaml_drc_fallback": env_kwargs.get("use_yaml_drc_fallback", False),
        }

        super().__init__(
            worker_cls=KiCadLLMWrapper,
            worker_ctor_kwargs=worker_ctor_kwargs,
            seed=seed,
            env_num=env_num,
            group_n=group_n,
            resources_per_worker=resources_per_worker,
            board_paths=board_paths,
        )
        self.state_format = state_format


def build_cadagent_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: Dict[str, Any],
    board_path: Optional[str] = None,
    board_paths: Optional[List[str]] = None,
    is_train: bool = True,
    env_kwargs: Dict[str, Any] = {},
) -> KiCadLLMVecEnv:
    """Construct a KiCadLLMVecEnv.

    Exactly one of ``board_path`` / ``board_paths`` must be supplied.
    """
    return KiCadLLMVecEnv(
        board_path=board_path,
        board_paths=board_paths,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )


__all__ = ["KiCadLLMVecEnv", "build_cadagent_envs"]
