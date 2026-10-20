"""LLM env factory + vectorised env binding.

Defines the canonical LLM env surface — a Ray-vectorised pool of KiCad routing
workers (:class:`KiCadLLMVecEnv`), each serialising dict observations to text for
an LLM agent — and the ``build_pcbworld_envs`` constructor.

Mirrors :mod:`methods.rl_agent.wrappers.factory` (``make_decoder_env_pool`` →
``SubprocDecoderVecEnv``). Unlike the RL subproc pool — which the factory wires
via ``env_fns`` closures, so it needs no binding subclass — the Ray backend
takes an injected ``worker_cls``, so the LLM side needs this thin
:class:`KiCadLLMVecEnv` subclass to bind :class:`KiCadLLMWrapper` and hand
every worker the same :class:`EnvConfig` (whole — no per-key list). It imports only *down* (the common Ray
backend) and *sideways within wrappers* (the LLM adapter) — never up into
``training``; training owns the keyword config it passes in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from configs.loader.schema import EnvConfig
from pcb_world.vec.backends.ray import RayVecBackend, _normalize_board_paths
from methods.llm_agent.wrappers.adapter import KiCadLLMWrapper
from methods.llm_agent.wrappers.prompts import check_prompt_masking


class KiCadLLMVecEnv(RayVecBackend):
    """Ray-based vectorised KiCad routing environment for the LLM agent.

    Manages ``env_num * group_n`` parallel KiCadLLMWrapper actors. All
    ``group_n`` workers in a GiGPO group share the same board — required
    for group-relative advantage to be meaningful.

    Args:
        env_config:   The whole env config every worker's ``PCBWorld`` is built with.
        state_format: Observation text format (``sexpr`` | ``xml``).
        prompt_version: Prompt version the text is rendered for (selects the obs
                      elements); must grant the same actions as
                      ``env_config.masking_rule`` (raises otherwise).
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
    """

    def __init__(
        self,
        env_config: EnvConfig,
        state_format: str,
        prompt_version: str,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: Dict[str, Any],
        board_path: Optional[str] = None,
        board_paths: Optional[List[str]] = None,
        is_train: bool = True,
    ) -> None:
        check_prompt_masking(prompt_version, env_config.masking_rule)
        board_paths = _normalize_board_paths(board_path, board_paths, env_num)
        super().__init__(
            worker_cls=KiCadLLMWrapper,
            worker_ctor_kwargs={"env_config": env_config, "state_format": state_format,
                                "prompt_version": prompt_version},
            seed=seed,
            env_num=env_num,
            group_n=group_n,
            resources_per_worker=resources_per_worker,
            board_paths=board_paths,
        )
        self.env_config = env_config
        self.state_format = state_format
        self.prompt_version = prompt_version


def build_pcbworld_envs(
    env_config: EnvConfig,
    state_format: str,
    prompt_version: str,
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: Dict[str, Any],
    board_path: Optional[str] = None,
    board_paths: Optional[List[str]] = None,
    is_train: bool = True,
) -> KiCadLLMVecEnv:
    """Construct a KiCadLLMVecEnv.

    Exactly one of ``board_path`` / ``board_paths`` must be supplied.
    """
    return KiCadLLMVecEnv(
        env_config=env_config,
        state_format=state_format,
        prompt_version=prompt_version,
        board_path=board_path,
        board_paths=board_paths,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
    )


__all__ = ["KiCadLLMVecEnv", "build_pcbworld_envs"]
