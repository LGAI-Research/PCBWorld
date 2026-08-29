"""LLM env adapter — the per-env Ray-actor body.

Holds one :class:`PCBWorld` and serialises its dict observation to
prompt-facing text for the LLM agent (the value/text-space analogue of the
RL index-space :class:`methods.rl_agent.wrappers.adapter.KiCadRLWrapper`). Driven as a
Ray actor by :class:`pcb_world.vec.backends.ray.RayVecBackend`, which the
:class:`methods.llm_agent.wrappers.factory.KiCadLLMVecEnv` binding injects.

Note on RLRouter singleton:
    PCBWorld's underlying C++ router is a per-process singleton. Each Ray
    actor runs in its own process, so multiple workers are safe.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from configs.loader.schema import EnvConfig, RewardOverrides
from methods.llm_agent.wrappers.state_converter import (
    format_state, format_state_split,
    format_state_sexpr, format_state_split_sexpr,
)

# Supported state serialization formats
_FORMAT_FUNCS = {
    "xml":   (format_state, format_state_split),
    "sexpr": (format_state_sexpr, format_state_split_sexpr),
}


# ---------------------------------------------------------------------------
# Adaptive max_steps helpers — internal to this module.
# ---------------------------------------------------------------------------

def _count_board_pins(obs_dict: Optional[Dict[str, Any]]) -> int:
    """Total pin count: ``sum(len(board_static.nets[*].pads))``.

    Returns ``0`` for ``None`` / malformed dicts (no observation yet).
    """
    if obs_dict is None:
        return 0
    nets = obs_dict.get("board_static", {}).get("nets", {})
    return sum(len(net.get("pads", {})) for net in nets.values())


def _count_board_nets(obs_dict: Optional[Dict[str, Any]]) -> int:
    """Number of nets in ``board_static.nets``. ``0`` for missing dicts."""
    if obs_dict is None:
        return 0
    return len(obs_dict.get("board_static", {}).get("nets", {}))


# ---------------------------------------------------------------------------
# Observation serialiser
# ---------------------------------------------------------------------------

def _serialize_obs(obs: dict, state_format: str = "sexpr") -> str:
    """Convert a PCBWorld observation dict to a string."""
    fmt, _ = _FORMAT_FUNCS[state_format]
    return fmt(obs)


def _serialize_obs_split(obs: dict, state_format: str = "sexpr") -> tuple:
    """Convert a PCBWorld observation dict to (board_static, dynamic) strings."""
    _, fmt_split = _FORMAT_FUNCS[state_format]
    return fmt_split(obs)


# ---------------------------------------------------------------------------
# Ray remote worker
# ---------------------------------------------------------------------------

class KiCadLLMWrapper:
    """Per-env LLM adapter — a Ray remote actor holding one PCBWorld.

    The value/text-space counterpart of the RL index-space
    :class:`methods.rl_agent.wrappers.adapter.KiCadRLWrapper`: each actor owns a single
    environment and exposes step / reset / get_action_mask as remote-callable
    methods (mirrors verl-agent's AlfworldWorker).
    """

    def __init__(
        self,
        board_path: str,
        max_steps: int,
        masking_rule: str,
        reward_rule: str,
        seed: int,
        state_format: str = "sexpr",
        corner_mode: int = 0,
        via_penalty: Optional[float] = None,
        reward_noise_std: float = 0.0,
        emit_drc_tokens: bool = True,
        use_yaml_drc_fallback: bool = False,
    ) -> None:
        self.state_format = state_format
        # Store seed so every reset() re-seeds the env — required for GiGPO
        # group-shared start states (workers sharing the same group_idx are
        # constructed with the same seed in KiCadLLMVecEnv.__init__).
        self._seed = seed
        # Canonical env-core config (shared schema with the RL factory) so
        # reload_board() can rebuild the underlying PCBWorld in place for a
        # different board. Fields the LLM path does not set (wirelength/drc
        # penalties, etc.) stay at the EnvConfig defaults = PCBWorld defaults.
        self._env_config = EnvConfig(
            max_steps=max_steps,
            masking_rule=masking_rule,
            reward_rule=reward_rule,
            corner_mode=corner_mode,
            reward_noise_std=reward_noise_std,
            emit_drc_tokens=emit_drc_tokens,
            use_yaml_drc_fallback=use_yaml_drc_fallback,
            reward=RewardOverrides(via_penalty=via_penalty),
        )
        self._board_path = board_path
        self.env = self._build_env(board_path)

    def _build_env(self, board_path: str):
        # Imported inside the actor process to avoid serialization issues
        # with the C++ KiCad router singleton.
        from pcb_world.core.env import PCBWorld

        return PCBWorld(board_path=board_path, env_config=self._env_config)

    def step(self, action: Dict[str, Any]):
        """Execute one action step.

        Args:
            action: PCBWorld-compatible action dict from cadagent_projection().

        Returns:
            (text_obs, reward, done, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Retain the raw (pre-serialisation) obs dict so callers that want
        # the native schema (eval dumps, visualiser) can opt-in via
        # ``get_last_obs_dict`` without re-running the env.
        self._last_obs_dict = obs

        text_obs = _serialize_obs(obs, self.state_format)
        # REVIEW(metric-semantics): bare ``terminated`` is NOT "fully routed"
        # since the voluntary all-nets-closed finish (v0.11.28) — give-up
        # episodes also terminate. Canonical success = env ``info["success"]``
        # / ``eval.eval_utils.success_from``; ``won`` kept as-is (LLM-owned).
        info["won"] = bool(terminated)
        # Preserve the raw termination/truncation split for value-bootstrap
        # schemes (e.g. PPO) — the collapsed `done` is lossy.
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info["action_mask"] = self.env.action_masks().tolist()
        info["action_mask_dict"] = self.env.action_mask_dict()
        # drc_violations is populated at episode end by PCBWorld.step()
        info.setdefault("drc_violations", 0)

        return text_obs, float(reward), bool(done), info

    def reset(self):
        """Reset the environment.

        Returns:
            (text_obs, info)
        """
        # Re-seed every reset so GiGPO group members (same ``_seed``) see
        # identical start state + np_random on every rollout iteration.
        obs, info = self.env.reset(seed=self._seed)
        self._last_obs_dict = obs

        text_obs = _serialize_obs(obs, self.state_format)
        info["action_mask"] = self.env.action_masks().tolist()
        info["action_mask_dict"] = self.env.action_mask_dict()

        return text_obs, info

    def get_action_mask(self):
        """Return the current boolean action mask as a list."""
        return self.env.action_masks().tolist()

    def get_action_mask_dict(self):
        """Return the current action mask as {action_name: bool}."""
        return self.env.action_mask_dict()

    def get_last_obs_dict(self) -> Optional[Dict[str, Any]]:
        """Return the raw observation dict from the most recent step/reset.

        ``None`` until the first ``reset()``.
        """
        return getattr(self, "_last_obs_dict", None)

    def apply_adaptive_max_steps(self, base_max_steps: int) -> int:
        """Compute and apply ``max(base, pin*3 + net*2)`` for this worker.

        Reads pin / net counts from the worker's most recent observation
        (cached as ``_last_obs_dict`` by ``reset`` / ``step``) and writes
        the result onto ``PCBWorld.max_steps`` so the env's internal
        ``self._step_count >= self.max_steps`` truncation respects the
        adaptive budget. Returns the effective value.
        """
        obs = getattr(self, "_last_obs_dict", None)
        effective = max(
            base_max_steps,
            _count_board_pins(obs) * 3 + _count_board_nets(obs) * 2,
        )
        self._env_config = replace(self._env_config, max_steps=effective)
        self.env.max_steps = effective
        return effective

    def reload_board(self, board_path: str, seed: Optional[int] = None) -> None:
        """Swap the underlying board without tearing down the Ray actor.

        Rebuilds the PCBWorld in place so the C++ RLRouter singleton is
        reinitialised against the new .kicad_pcb. The surrounding Ray actor
        (and its Python state) stays alive, avoiding actor startup cost
        when cycling through boards each training iter.
        """
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
        # Drop the reference and force GC before rebuilding so the old
        # RLRouter finalises before the new one claims the singleton slot.
        self.env = None
        self._last_obs_dict = None
        import gc
        gc.collect()

        if seed is not None:
            self._seed = seed
        self._board_path = board_path
        self.env = self._build_env(board_path)

    @property
    def board_path(self) -> str:
        return self._board_path

    @property
    def board_id(self) -> str:
        """Filename stem of the current board (matches boards_json entries)."""
        import os
        return os.path.splitext(os.path.basename(self._board_path))[0]

    def save_board(self, output_path: str) -> str:
        """Save the worker's current board state to ``output_path``.

        ``KiCadEngine.save`` always writes both ``<stem>.kicad_pcb`` and
        ``<stem>.kicad_pro``; returning the requested path lets the caller
        confirm it landed where expected (useful when logs are shared
        across workers).
        """
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.env._engine.save(output_path)
        return output_path

    def eval_inline_drc(self, **kwargs) -> dict:
        """Run canonical DRC scoring on the live routed engine.

        Same name/hook as the RL worker
        (:meth:`methods.rl_agent.wrappers.adapter.KiCadRLWrapper.eval_inline_drc`) so
        both branches expose one scoring contract over ``env_method`` — a thin
        delegation to :func:`eval.metrics.compute_metrics_inline`.
        Non-destructive: ``u_0`` comes from the env's reset-time capture, so
        the just-routed board is left untouched.
        """
        from eval.metrics import compute_metrics_inline
        return compute_metrics_inline(self.env, **kwargs)
