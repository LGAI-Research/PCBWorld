"""Trainer-agnostic rollout-manager base for the cadagent KiCad routing env.

Owns all cadagent-specific state and per-step logic that is shared across
training algorithm adapters (verl-agent today, future RL trainers tomorrow):

  - :class:`SimpleMemory` for valid-step action history
  - per-env total step counter (``_total_steps``) covering invalid attempts
  - v5 rejection-streak tracking (``_last_rejected``)
  - episode-static board context cache (``_board_static_cache``)
  - versioned prompt bundle dispatch (``get_cadagent_prompts``)
  - optional multi-board scheduler (``_board_scheduler``)
  - per-board adaptive max-steps budget (``effective_max_steps``)
  - episode-end canonical DRC scoring (``score_episodes`` ->
    worker ``eval_inline_drc`` -> buffered per-rollout rows,
    ``pop_episode_rows``) — the LLM counterpart of the RL eval rollout's
    inline scoring, sharing the same worker hook + row schema

Concrete training-framework adapters subclass this together with their own
framework's manager base (e.g. verl-agent's
``agent_system.environments.base.EnvironmentManagerBase``). Hooks let the
adapter override only what differs:

  - :meth:`_pack_observations` — framework-specific obs-dict shape
  - :meth:`_post_step_info`     — per-info bookkeeping (e.g. is_action_valid)
  - :meth:`_resolve_*`          — config schema differences

The base imports nothing from ``agent_system``, ``torch``, or ``numpy`` so it
works in publish-side environments without verl-agent installed.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from methods.llm_agent.wrappers.feedback import (
    NO_EFFECT_PREFIX,
    extract_action_body,
    is_no_progress,
    mark_preceding_start_route_no_effect,
    render_rejection_streak,
    update_rejection_streak,
)
from methods.llm_agent.wrappers.memory import SimpleMemory
from methods.llm_agent.wrappers.action_converter import extract_think_action
from methods.llm_agent.wrappers.state_converter import (
    format_valid_actions,
    split_board_static,
)
from methods.llm_agent.wrappers.prompts import get_cadagent_prompts


@runtime_checkable
class LLMRolloutBackend(Protocol):
    """The vec-backend surface :class:`KiCadLLMRolloutManager` drives.

    Formalises the LLM rollout backend contract so the manager depends on this
    Protocol rather than a concrete class — :class:`methods.llm_agent.wrappers.factory.
    KiCadLLMVecEnv` (a :class:`~pcb_world.vec.backends.ray.RayVecBackend` in
    worker_cls mode) structurally satisfies it today; a future non-Ray LLM
    rollout backend can be substituted without touching the manager.

    Action-restriction note — this is the LLM branch, NOT RL. The LLM never
    logit-masks. ``get_action_masks`` / ``get_action_mask_dicts`` are the
    per-stage *valid action-type list* (``List[bool]`` aligned with
    ACTION_REGISTRY / ``{name: bool}``), cached after each step/reset (and
    surviving the done short-circuit) and read at prompt-assembly time to
    render the valid-actions text and to validate the parsed action. They are
    deliberately NOT the RL pointer/mode/net_valid logit-mask machinery
    (``methods.rl_agent.models.v1.encoding.stack_*_masks``); the two branches restrict
    actions at different layers (RL at the logits, LLM at prompt + grammar +
    parse), so this surface stays LLM-specific by design.

    ``step``/``reset`` use the LLM-branch tuple shape (``image_obs`` is a
    placeholder, always ``None``); ``done`` is the collapsed terminated|
    truncated flag (the backend may short-circuit already-done workers).
    """

    env_num: int

    def reset(self) -> Tuple[List[str], Any, List[Dict]]: ...

    def step(
        self, actions: List[Any],
    ) -> Tuple[List[str], Any, List[Any], List[Any], List[Dict]]: ...

    def reload_boards(self, board_paths: List[str]) -> None: ...

    def apply_adaptive_max_steps(self, base_max_steps: int) -> int: ...

    def env_method(
        self, name: str, *args: Any, indices: Any = None, **kwargs: Any,
    ) -> List[Any]: ...

    @property
    def get_action_masks(self) -> List[Any]: ...

    @property
    def get_action_mask_dicts(self) -> List[Any]: ...

    @property
    def board_paths(self) -> List[str]: ...


class KiCadLLMRolloutManager:
    """KiCad PCB routing rollout adapter — trainer-agnostic mother class.

    Designed for multiple inheritance with a training framework's own
    manager base. The verl-agent adapter for instance does::

        class KiCadLLMVerlManager(KiCadLLMRolloutManager, EnvironmentManagerBase):
            def __init__(self, envs, projection_f, config, board_scheduler=None):
                KiCadLLMRolloutManager.__init__(
                    self, envs, projection_f, config, board_scheduler,
                )
            ...

    Subclasses inherit the full per-step rollout logic and only override the
    hooks that differ from the verl-agent default (see ``_pack_observations``
    / ``_post_step_info``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self, envs: "LLMRolloutBackend", projection_f, config, board_scheduler=None,
        score_episodes: bool = False,
    ):
        self.envs = envs
        self.projection_f = projection_f
        self.config = config

        self.memory = SimpleMemory()
        self._board_static_cache: List[str] = []
        self._total_steps: List[int] = []
        # Entry shape: {"body": str, "first_step": int, "last_step": int, "count": int}
        self._last_rejected: List[Optional[dict]] = []

        # Episode-end canonical scoring (validation / eval managers only —
        # off for training rollouts to avoid the per-episode DRC cost).
        self.score_episodes = bool(score_episodes)
        self._eval_reward_config: str = self._resolve_eval_reward_config(config)
        self._eval_check_angle: int = self._resolve_eval_check_angle(config)
        self._episode_return: List[float] = []
        self._episode_done: List[bool] = []
        self._episode_rows: List[dict] = []
        # Nested per-env compute_metrics dicts for the current episode (None
        # until that env is scored) — callers that need the un-flattened
        # result (e.g. the eval CLI's {stem}.json dumps) read this.
        self.episode_scored: List[Optional[dict]] = []

        self._state_format: str = getattr(envs, "state_format", "sexpr")

        prompt_version = self._resolve_prompt_version(config)
        self._prompts = get_cadagent_prompts(prompt_version)

        # Workers were just built with the scheduler's iter-0 board, so the
        # very first reset() must NOT advance the scheduler. Subsequent
        # resets do.
        self._board_scheduler = board_scheduler
        self._skip_scheduler_advance = board_scheduler is not None

        self._base_max_steps: int = self._resolve_base_max_steps(config)
        # Reads before the first reset (e.g. early in the trainer's rollout
        # loop) get the configured baseline; reset() updates it.
        self.effective_max_steps: int = self._base_max_steps
        self._history_length: int = self._resolve_history_length(config)

        # Last step's projection results — surfaced so callers can log
        # parse-fail counts, dump action_dicts, or tag valids onto info
        # without re-running ``projection_f``. Empty until the first step.
        self._last_actions: List[Any] = []
        self._last_valids: List[Any] = []

    # ------------------------------------------------------------------
    # Config-schema hooks (subclass overrides for non-OmegaConf configs)
    # ------------------------------------------------------------------

    @staticmethod
    def _cfg_get(node, key: str, default):
        """Read ``key`` from ``node`` whether it's an OmegaConf DictConfig
        (``.get``) or a plain object (``getattr``)."""
        if node is None:
            return default
        if hasattr(node, "get"):
            try:
                return node.get(key, default)
            except TypeError:
                pass
        return getattr(node, key, default)

    def _resolve_prompt_version(self, config) -> str:
        """Default schema: ``config.env.cadagent.prompt_version`` (verl-agent).

        Default sourced from the shared schema (configs/defaults/llm_train.yaml)
        so the LLM-train defaults live in one place.
        """
        from configs.loader.schema import LLMTrainConfig

        env_cfg = self._cfg_get(config, "env", None)
        cad_cfg = self._cfg_get(env_cfg, "cadagent", None)
        return self._cfg_get(cad_cfg, "prompt_version", LLMTrainConfig().prompt_version)

    def _resolve_base_max_steps(self, config) -> int:
        from configs.loader.schema import LLMTrainConfig

        env_cfg = self._cfg_get(config, "env", None)
        return int(self._cfg_get(env_cfg, "max_steps", LLMTrainConfig().max_steps))

    def _resolve_history_length(self, config) -> int:
        from configs.loader.schema import LLMTrainConfig

        env_cfg = self._cfg_get(config, "env", None)
        return int(self._cfg_get(env_cfg, "history_length", LLMTrainConfig().history_length))

    def _resolve_eval_reward_config(self, config) -> str:
        """Reward config used for episode-end scoring. Default sourced from
        ``LLMTrainConfig.eval_reward_config`` (which is the shared eval default,
        ``RLEvalConfig.reward_config``), so LLM val and the RL inline-eval path
        score with the SAME ruler (W5 unification)."""
        from configs.loader.schema import LLMTrainConfig

        env_cfg = self._cfg_get(config, "env", None)
        cad_cfg = self._cfg_get(env_cfg, "cadagent", None)
        return str(self._cfg_get(
            cad_cfg, "eval_reward_config", LLMTrainConfig().eval_reward_config,
        ))

    def _resolve_eval_check_angle(self, config) -> int:
        from configs.loader.schema import LLMTrainConfig

        env_cfg = self._cfg_get(config, "env", None)
        cad_cfg = self._cfg_get(env_cfg, "cadagent", None)
        return int(self._cfg_get(
            cad_cfg, "eval_check_angle", LLMTrainConfig().eval_check_angle,
        ))

    # ------------------------------------------------------------------
    # Output-shape hooks (subclass overrides for non-verl-agent contracts)
    # ------------------------------------------------------------------

    def _pack_observations(
        self, *,
        user_prompts: List[str],
        system_prompts: List[str],
        image_obs: Any,
        anchor: List[str],
    ) -> Dict[str, Any]:
        """Pack assembled prompts into the framework's obs-dict shape.

        Default matches verl-agent's contract
        (``{'text', 'system', 'image', 'anchor'}``). Override to swap.
        """
        return {
            "text": user_prompts,
            "system": system_prompts,
            "image": image_obs,
            "anchor": anchor,
        }

    def _post_step_info(self, infos: List[Dict], valids: List) -> List[Dict]:
        """Stamp per-info bookkeeping right before ``step`` returns.

        Default no-op. The verl-agent shim overrides to write
        ``info['is_action_valid'] = to_numpy(valids[i])``.
        """
        return infos

    # ------------------------------------------------------------------
    # Core rollout interface
    # ------------------------------------------------------------------

    def reset_state(self) -> Tuple[List[str], Any, List[Dict]]:
        """Low-level reset: advance scheduler if needed, reset envs, init
        per-env state.

        Returns the raw ``(text_obs, image_obs, infos)`` tuple from
        ``envs.reset()`` after per-env state has been re-initialised. Used
        by :meth:`reset` and exposed for subclasses that need to assemble a
        non-default obs dict shape.
        """
        if (
            self._board_scheduler is not None
            and self._board_scheduler.mode != "single"
        ):
            if self._skip_scheduler_advance:
                self._skip_scheduler_advance = False
            else:
                next_paths = self._board_scheduler.next_paths(self.envs.env_num)
                self.envs.reload_boards(next_paths)

        text_obs, image_obs, infos = self.envs.reset()

        # Adaptive rollout budget — pin-heavy boards need more steps than
        # the static config baseline. Trainer reads ``self.effective_max_steps``
        # as the rollout for-loop bound; per-worker budgets are also pushed
        # into PCBWorld so its internal truncation respects each board's
        # own limit.
        self.effective_max_steps = self.envs.apply_adaptive_max_steps(
            self._base_max_steps
        )

        self.memory.reset(batch_size=len(text_obs))
        self._total_steps = [0] * len(text_obs)
        self._last_rejected = [None] * len(text_obs)
        self._episode_return = [0.0] * len(text_obs)
        self._episode_done = [False] * len(text_obs)
        self._episode_rows = []
        self.episode_scored = [None] * len(text_obs)
        self._board_static_cache = [
            split_board_static(obs, self._state_format)[0] for obs in text_obs
        ]
        return text_obs, image_obs, infos

    def reset(self, kwargs: Optional[Dict] = None) -> Tuple[Dict[str, Any], List[Dict]]:
        """Reset envs and return the framework-shaped obs dict + infos."""
        text_obs, image_obs, infos = self.reset_state()
        system_prompts, user_prompts = self.build_text_obs(
            text_obs,
            action_masks=self.envs.get_action_masks,
            init=True,
        )
        obs_dict = self._pack_observations(
            user_prompts=user_prompts,
            system_prompts=system_prompts,
            image_obs=image_obs,
            anchor=list(text_obs),
        )
        return obs_dict, infos

    def step(self, text_actions: List[str]) -> Tuple[Dict[str, Any], Any, Any, List[Dict]]:
        """Project text actions, step envs, update history, return packed obs."""
        actions, valids = self.projection_f(
            text_actions,
            self.envs.get_action_mask_dicts,
        )
        # Surface for caller-side logging / dump_dir; cheap (just a ref).
        self._last_actions = list(actions)
        self._last_valids = list(valids)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self._update_history(text_actions, valids, infos)
        self._track_episode_end(rewards, dones, infos)
        system_prompts, user_prompts = self.build_text_obs(
            text_obs, action_masks=self.envs.get_action_masks,
        )
        infos = self._post_step_info(infos, valids)
        obs_dict = self._pack_observations(
            user_prompts=user_prompts,
            system_prompts=system_prompts,
            image_obs=image_obs,
            anchor=list(text_obs),
        )
        return obs_dict, rewards, dones, infos

    # ------------------------------------------------------------------
    # History bookkeeping (memory + rejection streak + no-effect marker)
    # ------------------------------------------------------------------

    def _update_history(
        self, text_actions: List[str], valids: List, infos: List[Dict],
    ) -> None:
        """Append valid+success steps to memory; update rejection streak otherwise.

        - Valid + ``info['action_success']``: append ``extract_think_action(text)``
          to memory. If ``is_no_progress(info)`` (env signalled empty_action),
          prefix with ``NO_EFFECT_PREFIX`` AND retroactively tag the
          immediately-preceding ``start_route`` via
          ``mark_preceding_start_route_no_effect`` (the env state machine
          allows at most one ``start_route`` between two successful actions,
          so a single look-back is sufficient). Any successful step clears
          the rejection streak.
        - Otherwise (parse fail or mask reject): update the rejection streak
          via ``update_rejection_streak`` so the v5 prompt slot can show the
          last rejected attempt with its ×N count.
        """
        if self.memory.keys is None:
            self.memory.keys = ["action", "step"]
        for i, text in enumerate(text_actions):
            self._total_steps[i] += 1
            step_no = self._total_steps[i]
            if valids[i] and infos[i].get("action_success", False):
                no_progress = is_no_progress(infos[i])
                entry = extract_think_action(text)
                if no_progress:
                    entry = NO_EFFECT_PREFIX + entry
                self.memory._data[i].append({"action": entry, "step": step_no})
                if no_progress:
                    mark_preceding_start_route_no_effect(self.memory._data[i])
                self._last_rejected[i] = None
            else:
                body = (
                    "[Parsing error]" if not valids[i]
                    else extract_action_body(text)
                )
                self._last_rejected[i] = update_rejection_streak(
                    self._last_rejected[i], body, step_no,
                )

    # ------------------------------------------------------------------
    # MCTS checkpoint (L2): LLM-layer per-env memory snapshot / restore
    # ------------------------------------------------------------------

    def snapshot_memory(self, env_idx: int) -> Dict[str, Any]:
        """Capture the LLM-layer path-dependent state for one env (MCTS L2).

        Parallel to the branch-agnostic ``PCBWorld.checkpoint()`` (L1 core):
        L1 captures board+config+session; this captures the manager-owned
        per-env state the LLM prompt depends on — the raw-action history
        (``memory``), the step counter that numbers it, and the rejection
        streak slot. Take this in lockstep with the env checkpoint for the same
        MCTS node, and restore both together so the prompt context matches the
        restored board.

        Deep-copies the history records: ``mark_preceding_start_route_no_effect``
        rewrites earlier entries in place, so a shallow list copy would alias a
        later mutation back into the snapshot.
        """
        return {
            "memory": copy.deepcopy(self.memory._data[env_idx]),
            "total_steps": self._total_steps[env_idx],
            "last_rejected": copy.deepcopy(self._last_rejected[env_idx]),
        }

    def restore_memory(self, env_idx: int, snap: Dict[str, Any]) -> None:
        """Restore the LLM-layer state captured by :meth:`snapshot_memory`.

        Lockstep counterpart to ``PCBWorld.restore()`` — the MCTS driver must
        call both for the same node so the prompt history matches the board.
        Deep-copies on the way in too, so the same snapshot can be restored on
        repeated MCTS visits without aliasing the live memory.
        """
        self.memory._data[env_idx] = copy.deepcopy(snap["memory"])
        self._total_steps[env_idx] = snap["total_steps"]
        self._last_rejected[env_idx] = copy.deepcopy(snap["last_rejected"])

    # ------------------------------------------------------------------
    # Episode-end canonical scoring
    # ------------------------------------------------------------------

    def _track_episode_end(self, rewards, dones, infos: List[Dict]) -> None:
        """Accumulate per-env returns; finish each env once it first reports
        done (scoring it when ``score_episodes`` is on)."""
        if not self._episode_done:
            return  # reset_state() not called yet — nothing to track
        for i, r in enumerate(rewards):
            if not self._episode_done[i]:
                self._episode_return[i] += float(r)
        self.mark_episode_done(
            [i for i, d in enumerate(dones) if bool(d)], infos=infos,
        )

    def mark_episode_done(
        self, indices: List[int], infos: Optional[List[Dict]] = None,
    ) -> None:
        """Mark envs as episode-finished; score each exactly once.

        Called internally on env-reported done, and externally by callers
        that end an episode outside the env (e.g. the eval CLI's
        no-progress early stop — scoring at the stop decision captures the
        board state the caller is freezing). Already-finished envs are
        skipped, so double calls are safe.
        """
        todo = [
            i for i in indices
            if i < len(self._episode_done) and not self._episode_done[i]
        ]
        for i in todo:
            self._episode_done[i] = True
        if self.score_episodes and todo:
            self._score_and_buffer(todo, infos)

    def finalize_episode(self, infos: Optional[List[Dict]] = None) -> None:
        """Finish (and score) every env that has not reported done yet.

        Defensive end-of-rollout hook: when the caller's step loop exits
        before some env reached terminated/truncated, this scores the
        engines as they stand so no episode leaves the rollout unscored.
        """
        self.mark_episode_done(
            list(range(len(self._episode_done))), infos=infos,
        )

    def _score_and_buffer(
        self, indices: List[int], infos: Optional[List[Dict]] = None,
    ) -> None:
        """Run worker-side canonical scoring for ``indices`` and buffer rows.

        Calls ``eval_inline_drc`` (-> ``eval.metrics.compute_metrics``, the
        same hook the RL eval rollout uses) on the still-live engines,
        keeps the nested dict in :attr:`episode_scored`, stamps it onto
        ``infos[i]['eval_metrics']`` when infos are given, and buffers a
        canonical per-rollout row. Identity completion (``board_index`` /
        ``rollout_idx`` / ``seed``) is the caller's job when draining
        :meth:`pop_episode_rows` — the manager only knows the worker-local
        board (``env_idx`` is attached as an extra key for that mapping).
        Scoring failures never propagate: they produce error rows instead.
        """
        from eval.eval_utils import assemble_rollout_row

        scored_list: List[Optional[dict]]
        score_error = ""
        try:
            scored_list = self.envs.env_method(
                "eval_inline_drc",
                reward_config_name=self._eval_reward_config,
                check_angle=self._eval_check_angle,
                indices=indices,
            )
        except Exception as exc:  # noqa: BLE001 — scoring must not kill rollout
            import traceback
            score_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            scored_list = [None] * len(indices)

        board_paths = list(getattr(self.envs, "board_paths", None) or [])
        for i, scored in zip(indices, scored_list):
            self.episode_scored[i] = scored
            runtime: Dict[str, Any] = {
                "status": "completed",
                "steps": int(self._total_steps[i]),
                "episode_return": float(self._episode_return[i]),
            }
            if infos is not None and i < len(infos):
                if scored is not None:
                    infos[i]["eval_metrics"] = scored
                runtime.update({
                    "terminated": bool(infos[i].get("terminated", False)),
                    "truncated": bool(infos[i].get("truncated", False)),
                    # REVIEW(metric-semantics): ``won`` = the adapter's bare
                    # ``bool(terminated)`` — conflates give-up termination with
                    # success (same flag as rollout/cadagent.py; LLM-owned,
                    # left as-is). The scored layer overwrites success
                    # whenever inline scoring ran.
                    "success": bool(infos[i].get("won", False)),
                })
            path = str(board_paths[i]) if i < len(board_paths) else ""
            stem = os.path.splitext(os.path.basename(path))[0] if path else ""
            row = assemble_rollout_row(
                scored,
                error=score_error,
                identity={"board_id": stem, "board_path": path},
                runtime=runtime,
                reward_config=self._eval_reward_config,
                check_angle=self._eval_check_angle,
            )
            row["env_idx"] = i
            self._episode_rows.append(row)

    def pop_episode_rows(self) -> List[dict]:
        """Drain the canonical per-rollout rows buffered since the last reset
        (or last drain). Rows carry ``env_idx``; the caller assigns
        ``board_index``/``rollout_idx``/``seed`` before aggregation."""
        rows, self._episode_rows = self._episode_rows, []
        return rows

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def build_text_obs(
        self,
        text_obs: List[str],
        action_masks: List[List[bool]],
        init: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Render ``(system_prompts, user_prompts)`` for the current step.

        - System prompt: role + state-format description + cached
          ``board_static`` (unchanged per episode).
        - User prompt: dynamic obs + valid-actions list. WITH_HIS template
          fires per env when EITHER the action history is non-empty OR a
          rejection streak is active; NO_HIS otherwise. When memory is
          empty but rejections are active, ``action_history`` gets a
          ``"(no valid actions yet)"`` placeholder so the template still
          renders cleanly.
        """
        prompts = self._prompts
        state_format_desc = prompts.get_state_format_desc(self._state_format)

        # Per-env action history strings (empty when init or history disabled).
        memory_contexts: List[str] = ["" for _ in range(len(text_obs))]
        if not init and self._history_length > 0:
            for env_idx in range(len(text_obs)):
                recent = self.memory._data[env_idx][-self._history_length:]
                # Use the step number captured at append time — list indices
                # don't track real step numbers because invalid attempts skip
                # memory, so an index-derived label would mis-number history.
                lines = [f"Step {rec['step']}: {rec['action']}" for rec in recent]
                memory_contexts[env_idx] = "\n".join(lines)

        system_prompts: List[str] = []
        user_prompts: List[str] = []
        for i in range(len(text_obs)):
            valid_actions_str = (
                format_valid_actions(action_masks[i]) if action_masks[i] else ""
            )
            _, dynamic_obs = split_board_static(text_obs[i], self._state_format)

            system = prompts.CADAGENT_SYSTEM_PROMPT.format(
                state_format_desc=state_format_desc,
                board_static=self._board_static_cache[i],
            )

            # Init-step or no-history-tracking: always NO_HIS.
            # Otherwise: WITH_HIS only when memory non-empty OR rejection
            # streak active — keeps the prompt structurally identical to
            # NO_HIS on the first generation step (no dead History section).
            history_text = memory_contexts[i] if not init else ""
            rejected = (
                render_rejection_streak(self._last_rejected[i])
                if (not init and i < len(self._last_rejected))
                else ""
            )
            if init or self._history_length <= 0 or (not history_text and not rejected):
                user = prompts.CADAGENT_USER_PROMPT_NO_HIS.format(
                    dynamic_observation=dynamic_obs,
                    valid_actions=valid_actions_str,
                )
            else:
                user = prompts.CADAGENT_USER_PROMPT.format(
                    current_step=self._total_steps[i] + 1,
                    valid_step_count=len(self.memory[i]),
                    # Placeholder so the template still renders when memory
                    # is empty but rejections are active (otherwise the model
                    # gets the streak feedback inside an empty history block).
                    action_history=history_text or "(no valid actions yet)",
                    dynamic_observation=dynamic_obs,
                    valid_actions=valid_actions_str,
                    rejected_attempts=rejected,
                )
            system_prompts.append(system)
            user_prompts.append(user)

        return system_prompts, user_prompts

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release env resources."""
        if hasattr(self.envs, "close"):
            self.envs.close()


__all__ = ["KiCadLLMRolloutManager", "LLMRolloutBackend"]
