"""Local RL policy provider for the interactive visualizer.

Loads a ``KiCadRLModel`` checkpoint saved by
``methods/rl_agent/training/train_{ppo,grpo}.py`` and emits one decoded action
per observation. Conforms structurally to the branch-neutral
:class:`methods._shared.agent.Agent` contract.

Unlike the LLM-style providers, this consumes the raw env obs dict (not
a text prompt). The caller MUST pass the obs list via the
``observations=`` kwarg of :meth:`generate`.

Action wire format
------------------
``generate`` returns each action as an LLM-format envelope:

    "<think>{human-readable RL action}</think>"
    "<action>{action_name} {arg1} {arg2} ...</action>"

The ``<action>`` body is parsed by
:func:`methods.llm_agent.wrappers.action_converter._parse_action_text` into the env
action_dict — i.e. the visualizer's existing manual/LLM apply path
(``cadagent_projection`` → ``env.step``) handles RL output unchanged.
``<think>`` carries the structured-action summary for the history pane.

Checkpoint contract (v55/v56, as written by ``scripts/train.py``):

    {
        "iteration":            int,
        "policy_state_dict":    dict,        # required
        "optimizer_state_dict": dict,        # not needed for inference
        "metrics":              dict,
        "args":                 vars(args),  # full training argparse — used
                                             # to rebuild the policy architecture
    }
"""

from __future__ import annotations

import time

# Codec helpers — the visualizer drives the bare PCBWorld (no wrapper), so we
# reuse the same stateless codec the wrapper uses (de-dup of former private
# copies _sorted_net_codes_from_obs / _net_valid_mask_from_obs).
from methods.rl_agent.models.v1.encoding import (
    net_valid_mask,
    sorted_net_codes_from_obs,
)


# Map routing_mode int → single-letter token consumed by
# pcb_world.core.action_schema.parse_mode (0=MarkObstacles, 1=Shove, 2=Walkaround).
from pcb_world.core.action_schema import MODE_INT_TO_LETTER as _MODE_INT_TO_LETTER


def _build_action_envelope(
    action_type: int,
    pointer_idx: int,
    routing_mode: int,
    cand_mm: list[tuple[float, float, int]],
    sorted_net_codes: list[int],
    policy_net_select: bool,
    human_text: str,
) -> str:
    """Convert ``(at, ptr, mode)`` → an LLM-format envelope that
    :func:`methods.llm_agent.wrappers.action_converter._parse_action_text` understands.

    Mirrors :meth:`methods.rl_agent.wrappers.adapter.KiCadRLWrapper._decode_action`
    one-to-one so the env action_dict produced by the visualizer matches
    what the training-time wrapper would have produced.

    Returns the full envelope. ``<think>`` carries ``human_text`` (the
    output of :func:`human_render.render_action_human`) so the history
    pane shows a readable summary alongside the parseable command.
    """
    from pcb_world.core.masking import (
        ACT_NET_SELECT, ACT_START_ROUTE, ACT_NET_END,
        ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_FINISH,
    )

    mode_letter = _MODE_INT_TO_LETTER.get(int(routing_mode), "w")
    at = int(action_type)
    ptr = int(pointer_idx)

    def _cand_xy(idx: int) -> tuple[float, float, int] | None:
        if 0 <= idx < len(cand_mm):
            x, y, ly = cand_mm[idx]
            return float(x), float(y), int(ly)
        return None

    body: str | None = None

    if at == ACT_NET_SELECT:
        if policy_net_select and 0 <= ptr < len(sorted_net_codes):
            body = f"net_select {sorted_net_codes[ptr]}"
        elif not policy_net_select and sorted_net_codes:
            # Trained without policy-side net selection — wrapper picked
            # randomly from unrouted nets. Visualizer's simplification:
            # pick the first net so the env still steps deterministically.
            # Suboptimal vs the wrapper's random pick but never invalid.
            body = f"net_select {sorted_net_codes[0]}"
    elif at == ACT_START_ROUTE:
        xy = _cand_xy(ptr)
        if xy is not None:
            body = f"start_route {xy[0]:.3f} {xy[1]:.3f} {xy[2]}"
    elif at == ACT_NET_END:
        body = "net_end"
    elif at == ACT_MAKE_LINE:
        xy = _cand_xy(ptr)
        if xy is not None:
            body = f"make_line {xy[0]:.3f} {xy[1]:.3f} {mode_letter}"
    elif at == ACT_MAKE_VIA:
        xy = _cand_xy(ptr)
        if xy is not None:
            body = f"make_via {xy[0]:.3f} {xy[1]:.3f} {mode_letter}"
    elif at == ACT_FINISH:
        body = f"finish {mode_letter}"
    # ACT_IDLE (6) and out-of-range pointers fall through with body=None,
    # letting cadagent_projection emit its _FALLBACK_ACTION (idle).

    if body is None:
        return f"<think>{human_text} (no valid env action — idle fallback)</think>"
    return f"<think>{human_text}</think><action>{body}</action>"


def _resolve_rl_ckpt_file(path: str) -> str:
    """Accept either a ckpt file or a run directory.

    Directories follow the training run layout (``checkpoints/<RUN_NAME>/``):
    prefer ``policy_best.pt`` if present, else the highest-numbered
    ``policy_iter_<N>.pt``.
    """
    import os
    import re
    import glob
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise ValueError(f"RL checkpoint path is neither file nor dir: {path!r}")
    best = os.path.join(path, "policy_best.pt")
    if os.path.isfile(best):
        return best
    iter_files: list[tuple[int, str]] = []
    for p in glob.glob(os.path.join(path, "policy_iter_*.pt")):
        m = re.match(r".*policy_iter_(\d+)\.pt$", p)
        if m:
            iter_files.append((int(m.group(1)), p))
    if not iter_files:
        raise ValueError(
            f"no policy_best.pt or policy_iter_<N>.pt in dir: {path!r}"
        )
    iter_files.sort()
    return iter_files[-1][1]


class RLPolicyProvider:
    """Local RL policy provider — see module docstring."""

    def __init__(self, args):
        from methods.rl_agent.policy.agent import KiCadRLAgent

        if not getattr(args, "checkpoint_path", None):
            raise ValueError(
                "RLPolicyProvider requires --checkpoint_path (a "
                "policy_iter_<N>.pt / policy_best.pt file or its run dir)."
            )
        ckpt_file = _resolve_rl_ckpt_file(args.checkpoint_path)
        device = getattr(args, "device", None) or "cuda"

        print("\n" + "=" * 60)
        print(f"  Loading RL policy: {ckpt_file}")
        print("=" * 60)
        t0 = time.time()

        # ``temperature == 0`` → greedy / argmax. The RL model has no
        # top-k/top-p surface (see decoder_only_policy.py:616), so this is
        # the only sampling knob.
        deterministic = bool(getattr(args, "temperature", 0.7) == 0.0)
        # Single load path (methods.rl_agent.models.loader via the agent facade):
        # legacy-arg inference + use_critic from the state_dict + tolerant
        # action-history/drc missing keys, hard mismatches raise.
        self._agent = KiCadRLAgent.from_checkpoint(
            ckpt_file, device=device, deterministic=deterministic,
        )
        self.policy = self._agent.model  # back-compat attribute
        ta: dict = self._agent.ckpt_args

        self.device = self._agent.device
        self.deterministic = deterministic
        # Match training obs distribution: when training ran with
        # ``--no-drc-tokens``, the tokenizer didn't see ``drc_violations``,
        # so we strip them here too. See ``BatchedStateTokenizer`` ~line 408.
        self.no_drc_tokens = bool(ta.get("no_drc_tokens", False))
        # When training ran with policy-driven net selection, infer net_valid_mask
        # at inference too — otherwise the policy keeps picking already-routed
        # nets (training-time distribution mismatch).
        self.policy_net_select = bool(ta.get("policy_net_select", False))
        # KiCadRLWrapper default is True (training arg likely missing
        # from older runs — match wrapper default rather than False so we
        # don't silently break parity for policies that learned with it on).
        self.mask_start_point = bool(ta.get("mask_start_point") or True)
        # Wrapper state we re-derive on the inference side: the candidate
        # key (x, y, layer) of the most recent ACT_START_ROUTE the policy
        # emitted, kept until ACT_NET_END / ACT_NET_SELECT (or auto-cleared
        # when the env reports is_routing=False, which handles env.reset too).
        self._last_start_route_xy: tuple[float, float, int] | None = None
        self.train_iteration = int(self._agent.train_iteration)
        self._train_args = ta  # kept for debugging / info echoes

        print(f"  Loaded in {time.time() - t0:.1f}s  "
              f"iter={self.train_iteration}  "
              f"use_critic={ta.get('use_critic', True)}  "
              f"no_drc_tokens={self.no_drc_tokens}  device={device}  "
              f"deterministic={self.deterministic}")

    def generate(self, prompt_pairs, observations=None, *,
                 action_masks=None, raw_tokens: bool = False):
        """Emit one decoded action per observation.

        ``prompt_pairs`` is accepted for interface uniformity but ignored:
        the RL policy consumes the raw obs dict, not text. Pass the obs
        list via ``observations=`` — length must match ``prompt_pairs``.

        ``action_masks`` (optional) — per-obs boolean masks of valid
        action types, shape ``(N, NUM_ACTIONS)`` (numpy or torch). Source
        is :meth:`PCBWorld.action_masks` (one row per obs). Without it
        the decoder can sample an action_type whose pointer/mode head has
        no valid output, producing NaN logits at terminal-like states.
        Matches the training-time path
        (``methods/rl_agent/training/collect.py``).

        ``raw_tokens=True`` switches the response's ``<think>`` body to
        the demo integer triple ``[at, ptr, mode]``. The ``<action>``
        body remains the LLM-format parseable command either way so
        ``cadagent_projection → env.step`` doesn't care.
        """
        import torch
        from pcb_world.vec.candidate_pool import (
            collect_raw_candidates,
            build_directional_candidates,
        )
        from methods.rl_agent.policy.human_render import (
            render_action_human, count_state_entities,
        )

        if observations is None:
            raise RuntimeError(
                "RLPolicyProvider.generate requires the `observations` kwarg "
                "(list of raw PCBWorld obs dicts). LLM/API providers ignore "
                "it; only the RL path threads obs through the worker."
            )
        if len(observations) != len(prompt_pairs):
            raise RuntimeError(
                f"observations length ({len(observations)}) does not match "
                f"prompt_pairs length ({len(prompt_pairs)})"
            )

        masks_t = None
        if action_masks is not None:
            import numpy as np
            arr = np.asarray(action_masks, dtype=bool)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[0] != len(observations):
                raise RuntimeError(
                    f"action_masks first dim ({arr.shape[0]}) does not match "
                    f"observations length ({len(observations)})"
                )
            masks_t = torch.from_numpy(arr)

        from pcb_world.core.masking import (
            ACT_START_ROUTE, ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_FINISH,
        )

        responses: list[str] = []
        token_counts: list[tuple[int, int, int]] = []
        for i, obs in enumerate(observations):
            if self.no_drc_tokens and "drc_violations" in obs:
                # Shallow-copy so caller's obs is untouched.
                obs = {**obs, "drc_violations": []}

            # Rebuild cand pool first — needed both for pointer_masks (built
            # before policy call) and for envelope construction after.
            # collect_raw_candidates mirrors the tokenizer at
            # state_tokenizer_batched.py:467.
            rh = obs.get("router_head", {})
            cn = rh.get("current_net", -1)
            cur_net = cn if isinstance(cn, int) and cn > 0 else None
            extra = None
            if rh.get("is_routing", False):
                hxy = rh["current_xy"]
                _pa = obs.get("_aug") or {}
                extra = build_directional_candidates(
                    (hxy[0], hxy[1]), rh["current_layer"],
                    mode=_pa.get("directional_candidates"),
                )
            raw_cands = collect_raw_candidates(obs, cur_net, extra)
            cand_mm = [(x, y, ly) for (x, y, ly, _ct) in raw_cands]

            # Auto-clear stale start-route state when no net is active
            # (env.reset → current_net=-1, ACT_NET_END → current_net=-1).
            # NB: After ACT_FINISH ``is_routing=False`` but ``current_net``
            # is still set — the wrapper keeps ``_start_route_xy`` in that
            # state (so the next ACT_START_ROUTE on the same net can't
            # re-pick the just-used pad), so we mirror that here.
            if rh.get("current_net", -1) == -1:
                self._last_start_route_xy = None

            # Without action_masks, the pointer/mode head can produce NaN
            # logits at terminal-like states (empty cand pool, no valid net).
            # Threading env.action_masks() through matches training-time
            # collect.py call shape.
            per_obs_mask = masks_t[i:i + 1] if masks_t is not None else None

            # net_valid_mask: only meaningful when the policy was trained
            # with policy_net_select=True (otherwise the wrapper picked
            # nets externally and the policy's pointer logits over the
            # net pool were never used in the loss). Reconstructed from
            # obs.routing_geometry so the visualizer can stay on the
            # bare PCBWorld (no wrapper).
            extra_kwargs: dict = {}
            if self.policy_net_select:
                import numpy as np
                sorted_codes_for_mask = sorted_net_codes_from_obs(obs)
                nvm = net_valid_mask(
                    sorted_net_codes=sorted_codes_for_mask, last_obs=obs,
                )
                extra_kwargs["net_valid_mask"] = torch.from_numpy(
                    nvm[None, :]
                )
                extra_kwargs["allow_net_select_lp"] = True

            # pointer_masks (mask_start_point): exclude the cand equal to
            # the active start_route key (x, y, layer) — mirrors
            # KiCadRLWrapper.start_route_pointer_indices (layer-aware).
            # Without this, the policy can re-pick the start pad
            # as a routing target (visible in the GUI as a
            # start_route→finish FAIL loop on the same coord).
            # Provider-side state since the bare PCBWorld has no wrapper.
            if (
                self.mask_start_point
                and self._last_start_route_xy is not None
                and cand_mm
            ):
                sx, sy, sl = self._last_start_route_xy
                pm_idxs = [
                    k for k, (cx, cy, ly) in enumerate(cand_mm)
                    if abs(cx - sx) < 0.01 and abs(cy - sy) < 0.01
                    and ly == sl
                ]
                if pm_idxs:
                    extra_kwargs["pointer_masks"] = torch.tensor(
                        [pm_idxs], dtype=torch.long,
                    )

            # Inference through the agent facade (no_grad inside .act).
            actions, _logp = self._agent.act(
                [obs],
                action_masks=per_obs_mask,
                **extra_kwargs,
            )
            at, ptr, rm = (int(x) for x in actions[0].tolist())

            # Update _last_start_route_xy after decoding the action — same
            # transition rules as KiCadRLWrapper.step (lines 423-432).
            # Done here, not inside the env.step caller, because the
            # visualizer uses the bare PCBWorld.
            if at == ACT_START_ROUTE:
                if 0 <= ptr < len(cand_mm):
                    self._last_start_route_xy = (
                        float(cand_mm[ptr][0]), float(cand_mm[ptr][1]),
                        int(cand_mm[ptr][2]),
                    )
                # else: rejected pointer — wrapper leaves state unchanged.
            elif at not in (ACT_MAKE_LINE, ACT_MAKE_VIA, ACT_FINISH):
                self._last_start_route_xy = None

            # Summary for the <think> block — demo integer triple when
            # raw_tokens, else verbose human-readable form.
            human_text = render_action_human(
                at, ptr, rm, cand_mm, raw_tokens=raw_tokens,
            )
            # Build an LLM-format envelope so the visualizer's existing
            # cadagent_projection → env.step path consumes the RL output
            # unchanged. Without this, the GUI would treat the response
            # as unparseable text and fall through to the idle fallback,
            # leaving the board unmodified.
            sorted_codes = sorted_net_codes_from_obs(obs)
            envelope = _build_action_envelope(
                at, ptr, rm,
                cand_mm=cand_mm,
                sorted_net_codes=sorted_codes,
                policy_net_select=bool(
                    self._train_args.get("policy_net_select", True)
                ),
                human_text=human_text,
            )
            responses.append(envelope)
            # Counts: (sys=0, user=full state-token count, out=1). The user
            # column matches what the policy actually consumes — see
            # human_render.count_state_entities for the per-entity breakdown.
            token_counts.append((0, count_state_entities(
                obs,
                obstacle_obs=bool(self._train_args.get("obstacle_obs", False)),
                action_history_len=int(
                    self._train_args.get("action_history_len", 1)
                ),
            ), 1))

        return responses, token_counts

    def close(self) -> None:
        # Drop refs and let CUDA / GC reclaim. No explicit policy.close().
        self.policy = None
