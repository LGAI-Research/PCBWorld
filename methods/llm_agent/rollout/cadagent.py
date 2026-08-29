"""Eval script for the cadagent KiCad PCB routing environment.

Two modes:
    --mode fixed   (default) Run a single reset + one fixed net_select step.
    --mode llm     Load a model via vLLM and run full rollout episodes.

Supported model families & sizes:
    +-----------+---------------------------+-------------------+----------+
    | Family    | Model ID (HuggingFace)    | Params            | Type     |
    +-----------+---------------------------+-------------------+----------+
    | Qwen2.5   | Qwen/Qwen2.5-3B-Instruct | 3B                | Causal   |
    |           | Qwen/Qwen2.5-7B-Instruct | 7B                | Causal   |
    |           | Qwen/Qwen2.5-14B-Instruct| 14B               | Causal   |
    |           | Qwen/Qwen2.5-32B-Instruct| 32B               | Causal   |
    |           | Qwen/Qwen2.5-72B-Instruct| 72B               | Causal   |
    +-----------+---------------------------+-------------------+----------+
    | Qwen2.5-VL| Qwen/Qwen2.5-VL-3B-Instruct  | 3B           | Vision   |
    |           | Qwen/Qwen2.5-VL-7B-Instruct  | 7B           | Vision   |
    |           | Qwen/Qwen2.5-VL-32B-Instruct | 32B          | Vision   |
    |           | Qwen/Qwen2.5-VL-72B-Instruct | 72B          | Vision   |
    +-----------+---------------------------+-------------------+----------+
    | LLaMA 3.1 | meta-llama/Llama-3.1-8B-Instruct   | 8B      | Causal   |
    |           | meta-llama/Llama-3.1-70B-Instruct  | 70B      | Causal   |
    |           | meta-llama/Llama-3.1-405B-Instruct | 405B     | Causal   |
    +-----------+---------------------------+-------------------+----------+
    | LLaMA 3.3 | meta-llama/Llama-3.3-70B-Instruct  | 70B      | Causal   |
    +-----------+---------------------------+-------------------+----------+
    | Mistral   | mistralai/Mistral-7B-Instruct-v0.3  | 7B      | Causal   |
    | Mixtral   | mistralai/Mixtral-8x7B-Instruct-v0.1| 47B(MoE)| Causal   |
    |           | mistralai/Mixtral-8x22B-Instruct-v0.1|141B(MoE)| Causal  |
    +-----------+---------------------------+-------------------+----------+
    | Gemma 2   | google/gemma-2-9b-it      | 9B                | Causal   |
    |           | google/gemma-2-27b-it     | 27B               | Causal   |
    +-----------+---------------------------+-------------------+----------+
    | DeepSeek  | deepseek-ai/DeepSeek-V2-Lite-Chat   | 16B    | Causal   |
    |           | deepseek-ai/DeepSeek-V2.5           | 236B(MoE)| Causal |
    |           | deepseek-ai/DeepSeek-V3-0324        | 671B(MoE)| Causal |
    +-----------+---------------------------+-------------------+----------+
    | InternVL2 | OpenGVLab/InternVL2-8B    | 8B                | Vision   |
    |           | OpenGVLab/InternVL2-26B   | 26B               | Vision   |
    +-----------+---------------------------+-------------------+----------+

    Notes:
    - Vision models are auto-detected and loaded with the appropriate class.
    - Any HuggingFace model supporting AutoModelForCausalLM can be used
      even if not listed above.
    - bf16 VRAM estimate: ~2 GB per 1B params. Use --gpu_devices or
      --num_gpus for models that exceed a single GPU's memory.

Usage:
    # Fixed-action smoke test (no GPU needed)
    python methods/llm_agent/rollout/cadagent.py --board_path /path/to/board.kicad_pcb

    # LLM rollout (requires GPU)
    python methods/llm_agent/rollout/cadagent.py --mode llm --model_path Qwen/Qwen2.5-VL-7B-Instruct \
        --rollout_episodes 10 --max_steps 50

    # Large model on specific GPUs
    python methods/llm_agent/rollout/cadagent.py --mode llm --model_path meta-llama/Llama-3.1-70B-Instruct \
        --gpu_devices 0,1,2,3 --rollout_episodes 5
"""

from __future__ import annotations

import argparse
import sys
import os

from methods.llm_agent.wrappers.action_converter import cadagent_projection
from methods.llm_agent.training.manager import KiCadLLMRolloutManager


# ---------------------------------------------------------------------------
# sys.path bootstrap
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    from eval.args import build_eval_parser
    return build_eval_parser(description="cadagent environment eval").parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def _build_envs(args, board_path: str | None = None):
    """Build KiCadLLMVecEnv, optionally overriding the initial board path.

    When ``board_path`` is ``None`` the workers are built against
    ``args.board_path`` (legacy single-board behaviour).
    """
    from methods.llm_agent.wrappers.factory import KiCadLLMVecEnv

    return KiCadLLMVecEnv(
        board_path=board_path if board_path is not None else args.board_path,
        seed=args.seed,
        env_num=args.env_num,
        group_n=1,
        resources_per_worker={"num_cpus": args.num_cpus_per_worker, "num_gpus": 0},
        is_train=False,
        env_kwargs={
            "max_steps":             args.max_steps,
            "masking_rule":          args.masking_rule,
            "reward_rule":           args.reward_rule,
            "state_format":          args.state_format,
            "corner_mode":           args.corner_mode,
            "via_penalty":           args.via_penalty,
            "reward_noise_std":      args.reward_noise_std,
            "emit_drc_tokens":       not args.no_drc_tokens,
            "use_yaml_drc_fallback": args.use_yaml_drc_fallback,
        },
    )


def _resolve_eval_boards(args) -> list[tuple[str, str]]:
    """Return ``[(board_id, board_path), ...]`` for the current CLI args.

    - ``--boards_order=single``: single-element list derived from
      ``--board_path`` (legacy behaviour preserved).
    - ``--boards_order=round_robin``: full list from ``--boards_json``
      filtered by ``--boards_difficulty`` / ``--boards_split``, sorted
      ascending by pad count (gentle curriculum order). Each board gets
      ``--rollout_episodes`` episodes.
    """
    boards_order = getattr(args, "boards_order", "single")
    if boards_order == "single":
        bid = os.path.splitext(os.path.basename(args.board_path))[0]
        return [(bid, args.board_path)]

    from methods._shared.board_scheduler import BoardScheduler, BoardSchedulerConfig
    sched = BoardScheduler(BoardSchedulerConfig(
        mode=boards_order,
        single_board=args.board_path,
        boards_json=args.boards_json,
        difficulty=args.boards_difficulty,
        split=args.boards_split,
    ))
    paths = list(sched.paths)
    # board_id: file stem when those are unique (flat synth layout,
    # ``<dir>/board_00003.kicad_pcb``). For per-board-dir layouts where every
    # board shares one filename (e.g. d3's ``<dir>/processed_v9_guide_v3.kicad_pcb``)
    # the stems all collide and would overwrite each other under per_board/,
    # so fall back to the parent directory name. Mirrors
    # eval_rollouts.select_real_boards.
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(set(stems)) == len(stems):
        return list(zip(stems, paths))
    return [(os.path.basename(os.path.dirname(p)), p) for p in paths]


def _iter_boards_with_reload(envs, eval_boards, env_num: int):
    """Yield ``(board_idx, board_id, board_path)`` tuples.

    For every board after the first, the workers are hot-swapped via
    :meth:`KiCadLLMVecEnv.reload_boards` before the tuple is yielded so
    callers can issue ``envs.reset()`` on the new board directly.
    """
    for idx, (bid, path) in enumerate(eval_boards):
        if idx > 0:
            envs.reload_boards([path] * env_num)
        yield idx, bid, path


# (display_label, result_dict_key) pairs for the human console summary in
# _print_board_summary. The machine summary (per_rollout/per_board CSV +
# eval_overall_summary.json) is produced by the shared evaluator handler via
# _result_to_canonical_row -> EvalSummary, not from this list.
# Order matters: this is the print order in the per-board summary block (mirrors
# the per-board DRC summary layout).
# ``ratsnest_reduction``, ``track_count`` come straight from PCBWorld's per-step
# info (no DRC re-eval needed); ``final_potential`` is populated by
# PCBWorld only at the terminal/truncated step, so it is NaN for early-
# stopped runs — :func:`_mean_std` skips NaNs so the mean reports over the
# subset that has it.
_METRIC_KEYS = [
    ("reward", "reward"),
    ("steps", "steps"),
    ("ratsnest_red", "ratsnest_reduction"),
    ("tracks", "track_count"),
    ("via", "via_count"),
    ("wirelength", "wirelength"),
    ("drv", "drc_violations"),
    ("final_potential", "final_potential"),
    ("system_tokens", "system_tokens"),
    ("response_tokens", "response_tokens"),
]


def _is_finite(x) -> bool:
    """True if ``x`` is a real number (not None, not NaN)."""
    if x is None:
        return False
    try:
        return x == x  # NaN != NaN
    except Exception:  # noqa: BLE001
        return False


def _mean_std(values: list) -> tuple[float, float]:
    """Mean / sample-stdev over numeric, non-NaN, non-None entries.

    Returns (NaN, NaN) when no finite values are present so missing measures
    (e.g. ``final_potential`` on early-stopped runs) bubble through summary
    print and CSV columns without blowing up the rollup.
    """
    finite = [float(v) for v in values if _is_finite(v)]
    n = len(finite)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(finite) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in finite) / (n - 1)
    return mean, var ** 0.5


def _result_to_canonical_row(
    result: dict, *, board_index: int, board_path: str, rollout_idx: int
) -> dict:
    """Map one LLM episode result dict to a canonical ``PER_ROLLOUT_FIELDS`` row.

    Assembly goes through the shared kernel
    (:func:`eval.eval_utils.assemble_rollout_row`) so RL and LLM rows come
    from one place. The metric columns come from the nested ``eval_metrics``
    block (episode-end canonical DRC scoring via the worker's
    ``eval_inline_drc`` — the same ruler as the RL eval path), which
    overwrites the env-info fallbacks below. The fallbacks remain only for
    crashed episodes and pre-scoring dumps (old resume dirs) without an
    ``eval_metrics`` block. The bespoke per-episode result dict and the
    ``{stem}.json`` dump are kept as-is (``visualize`` reads
    ``result["won"]``); this mapper exists only for the summary/aggregation
    handler.

    Name reconciliation: ``reward`` -> ``episode_return``, ``wirelength`` ->
    ``wirelength_mm``, ``won`` -> ``success``/``terminated`` (success/clean_pass
    go through the shared semantics helpers in :mod:`eval.eval_utils`).
    RL-only columns are left blank (parsed as NaN, excluded from means).
    """
    from configs.loader.schema import DEFAULTS
    from eval import eval_utils as u

    won = bool(result.get("won", False))
    crashed = bool(result.get("crashed", False))
    scored = result.get("eval_metrics")
    # The result dict carries no unrouted count, so success falls back to
    # ``won`` — which the adapter stamps as bare ``bool(terminated)``
    # (methods/llm_agent/wrappers/adapter.py). A voluntary all-nets-closed
    # finish also sets ``terminated`` without implying full routing, so
    # ``won`` (and the ``terminated: won`` mapping below) conflates give-up
    # with success — intentional (LLM-owned definition); the canonical
    # ``eval_metrics`` block overwrites success/clean_pass whenever inline
    # scoring ran, so this only shows on crashed/unscored episodes.
    success = u.success_from(terminated=won, crashed=crashed)
    return u.assemble_rollout_row(
        scored if isinstance(scored, dict) else None,
        identity={
            "board_index": board_index,
            "board_id": result.get("board_id", ""),
            "board_path": board_path,
            "rollout_idx": rollout_idx,
        },
        runtime={
            "status": "crashed" if crashed else "completed",
            "crashed": crashed,
            "success": success,
            "clean_pass": u.clean_pass_from(
                success, result.get("drv_errors_only_count"),
            ),
            "terminated": won,
            "steps": result.get("steps", ""),
            "episode_return": result.get("reward", ""),
            "ratsnest_reduction": result.get("ratsnest_reduction", ""),
            "drc_violations": result.get("drc_violations", ""),
            "wirelength_mm": result.get("wirelength", ""),
            "via_count": result.get("via_count", ""),
            "track_count": result.get("track_count", ""),
            "final_potential": result.get("final_potential", ""),
            "early_stop": bool(result.get("early_stopped", False)),
            "system_tokens": result.get("system_tokens", ""),
            "user_tokens": result.get("user_tokens", ""),
            "response_tokens": result.get("response_tokens", ""),
            "call_count": result.get("call_count", ""),
        },
        reward_config=DEFAULTS.reward_config,
        check_angle=DEFAULTS.check_angle,
    )


def _crashed_result(board_id: str, env_i: int) -> dict:
    """Placeholder result for an episode that crashed mid-run.

    A worker death (native router SIGSEGV / OOM) or any unexpected exception
    during the episode body lands here: metrics are NaN (skipped by
    :func:`_mean_std`), ``won`` is False (counts as a non-success), and
    ``crashed`` flags it. The episode is intentionally NOT dumped, so a later
    resume run re-attempts it. Same key set as a normal result row.
    """
    nan = float("nan")
    return {
        "board_id": board_id, "env": env_i, "steps": 0, "reward": nan,
        "won": False, "drc_violations": nan, "wirelength": nan,
        "via_count": nan, "track_count": nan, "ratsnest_reduction": nan,
        "final_potential": nan,
        "system_tokens": nan, "user_tokens": nan,
        "response_tokens": nan, "call_count": 0,
        "early_stopped": False, "crashed": True,
    }


def _load_episode_results(
    dump_dir: str, board_id: str, ep: int, env_num: int
) -> list[dict] | None:
    """Return per-env result rows for an already-dumped episode, or None.

    Used for resume: an episode counts as complete only when, for every env,
    both ``{stem}.json`` and ``{stem}.kicad_pcb`` exist and the JSON carries a
    ``result`` block. Partial dumps (e.g. crashed mid-dump) return None so the
    episode is re-run and overwritten. Reconstructs the in-memory result rows
    from the JSON's ``result`` block so the summary CSV stays complete across
    resumes.
    """
    import json
    rows: list[dict] = []
    for i in range(env_num):
        stem = f"{board_id}_episode_{ep:02d}_env_{i:02d}"
        jpath = os.path.join(dump_dir, f"{stem}.json")
        pcbpath = os.path.join(dump_dir, f"{stem}.kicad_pcb")
        if not (os.path.isfile(jpath) and os.path.isfile(pcbpath)):
            return None
        try:
            with open(jpath) as f:
                doc = json.load(f)
        except Exception:  # noqa: BLE001 — unreadable/partial JSON → re-run
            return None
        res = doc.get("result")
        if not isinstance(res, dict):
            return None
        nan = float("nan")
        rows.append({
            "board_id": board_id, "env": i,
            "steps": res.get("steps", 0), "reward": res.get("reward", nan),
            "won": bool(res.get("won", False)),
            "drc_violations": res.get("drc_violations", nan),
            "wirelength": res.get("wirelength", nan),
            "via_count": res.get("via_count", nan),
            "track_count": res.get("track_count", nan),
            "ratsnest_reduction": res.get("ratsnest_reduction", nan),
            "final_potential": res.get("final_potential", nan)
                if res.get("final_potential") is not None else nan,
            "system_tokens": res.get("system_tokens", nan),
            "user_tokens": res.get("user_tokens", nan),
            "response_tokens": res.get("response_tokens", nan),
            "call_count": res.get("call_count", 0),
            "early_stopped": bool(res.get("early_stopped", False)),
            # None for dumps written before episode-end scoring existed —
            # the canonical-row mapper then falls back to the env-info
            # fields above, keeping old dump dirs resumable.
            "eval_metrics": res.get("eval_metrics"),
            "resumed": True,
        })
    return rows


def _print_board_summary(board_id: str, results: list[dict], args) -> None:
    """Print a summary block for one board's rollouts.

    Layout mirrors the per-board DRC summary
    (mean ± sample stdev over all runs, with success-only min/mean/max
    appended for the routing-quality metrics). ``results`` contains all
    per-(episode, env) records for a single board; no cross-board
    aggregation is performed — each board is reported in isolation so
    cross-difficulty averages never appear.

    Notes vs. the post-hoc DRC scorer (eval.metrics.evaluate_one):
      - We do NOT call ``evaluate_one`` per board, so DRV severity split
        (errors_only / errors_and_promoted), ``track_angle_drv``,
        ``total_drv_count`` and ``clean_pass_rate`` are absent. Only
        the env-reported single ``drc_violations`` count is shown.
      - ``final_potential`` is reported only for runs where the env
        reached a terminal/truncated step (PCBWorld only stamps the
        field then). Early-stopped runs contribute NaN and are skipped
        by the mean.
    """
    total = len(results)
    successes = sum(1 for r in results if r["won"])
    early_stops = sum(1 for r in results if r.get("early_stopped"))
    finals_seen = sum(1 for r in results if _is_finite(r.get("final_potential")))

    _print_section(f"SUMMARY [{board_id}]")
    print(
        f"  Episodes    : {args.rollout_episodes} x {args.env_num} envs = "
        f"{total} runs  (early-stopped: {early_stops})"
    )
    print(
        f"  Success rate: {successes}/{total} "
        f"({100 * successes / max(total, 1):.1f}%)"
    )
    print(
        f"  Final-Φ runs: {finals_seen}/{total} "
        f"(env stamps final_potential only at terminal/truncated steps)"
    )

    # mean ± stdev over all runs — DRC-summary-style block.
    print("\n  mean ± stdev across all runs:")
    metric_pad = max(len(label) for label, _ in _METRIC_KEYS)
    for label, key in _METRIC_KEYS:
        mean, std = _mean_std([r.get(key) for r in results])
        print(f"    {label:<{metric_pad}}  {mean:>10.4f} ± {std:>9.4f}")

    # Success-only min/mean/max for the routing-quality metrics that
    # only make sense on completed runs (e.g. wirelength on a failed
    # episode is just whatever partial trace exists).
    success_keys = [
        ("reward", "reward"),
        ("steps", "steps"),
        ("wirelength", "wirelength"),
        ("via", "via_count"),
        ("tracks", "track_count"),
    ]
    successful = [r for r in results if r["won"]]
    if successful:
        print("\n  success-only min / mean / max:")
        succ_pad = max(len(label) for label, _ in success_keys)
        for label, key in success_keys:
            vals = [r[key] for r in successful if _is_finite(r.get(key))]
            if vals:
                print(
                    f"    {label:<{succ_pad}}  "
                    f"min={min(vals):>10.4f}  "
                    f"mean={sum(vals) / len(vals):>10.4f}  "
                    f"max={max(vals):>10.4f}"
                )
    else:
        print("\n  success-only min / mean / max: (no successful runs)")

    # Token usage block — unchanged.
    total_sys = sum(r["system_tokens"] for r in results)
    total_usr = sum(r["user_tokens"] for r in results)
    total_resp = sum(r["response_tokens"] for r in results)
    total_calls = sum(r["call_count"] for r in results)
    print(f"\n  --- Token Usage ---")
    print(
        f"  Total   : system={total_sys:,}  user={total_usr:,}  "
        f"response={total_resp:,}  (all={total_sys+total_usr+total_resp:,})"
    )
    print(
        f"  Avg/ep  : system={total_sys/max(total,1):,.1f}  "
        f"user={total_usr/max(total,1):,.1f}  "
        f"response={total_resp/max(total,1):,.1f}"
    )
    print(
        f"  Avg/call: system={total_sys/max(total_calls,1):,.1f}  "
        f"user={total_usr/max(total_calls,1):,.1f}  "
        f"response={total_resp/max(total_calls,1):,.1f}  ({total_calls} calls)"
    )


class _EvalRolloutManager(KiCadLLMRolloutManager):
    """``KiCadLLMRolloutManager`` subclass that reads CLI argparse args.

    The base's default ``_resolve_*`` hooks expect verl-agent's nested
    OmegaConf shape (``config.env.cadagent.prompt_version``); eval scripts
    pass a flat argparse Namespace, so each hook is overridden to read the
    matching CLI flag directly. Multi-board cycling is handled externally
    by ``_iter_boards_with_reload`` (not the base's ``_board_scheduler``),
    so the scheduler arg stays ``None``.
    """

    def __init__(self, envs, args):
        # score_episodes: every eval episode ends with canonical DRC scoring
        # (worker eval_inline_drc) — the same ruler as the RL eval path.
        super().__init__(
            envs, cadagent_projection, args, board_scheduler=None,
            score_episodes=True,
        )

    def _resolve_prompt_version(self, args) -> str:
        return args.prompt_version

    def _resolve_base_max_steps(self, args) -> int:
        return int(args.max_steps)

    def _resolve_history_length(self, args) -> int:
        return int(args.history_length)


def _make_manager(envs, args) -> _EvalRolloutManager:
    """Construct the eval-side rollout manager bound to ``envs``."""
    return _EvalRolloutManager(envs, args)


def _info_action_masks(info_list: list, env_num: int) -> list:
    """Extract per-env action masks from infos (preferred over the env's
    live property when restoring frozen early-stop state)."""
    return [info_list[i].get("action_mask", []) for i in range(env_num)]


def _print_step_prompts(step: int, env_num: int, sys_prompts, usr_prompts) -> None:
    """Verbose per-step prompt dump (--verbose or step 0)."""
    for i in range(env_num):
        print(f"\n[step {step}] System prompt (env {i}):")
        print(sys_prompts[i])
        print(f"\n[step {step}] User prompt (env {i}):")
        print(usr_prompts[i])


# ---------------------------------------------------------------------------
# Fixed mode (no LLM)
# ---------------------------------------------------------------------------

def run_fixed(args) -> None:
    eval_boards = _resolve_eval_boards(args)
    envs = _build_envs(args, board_path=eval_boards[0][1])
    mgr = _make_manager(envs, args)
    print(f"  Workers spawned: {envs.num_processes}")
    print(f"  Prompt version : {mgr._prompts.version}")
    print(f"  Boards to smoke-test: {len(eval_boards)}")

    for board_idx, board_id, _board_path in _iter_boards_with_reload(
        envs, eval_boards, args.env_num
    ):
        _print_section(
            f"Board {board_idx + 1}/{len(eval_boards)}: {board_id}"
        )

        # 1. Reset
        _print_section("1. INITIAL STATE (after reset)")
        obs_dict, _ = mgr.reset()
        for i in range(args.env_num):
            print(f"\n{'─'*50}")
            print(f"  [env {i}] SYSTEM PROMPT")
            print(f"{'─'*50}")
            print(obs_dict["system"][i])
            print(f"\n{'─'*50}")
            print(f"  [env {i}] USER PROMPT")
            print(f"{'─'*50}")
            print(obs_dict["text"][i])

        # 2. Fixed actions (parsed via mgr.step's projection)
        _print_section("2. ACTION — net_select")
        llm_outputs = [
            f"<think>Selecting net {i + 1}.</think>\n<action>net_select {i + 1}</action>"
            for i in range(args.env_num)
        ]
        obs_dict, rewards, dones, _ = mgr.step(llm_outputs)
        for i in range(args.env_num):
            print(
                f"[env {i}] parsed: {mgr._last_actions[i]}  "
                f"(valid={mgr._last_valids[i]})"
            )

        # 3. Next state
        _print_section("3. NEXT STATE (after net_select)")
        for i in range(args.env_num):
            print(f"\n{'─'*50}")
            print(f"  [env {i}] SYSTEM PROMPT  (reward={rewards[i]}, done={dones[i]})")
            print(f"{'─'*50}")
            print(obs_dict["system"][i])
            print(f"\n{'─'*50}")
            print(f"  [env {i}] USER PROMPT  (reward={rewards[i]}, done={dones[i]})")
            print(f"{'─'*50}")
            print(obs_dict["text"][i])

    mgr.close()


# ---------------------------------------------------------------------------
# LLM mode (vLLM)
# ---------------------------------------------------------------------------

def _run_one_episode(
    args, provider, mgr, envs, board_id, board_path, ep,
    prompts_version, extract_think_action, is_no_progress,
) -> list[dict]:
    """Run a single episode (all envs) and return per-env result rows.

    Extracted from :func:`_run_rollout` so the caller can wrap the whole
    episode in try/except: a worker death (native router SIGSEGV) or any
    unexpected error is caught upstream, recorded as a crashed (NaN) result,
    and the run continues. On success this dumps ``{stem}.json`` +
    ``{stem}.kicad_pcb`` per env (when ``--dump_dir`` is set), which the
    resume path keys off of. A crash raises before the dump, so the episode
    stays un-dumped and a later resume re-attempts it.
    """
    import json

    ep_results: list[dict] = []
    obs_dict, info_list = mgr.reset()
    text_obs_list = list(obs_dict["anchor"])
    effective_max_steps = mgr.effective_max_steps
    if effective_max_steps != args.max_steps:
        print(
            f"  [adaptive] max_steps {args.max_steps} -> {effective_max_steps}"
        )

    dones = [False] * args.env_num
    total_rewards = [0.0] * args.env_num
    # Early-stop bookkeeping. ``no_progress_streak`` counts consecutive
    # steps that did NOT alter board state (parse-error, mask-rejected,
    # or empty_action). On threshold the env is marked done and its
    # final obs/info are frozen so later batch steps don't overwrite
    # the reported metrics.
    no_progress_streak = [0] * args.env_num
    frozen_obs: list[str | None] = [None] * args.env_num
    frozen_info: list[dict | None] = [None] * args.env_num
    # PCBWorld only stamps board-state measures (track_count,
    # via_count, wirelength, ratsnest_reduction) on VALID steps — at episode
    # end (terminated / truncated / early-stop on invalid step) the
    # final info_list[i] may not carry them. Track the most recent
    # value seen per env so the per-episode summary reflects the real
    # board state, not a "0/NaN because the last step was invalid"
    # artifact. Mirrors how the post-hoc scorer reads from the
    # saved board (we infer from streamed infos instead).
    last_known: list[dict] = [
        {
            "track_count": 0, "via_count": 0,
            "wirelength": 0.0, "ratsnest_reduction": float("nan"),
        }
        for _ in range(args.env_num)
    ]
    ep_system_tokens = 0
    ep_user_tokens = 0
    ep_response_tokens = 0
    ep_call_count = 0
    step = 0

    # Per-env rollout dumps — eval-exp.sh's visualiser step consumes
    # board_path + initial_obs_dict + per-step obs_after_dict.
    rollout_dumps: list | None = None
    if args.dump_dir:
        initial_obs_dicts = envs.get_last_obs_dicts()
        rollout_dumps = [
            {
                "version": prompts_version,
                "board_id": board_id,
                "board_path": board_path,
                "episode": ep,
                "env": i,
                "state_format": args.state_format,
                "initial_obs": text_obs_list[i],
                "initial_obs_dict": initial_obs_dicts[i],
                "steps": [],
            }
            for i in range(args.env_num)
        ]

    while not all(dones) and step < effective_max_steps:
        # Manager has already assembled prompts in obs_dict.
        prompt_pairs = list(zip(obs_dict["system"], obs_dict["text"]))
        if args.verbose or step == 0:
            _print_step_prompts(
                step, args.env_num, obs_dict["system"], obs_dict["text"],
            )

        # Provider call (vLLM batched / API per-prompt).
        llm_outputs, token_counts = provider.generate(prompt_pairs)

        # mgr.step does projection + envs.step + memory/streak update,
        # then returns the next obs_dict (system/user prompts already
        # rendered for the next round). ``_last_actions`` /
        # ``_last_valids`` are surfaced for our dump + status line.
        obs_dict, rewards, step_dones, info_list = mgr.step(llm_outputs)
        text_obs_list = list(obs_dict["anchor"])
        actions = mgr._last_actions
        valids = mgr._last_valids

        for i in range(args.env_num):
            sys_tok, usr_tok, out_tok = token_counts[i]
            ep_system_tokens += sys_tok
            ep_user_tokens += usr_tok
            ep_response_tokens += out_tok
            ep_call_count += 1
            status = "VALID" if valids[i] else "FALLBACK"
            print(
                f"\n  [step {step}][env {i}] {status}  "
                f"(system={sys_tok}, user={usr_tok}, "
                f"response={out_tok} tokens)"
            )
            print(llm_outputs[i])

        if rollout_dumps is not None:
            obs_after_dicts = envs.get_last_obs_dicts()
            for i in range(args.env_num):
                act = actions[i]
                rollout_dumps[i]["steps"].append({
                    "step": step,
                    "action_text": llm_outputs[i],
                    "action_parsed": extract_think_action(llm_outputs[i]),
                    "action_dict": act if isinstance(act, dict) else str(act),
                    "valid": bool(valids[i]),
                    "action_success": bool(info_list[i].get("action_success", False)),
                    "reward": float(rewards[i]),
                    "obs_after": text_obs_list[i],
                    "obs_after_dict": obs_after_dicts[i],
                    "done": bool(step_dones[i]),
                })

        # Snapshot board-state measures from valid steps. PCBWorld
        # only stamps these on ``valid_*`` steps, so we cache the
        # latest seen value to use as the per-episode summary fallback
        # when the final step is invalid / truncated.
        for i in range(args.env_num):
            info = info_list[i]
            for k in ("track_count", "via_count", "wirelength", "ratsnest_reduction"):
                if k in info:
                    last_known[i][k] = info[k]

        # Non-progress streak → optional early stop. A step counts as
        # non-progress when the board didn't change: invalid (parse
        # error / mask reject) OR accepted-but-empty_action.
        for i in range(args.env_num):
            if dones[i]:
                continue
            progressed = (
                valids[i]
                and info_list[i].get("action_success", False)
                and not is_no_progress(info_list[i])
            )
            no_progress_streak[i] = (
                0 if progressed else no_progress_streak[i] + 1
            )
            if (args.early_stop_no_progress > 0
                    and no_progress_streak[i] >= args.early_stop_no_progress
                    and not step_dones[i]):
                print(
                    f"  [early-stop][env {i}] non-progress streak "
                    f"{no_progress_streak[i]} >= "
                    f"{args.early_stop_no_progress} → marking done"
                )
                # Score at the stop decision (before the freeze copy) so the
                # canonical metrics capture the board state being frozen and
                # ``eval_metrics`` lands inside the frozen info snapshot.
                mgr.mark_episode_done([i], infos=info_list)
                frozen_obs[i] = text_obs_list[i]
                frozen_info[i] = dict(info_list[i])
                dones[i] = True

        for i in range(args.env_num):
            total_rewards[i] += rewards[i]
            if step_dones[i]:
                dones[i] = True

        step += 1

    # Defensive: score any env the loop left unfinished (no env-side
    # done, no early stop) so every episode leaves with canonical metrics.
    mgr.finalize_episode(infos=info_list)

    # Restore frozen snapshots for envs early-stopped on no-progress.
    for i in range(args.env_num):
        if frozen_info[i] is not None:
            text_obs_list[i] = frozen_obs[i]
            info_list[i] = frozen_info[i]

    # Final observation — render with NO_HIS template (init=True) so
    # the snapshot matches the "model's view of the final state" that
    # the original inline ``_build_prompt(...)`` produced.
    _, final_user = mgr.build_text_obs(
        text_obs_list,
        action_masks=_info_action_masks(info_list, args.env_num),
        init=True,
    )
    for i in range(args.env_num):
        print(f"\n{'─'*50}")
        print(f"  [env {i}] FINAL OBSERVATION (after step {step})")
        print(f"{'─'*50}")
        print(final_user[i])

    # Episode summary. ``final_potential`` is stamped by PCBWorld
    # only on the terminal/truncated step (see pcb_world/core/env.py),
    # so it is NaN for early-stopped runs and skipped by ``_mean_std``.
    # Board-state measures (track_count / via_count / wirelength /
    # ratsnest_reduction) likewise only appear on valid steps; if the final
    # step is invalid we fall back to the most recent valid-step
    # value tracked in ``last_known``.
    for i in range(args.env_num):
        won = info_list[i].get("won", False)
        drv = info_list[i].get("drc_violations", 0)
        wl = info_list[i].get("wirelength", last_known[i]["wirelength"])
        via = info_list[i].get("via_count", last_known[i]["via_count"])
        tracks = info_list[i].get("track_count", last_known[i]["track_count"])
        ratsnest_reduction = info_list[i].get(
            "ratsnest_reduction", last_known[i]["ratsnest_reduction"],
        )
        final_phi = info_list[i].get("final_potential", float("nan"))
        early_stopped = frozen_info[i] is not None
        result = {
            "board_id": board_id,
            "env": i, "steps": step, "reward": total_rewards[i],
            "won": won, "drc_violations": drv, "wirelength": wl,
            "via_count": via,
            "track_count": tracks,
            "ratsnest_reduction": ratsnest_reduction,
            "final_potential": final_phi,
            "system_tokens": ep_system_tokens, "user_tokens": ep_user_tokens,
            "response_tokens": ep_response_tokens, "call_count": ep_call_count,
            "early_stopped": early_stopped,
            # Nested canonical scoring (compute_metrics) from the manager's
            # episode-end hook; None if scoring failed for this env.
            "eval_metrics": mgr.episode_scored[i],
        }
        ep_results.append(result)
        tag = "  [early-stop]" if early_stopped else ""
        phi_str = (
            f"  Φ={final_phi:+.3f}" if _is_finite(final_phi) else ""
        )
        print(
            f"  [{board_id}][env {i}] steps={step}  "
            f"reward={total_rewards[i]:.3f}  success={won}  "
            f"ratsnest_reduction={ratsnest_reduction:.3f}  "
            f"DRV={drv}  WL={wl:.2f}mm  vias={via}  "
            f"tracks={tracks}{phi_str}{tag}"
        )

    # Episode-end: write rollout dumps + final board snapshots. This is the
    # last thing the episode does — if anything above crashed, no dump exists
    # and the resume path will re-run this (board, episode).
    if rollout_dumps is not None:
        pcb_paths: list[str] = []
        for i in range(args.env_num):
            result_i = next(
                (r for r in ep_results if r["env"] == i),
                {"steps": step, "reward": total_rewards[i],
                 "won": info_list[i].get("won", False),
                 "drc_violations": info_list[i].get("drc_violations", 0),
                 "wirelength": info_list[i].get("wirelength", 0.0)},
            )
            # Mirror the DRC-scorer headline measures (those
            # we can compute without a separate ``evaluate_one`` pass).
            # NaN ``final_potential`` is preserved as None in JSON for
            # readability — readers can detect missing-final via this.
            # Token counts are persisted too so a resume reconstructs the
            # summary row faithfully (see _load_episode_results).
            final_phi = result_i.get("final_potential", float("nan"))
            rollout_dumps[i]["result"] = {
                "steps": result_i["steps"],
                "reward": result_i["reward"],
                "won": result_i["won"],
                "drc_violations": result_i["drc_violations"],
                "wirelength": result_i["wirelength"],
                "via_count": result_i.get("via_count", 0),
                "track_count": result_i.get("track_count", 0),
                "ratsnest_reduction": result_i.get("ratsnest_reduction"),
                "final_potential": (
                    final_phi if _is_finite(final_phi) else None
                ),
                "early_stopped": result_i.get("early_stopped", False),
                "system_tokens": result_i.get("system_tokens", 0),
                "user_tokens": result_i.get("user_tokens", 0),
                "response_tokens": result_i.get("response_tokens", 0),
                "call_count": result_i.get("call_count", 0),
                # Canonical scoring block — lets a resume rebuild the same
                # canonical row without re-scoring, and downstream tools read
                # KiCad-measured metrics straight from the dump.
                "eval_metrics": result_i.get("eval_metrics"),
            }
            stem = f"{board_id}_episode_{ep:02d}_env_{i:02d}"
            fpath = os.path.join(args.dump_dir, f"{stem}.json")
            with open(fpath, "w") as f:
                json.dump(rollout_dumps[i], f, ensure_ascii=False, indent=2)
            print(f"  [env {i}] dump -> {fpath}")
            pcb_paths.append(
                os.path.join(args.dump_dir, f"{stem}.kicad_pcb")
            )

        # Batch-save all workers in one Ray round-trip.
        for pcb in envs.save_boards(pcb_paths):
            print(f"  pcb     -> {pcb}  (+ {os.path.splitext(pcb)[0]}.kicad_pro)")

    return ep_results


def _run_rollout(args, provider) -> None:
    """Shared LLM-driven rollout driver for ``--mode llm`` / ``--mode api``.

    ``provider`` must satisfy the :class:`methods._shared.agent.Agent` protocol —
    ``generate`` + ``close`` (e.g. :class:`methods.llm_agent.policy.agent.KiCadLLMAgent`,
    or the RL serving agent). The rest of the per-step flow (prompt assembly, memory +
    rejection streak bookkeeping, projection) is delegated to
    :class:`KiCadLLMRolloutManager` via the ``_EvalRolloutManager`` subclass
    so eval and verl-agent training share one source of truth.

    Bookkeeping kept here (not in the manager): provider token-count
    aggregation, per-step verbose printing, ``--dump_dir`` JSON + PCB
    snapshots, ``--early_stop_no_progress`` frozen-state restoration, and
    per-board / per-episode summary printing + CSV.
    """
    import json
    from methods.llm_agent.wrappers.action_converter import extract_think_action
    from methods.llm_agent.wrappers.feedback import is_no_progress

    eval_boards = _resolve_eval_boards(args)
    envs = _build_envs(args, board_path=eval_boards[0][1])
    mgr = _make_manager(envs, args)
    prompts_version = mgr._prompts.version

    print(
        f"  Workers: {envs.num_processes}  |  "
        f"Episodes/board: {args.rollout_episodes}  |  "
        f"Boards: {len(eval_boards)}"
    )
    print(f"  Prompt version : {prompts_version}")

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        print(f"  Dump dir: {args.dump_dir}  (prompt version tag: {prompts_version})")

    from methods._shared.board_loader import BoardSpec

    canonical_rows: list[dict] = []
    boards: list[BoardSpec] = []
    for board_idx, board_id, _board_path in _iter_boards_with_reload(
        envs, eval_boards, args.env_num
    ):
        if len(eval_boards) > 1:
            _print_section(f"Board {board_idx + 1}/{len(eval_boards)}: {board_id}")

        board_results: list[dict] = []

        for ep in range(args.rollout_episodes):
            # Resume: skip episodes already fully dumped (both .json and
            # .kicad_pcb present for every env). Reconstruct their result rows
            # so the summary CSV stays complete across resumes.
            if args.dump_dir:
                cached = _load_episode_results(
                    args.dump_dir, board_id, ep, args.env_num
                )
                if cached is not None:
                    print(
                        f"  [resume] {board_id} episode {ep + 1}/"
                        f"{args.rollout_episodes} already complete — skipping"
                    )
                    board_results.extend(cached)
                    continue

            _print_section(f"Episode {ep + 1}/{args.rollout_episodes}")

            try:
                ep_results = _run_one_episode(
                    args, provider, mgr, envs, board_id, _board_path, ep,
                    prompts_version, extract_think_action, is_no_progress,
                )
                board_results.extend(ep_results)
            except Exception as e:  # noqa: BLE001
                # Worker death (native router SIGSEGV / OOM) or any unexpected
                # error: record this episode as crashed (NaN), rebuild the dead
                # worker(s) against the current board, and continue. The episode
                # was not dumped, so a later resume re-attempts it.
                import traceback
                print(
                    f"  [crash] {board_id} episode {ep + 1}/"
                    f"{args.rollout_episodes}: {type(e).__name__}: {e}"
                )
                traceback.print_exc()
                board_results.extend(
                    _crashed_result(board_id, i) for i in range(args.env_num)
                )
                try:
                    envs.rebuild_workers()
                except Exception as rebuild_err:  # noqa: BLE001
                    print(
                        f"  [crash] worker rebuild failed: "
                        f"{type(rebuild_err).__name__}: {rebuild_err}"
                    )
                continue

        _print_board_summary(board_id, board_results, args)
        boards.append(BoardSpec(board_idx, board_id, _board_path or ""))
        for rollout_idx, result in enumerate(board_results):
            canonical_rows.append(_result_to_canonical_row(
                result, board_index=board_idx,
                board_path=_board_path or "", rollout_idx=rollout_idx,
            ))

    # Aggregate + export through the single shared evaluator handler so the LLM
    # eval emits the same canonical artifacts (per_rollout.csv / per_board.csv /
    # eval_overall_summary.json) as the RL path, and Stage-2/3 can consume them.
    if args.dump_dir and canonical_rows:
        from eval.evaluator import export_csv
        from eval.metrics import EvalSummary

        metrics = EvalSummary.from_rollouts(canonical_rows, boards)
        export_csv(metrics, args.dump_dir)
        print(
            f"\n  eval artifacts -> {args.dump_dir}/"
            "{per_rollout.csv, per_board.csv, eval_overall_summary.json}"
        )

    mgr.close()


def run_llm(args) -> None:
    from methods.llm_agent.policy.agent import KiCadLLMAgent
    agent = KiCadLLMAgent.from_args(args, mode="llm")
    try:
        _run_rollout(args, agent)
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# API mode (OpenAI / Anthropic / Google)
# ---------------------------------------------------------------------------


def run_api(args) -> None:
    from methods.llm_agent.policy.agent import KiCadLLMAgent
    agent = KiCadLLMAgent.from_args(args, mode="api")
    _print_section(f"API mode: {args.api_provider} / {agent.backend.model}")
    try:
        _run_rollout(args, agent)
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    _print_section("cadagent environment eval")
    print(f"  mode              : {args.mode}")
    print(f"  board_path        : {args.board_path}")
    print(f"  env_num           : {args.env_num}")
    print(f"  max_steps         : {args.max_steps}")
    print(f"  masking_rule      : {args.masking_rule}")
    print(f"  state_format      : {args.state_format}")
    if args.mode == "llm":
        print(f"  model_path        : {args.model_path}")
        print(f"  rollout_episodes  : {args.rollout_episodes}")
        print(f"  temperature       : {args.temperature}")
        print(f"  max_new_tokens    : {args.max_new_tokens}")
        print(f"  tensor_parallel   : {args.tensor_parallel_size}")
        print(f"  gpu_mem_util      : {args.gpu_memory_utilization}")
        print(f"  max_model_len     : {args.max_model_len}")
        print(f"  prefix_caching    : {args.enable_prefix_caching}")
        print(f"  chunked_prefill   : {args.enable_chunked_prefill}")
    elif args.mode == "api":
        # APIProvider falls back to the per-provider default when api_model
        # is unset; show the user what they actually requested (the resolved
        # name is also re-printed by run_api's "API mode: ..." header).
        print(f"  api_provider      : {args.api_provider}")
        print(f"  api_model         : {args.api_model or '(provider default)'}")
        print(f"  rollout_episodes  : {args.rollout_episodes}")
        print(f"  temperature       : {args.temperature}")
        print(f"  max_new_tokens    : {args.max_new_tokens}")

    if not os.path.isfile(args.board_path):
        print(f"\n[ERROR] Board file not found: {args.board_path}")
        sys.exit(1)

    if args.mode == "fixed":
        run_fixed(args)
    elif args.mode == "llm":
        run_llm(args)
    elif args.mode == "api":
        run_api(args)

    _print_section("DONE")


if __name__ == "__main__":
    main()
