"""Generate API-Seq few-shot examples by querying a strong LLM.

Sister of ``prepare_plan_only_fewshot.py`` — same output schema, same
downstream consumer (``eval_apiseq_llm.load_fewshot_pool``) — but the
action sequence comes from a real LLM call instead of the deterministic
auto-router. The motivation: synthetic auto-router examples teach the
LLM to copy a *robotic* sequence (start_route → finish for every 2-pad
net), while LLM-generated examples retain pad-ordering decisions, mode
choices, and via placement that look like real solutions.

Pipeline per board:
    1. Render the initial board state (board_static sexpr).
    2. Build the v3 system+user prompt (no plan injection — we want the
       LLM's natural decision, not a transcription).
    3. Sample N completions from the strong model (default gpt-5.4 with
       low temperature for determinism).
    4. For each completion, replay through PCBWorld and score with
       ``eval.metrics.evaluate_one`` (same metric as the eval).
    5. Keep the BEST sample per board (by clean_pass > success >
       routability), filtered by --routability-min.
    6. Across all boards, take the top --top-k overall and write each
       as a JSON cache entry plus its rolled-out .kicad_pcb sidecar.

Usage:
    # Default: 8 synth_2L val boards, gpt-5.4, keep top 4
    python experiments/kdd/llm_eval/prepare_plan_only_fewshot_llm.py \\
        --src-dir $CADAGENT_DATA_ROOT/synthetic/synth_2L_v2/val \\
        -o cache/synth_2L_apiseq_fewshot_llm \\
        --limit 8 --top-k 4 --api-model gpt-5.4

    # Mixed: throw in a couple of easy real boards too
    python experiments/kdd/llm_eval/prepare_plan_only_fewshot_llm.py \\
        --src-files $CADAGENT_DATA_ROOT/.../easy_real_board.kicad_pcb \\
        --src-dir $CADAGENT_DATA_ROOT/.../synth_2L_v2/val \\
        -o cache/apiseq_fewshot_mixed --limit 6 --top-k 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # llm_eval→paper_repro→scripts→repo
_KICAD_RL_DIR = _PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"
for p in (_PROJECT_ROOT, _KICAD_RL_DIR, _PROJECT_ROOT / "scripts"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Board discovery
# ---------------------------------------------------------------------------

def _gather_boards(
    src_dir: Path | None,
    src_files: list[Path],
    limit: int,
) -> list[Path]:
    """Return up to ``limit`` .kicad_pcb files. ``src_files`` come first
    (in the order given), then alphabetical files from ``src_dir``."""
    out: list[Path] = []
    for f in src_files:
        if f.is_file() and f.suffix == ".kicad_pcb":
            out.append(f.resolve())
    if src_dir and src_dir.is_dir():
        for f in sorted(src_dir.glob("*.kicad_pcb")):
            if len(out) >= limit:
                break
            out.append(f.resolve())
    return out[:limit] if limit > 0 else out


# ---------------------------------------------------------------------------
# Single-board: prompt → LLM → replay → score
# ---------------------------------------------------------------------------

def _board_static_for(board_path: Path, state_format: str = "sexpr") -> str:
    """Reset the env on ``board_path`` and return board_static sexpr."""
    from pcb_world.core.env import PCBWorld
    from pcb_world.engine.drc_config import DEFAULT_DRC_CONFIG_PATH
    from methods.llm_agent.wrappers.state_converter import format_state_split_sexpr, format_state_split

    env = PCBWorld(
        board_path=str(board_path), state_format=state_format,
        max_steps=10_000, use_yaml_drc_fallback=True,
        drc_config_path=str(DEFAULT_DRC_CONFIG_PATH),
    )
    try:
        obs, _ = env.reset()
        if state_format == "sexpr":
            static, _ = format_state_split_sexpr(obs)
        else:
            static, _ = format_state_split(obs)
        return static
    finally:
        try:
            env.close()
        except Exception:
            pass


def _generate_action_sequences(
    system_prompt: str,
    user_prompt: str,
    api_model: str,
    api_key: str | None,
    temperature: float,
    n_samples: int,
    max_tokens: int,
) -> tuple[list[str], dict]:
    """Single OpenAI call returning ``n_samples`` completions."""
    from openai import OpenAI
    client = OpenAI(**({"api_key": api_key} if api_key else {}))

    is_reasoning = any(api_model.startswith(p) for p in ("o1", "o3", "o4", "gpt-5"))
    if is_reasoning:
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {"max_completion_tokens": max_tokens}
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {"max_tokens": max_tokens, "temperature": temperature}

    resp = client.chat.completions.create(
        model=api_model, messages=messages, n=n_samples, **kwargs,
    )
    texts = [c.message.content or "" for c in resp.choices]
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
        "total_tokens": getattr(resp.usage, "total_tokens", 0),
    }
    return texts, usage


def _replay_and_score(
    board_path: Path, response: str, save_pcb_path: Path,
    reward_config: str = "drc_dense_promoted", check_angle: int = 45,
) -> dict:
    """Parse the LLM response, replay through the env, then run the
    same eval.metrics evaluator the actual benchmark uses so the
    quality filter matches the eval-time metric exactly."""
    from eval_plan_only_llm_v8_standalone import (   # noqa: E402  (deferred for sys.path)
        extract_action_sequence, replay_actions_and_eval,
    )
    from eval.metrics import evaluate_one as _eval_routed

    actions = extract_action_sequence(response)
    if not actions:
        return {
            "actions_text": "",
            "n_actions": 0,
            "success": False, "routability": 0.0, "drv_count": 9999,
            "clean_pass": False, "final_potential": -9999.0,
            "error": "no_actions_parsed",
        }

    # 1. Replay through the env so the engine writes a routed PCB to disk.
    save_pcb_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        replay_metrics = replay_actions_and_eval(board_path, actions, save_pcb_path)
    except Exception as exc:
        return {
            "actions_text": "\n".join(_render_action(a) for a in actions),
            "n_actions": len(actions),
            "success": False, "routability": 0.0, "drv_count": 9999,
            "clean_pass": False, "final_potential": -9999.0,
            "error": f"replay_failed: {type(exc).__name__}: {exc}",
        }

    # 2. Score the saved board with the canonical evaluator
    #    (eval.metrics.evaluate_one, via experiments/_lib/metrics).
    pro = board_path.with_suffix(".kicad_pro")
    pro_arg = str(pro) if pro.exists() else None
    try:
        m = _eval_routed(
            str(save_pcb_path), pro_arg,
            reward_config_name=reward_config, check_angle=check_angle,
        )
    except Exception as exc:
        m = {}
        return {
            "actions_text": "\n".join(_render_action(a) for a in actions),
            "n_actions": len(actions),
            "success": replay_metrics.get("success", False),
            "routability": replay_metrics.get("routability", 0.0),
            "drv_count": 9999, "clean_pass": False, "final_potential": -9999.0,
            "error": f"eval_failed: {type(exc).__name__}: {exc}",
        }

    return {
        "actions_text": "\n".join(_render_action(a) for a in actions),
        "n_actions": len(actions),
        "success": bool(m.get("success", False)),
        "routability": float(m.get("routability", 0.0)),
        "drv_count": int(m.get("drv_errors_and_promoted_count", 0)),
        "clean_pass": bool(m.get("clean_pass", False)),
        "final_potential": float(m.get("final_potential", 0.0)),
        "track_count": int(m.get("track_count", 0)),
        "via_count": int(m.get("via_count", 0)),
        "wirelength_mm": float(m.get("wirelength_mm", 0.0)),
        "track_angle_drv_count": int(m.get("track_angle_drv", {}).get("count", 0)),
        "error": "",
    }


from pcb_world.core.action_schema import (
    ACTION_NAMES as _ALL_ACTION_NAMES,
    ACT_IDLE as _ACT_IDLE,
    MODE_INT_TO_LETTER as _MODE_INT_TO_LETTER,
)
# Selectable actions only (idle excluded); derived from canonical action_schema.
_ACTION_NAMES = _ALL_ACTION_NAMES[:_ACT_IDLE]


def _render_action(a: dict) -> str:
    """Convert a parsed action dict back to its single-line text form
    (same shape the LLM should have emitted). Used for the cached
    ``action_sequence`` field so few-shot consumers see the canonical
    text — not the engine-internal dict."""
    name = _ACTION_NAMES[a["action_type"]]
    if name == "net_select":
        return f"net_select {a.get('net_id', 0)}"
    if name == "start_route":
        return f"start_route {a.get('x_mm', 0):.3f} {a.get('y_mm', 0):.3f} {a.get('layer', 1)}"
    if name == "net_end":
        return "net_end"
    if name == "finish":
        m = _MODE_INT_TO_LETTER.get(a.get("routing_mode", 2), "w")
        return f"finish {m}"
    # make_line / make_via
    m = _MODE_INT_TO_LETTER.get(a.get("routing_mode", 2), "w")
    return f"{name} {a.get('x_mm', 0):.3f} {a.get('y_mm', 0):.3f} {m}"


# ---------------------------------------------------------------------------
# Per-board orchestration
# ---------------------------------------------------------------------------

def _quality_key(s: dict) -> tuple:
    """Sort key for picking the best sample. Higher is better."""
    return (
        int(s.get("clean_pass", False)),
        int(s.get("success", False)),
        float(s.get("routability", 0.0)),
        -float(s.get("drv_count", 9999)),
        float(s.get("final_potential", -9999.0)),
    )


def _process_board(
    board_path: Path,
    out_dir: Path,
    args,
    system_prompt: str,
) -> dict | None:
    """Sample `n_samples` LLM completions, score each, return the best.

    The "best" is the highest-quality completion under ``_quality_key``;
    we save its action_sequence (alongside the rolled-out .kicad_pcb)
    as the candidate few-shot entry for this board. If every completion
    failed the routability filter, returns None.
    """
    from eval_plan_only_llm_v8_standalone import build_user_prompt, FewShotExample
    bid = board_path.stem
    board_dir = out_dir / "_attempts" / bid
    board_dir.mkdir(parents=True, exist_ok=True)

    try:
        board_static = _board_static_for(board_path)
    except Exception as exc:
        traceback.print_exc()
        return None

    user_prompt = build_user_prompt(board_static, examples=[])
    (board_dir / "prompt.txt").write_text(
        "==== SYSTEM ====\n" + system_prompt + "\n==== USER ====\n" + user_prompt
    )

    try:
        responses, usage = _generate_action_sequences(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_model=args.api_model,
            api_key=args.api_key,
            temperature=args.temperature,
            n_samples=args.n_samples,
            max_tokens=args.max_new_tokens,
        )
    except Exception as exc:
        traceback.print_exc()
        return None

    samples: list[dict] = []
    for i, response in enumerate(responses):
        (board_dir / f"sample_{i:02d}.response.txt").write_text(response)
        save_pcb = board_dir / f"sample_{i:02d}.kicad_pcb"
        m = _replay_and_score(
            board_path, response, save_pcb,
            reward_config=args.reward_config, check_angle=args.check_angle,
        )
        m["sample_idx"] = i
        m["pcb_path"] = str(save_pcb) if save_pcb.exists() else ""
        samples.append(m)
        with (board_dir / f"sample_{i:02d}.json").open("w") as f:
            json.dump(m, f, indent=2)

    samples.sort(key=_quality_key, reverse=True)
    best = samples[0] if samples else None
    if not best:
        return None

    if best.get("routability", 0.0) < args.routability_min:
        print(f"  [drop] {bid}: best routability "
              f"{best.get('routability', 0):.3f} < {args.routability_min}")
        return None
    if args.success_only and not best.get("success", False):
        print(f"  [drop] {bid}: best is not success")
        return None

    return {
        "board_id": bid,
        "source_pcb": str(board_path),
        "board_static": board_static,
        "action_sequence": best["actions_text"],
        "best_pcb_path": best["pcb_path"],
        "stats": {k: best.get(k) for k in (
            "success", "clean_pass", "routability", "drv_count",
            "track_angle_drv_count", "final_potential",
            "track_count", "via_count", "wirelength_mm", "n_actions",
        )},
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", type=Path, default=None,
                   help="Directory of unrouted .kicad_pcb (e.g. synth_2L_v2/val).")
    p.add_argument("--src-files", type=Path, nargs="*", default=[],
                   help="Explicit .kicad_pcb files (e.g. easy real boards).")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="Cache dir for example JSONs + rolled-out PCBs.")
    p.add_argument("--limit", type=int, default=8,
                   help="Total board attempts (src_files first, then src-dir).")
    p.add_argument("--top-k", type=int, default=4,
                   help="Across all attempted boards, keep the top-K best as "
                        "few-shot examples.")
    p.add_argument("--n-samples", type=int, default=3,
                   help="LLM completions per board (best is kept).")
    p.add_argument("--api-model", default="gpt-5.4")
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Lower than benchmark default for stable example "
                        "extraction.")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--routability-min", type=float, default=0.8,
                   help="Drop boards whose best sample's routability is "
                        "below this threshold.")
    p.add_argument("--success-only", action="store_true",
                   help="Only keep boards whose best sample is fully routed "
                        "(success=True).")
    p.add_argument("--reward-config", default="drc_dense_promoted")
    p.add_argument("--check-angle", type=int, choices=(45, 90), default=45)
    p.add_argument("--prompt-version", choices=("v3",), default="v3",
                   help="Which prompt to use for generation. v3 (default) "
                        "is preferred — v4 injects the routing plan and "
                        "would skip the LLM's natural decision-making.")
    p.add_argument("-f", "--force", action="store_true")
    return p.parse_args()


def _system_prompt(version: str) -> str:
    if version == "v3":
        from eval_plan_only_llm_v8_standalone import _SYSTEM_PROMPT_V8 as _SYSTEM_PROMPT_V3   # type: ignore
        return _SYSTEM_PROMPT_V3
    raise ValueError(f"unknown prompt version: {version}")


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    boards = _gather_boards(args.src_dir, args.src_files, args.limit)
    if not boards:
        print("[ERROR] no .kicad_pcb files found", file=sys.stderr)
        return 2

    system_prompt = _system_prompt(args.prompt_version)

    print(f"  output      : {args.output}")
    print(f"  boards      : {len(boards)}")
    print(f"  api_model   : {args.api_model}")
    print(f"  n_samples   : {args.n_samples}  temperature: {args.temperature}")
    print(f"  filters     : routability >= {args.routability_min}, "
          f"success_only={args.success_only}")
    print(f"  prompt      : {args.prompt_version}")
    print()

    candidates: list[dict] = []
    for b in boards:
        print(f"=== {b.name} ===")
        try:
            cand = _process_board(b, args.output, args, system_prompt)
        except Exception as exc:
            traceback.print_exc()
            cand = None
        if cand is None:
            continue
        s = cand["stats"]
        print(f"  best: success={s['success']}  clean={s['clean_pass']}  "
              f"rout={s['routability']:.3f}  drv={s['drv_count']}  "
              f"actions={s['n_actions']}")
        candidates.append(cand)

    candidates.sort(key=lambda c: _quality_key(c["stats"]), reverse=True)
    kept = candidates[:args.top_k] if args.top_k > 0 else candidates

    print()
    print(f"=== keeping {len(kept)} of {len(candidates)} as few-shot ===")
    for k in kept:
        json_path = args.output / f"{k['board_id']}.json"
        pcb_path = args.output / f"{k['board_id']}.kicad_pcb"
        # Mirror the routed board into the cache dir for inspection.
        if k.get("best_pcb_path"):
            try:
                shutil.copyfile(k["best_pcb_path"], pcb_path)
                src_pro = Path(k["source_pcb"]).with_suffix(".kicad_pro")
                if src_pro.exists():
                    shutil.copyfile(src_pro, pcb_path.with_suffix(".kicad_pro"))
            except Exception:
                pass
        json_path.write_text(json.dumps({
            "board_id": k["board_id"],
            "source_pcb": k["source_pcb"],
            "state_format": "sexpr",
            "board_static": k["board_static"],
            "action_sequence": k["action_sequence"],
            "stats": k["stats"],
            "generator": {"model": args.api_model, "prompt": args.prompt_version},
        }, indent=2))
        s = k["stats"]
        print(f"  -> {json_path.name}  rout={s['routability']:.3f} "
              f"drv={s['drv_count']} actions={s['n_actions']}")

    if not kept:
        print("[ERROR] no candidate met the quality filter — "
              "try lowering --routability-min or sampling more (--n-samples).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
