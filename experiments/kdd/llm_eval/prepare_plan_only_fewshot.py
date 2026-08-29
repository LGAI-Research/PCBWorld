"""Pre-compute API-Seq few-shot examples by replaying a deterministic
auto-router on synth_2L val boards.

Sister script to ``prepare_synth_fewshot.py`` (which produces routed
*boards* for CAD-Gen few-shot). For API-Seq we instead need the
*action sequence* — the chain of CAD API calls that would
have produced a routing — paired with the *initial board state text*.

The auto-router walks each net pad-by-pad and emits the equivalent of
PCBWorld.step({...}) calls:

    net_select <net_id>
    start_route <pad0.x> <pad0.y> <pad0.layer>
    make_line <pad1.x> <pad1.y> w        # if same layer as head
    make_via  <pad1.x> <pad1.y> w        # if different layer
    ...
    finish w
    net_end

Output cache layout (used by ``eval_apiseq_llm.py --fewshot-pool``):

    <out_dir>/<board_stem>.json       {board_id, board_static, action_sequence,
                                       routed_pcb (optional), stats}
    <out_dir>/<board_stem>.kicad_pcb  routed copy (so you can sanity-check)
    <out_dir>/<board_stem>.kicad_pro  paired pro

Usage:
    python experiments/kdd/llm_eval/prepare_plan_only_fewshot.py \
        $CADAGENT_DATA_ROOT/synthetic/synth_2L_v2/val \
        -o cache/synth_2L_apiseq_fewshot/ \
        --limit 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent.parent.parent.parent  # llm_eval→paper_repro→scripts→repo
_KICAD_RL_DIR = _THIS_DIR / "build_rl" / "pcbnew" / "python" / "rl"
for p in (_THIS_DIR, _KICAD_RL_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Action sequence builder
# ---------------------------------------------------------------------------

def _layer_for_pad(engine, pad) -> int:
    """Human layer (1 = F.Cu, 2 = B.Cu) for a pad. THT pads default to 1."""
    h = engine._b2h(pad.layer)
    return h if h >= 1 else 1


def auto_route_actions(board_path: Path) -> str:
    """Build a complete action-sequence text by replaying a chain router
    on every net of the given board.

    The text format matches what ``eval_apiseq_llm.parse_action_line``
    expects: one action per line, no envelope. Caller wraps with
    ``<actions>...</actions>`` if needed.

    This is also used by ``eval_apiseq_llm.py --dry-run`` to produce a
    "self-known" stand-in for the LLM response.
    """
    from pcb_world.engine.kicad_engine import KiCadEngine

    engine = KiCadEngine(str(board_path))
    try:
        engine.build_connectivity()
        nets: dict[int, list] = defaultdict(list)
        for p in engine.get_pads():
            nets[p.net_code].append(p)

        lines: list[str] = []
        for nc, pads in sorted(nets.items()):
            if nc == 0 or len(pads) < 2:
                continue
            # Stable per-net pad order keeps the action sequence
            # deterministic across runs (the engine's get_pads() order
            # is itself deterministic, so this is mostly defensive).
            pads = sorted(pads, key=lambda p: (p.x_mm, p.y_mm, p.layer))

            lines.append(f"net_select {nc}")
            head_layer = _layer_for_pad(engine, pads[0])
            lines.append(
                f"start_route {pads[0].x_mm:.3f} {pads[0].y_mm:.3f} {head_layer}"
            )
            for tgt in pads[1:]:
                tgt_layer = _layer_for_pad(engine, tgt)
                if tgt_layer == head_layer:
                    lines.append(f"make_line {tgt.x_mm:.3f} {tgt.y_mm:.3f} w")
                else:
                    lines.append(f"make_via {tgt.x_mm:.3f} {tgt.y_mm:.3f} w")
                    head_layer = tgt_layer
            lines.append("finish w")
            lines.append("net_end")
        return "\n".join(lines)
    finally:
        engine = None  # release C++ singleton


# ---------------------------------------------------------------------------
# Replay (record stats + save routed copy)
# ---------------------------------------------------------------------------

def replay_and_save(board_path: Path, actions_text: str, out_pcb: Path) -> dict:
    """Run the action sequence through ``PCBWorld`` and save the
    rolled-out board, returning routing stats so the cache JSON can
    advertise example quality.
    """
    from pcb_world.core.env import PCBWorld
    from pcb_world.engine.drc_config import DEFAULT_DRC_CONFIG_PATH
    # Parse the same way ``eval_plan_only_llm_v8_standalone.py`` will when the
    # LLM emits this sequence — keeps the prep step's "ground-truth replay"
    # identical to what the evaluator actually runs at inference time.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_plan_only_llm_v8_standalone import parse_action_line  # type: ignore

    env = PCBWorld(
        board_path=str(board_path), max_steps=10_000,
        use_yaml_drc_fallback=True,
        drc_config_path=str(DEFAULT_DRC_CONFIG_PATH),
    )
    try:
        env.reset()
        u_0 = env._initial_unconnected
        for line in actions_text.splitlines():
            action = parse_action_line(line)
            if action is None:
                continue
            try:
                env.step(action)
            except Exception:
                pass
        snap = env._engine.get_reward_snapshot(run_drc=True)
        out_pcb.parent.mkdir(parents=True, exist_ok=True)
        env._engine.save(str(out_pcb))
        return {
            "initial_unrouted": int(u_0),
            "unrouted_remaining": int(snap.unrouted_count),
            "track_count": int(snap.track_count),
            "via_count": int(snap.via_count),
            "drc_violations": int(snap.drc_violation_count),
            "wirelength_mm": float(snap.total_wirelength),
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Initial state extraction  (mirrors eval_apiseq_llm.initial_board_state_text
# but standalone to avoid the eval script importing this module's heavyweight
# deps on every call).
# ---------------------------------------------------------------------------

def extract_initial_board_state(board_path: Path, state_format: str = "sexpr") -> str:
    from pcb_world.core.env import PCBWorld
    from pcb_world.engine.drc_config import DEFAULT_DRC_CONFIG_PATH
    from methods.llm_agent.wrappers.state_converter import (
        format_state_split_sexpr, format_state_split,
    )

    env = PCBWorld(
        board_path=str(board_path), state_format=state_format,
        max_steps=10_000, use_yaml_drc_fallback=True,
        drc_config_path=str(DEFAULT_DRC_CONFIG_PATH),
    )
    try:
        obs, _info = env.reset()
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src_dir", type=Path,
                   help="Directory of unrouted .kicad_pcb (e.g. synth_2L_v2/val).")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="Cache dir for example JSONs + rolled-out PCBs.")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--state-format", choices=["sexpr", "xml"], default="sexpr")
    p.add_argument("-f", "--force", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.src_dir.is_dir():
        print(f"[ERROR] not a directory: {args.src_dir}", file=sys.stderr)
        return 2
    boards = sorted(args.src_dir.glob("*.kicad_pcb"))[: args.limit]
    if not boards:
        print(f"[ERROR] no .kicad_pcb under {args.src_dir}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"  source : {args.src_dir}")
    print(f"  out    : {args.output}")
    print(f"  boards : {len(boards)}")
    print()

    n_ok = n_skip = n_fail = 0
    for src in boards:
        json_path = args.output / f"{src.stem}.json"
        pcb_path = args.output / f"{src.stem}.kicad_pcb"
        if json_path.exists() and not args.force:
            print(f"  [skip] {json_path.name} (use --force to overwrite)")
            n_skip += 1
            continue

        try:
            # 1. Capture the initial board state text the LLM will see.
            board_static = extract_initial_board_state(src, args.state_format)
            # 2. Build an action sequence by walking nets pad-by-pad.
            actions_text = auto_route_actions(src)
            # 3. Roll out the sequence to verify it actually routes
            #    something + capture stats. The .kicad_pcb sidecar lets a
            #    human eyeball the example quality.
            stats = replay_and_save(src, actions_text, pcb_path)
        except Exception as exc:
            traceback.print_exc()
            print(f"  [FAIL] {src.name}: {type(exc).__name__}: {exc}")
            n_fail += 1
            continue

        # Mirror the source .kicad_pro so the routed copy carries BDS
        # / NetSettings (engine.save also emits one, but sourcing the
        # original keeps consumers consistent with synth few-shot).
        src_pro = src.with_suffix(".kicad_pro")
        if src_pro.exists():
            shutil.copyfile(src_pro, pcb_path.with_suffix(".kicad_pro"))

        json_path.write_text(json.dumps({
            "board_id": src.stem,
            "source_pcb": str(src),
            "state_format": args.state_format,
            "board_static": board_static,
            "action_sequence": actions_text,
            "routed_pcb": str(pcb_path),
            "stats": stats,
        }, indent=2))

        n_actions = len(actions_text.splitlines())
        print(
            f"  [ok] {src.name}: actions={n_actions} "
            f"tracks={stats['track_count']} vias={stats['via_count']} "
            f"ratsnest_left={stats['unrouted_remaining']}"
        )
        if args.verbose:
            print(f"    static_chars={len(board_static)}  "
                  f"action_chars={len(actions_text)}")
        n_ok += 1

    print()
    print(f"OK: {n_ok}  Skipped: {n_skip}  Failed: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
