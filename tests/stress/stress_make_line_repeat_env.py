"""Stress test using the *actual* gym env (PCBWorld) instead of raw engine.

Mirrors scripts/_stress_make_line_repeat.py but exercises the env action
pipeline:
  reset -> net_select(NET1) -> start_route(10,10) -> make_line(110,10) x N

env.make_line uses force_finish=True (pcb_world/core/action.py), so each
call commits + ends the routing session. We log per-step:
  - action_mask (which actions the env says are legal next)
  - is_routing (engine flag)
  - dispatch success / action_class (valid_effective / valid_empty / ...)
  - reward
  - tracks committed so far (net1 only)
  - placer head
  - terminated / truncated

Board: same synthetic NET1 with pads at (10,10) and (110,50) on F.Cu
(reuses the fixture file written by _stress_make_line_repeat.py if present).

Run (from the repo root, with the C++ router built):
  conda activate cadagent
  python tests/stress/stress_make_line_repeat_env.py
"""

from __future__ import annotations

import math
import sys
import textwrap
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_MODULE_DIR = PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"
OUTPUT_DIR = PROJECT_ROOT / "var" / "tests" / "output" / "stress_make_line_repeat"

sys.path.insert(0, str(RL_MODULE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from pcb_world.core.env import PCBWorld  # noqa: E402
from pcb_world.engine.drc_config import DEFAULT_DRC_CONFIG_PATH  # noqa: E402
from pcb_world.core.masking import (  # noqa: E402
    ACT_IDLE,
    ACT_MAKE_LINE,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACT_NET_END,
    ACT_FINISH,
    ACTION_NAMES,
)


START_PIN = (10.0, 10.0)
FAR_PIN = (110.0, 50.0)
TARGET = (110.0, 10.0)
LAYER_HUMAN = 1  # 1 = F.Cu (env uses 1-indexed human layers)
NET_ID = 1       # NET1
N_ITER = 8


BOARD_TEMPLATE = textwrap.dedent("""\
(kicad_pcb
  (version 20241229)
  (generator "stress_make_line_repeat")
  (generator_version "9.0.5")
  (general (thickness 1.6) (legacy_teardrops no))
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive") (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user) (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen") (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user "B.Mask") (39 "F.Mask" user "F.Mask")
    (40 "Dwgs.User" user "User.Drawings") (41 "Cmts.User" user "User.Comments")
    (44 "Edge.Cuts" user))
  (setup (pad_to_mask_clearance 0) (allow_soldermask_bridges_in_footprints no))
  (net 0 "") (net 1 "NET1")
  (net_class "Default" "Default net class"
    (clearance 0.2) (trace_width 0.2)
    (via_dia 0.6) (via_drill 0.3) (uvia_dia 0.3) (uvia_drill 0.1))
  (footprint "SamplePad:FCu" (layer "F.Cu") (at 10 10)
    (uuid "00000000-0000-0000-0000-000000000001")
    (property "Reference" "P1" (at 0 -1) (layer "F.SilkS")
      (effects (font (size 0.6 0.6) (thickness 0.1))))
    (property "Value" "Pad1" (at 0 1) (layer "F.Fab")
      (effects (font (size 0.6 0.6) (thickness 0.1))))
    (pad "1" smd roundrect (at 0 0) (size 1.0 1.0)
      (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25)
      (net 1 "NET1")
      (uuid "00000000-0000-0000-0000-00000000aa01")))
  (footprint "SamplePad:FCu" (layer "F.Cu") (at 110 50)
    (uuid "00000000-0000-0000-0000-000000000002")
    (property "Reference" "P2" (at 0 -1) (layer "F.SilkS")
      (effects (font (size 0.6 0.6) (thickness 0.1))))
    (property "Value" "Pad2" (at 0 1) (layer "F.Fab")
      (effects (font (size 0.6 0.6) (thickness 0.1))))
    (pad "1" smd roundrect (at 0 0) (size 1.0 1.0)
      (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25)
      (net 1 "NET1")
      (uuid "00000000-0000-0000-0000-00000000aa02")))
  (gr_rect (start 0.0 0.0) (end 130.0 60.0)
    (stroke (width 0.15) (type solid)) (fill none) (layer "Edge.Cuts")
    (uuid "00000000-0000-0000-0000-0000000000ee")))
""")


def write_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BOARD_TEMPLATE)


def _net1_summary(engine):
    tracks = engine.get_tracks()
    net1 = [t for t in tracks if t.net_name == "NET1"]
    L = sum(math.hypot(t.x2_mm - t.x1_mm, t.y2_mm - t.y1_mm) for t in net1)
    segs = [
        (round(t.x1_mm, 3), round(t.y1_mm, 3),
         round(t.x2_mm, 3), round(t.y2_mm, 3))
        for t in net1
    ]
    return len(net1), round(L, 3), segs


def _fmt_mask(mask):
    return "[" + ",".join(
        f"{n}={'1' if m else '0'}" for n, m in zip(ACTION_NAMES, mask)
    ) + "]"


def _classify(info: dict) -> str:
    # env writes the same action_class taxonomy into info["action_class"] when
    # available; if not, infer from info["empty_action"] / info["success"].
    if "action_class" in info:
        return info["action_class"]
    if info.get("idle"):
        return "idle"
    if info.get("empty_action"):
        return "valid_empty"
    if info.get("success") is False:
        return "valid_dispatch_fail"
    return "valid_effective"


def _step(env, action: dict, label: str) -> dict:
    obs, reward, terminated, truncated, info = env.step(action)
    engine = env._engine
    head = engine.get_route_head()
    is_routing = engine.is_routing()
    n_tracks, total_mm, segs = _net1_summary(engine)
    mask = env._get_action_mask()
    row = dict(
        label=label,
        action=ACTION_NAMES[int(action["action_type"])],
        reward=round(float(reward), 4),
        terminated=terminated,
        truncated=truncated,
        is_routing=is_routing,
        head=(round(head[0], 3), round(head[1], 3), head[2]),
        net1_tracks=n_tracks,
        net1_len=total_mm,
        action_class=_classify(info),
        success=info.get("success"),
        empty=info.get("empty_action"),
        mask=mask.tolist() if hasattr(mask, "tolist") else list(mask),
        segs=segs,
    )
    return row


def _print_row(row: dict) -> None:
    print(
        f"[{row['label']:>14}] act={row['action']:<11} "
        f"class={row['action_class']:<20} "
        f"reward={row['reward']:>8.4f} "
        f"is_routing={int(row['is_routing'])} "
        f"head={row['head']} "
        f"net1_tracks={row['net1_tracks']} len={row['net1_len']}mm "
        f"term={int(row['terminated'])} trunc={int(row['truncated'])} "
        f"\n                    mask={_fmt_mask(row['mask'])}"
    )


def main() -> int:
    warnings.simplefilter("default")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    board_path = OUTPUT_DIR / "stress_board.kicad_pcb"
    if not board_path.exists():
        write_board(board_path)

    print("== stress (env): same net + start_route, repeated make_line ==")
    print(f"board    : {board_path}")
    print(f"pins     : NET1 P1={START_PIN}  P2={FAR_PIN}")
    print(f"target   : {TARGET}  (NOT a pin)")
    print(f"corner   : MITERED_90  (env corner_mode=2)")
    print(f"iters    : {N_ITER}")
    print()

    # Synthetic board has no .kicad_pro -> opt into YAML DRC fallback so env
    # doesn't raise. The actual experiment is unaffected by which DRC defaults
    # the env applies.
    env = PCBWorld(
        board_path=str(board_path),
        max_steps=N_ITER + 10,
        masking_rule="default",
        corner_mode=2,                    # MITERED_90
        use_yaml_drc_fallback=True,
        drc_config_path=str(DEFAULT_DRC_CONFIG_PATH),
        reward_rule="drc_only_dense",
    )
    obs, info = env.reset()

    print("after reset:")
    print(f"   is_routing={env._engine.is_routing()} "
          f"current_net={env._dispatcher.current_net_id} "
          f"mask={_fmt_mask(env._get_action_mask())}")
    print()

    # 1. net_select NET1
    row = _step(env, {
        "action_type": ACT_NET_SELECT,
        "net_id": NET_ID,
        "x_mm": 0.0, "y_mm": 0.0, "layer": 1, "routing_mode": 2,
    }, "net_select")
    _print_row(row)

    # 2. start_route at P1
    row = _step(env, {
        "action_type": ACT_START_ROUTE,
        "x_mm": START_PIN[0], "y_mm": START_PIN[1], "layer": LAYER_HUMAN,
        "net_id": NET_ID, "routing_mode": 2,
    }, "start_route")
    _print_row(row)

    # 3..N+2: make_line to TARGET repeatedly
    last_key = None
    for i in range(1, N_ITER + 1):
        row = _step(env, {
            "action_type": ACT_MAKE_LINE,
            "x_mm": TARGET[0], "y_mm": TARGET[1],
            "routing_mode": 2,
            "net_id": NET_ID, "layer": LAYER_HUMAN,
        }, f"make_line#{i}")
        _print_row(row)
        key = (row["head"], row["is_routing"], row["net1_tracks"], row["net1_len"])
        if key == last_key:
            print(f"   ^ engine state unchanged from previous iter")
        last_key = key

    # Final committed board
    out_final = OUTPUT_DIR / "env_stress_final.kicad_pcb"
    env._engine.save(str(out_final))
    print()
    print("=== final committed segments on NET1 ===")
    _, total, segs = _net1_summary(env._engine)
    for j, s in enumerate(segs):
        print(f"   #{j}: ({s[0]}, {s[1]}) -> ({s[2]}, {s[3]})")
    print(f"   total = {total} mm")

    drc = env._engine.run_drc()
    print(f"\nDRC violations: {len(drc)}")
    for v in drc[:20]:
        print(f"   {v!r}")
    print(f"\nfinal board saved: {out_final}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
