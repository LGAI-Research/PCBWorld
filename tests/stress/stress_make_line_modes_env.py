"""Routing-mode sweep on the parallel two-net scenario via env.

Same board / coords as scripts/_stress_make_line_alternate_env.py:
  NET1 pads: (10, 10)  <->  (110, 10)
  NET2 pads: (10, 30)  <->  (110, 30)
  edge cuts 0..130 x 0..50, F.Cu only, 90 degree corner mode.

For each routing_mode in {MARK_OBSTACLES=0, SHOVE=1, WALKAROUND=2}:
  - Fresh PCBWorld
  - net_select(NET1) (single up-front; relies on stale current_net_id
    to keep has_net=True so start_route stays mask-legal across nets)
  - iter 1: start_route(NET1.p1, F.Cu) + make_line(NET1.p2, mode)
  - iter 2: start_route(NET2.p1, F.Cu) + make_line(NET2.p2, mode)

Log per (mode, iter):
  - action_class, reward
  - segments committed per net (count + total mm)
  - whole-board DRC violations after each commit

Run (from the repo root, with the C++ router built):
  conda activate cadagent
  python tests/stress/stress_make_line_modes_env.py
"""

from __future__ import annotations

import math
import sys
import textwrap
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_MODULE_DIR = PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"
OUTPUT_DIR = PROJECT_ROOT / "var" / "tests" / "output" / "stress_make_line_modes_env"

sys.path.insert(0, str(RL_MODULE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from pcb_world.core.env import PCBWorld  # noqa: E402
from pcb_world.engine.drc_config import DEFAULT_DRC_CONFIG_PATH  # noqa: E402
from pcb_world.core.masking import (  # noqa: E402
    ACT_MAKE_LINE,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACTION_NAMES,
)


# Routing mode IDs match pcb_world/core/action.py + kicad_rl_router constants
# 0=MARK_OBSTACLES, 1=SHOVE, 2=WALKAROUND
MODES = [
    (0, "mark_obstacles"),
    (1, "shove"),
    (2, "walkaround"),
]

NETS = {
    1: {"name": "NET1", "p1": (10.0, 10.0), "p2": (110.0, 10.0)},
    2: {"name": "NET2", "p1": (10.0, 30.0), "p2": (110.0, 30.0)},
}
LAYER_HUMAN = 1


_PAD_TPL = textwrap.dedent("""\
  (footprint "SamplePad:FCu" (layer "F.Cu") (at {x} {y})
    (uuid "{fuuid}")
    (property "Reference" "{ref}" (at 0 -1) (layer "F.SilkS")
      (effects (font (size 0.6 0.6) (thickness 0.1))))
    (property "Value" "{val}" (at 0 1) (layer "F.Fab")
      (effects (font (size 0.6 0.6) (thickness 0.1))))
    (pad "1" smd roundrect (at 0 0) (size 1.0 1.0)
      (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25)
      (net {net_code} "{net_name}")
      (uuid "{puuid}")))
""")


def _board_text() -> str:
    pads = []
    idx = 0
    for net_code, n in NETS.items():
        for pin_key in ("p1", "p2"):
            x, y = n[pin_key]
            idx += 1
            pads.append(_PAD_TPL.format(
                x=x, y=y,
                fuuid=f"00000000-0000-0000-0000-0000000000{idx:02x}",
                ref=f"P{net_code}_{pin_key.upper()}",
                val=f"Pad{net_code}{pin_key.upper()}",
                net_code=net_code, net_name=n["name"],
                puuid=f"00000000-0000-0000-0000-00000000aa{idx:02x}",
            ))
    nets_decl = "\n  ".join([f"(net {nc} \"{n['name']}\")" for nc, n in NETS.items()])
    return textwrap.dedent(f"""\
        (kicad_pcb
          (version 20241229)
          (generator "stress_make_line_modes")
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
          (net 0 "")
          {nets_decl}
          (net_class "Default" "Default net class"
            (clearance 0.2) (trace_width 0.2)
            (via_dia 0.6) (via_drill 0.3) (uvia_dia 0.3) (uvia_drill 0.1))
        """) + "\n".join(pads) + textwrap.dedent("""
          (gr_rect (start 0.0 0.0) (end 130.0 50.0)
            (stroke (width 0.15) (type solid)) (fill none) (layer "Edge.Cuts")
            (uuid "00000000-0000-0000-0000-0000000000ee")))
        """)


def _summary(engine, net_name: str | None = None):
    tracks = engine.get_tracks()
    if net_name is not None:
        tracks = [t for t in tracks if t.net_name == net_name]
    L = sum(math.hypot(t.x2_mm - t.x1_mm, t.y2_mm - t.y1_mm) for t in tracks)
    segs = [
        (round(t.x1_mm, 3), round(t.y1_mm, 3),
         round(t.x2_mm, 3), round(t.y2_mm, 3))
        for t in tracks
    ]
    return len(tracks), round(L, 3), segs


def _classify(info: dict) -> str:
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
    _, reward, _, _, info = env.step(action)
    return dict(label=label, reward=round(float(reward), 4),
                cls=_classify(info))


def run_mode(mode_id: int, mode_name: str) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    board_path = OUTPUT_DIR / f"board_{mode_name}.kicad_pcb"
    board_path.write_text(_board_text())

    env = PCBWorld(
        board_path=str(board_path),
        max_steps=20,
        masking_rule="default",
        corner_mode=2,                      # MITERED_90
        use_yaml_drc_fallback=True,
        drc_config_path=str(DEFAULT_DRC_CONFIG_PATH),
        reward_rule="drc_only_dense",
    )
    env.reset()

    rows = []
    cumr = 0.0

    # single net_select(NET1) so subsequent start_route is mask-legal
    r = _step(env, {
        "action_type": ACT_NET_SELECT, "net_id": 1,
        "x_mm": 0.0, "y_mm": 0.0, "layer": LAYER_HUMAN, "routing_mode": mode_id,
    }, "net_select")
    cumr += r["reward"]; rows.append(r)

    for i, net_code in enumerate((1, 2), start=1):
        net = NETS[net_code]
        # start_route — routing_mode is ignored here (only set by make_line)
        r = _step(env, {
            "action_type": ACT_START_ROUTE,
            "x_mm": net["p1"][0], "y_mm": net["p1"][1], "layer": LAYER_HUMAN,
            "net_id": net_code, "routing_mode": mode_id,
        }, f"i{i}.start_route")
        cumr += r["reward"]; rows.append(r)

        r = _step(env, {
            "action_type": ACT_MAKE_LINE,
            "x_mm": net["p2"][0], "y_mm": net["p2"][1],
            "routing_mode": mode_id,             # << mode swept here
            "net_id": net_code, "layer": LAYER_HUMAN,
        }, f"i{i}.make_line[{mode_name}]")
        cumr += r["reward"]; rows.append(r)

    # End-state
    n1, l1, s1 = _summary(env._engine, "NET1")
    n2, l2, s2 = _summary(env._engine, "NET2")
    drc = env._engine.run_drc()
    out = OUTPUT_DIR / f"final_{mode_name}.kicad_pcb"
    env._engine.save(str(out))

    return dict(
        mode_id=mode_id, mode_name=mode_name,
        rows=rows, cumr=round(cumr, 4),
        net1=(n1, l1, s1), net2=(n2, l2, s2),
        drc=len(drc), drc_list=[repr(v) for v in drc],
        save=str(out),
    )


def main() -> int:
    warnings.simplefilter("default")
    print("== make_line routing_mode sweep (env) ==")
    print(f"board    : 130x50 mm, 2 nets parallel horizontal")
    print(f"  NET1 (10,10) <-> (110,10)")
    print(f"  NET2 (10,30) <-> (110,30)")
    print(f"corner   : MITERED_90")
    print(f"modes    : {[m[1] for m in MODES]}")
    print()

    results = []
    for mid, name in MODES:
        print(f"=== mode = {name} (id={mid}) ===")
        res = run_mode(mid, name)
        for row in res["rows"]:
            print(f"   [{row['label']:>26}] class={row['cls']:<20} reward={row['reward']:+.4f}")
        n1, l1, s1 = res["net1"]
        n2, l2, s2 = res["net2"]
        print(f"   NET1: {n1} segs, {l1} mm  segs={s1}")
        print(f"   NET2: {n2} segs, {l2} mm  segs={s2}")
        print(f"   DRC : {res['drc']} violations")
        for v in res["drc_list"][:5]:
            print(f"        {v}")
        print(f"   cumulative reward = {res['cumr']}")
        print(f"   saved: {res['save']}")
        print()
        results.append(res)

    print("=== comparison table ===")
    hdr = f"{'mode':<16} {'net1_segs':>10} {'net1_len':>10} {'net2_segs':>10} {'net2_len':>10} {'drc':>5} {'cum_r':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        n1, l1, _ = r["net1"]
        n2, l2, _ = r["net2"]
        print(f"{r['mode_name']:<16} {n1:>10} {l1:>10.3f} {n2:>10} {l2:>10.3f} {r['drc']:>5} {r['cumr']:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
