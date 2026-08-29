"""Two-net alternation stress test via gym env (PCBWorld).

Board (synthesized in this script):
  NET1 pads:  (10, 10)  &  (110, 50)
  NET2 pads:  (50, 10)  &  (150, 20)
  Edge.Cuts:  0..160 x 0..60  (large enough to fit both nets)
  Corner mode: MITERED_90  (no diagonals)
  Routing mode: WALKAROUND  (env default)

Sequence (per "round"):
  net_select(N)  ->  start_route(N.pin1, F.Cu)  ->  make_line(N.pin2)

Then we alternate the net between rounds:
  round 1: NET1
  round 2: NET2
  round 3: NET1   <-- will fail (already routed)
  round 4: NET2   <-- will fail (already routed)
  ...

What to observe
---------------
- Geometry under 90 degrees: NET1 L-shape (horizontal 10,10->110,10 then
  vertical to 110,50) overlaps with NET2's natural L-shape
  (50,10->150,10 then vertical to 150,20). So NET2's walkaround
  placer must detour around NET1's already-committed track.
- DRC: any crossings get flagged.
- env machinery: after both nets are routed, ratsnest is empty so
  net_select fails -> mask_reject on every further round.

Run (from the repo root, with the C++ router built):
  conda activate cadagent
  python tests/stress/stress_make_line_alternate_env.py
"""

from __future__ import annotations

import math
import sys
import textwrap
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_MODULE_DIR = PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"
OUTPUT_DIR = PROJECT_ROOT / "var" / "tests" / "output" / "stress_make_line_alternate"

sys.path.insert(0, str(RL_MODULE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from pcb_world.core.env import PCBWorld  # noqa: E402
from pcb_world.engine.drc_config import DEFAULT_DRC_CONFIG_PATH  # noqa: E402
from pcb_world.core.masking import (  # noqa: E402
    ACT_MAKE_LINE,
    ACT_NET_END,
    ACT_NET_SELECT,
    ACT_START_ROUTE,
    ACTION_NAMES,
)


NETS = {
    1: {"name": "NET1", "p1": (10.0, 10.0), "p2": (60.0, 10.0)},
    2: {"name": "NET2", "p1": (10.0, 30.0), "p2": (60.0, 30.0)},
}
LAYER_HUMAN = 1
N_ROUNDS = 8  # alternates start_route+make_line for Y=10 (NET1) and Y=30 (NET2)


# pad-footprint template (mm-units inside the .kicad_pcb)
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
    pad_index = 0
    for net_code, n in NETS.items():
        for pin_key in ("p1", "p2"):
            x, y = n[pin_key]
            pad_index += 1
            pads.append(_PAD_TPL.format(
                x=x, y=y,
                fuuid=f"00000000-0000-0000-0000-0000000000{pad_index:02x}",
                ref=f"P{net_code}_{pin_key.upper()}",
                val=f"Pad{net_code}{pin_key.upper()}",
                net_code=net_code, net_name=n["name"],
                puuid=f"00000000-0000-0000-0000-00000000aa{pad_index:02x}",
            ))
    nets_decl = "\n  ".join([f"(net {nc} \"{n['name']}\")" for nc, n in NETS.items()])
    return textwrap.dedent(f"""\
        (kicad_pcb
          (version 20241229)
          (generator "stress_make_line_alternate")
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
          (gr_rect (start 0.0 0.0) (end 80.0 50.0)
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


def _ratsnest_remaining(engine) -> dict[int, int]:
    edges = engine.get_ratsnest()
    out: dict[int, int] = {}
    for e in edges:
        out[int(e.net_code)] = out.get(int(e.net_code), 0) + 1
    return out


def _fmt_mask(mask) -> str:
    return "[" + ",".join(
        f"{n}={'1' if m else '0'}" for n, m in zip(ACTION_NAMES, mask)
    ) + "]"


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
    obs, reward, terminated, truncated, info = env.step(action)
    engine = env._engine
    head = engine.get_route_head()
    rats = _ratsnest_remaining(engine)
    n_all, total_all, _ = _summary(engine)
    return dict(
        label=label,
        action=ACTION_NAMES[int(action["action_type"])],
        reward=round(float(reward), 4),
        terminated=terminated, truncated=truncated,
        is_routing=engine.is_routing(),
        head=(round(head[0], 3), round(head[1], 3), head[2]),
        all_tracks=n_all, all_len=total_all,
        rats=rats,
        action_class=_classify(info),
        mask=env._get_action_mask().tolist(),
    )


def _print_row(r: dict) -> None:
    print(
        f"  [{r['label']:>18}] act={r['action']:<11} "
        f"class={r['action_class']:<20} "
        f"reward={r['reward']:>8.4f} "
        f"is_routing={int(r['is_routing'])} "
        f"head={r['head']} "
        f"tracks={r['all_tracks']} len={r['all_len']}mm "
        f"rats={r['rats']} "
        f"\n                       mask={_fmt_mask(r['mask'])}"
    )


def main() -> int:
    warnings.simplefilter("default")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    board_path = OUTPUT_DIR / "alt_board.kicad_pcb"
    board_path.parent.mkdir(parents=True, exist_ok=True)
    board_path.write_text(_board_text())

    print("== alternate-net stress (env) ==")
    print(f"board    : {board_path}")
    for nc, n in NETS.items():
        print(f"  {n['name']} (net_code={nc})  pads {n['p1']}  <->  {n['p2']}")
    print(f"corner   : MITERED_90  (env corner_mode=2)")
    print(f"rounds   : {N_ROUNDS}  (alternates NET1, NET2, NET1, ...)")
    print()

    env = PCBWorld(
        board_path=str(board_path),
        max_steps=N_ROUNDS * 2 + 8,
        masking_rule="default",
        corner_mode=2,
        use_yaml_drc_fallback=True,
        drc_config_path=str(DEFAULT_DRC_CONFIG_PATH),
        reward_rule="drc_only_dense",
    )
    env.reset()
    print(f"after reset: is_routing={env._engine.is_routing()} "
          f"rats={_ratsnest_remaining(env._engine)} "
          f"mask={_fmt_mask(env._get_action_mask())}")
    print()

    cumulative_reward = 0.0

    # ONE net_select up front (NET1). After this, the loop only does
    # start_route + make_line, alternating coords between Y=10 (NET1 pads)
    # and Y=30 (NET2 pads). The C++ engine picks the net automatically from
    # the start coordinate, so we do NOT need to net_select again.
    print("--- prelude: single net_select(NET1) ---")
    row = _step(env, {
        "action_type": ACT_NET_SELECT, "net_id": 1,
        "x_mm": 0.0, "y_mm": 0.0, "layer": LAYER_HUMAN, "routing_mode": 2,
    }, "net_select(N1)")
    cumulative_reward += row["reward"]
    _print_row(row)
    print()

    for r in range(1, N_ROUNDS + 1):
        net_code = 1 if (r % 2 == 1) else 2
        net = NETS[net_code]
        print(f"--- iter {r}: start_route + make_line on {net['name']} ({net['p1']} -> {net['p2']}) ---")

        # start_route
        row = _step(env, {
            "action_type": ACT_START_ROUTE,
            "x_mm": net["p1"][0], "y_mm": net["p1"][1], "layer": LAYER_HUMAN,
            "net_id": net_code, "routing_mode": 2,
        }, f"i{r}.start_route")
        cumulative_reward += row["reward"]
        _print_row(row)

        # make_line to the other pin
        row = _step(env, {
            "action_type": ACT_MAKE_LINE,
            "x_mm": net["p2"][0], "y_mm": net["p2"][1],
            "routing_mode": 2,
            "net_id": net_code, "layer": LAYER_HUMAN,
        }, f"i{r}.make_line")
        cumulative_reward += row["reward"]
        _print_row(row)

        print(f"   net1 tracks: {_summary(env._engine, 'NET1')[:2]}  "
              f"net2 tracks: {_summary(env._engine, 'NET2')[:2]}  "
              f"cum_reward={round(cumulative_reward, 4)}")
        env._engine.save(str(OUTPUT_DIR / f"after_iter{r:02d}.kicad_pcb"))
        print()

    # Final inspection
    print("=== final segments per net ===")
    for net_code, n in NETS.items():
        cnt, total, segs = _summary(env._engine, n["name"])
        print(f"  {n['name']}: {cnt} segments, total {total} mm")
        for j, s in enumerate(segs):
            print(f"     #{j}: ({s[0]}, {s[1]}) -> ({s[2]}, {s[3]})")
    drc = env._engine.run_drc()
    print(f"\nDRC violations: {len(drc)}")
    for v in drc[:30]:
        print(f"   {v!r}")
    out = OUTPUT_DIR / "alt_final.kicad_pcb"
    env._engine.save(str(out))
    print(f"\ncumulative reward over {N_ROUNDS} rounds = {round(cumulative_reward, 4)}")
    print(f"final board saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
