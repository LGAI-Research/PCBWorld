"""Stress test: same start_route, repeated make_line to a fixed off-pin target.

Scenario
--------
- Board: single NET1 with two SMD pads at (10, 10) and (110, 50) on F.Cu.
- 90 degree corner constraint (CORNER_MITERED_90).
- start_route once at (10, 10).
- Repeatedly call move(110, 10) + fix_route(110, 10, force_finish=False).
  Target (110, 10) is an interior point, NOT the other pin (which is at 110, 50).
  Without force_finish, the placer should stay alive between calls.

What to observe per iteration
-----------------------------
- placer head (get_route_head)
- router state (0=IDLE, 3=ROUTE_TRACK)
- is_routing flag, failure_reason
- committed track count and total length on NET1
- WIP segments (placer-held, not yet committed)
- unrouted count / current target

This script is intentionally a probe: we want to see whether
  (a) iteration 1 emits a horizontal segment 10,10 -> 110,10 then the head
      moves toward (110, 50),
  (b) iterations 2..N are no-ops because head == target,
  (c) repeated calls leak/duplicate tracks, stall, or crash.

Run
---
  python tests/stress/stress_make_line_repeat.py
  (the script puts the built router module dir and the repo root on sys.path
  itself; run it from the repo root)
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_MODULE_DIR = PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"
OUTPUT_DIR = PROJECT_ROOT / "var" / "tests" / "output" / "stress_make_line_repeat"

sys.path.insert(0, str(RL_MODULE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import kicad_rl_router as krl  # noqa: E402


START_PIN = (10.0, 10.0)
FAR_PIN = (110.0, 50.0)
TARGET = (110.0, 10.0)
LAYER = 0  # F.Cu (board layer index 0)
N_ITER = 8


BOARD_TEMPLATE = textwrap.dedent("""\
(kicad_pcb
  (version 20241229)
  (generator "stress_make_line_repeat")
  (generator_version "9.0.5")
  (general
    (thickness 1.6)
    (legacy_teardrops no)
  )
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user "B.Mask")
    (39 "F.Mask" user "F.Mask")
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (44 "Edge.Cuts" user)
  )
  (setup
    (pad_to_mask_clearance 0)
    (allow_soldermask_bridges_in_footprints no)
  )

  (net 0 "")
  (net 1 "NET1")

  (net_class "Default" "Default net class"
    (clearance 0.2)
    (trace_width 0.2)
    (via_dia 0.6)
    (via_drill 0.3)
    (uvia_dia 0.3)
    (uvia_drill 0.1)
  )

  (footprint "SamplePad:FCu"
    (layer "F.Cu")
    (at 10 10)
    (uuid "00000000-0000-0000-0000-000000000001")
    (property "Reference" "P1"
      (at 0 -1) (layer "F.SilkS")
      (effects (font (size 0.6 0.6) (thickness 0.1)))
    )
    (property "Value" "Pad1"
      (at 0 1) (layer "F.Fab")
      (effects (font (size 0.6 0.6) (thickness 0.1)))
    )
    (pad "1" smd roundrect
      (at 0 0) (size 1.0 1.0)
      (layers "F.Cu" "F.Paste" "F.Mask")
      (roundrect_rratio 0.25)
      (net 1 "NET1")
      (uuid "00000000-0000-0000-0000-00000000aa01")
    )
  )

  (footprint "SamplePad:FCu"
    (layer "F.Cu")
    (at 110 50)
    (uuid "00000000-0000-0000-0000-000000000002")
    (property "Reference" "P2"
      (at 0 -1) (layer "F.SilkS")
      (effects (font (size 0.6 0.6) (thickness 0.1)))
    )
    (property "Value" "Pad2"
      (at 0 1) (layer "F.Fab")
      (effects (font (size 0.6 0.6) (thickness 0.1)))
    )
    (pad "1" smd roundrect
      (at 0 0) (size 1.0 1.0)
      (layers "F.Cu" "F.Paste" "F.Mask")
      (roundrect_rratio 0.25)
      (net 1 "NET1")
      (uuid "00000000-0000-0000-0000-00000000aa02")
    )
  )

  (gr_rect
    (start 0.0 0.0)
    (end 130.0 60.0)
    (stroke (width 0.15) (type solid))
    (fill none)
    (layer "Edge.Cuts")
    (uuid "00000000-0000-0000-0000-0000000000ee")
  )
)
""")


def write_board(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BOARD_TEMPLATE)


def _track_len(t) -> float:
    import math
    return math.hypot(t.x2_mm - t.x1_mm, t.y2_mm - t.y1_mm)


def _snapshot(router, label: str) -> dict:
    head = router.get_route_head()  # (x_mm, y_mm, layer) or (0,0,-1)
    state = router.get_router_state()
    is_routing = router.is_routing()
    fail = router.get_failure_reason()
    target = router.get_routing_target()

    try:
        wip = router.get_wip_segments()
    except Exception as e:  # noqa: BLE001
        wip = f"<err: {e!r}>"

    tracks = router.get_tracks()
    net1 = [t for t in tracks if t.net_name == "NET1"]
    total = sum(_track_len(t) for t in net1)
    seg_list = [
        (round(t.x1_mm, 3), round(t.y1_mm, 3),
         round(t.x2_mm, 3), round(t.y2_mm, 3))
        for t in net1
    ]
    return dict(
        label=label,
        head=(round(head[0], 3), round(head[1], 3), head[2]),
        state=state,
        is_routing=is_routing,
        fail=fail or "",
        target=(round(target[0], 3), round(target[1], 3), target[2]),
        wip_len=(len(wip) if hasattr(wip, "__len__") else wip),
        tracks=len(net1),
        total_mm=round(total, 3),
        segs=seg_list,
    )


def _fmt(row: dict) -> str:
    return (
        f"[{row['label']:>10}] "
        f"head={row['head']} "
        f"state={row['state']} "
        f"is_routing={row['is_routing']} "
        f"target={row['target']} "
        f"wip={row['wip_len']} "
        f"tracks={row['tracks']} "
        f"len={row['total_mm']}mm "
        f"fail={row['fail']!r}"
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    board_path = OUTPUT_DIR / "stress_board.kicad_pcb"
    write_board(board_path)

    print("== stress: same start_route, repeated make_line ==")
    print(f"board    : {board_path}")
    print(f"pins     : NET1 P1={START_PIN}  P2={FAR_PIN}")
    print(f"target   : {TARGET}  (NOT a pin -- interior point)")
    print(f"corner   : MITERED_90  (no diagonals)")
    print(f"iters    : {N_ITER}")
    print()

    router = krl.RLRouter(str(board_path), "")
    router.build_connectivity()
    router.set_corner_mode(krl.CORNER_MITERED_90)
    router.set_routing_mode(krl.MODE_WALKAROUND)  # baseline placer mode

    # Pad sanity
    pads = router.get_pads()
    print("pads :")
    for p in pads:
        print(f"   net={p.net_name} pad={p.pad_name} at=({p.x_mm:.3f}, {p.y_mm:.3f}) layer={p.layer}")
    print()

    rows = []
    rows.append(_snapshot(router, "pre-start"))
    print(_fmt(rows[-1]))

    ok = router.start_route(START_PIN[0], START_PIN[1], LAYER)
    print(f"\nstart_route({START_PIN[0]}, {START_PIN[1]}, layer={LAYER}) -> {ok}")
    rows.append(_snapshot(router, "post-start"))
    print(_fmt(rows[-1]))
    print()

    last_state = None
    for i in range(1, N_ITER + 1):
        mv_ok = router.move(TARGET[0], TARGET[1])
        fx_ok = router.fix_route(TARGET[0], TARGET[1], False)
        snap = _snapshot(router, f"iter{i}")
        snap["move_ok"] = mv_ok
        snap["fix_ok"] = fx_ok
        rows.append(snap)
        print(f"  move={mv_ok} fix={fx_ok}  " + _fmt(snap))

        # Save board snapshot after iter 1 and iter N for visual inspection
        if i in (1, 2, N_ITER):
            out = OUTPUT_DIR / f"after_iter{i:02d}.kicad_pcb"
            router.save(str(out))

        # Detect stall / change-of-state
        state_key = (snap["head"], snap["state"], snap["tracks"], snap["total_mm"])
        if state_key == last_state:
            print(f"      ^ state unchanged from previous iter -> idempotent / stalled")
        last_state = state_key

    # Finalize so we can see what the router thinks the committed board looks like
    print()
    print("finalizing (fix_route(target, force_finish=True))...")
    final_ok = router.fix_route(TARGET[0], TARGET[1], True)
    print(f"  force_finish -> {final_ok}")
    rows.append(_snapshot(router, "finalized"))
    print(_fmt(rows[-1]))

    # Try a fresh start_route + make_line straight to the OTHER pin to see what
    # the placer thinks the natural 90-degree path is, for context.
    router2 = krl.RLRouter(str(board_path), "")
    router2.build_connectivity()
    router2.set_corner_mode(krl.CORNER_MITERED_90)
    router2.set_routing_mode(krl.MODE_WALKAROUND)
    print()
    print("-- reference run: single-shot start_route + move + force_finish to FAR_PIN --")
    print(f"   start_route({START_PIN}) move({FAR_PIN}) fix_route({FAR_PIN}, force_finish=True)")
    if router2.start_route(START_PIN[0], START_PIN[1], LAYER):
        router2.move(FAR_PIN[0], FAR_PIN[1])
        ref_ok = router2.fix_route(FAR_PIN[0], FAR_PIN[1], True)
        print(f"   final ok={ref_ok}")
        print(_fmt(_snapshot(router2, "reference")))
    else:
        print("   start_route failed")

    print()
    print("=== detailed segments (final state, stress run) ===")
    final_segs = rows[-1]["segs"]
    for i, s in enumerate(final_segs):
        print(f"   #{i}: ({s[0]}, {s[1]}) -> ({s[2]}, {s[3]})")

    drc = router.run_drc()
    print(f"\nDRC violations on committed board: {len(drc)}")
    for v in drc[:20]:
        print(f"   {v!r}")

    out_final = OUTPUT_DIR / "stress_final.kicad_pcb"
    router.save(str(out_final))
    print(f"\nfinal board saved: {out_final}")
    print(f"intermediate boards in: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
