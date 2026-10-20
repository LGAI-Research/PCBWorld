"""Write a copy of a fixture board with its pre-routed tracks replaced.

For tests that need copper the router would not lay by itself — a track ending a
few nm short of its pad, two tracks meeting on a straight run, two tracks leaving one
vertex at 45° or running on top of each other. The tracks go in as board text; pads
and rules stay the fixture's.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "offgrid_track_board.kicad_pcb"

_SEGMENT = """  (segment
    (start {} {})
    (end {} {})
    (width 0.2)
    (layer "F.Cu")
    (net 1)
    (uuid "00000000-0000-4000-8000-{:012x}")
  )
"""


def offgrid_board_with_tracks(tmp_path: Path, tracks) -> str:
    """Copy ``offgrid_track_board`` (NET1 pads A(10,10), C(40,10), B(25,20)) into
    ``tmp_path`` with its one pre-routed track replaced by ``tracks`` — (x1, y1, x2, y2)
    in mm, NET1 on F.Cu, 0.2 wide, UUIDs in list order. Returns the board path; the
    fixture's ``.kicad_pro`` is copied next to it."""
    text = _FIXTURE.read_text(encoding="utf-8")
    assert text.count("  (segment\n") == 1, "the fixture is expected to carry exactly one track"
    start = text.index("  (segment\n")
    end = text.index("\n  )\n", start) + len("\n  )\n")
    board = Path(tmp_path) / _FIXTURE.name
    board.write_text(text[:start] + "".join(_SEGMENT.format(*t, n) for n, t in enumerate(tracks, 1)) + text[end:],
                     encoding="utf-8")
    shutil.copyfile(_FIXTURE.with_suffix(".kicad_pro"), board.with_suffix(".kicad_pro"))
    return str(board)
