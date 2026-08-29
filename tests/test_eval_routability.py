"""Routability is measured on pad groups, not on the KiCad ratsnest.

The ratsnest counts every copper cluster, so a dangling island inflates it,
which would push ``(u_0 - u_t) / u_0`` below zero.  Routability is defined
instead as the fraction of pin-to-pin connections actually made, which
dangling copper cannot affect at all.
"""
from pathlib import Path

import pytest

from eval.metrics import evaluate_one

FIXTURE = Path(__file__).parent / "fixtures" / "sample_board.kicad_pcb"

# A segment on an already-routed net, placed far from every pad and every other
# track: its own copper cluster, holding no pad. KiCad counts it as one more
# ratsnest edge; it must not move routability or success.
ISLAND_SEGMENT = """
  (segment
    (start 500 500)
    (end 505 500)
    (width 0.2)
    (layer "F.Cu")
    (net 1)
    (uuid "00000000-0000-4000-8000-00000000dead")
  )
"""


def _board_with_island(tmp_path: Path) -> Path:
    text = FIXTURE.read_text()
    cut = text.rstrip().rfind(")")
    out = tmp_path / "island.kicad_pcb"
    out.write_text(text[:cut] + ISLAND_SEGMENT + text[cut:])
    # Engine load contract: pro sibling required — the island injection only
    # adds copper, so the source fixture's rules apply unchanged.
    out.with_suffix(".kicad_pro").write_bytes(
        FIXTURE.with_suffix(".kicad_pro").read_bytes()
    )
    return out


def test_fully_routed_board_scores_one():
    result = evaluate_one(str(FIXTURE), None)

    assert result["routability"] == pytest.approx(1.0)
    assert result["success"] is True
    assert result["extras"]["unrouted_edges_remaining"] == 0


def test_dangling_island_leaves_routability_untouched(tmp_path):
    clean = evaluate_one(str(FIXTURE), None)
    islanded = evaluate_one(str(_board_with_island(tmp_path)), None)

    # The island is visible in the ratsnest ...
    assert (
        islanded["extras"]["unrouted_edges_remaining"]
        > clean["extras"]["unrouted_edges_remaining"]
    )
    # ... but carries no pad, so it changes neither pad grouping nor the metric.
    assert islanded["routability"] == pytest.approx(clean["routability"])
    assert islanded["routability"] == pytest.approx(1.0)
    assert islanded["success"] is True
    assert (
        islanded["extras"]["pad_groups_remaining"]
        == clean["extras"]["pad_groups_remaining"]
    )


def test_denominator_is_the_boards_own_initial_pad_grouping():
    extras = evaluate_one(str(FIXTURE), None)["extras"]

    # required = pad groups that had to be merged away; a fully routed board
    # merges all of them, which is what makes routability exactly 1.0.
    assert extras["pad_connections_required"] == (
        extras["pad_groups_initial"] - extras["pad_groups_remaining"]
    )
