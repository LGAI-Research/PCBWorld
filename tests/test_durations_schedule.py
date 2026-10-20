"""xdist file scheduling in tests/conftest.py: which durations seed a file reads and
writes, and the order files are handed to workers (``--dist loadfile --no-loadscope-reorder``).
"""
from __future__ import annotations

import json

from tests.conftest import _schedule_order, _seed_for


def test_each_tree_uses_its_nearest_seed(tmp_path):
    (tmp_path / "tests" / "sub").mkdir(parents=True)
    (tmp_path / "other" / "tests" / "sub").mkdir(parents=True)
    (tmp_path / "tests" / "durations.json").write_text("{}")
    (tmp_path / "other" / "tests" / "durations.json").write_text("{}")
    root = str(tmp_path)

    assert _seed_for("tests/sub/test_a.py", root) == str(tmp_path / "tests" / "durations.json")
    # a second tree keeps its own seed, never the first tree's
    assert (_seed_for("other/tests/sub/test_b.py", root)
            == str(tmp_path / "other" / "tests" / "durations.json"))
    assert _seed_for("elsewhere/test_c.py", root) is None


def test_longest_first_and_the_second_block_reversed():
    recorded = {f"f{i}": float(10 - i) for i in range(8)}      # f0 longest … f7 shortest
    order = _schedule_order([f"f{i}" for i in (5, 0, 7, 2, 1, 6, 3, 4)], recorded, 3)
    # first block: the 3 longest, one per worker in worker order
    assert order[:3] == ["f0", "f1", "f2"]
    # second block reversed: gw0 (longest file) is pre-given the shortest of f3..f5
    assert order[3:6] == ["f5", "f4", "f3"]
    assert order[6:] == ["f6", "f7"]


def test_unrecorded_files_go_last_in_collection_order():
    order = _schedule_order(["new_b", "old", "new_a"], {"old": 1.0}, 0)
    assert order == ["old", "new_b", "new_a"]


def test_tracked_seeds_name_only_their_own_tree():
    """--update-durations writes each file into its own tree's seed, so the public
    seed never names the private tree (public-hygiene)."""
    from pathlib import Path

    seed = Path(__file__).resolve().parent / "durations.json"
    assert all(k.startswith("tests/") for k in json.loads(seed.read_text()))
