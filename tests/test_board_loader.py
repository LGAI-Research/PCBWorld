"""Path-building for load_boards_from_split_json — flat vs per-board-dir layout.

Pure-filesystem (tmp_path); no router/real-data needed. Covers the two board
layouts the eval-split loader must resolve, mirroring resolve_board_list:
  - flat (synth):        <ds>/<bid>.kicad_pcb          (no board_filename)
  - per-board-dir (d3):  <ds>/<bid>/<board_filename>   (board_filename set)

Plus the training-side resolver's missing-board contract (resolve_board_list):
every multi-board mode warns and skips, none hands a nonexistent path to the
engine.
"""
import json

import pytest

from methods._shared.board_loader import (
    load_boards_from_split_json,
    resolve_board_list,
)


def _write_split(tmp_path, *, board_filename=None, per_board_dir=False):
    ds = tmp_path / "boards"
    ds.mkdir()
    bids = ["b0", "b1"]
    for bid in bids:
        if per_board_dir:
            (ds / bid).mkdir()
            (ds / bid / board_filename).write_text("(kicad_pcb)")
        else:
            (ds / f"{bid}.kicad_pcb").write_text("(kicad_pcb)")
    split = {"easy": {"val": bids}, "dataset_dirs": {"val": str(ds)}}
    if board_filename is not None:
        split["board_filename"] = board_filename
    p = tmp_path / "split.json"
    p.write_text(json.dumps(split))
    return p, ds


def test_flat_layout_default(tmp_path):
    p, ds = _write_split(tmp_path, per_board_dir=False)
    boards = load_boards_from_split_json(p, "easy", "val")
    assert [b.path for b in boards] == [str(ds / "b0.kicad_pcb"), str(ds / "b1.kicad_pcb")]


def test_per_board_dir_from_json_board_filename(tmp_path):
    p, ds = _write_split(tmp_path, board_filename="routed.kicad_pcb", per_board_dir=True)
    boards = load_boards_from_split_json(p, "easy", "val")
    assert [b.path for b in boards] == [
        str(ds / "b0" / "routed.kicad_pcb"),
        str(ds / "b1" / "routed.kicad_pcb"),
    ]


def test_board_filename_arg_overrides_json(tmp_path):
    # json says one name, arg wins (both files exist; arg picks argname.kicad_pcb).
    ds = tmp_path / "boards"
    ds.mkdir()
    for bid in ["b0", "b1"]:
        (ds / bid).mkdir()
        (ds / bid / "argname.kicad_pcb").write_text("(kicad_pcb)")
    split = {"easy": {"val": ["b0", "b1"]}, "dataset_dirs": {"val": str(ds)},
             "board_filename": "jsonname.kicad_pcb"}
    p = tmp_path / "split.json"
    p.write_text(json.dumps(split))
    boards = load_boards_from_split_json(p, "easy", "val", board_filename="argname.kicad_pcb")
    assert all(b.path.endswith("argname.kicad_pcb") for b in boards)
    assert len(boards) == 2


def _write_train_split(tmp_path, ids, present):
    """A train split listing ``ids`` while only ``present`` exist on disk."""
    ds = tmp_path / "boards"
    ds.mkdir()
    for bid in present:
        (ds / f"{bid}.kicad_pcb").write_text("(kicad_pcb)")
    p = tmp_path / "split.json"
    p.write_text(json.dumps(
        {"easy": {"train": ids}, "dataset_dirs": {"train": str(ds)}}))
    return p, ds


# per_env_random shares the branch; per_env_epoch is what the paper recipe runs.
@pytest.mark.parametrize("order", ["per_env_epoch", "per_env_random"])
def test_multi_board_modes_skip_missing_boards(tmp_path, capsys, order):
    """A split file listing boards that are not on disk must not reach the C++.

    These two modes took a no-pre-scan shortcut and handed every listed path
    straight to the pool, so following the docs (generate a small set, then
    train with a split file listing the full 10k) died deep in a worker with a
    raw ``failed to load board`` from the engine, naming no split file.
    """
    p, ds = _write_train_split(tmp_path, ["b0", "b1", "b2"], present=["b1"])
    paths, pads = resolve_board_list(
        boards_order=order, single_board="", boards_json=str(p),
        difficulty="easy", split="train")
    assert paths == [str(ds / "b1.kicad_pcb")]
    assert pads == [0]
    out = capsys.readouterr().out
    assert "WARN: missing" in out and "b0.kicad_pcb" in out and "b2.kicad_pcb" in out
    assert "2 missing skipped" in out


def test_per_env_epoch_raises_when_no_board_exists(tmp_path):
    p, _ = _write_train_split(tmp_path, ["b0", "b1"], present=[])
    with pytest.raises(RuntimeError, match="No usable boards"):
        resolve_board_list(
            boards_order="per_env_epoch", single_board="", boards_json=str(p),
            difficulty="easy", split="train")
