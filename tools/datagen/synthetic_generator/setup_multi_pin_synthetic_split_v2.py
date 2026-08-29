"""v2 of setup_multi_pin_synthetic_split.py.

Builds the combined dir + split JSON for the migrated (v2) 2-layer dataset
that ships a `.kicad_pro` next to every `.kicad_pcb`. Symlinks BOTH files.

Source (v2 = migrated to .kicad_pro, train trimmed to first 10K boards):
  pcb_dataset_synthetic_multi_pin_2layer_v2/        train (10K)
  pcb_dataset_synthetic_multi_pin_2layer_test_v2/   test  (128)

Produces:
  pcb_dataset_multi_pin_2layer_combined_v2/    symlinks: pcb + pro pairs
  configs/datasets/misc/multi_pin_2layer_v2.json      {"easy": {"train": [...], "test": [...]}}
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN_SRC = REPO / "pcb_dataset_synthetic_multi_pin_2layer_v2"
TEST_SRC = REPO / "pcb_dataset_synthetic_multi_pin_2layer_test_v2"
DST = REPO / "pcb_dataset_multi_pin_2layer_combined_v2"
SPLIT_DIR = REPO / "configs" / "splits"
SPLIT_PATH = SPLIT_DIR / "multi_pin_2layer_v2.json"


def _link_pair(src_pcb: Path, dst_pcb: Path) -> None:
    src_pro = src_pcb.with_suffix(".kicad_pro")
    dst_pro = dst_pcb.with_suffix(".kicad_pro")
    for d in (dst_pcb, dst_pro):
        if d.is_symlink() or d.exists():
            d.unlink()
    dst_pcb.symlink_to(src_pcb.resolve())
    if src_pro.exists():
        dst_pro.symlink_to(src_pro.resolve())


def main() -> None:
    assert TRAIN_SRC.is_dir(), f"missing {TRAIN_SRC}"
    assert TEST_SRC.is_dir(), f"missing {TEST_SRC}"
    DST.mkdir(exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    def by_idx(p: Path) -> int:
        return int(p.stem.split("_", 1)[1])

    train_files = sorted(TRAIN_SRC.glob("board_*.kicad_pcb"), key=by_idx)
    test_files = sorted(TEST_SRC.glob("board_*.kicad_pcb"), key=by_idx)
    print(f"train source: {len(train_files)} boards")
    print(f"test  source: {len(test_files)} boards")

    train_ids: list[str] = []
    for src in train_files:
        bid = src.stem
        _link_pair(src, DST / f"{bid}.kicad_pcb")
        train_ids.append(bid)

    test_ids: list[str] = []
    for src in test_files:
        n = int(src.stem.split("_")[1])
        bid = f"testboard_{n:05d}"
        _link_pair(src, DST / f"{bid}.kicad_pcb")
        test_ids.append(bid)

    split = {"easy": {"train": train_ids, "test": test_ids}}
    SPLIT_PATH.write_text(json.dumps(split, indent=2))
    print(f"wrote {SPLIT_PATH} ({len(train_ids)} train, {len(test_ids)} test)")
    n_pcb = len(list(DST.glob("*.kicad_pcb")))
    n_pro = len(list(DST.glob("*.kicad_pro")))
    print(f"combined dir: {DST}  ({n_pcb} pcb, {n_pro} pro)")


if __name__ == "__main__":
    main()
