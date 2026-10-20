"""Pool per-checkpoint rollout cells into one cell (docs/QUICKSTART.md §2).

    python experiments/_lib/pool_seed_cells.py <cell_dir> <part_dir>...

eval/pipeline.py rolls out one checkpoint per run, while a paper cell pools the checkpoints of
several training seeds. Each part is a Stage-1 output directory named like the cell (its boards
are ``boards/<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb``, ``s<SS>`` = the checkpoint's seed). The
pooled cell gets every part's boards (with their ``.kicad_pro``/``.kicad_prl``) in ``boards/``,
their ``per_rollout.csv`` rows with ``artifact_path`` at the moved files, each part's run record
and manifest under ``parts/<i>/``, and the first part's run record as its own — the record a
re-scoring reads for the scoring settings and the checkpoint's training reward, so the parts
must agree on that reward. Everything is checked before anything moves: a cell that already has
rows, a part named otherwise, a missing artifact, two parts writing the same board file or
differing training rewards leave the parts as they were.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from configs.loader.record import RUN_RECORD  # noqa: E402
from eval import eval_utils as u  # noqa: E402
from eval.pipeline import cell_training_reward  # noqa: E402

SIDECARS = (".kicad_pcb", ".kicad_pro", ".kicad_prl")
RECORDS = (RUN_RECORD, "manifest.json")


def pool(cell_dir: Path, parts: list[Path]) -> int:
    cell = cell_dir.name
    if (cell_dir / "per_rollout.csv").exists():
        raise SystemExit(f"{cell_dir} already has per_rollout.csv — pool into a fresh cell")
    boards = cell_dir / "boards"
    rows, moves, training = [], {}, []
    for part in parts:
        if part.name != cell:
            raise SystemExit(f"part {part} is not named like the cell {cell!r}")
        training.append(cell_training_reward(part))
        for row in u.read_csv(part / "per_rollout.csv"):
            if row.get("artifact_path"):
                src = Path(row["artifact_path"])
                src = src if src.is_absolute() else part / src
                dst = boards / src.name
                if not src.is_file():
                    raise SystemExit(f"{part}: artifact {src} is missing")
                if dst in moves or dst.exists():
                    raise SystemExit(f"{dst} written twice — two parts with the same checkpoint seed?")
                moves[dst] = src
                row["artifact_path"] = str(dst)
            rows.append(row)
    if any(t != training[0] for t in training):
        raise SystemExit(f"the parts' checkpoints were trained with different rewards: {training}")

    boards.mkdir(parents=True, exist_ok=True)
    for i, part in enumerate(parts):
        (cell_dir / "parts" / str(i)).mkdir(parents=True)
        for name in RECORDS:
            if (part / name).is_file():
                shutil.copy2(part / name, cell_dir / "parts" / str(i) / name)
    if (parts[0] / RUN_RECORD).is_file():
        shutil.copy2(parts[0] / RUN_RECORD, cell_dir / RUN_RECORD)
    for dst, src in moves.items():
        for suffix in SIDECARS:
            if src.with_suffix(suffix).is_file():
                src.with_suffix(suffix).replace(dst.with_suffix(suffix))
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    n = pool(Path(sys.argv[1]).resolve(), [Path(p).resolve() for p in sys.argv[2:]])
    print(f"[pool] {n} rollout rows from {len(sys.argv) - 2} checkpoints -> {sys.argv[1]}")
