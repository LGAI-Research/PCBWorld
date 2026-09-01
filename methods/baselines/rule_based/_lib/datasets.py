"""Dataset path resolver.

Maps ``(dataset_name, grid?, split?, board_id)`` to:
    * raw unrouted ``.kicad_pcb``
    * matching ``.kicad_pro``
    * pre-converted Specctra DSN (``_unrouted.dsn``)
    * pre-converted ORP (``.orp``)

All locations come from env vars, falling back to the configs/paths.yaml
registry (no hardcoded host-specific paths in this module).
The runner reads these to (a) feed each baseline its preferred input format,
and (b) feed ``eval.metrics.evaluate_one`` the matching ``.kicad_pro``.

Supported datasets:

| dataset        | required env vars                                |
|----------------|--------------------------------------------------|
| pcbench        | PCBENCH_PCB_ROOT, PCBENCH_DSN_ROOT               |
| synthetic_2l   | SYNTH2L_PCB_ROOT, SYNTH2L_DSN_ROOT               |
| synthetic_1l   | SYNTH1L_PCB_ROOT_<G>, SYNTH1L_DSN_ROOT_<G>       |
|                |   where <G> ∈ {10, 50, 100, 200, 500}            |

For synthetic_1l, the env-var names are suffixed with the grid number so
multiple grids can be configured side-by-side. The runner's ``--grid``
flag selects which pair to look at.

All envs accept absolute paths. When a specific var is unset, the location
falls back to the named logical dataset in ``configs/paths.yaml`` (resolved
through ``configs.loader.paths``, i.e. under the repo-wide data root); with
an empty data root, resolution raises with a message naming
``CADAGENT_DATA_ROOT``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Root resolution — the specific env var wins; otherwise the location comes
# from the logical dataset registered in configs/paths.yaml (env
# CADAGENT_DATA_ROOT overrides the root, the yaml default falls back). No
# machine-specific path is hardcoded here.
# ---------------------------------------------------------------------------

_DATASET_NAMES = {
    "PCBENCH_PCB_ROOT": "pcbench_exacad",
    "PCBENCH_DSN_ROOT": "pcbench_exacad_dsn",
    "SYNTH2L_PCB_ROOT": "synth_2L_v2",
    "SYNTH2L_DSN_ROOT": "synth_2L_v2_dsn",
}
for _g in (10, 50, 100, 200, 500):
    _DATASET_NAMES[f"SYNTH1L_PCB_ROOT_{_g}"] = f"synth_1L_grid{_g}_5net_v02"
    _DATASET_NAMES[f"SYNTH1L_DSN_ROOT_{_g}"] = f"synth_1L_grid{_g}_5net_v02_dsn"

PCBENCH_STEM = "processed_v9_guide_v3"
SUPPORTED_GRIDS = (10, 50, 100, 200, 500)
SUPPORTED_SPLITS = ("test", "train", "val")


# ---------------------------------------------------------------------------
# Per-board path record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardPaths:
    """All canonical paths for a single board sample."""

    board_id: str
    raw_pcb: Path
    pro: Path
    dsn: Path
    orp: Path

    def __post_init__(self) -> None:
        # ``Path`` objects are immutable, so frozen=True works; this method
        # exists only to give a clear error when a caller forgets to check
        # ``exists()``.
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _env_root(name: str) -> Path:
    val = os.environ.get(name)
    if val:
        return Path(val)
    # Lazy import with a repo-root bootstrap so this module also works when
    # imported without the runner's sys.path setup (e.g. from the _lib dir).
    repo_root = str(Path(__file__).resolve().parents[4])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from configs.loader.paths import resolve_dataset

    return resolve_dataset(_DATASET_NAMES[name])


def _require_grid(grid: int | None) -> int:
    if grid is None or grid not in SUPPORTED_GRIDS:
        raise ValueError(
            f"synthetic_1l requires --grid ∈ {SUPPORTED_GRIDS}, got {grid!r}"
        )
    return int(grid)


def _require_split(split: str | None) -> str:
    if split is None or split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"--split must be one of {SUPPORTED_SPLITS}, got {split!r}"
        )
    return split


# ---------------------------------------------------------------------------
# pcbench
# ---------------------------------------------------------------------------


def _pcbench_root_pair() -> tuple[Path, Path]:
    pcb = _env_root("PCBENCH_PCB_ROOT")
    dsn = _env_root("PCBENCH_DSN_ROOT")
    return pcb, dsn


def _pcbench_board(folder: str) -> BoardPaths:
    pcb_root, dsn_root = _pcbench_root_pair()
    pcb_folder = pcb_root / folder
    dsn_folder = dsn_root / folder
    return BoardPaths(
        board_id=folder,
        raw_pcb=pcb_folder / f"{PCBENCH_STEM}_unrouted.kicad_pcb",
        pro=pcb_folder / f"{PCBENCH_STEM}.kicad_pro",
        dsn=dsn_folder / f"{PCBENCH_STEM}_unrouted.dsn",
        orp=dsn_folder / f"{PCBENCH_STEM}.orp",
    )


def _pcbench_iter() -> Iterable[BoardPaths]:
    pcb_root, _ = _pcbench_root_pair()
    if not pcb_root.is_dir():
        raise FileNotFoundError(
            f"PCBENCH_PCB_ROOT={pcb_root!s} does not exist"
        )
    for folder in sorted(p.name for p in pcb_root.iterdir() if p.is_dir()):
        yield _pcbench_board(folder)


# ---------------------------------------------------------------------------
# synth_2L_v2
# ---------------------------------------------------------------------------


def _synth2l_root_pair() -> tuple[Path, Path]:
    pcb = _env_root("SYNTH2L_PCB_ROOT")
    dsn = _env_root("SYNTH2L_DSN_ROOT")
    return pcb, dsn


def _synth2l_board(split: str, board_id: str) -> BoardPaths:
    pcb_root, dsn_root = _synth2l_root_pair()
    return BoardPaths(
        board_id=board_id,
        raw_pcb=pcb_root / split / f"{board_id}.kicad_pcb",
        pro=pcb_root / split / f"{board_id}.kicad_pro",
        dsn=dsn_root / split / f"{board_id}_unrouted.dsn",
        orp=dsn_root / split / f"{board_id}.orp",
    )


def _synth2l_iter(split: str) -> Iterable[BoardPaths]:
    pcb_root, _ = _synth2l_root_pair()
    split_dir = pcb_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"{split_dir} does not exist")
    for pcb in sorted(split_dir.glob("*.kicad_pcb")):
        yield _synth2l_board(split, pcb.stem)


# ---------------------------------------------------------------------------
# synth_1L grid<G>
# ---------------------------------------------------------------------------


def _synth1l_root_pair(grid: int) -> tuple[Path, Path]:
    pcb = _env_root(f"SYNTH1L_PCB_ROOT_{grid}")
    dsn = _env_root(f"SYNTH1L_DSN_ROOT_{grid}")
    return pcb, dsn


def _synth1l_board(grid: int, split: str, board_id: str) -> BoardPaths:
    pcb_root, dsn_root = _synth1l_root_pair(grid)
    return BoardPaths(
        board_id=board_id,
        raw_pcb=pcb_root / split / f"{board_id}.kicad_pcb",
        pro=pcb_root / split / f"{board_id}.kicad_pro",
        dsn=dsn_root / split / f"{board_id}_unrouted.dsn",
        orp=dsn_root / split / f"{board_id}.orp",
    )


def _synth1l_iter(grid: int, split: str) -> Iterable[BoardPaths]:
    pcb_root, _ = _synth1l_root_pair(grid)
    split_dir = pcb_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"{split_dir} does not exist")
    for pcb in sorted(split_dir.glob("*.kicad_pcb")):
        yield _synth1l_board(grid, split, pcb.stem)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iter_boards(
    dataset: str,
    *,
    grid: int | None = None,
    split: str | None = None,
) -> Iterable[BoardPaths]:
    """Yield ``BoardPaths`` for every board in the dataset."""
    if dataset == "pcbench":
        yield from _pcbench_iter()
    elif dataset == "synthetic_2l":
        yield from _synth2l_iter(_require_split(split or "test"))
    elif dataset == "synthetic_1l":
        yield from _synth1l_iter(
            _require_grid(grid), _require_split(split or "test"),
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. Must be one of "
            f"{{pcbench, synthetic_1l, synthetic_2l}}."
        )


def select_boards(
    dataset: str,
    *,
    grid: int | None = None,
    split: str | None = None,
    limit: int | None = None,
    sample_substrings: Sequence[str] | None = None,
) -> list[BoardPaths]:
    """Apply ``--limit`` / ``--sample`` filters on top of ``iter_boards``."""
    boards = list(iter_boards(dataset, grid=grid, split=split))
    if sample_substrings:
        wanted = list(sample_substrings)
        boards = [
            b for b in boards if any(w in b.board_id for w in wanted)
        ]
    if limit is not None:
        boards = boards[: int(limit)]
    return boards


def algorithm_tag(
    baseline: str, dataset: str, *, grid: int | None = None,
) -> str:
    """Folder name to use under ``<output-root>/<dataset>/...``.

    The unified runner places its results at
    ``<output-root>/<dataset_tag>/<algorithm_tag>/seed<N>/...``. Both tags
    are stable strings so reruns overwrite predictably.
    """
    if dataset == "synthetic_1l":
        return f"{baseline}_grid{_require_grid(grid)}"
    return baseline


def dataset_tag(dataset: str, *, grid: int | None = None) -> str:
    """Top-level folder name for the dataset under ``<output-root>``."""
    if dataset == "synthetic_1l":
        return f"synth_1L_grid{_require_grid(grid)}_5net_v02_test"
    if dataset == "synthetic_2l":
        return "synth_2L_v2_test"
    if dataset == "pcbench":
        return "PCBench"
    raise ValueError(f"Unknown dataset: {dataset!r}")
