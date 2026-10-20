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

For pcbench, ``difficulty`` (``easy``/``medium``/``hard`` = D3-A/B/C) selects
the boards listed under ``configs/datasets/d3.json[difficulty][split]``. Walking
every folder under ``PCBENCH_PCB_ROOT`` instead (the whole corpus, which also
holds the boards excluded from D3) needs the explicit ``all_corpus=True``; one
of the two is required.

All envs accept absolute paths. When a specific var is unset, the location
falls back to the named logical dataset in ``configs/paths.yaml`` (resolved
through ``configs.loader.paths``, i.e. under the repo-wide data root); with
an empty data root, resolution raises with a message naming
``PCBWORLD_DATA_ROOT``.
"""

from __future__ import annotations

import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Root resolution — the specific env var wins; otherwise the location comes
# from the logical dataset registered in configs/paths.yaml (env
# PCBWORLD_DATA_ROOT overrides the root, the yaml default falls back). No
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
D3_SPLIT_JSON = Path(__file__).resolve().parents[4] / "configs" / "datasets" / "d3.json"
D3_DIFFICULTIES = ("easy", "medium", "hard")
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


def _pcbench_roots() -> tuple[Path, Path]:
    """Both pcbench roots, resolved once per selection; the PCB root must exist."""
    pcb_root, dsn_root = _pcbench_root_pair()
    if not pcb_root.is_dir():
        raise FileNotFoundError(
            f"PCBENCH_PCB_ROOT={pcb_root!s} does not exist"
        )
    return pcb_root, dsn_root


def _pcbench_board(folder: str, pcb_root: Path, dsn_root: Path) -> BoardPaths:
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
    pcb_root, dsn_root = _pcbench_roots()
    for folder in sorted(p.name for p in pcb_root.iterdir() if p.is_dir()):
        yield _pcbench_board(folder, pcb_root, dsn_root)


@functools.lru_cache(maxsize=None)
def _d3_split() -> dict:
    try:
        return json.loads(D3_SPLIT_JSON.read_text())
    except json.JSONDecodeError as exc:  # not a selection error: keep it out of the CLI's ValueError path
        raise RuntimeError(f"{D3_SPLIT_JSON} is not valid JSON: {exc}") from exc


def d3_board_ids(difficulty: str, split: str) -> list[str]:
    """Board folder names of one D3 tier, in d3.json order."""
    if difficulty not in D3_DIFFICULTIES:
        raise ValueError(
            f"--difficulty must be one of {D3_DIFFICULTIES}, got {difficulty!r}"
        )
    tier = _d3_split()[difficulty]
    if split not in tier:
        raise ValueError(
            f"{D3_SPLIT_JSON} has no {difficulty}.{split} "
            f"(splits: {sorted(tier)})"
        )
    return list(tier[split])


def _pcbench_tier_iter(difficulty: str, split: str) -> Iterable[BoardPaths]:
    folders = d3_board_ids(difficulty, split)
    pcb_root, dsn_root = _pcbench_roots()
    for folder in folders:
        yield _pcbench_board(folder, pcb_root, dsn_root)


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
    difficulty: str | None = None,
    all_corpus: bool = False,
) -> Iterable[BoardPaths]:
    """Yield ``BoardPaths`` for every board in the dataset.

    pcbench needs exactly one of ``difficulty`` (one D3 tier of
    ``configs/datasets/d3.json``; ``split`` defaults to ``test``) and
    ``all_corpus`` (every corpus folder, D3 or not; it has no split).
    """
    check_pcbench_selection(dataset, difficulty, all_corpus)
    if dataset == "pcbench":
        if difficulty is not None:
            yield from _pcbench_tier_iter(difficulty, split or "test")
        else:
            if split not in (None, "test"):
                raise ValueError(
                    f"--split {split} selects within a D3 tier (--difficulty); "
                    "--all-corpus walks every corpus folder"
                )
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


def check_pcbench_selection(
    dataset: str, difficulty: str | None, all_corpus: bool,
) -> None:
    """Raise unless the D3 selection flags fit the dataset."""
    if dataset != "pcbench":
        if difficulty is not None or all_corpus:
            raise ValueError(
                "--difficulty / --all-corpus select PCBench boards and are "
                f"valid only with --dataset pcbench, got --dataset {dataset!r}"
            )
        return
    if difficulty is not None and all_corpus:
        raise ValueError("--difficulty and --all-corpus are mutually exclusive")
    if difficulty is None and not all_corpus:
        raise ValueError(
            "--dataset pcbench needs --difficulty easy|medium|hard (D3-A/B/C of "
            "configs/datasets/d3.json); --all-corpus routes every corpus "
            "folder instead, including the boards excluded from D3"
        )


def missing_inputs(board: BoardPaths, baseline: str) -> list[str]:
    """``<input>=<path>`` for each file ``baseline`` needs for ``board`` that is not on disk."""
    sibling = "orp" if baseline == "orthoroute" else "dsn"
    return [f"{name}={getattr(board, name)}" for name in ("raw_pcb", "pro", sibling)
            if not getattr(board, name).is_file()]


def partition_by_inputs(
    boards: Sequence[BoardPaths], baseline: str,
) -> tuple[list[BoardPaths], list[tuple[str, str]]]:
    """Split ``boards`` into (complete, [(board_id, missing paths), ...])."""
    valid: list[BoardPaths] = []
    missing: list[tuple[str, str]] = []
    for b in boards:
        gaps = missing_inputs(b, baseline)
        if gaps:
            missing.append((b.board_id, ", ".join(gaps)))
        else:
            valid.append(b)
    return valid, missing


def describe_missing(missing: Sequence[tuple[str, str]], head: int = 10) -> str:
    """``board: why`` lines, the first ``head`` of them plus a count of the rest."""
    lines = [f"  {bid}: {why}" for bid, why in missing[:head]]
    if len(missing) > head:
        lines.append(f"  ... and {len(missing) - head} more")
    return "\n".join(lines)


def select_boards(
    dataset: str,
    *,
    grid: int | None = None,
    split: str | None = None,
    limit: int | None = None,
    sample_substrings: Sequence[str] | None = None,
    difficulty: str | None = None,
    all_corpus: bool = False,
) -> list[BoardPaths]:
    """Apply ``--limit`` / ``--sample`` filters on top of ``iter_boards``.

    A ``--sample`` substring that matches no board of the selection (e.g. a
    corpus board outside the chosen D3 tier) is an error, not a silent drop,
    and so is a ``--limit`` below 1.
    """
    if limit is not None and int(limit) < 1:
        raise ValueError(f"--limit must be >= 1 (omit it for the whole selection), got {limit}")
    boards = list(iter_boards(dataset, grid=grid, split=split,
                              difficulty=difficulty, all_corpus=all_corpus))
    if sample_substrings:
        wanted = list(sample_substrings)
        unmatched = [w for w in wanted if not any(w in b.board_id for b in boards)]
        if unmatched:
            where = (f"D3 tier {difficulty}.{split or 'test'}" if difficulty is not None
                     else f"--dataset {dataset}")
            raise ValueError(
                f"--sample matched no board of {where}: {unmatched[:10]}"
                f"{' ...' if len(unmatched) > 10 else ''}"
            )
        boards = [
            b for b in boards if any(w in b.board_id for w in wanted)
        ]
    if limit is not None:
        boards = boards[: int(limit)]
    return boards


def select_boards_with_inputs(
    dataset: str, baseline: str, **selection,
) -> tuple[list[BoardPaths], list[BoardPaths], list[tuple[str, str]]]:
    """The runners' board selection: ``select_boards(dataset, **selection)`` split by inputs.

    Returns (selected, complete, [(board_id, missing inputs), ...]). When the
    selection is a whole D3 tier (however ``--limit`` / ``--sample`` were
    given), a board lacking an input ``baseline`` needs is an error instead.
    """
    boards = select_boards(dataset, **selection)
    valid, missing = partition_by_inputs(boards, baseline)
    difficulty = selection.get("difficulty")
    if missing and difficulty is not None:
        split = selection.get("split") or "test"
        if len(boards) == len(d3_board_ids(difficulty, split)):
            raise ValueError(
                f"D3 tier {difficulty}.{split} is incomplete: "
                f"{len(missing)} of {len(boards)} board(s) lack inputs for "
                f"{baseline}:\n{describe_missing(missing)}"
            )
    return boards, valid, missing


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


def dataset_tag(
    dataset: str,
    *,
    grid: int | None = None,
    difficulty: str | None = None,
    split: str | None = None,
    all_corpus: bool = False,
) -> str:
    """Top-level folder name for the dataset under ``<output-root>``."""
    if dataset == "synthetic_1l":
        return f"synth_1L_grid{_require_grid(grid)}_5net_v02_test"
    if dataset == "synthetic_2l":
        return "synth_2L_v2_test"
    if dataset == "pcbench":
        check_pcbench_selection(dataset, difficulty, all_corpus)
        if difficulty is not None:
            return f"PCBench_{difficulty}_{split or 'test'}"
        return "PCBench_corpus"
    raise ValueError(f"Unknown dataset: {dataset!r}")
