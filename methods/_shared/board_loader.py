"""Board identity + board-list loading for the eval pipeline."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PAD_RE = re.compile(r"\(pad\s")


@lru_cache(maxsize=None)
def count_pads(path: str) -> int:
    """Number of ``(pad ...`` tokens in a ``.kicad_pcb`` file — a cheap board
    size / difficulty proxy used to order boards small-first.

    Shared by the training board resolver (``resolve_board_list`` round_robin
    curriculum) and the eval rollout scheduler. Cached per path so repeated
    scheduling passes (e.g. per-eval validation) don't re-read the file.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return len(_PAD_RE.findall(text))


@dataclass
class BoardSpec:
    """Identity + on-disk location for one board."""
    index: int
    board_id: str
    path: str


def load_boards_from_dir_or_list(
    *,
    boards_dir: Path | None = None,
    boards_list: Path | None = None,
) -> list[BoardSpec]:
    """Load a list of boards from either a directory or a manifest file.

    The directory variant accepts two layouts:
      * flat:   ``<dir>/*.kicad_pcb``
      * nested: ``<dir>/<board>/processed_v9_guide_v3.kicad_pcb`` (real PCBs)

    The manifest variant reads one path per non-comment line; entries may name
    their location as ``${CADAGENT_DATA_ROOT}/...`` (resolved through
    ``configs.loader.paths``: env var wins, the paths.yaml default falls back)
    so the tracked lists carry no machine-specific absolute root.
    """
    paths: list[Path] = []
    if boards_dir is not None:
        flat = sorted(Path(boards_dir).glob("*.kicad_pcb"))
        nested = sorted(Path(boards_dir).glob("*/processed_v9_guide_v3.kicad_pcb"))
        if flat:
            paths = flat
        elif nested:
            paths = nested
        else:
            raise FileNotFoundError(
                f"No *.kicad_pcb or */processed_v9_guide_v3.kicad_pcb in {boards_dir}"
            )
    elif boards_list is not None:
        from configs.loader.paths import expand_data_path

        with open(boards_list) as f:
            paths = [
                Path(expand_data_path(line.strip())) for line in f
                if line.strip() and not line.startswith("#")
            ]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Board not found: {p}")
    else:
        raise ValueError("provide boards_dir or boards_list")

    boards: list[BoardSpec] = []
    for i, p in enumerate(paths):
        if p.name == "processed_v9_guide_v3.kicad_pcb":
            board_id = p.parent.name
        else:
            board_id = p.stem
        boards.append(BoardSpec(index=i, board_id=board_id, path=str(p.resolve())))
    return boards


def load_boards_from_split_json(
    split_json: str | Path,
    difficulty: str,
    split: str,
    *,
    dataset_dir: str | None = None,
    board_filename: str | None = None,
) -> list[BoardSpec]:
    """Load boards listed under ``data[difficulty][split]`` of ``split_json``.

    The split file must either have top-level ``dataset_dirs[split]``
    pointing at the .kicad_pcb directory, or the caller must supply
    ``dataset_dir``. Missing files are warned and skipped.

    Board-file layout mirrors :func:`resolve_board_list`: with a
    ``board_filename`` (arg wins over a top-level ``"board_filename"`` key in
    the split json), boards live at ``<ds>/<bid>/<board_filename>`` (real
    per-board-dir sets, e.g. d3); otherwise the flat ``<ds>/<bid>.kicad_pcb``.
    """
    with open(split_json) as f:
        data = json.load(f)
    if difficulty not in data or split not in data[difficulty]:
        raise ValueError(
            f"{split_json} has no [{difficulty}][{split}] "
            f"(got {list(data.keys())})"
        )
    if dataset_dir is not None:
        ds = dataset_dir
    else:
        dataset_dirs = data.get("dataset_dirs") or {}
        if split not in dataset_dirs:
            raise ValueError(
                f"{split_json} top-level 'dataset_dirs' missing entry for "
                f"split={split!r} (have: {sorted(dataset_dirs)})."
            )
        ds = dataset_dirs[split]
    from configs.loader.paths import expand_data_path

    ds = expand_data_path(ds)
    bn = board_filename if board_filename is not None else data.get("board_filename")
    boards: list[BoardSpec] = []
    next_index = 0
    for bid in data[difficulty][split]:
        path = os.path.join(ds, bid, bn) if bn else os.path.join(ds, f"{bid}.kicad_pcb")
        if not os.path.exists(path):
            print(f"[eval] WARN: missing {path} — skipping")
            continue
        boards.append(BoardSpec(index=next_index, board_id=str(bid), path=path))
        next_index += 1
    if not boards:
        raise RuntimeError(f"No boards resolved for {difficulty}/{split}")
    return boards


# ---------------------------------------------------------------------------
# Training board-list resolution (curriculum / multi-board scheduling)
# ---------------------------------------------------------------------------
# Shared by both training branches (RL trainer + LLM BoardScheduler); lives
# here in the neutral board-loading home rather than under training/rl so the
# LLM scheduler need not reach into the RL package.

def resolve_board_list(
    *,
    boards_order: str,
    single_board: str,
    boards_json: str | None,
    difficulty: str,
    split: str,
    dataset_dir: str | None = None,
    board_filename: str | None = None,
) -> tuple[list[str], list[int]]:
    """Resolve the list of boards for training.

    ``single`` mode: returns ``([single_board], [0])`` (pad count unused).

    Multi-board modes: reads ``boards_json``, resolves the per-split
    dataset directory from ``dataset_dir`` when provided or from the JSON's
    top-level ``dataset_dirs[split]`` otherwise, then resolves each board id
    to an on-disk path. Two on-disk layouts are supported:

      - flat (default): ``<ds>/<bid>.kicad_pcb`` — used by the synthetic
        datasets where ``bid`` already names the file stem.
      - per-board-dir: ``<ds>/<bid>/<board_filename>`` — used by datasets
        like the real PCB d3 set where ``bid`` names a directory holding
        several variants. Enabled by either the ``board_filename`` argument
        or a top-level ``"board_filename"`` key in ``boards_json`` (the
        argument takes precedence).

    In every multi-board mode a board listed in the split file but absent on
    disk is warned about and skipped; an empty result raises. In
    ``round_robin`` mode the resolved paths are additionally sorted by total
    pad count (gentle curriculum).
    """
    if boards_order == "single":
        return [single_board], [0]
    if boards_order not in ("round_robin", "per_env_random", "per_env_epoch"):
        raise ValueError(f"Unknown boards_order: {boards_order}")
    if not boards_json:
        raise ValueError(
            f"boards_order={boards_order!r} requires an explicit boards_json "
            "(there is no default — specify a split from the "
            "configs/datasets/ registry)."
        )

    import json
    import os

    with open(boards_json) as f:
        split_data = json.load(f)
    if difficulty not in split_data:
        raise ValueError(f"difficulty={difficulty} not in {boards_json}")
    if split not in split_data[difficulty]:
        raise ValueError(
            f"split={split} not in {boards_json}[{difficulty}] "
            f"(available: {list(split_data[difficulty].keys())})"
        )
    board_ids: list[str] = split_data[difficulty][split]
    if dataset_dir is not None:
        ds = dataset_dir
    else:
        dataset_dirs = split_data.get("dataset_dirs") or {}
        if split not in dataset_dirs:
            raise ValueError(
                f"{boards_json} top-level 'dataset_dirs' missing entry for "
                f"split={split!r} (have: {sorted(dataset_dirs)}). "
                f"Add 'dataset_dirs[{split!r}]' pointing at the directory "
                f"that holds <board_id>.kicad_pcb files."
            )
        ds = dataset_dirs[split]
    from configs.loader.paths import expand_data_path

    ds = expand_data_path(ds)

    # board_filename arg wins over boards_json metadata; either enables the
    # per-board-dir layout. None (default) keeps the flat <bid>.kicad_pcb
    # layout so synth datasets stay byte-identical to the legacy behavior.
    bn = board_filename if board_filename is not None else split_data.get("board_filename")

    def _board_path(bid: str) -> str:
        if bn:
            return os.path.join(ds, bid, bn)
        return os.path.join(ds, f"{bid}.kicad_pcb")

    if boards_order in ("per_env_epoch", "per_env_random"):
        # Existence check only — no *pad* pre-scan (that is round_robin's
        # curriculum sort, which opens every board). Missing boards are warned
        # and skipped exactly as round_robin and the eval loader do: without
        # it a split file that does not match what is on disk surfaced only
        # mid-training, as a raw engine "failed to load board" crash naming no
        # split file. One stat per board costs ~5s cold / ~0s warm for a
        # 10k-board split on NFS. The warnings are head-capped because a
        # wholly-absent 10k split would otherwise print 10k lines.
        head = 5
        paths: list[str] = []
        missing: list[str] = []
        for bid in board_ids:
            path = _board_path(bid)
            (paths if os.path.exists(path) else missing).append(path)
        for path in missing[:head]:
            print(f"[boards] WARN: missing {path} — skipping")
        if len(missing) > head:
            print(f"[boards] WARN: ... ({len(missing) - head} more missing "
                  "boards elided)")
        if not paths:
            raise RuntimeError(
                f"No usable boards in {boards_json}[{difficulty}][{split}] — "
                f"none of its {len(missing)} boards exist under {ds}"
            )
        pads = [0] * len(paths)
        skipped = f", {len(missing)} missing skipped" if missing else ""
        print(f"[boards] {boards_order} over {len(paths)} boards"
              f"{skipped} (no pad pre-scan)")
        return paths, pads

    resolved: list[tuple[str, int]] = []
    for bid in board_ids:
        path = _board_path(bid)
        if not os.path.exists(path):
            print(f"[boards] WARN: missing {path} — skipping")
            continue
        try:
            n_pads = count_pads(path)
        except Exception as e:  # noqa: BLE001
            print(f"[boards] WARN: failed to count pads in {path}: {e} — skipping")
            continue
        resolved.append((path, n_pads))

    if not resolved:
        raise RuntimeError(
            f"No usable boards in {boards_json}[{difficulty}][{split}]"
        )
    resolved.sort(key=lambda x: x[1])
    n_total = len(resolved)
    print(f"[boards] round_robin over {n_total} boards (ascending pads):")
    # Full listing only for small pools — a 100k-board train pool dumped one
    # line per board (~15MB launch log). Large pools show head/tail + elision.
    if n_total <= 50:
        idx = list(range(n_total))
    else:
        idx = [0, 1, 2, n_total - 3, n_total - 2, n_total - 1]
    prev = -1
    for i in idx:
        if i != prev + 1:
            print(f"  ... ({n_total - len(idx)} boards elided) ...")
        path, n = resolved[i]
        print(f"  [{i:2d}] pads={n:4d}  {path}")
        prev = i
    paths = [p for p, _ in resolved]
    pads = [n for _, n in resolved]
    return paths, pads
