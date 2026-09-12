"""Board selection for multi-board training — shared by RL and LLM.

Encapsulates the "which board(s) do I train on in iter N?" logic so neither
branch reimplements curriculum/sampling. Lives at the neutral ``training``
level (not under ``rl/`` or ``llm/``) because both the RL trainer and the LLM
``KiCadLLMVerlManager`` consume it.

Modes:

    - ``single``:          always the single board.
    - ``round_robin``:     rotate through the boards listed in ``boards_json``
                           (sorted ascending by pad count inside
                           :func:`methods._shared.board_loader.resolve_board_list`). One
                           board per iter, replicated across all groups.
    - ``per_env_random``:  every iter, draw ``count`` boards independently with
                           replacement. Each group gets its own board.
    - ``per_env_epoch``:   shuffle the full board list once, deal out ``count``
                           boards per iter without replacement; reshuffle on
                           exhaustion.

The core selection returns board *indices* (:meth:`next_indices`); the RL
trainer replicates them by its GRPO group factor and maps to paths itself,
while LLM uses :meth:`next_paths` (indices mapped to paths). Both share one
RNG/epoch implementation, so the curriculum logic is defined exactly once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BoardSchedulerConfig:
    """Config bag for :class:`BoardScheduler`."""

    mode: str = "single"
    single_board: Optional[str] = None
    boards_json: Optional[str] = None   # required for multi-board modes (no default)
    difficulty: str = "easy"
    split: str = "train"
    seed: int = 0  # shuffle/sample seed for per_env_random / per_env_epoch
    dataset_dir: Optional[str] = None      # override boards_json dataset_dirs[split]
    board_filename: Optional[str] = None   # per-board-dir layout (real PCB sets)


class BoardScheduler:
    """Stateful iterator over training boards (index- and path-valued).

    A single iteration's ``round_robin`` pick is one board replicated across
    all groups (within-iter GRPO comparisons stay meaningful); ``per_env_*``
    deal one board per group.
    """

    _SUPPORTED_MODES = ("single", "round_robin", "per_env_random", "per_env_epoch")

    def __init__(self, cfg: BoardSchedulerConfig) -> None:
        if cfg.mode not in self._SUPPORTED_MODES:
            raise NotImplementedError(
                f"BoardScheduler mode {cfg.mode!r} not implemented. "
                f"Supported: {self._SUPPORTED_MODES}"
            )

        from methods._shared.board_loader import resolve_board_list

        self._cfg = cfg
        # ``single`` / ``round_robin`` resolve with the full pad pre-scan
        # (round_robin needs the ascending-pads order). The per_env_* modes
        # pass through to the loader's no-pre-scan fast path: sampling
        # shuffles indices so order is irrelevant, and a 100k-board pool
        # costs minutes of NFS reads per launch when scanned up front —
        # consumers that want pad counts call the lru-cached ``count_pads``
        # lazily on the boards actually used.
        self._paths, self._pads = resolve_board_list(
            boards_order=cfg.mode,
            single_board=cfg.single_board or "",
            boards_json=cfg.boards_json,
            difficulty=cfg.difficulty,
            split=cfg.split,
            dataset_dir=cfg.dataset_dir,
            board_filename=cfg.board_filename,
        )
        if not self._paths:
            raise RuntimeError(f"BoardScheduler resolved 0 boards for {cfg!r}")

        # Incremented on every ``next_indices`` call; first call → iter 0.
        self._iter_idx = -1
        self._rng = random.Random(cfg.seed)
        self._epoch_order: List[int] = []
        self._epoch_pos: int = 0
        self._epochs_completed: int = 0

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def next_indices(self, count: int) -> List[int]:
        """Advance the iter counter and return ``count`` board indices.

        - ``single`` / ``round_robin``: one index, replicated ``count`` times.
        - ``per_env_random``: ``count`` independent draws (with replacement).
        - ``per_env_epoch``: ``count`` from a shuffled epoch order (no
          replacement); reshuffled (and ``epochs_completed`` bumped) when
          exhausted.
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")

        self._iter_idx += 1
        n_boards = len(self._paths)

        if self._cfg.mode == "single":
            return [0] * count

        if self._cfg.mode == "round_robin":
            return [self._iter_idx % n_boards] * count

        if self._cfg.mode == "per_env_random":
            return [self._rng.randrange(n_boards) for _ in range(count)]

        if self._cfg.mode == "per_env_epoch":
            out: List[int] = []
            for _ in range(count):
                if self._epoch_pos >= len(self._epoch_order):
                    self._epoch_order = list(range(n_boards))
                    self._rng.shuffle(self._epoch_order)
                    self._epoch_pos = 0
                    self._epochs_completed += 1
                out.append(self._epoch_order[self._epoch_pos])
                self._epoch_pos += 1
            return out

        raise NotImplementedError(self._cfg.mode)  # pragma: no cover

    def next_paths(self, env_num: int) -> List[str]:
        """``next_indices`` mapped to board paths (one per group)."""
        return [self._paths[i] for i in self.next_indices(env_num)]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._cfg.mode

    @property
    def paths(self) -> List[str]:
        """Full list of candidate board paths (read-only copy)."""
        return list(self._paths)

    @property
    def pads(self) -> List[int]:
        """Per-board pad counts aligned with :attr:`paths` (read-only copy)."""
        return list(self._pads)

    @property
    def iter_idx(self) -> int:
        """Most recent iter index from ``next_indices`` (-1 before any call)."""
        return self._iter_idx

    @property
    def num_boards(self) -> int:
        return len(self._paths)

    @property
    def epochs_completed(self) -> int:
        """Number of full epoch reshuffles so far (per_env_epoch only)."""
        return self._epochs_completed

    @property
    def epoch_remaining(self) -> int:
        """Boards left in the current shuffled epoch (per_env_epoch only)."""
        return max(0, len(self._epoch_order) - self._epoch_pos)

    # ------------------------------------------------------------------
    # Checkpoint hooks
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "iter_idx": self._iter_idx,
            "rng_state": self._rng.getstate(),
            "epoch_order": list(self._epoch_order),
            "epoch_pos": self._epoch_pos,
            "epochs_completed": self._epochs_completed,
        }

    def load_state_dict(self, d: dict) -> None:
        self._iter_idx = int(d["iter_idx"])
        if "rng_state" in d:
            self._rng.setstate(tuple(d["rng_state"]))
        if "epoch_order" in d:
            self._epoch_order = list(d["epoch_order"])
        if "epoch_pos" in d:
            self._epoch_pos = int(d["epoch_pos"])
        if "epochs_completed" in d:
            self._epochs_completed = int(d["epochs_completed"])


__all__ = ["BoardScheduler", "BoardSchedulerConfig"]
