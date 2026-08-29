"""Per-env RNG stream must survive board reloads (advance_rng_on_reload).

A board hot-swap rebuilds the worker's KiCadRLWrapper, and the wrapper seeds
its numpy Generator once at construction — so without the reload counter the
whole per-env random stream (augmentation, slot_perm, auto net-select) rewinds
to its first draws on every reload. Training reloads once per iteration, which
would replay the same transforms for the entire run.

Each test spawns its own subprocess pool (multi-second), so this file stays
separate for pytest-xdist's loadfile scheduler.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from methods.rl_agent.wrappers.factory import make_decoder_env_pool
from tests.helpers.env_kwargs import full_env_kwargs


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BOARD = FIXTURES / "simple_routing_board.kicad_pcb"

N_RELOADS = 5


def _aug_signature(obs: dict) -> tuple:
    """The per-episode affine aug params the wrapper drew from its RNG."""
    aug = obs["_aug"]
    return (
        bool(aug["axis_swap"]),
        int(aug["flip_x"]),
        int(aug["flip_y"]),
        round(float(aug["nn_dx"]), 12),
        round(float(aug["nn_dy"]), 12),
    )


def _signatures_across_reloads(*, advance: bool) -> list[tuple]:
    pool = make_decoder_env_pool(
        str(BOARD), n_envs=1,
        **full_env_kwargs(
            max_steps=20,
            masking_rule="default_no_finish",
            reward_rule="drc_only_dense",
            # aug + the reload-advance switch are train-only knobs, so they
            # travel in the bundle rather than as defaulted parameters.
            train_extras={"aug_flip": True, "aug_rotate": True,
                          "aug_trans": True,
                          "advance_rng_on_reload": advance},
        ),
    )
    try:
        sigs = []
        for _ in range(N_RELOADS):
            # Mirrors the trainer: reload the board, then start an episode.
            pool.reload_boards([str(BOARD)])
            sigs.append(_aug_signature(pool.reset_all()[0]))
        return sigs
    finally:
        pool.close()


@pytest.mark.skipif(not BOARD.exists(), reason="fixture board missing")
def test_rng_advances_across_reloads():
    """advance_rng_on_reload=True: the stream keeps moving across rebuilds."""
    sigs = _signatures_across_reloads(advance=True)
    assert len(set(sigs)) > 1, (
        f"aug params identical across {N_RELOADS} reloads — the per-env RNG "
        f"was rewound to its seed on every rebuild: {sigs[0]}"
    )


@pytest.mark.skipif(not BOARD.exists(), reason="fixture board missing")
def test_rng_rewinds_when_not_advancing():
    """Default (eval pools): reload rewinds, so a board's draws are
    independent of how many boards the worker processed before it."""
    sigs = _signatures_across_reloads(advance=False)
    assert len(set(sigs)) == 1, (
        f"eval pools must stay reload-order independent, got {sigs}"
    )
