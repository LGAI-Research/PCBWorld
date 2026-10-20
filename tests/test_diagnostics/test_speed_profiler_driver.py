"""The speed profiler's trainer argv parses with the real training parser."""

from __future__ import annotations

from tools.diagnostics.speed_profiler.driver import ProfileConfig, trainer_args


def test_trainer_argv_parses_with_the_training_parser(tmp_path) -> None:
    """Every flag ``build_trainer`` passes must be a separate argv token the PPO parser
    accepts — a joined ``"--flag value"`` token is an unrecognized argument."""
    args = trainer_args(ProfileConfig(), str(tmp_path))
    assert args.rules_source == "board_or_yaml"
    assert args.drc_config_path == "experiments/kdd/configs/drc/synth_2L_v2.yaml"
