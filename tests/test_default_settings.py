"""Defaults that must compose (``configs/rl.yaml``): the default ``override_n_max_slots`` lifts
only a checkpoint without the slot embedding and is skipped (logged) for one with it; the
default ``eval_split`` is skipped (logged) on a single-board run, which has no split, and on a
``--boards-json`` without that split it fails naming the default and ``--no-eval-split``. An
override or split asked for explicitly still fails. End to end: ``tests/integration/
test_default_train_eval.py``."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

BOARD = Path(__file__).resolve().parent / "fixtures" / "simple_routing_board.kicad_pcb"


def _policy(*, disable_slot_emb: bool, n_max_slots: int = 512):
    return SimpleNamespace(tokenizer=SimpleNamespace(vocab=SimpleNamespace(
        n_max_slots=n_max_slots, disable_slot_emb=disable_slot_emb, slot_perm=False)))


class TestSlotOverride:
    def test_lifts_a_checkpoint_without_the_slot_embedding(self):
        from methods.rl_agent.rollout.transformer import apply_n_max_slots_override

        policy = _policy(disable_slot_emb=True)
        apply_n_max_slots_override(policy, 1280, explicit=False)
        assert policy.tokenizer.vocab.n_max_slots == 1280

    def test_default_override_is_skipped_for_a_slot_embedding_checkpoint(self, capsys):
        from methods.rl_agent.rollout.transformer import apply_n_max_slots_override

        policy = _policy(disable_slot_emb=False)
        apply_n_max_slots_override(policy, 1280, explicit=False)
        assert policy.tokenizer.vocab.n_max_slots == 512
        assert "default override 1280 skipped" in capsys.readouterr().out

    def test_explicit_override_of_a_slot_embedding_checkpoint_raises(self):
        from methods.rl_agent.rollout.transformer import apply_n_max_slots_override

        with pytest.raises(RuntimeError, match="disable_slot_emb=False"):
            apply_n_max_slots_override(_policy(disable_slot_emb=False), 1280, explicit=True)


class TestSingleBoardValidation:
    def _trainer(self, *argv: str):
        from methods.rl_agent.training.loop import PPOTrainer
        from methods.rl_agent.training.train_ppo import parse_train_args

        return PPOTrainer(parse_train_args(["--board", str(BOARD), *argv]))

    def test_default_eval_split_is_skipped_and_logged(self, capsys):
        trainer = self._trainer()
        assert trainer.args.eval_split is not None   # the default asks for validation
        trainer._setup_eval_boards()
        assert trainer.eval_boards is None
        assert "validation: off — single-board run" in capsys.readouterr().out

    def test_explicit_eval_split_raises(self):
        with pytest.raises(RuntimeError, match="--eval-split requires a multi-board"):
            self._trainer("--eval-split", "val")._setup_eval_boards()

    def test_async_val_on_a_single_board_raises(self):
        with pytest.raises(RuntimeError, match="--async-val requires a multi-board"):
            self._trainer("--async-val")._setup_eval_boards()

    def test_no_eval_split_turns_validation_off(self):
        trainer = self._trainer("--no-eval-split")
        assert trainer.args.eval_split is None
        trainer._setup_eval_boards()
        assert trainer.eval_boards is None


class TestSplitJsonWithoutVal:
    def _trainer(self, tmp_path, *argv: str):
        from methods.rl_agent.training.loop import PPOTrainer
        from methods.rl_agent.training.train_ppo import parse_train_args

        split = tmp_path / "split.json"
        split.write_text(json.dumps({"easy": {"train": ["b"], "test": ["b"]},
                                     "dataset_dirs": {"train": str(tmp_path), "test": str(tmp_path)}}))
        return PPOTrainer(parse_train_args(["--board", str(BOARD), "--boards-json", str(split),
                                            "--boards-order", "per_env_epoch", *argv]))

    def test_default_eval_split_names_its_source_and_the_off_switch(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"came from defaults .*--no-eval-split"):
            self._trainer(tmp_path)._setup_eval_boards()

    def test_explicit_eval_split_keeps_the_loader_error(self, tmp_path):
        with pytest.raises(ValueError, match=r"has no \[easy\]\[val\]"):
            self._trainer(tmp_path, "--eval-split", "val")._setup_eval_boards()

    def test_no_eval_split_trains_without_validation(self, tmp_path):
        trainer = self._trainer(tmp_path, "--no-eval-split")
        trainer._setup_eval_boards()
        assert trainer.eval_boards is None
