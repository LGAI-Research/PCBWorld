"""Golden tests for configs.loader.schema.LLMTrainConfig.

LLMTrainConfig extracts the pcbworld-specific slice of the verl Hydra config
(the ``env.pcbworld.*`` block + the ``env.*`` knobs pcbworld reads). These
tests pin from_verl_config behaviour and the EnvConfig it carries. Imports are
function-local (the repo's eval/* import at collection time can tip a
native double-init segfault).
"""
from __future__ import annotations


def test_defaults_match_yaml():
    """LLMTrainConfig() defaults = configs/llm.yaml (llm_train:)."""
    from configs.loader.schema import EnvConfig, LLMTrainConfig

    d = LLMTrainConfig()
    assert (d.env.max_steps, d.env.masking_rule, d.env.reward_rule, d.state_format) == (
        200, "pcbworld_masking", "default", "sexpr")
    # every other env field is the shared env: default (RL's)
    rl = EnvConfig().to_env_kwargs()
    assert {k: v for k, v in d.env.to_env_kwargs().items() if rl[k] != v} == {
        "max_steps": 200, "reward_rule": "default"}
    assert (d.prompt_version, d.history_length) == ("latest", 0)
    assert (d.boards_order, d.boards_difficulty, d.boards_split) == (
        "single", "easy", "train")
    assert (d.val_boards_order, d.val_boards_split) == ("single", None)
    assert (d.score_train_episodes, d.guided_decoding_grammar, d.seed) == (
        False, None, 0)
    assert d.board_path is None
    # val split inherits boards_split when unset
    assert d.resolved_val_boards_split == "train"
    # val scoring config is sourced from the shared eval config (RLEvalConfig)
    # so LLM val and the RL inline-eval path score with the same ruler.
    from configs.loader.schema import RLEvalConfig

    rl_eval = RLEvalConfig()
    assert d.eval_reward_config == rl_eval.reward_config
    assert d.eval_check_angle is None   # the env corner_mode (the manager resolves it)


def test_best_metric_key_shared_canonical():
    """Best-checkpoint criterion is one shared schema value (RLEvalConfig).

    RL (PPO+GRPO on_validation) and LLM (verl trainer.best_metric_key in
    run_pcbworld*.sh) both select the ckpt maximizing this val tag.
    """
    from configs.loader.schema import RLEvalConfig

    e = RLEvalConfig()
    assert e.best_metric_key == "val/fp_mean_of_means"
    assert e.best_metric_mode == "max"


def test_env_pcbworld_rejects_unknown_keys():
    """A typo or a key the schema does not know raises instead of being ignored."""
    import pytest

    from configs.loader.schema import LLMTrainConfig

    with pytest.raises(ValueError, match="unknown keys \\['masking_rul'\\]"):
        LLMTrainConfig.from_verl_config({"env": {"pcbworld": {"masking_rul": 1}}})


def test_env_pcbworld_rejects_the_env_block_keys():
    """max_steps / history_length / seed are read from env.* only — under env.pcbworld
    they would be ignored (seed, history_length) or shadowed, so they raise."""
    import pytest

    from configs.loader.schema import LLMTrainConfig

    for key in ("max_steps", "history_length", "seed"):
        with pytest.raises(ValueError, match=f"\\['{key}'\\] not allowed .* env\\.{key}"):
            LLMTrainConfig.from_verl_config({"env": {"pcbworld": {key: 1}}})


def test_from_verl_config_reads_env_and_pcbworld():
    """env-core/board/prompt from env.pcbworld.*; max_steps/history/seed from env.*."""
    from configs.loader.schema import LLMTrainConfig

    # Plain dicts mimic omegaconf's .get(key, default) surface (incl. nesting).
    cfg = {"env": {
        "max_steps": 30, "seed": 7, "history_length": 2,
        "pcbworld": {
            "board_path": "/b.kicad_pcb", "reward_rule": "grpo_final",
            "masking_rule": "relaxed", "state_format": "xml", "corner_mode": 90,
            "via_penalty": 0.5, "reward_noise_std": 0.1, "emit_drc_tokens": False,
            "prompt_version": "v4", "boards_order": "round_robin",
            "boards_json": "/data/split.json", "boards_difficulty": "hard",
            "boards_split": "train_small", "val_boards_order": "per_env_epoch",
            "val_boards_split": "test", "score_train_episodes": True,
            "guided_decoding_grammar": "pcbworld_v1",
            "eval_reward_config": "drc_sparse", "eval_check_angle": 90,
        },
    }}
    lt = LLMTrainConfig.from_verl_config(cfg)
    assert (lt.eval_reward_config, lt.eval_check_angle) == ("drc_sparse", 90)
    assert (lt.env.max_steps, lt.seed, lt.history_length) == (30, 7, 2)
    assert (lt.board_path, lt.env.reward_rule, lt.env.masking_rule) == (
        "/b.kicad_pcb", "grpo_final", "relaxed")
    assert (lt.state_format, lt.env.corner_mode, lt.env.reward.via_penalty) == ("xml", 90, 0.5)
    assert (lt.env.emit_drc_tokens, lt.prompt_version) == (False, "v4")
    assert (lt.boards_order, lt.boards_difficulty, lt.boards_split) == (
        "round_robin", "hard", "train_small")
    assert (lt.val_boards_order, lt.resolved_val_boards_split) == (
        "per_env_epoch", "test")
    assert (lt.score_train_episodes, lt.guided_decoding_grammar) == (
        True, "pcbworld_v1")
    assert lt.env.reward_noise_std == 0.1


def test_from_verl_config_defaults_on_missing():
    """Empty / partial config → every key falls back to the YAML default, and
    val_boards_split inherits the (overridden) boards_split."""
    from configs.loader.schema import LLMTrainConfig

    assert LLMTrainConfig.from_verl_config({}) == LLMTrainConfig()
    # boards_split override with val_boards_split unset -> resolved inherits it.
    lt = LLMTrainConfig.from_verl_config(
        {"env": {"pcbworld": {"boards_split": "test"}}})
    assert lt.val_boards_split is None
    assert lt.resolved_val_boards_split == "test"
