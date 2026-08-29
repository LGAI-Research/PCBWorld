"""Golden tests for configs.loader.schema.LLMTrainConfig.

LLMTrainConfig extracts the cadagent-specific slice of the verl Hydra config
(the ``env.cadagent.*`` block + the ``env.*`` knobs cadagent reads). These
tests pin from_verl_config / to_env_kwargs behaviour. Imports are
function-local (the repo's eval/* import at collection time can tip a
native double-init segfault).
"""
from __future__ import annotations


def test_defaults_match_yaml():
    """LLMTrainConfig() defaults = configs/defaults/llm_train.yaml."""
    from configs.loader.schema import LLMTrainConfig

    d = LLMTrainConfig()
    assert (d.max_steps, d.masking_rule, d.reward_rule, d.state_format) == (
        200, "strict", "default", "sexpr")
    assert (d.corner_mode, d.via_penalty, d.reward_noise_std, d.emit_drc_tokens) == (
        0, None, 0.0, True)
    assert (d.prompt_version, d.history_length) == ("v1", 0)
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
    assert d.eval_check_angle == rl_eval.check_angle


def test_best_metric_key_shared_canonical():
    """Best-checkpoint criterion is one shared schema value (RLEvalConfig).

    RL (PPO+GRPO on_validation) and LLM (verl trainer.best_metric_key in
    run_cadagent*.sh) both select the ckpt maximizing this val tag.
    """
    from configs.loader.schema import RLEvalConfig

    e = RLEvalConfig()
    assert e.best_metric_key == "val/fp_mean_of_means"
    assert e.best_metric_mode == "max"


def test_to_env_kwargs_surface():
    """to_env_kwargs is exactly the PCBWorld keyword surface (8 keys)."""
    from configs.loader.schema import LLMTrainConfig

    kw = LLMTrainConfig().to_env_kwargs()
    assert set(kw) == {
        "max_steps", "masking_rule", "reward_rule", "state_format",
        "corner_mode", "via_penalty", "reward_noise_std", "emit_drc_tokens",
    }
    # board scheduling / prompt / scoring knobs do NOT leak into env_kwargs.
    for leaked in ("boards_order", "boards_json", "prompt_version",
                   "score_train_episodes", "guided_decoding_grammar",
                   "val_boards_order", "seed", "board_path", "history_length"):
        assert leaked not in kw


def test_from_verl_config_reads_env_and_cadagent():
    """env-core/board/prompt from env.cadagent.*; max_steps/history/seed from env.*."""
    from configs.loader.schema import LLMTrainConfig

    # Plain dicts mimic omegaconf's .get(key, default) surface (incl. nesting).
    cfg = {"env": {
        "max_steps": 30, "seed": 7, "history_length": 2,
        "cadagent": {
            "board_path": "/b.kicad_pcb", "reward_rule": "grpo_final",
            "masking_rule": "relaxed", "state_format": "xml", "corner_mode": 2,
            "via_penalty": 0.5, "reward_noise_std": 0.1, "emit_drc_tokens": False,
            "prompt_version": "v4", "boards_order": "round_robin",
            "boards_json": "/data/split.json", "boards_difficulty": "hard",
            "boards_split": "train_small", "val_boards_order": "per_env_epoch",
            "val_boards_split": "test", "score_train_episodes": True,
            "guided_decoding_grammar": "cadagent_v1",
            "eval_reward_config": "drc_sparse", "eval_check_angle": 90,
        },
    }}
    lt = LLMTrainConfig.from_verl_config(cfg)
    assert (lt.eval_reward_config, lt.eval_check_angle) == ("drc_sparse", 90)
    assert (lt.max_steps, lt.seed, lt.history_length) == (30, 7, 2)
    assert (lt.board_path, lt.reward_rule, lt.masking_rule) == (
        "/b.kicad_pcb", "grpo_final", "relaxed")
    assert (lt.state_format, lt.corner_mode, lt.via_penalty) == ("xml", 2, 0.5)
    assert (lt.emit_drc_tokens, lt.prompt_version) == (False, "v4")
    assert (lt.boards_order, lt.boards_difficulty, lt.boards_split) == (
        "round_robin", "hard", "train_small")
    assert (lt.val_boards_order, lt.resolved_val_boards_split) == (
        "per_env_epoch", "test")
    assert (lt.score_train_episodes, lt.guided_decoding_grammar) == (
        True, "cadagent_v1")
    assert lt.to_env_kwargs() == {
        "max_steps": 30, "masking_rule": "relaxed", "reward_rule": "grpo_final",
        "state_format": "xml", "corner_mode": 2, "via_penalty": 0.5,
        "reward_noise_std": 0.1, "emit_drc_tokens": False,
    }


def test_from_verl_config_defaults_on_missing():
    """Empty / partial config → every key falls back to the YAML default, and
    val_boards_split inherits the (overridden) boards_split."""
    from configs.loader.schema import LLMTrainConfig

    assert LLMTrainConfig.from_verl_config({}).to_env_kwargs() == (
        LLMTrainConfig().to_env_kwargs())
    # boards_split override with val_boards_split unset -> resolved inherits it.
    lt = LLMTrainConfig.from_verl_config(
        {"env": {"cadagent": {"boards_split": "test"}}})
    assert lt.val_boards_split is None
    assert lt.resolved_val_boards_split == "test"
