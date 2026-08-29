set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# W&B logging guard (see run_cadagent.sh): drop to console-only when
# WANDB_API_KEY is unset so the vendored Tracking's bare wandb.init() does
# not crash offline. Override with LOGGER="['console']" / WANDB_PROJECT=...
if [[ -z "${LOGGER:-}" ]]; then
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        LOGGER="['console','wandb']"
    else
        echo "[run_cadagent] WANDB_API_KEY unset -> console-only logging" >&2
        LOGGER="['console']"
    fi
fi
WANDB_PROJECT_NAME="${WANDB_PROJECT:-verl_agent_cadagent}"

# Multi-pin 2-layer synthetic experiment.
# Uses configs/datasets/misc/multi_pin_2layer_v2.json (10k train + 128 test + 128 val
# easy boards, 5-net 13-pad, 100x100mm, 2-layer). Per-split dataset dir is read
# from the JSON's top-level dataset_dirs[split] entry.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Boards-list scheduling — per_env_epoch gives each group a different board
# per training step (8 boards/step in our 8-group setup) with shuffled
# no-replacement order across the full 10k-board epoch. Falls back gracefully
# to round_robin / per_env_random / single via env override.
BOARDS_ORDER=${BOARDS_ORDER:-per_env_epoch}
BOARDS_JSON=${BOARDS_JSON:-"$REPO_ROOT/configs/datasets/misc/multi_pin_2layer_v2.json"}
BOARDS_DIFFICULTY=${BOARDS_DIFFICULTY:-easy}
BOARDS_SPLIT=${BOARDS_SPLIT:-train}

# Single-board fallback (used by env.cadagent.board_path init; round-robin
# overrides per episode). Keep pointing at the simple fixture.
BOARD_PATH=${BOARD_PATH:-"$REPO_ROOT/tests/fixtures/simple_routing_board.kicad_pcb"}

# Redirect Ray temp dir (session dirs + logs + object spill) to RayCache/
# one level above the repo root. Avoids polluting /tmp and keeps logs in a
# stable location across runs.
RAY_LOG_DIR="$(dirname "$REPO_ROOT")/RayCache"
mkdir -p "$RAY_LOG_DIR"
export RAY_TMPDIR="$RAY_LOG_DIR"

# GRPO with group_size=8: train_batch_size=8, val_batch_size=3, rollout.n=8
# => CadagentEnvs.num_processes = 8 * 8 = 64 workers
train_data_size=8
val_data_size=3
group_size=8

# KiCad router is CPU-bound; allocate 0.2 CPU per worker.
num_cpus_per_env_worker=0.2

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=192 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj] \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.stop='["</action>"]' \
    +actor_rollout_ref.rollout.include_stop_str_in_output=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=3 \
    env.env_name=cadagent \
    env.seed=0 \
    env.max_steps=30 \
    env.history_length=2 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.resources_per_worker.num_gpus=0 \
    env.cadagent.board_path=$BOARD_PATH \
    +env.cadagent.boards_order=$BOARDS_ORDER \
    +env.cadagent.boards_json=$BOARDS_JSON \
    +env.cadagent.boards_difficulty=$BOARDS_DIFFICULTY \
    +env.cadagent.boards_split=$BOARDS_SPLIT \
    +env.cadagent.val_boards_order=round_robin \
    +env.cadagent.val_boards_split=val \
    env.cadagent.masking_rule=strict \
    env.cadagent.reward_rule=grpo_final \
    env.cadagent.state_format=sexpr \
    env.cadagent.prompt_version=v5 \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${WANDB_PROJECT_NAME}" \
    trainer.experiment_name='grpo_qwen2.5_3b_multi_pin_2layer' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    +trainer.best_metric_key='val/fp_mean_of_means' \
    +trainer.best_metric_mode=max \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.log_val_generations=2 \
    trainer.val_before_train=False $@
