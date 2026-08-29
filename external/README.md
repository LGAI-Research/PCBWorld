# external/ — third-party frameworks and baselines

Upstream code this project builds on, kept out of the main tree. Nothing here is
vendored: each entry is a git submodule pinned to a public upstream, or a
checkout/binary fetched at setup time. What this repo *does* own are the two
`*-patch/` overlays, `patcher.sh`, and this file.

Only `README.md`, `patcher.sh` and `*-patch/` are tracked here. Everything else
is a submodule mount point or a setup-time download — in a fresh clone those
directories are empty (submodules) or missing (downloads) until you run
[tools/setup/setup_all.sh](../tools/setup/setup_all.sh), which does the
submodule init and the pinned baseline downloads in one pass.

| Entry | Kind | Upstream |
|---|---|---|
| `RAGEN/` | submodule (has nested submodules) | https://github.com/mll-lab-nu/RAGEN.git |
| `verl-agent/` | submodule | https://github.com/langfengQ/verl-agent.git |
| `OrthoRoute/` | submodule | https://github.com/bbenchoff/OrthoRoute.git |
| `KiCadRoutingTools/` | cloned at a pinned commit by `fetch_baselines.sh` (gitignored) | https://github.com/drandyhaas/KiCadRoutingTools |
| `freerouting/` | downloaded jar (gitignored) | https://github.com/freerouting/freerouting |

---

## Patcher

`patcher.sh` copies **every** file from a patch directory into the matching
submodule checkout, preserving directory structure (plain `cp` — it overwrites
the upstream file, it is not a `git apply` patch series; `__pycache__` and
`.DS_Store` are skipped). Run it from anywhere; the script resolves paths
relative to itself.

```bash
bash external/patcher.sh ragen        # RAGEN-patch/      -> RAGEN/
bash external/patcher.sh verl-agent   # verl-agent-patch/ -> verl-agent/
bash external/patcher.sh all          # both
```

**Initialise the submodule first.** The script refuses to run when the target is
not a checkout — a fresh clone leaves an *empty* directory at each submodule
mount point, and copying into it makes the later `git submodule update --init`
fail on untracked files. Re-run the patcher after updating a submodule.

### `RAGEN-patch/` file list

The RAGEN overlay is small: it does not add the PCBWorld env to RAGEN. It backports
LoRA support for the fsdp2 critic and records the matching config knob.

| File | Purpose |
|---|---|
| `config/base.yaml` | upstream training config plus the commented `exclude_modules` LoRA knob |
| `config/eval.yaml` | the same one-line addition to the eval config |
| `ragen/workers/fsdp_workers.py` | adds `build_peft_model` (LoRA via `LoraConfig`, incl. the critic value head) and the `get_shard_placement_fn` import it needs |

### `verl-agent-patch/` file list

| File | Purpose |
|---|---|
| `agent_system/environments/env_manager.py` | registers the PCBWorld env with the env manager |
| `agent_system/multi_turn_rollout/rollout_loop.py` | multi-turn rollout loop adaptation |
| `examples/run_cadagent.sh` | GRPO training launcher (synthetic 2-layer) |
| `examples/run_cadagent_multi_pin_2layer.sh` | GRPO training launcher (multi-pin 2-layer) |
| `scripts/model_merger.py` | FSDP shard -> HF checkpoint merge |
| `verl/trainer/config/ppo_trainer.yaml` | PPO trainer config |
| `verl/trainer/main_ppo.py` | trainer entry adaptation |
| `verl/trainer/ppo/metric_utils.py` | metric plumbing |
| `verl/trainer/ppo/ray_trainer.py` | Ray trainer adaptation |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | vLLM rollout adaptation |

---

## RAGEN

Upstream docs: [RAGEN/README.md](RAGEN/README.md) (present once the submodule is
checked out — the directory is empty in a fresh clone)

**RAGEN** (Reasoning AGENT) is a reinforcement-learning framework for training
multi-turn reasoning agents. It is built around the StarPO
(State-Thinking-Actions-Reward Policy Optimization) algorithm and ships ten
built-in environments behind a gym-compatible interface.

### Setup

RAGEN carries its own submodules — `verl` above all, which its code imports — so
this one needs `--recursive`. It also has its own installer, `scripts/setup_ragen.sh`
(see its README); the patcher only overlays our files on top of the checkout.

```bash
git submodule update --init --recursive external/RAGEN
bash external/patcher.sh ragen
```

### Basic usage

Upstream's own examples, run from the `external/RAGEN` directory — RAGEN drives
its built-in environments, not the PCBWorld env (that integration is
verl-agent's, below):

```bash
# training (no rollout filter — the default)
python train.py --config-name _2_sokoban

# training with SNR-adaptive filtering
python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=top_p \
  actor_rollout_ref.rollout.rollout_filter_value=0.9

# evaluation
python -m ragen.llm_agent.agent_proxy --config-name _2_sokoban
```

> Both filter keys live under `actor_rollout_ref.rollout` (`config/base.yaml`);
> upstream's README drops the `.rollout` from the first one, which Hydra rejects.

### Troubleshooting

**No CUDA toolkit, or `CUDA_HOME` unset.** Use this when CUDA is not installed
system-wide and you manage it inside the conda environment instead:

```bash
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit -y

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat >> $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh << 'EOF'
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF
```

**`lib64` symlink error.** A conda-installed CUDA puts its libraries in `lib/`,
but RAGEN looks for `lib64/`:

```bash
ln -s $CONDA_PREFIX/lib $CONDA_PREFIX/lib64
```

**FlashInfer cache error.** Clear the cache and re-run:

```bash
rm -rf ~/.cache/flashinfer/
```

---

## verl-agent

Upstream docs: [verl-agent/README.md](verl-agent/README.md) (present once the
submodule is checked out — the directory is empty in a fresh clone)

**verl-agent** extends [veRL](https://github.com/volcengine/verl) for
reinforcement-learning training of LLM agents. Its **step-independent
multi-turn rollout** mechanism lets the per-step input structure, history
handling, and memory module be customised independently, which is what makes
long-horizon multi-turn RL practical. It supports several RL algorithms,
including GiGPO (Group-in-Group Policy Optimization).

### Setup

```bash
git submodule update --init external/verl-agent
bash external/patcher.sh verl-agent
```

### Basic usage

Run from the `external/verl-agent` directory. Both launchers come from
`verl-agent-patch/` and are in place once the patcher has run. The patched
`verl.trainer.main_ppo` and `env_manager` import `methods.llm_agent.*` from this
repository, which is not installed as a package — put its root on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/this/repo:$PYTHONPATH
bash examples/run_cadagent.sh                    # GRPO, synthetic 2-layer
bash examples/run_cadagent_multi_pin_2layer.sh   # GRPO, multi-pin 2-layer
```

Both default to console-only logging and switch to console+W&B when
`WANDB_API_KEY` is set; override with `LOGGER="['console']"`.

---

## OrthoRoute

Upstream source for the GPU rule-based router baseline. This repo ships setup
plus a thin wrapper only — the source is not vendored. The runner
([methods/baselines/rule_based/README.md](../methods/baselines/rule_based/README.md))
invokes `external/OrthoRoute/main.py`.

### Setup

```bash
git submodule update --init external/OrthoRoute
pip install -e external/OrthoRoute
```

[tools/setup/fetch_baselines.sh](../tools/setup/fetch_baselines.sh) does the init
and checks the checkout is at the pinned commit (`f45dc68`), failing if it is
not. It does **not** install: it prints the editable install above as a next
step. [methods/baselines/rule_based/scripts/setup_env.sh](../methods/baselines/rule_based/scripts/setup_env.sh)
performs it (it skips OrthoRoute if `external/OrthoRoute/main.py` is absent).

---

## KiCadRoutingTools (KRT)

Upstream source for the KRT rule-based router baseline. It is a plain checkout,
not a submodule: [tools/setup/fetch_baselines.sh](../tools/setup/fetch_baselines.sh)
clones it to `external/KiCadRoutingTools` and checks out the pinned commit
`d9557ad1`, whose `route.py` defaults are the ones the paper reports.
`external/KiCadRoutingTools/` is gitignored, so it is absent in a fresh clone.

### Setup

```bash
bash tools/setup/fetch_baselines.sh
export KRT_ROOT="$PWD/external/KiCadRoutingTools"
pip install -e methods/baselines/rule_based/krt
```

`KRT_ROOT` is the one thing the rule-based bootstrap does not set for you — see
[methods/baselines/rule_based/README.md](../methods/baselines/rule_based/README.md).

---

## freerouting

Release JAR for the Freerouting rule-based baseline (`freerouting-2.1.0.jar`,
~66 MB). It is a binary rather than source, so it is fetched by download instead
of being tracked as a submodule; `external/freerouting/*.jar` is gitignored.

### Setup

[tools/setup/fetch_baselines.sh](../tools/setup/fetch_baselines.sh) downloads it
from the pinned release URL to `external/freerouting/freerouting-2.1.0.jar` when
that file is missing, then verifies its sha256 on every run. To fetch it
manually:

```bash
mkdir -p external/freerouting
curl -L -o external/freerouting/freerouting-2.1.0.jar \
    https://github.com/freerouting/freerouting/releases/download/v2.1.0/freerouting-2.1.0.jar
```
