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
bash external/patcher.sh verl-agent   # verl-agent-patch/ -> verl-agent/
bash external/patcher.sh all          # every overlay
```

**Initialise the submodule first.** The script refuses to run when the target is
not a checkout — a fresh clone leaves an *empty* directory at each submodule
mount point, and copying into it makes the later `git submodule update --init`
fail on untracked files. Re-run the patcher after updating a submodule.

### `verl-agent-patch/` file list

| File | Purpose |
|---|---|
| `agent_system/environments/env_manager.py` | registers the PCBWorld env with the env manager |
| `agent_system/multi_turn_rollout/rollout_loop.py` | multi-turn rollout loop adaptation |
| `examples/run_pcbworld.sh` | GRPO training launcher (synthetic 2-layer) |
| `examples/run_pcbworld_multi_pin_2layer.sh` | GRPO training launcher (multi-pin 2-layer) |
| `scripts/model_merger.py` | FSDP shard -> HF checkpoint merge |
| `verl/trainer/config/ppo_trainer.yaml` | PPO trainer config |
| `verl/trainer/main_ppo.py` | trainer entry adaptation |
| `verl/trainer/ppo/metric_utils.py` | metric plumbing |
| `verl/trainer/ppo/ray_trainer.py` | Ray trainer adaptation |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | vLLM rollout adaptation |

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
bash examples/run_pcbworld.sh                    # GRPO, synthetic 2-layer
bash examples/run_pcbworld_multi_pin_2layer.sh   # GRPO, multi-pin 2-layer
```

Both default to console-only logging and switch to console+W&B when
`WANDB_API_KEY` is set; override with `LOGGER="['console']"`.

---

## OrthoRoute

Upstream source of the OrthoRoute router, kept for the KDD-era baseline record (its limits:
[methods/baselines/rule_based/README.md](../methods/baselines/rule_based/README.md)). This repo ships setup
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
not. It does **not** install: run the editable install above by hand, or let
[methods/baselines/rule_based/scripts/setup_env.sh](../methods/baselines/rule_based/scripts/setup_env.sh)
perform it (it skips OrthoRoute if `external/OrthoRoute/main.py` is absent).

---

## KiCadRoutingTools (KRT)

Upstream source for the KRT rule-based router baseline. It is a plain checkout,
not a submodule: [tools/setup/fetch_baselines.sh](../tools/setup/fetch_baselines.sh)
clones it to `external/KiCadRoutingTools` and checks out the pinned commit
`561f158`, the KRT the KDD and CADET papers used; the rule-based runner passes the
routing hyperparameters explicitly.
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
