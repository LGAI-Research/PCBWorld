---
date: 2026-05-06
purpose: How to run cadagent eval.metrics on routed PCBs and aggregate per-method comparison numbers (succ / DRV / WL / via / Φ). Designed for adding NEW methods (e.g. a new LLM) under the existing routed-PCB root, reusing the existing source `.kicad_pro` files.
audience: Someone who has just produced `.kicad_pcb` files for a new method and wants metric numbers consistent with the rest of the benchmark.
---

# eval.metrics aggregation — reusable scripts

## Where things live

`$CADAGENT_DATA_ROOT` is the dataset root (layout: `configs/paths.yaml`);
`$ROUTED_ROOT` is the routed-results tree these numbers aggregate over (a
benchmark result archive — not distributed with this repo).

```
# Source .kicad_pro files (defines the project's net-class DRC rules)
$CADAGENT_DATA_ROOT/synthetic/synth_2L_v2/test/<sample>.kicad_pro
$CADAGENT_DATA_ROOT/pcbench/exacad_sorted/<sample>/processed_v9_guide_v3.kicad_pro

# Routed PCB root (one folder per method dir; YOU add new method folders here)
$ROUTED_ROOT/<dataset-folder>/<method>/routed/
        ↑ dataset-folder: synthetic_2L_v2 (synth) or PCBench (real)
```

You add a new method by creating a folder under that root following the
naming convention below. Source `.kicad_pro` paths stay the same — you just
need to **stage them with sample-id-only filenames** for `$EVAL_PY`
matching (helper script provided).

## What you bring (new method)

- `.kicad_pcb` files placed at:
  ```
  $ROUTED_ROOT/<dataset>/<base_method_name>/routed/
      <sample-id>_<base_method_name>.kicad_pcb
  ```
  For RL/stochastic: one folder per `(seed, rollout)` tuple, named
  `<base>_seed<N>...r<N>` (regex below).

## Naming convention

```
<your_method_dirname>
   |              |
   ├──────────────┤
   <base>_seed<N>...r<N>          # for RL/stochastic methods (must contain _seed<N> and _r<N>)
   <base>                         # for deterministic methods (one routing per board)

<sample-id>_<your_method_dirname>.kicad_pcb
└── sample-id ──────┘ └── must EXACTLY match dirname ──┘
```

The default RL regex is `_seed(\d+).*?_r(\d+)$` — your dir name must end with
`_r<digits>` and contain `_seed<digits>` somewhere before that. You can
override with `--rl-pattern`.

`<sample-id>` for the matching `.kicad_pro` lookup:
- **synth (synthetic_2L_v2)**: `board_00000`, ..., `board_00127`
- **PCBench (real)**: `0001_rufs__autosave-simple_kicad_schema_and_pcb_v1`, ...
  (sample-id = exacad_sorted folder name)

## End-to-end example: a new "Qwen3-30B" method on PCBench

Suppose your LLM has produced 5 sampled rollouts × 3 seeds × 100 PCBench
boards = 1500 `.kicad_pcb` files.

### 1. Place routed files under the existing root

```
$ROUTED_ROOT/PCBench/
└── Qwen3-30B/                                                # method base name
    └── routed/
        ├── 0001_rufs_..._adapter_Qwen3-30B_seed1_r0.kicad_pcb
        ├── 0001_rufs_..._adapter_Qwen3-30B_seed1_r1.kicad_pcb
        ├── ...
        ├── 0100_smt-zvs-driver_..._Qwen3-30B_seed3_r4.kicad_pcb
```

OR (if you prefer one-folder-per-(seed,rollout), to mirror Transformer_PPO layout):

```
$ROUTED_ROOT/PCBench/
└── Qwen3-30B/
    └── per_run/
        ├── Qwen3-30B_seed1_r0/
        │   └── 0001_rufs_..._adapter_Qwen3-30B_seed1_r0.kicad_pcb
        │       (all 100 boards)
        ├── Qwen3-30B_seed1_r1/
        ├── ...
```

The key is: each routed `.kicad_pcb` file's name must end in `_<algo>.kicad_pcb`,
where `<algo>` is the directory name passed to `$EVAL_PY --algorithm`.

### 2. Stage `.kicad_pro` files (one-time, sample-id-named)

Helper script (run once per dataset):

```bash
# For PCBench (sample-ids like 0001_rufs_...)
mkdir -p /tmp/qwen_pro_pcbench
for d in $CADAGENT_DATA_ROOT/pcbench/exacad_sorted/*/; do
    sample=$(basename "$d")
    src="$d/processed_v9_guide_v3.kicad_pro"
    [ -f "$src" ] && ln -sf "$src" "/tmp/qwen_pro_pcbench/${sample}.kicad_pro"
done
ls /tmp/qwen_pro_pcbench | wc -l   # should match your sample count

# For synth (sample-ids like board_00000)
mkdir -p /tmp/qwen_pro_synth
for f in $CADAGENT_DATA_ROOT/synthetic/synth_2L_v2/test/board_*.kicad_pro; do
    ln -sf "$f" "/tmp/qwen_pro_synth/$(basename $f)"
done
```

### 3. Run the per-board scorer per `<method>` dir

> **`$EVAL_PY` below is a command-line wrapper that is not part of this
> repository.** The scoring it performs is `eval.metrics.evaluate_one` per board
> plus `eval.aggregation.aggregate` for the summary — call those directly, or use
> `methods/baselines/rule_based/run_rule_based_routers.py`, which scores each
> board as it routes it. The `--routed-dir / --pro-dir / --dataset-name /
> --algorithm / --output-dir / --check-angle` options below describe that
> wrapper's interface, not a shipped CLI. The same holds for `reeval_all.py`
> further down.

You may need to point to YOUR cadagent install. Set these once:

```bash
export CADAGENT_ROOT=/path/to/your/cadagent     # this repo's checkout root
export PY=/path/to/cadagent/conda/env/python    # the conda env that built cadagent
export EVAL_PY=/path/to/your/scoring/wrapper    # not shipped — see the note above
export EVAL_OUTDIR="$CADAGENT_ROOT/eval"        # where summaries land

export PYTHONPATH=$CADAGENT_ROOT/build_rl/pcbnew/python/rl:$CADAGENT_ROOT
```

Then for one method dir:

```bash
ALGO=Qwen3-30B_seed1_r0
ROUTED=$ROUTED_ROOT/PCBench/Qwen3-30B/per_run/$ALGO

$PY $EVAL_PY \
    --routed-dir $ROUTED \
    --pro-dir /tmp/qwen_pro_pcbench \
    --dataset-name PCBench \
    --algorithm $ALGO \
    --output-dir $EVAL_OUTDIR \
    --check-angle 90 \
    --force
```

`--check-angle`: `45` for octilinear (45° miters allowed), `90` for pure
Manhattan. Picking wrong inflates `track_angle_drv`. If unsure, run both on
one board and compare.

This produces:
```
$EVAL_OUTDIR/PCBench/$ALGO/
├── logs/<sample>_<ALGO>.json   (per-board)
├── summary.json
└── summary.txt
```

Loop over all `<method>` dirs (parallel helpers live in the original
authors' staging area, not vendored here):

```bash
for dir in $ROUTED_ROOT/Qwen3-30B/per_run/Qwen3-30B_seed*_r*; do
    algo=$(basename "$dir")
    $PY $EVAL_PY --routed-dir "$dir" --pro-dir /tmp/qwen_pro_pcbench \
                 --dataset-name PCBench --algorithm "$algo" \
                 --output-dir $EVAL_OUTDIR --force
done
```

**reeval_all.py CAVEAT**: it auto-selects `--check-angle` based on the dir name
(`90` if `"orthoroute"` in label, else `45`). For your Qwen output:
- if it's 45° octilinear → reeval_all.py uses 45 (correct)
- if it's pure Manhattan → reeval_all.py uses 45 (WRONG, need to call
  `$EVAL_PY` directly with `--check-angle 90`)

### 4. Aggregate

```bash
cd methods/baselines/rule_based/eval/_scripts

# All seeds × all rollouts (RL: max-Φ rollout per (seed, board), seed-mean, board-mean)
$PY aggregate_eval.py \
    --eval-root $EVAL_OUTDIR \
    --dataset PCBench \
    --method-filter '^Qwen3-30B_seed\d+_r\d+$' \
    --mode rl

# Single seed × multiple rollouts (best-of-N within seed=1)
$PY aggregate_eval.py \
    --eval-root $EVAL_OUTDIR \
    --dataset PCBench \
    --method-filter '^Qwen3-30B_seed1_r\d+$' \
    --mode rl

# Deterministic single-output method (no seed/rollout in name)
$PY aggregate_eval.py \
    --eval-root $EVAL_OUTDIR \
    --dataset PCBench \
    --method-filter '^Qwen3-30B_zeroshot$' \
    --mode single
```

Output:
```
=== aggregate (rl mode) ===
  name       ^Qwen3-30B_seed\d+_r\d+$
  n          99            ← board count
  n_seeds    3             ← seeds averaged per board
  succ_pct   88.636        ← Succ. (errors-only DRV == 0 AND fully routed)
  drv_e      0.593         ← errors only
  drv_ep     0.657         ← errors + 3 promoted warnings
  wl         181.544       ← wirelength mm
  via        1.308
  track      40.861
  phi        0.968         ← final_potential (Φ)
```

## Aggregation modes — when to use which

| Mode    | When to use                           | Aggregation rule |
|---------|---------------------------------------|------------------|
| single  | one routing per board                 | per-board mean |
| rl      | (seed, rollout) tuples in dir names   | per (seed, board) max-Φ rollout → seed-mean → board-mean |

`rl` works in degenerate cases:
- 1 seed × N rollouts → n_seeds=1, max-Φ from N (best-of-N within that seed)
- M seeds × 1 rollout → n_seeds=M, single rollout used as-is
- 1 seed × 1 rollout → equivalent to `--mode single`

## Compat-issue exclusion (optional)

`--exclude-compat` drops boards whose error-only DRV codes include
{17, 22, 23, 14, 15, 16} (track-width / via diameter / hole-size / hole packing).
These indicate dim mismatch with project net-class rules — separate from
pure routing-completion failures.

For multi-method intersection (V1) / per-baseline (V2) tables across many
baselines, see `aggregate_eval.py` in this directory.

### PCBench fair-comparison set (95/100)

For comparing methods on the **first 100 PCBench boards**, the recommended
fair-comparison set is `pcbench_fair95.txt` in this directory — it lists 95
sample-IDs that are NOT compat-flagged by any of FR / KRT / Transformer_PPO
under our reference setup. The 5 excluded:

| sample | flagged by | reason |
|---|---|---|
| 0004_S1G-Mod_JST_Adapter | FR | track-width / via dim DRV |
| 0024_memsarray_mems_modules | FR (and KRT) | dim DRV in both |
| 0035_KiCad-Like-a-Pro-Tutorial_rf24-breakout-v1 | FR | track-width DRV |
| 0077_HaveSome_PCB_HaveSomePCB | FR | track-width DRV |
| 0096_karabas-nano_wifi_revA | PPO | inference missing across all 40 rollouts (board too large/complex for the model) |

How the set was built (phi-max-aware logic):

1. For each baseline (FR/KRT/PPO), find boards where the **chosen** evaluation
   has compat DRV codes. For PPO, the "chosen" rollout is the max-Φ pick per
   (seed, board) — borderline rollouts that aren't picked don't count.
2. Union across baselines → 5 boards excluded.
3. 100 − 5 = 95 boards in the fair set.

Note: a small number of boards (e.g. 0020_Cherry-Mx-Bitboard) have a few
borderline best-ckpt rollouts with floating-point rounding DRV (e.g. via at
0.799999 mm vs project min 0.8 mm — 2/20 best-ckpt rollouts on 0020). But
the max-Φ pick across all 4 seeds happens to be clean (rollout r00, Φ=+1.61
for every seed) — so this board is correctly NOT excluded from the fair set.

Use the file directly:

```bash
$PY aggregate_eval.py \
    --eval-root $EVAL_OUTDIR \
    --dataset PCBench \
    --method-filter '^Qwen3-30B_seed\d+_r\d+$' \
    --mode rl \
    --samples-file methods/baselines/rule_based/eval/_scripts/pcbench_fair95.txt
```

## Source `.kicad_pro` lookup rule

`$EVAL_PY` strips the trailing `_<algorithm>` from the routed file's
stem to get the sample-id, then looks for `<sample-id>.kicad_pro` in the
`--pro-dir`. Hence the staging step that creates symlinks named
`<sample-id>.kicad_pro` (the source files have different names like
`processed_v9_guide_v3.kicad_pro`).

## --check-angle quick reference

### Dataset defaults

| Dataset                                  | Default angle | Status |
|------------------------------------------|:-------------:|--------|
| synthetic_2L_v2 (Sym, "synth")           | **45**         | actively used |
| PCBench / exacad_sorted (Real)           | **45**         | actively used |
| grid (planned)                           | **90**         | not yet run — Manhattan board layout |

So for **all currently-evaluated baselines on synth or PCBench**, the
"natural" angle convention is **45**. Use `--check-angle 45` unless your
specific method is known to produce purely orthogonal routes.

### By router output style

| Router output style     | Use            |
|-------------------------|----------------|
| Octilinear (45° miters) | `--check-angle 45` |
| Manhattan (90° corners) | `--check-angle 90` |

| Method            | Convention | Recommended |
|-------------------|------------|-------------|
| Freerouting       | octilinear | 45 |
| KiCadRoutingTools | octilinear | 45 |
| OrthoRoute        | Manhattan  | 90 |
| Transformer_PPO   | Manhattan  | 90 |
| LLM (depends)     | varies     | start with 45 (dataset default); switch to 90 if `track_angle_drv` is huge |

To check: pick one board, run `$EVAL_PY` with `--check-angle 45` and
again with `90`. The mode that reports `track_angle_drv = 0` (or much lower)
is the correct one.

## Troubleshooting

- **"No method dirs match filter"** → eval logs not yet under
  `<eval-root>/<dataset>/<method>/logs/`. Re-run Step 3.
- **`succ_pct` very low across the board** → likely wrong `--check-angle`.
- **Eval log says `"PNS_RL_ROUTER: failed to load board"`** → layer alias
  issue (e.g. `(layer "Top")` instead of `(layer "F.Cu")`). See
  `pcbench_runner.py:fix_layer_names()` for in-place repair.
- **`<sample-id>` not in `--pro-dir`** → check that staged `.kicad_pro` has
  exactly the sample-id filename (no `_<algo>` suffix on the pro side).
- **All boards have `error` field** → cadagent build_rl env vars not set.
  Re-export `PYTHONPATH` and `LD_LIBRARY_PATH` per Step 3.

## Files in this directory

- `aggregate.py` — canonical RQ2 aggregator (single-source-of-truth for paper Table 2 + Table A1).
- `aggregate_eval.py` — general-purpose per-method aggregator with `single` / `rl` modes (upstream).
- `aggregate_routable_only.py` — re-aggregator with `routability == 1.0` filter (upstream; `RQ2_EVAL_ROOT` / `RQ2_OUT_ROOT` are now REQUIRED env vars — no baked-in tree defaults — and `RQ2_FAIR95` stays overridable).
- `pcbench_fair95.txt` — 95-board fair-comparison subset used by the paper.
- `README_upstream.md` — this file (upstream evaluation notes).

For routing the baselines themselves, see `methods/baselines/rule_based/run_rule_based_routers.py`.
