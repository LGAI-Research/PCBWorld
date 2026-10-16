# `methods/baselines/rule_based/` — Rule-based PCB routers

Three external rule-based routers — Freerouting, KiCadRoutingTools (KRT) and
OrthoRoute — behind one runner, so they can be run side-by-side on the project's
datasets on the same terms. Per board the chain is: raw `.kicad_pcb` plus its
pre-made `.dsn`/`.orp` sibling → tool-native output → routed `.kicad_pcb` →
`eval.metrics` log, all under one output tree per `(dataset, algorithm, seed)`.

## Layout

```
methods/baselines/rule_based/
├── run_rule_based_routers.py    UNIFIED RUNNER — all three baselines, eval included
├── run_orthoroute_pipeline.py   OrthoRoute-only chain (kicad_pcb → .orp → .ORS → routed)
├── _lib/datasets.py             dataset / path resolution (env var, then configs/paths.yaml)
├── _converters/                 SES/ORS → kicad_pcb (preserves base-board net refs; no pcbnew)
├── krt/                         pip-installable thin wrapper around KRT (CLI `krt-route`)
├── scripts/setup_env.sh         one-shot env bootstrap
├── eval/                        paper RQ2 snapshot + aggregator (per-board logs gitignored)
└── _run_outputs/                gitignored; the default `--output-root`
```

The external tools themselves are not vendored here: the Freerouting jar, the
KRT checkout and the OrthoRoute source live under the repo-root `external/`
tree, fetched pinned and sha256-verified by
[tools/setup/fetch_baselines.sh](../../../tools/setup/fetch_baselines.sh).

## Quick start

Everything here runs in the single **`pcbworld`** conda env
([environment.yml](../../../environment.yml) carries openjdk for Freerouting and
rust for KRT; cupy/scipy/shapely come from the pip lock).

```bash
conda activate pcbworld
bash tools/setup/fetch_baselines.sh
pip install -e methods/baselines/rule_based/krt -e external/OrthoRoute
export KRT_ROOT="$PWD/external/KiCadRoutingTools"

# smallest end-to-end smoke run (1 board, deterministic baseline, ~30 s)
python methods/baselines/rule_based/run_rule_based_routers.py --baseline krt --dataset synthetic_2l --limit 1
```

The run drops a routed `.kicad_pcb` and a per-board eval json under
`_run_outputs/synth_2L_v2_test/krt/seed0/`; compare it against the paper numbers
in [`eval/RQ2_TABLE.md`](eval/RQ2_TABLE.md) to confirm the setup reproduces.

For a bare conda env that was **not** created from `environment.yml`,
[`scripts/setup_env.sh`](scripts/setup_env.sh) installs the same set into the
active env (openjdk, rust, cupy, the two editable installs, and the Freerouting
jar if missing) and smoke-tests every step. It is idempotent; a fresh install
takes roughly ten minutes and a few GB.

## What you must provide yourself

| requirement | how | when absent |
|---|---|---|
| `KRT_ROOT` | defaults to the pinned `external/KiCadRoutingTools` checkout; `export KRT_ROOT=/path/to/KRT` for another one | falls back to the `krt_tool` entry of [`configs/paths.yaml`](../../../configs/paths.yaml); with neither, `setup_env.sh` aborts with `route.py missing` |
| OrthoRoute source | `git submodule update --init external/OrthoRoute` | the runner does **not** check up front: every board becomes a `status="error"` row (`orthoroute headless rc=2`) in `manifest.json` |
| `.dsn` / `.orp` siblings | pre-generate them (below) — they are never generated at run time | the runner drops the board before routing and warns with the list of boards and which sibling each one lacks |
| dataset roots | `PCBENCH_PCB_ROOT`, `PCBENCH_DSN_ROOT`, `SYNTH2L_PCB_ROOT`, `SYNTH2L_DSN_ROOT`, `SYNTH1L_PCB_ROOT_<grid>`, `SYNTH1L_DSN_ROOT_<grid>` | [`_lib/datasets.py`](_lib/datasets.py) falls back to the matching logical dataset in `configs/paths.yaml` under `PCBWORLD_DATA_ROOT`; with no data root, resolution fails loudly naming the variable |

Routing needs a `.dsn` (Freerouting, KRT) and a `.orp` (OrthoRoute) next to each
raw `.kicad_pcb`. Producing them needs KiCad's `pcbnew` module, so it happens
offline through the prep scripts in the engine repository — any interpreter that
imports `pcbnew` works (the engine's own build, or a host KiCad's `python3`):

```bash
PYTHONPATH=build_rl/pcbnew python engine/pcbnew_prep/make_dsn_orp_v3.py   # PCBench; _synth / _synth_1l for the synthetic sets
```

`run_orthoroute_pipeline.py` resolves the same interpreter for its Stage 1, and
an interpreter that cannot import `pcbnew` is a hard error there rather than
something worked around.

## Running

```bash
python methods/baselines/rule_based/run_rule_based_routers.py \
    --baseline {freerouting|krt|orthoroute} --dataset {pcbench|synthetic_1l|synthetic_2l}
```

`--help` lists every flag (board selection with `--limit` / `--sample`,
`--workers`, `--output-root`, `--reward-config`, `--check-angle`, `--max-passes`,
`--no-eval`). The ones that carry a constraint:

* `--grid` — required by, and valid only for, `--dataset synthetic_1l`.
* `--split` — synthetic sets only.
* `--dataset synthetic_1l` is **freerouting-only** (vias are prohibited and the
  DSN is patched to snap to 90°, so `--check-angle` switches to 90 there);
  `--baseline krt|orthoroute` with it is rejected at parse time.
* `--seeds N` only means something for Freerouting; the two deterministic
  baselines collapse it to 1 and log that.
* `--use-gpu` / `--no-gpu` — OrthoRoute only; see the table below.
* `--check-angle {45,90}` — which corner geometry counts as routed copper. The two sets
  are disjoint: `45` allows a 45° miter or a straight joint, `90` a right angle or a
  straight joint, so the wrong one turns every corner into a `track_angle_drv`. It defaults
  to `45` and only `--dataset synthetic_1l` flips it to `90`, so an orthogonal router scored
  on another dataset needs `--check-angle 90` passed by hand. Freerouting and KRT emit
  octilinear copper (45); OrthoRoute emits Manhattan copper (90). The count lands in
  `total_drv_count` only — the headline DRV column counts DRC errors, not angles.
* `--timeout` — per-board wall clock. The same default applies to all three
  baselines for a symmetric comparison, well above every tool's own budget;
  boards that hit it appear as `status="timeout"` rows in `manifest.json`.

[`run_orthoroute_pipeline.py`](run_orthoroute_pipeline.py) is the alternative
entry point for OrthoRoute alone: it takes a directory of v10 PCBench board
folders instead of a registered dataset and uses the canonical
`make_dsn_orp_v3.py` converter for Stage 1 when siblings are missing. It writes
`processed_v10_guide_v3_unrouted.dsn` and `processed_v10_guide_v3.orp` under the
output root; `<out-root>/_orp_stage1_report.json` records each input board, reused
ORPs, and converter outcomes. A failed conversion or missing/empty ORP stops the
pipeline before routing.

## Choosing a baseline

| baseline | invoked as | determinism | raw artifact | its own budget |
|---|---|---|---|---|
| freerouting | the pinned jar in a headless JVM (`-de/-do/-mp`); Java env vars injected for the 1-layer no-via case | **stochastic** — the JVM reseeds per process, so the same DSN gives a different SES each run; use `--seeds` to sweep | `.ses` | max passes (`--max-passes`) |
| krt | `krt-route` → `route.py` in the KRT tree; the runner parses the DSN for per-class track width / clearance / via size and drill and forwards them, with fixed routing knobs set inline in [`run_rule_based_routers.py`](run_rule_based_routers.py) | deterministic | `.metrics.json` | A\* + ripup iteration cap (typically < 15 s/board) |
| orthoroute | `external/OrthoRoute/main.py headless`, then the in-tree `_converters.ors_to_kicadpcb` injects tracks/vias into the unrouted board; hyperparameters come from upstream's `PathFinderConfig` | deterministic **on CPU** (the default); GPU is faster but per-board DRV / `track_count` can wobble by ±1 from atomic-float ordering | `.ORS` | iteration cap (typically < 60 s/board) |

Tool stdout always lands in `raw/<board>.routing.log`, whichever baseline ran.

## Contracts

### Input layout per dataset

The PCBench aliases `pcbench_exacad` and `pcbench_exacad_dsn` resolve to the v10 sorted PCB and DSN/ORP roots in `configs/paths.yaml`.

| dataset | raw `.kicad_pcb` | `.kicad_pro` | `.dsn` (unrouted) | `.orp` |
|---|---|---|---|---|
| `pcbench` | `${PCBENCH_PCB_ROOT}/<folder>/processed_v10_guide_v3_unrouted.kicad_pcb` | sibling | `${PCBENCH_DSN_ROOT}/<folder>/processed_v10_guide_v3_unrouted.dsn` | `${PCBENCH_DSN_ROOT}/<folder>/processed_v10_guide_v3.orp` |
| `synthetic_2l` | `${SYNTH2L_PCB_ROOT}/<split>/board_NNNNN.kicad_pcb` | sibling | `${SYNTH2L_DSN_ROOT}/<split>/board_NNNNN_unrouted.dsn` | sibling |
| `synthetic_1l` (grid `<G>`) | `${SYNTH1L_PCB_ROOT_<G>}/<split>/board_NNNNN.kicad_pcb` | sibling | `${SYNTH1L_DSN_ROOT_<G>}/<split>/board_NNNNN_unrouted.dsn` | sibling |

`<folder>` for pcbench is the project ID prefix
(e.g. `0001_rufs__autosave-simple_kicad_schema_and_pcb_v1`).

### Output tree

```
<output-root>/<dataset_tag>/<algorithm_tag>/seed<N>/
├── raw/        board_id.{ses|ORS|metrics.json} + board_id.routing.log
├── routed/     board_id_<algorithm>.kicad_pcb
├── eval/       logs/board_id_<algorithm>.json  +  summary.json
└── manifest.json
```

`<dataset_tag>` follows the tags used by [`eval/`](eval/README.md) (`d2a` ↔
`synth_2L_v2_test`, `d3a` ↔ `PCBench`, …) so trees line up side by side.

Scoring is built into the runner (`--no-eval` skips it): once a routed board
exists, `eval.metrics.evaluate_one` runs on it and `eval.aggregation.aggregate`
writes `summary.json`. There is no standalone eval CLI — to re-score an existing
`routed/` tree, e.g. after changing the reward config, either re-run the routing
or call those two library functions over the tree yourself; `<board>_<algorithm>`
is the file-name convention to strip when recovering a board id.

## Reference comparison (paper RQ2)

The aggregated paper numbers are checked in as
[`eval/RQ2_TABLE.md`](eval/RQ2_TABLE.md); the per-board logs behind them are
**not tracked** (`eval/**/logs/` is gitignored, each `seed*/SOURCE_PATH.txt`
records where they came from). With the logs in place,
`python methods/baselines/rule_based/eval/_scripts/aggregate.py` reproduces both
paper tables exactly — see [`eval/README.md`](eval/README.md) for the source
paths, the aggregation rule and the fair-95 PCBench board list.

Re-running the routers reproduces those logs subject to:

| baseline | reproducibility against the canonical logs |
|---|---|
| krt, synth_2l | bit-perfect |
| krt, pcbench | routability / DRV / Φ match; `track_count` and `wirelength_mm` may differ by O(1 segment) from track-segmentation post-processing |
| orthoroute | bit-perfect on synth_2l under the CPU default; the paper numbers were produced on GPU, which agrees at the table-row level but drifts per board |
| freerouting | stochastic — match the paper's seed sweep with `--seeds 4` |
