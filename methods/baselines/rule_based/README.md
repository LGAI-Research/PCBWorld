# `methods/baselines/rule_based/` — Rule-based PCB routers

Three rule-based routers wrapped behind one entry point so they can be run
side-by-side on the project's datasets. Each run produces a routed
`.kicad_pcb` and a per-board eval log keyed off the project's reward / DRV
definitions.

```
                            ┌──────────────────────────────────┐
   raw .kicad_pcb (+ sibling│ methods/baselines/rule_based/run_rule_based_routers  │      <output-root>/<dataset>/<algo>/seed<N>/
       .kicad_pro, .dsn,    │   --baseline ∈ {fr, krt,         │ ─→  ├── raw/      (.ses or .ORS or .log)
       .orp)  ──────────────▶│   orthoroute}                    │      ├── routed/   (board_id_<algo>.kicad_pcb)
                            │   --dataset  ∈ {pcbench,         │      ├── eval/     (eval.metrics logs + summary)
                            │       synthetic_1l/2l}           │      └── manifest.json
                            └──────────────────────────────────┘
```

## Layout

```
methods/baselines/rule_based/
├── README.md                          (this file)
├── .gitignore
│
├── run_rule_based_routers.py                    UNIFIED RUNNER
│
├── _lib/                              dataset / path resolver (env-var driven)
├── _converters/                       SES/ORS → kicad_pcb (verbatim, no pcbnew)
│
├── krt/                               pip-installable thin wrapper (`krt-route`)
│                                      (the Freerouting jar and the OrthoRoute
│                                       source live under repo-root external/
│                                       — see Install)
│
├── scripts/setup_env.sh               one-shot env bootstrap
│
├── eval/                              paper-canonical eval logs + aggregator
│   ├── d2a/{Freerouting,OrthoRoute,KiCadRoutingTools}/seed*/
│   ├── d3a/{Freerouting,OrthoRoute,KiCadRoutingTools}/seed*/
│   ├── RQ2_TABLE.md                   paper Table 2 (mean) + Table A1 (mean+std)
│   └── _scripts/aggregate.py          reproduce table from logs
│
└── _run_outputs/                      (gitignored — produced by run_rule_based_routers.py)
```

## Quick start

> **conda env:** everything here runs in the single **`cadagent`** env —
> [environment.yml](../../../environment.yml) carries openjdk 21 (Freerouting) and
> rust (KRT); cupy/scipy/shapely are in the pip lock. Externals (jar · KRT ·
> OrthoRoute) come from [tools/setup/fetch_baselines.sh](../../../tools/setup/fetch_baselines.sh)
> (pinned + sha256-verified).

```bash
conda activate cadagent
bash tools/setup/fetch_baselines.sh
pip install -e methods/baselines/rule_based/krt -e external/OrthoRoute
export KRT_ROOT="$PWD/external/KiCadRoutingTools"

# Smallest end-to-end smoke run (1 board, both deterministic baselines, ~30 s)
python methods/baselines/rule_based/run_rule_based_routers.py --baseline krt        --dataset synthetic_2l --limit 1
python methods/baselines/rule_based/run_rule_based_routers.py --baseline orthoroute --dataset synthetic_2l --limit 1
```

Each invocation drops a `.kicad_pcb` under
`methods/baselines/rule_based/_run_outputs/synth_2L_v2_test/<algo>/seed0/routed/` and a per-board
eval json under `…/seed0/eval/logs/`. Compare against the paper-canonical
log in `methods/baselines/rule_based/eval/d2a/<PaperName>/seed0/logs/` to verify reproducibility
(see [Reference comparison](#reference-comparison)).

## Install

The standard path is the repo-wide one (the single-env policy in the top-level
[README](../../../README.md)): `conda env create -f environment.yml` +
`tools/setup/fetch_baselines.sh` + the two editable installs from the Quick
start above.

### One-shot bootstrap (`setup_env.sh`)

For a bare conda env that was not created from `environment.yml`:

```bash
bash methods/baselines/rule_based/scripts/setup_env.sh
```

What it installs (all into the active conda env):

1. `openjdk≥21` via conda-forge (Freerouting Java runtime)
2. `rust` via conda-forge (KRT's `grid_router.so` is rebuilt locally on first
   `krt-route` call; the upstream prebuilt copy targets a newer glibc than
   e.g. Ubuntu 20.04 ships)
3. `cupy-cuda12x` via pip (OrthoRoute GPU path; assumes CUDA 12.x driver on host)
4. `pip install -e external/OrthoRoute` (editable; the vendored upstream submodule)
5. `scipy` / `shapely` for the upstream KRT `route.py`, plus
   `pip install -e methods/baselines/rule_based/krt` (thin wrapper; CLI `krt-route`)
6. Smoke tests of each step — including auto-downloading the Freerouting jar to
   `external/freerouting/` if missing.

Tested fresh-env install: **~9.5 min, ~2.5 GB** (Python 3.12 + openjdk 25 +
rust + cupy-cuda12x + OrthoRoute + krt-runner).

The script is idempotent — re-running skips already-installed steps.

### Prerequisites the bootstrap does NOT cover

`setup_env.sh` auto-fetches the Freerouting jar and installs the OrthoRoute
submodule if present, so the only thing you MUST handle yourself is `KRT_ROOT`:

#### 1. Freerouting JAR (~66 MB) — auto-downloaded

`setup_env.sh` downloads it to `external/freerouting/freerouting-2.1.0.jar` (from the
pinned v2.1.0 release URL) when missing; the jar is gitignored (binary). To fetch
it manually instead:

```bash
mkdir -p external/freerouting
curl -L -o external/freerouting/freerouting-2.1.0.jar \
    https://github.com/freerouting/freerouting/releases/download/v2.1.0/freerouting-2.1.0.jar
```


#### 2. `KRT_ROOT` (KiCadRoutingTools install tree)

The KRT wrapper forwards to `route.py` inside an external KRT checkout. The
default `KRT_ROOT` is `external/KiCadRoutingTools` — the pinned checkout made by
`tools/setup/fetch_baselines.sh`. To use a checkout elsewhere:

```bash
export KRT_ROOT=/path/to/KRT
```

Without either, `setup_env.sh` step [6/6] aborts with `route.py missing`.

#### 3. `external/OrthoRoute` (OrthoRoute source submodule)

OrthoRoute is a git submodule at `external/OrthoRoute` (pinned upstream). Initialize it
once after clone:

```bash
git submodule update --init external/OrthoRoute
```

The bootstrap installs it only if `external/OrthoRoute/main.py` exists (i.e. the
submodule is checked out). If it is missing, `setup_env.sh` skips OrthoRoute and
`run_rule_based_routers.py --baseline orthoroute ...` fails with a missing `main.py`.

#### 4. `pcbnew` (for DSN/ORP generation only)

Routing needs `.dsn` (Freerouting, KRT) and `.orp` (OrthoRoute) files. We
**do not** generate them at runtime — the runner expects them to already
exist next to each raw `.kicad_pcb`. To produce them from a fresh
`.kicad_pcb`, run the prep scripts under `engine/pcbnew_prep/`, which
import `pcbnew` (the KiCad Python module, ships with the KiCad app).

```bash
# Example: pcbnew Python typically lives at /usr/lib/python3/dist-packages
/usr/bin/python3 engine/pcbnew_prep/make_dsn_orp_v3.py        # PCBench
/usr/bin/python3 engine/pcbnew_prep/make_dsn_orp_synth.py     # synthetic_2L
/usr/bin/python3 engine/pcbnew_prep/make_dsn_orp_synth_1l.py  # synthetic_1L
```

`pcbnew` is not pip/conda-installable — install KiCad (v9 recommended) on
the host. See `engine/pcbnew_prep/README.md` (in the engine repository) for
details.

#### 5. Dataset roots (env vars)

The runner resolves dataset paths via env vars. Set them to point at your
local mirror of the data:

```bash
export PCBENCH_PCB_ROOT=/path/to/exacad_sorted
export PCBENCH_DSN_ROOT=/path/to/exacad_sorted_dsn
export SYNTH2L_PCB_ROOT=/path/to/synth_2L_v2
export SYNTH2L_DSN_ROOT=/path/to/synth_2L_v2_dsn
# Per-grid roots for synth_1L:
export SYNTH1L_PCB_ROOT_50=/path/to/synth_1L_grid50_5net_v02
export SYNTH1L_DSN_ROOT_50=/path/to/synth_1L_grid50_5net_v02_dsn
# (similarly for grid 10 / 100 / 200 / 500)
```

Defaults baked into [`_lib/datasets.py`](_lib/datasets.py) point at the
project's NFS mount; outside users **must** override them.

## CLI

```
python methods/baselines/rule_based/run_rule_based_routers.py \
    --baseline {freerouting|krt|orthoroute} \
    --dataset {pcbench|synthetic_1l|synthetic_2l} \
    [--grid {10|50|100|200|500}]      # synthetic_1l only
    [--split test]                    # synth_* only; default test
    [--seeds N]                       # freerouting only; folders seed0..seed{N-1}
    [--limit N]                       # first N boards
    [--sample SUBSTR ...]             # repeatable substring filter
    [--workers W]                     # board-level concurrency (per seed); default 1
    [--timeout SEC]                   # wrapper-level wall-clock per board
    [--output-root DIR]               # default: methods/baselines/rule_based/_run_outputs
    [--no-eval]                       # skip eval.metrics
    [--reward-config NAME]            # default drc_dense_promoted
    [--check-angle 45|90]             # eval routing-angle check; auto-90 for synth_1l
    [--max-passes N]                  # freerouting -mp; default 10
    [--no-gpu | --use-gpu]            # OrthoRoute: CPU-only (default) or GPU
```

### Constraints

* `--dataset synthetic_1l` is **freerouting-only**: vias are prohibited
  (`FREEROUTING__ROUTER__VIAS_ALLOWED=false`, `VIA_COSTS=10000`) and the DSN
  is patched with `(snap_angle ninety_degree)` so routing stays orthogonal.
  `--check-angle` defaults to 90 in this mode. Passing `--baseline krt` or
  `orthoroute` with `--dataset synthetic_1l` is rejected at CLI parse time.
* `--seeds N>1` only affects Freerouting (its JVM reseeds from
  `System.nanoTime()` per process; the same DSN produces a different SES
  each run). KRT and OrthoRoute are deterministic — `--seeds` collapses to 1
  with a warning.
* **OrthoRoute defaults to CPU-only** for bit-perfect reproducibility. GPU
  mode (`--use-gpu`) is faster but per-board DRV / track_count can wobble
  by ±1 due to atomic-float ordering. Paper RQ2 numbers were produced under
  GPU; aggregate metrics still agree with CPU mode at the table-row level.

### Data layout assumed

| dataset | raw `.kicad_pcb` | `.kicad_pro` | `.dsn` (unrouted) | `.orp` |
|---|---|---|---|---|
| `pcbench` | `${PCBENCH_PCB_ROOT}/<folder>/processed_v9_guide_v3_unrouted.kicad_pcb` | sibling `.kicad_pro` | `${PCBENCH_DSN_ROOT}/<folder>/processed_v9_guide_v3_unrouted.dsn` | sibling `.orp` |
| `synthetic_2l` | `${SYNTH2L_PCB_ROOT}/<split>/board_NNNNN.kicad_pcb` | sibling `.kicad_pro` | `${SYNTH2L_DSN_ROOT}/<split>/board_NNNNN_unrouted.dsn` | sibling `.orp` |
| `synthetic_1l` (grid `<G>`) | `${SYNTH1L_PCB_ROOT_<G>}/<split>/board_NNNNN.kicad_pcb` | sibling `.kicad_pro` | `${SYNTH1L_DSN_ROOT_<G>}/<split>/board_NNNNN_unrouted.dsn` | sibling `.orp` |

`<folder>` for pcbench is the project ID prefix
(e.g. `0001_rufs__autosave-simple_kicad_schema_and_pcb_v1`).

### Hyperparameters carried through

* **Freerouting** — JAR invoked with `-de <dsn> -do <ses> -mp <max_passes>`
  in headless JVM. Java env vars (`FREEROUTING__ROUTER__...`) injected for
  the synth_1L no-via / 90° case.
* **KRT** — invoked through `krt-route` (forwards to `route.py` in the KRT
  install tree). The runner parses the DSN to extract per-class track-width
  / clearance / via-size / via-drill, then passes them via CLI flags. Fixed
  routing knobs (`--grid-step 0.1`, `--max-iterations 200000`,
  `--heuristic-weight 1.9`, `--ordering mps`, …) are set inline in
  `methods/baselines/rule_based/run_rule_based_routers.py`.
* **OrthoRoute** — invoked via `external/OrthoRoute/main.py headless <orp> -o <ors>
  --cpu-only` (default; pass `--use-gpu` for GPU). The runner then calls the
  in-tree `ors_to_kicadpcb.convert()` to inject tracks/vias into the
  unrouted `.kicad_pcb`. OrthoRoute reads its hyperparams from
  `PathFinderConfig` defaults (see
  `external/OrthoRoute/orthoroute/algorithms/manhattan/unified_pathfinder.py:583+`).

### Wrapper timeouts

The `--timeout` flag caps subprocess wall-clock per board. **All three
baselines default to 1800 s** for a symmetric fair comparison; the internal
budget each tool actually targets is lower:

| baseline    | wrapper default | internal budget |
|---|---|---|
| freerouting | 1800 s | `-mp 10` max passes (the paper's setting) |
| krt         | 1800 s | `--max-iterations 200000` A* + ripup (typically <15 s) |
| orthoroute  | 1800 s | `PathFinderConfig.max_iterations = 40` (typically <60 s) |

In our PCBench OrthoRoute run, the slowest board converged in 48 s; in
PCBench Freerouting, large boards can take several hundred seconds. 1800 s
gives all three a comfortable safety margin without favouring any one tool.
Look for `status="timeout"` rows in `manifest.json` if you ever need to
raise this further.

## Output tree

```
<output-root>/<dataset_tag>/<algorithm_tag>/seed<N>/
├── raw/
│   ├── board_id.ses               (freerouting only)
│   ├── board_id.ORS               (orthoroute only)
│   ├── board_id.metrics.json      (krt only)
│   └── board_id.routing.log       (tool stdout, all baselines)
├── routed/
│   └── board_id_<algorithm>.kicad_pcb
├── eval/
│   ├── logs/board_id_<algorithm>.json       # eval.metrics.evaluate_one
│   └── summary.json                          # eval.aggregation.aggregate
└── manifest.json
```

`<dataset_tag>` follows the convention used by `methods/baselines/rule_based/eval/` (`d2a` ↔
`synth_2L_v2_test`, `d3a` ↔ `PCBench`, etc.) for easy side-by-side
comparison.

The eval step is built into the runner — once a board's
`<routed>/<board>_<algo>.kicad_pcb` is produced, `eval.metrics.evaluate_one`
is invoked automatically. There is no standalone eval CLI; to re-score already
routed boards (e.g. after changing the reward config), call the library
function over the `routed/` directory:

```python
import json
from pathlib import Path
from eval import aggregation, metrics

seed_dir = Path("methods/baselines/rule_based/_run_outputs"
                "/synth_2L_v2_test/orthoroute/seed0")
pro_dir = Path("<dataset dir holding the source .kicad_pro files>")
logs = seed_dir / "eval" / "logs"
logs.mkdir(parents=True, exist_ok=True)

per_board = []
for pcb in sorted((seed_dir / "routed").glob("*.kicad_pcb")):
    board_id = pcb.stem.rsplit("_", 1)[0]          # strip the trailing _<algorithm>
    m = metrics.evaluate_one(str(pcb), str(pro_dir / f"{board_id}.kicad_pro"),
                             reward_config_name="drc_dense_promoted",
                             check_angle=45)        # 90 for the 1-layer D1 sets
    (logs / f"{pcb.stem}.json").write_text(json.dumps(m, indent=2, default=str))
    per_board.append(m)

(seed_dir / "eval" / "summary.json").write_text(
    json.dumps(aggregation.aggregate(per_board), indent=2, default=str))
```

## Reference comparison (paper RQ2)

Paper-canonical per-board eval logs ship under `methods/baselines/rule_based/eval/<d2a|d3a>/<PaperName>/seed*/`
(see [`eval/README.md`](eval/README.md) for the exact NFS source paths and
the fair-95 PCBench board list). Aggregating these logs reproduces the
paper Table 2 (mean) + Table A1 (mean+/-std) exactly:

```bash
python methods/baselines/rule_based/eval/_scripts/aggregate.py
```

Re-running `run_rule_based_routers.py` reproduces the per-board logs subject to:

| baseline    | dataset      | reproducibility under this runner |
|---|---|---|
| krt         | synth_2l     | **bit-perfect** vs canonical logs |
| krt         | pcbench      | routability / DRV / Φ match; `track_count` and `wirelength_mm` may differ from the canonical logs by O(1 segment) because of track-segmentation post-processing |
| orthoroute  | synth_2l     | **bit-perfect under `--no-gpu`** (the default). `--use-gpu` matches at the aggregate level but per-board DRV / `track_count` can drift by ±1 due to atomic-float ordering on the GPU. |
| orthoroute  | pcbench      | The canonical paper-Table numbers ship under `methods/baselines/rule_based/eval/d3a/OrthoRoute/seed0/` and are reproduced exactly by `_scripts/aggregate.py`. |
| freerouting | all          | Stochastic (JVM reseeds from `System.nanoTime()`). Use `--seeds 4` to match the paper's seed sweep. |
