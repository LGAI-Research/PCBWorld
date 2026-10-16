# PCBench → KiCad 10.0.7 D3 preparation

Rebuilds the live D3 corpus from the public [PCBench](https://github.com/PCBench/PCBench)
collection. Board files are user data and are not redistributed by this repository.

## Toolchains

The prep chain requires both matched toolchains before processing:

- The **Engine KiCad 10.0.7 build**, including its uncapped `kicad-cli` and `pcbnew` module.
  On Linux the CLI is `build_rl/kicad/kicad-cli`; on macOS it is
  `build_rl/kicad/KiCad.app/Contents/MacOS/kicad-cli`. The module is under `build_rl/pcbnew`.
  Build with `BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh` when absent.
- The **independent stock KiCad 10.0.7 CLI and Python binding**, used by Step10 save/refill/
  reopen validation. On Arc these resolve to `/usr/bin/kicad-cli` and `/usr/bin/python3`;
  on macOS they resolve to `~/Applications/KiCad-10.0.7/KiCad.app/Contents/MacOS/kicad-cli`
  and its bundled `Python.framework/Versions/Current/bin/python3` (Python 3.9.13).

`prepare_pcbench.sh` checks all four versions before conversion. There is no KiCad 9, 10.0.6,
PATH, or stock-CLI fallback for the Engine DRC path. Overrides, when needed, are explicit:
`PCBENCH_KICAD_CLI` (Engine CLI), `PCBNEW_PYTHON` (interpreter for the Engine `pcbnew` build),
`KICAD_STOCK_CLI`, and `KICAD_STOCK_PYTHON` (or `KICAD_PYTHON`). The selected Engine CLI must
carry the Engine `ENGINE_VERSION` build stamp; all reported KiCad versions must be 10.0.7.

## Source pin and safety

The public clone is pinned to commit
`dec3be75cbdef74787625f9043c7391cd473bb64`. A new clone is checked out at that exact commit.
An existing clone is accepted only if it is already at that commit and clean; the quickstart
never resets, checks out, fetches into, or overwrites an existing clone with unexpected state.

All generated data lives under `PCBWORLD_DATA_ROOT` (default in the repo launcher:
`third_party/PCBWorld/var/datasets`). The v10 roots are siblings of the historical v9 roots:

- `pcbench_work/v10`
- `pcbench_work/newdrc_v10`
- `pcbench/exacad_sorted_v10`
- `pcbench/exacad_sorted_dsn_v10`

The quickstart requires distinct v10 output roots below `PCBWORLD_DATA_ROOT` and refuses any
output override that overlaps the raw PCBench clone or these preserved paths:
`pcbench_work/v9`, `pcbench_work/newdrc`, `pcbench/exacad_sorted`, and
`pcbench/exacad_sorted_dsn`.

## Pipeline and artifact names

```
PCBench/PCBs/<source-folder>/processed.kicad_pcb
  │  0. convert_v10.py: source folder → pcbench_work/v10/<source-folder>/
  │     Engine pcbnew load/save → processed_v10.kicad_pcb + .kicad_pro
  │  1. drc_fix_v10.py: pcbench_work/v10 → pcbench_work/newdrc_v10
  │     preserves the historical D3 severity overrides, rule floors, and
  │     min(0.25 mm, Default net-class clearance) hole-to-hole floor
  │  2. make_guide.py: processed_v10 → processed_v10_guide_v3 (+ unrouted guide)
  │  3. sort_prefix.py: successful guide boards → exacad_sorted_v10/<NNNN>_<source-folder>/
  │     CSV includes raw source ID (`sample`) and rank-based folder (`sorted_id`)
  │  4. engine/pcbnew_prep/make_dsn_orp_v3.py: sorted guides → exacad_sorted_dsn_v10/
  ▼
processed_v10_guide_v3.kicad_pcb
processed_v10_guide_v3_unrouted.kicad_pcb
processed_v10_guide_v3_unrouted.dsn
processed_v10_guide_v3.orp
```

Stage reports stay beside their corresponding v10 data:

- `pcbench_work/v10/_results.json` — conversion outcome for every selected raw source folder.
- `pcbench_work/newdrc_v10/_results.json` — DRC pass, legitimate `drc_rejected`, or explicit
  stage error for every converted input; `_failures.json` contains conversion and stage errors.
- `pcbench_work/newdrc_v10/_guide_results.json` — guide `pass`, DRC `fail` rejection, `skip`,
  or `error` for every DRC-passing board.
- `pcbench_work/newdrc_v10/_sort_results.json` — sorted, guide-rejected, or stage-error outcome.
- `pcbench/exacad_sorted_dsn_v10/_run-report.json` — DSN/ORP outcome for every sorted folder.

A DRC rejection is not a tool failure. Timeout, nonzero CLI exit, missing DRC output, and
malformed JSON are explicit stage errors and never count as zero violations. Keep the existing D3
ignore overrides for `solder_mask_bridge`, `drill_out_of_range`, `malformed_courtyard`,
`courtyards_overlap`, `annular_width`, `copper_edge_clearance`, and `hole_clearance`; add
`pth_inside_courtyard` and `npth_inside_courtyard` as `ignore` to match v10 defaults that changed
from v9. Under the **unpatched KiCad 10 defaults**, 53 of 54 boards that passed v9 but failed v10
failed only these two checks; two frozen benchmark boards were affected solely by them. The
compatibility overrides above apply to current v10 runs, and the full patched run reported 679
DRC passes. Do not blanket-ignore `tracks_crossing` or `invalid_outline`. The same rule floors
remain zero, and the hole-to-hole floor remains `min(0.25 mm, Default net-class clearance)`.

`make_guide.py` keeps its best-attempt files for a valid DRC rejection (`status: fail`), but the
sorter excludes them. The quickstart validates each report against its stage inputs and continues
through later stages when a command exits nonzero but has produced a complete per-board ledger.
DRC `drc_rejected` and guide `fail`/`guide_rejected` outcomes are normal filters. Conversion
failures, other board/stage errors, unexplained nonzero exits, or missing, malformed, duplicate,
or incomplete ledgers make the quickstart return nonzero; an incomplete ledger stops the run.
Every emitted sorted board must have a successful guide result before DSN/ORP generation. The
historical 679 v9 pass count is not a forced v10 target.

## Run

The quickstart resolves the PCBWorld root from its own location. From the `/root/gnn` monorepo,
prefer `scripts/pcbw`; it activates the `pcbworld` environment and exposes Engine `pcbnew`:

```bash
export PCBWORLD_DATA_ROOT=/data/pcbworld
scripts/pcbw bash third_party/PCBWorld/tools/quickstart/prepare_pcbench.sh
```

From a standalone PCBWorld checkout, activate the matching `pcbworld` environment (with `python`
available) and run from the repository root. When the monorepo wrapper is absent, the script uses
the active environment's `python`; it still sets the Engine CLI/PYTHONPATH and its preflight
checks both Engine and stock KiCad versions before processing:

```bash
conda activate pcbworld
export PCBWORLD_DATA_ROOT=/data/pcbworld
bash tools/quickstart/prepare_pcbench.sh
```

For a trial, give each stage its own new v10 root so trial output cannot mix with a full run.
From the monorepo root:

```bash
DATA=/data/pcbworld
PCBWORLD_DATA_ROOT="$DATA" \
PCBENCH_V10_ROOT="$DATA/pcbench_work/v10_trial" \
PCBENCH_NEWDRC_OUT="$DATA/pcbench_work/newdrc_v10_trial" \
PCBENCH_SORTED_OUT="$DATA/pcbench/exacad_sorted_v10_trial" \
PCBENCH_DSN_OUT="$DATA/pcbench/exacad_sorted_dsn_v10_trial" \
  scripts/pcbw bash third_party/PCBWorld/tools/quickstart/prepare_pcbench.sh --limit 20 --workers 2
```

From a standalone checkout, activate `pcbworld` first and use the same roots with the local path:

```bash
conda activate pcbworld
DATA=/data/pcbworld
PCBWORLD_DATA_ROOT="$DATA" \
PCBENCH_V10_ROOT="$DATA/pcbench_work/v10_trial" \
PCBENCH_NEWDRC_OUT="$DATA/pcbench_work/newdrc_v10_trial" \
PCBENCH_SORTED_OUT="$DATA/pcbench/exacad_sorted_v10_trial" \
PCBENCH_DSN_OUT="$DATA/pcbench/exacad_sorted_dsn_v10_trial" \
  bash tools/quickstart/prepare_pcbench.sh --limit 20 --workers 2
```

After the **full** prep run completes, inspect all stage ledgers and status counts. The runner
can return nonzero for isolated per-board failures after completing downstream stages; reported
failures may still leave a usable sorted corpus. If the corpus is usable, build the live v10 D3
manifest separately. The builder verifies its required source IDs and refuses to overwrite frozen
`d3.json`; the default output is `configs/datasets/d3_v2.json`. From `/root/gnn`:

```bash
scripts/pcbw bash third_party/PCBWorld/experiments/kdd/d3_dataset/run.sh
```

From the standalone checkout root, with the matching environment active:

```bash
bash experiments/kdd/d3_dataset/run.sh
```

For a trial corpus, point both the sorter and the manifest output at trial paths instead of
changing the live alias target. From `/root/gnn`:

```bash
DATA=/data/pcbworld
PCBWORLD_DATA_ROOT="$DATA" scripts/pcbw bash \
  third_party/PCBWorld/experiments/kdd/d3_dataset/run.sh \
  --sorted-dir "$DATA/pcbench/exacad_sorted_v10_trial" \
  --out "$DATA/pcbench_work/d3_v2_trial.json"
```

From the standalone checkout root, with the matching environment active:

```bash
DATA=/data/pcbworld
PCBWORLD_DATA_ROOT="$DATA" bash experiments/kdd/d3_dataset/run.sh \
  --sorted-dir "$DATA/pcbench/exacad_sorted_v10_trial" \
  --out "$DATA/pcbench_work/d3_v2_trial.json"
```

The output roots can be overridden independently with `PCBENCH_V10_ROOT`,
`PCBENCH_NEWDRC_OUT`, `PCBENCH_SORTED_OUT`, and `PCBENCH_DSN_OUT`. The raw input can be
pointed at the exact pinned clone with `PCBENCH_PCBS_ROOT`; its Git checkout root may be
set explicitly with `PCBENCH_CLONE_ROOT`. `PCBENCH_PCB_ROOT` and `PCBENCH_DSN_ROOT` are set
by the quickstart for the existing Engine DSN/ORP converter.

The live aliases `d3a`, `d3b`, and `d3c` point to `configs/datasets/d3_v2.json` after it has
been generated. `configs/datasets/d3.json` remains the frozen v1 manifest. Both easy train-only
and medium test override identities are resolved from bare PCBench source folder IDs, not
current sort ranks.
