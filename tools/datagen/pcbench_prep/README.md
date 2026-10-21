# `tools/datagen/pcbench_prep/` — PCBench → `exacad_sorted` (D3) reconstruction

Rebuilds the D3 real-board benchmark set from a clone of the public
[PCBench](https://github.com/PCBench/PCBench) collection. Board content is never
redistributed by this repository — users clone PCBench and run this chain.

## Prerequisites

- **KiCad 9 `kicad-cli` and the `pcbnew` Python module** — the steps shell out to
  `kicad-cli pcb drc` and run `pcbnew` in a child interpreter (nothing GPL in-process),
  resolving both through [kicad_tools.py](kicad_tools.py) (`KICAD_CLI` / `PCBNEW_PYTHON`
  override) and printing the pair they chose as their first line. Prefer the engine's own
  build (`BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh`, ~4.5 min on 64 cores;
  [engine/README.md](../../../engine/README.md)): it is the KiCad the environment routes
  with, and its DRC reports are uncapped where a stock KiCad 9's are not (see below).
- ~4 GB of disk: the clone is 1.2 GB, the intermediate trees add about as much.

## Pipeline

```
PCBench/PCBs/<name>/processed.kicad_pcb   (public clone: KiCad 5 file format, no project file)
   │  0. convert_v9.py     PCBENCH_PCBS_ROOT → PCBENCH_V9_ROOT
   │     pcbnew load + save: KiCad 9 board + the .kicad_pro KiCad 9 derives from the legacy
   │     (setup …) block; diff-pair width / gap follow the class track width / clearance unless
   │     the board set its own value (netclass_defaults.py); metadata.json / final.json copied along
   │  1. drc_fix_v9.py     PCBENCH_V9_ROOT → PCBENCH_NEWDRC_OUT
   │     patch the minimum .kicad_pro entries (severities → ignore, a few rule floors → 0,
   │     hole-to-hole = min(0.25 mm, Default net-class clearance)); the boards that then
   │     pass DRC are the D3 pool, as <name>/processed_v9.kicad_pcb + .kicad_pro
   │  2. make_guide.py     per-net uniform-width guide board + .kicad_pro + *_unrouted
   │     strip; the guide board is what the benchmark loads (algorithm: make_guide.md)
   │  3. sort_prefix.py    PCBENCH_NEWDRC_OUT → PCBENCH_SORTED_OUT
   │     order = (pins, nets, components) ascending from final.json + the footprint count;
   │     folders copied as <NNNN>_<name>/ next to pcb_characteristics_exacad_sorted.csv
   │  4. make_dsn_orp.py   PCBENCH_SORTED_OUT → PCBENCH_DSN_ROOT
   │     the .dsn/.orp siblings the rule-based routers read: the engine's
   │     engine/pcbnew_prep/make_dsn_orp_v3.py in the pcbnew interpreter, results in _results.json;
   │     a board it cannot write exactly is a fail row (exit 3; on the full corpus 11, none in a
   │     D3 eval set), and the chain stops only when the converter did not finish
   ▼
exacad_sorted/       (consumed via configs/paths.yaml `pcbench_exacad`;
                      split JSON: experiments/kdd/d3_dataset/build.py)
exacad_sorted_dsn/   (configs/paths.yaml `pcbench_exacad_dsn`)
```

## Run

```bash
export PCBWORLD_DATA_ROOT=/data/pcbworld     # your dataset root — everything lands under it
bash tools/quickstart/prepare_pcbench.sh     # clone + all five steps, one command
```

[`prepare_pcbench.sh`](../../quickstart/prepare_pcbench.sh) clones PCBench (`main` @
`dec3be75`), builds the KiCad tools when the engine build lacks them, and runs the chain with
the stem and suffix the split json expects; `--limit N` makes a trial run. Driving one step
by hand needs the diagram's `PCBENCH_*` roots exported (the wrapper's defaults show the
layout; flags are in each script's `--help`). `sort_prefix.py` owns the `<NNNN>_<name>/`
entries of its output directory — a re-run replaces them and drops what an earlier run left,
so a full run after a trial leaves exactly the full set.

## Determinism

With the engine's own 9.0.8 build, 1 182 board folders convert and **679** pass the DRC
filter — the paper's set exactly. KiCad mints fresh UUIDs on the step-0 round trip, so two
runs are not byte-identical; geometry, nets, design rules and DRC verdicts are, and steps
1–3 reproduce every file byte for byte.

Step 2 would otherwise pick its offender nets by race, since `kicad-cli pcb drc` runs the
copper-clearance test on a thread pool: it passes `--all-track-errors`, and, because a stock
KiCad stops reporting a violation type after 199 hits (clearance: 499) with no way to raise
that cap, treats a count at those values as unusable and narrows every reducible net instead
of the offenders ([make_guide.md](make_guide.md)). The engine's uncapped `kicad-cli` gets
the full list and narrows only the offenders on the 15 boards that reach those values; the
shipped `d3` is that output, identical to the paper's set (UUIDs aside) but for the per-net
widths of 40 of the 679 guide boards, which predate these guards, and for the diff-pair width / gap
of the net classes a KiCad 5 board left unset: they follow the class track width / clearance
([netclass_defaults.py](netclass_defaults.py)) where the paper's set kept KiCad's 0.2 / 0.25 mm
defaults, so nets KiCad pairs by name route differently on 72 of D3's 677 boards.

`bash experiments/kdd/d3_dataset/run.sh --out /tmp/d3.json` rebuilds the split json from the
CSV + layout, reproducing [d3.json](../../../configs/datasets/d3.json) up to one recorded
manual edit, the `medium.test` swap under the json's `_test_overrides` key (`dataset_dirs`
comes out as the absolute corpus path; the shipped file spells it `${PCBWORLD_DATA_ROOT}/…`).

The DSN/ORP converter itself is in `engine/pcbnew_prep/` (it imports `pcbnew` directly, hence
lives in the engine bundle); step 4 only runs it.
