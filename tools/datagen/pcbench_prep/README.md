# `tools/datagen/pcbench_prep/` — PCBench → `exacad_sorted` (D3) reconstruction

Rebuilds the D3 real-board benchmark set from a clone of the public
[PCBench](https://github.com/PCBench/PCBench) collection. Board content is never
redistributed by this repository — users clone PCBench and run this chain.

## Prerequisites

- **`kicad-cli` and the `pcbnew` Python module built from the engine's pinned source** —
  `BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh` (about 4.5 minutes on 64 cores;
  [engine/README.md](../../../engine/README.md) "Build"). The scripts shell out to
  `kicad-cli pcb drc` and run `pcbnew` in a child interpreter — nothing GPL is imported
  in-process — and resolve both through [kicad_tools.py](kicad_tools.py): `$KICAD_CLI` →
  `build_rl/kicad/kicad-cli` → `kicad-cli` on `PATH`; `$PCBNEW_PYTHON` → the running
  interpreter with `build_rl/pcbnew` on `PYTHONPATH` (when `import pcbnew` works there) →
  `/usr/bin/python3`. Every step prints its choice as its first line
  (`tools: kicad-cli=… (9.0.8)  pcbnew=… (9.0.8)`) and stops if a tool is unusable. The
  engine's build is the KiCad the environment routes with — the same 9.0.8 source, the same
  patched DRC — rather than whichever 9.0.x a distribution ships (the paper's set was made
  with the apt 9.0.8, the 2026-08-30 reproduction with 9.0.9 because the PPA had moved on).
- Alternative: any KiCad 9 install that provides `kicad-cli` and `pcbnew` (with an apt/PPA
  KiCad, `pcbnew` lives in the system `/usr/bin/python3`). `KICAD_CLI` / `PCBNEW_PYTHON`
  point the scripts at tools the resolution above does not find. Such a build caps its DRC
  reports, which the guide step accounts for (see "Determinism" below).
- ~4 GB of disk: the clone is 1.2 GB, the intermediate trees add about as much.

## Pipeline

```
PCBench/PCBs/<name>/processed.kicad_pcb   (public clone: KiCad 5 file format, no project file)
   │  0. convert_v9.py     PCBENCH_PCBS_ROOT → PCBENCH_V9_ROOT
   │     pcbnew load + save: KiCad 9 format plus the .kicad_pro that KiCad 9 derives
   │     from the legacy (setup …) block; metadata.json / final.json copied along
   │  1. drc_fix_v9.py     PCBENCH_V9_ROOT → PCBENCH_NEWDRC_OUT
   │     patch the minimum .kicad_pro entries (rule_severities → ignore, a few
   │     rules floors → 0, hole-to-hole minimum = min(0.25 mm, Default net-class
   │     clearance)), keep boards whose DRC passes under KiCad 9 (the D3
   │     candidate pool). Kept boards land as <name>/processed_v9.kicad_pcb +
   │     .kicad_pro; _results.json / _failures.json summarize the run
   │  2. make_guide.py --base-dir $PCBENCH_NEWDRC_OUT --stem processed_v9 --suffix _guide_v3
   │     per-net uniform-width guide board + .kicad_pro + *_unrouted strip
   │     (algorithm details: GUIDE_GENERATION.md)
   │  3. sort_prefix.py    PCBENCH_NEWDRC_OUT → PCBENCH_SORTED_OUT
   │     order = (pins, nets, components) ascending, read from final.json + the
   │     footprint count; folders copied as <NNNN>_<name>/ in that order next to
   │     pcb_characteristics_exacad_sorted.csv
   ▼
exacad_sorted/   (consumed via configs/paths.yaml `pcbench_exacad`;
                  split JSON: experiments/kdd/d3_dataset/build.py)
```

## Run

```bash
git clone --depth 1 https://github.com/PCBench/PCBench.git /data/PCBench    # main @ dec3be75
export CADAGENT_DATA_ROOT=/data/cadagent                 # your dataset root
export PCBENCH_PCBS_ROOT=/data/PCBench/PCBs
export PCBENCH_V9_ROOT=/data/pcbench_work/v9
export PCBENCH_NEWDRC_OUT=/data/pcbench_work/newdrc
export PCBENCH_SORTED_OUT=$CADAGENT_DATA_ROOT/pcbench/exacad_sorted
PP=tools/datagen/pcbench_prep
python $PP/convert_v9.py --workers 16        # --limit N: only the first N boards (a trial run)
python $PP/drc_fix_v9.py --workers 16        # --limit N likewise
python $PP/make_guide.py --base-dir $PCBENCH_NEWDRC_OUT --stem processed_v9 --suffix _guide_v3 --workers 16
python $PP/sort_prefix.py
bash experiments/kdd/d3_dataset/run.sh --out /tmp/d3.json   # optional: regenerate the split and diff it against configs/datasets/d3.json
```

The root README's Quick start §3 is this block on a `--limit 30` trial subset (its
`quickstart` walk runs it by machine). `sort_prefix.py` owns the `<NNNN>_<name>/` entries of
its output directory: a re-run replaces them and removes the ones an earlier run left, so
a full run after a trial leaves exactly the full set.

## Expected result

Verified end to end on 2026-08-31 with the source-built tools (PCBench `main` @ `dec3be75`,
`kicad-cli` and `pcbnew` 9.0.8 from `BUILD_CLI=1 BUILD_PCBNEW=1`, `--workers 16` on a 64-core
host):

| step | boards | wall time | note |
|---|---|---|---|
| clone | 1 183 entries | ~1 min | 1 182 board folders (+ `master_metadata.json`), each with `processed.kicad_pcb` and `final.json` |
| 0 convert | 1 182 / 1 182 | 15 s | |
| 1 DRC filter | **679** pass / 503 fail | 51 s | the same 679-board set the paper used (0 boards differ) |
| 2 guide | 679 / 679 pass | 82 s | deterministic (rerun byte-identical); against the paper's boards: unrouted strips identical 679/679, guide boards 639/679 (below) |
| 3 sort | 679 `<NNNN>_<name>/` + CSV | 2 s | folder order and CSV values identical to the paper's `pcb_characteristics_exacad_sorted.csv` |

Determinism. KiCad assigns fresh UUIDs on the step-0 round trip, so two runs are not
byte-identical; geometry, nets, design rules and the DRC verdicts are, and steps 1–3 are
deterministic (rerunning the guide step reproduces every file byte for byte). Step 2 needed
two guards to get there, both against `kicad-cli pcb drc`, whose copper-clearance test runs
on a thread pool: (a) by default it reports only the first violation per track, and which
one is "first" is a race — `make_guide.py` passes `--all-track-errors`; (b) a stock KiCad
stops reporting a violation type after 199 (clearance: 499) hits and kicad-cli cannot raise
that cap, so on boards with more violations the reported subset is a race — with a capped
CLI `make_guide.py` treats a count at or above those values as unusable (`drc_truncated`)
and narrows every reducible net instead of the offenders. The engine's `kicad-cli` has no
cap (its `drc_engine.cpp` patch reports every violation), so there the guard is switched
off (`kicad_tools.kicad_cli_uncapped()`) and the complete violation list picks the offender
nets on every board — 15 boards reach the stock cap values with real violations and now
narrow only their offenders instead of every net.

The 2026-08-30 run — apt KiCad 9.0.9, `--workers 16`, ~7 min — produced the same
counts with the narrow-all behaviour on those 15 boards (its output was first registered as
`d3_v2`). Compared file by file after removing what KiCad mints afresh on every convert
(`uuid` / `tstamp` / sheet `path` values and the item order they define), the engine-CLI run
differs from it exactly there: 13 of those 15 boards carry different (wider, offender-only)
per-net widths, the other 666 boards are identical on every file. A stock-CLI rebuild
therefore reproduces the set everywhere except those boards; the shipped `d3_v2` is the
engine-CLI output.

The paper's boards were generated before these guards, i.e. from one non-deterministic
sample: 40 of the 679 guide boards differ from them in the width chosen for a few nets
(and, as a consequence, in the per-width net classes the guide step writes), the other 639
are identical — the offender path brings the set closer to the paper's than the narrow-all
run did (634). The `.kicad_pro` written by the pcbnew round trip carries KiCad's defaults
for `min_copper_edge_clearance` / `min_hole_clearance` / `min_hole_to_hole`
(0.5 / 0.25 / 0.25 mm); step 1 zeroes the first two and sets the third to min(0.25 mm,
Default net-class clearance) — the value the paper's set carries on all 679 boards, and one
that routers and DRC honor as authoritative once the `.kicad_pro` is loaded. Against the
paper's set: folder order and CSV identical, `.kicad_pro` design rules identical 679/679,
boards and unrouted strips identical 679/679 (UUIDs aside) — the 40-board width choice is
the only remaining difference. The rebuilt set ships as the logical dataset `d3_v2`
([configs/datasets/d3_v2.json](../../../configs/datasets/d3_v2.json), dataset dir
`pcbench_exacad_v2`); the paper's numbers were measured on `d3`.

`experiments/kdd/d3_dataset/build.py` reproduces the shipped
[configs/datasets/d3.json](../../../configs/datasets/d3.json) from the CSV + layout up
to two recorded manual edits: `0170_hackaday_esp-14_power_meter__autosave-esp-14` is
dropped from `medium.train` (an autosave remnant with zero effective clearance), and
the `medium.test` swap listed under the JSON's `_test_overrides` key.

Related: `engine/pcbnew_prep/` converts prepared boards to DSN/ORP for the
rule-based baselines (that step imports `pcbnew` directly, hence lives in the
engine bundle; the same source-built module serves it with `PYTHONPATH=build_rl/pcbnew`).
