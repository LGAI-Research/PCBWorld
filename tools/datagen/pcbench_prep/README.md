# `tools/datagen/pcbench_prep/` — PCBench → `exacad_sorted` (D3) reconstruction

Rebuilds the D3 real-board benchmark set from a clone of the public
[PCBench](https://github.com/PCBench/PCBench) collection. Board content is never
redistributed by this repository — users clone PCBench and run this chain.

These scripts shell out to `kicad-cli` (KiCad 9) and, for zone refill, to a
`pcbnew`-python child process; nothing here imports GPL code in-process.

## Pipeline

```
PCBench/PCBs  (public clone; KiCad-9-format boards = the version_9 form)
   │  1. drc_fix_v9.py      PCBENCH_V9_ROOT → PCBENCH_NEWDRC_OUT
   │     patch the minimum .kicad_pro entries (rule_severities → ignore, a few
   │     rules floors → 0), keep boards whose DRC passes under KiCad 9 (the D3
   │     candidate pool). Kept boards land as <name>/processed_v9.kicad_pcb +
   │     .kicad_pro; _results.json / _failures.json summarize the run
   │  2. make_guide.py --base-dir <newdrc-out> --stem processed_v9 --suffix _guide_v3
   │     per-net uniform-width guide board + .kicad_pro + *_unrouted strip
   │     (algorithm details: GUIDE_GENERATION.md)
   │  3. difficulty sort + 4-digit prefix layout  — manual, not scripted here
   │     order = pcb_characteristics CSV sorted by pins (the CSV machinery ships
   │     in PCBench itself: Scripts/Data_extraction + statistics.ipynb);
   │     folders are copied as <NNNN>_<name>/ in that order
   ▼
exacad_sorted/   (consumed via configs/paths.yaml `pcbench_exacad`;
                  split JSON: experiments/kdd/d3_dataset/build.py)
```

Step 3 has no script in this tree: the board folders from step 2 are copied into
`exacad_sorted/` in CSV row order, each renamed to `<NNNN>_<name>` with a
4-digit index, alongside `pcb_characteristics_exacad_sorted.csv` — the layout
`experiments/kdd/d3_dataset/build.py` expects when it derives the difficulty
splits.

Related: `engine/pcbnew_prep/` converts prepared boards to DSN/ORP for the
rule-based baselines (that step imports `pcbnew` directly, hence lives in the
engine bundle).
