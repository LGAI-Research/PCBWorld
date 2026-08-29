# `experiments/kdd/d3_dataset/` — D3 real-board split builder

Rebuild [`configs/datasets/d3.json`](../../../configs/datasets/d3.json),
the boards-json consumed by the LLM eval split aliases:

| alias | difficulty | split | n  |
|-------|------------|-------|---:|
| `d3a` | easy       | test  | 99 |
| `d3b` | medium     | test  | 10 |
| `d3c` | hard       | test  | 10 |

The wrapper is a thin shell around
[`experiments/kdd/d3_dataset/build.py`](build.py).

## Usage

```bash
# rebuild in place against the PCB-bench source
bash experiments/kdd/d3_dataset/run.sh

# write to a different path (e.g. for a diff)
bash experiments/kdd/d3_dataset/run.sh --out /tmp/d3.json

# override the source dataset
bash experiments/kdd/d3_dataset/run.sh \
    --sorted-dir /path/to/exacad_sorted \
    --csv        /path/to/pcb_characteristics_exacad_sorted.csv

# forward extra knobs (anything after `--`) straight to build.py
bash experiments/kdd/d3_dataset/run.sh -- --medium-n 20 --hard-trim-frac 0.05
```

Environment overrides understood by `run.sh`:

* `T3_SORTED_DIR` — default for `--sorted-dir`
  (fallback `$CADAGENT_DATA_ROOT/pcbench/exacad_sorted`).

## Difficulty rule

The source CSV is monotone non-decreasing in `pins`, so CSV row order ≡
on-disk difficulty order. The builder splits it as:

* rows `[0, --easy-rows)`                              → **easy** (100 boards)
* rows `[--easy-rows, ...)` with `pins ≤ T`            → **medium** (287)
* rows `[--easy-rows, ...)` with `pins > T`            → **hard** (292)

where `T = --medium-pin-threshold` (default 100).

These are the raw CSV classifications. One board is dropped from the pool
outright (reason + name in
[`configs/datasets/README.md`](../../../configs/datasets/README.md)) and it
falls in the medium tier, so the shipped
[`configs/datasets/d3.json`](../../../configs/datasets/d3.json) carries 286
medium boards, 678 in total.

## Test-set selection

* **easy.test (99)** — every easy board *except* the one matching
  `--easy-train-only-glob` (default `0096_*`). That one stays in
  `easy.train` only, as a fixed single-board sanity sample.
* **medium.test (10) / hard.test (10)** — for each:
  1. keep only `layer == 2` boards (4-layer / 8-layer support is
     untested in current routing pipeline);
  2. **hard only**: drop the top `--hard-trim-frac` (default 10%) of
     pins as outliers — the long tail (max ≈ 2100 pins) would otherwise
     swamp the higher deciles;
  3. sort by `(pins, nets, sample)` for determinism;
  4. split into `--medium-n` / `--hard-n` equal-size quantile bins;
  5. take the lower-median pin board from each bin.

Train sets always contain every board in that difficulty bucket — this
builder never down-samples train.

## Output schema

```json
{
  "easy":   {"train": ["0001_…", ...], "test": ["0001_…", ...]},
  "medium": {"train": [...],           "test": [...]},
  "hard":   {"train": [...],           "test": [...]},
  "dataset_dirs": {
    "train": "<absolute path to exacad_sorted>",
    "test":  "<absolute path to exacad_sorted>"
  }
}
```

Board names carry their 4-digit folder prefix (e.g. `0113_maytal_Maytal`)
so `dataset_dirs.{train,test} + "/" + name` resolves directly to the
on-disk folder.

## Related

* Quickstart split aliases: [`configs/quickstart/kdd/splits.json`](../../../configs/quickstart/kdd/splits.json)
* Underlying generator:     [`experiments/kdd/d3_dataset/build.py`](build.py)
* Consumers:                `experiments/kdd/table1_llm/run.sh`, `experiments/kdd/table2/run_*.sh`
