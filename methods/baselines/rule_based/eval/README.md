# `methods/baselines/rule_based/eval/` — Paper RQ2 baseline eval logs (d2a / d3a)

Per-board `eval/metrics.py` JSON for every routed `.kicad_pcb` that produced
the paper's RQ2 numbers (Table `tab:rq2` mean, Table `tab:rq2-std` mean+/-std).
Aggregating these logs with `_scripts/aggregate.py` exactly reproduces the
three baseline rows in both tables.

## Layout

```
methods/baselines/rule_based/eval/
├── README.md                           (this file)
├── RQ2_TABLE.md                        paper Table values (mean and mean+/-std)
│
├── d2a/                                 Synth 2L, n=128, full set
│   ├── Freerouting/seed{0,1,2,3}/logs/board_NNNNN_Freerouting.json
│   ├── OrthoRoute/seed0/logs/board_NNNNN_OrthoRoute.json
│   └── KiCadRoutingTools/seed0/logs/board_NNNNN_KiCadRoutingTools.json
│
├── d3a/                               PCBench, fair-95 subset used in table
│   ├── Freerouting/seed{0,1,2,3}/logs/<board-id>_Freerouting.json
│   ├── OrthoRoute/seed0/logs/<board-id>_OrthoRoute.json
│   └── KiCadRoutingTools/seed0/logs/<board-id>_KiCadRoutingTools.json
│
└── _scripts/
    ├── aggregate.py                    canonical aggregator — run to reproduce RQ2
    ├── aggregate_routable_only.py      upstream original (kept for reference)
    ├── aggregate_eval.py               upstream general-purpose aggregator
    ├── pcbench_fair95.txt              95-board fair-comparison subset
    └── README_upstream.md              upstream notes on the eval pipeline
```

The `logs/*.json` files are **gitignored** (large, redundant). The aggregator
reads them at runtime; the table snapshot is checked in as `RQ2_TABLE.md`.
Each `seed*/` has a `SOURCE_PATH.txt` recording the canonical NFS path the
logs were copied from.

## Aggregation rule (set by paper)

| metric | rule |
|---|---|
| Routability | full-set mean over all boards |
| DRV / WL / Via | mean over boards with `routability == 1.0` only ("routable-only") |
| Time | reported separately, from per-tool wall-clock csv (not from these logs) |
| Std (Freerouting only) | sample stdev `n-1` across 4 seeds, computed on per-seed routable-only means |
| d3a | restricted to the 95 boards in `_scripts/pcbench_fair95.txt` (excludes 0004, 0024, 0035, 0077, 0096) |

OrthoRoute and KiCadRoutingTools are deterministic in this setup — single
seed, no stdev reported.

## Reproduce the table

```bash
python methods/baselines/rule_based/eval/_scripts/aggregate.py
```

Output matches the paper exactly:

```
dataset  method              seeds     n           Rout.              DRV                 WL              Via
--------------------------------------------------------------------------------------------------------------
d2a       Freerouting             4   128   1.00 +/- 0.00   0.00 +/- 0.00   407.38 +/- 2.14   2.48 +/- 0.01
d2a       OrthoRoute              1   128            0.24           43.00            247.13            8.00
d2a       KiCadRoutingTools       1   128            1.00            0.00            373.87            8.09
d3a     Freerouting             4    95   0.98 +/- 0.01   0.09 +/- 0.02   158.82 +/- 2.18   0.72 +/- 0.10
d3a     OrthoRoute              1    95            0.99          111.93            294.04           21.86
d3a     KiCadRoutingTools       1    95            0.94            1.01            156.01            3.74
```

## Source paths (canonical result tree)

All canonical logs were copied verbatim from the archived benchmark result
tree (`bench_results_evalfinal/`, not distributed with this repo); each seed
dir's `SOURCE_PATH.txt` records its exact source.

| target dir | canonical source | n |
|---|---|---|
| `d2a/Freerouting/seed0` | `synth_2L_v2_test/freerouting_via1x` | 128 |
| `d2a/Freerouting/seed1` | `synth_2L_v2_test/freerouting_via1x_seed1` | 128 |
| `d2a/Freerouting/seed2` | `synth_2L_v2_test/freerouting_via1x_seed2` | 128 |
| `d2a/Freerouting/seed3` | `synth_2L_v2_test/freerouting_via1x_seed3` | 128 |
| `d2a/OrthoRoute/seed0` | `synth_2L_v2_test/orthoroute_gpu_20260506` | 128 |
| `d2a/KiCadRoutingTools/seed0` | `synth_2L_v2_test/kicadroutingtools_via1x` | 128 |
| `d3a/Freerouting/seed0` | `PCBench/freerouting_via1x_s00_r00` | 100 |
| `d3a/Freerouting/seed1` | `PCBench/freerouting_via1x_s00_r00_seed1` (cadagent/eval mining) | 100 |
| `d3a/Freerouting/seed2` | `PCBench/freerouting_via1x_s00_r00_seed2` (cadagent/eval mining) | 100 |
| `d3a/Freerouting/seed3` | `PCBench/freerouting_via1x_s00_r00_seed3` (cadagent/eval mining) | 99 |
| `d3a/OrthoRoute/seed0` | `PCBench/orthoroute_gpu_20260506` | 100 |
| `d3a/KiCadRoutingTools/seed0` | `PCBench/kicadroutingtools_via1x_s00_r00` | 100 |

## Reproducing the logs from this repo

The unified runner `methods/baselines/rule_based/run_rule_based_routers.py` reproduces these per-board
logs end-to-end (raw routing -> .kicad_pcb -> eval). For the two
deterministic baselines:

```bash
# KiCadRoutingTools — bit-perfect match on d2a verified.
python methods/baselines/rule_based/run_rule_based_routers.py --baseline krt --dataset synthetic_2l \
    --output-root methods/baselines/rule_based/_run_outputs

# OrthoRoute — close match; small per-board DRV variation due to GPU
# tie-breaking. Aggregate-level numbers match the paper.
python methods/baselines/rule_based/run_rule_based_routers.py --baseline orthoroute --dataset synthetic_2l \
    --output-root methods/baselines/rule_based/_run_outputs
```

Freerouting is stochastic — each invocation produces a different SES, so
exact reproduction requires running 4 seeds and aggregating:

```bash
python methods/baselines/rule_based/run_rule_based_routers.py --baseline freerouting --dataset synthetic_2l \
    --seeds 4 --output-root methods/baselines/rule_based/_run_outputs
```
