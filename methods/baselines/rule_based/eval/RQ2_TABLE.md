# RQ2 baseline numbers — paper snapshot

Three rule-based baselines on d2a (Synth 2L, n=128) and d3a (PCBench fair-95).
Numbers are computed by `_scripts/aggregate.py` from `T{2,3-A}/<algo>/seed*/logs/`.

Rule recap:
- Routability = full-set mean.
- DRV / WL / Via = mean over boards with routability=1.0.
- Freerouting reports mean +/- sample stdev (n-1) across 4 seeds; OrthoRoute
  and KiCadRoutingTools are deterministic (single seed).
- Time is taken from the per-tool wall-clock csv (not the per-board eval
  JSON) and is reproduced here for completeness only.

## Main table (`tab:rq2`, mean)

### d2a (Synth 2L)

| Method            | Rout. | DRV   | WL     | Via  | Time |
|---                | ---:  | ---:  | ---:   | ---: | ---: |
| Freerouting       | 1.00  | 0.00  | 407.38 | 2.48 | 2.55 |
| OrthoRoute        | 0.24  | 43.00 | 247.13 | 8.00 | 2.54 |
| KiCadRoutingTools | 1.00  | 0.00  | 373.87 | 8.09 | 0.82 |

### d3a (PCBench, fair-95)

| Method            | Rout. | DRV    | WL     | Via   | Time |
|---                | ---:  | ---:   | ---:   | ---:  | ---: |
| Freerouting       | 0.98  | 0.09   | 158.82 | 0.72  | 3.90 |
| OrthoRoute        | 0.99  | 111.93 | 294.04 | 21.86 | 2.20 |
| KiCadRoutingTools | 0.94  | 1.01   | 156.01 | 3.74  | 0.65 |

## Per-seed std (`tab:rq2-std`, mean +/- sample stdev n-1)

### d2a (Synth 2L)

| Method            | Rout.         | DRV           | WL              | Via           | Time          |
|---                | ---:          | ---:          | ---:            | ---:          | ---:          |
| Freerouting       | 1.00 +/- 0.00 | 0.00 +/- 0.00 | 407.38 +/- 2.14 | 2.48 +/- 0.01 | 2.55 +/- 0.04 |
| OrthoRoute        | 0.24          | 43.00         | 247.13          | 8.00          | 2.54          |
| KiCadRoutingTools | 1.00          | 0.00          | 373.87          | 8.09          | 0.82          |

### d3a (PCBench, fair-95)

| Method            | Rout.         | DRV           | WL              | Via           | Time          |
|---                | ---:          | ---:          | ---:            | ---:          | ---:          |
| Freerouting       | 0.98 +/- 0.01 | 0.09 +/- 0.02 | 158.82 +/- 2.18 | 0.72 +/- 0.10 | 3.90 +/- 0.51 |
| OrthoRoute        | 0.99          | 111.93        | 294.04          | 21.86         | 2.20          |
| KiCadRoutingTools | 0.94          | 1.01          | 156.01          | 3.74          | 0.65          |
