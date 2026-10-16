# V10 guide generation (`make_guide.py`)

Step 2 in the KiCad 10.0.7 chain described in [README.md](README.md). Each DRC-passing
`processed_v10` board is transformed into a per-net uniform-width guide. The guide board,
project settings, and routing-free variant are written beside the input.

## Inputs and outputs

For every folder under `--base-dir`, with `--stem processed_v10`:

| File | Role |
|---|---|
| `processed_v10.kicad_pcb` | Routed board; net declarations and observed track widths are read from it |
| `processed_v10.kicad_pro` | Net classes and DRC settings |

With `--suffix _guide_v3`, the files written are:

| File | Content |
|---|---|
| `processed_v10_guide_v3.kicad_pcb` | Guide board with one track width per net |
| `processed_v10_guide_v3.kicad_pro` | Guide net classes and preserved project settings |
| `processed_v10_guide_v3_unrouted.kicad_pcb` | Guide board with segments, arcs, and vias removed |

A valid DRC-rejected guide (`status: fail`) retains its historical best-attempt files, but the
v10 sorter excludes it. A tool, refill, parse, or missing-input error is `status: error` or
`skip`, is recorded in the JSON log, and makes the stage exit nonzero.

## Invocation

Run from the `/root/gnn` monorepo root through `scripts/pcbw`. The launcher activates the
`pcbworld` environment and exposes the Engine `pcbnew` build; avoid invoking the script with an
arbitrary system `python`.

```bash
export PCBWORLD_DATA_ROOT=/data/pcbworld
scripts/pcbw python third_party/PCBWorld/tools/datagen/pcbench_prep/make_guide.py \
  --base-dir "$PCBWORLD_DATA_ROOT/pcbench_work/newdrc_v10" \
  --stem processed_v10 --suffix _guide_v3 --workers 16 \
  --log "$PCBWORLD_DATA_ROOT/pcbench_work/newdrc_v10/_guide_results.json"
```

The script requires the Engine's uncapped KiCad 10.0.7 CLI and Engine `pcbnew` build, plus the
independent stock KiCad 10.0.7 CLI/Python. It checks all four before processing. See
[README.md](README.md) for explicit toolchain overrides and host paths.

## Search and DRC behavior

Each net starts at its widest observed track width. The guide search trials use the current
class settings, rewrite every segment/arc width for the candidate, refill zones, and run
`kicad-cli pcb drc --format json --severity-error --all-track-errors`. A clean, validated DRC
report accepts the candidate. With violations, only implicated nets are narrowed; if the
reported violation list is capped or cannot identify a net, all reducible nets are narrowed.
After the trial loop, one final all-min-width attempt is made. The candidate with the lower
valid violation count is retained when neither is clean.

A DRC command is not a valid report unless it exits zero and writes parseable JSON containing
a `violations` array of objects. Timeout, nonzero exit, missing report, malformed JSON, or
failed zone refill are stage errors, not zero violations. If every trial errors, that board is
reported as `error`; if a clean or violating report was obtained, its pass/rejection result is
reported explicitly. The log contains one result per input folder and includes stage errors.

## Result statuses

| Status | Meaning |
|---|---|
| `pass` | Saved guide has a successful zero-violation final DRC report |
| `fail` | Best valid candidate still has DRC violations; sorter excludes it |
| `skip` | Required guide input is absent or the board has no usable default class/routed segments |
| `error` | DRC, zone refill, parsing, or other board processing failed |

`--log` is the complete result list. `fail` is a legitimate guide DRC rejection; `error` and
`skip` indicate an incomplete stage and cause a nonzero exit.
