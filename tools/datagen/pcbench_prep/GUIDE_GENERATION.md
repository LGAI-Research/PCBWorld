# `_guide` board generation (`make_guide.py`)

Step 2 of the chain in [README.md](README.md). For every board folder produced by
`drc_fix_v9.py`, the actual per-net track widths of the routed board are folded into
the board's net classes, so a router that only reads net classes reproduces the
original per-net widths. The routed board, its net-class settings and a routing-free
copy are written next to the input.

Each net gets **one** width: the widest track observed on that net, narrowed only if
that choice breaks DRC.

## 1. Input files

Per board folder under `--base-dir` (`<newdrc-out>/<board-name>/`), with
`<stem>` = `--stem` (default `processed_v9`):

| File | Role |
|---|---|
| `<stem>.kicad_pcb` | The routed board. Net declarations, `(segment ...)` / `(arc ...)` widths, and — on legacy-format boards — the `(net_class ...)` blocks are read from it |
| `<stem>.kicad_pro` | Net classes (`net_settings.classes`, `netclass_patterns`) and DRC rules. Optional: a missing or unparsable file is treated as `{}` |

A folder with no `<stem>.kicad_pcb` is skipped.

## 2. Output files

Written into the same folder, with `<suffix>` = `--suffix` (default `_guide`; the D3
set is built with `_guide_v3`):

| File | Content |
|---|---|
| `<stem><suffix>.kicad_pcb` | The board with every segment/arc width rewritten to its net's guide width. On legacy boards the `(net_class ...)` blocks are rebuilt as well |
| `<stem><suffix>.kicad_pro` | A copy of the input pro whose `net_settings.classes` is replaced by the guide classes (and whose `netclass_patterns` are repointed at them) |
| `<stem><suffix>_unrouted.kicad_pcb` | The guide board with all `(segment ...)`, `(arc ...)` and `(via ...)` blocks removed. Zones are **not** removed |

The D3 split json names `processed_v9_guide_v3.kicad_pcb` as its `board_filename`,
i.e. the routed guide board is the artifact the benchmark loads.

All three files are written whether or not the board ends up DRC-clean — a board that
never reached zero violations is saved from its best attempt and reported as `fail` in
the log (see §7).

## 3. Invocation

```bash
python3 tools/datagen/pcbench_prep/make_guide.py --base-dir <newdrc-out>
```

| Flag | Default | Meaning |
|---|---|---|
| `--base-dir` | *(required)* | Root holding the per-board folders (the `drc_fix_v9.py` output directory); subfolders are scanned in name order |
| `--stem` | `processed_v9` | Input file stem inside each folder |
| `--suffix` | `_guide` | Suffix of the generated files |
| `--workers` | `8` | Process pool size; `<= 1` runs serially |
| `--limit` | `0` | `0` = all folders, `N` = only the first `N` of the folder list |
| `--targets` | *(all)* | Explicit folder names to process instead of scanning `--base-dir` |
| `--log` | `/tmp/make_guide_log.json` | Per-board result log (overwritten on each run) |
| `--verbose` | off | Per-board progress and tracebacks (serial mode prints one line per board) |

External requirements (resolved by [kicad_tools.py](kicad_tools.py); the run prints the
choice on its first line):

- `kicad-cli` — every DRC trial shells out to `kicad-cli pcb drc --format json
  --severity-error --all-track-errors`. Default: the engine's own build
  (`build_rl/kicad/kicad-cli`, from `BUILD_CLI=1 BUILD_PCBNEW=1 bash engine/build_rl_router.sh`),
  else `kicad-cli` on `PATH`; `KICAD_CLI` overrides.
- an interpreter that imports `pcbnew` — used for zone refill before each DRC run. Default:
  the running interpreter with the engine's `build_rl/pcbnew` on `PYTHONPATH`, else
  `/usr/bin/python3` (an apt-installed KiCad); `PCBNEW_PYTHON` overrides. The helper script is
  written to `/tmp/_make_guide_refill_helper.py`. If the refill fails or times out, DRC runs on
  the un-refilled board instead, which can change zone-related violation counts.

## 4. Flow

```
<stem>.kicad_pcb + <stem>.kicad_pro
   ├─ extract_net_id_to_name()        → {net_id: name}
   ├─ extract_pro_net_classes()       → v9 classes + netclass_patterns
   ├─ extract_pcb_net_class_blocks()  → legacy (net_class ...) blocks
   │      is_v9 = no pcb net_class blocks and at least one pro class
   ├─ derive_net_to_class()           → {net_name: class}
   │      explicit `nets` list > first matching netclass_pattern > Default
   ├─ extract_net_width_range()       → {net_id: (min_mm, max_mm)}
   └─ extract_setup_legacy_rules()    → legacy DRC rule floors
              ↓
   trial loop, N_TRIAL = 5, every net starting at t = 0 (= its max width)
   ┌───────────────────────────────────────────────────────────────┐
   │ _compute_widths_per_t()   → per-net width from t              │
   │ apply_net_widths_to_tracks() → rewrite segment/arc widths     │
   │ build_groups() + assign_class_names() → Default / guide_W*    │
   │ build_guide_pro()            → guide .kicad_pro               │
   │ rebuild_pcb_net_class_blocks() → legacy boards only           │
   │ run_drc()                    → kicad-cli violations           │
   │   zero violations → accept and stop                           │
   │   otherwise: extract_offending_net_ids() → raise t by one     │
   │              step (1/(N_TRIAL-1)) for the offending nets only │
   │              (no net identified → narrow every reducible net) │
   └───────────────────────────────────────────────────────────────┘
              ↓ loop exhausted
   one final attempt with every net that has segments pinned to t = 1 (its
   min width); if that is clean it wins, otherwise the attempt with fewer
   violations is kept
              ↓
   write <suffix>.kicad_pcb / .kicad_pro, then
   remove_routing_v9() → <suffix>_unrouted.kicad_pcb
```

## 5. Key functions

All in [make_guide.py](make_guide.py).

| Function | Role |
|---|---|
| `extract_net_id_to_name` | Parses the top-level `(net id name)` declarations into `{id: name}` |
| `extract_net_width_range` | Per-net `(min, max)` width over all `(segment ...)` and `(arc ...)` blocks (multi-line included), excluding net `0` |
| `extract_pro_net_classes` | Reads `net_settings.classes` and `netclass_patterns` from a v9 `.kicad_pro` |
| `extract_pcb_net_class_blocks` | Reads the `(net_class ...)` blocks of a legacy `.kicad_pcb` |
| `derive_net_to_class` | Resolves each net to a class: explicit `nets` entry, else first matching pattern (`fnmatchcase`, case-sensitive), else `Default` |
| `extract_setup_legacy_rules` | Maps legacy `(setup ...)` DRC keys (`trace_min`, …) onto the modern rule names |
| `_detect_default_width` | Width used for nets with no segments: `Default` class `track_width`, else the smallest observed width, else `0.25` |
| `_compute_widths_per_t` | Width of each net from its `t`: `max·(1−t) + floor·t`, where floor is the net's observed min raised to `min_track_width`. Nets with no segments stay at the default width |
| `apply_net_widths_to_tracks` | Rewrites the `(width ...)` of every segment/arc block to its net's value |
| `extract_uuid_to_net` / `extract_offending_net_ids` | Map DRC violation items back to net ids — by item UUID, falling back to the `[net_name]` pattern in the description |
| `build_groups` | Groups nets by `(rounded width, preserved class parameters)`; parameters are copied verbatim from the class the net belonged to |
| `assign_class_names` | Names each group: the smallest width becomes `Default`, the rest `guide_W<width>`; a name collision gets a `_v2`, `_v3`, … suffix |
| `build_guide_pro` | Deep-copies the original pro, replaces `net_settings.classes` with the guide classes, and repoints `netclass_patterns` (a pattern equal to a net name → that net's guide class, anything else → `Default`) |
| `rebuild_pcb_net_class_blocks` | Legacy boards only: drops the old `(net_class ...)` blocks and inserts the guide ones before the first `(module`/`(footprint` |
| `remove_routing_v9` | Drops the `(segment ...)`, `(arc ...)` and `(via ...)` blocks |
| `run_drc` / `drc_passes` | Writes the candidate pcb+pro to a temp dir, refills zones, runs `kicad-cli pcb drc`, and counts violations by type; a candidate passes only at zero violations |
| `process_board` | Per-board driver: parse → trial loop → save → result record |

## 6. Class parameters and naming

- Preserved per-class parameters are `clearance`, `via_dia`, `via_drill`, `uvia_dia`,
  `uvia_drill` — only those actually present on some input class. They are carried into
  the guide class unchanged; only the track width is new. In the `.kicad_pro` they are
  written back under their KiCad names (`via_diameter`, `microvia_diameter`, …).
- Two nets share a guide class only if both their width **and** their preserved
  parameters match, so a board that uses different clearances per class keeps that
  distinction.
- Class names are formatted with up to four decimals, trailing zeros stripped: a 1.0 mm
  group becomes `guide_W1`, a 0.8128 mm group `guide_W0.8128`. Group keys round the
  width to 6 decimals; rewritten segment widths are printed with up to 5 decimals.
- The `Default` class carries no `nets` list in the pro (KiCad convention); every other
  guide class lists its nets sorted.
- The legacy `.kicad_pcb` blocks are emitted as:

  ```
    (net_class Default "This is the default net class."
      (clearance 0.2)
      (trace_width 0.25)
      (via_dia 0.8)
      (via_drill 0.4)
      (add_net "Net-(D1-Pad2)")
    )
  ```

  A group missing `via_dia`/`via_drill` is dropped from the legacy blocks (those fields
  are mandatory there). Net names containing `(`, `)`, `/`, `+`, `-` or a space are
  quoted.

## 7. Result log

`--log` receives a JSON list, one record per board:

| Field | Meaning |
|---|---|
| `folder` | Board folder name |
| `status` | `pass` (final candidate DRC-clean) · `fail` (files written, violations remain) · `skip` · `error` |
| `trial` | 1…`N_TRIAL` for a trial-loop result, `N_TRIAL + 1` when the all-min fallback was taken |
| `final` | Violation counts by type for the saved candidate |
| `n_nets_at_max` / `n_nets_reduced` / `n_nets_at_min` | How many nets kept their max width, were narrowed at all, or hit their min |
| `offender_history` | Per failing trial: number of nets blamed by DRC and number actually narrowed |
| `msg` | Reason for `skip` / `error` |

`skip` reasons: `no pcb` (no `<stem>.kicad_pcb`), `no Default class` (neither the pro nor
the pcb defines a class named `Default`), `no routed segments` (no segment/arc carries a
net). `error: all DRC trials errored` means `kicad-cli` produced no usable report for any
trial — check the `tools:` line the run printed first (which `kicad-cli` it resolved).

The run also prints a status tally at the end.

## 8. Pitfalls

| Issue | What to do |
|---|---|
| **`--stem` mismatch** | The stem must match the files on disk. `drc_fix_v9.py` writes `processed_v9.kicad_pcb` / `.kicad_pro`; a different stem silently skips every folder with `no pcb` |
| **Nets with mixed widths** | The widest track on the net wins — the design intent is assumed to be the wider one. Only nets that DRC blames get narrowed, and only far enough to clear the violation |
| **Nets without segments** | A net routed only through a zone, or unconnected, has no width to observe and is pinned to the default width, so it lands in the `Default` class |
| **Widening can break DRC** | Raising a net to its max width can create clearance violations that the original board did not have; that is exactly what the trial loop detects and walks back |
| **Zone refill** | Without a `pcbnew`-capable interpreter (the `tools:` line names it), DRC sees unfilled zones and the accept/reject decision is made on different violation counts |
| **`fail` boards are still written** | `status: fail` means the guide files exist but carry known violations; filter on the log before using a batch |
| **Runtime** | Each board runs DRC up to `N_TRIAL + 1` times, each with a zone refill; `--workers` parallelises across boards, not within one |
