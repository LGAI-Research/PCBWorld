"""Replay plan-only (LLM API-sequence) baselines and emit rollout-stage artifacts only.

IMPORTANT -- conda env: run this in the default ``cadagent`` env (torch / pcbnew /
RLRouter), same as ``eval/pipeline.py``'s transformer rollout. This is the plan-only
counterpart to ``eval/rollout/rule_based.py``: a standalone Stage-1
entrypoint that produces a flat ``boards/`` + ``per_rollout.csv`` that the shared
Stage-2 (DRC) / Stage-3 (aggregate) in ``eval/pipeline.py`` then consume unchanged.

What it does (NO LLM calls): for each saved plan-only response it

  1. reads the recorded action sequence from
     ``<responses-root>/apiseq_<split>_zs_<model>/per_board/test__<board_id>/sample_<NN>.response.txt``
     (pure routing actions: net_select / start_route / make_line / make_via /
     finish / net_end -- no design-rule-setting calls),
  2. replays it through a fresh ``PCBWorld`` built on the *native* board
     (``<dataset>/<board_id>.kicad_pcb`` + its companion ``.kicad_pro``) with
     **``use_yaml_drc_fallback=False``**, and
  3. saves the routed ``.kicad_pcb`` (``RLRouter.save`` co-locates a native
     ``.kicad_pro``) under the uniform cell layout.

Why ``use_yaml_drc_fallback=False``
-----------------------------------
With the fallback enabled the engine ignores each board's native ``.kicad_pro``
and applies the loose template rules of ``configs/drc/default.yaml``
(clearance 0.15 / track 0.20 / via 0.60). The router draws tracks at the active
netclass width, so the replay would emit 0.2 mm tracks / 0.6 mm vias, which then
score as spurious DRC violations against the native per-board rules
(track 0.3 / via 1.2 / clearance 0.3). Loading the native ``.kicad_pro`` replays
at native track/via widths, so geometry and scoring rules agree.

Output layout (mirrors eval/pipeline.py rollout stage; consumed by --stages eval,aggregate):

    <output-dir>/                      # the cell dir, e.g. .../d2a/plan_only_gpt-5.4
      ├── boards/<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb   (+ co-located .kicad_pro)
      ├── per_rollout.csv              (eval.eval_utils.PER_ROLLOUT_FIELDS; DRC-scoring
      │                                 cols blank/NaN until Stage 2 fills them)
      └── manifest.json

Then score with the shared post-hoc path::

    python eval/pipeline.py --stages eval,aggregate \
        --output-dir <output-dir> --rollout-mode parallel --n-envs 64 --check-angle 45

KiCad singleton note: ``RLRouter`` is a per-process singleton and not fork-safe,
so parallelism uses a ``forkserver`` ProcessPool; each worker builds and closes
its env per sample (never two live envs in one process).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent            # <repo root>/methods/llm_agent/rollout
_PROJECT_ROOT = _THIS_DIR.parents[2]                   # <repo root>
for _p in (_PROJECT_ROOT, _THIS_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from eval import eval_utils as u  # noqa: E402  (PER_ROLLOUT_FIELDS, write_csv)

log = logging.getLogger("rollout.plan_only")

from configs.loader.paths import resolve_dataset  # noqa: E402 — single-source path resolver
from configs.loader.schema import SCHEMA_VERSION  # noqa: E402 — shared manifest version

# Single source of truth (configs/paths.yaml): the responses live at the
# ``plan_only_responses`` dataset under $CADAGENT_DATA_ROOT; --responses-root
# overrides. Resolved lazily so importing this module never needs the data root.

# D2 synth model tags (the kdd cell is ``plan_only_<model>``). Order is cosmetic.
SYNTH_MODELS = [
    "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
    "qwen2.5-72b", "qwen3.5-122b", "qwen3.5-397b", "qwen3-coder-480b",
]

# ---------------------------------------------------------------------------
# Action parser for the recorded API-sequence grammar. Self-contained so this
# entrypoint needs no LLM-calling code.
# ---------------------------------------------------------------------------

from pcb_world.core.action_schema import MODE_LETTER_TO_INT as _MODE_LETTER_TO_INT

_ACTION_TYPE_MAP = {
    "net_select":  0,
    "start_route": 1,
    "net_end":     2,
    "make_line":   3,
    "make_via":    4,
    "finish":      5,
}

_ACTION_SCHEMA: dict[str, list[tuple[str, type]]] = {
    "net_select":  [("net_id", int)],
    "start_route": [("x_mm", float), ("y_mm", float), ("layer", int)],
    "net_end":     [],
    "make_line":   [("x_mm", float), ("y_mm", float), ("routing_mode", str)],
    "make_via":    [("x_mm", float), ("y_mm", float), ("routing_mode", str)],
    "finish":      [("routing_mode", str)],
}

_PARAM_DEFAULTS = {
    "x_mm": 0.0, "y_mm": 0.0, "layer": 1, "net_id": 0, "routing_mode": 2,
}


def parse_action_line(line: str) -> dict | None:
    """Return a step-ready action dict or ``None`` if the line is unparseable."""
    s = line.strip()
    if not s or s.startswith("#") or s.startswith(";"):
        return None
    m = re.match(r"<action>\s*(.*?)\s*</action>", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    tokens = s.split()
    if not tokens:
        return None
    name = tokens[0]
    if name not in _ACTION_SCHEMA:
        return None
    schema = _ACTION_SCHEMA[name]
    params: dict = {}
    values = tokens[1:]
    for i, (pname, conv) in enumerate(schema):
        if i < len(values):
            v = values[i]
            try:
                if pname == "routing_mode":
                    params[pname] = _MODE_LETTER_TO_INT.get(v, _PARAM_DEFAULTS[pname])
                    if v not in _MODE_LETTER_TO_INT:
                        try:
                            params[pname] = int(v)
                        except ValueError:
                            params[pname] = _PARAM_DEFAULTS[pname]
                elif conv is int:
                    params[pname] = int(float(v))
                else:
                    params[pname] = conv(v)
            except (ValueError, TypeError):
                params[pname] = _PARAM_DEFAULTS[pname]
        else:
            params[pname] = _PARAM_DEFAULTS[pname]
    return {"action_type": _ACTION_TYPE_MAP[name], **params}


_ACTIONS_TAG_RE = re.compile(r"<actions>\s*(.*?)\s*(?:</actions>|$)", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)\n```", re.DOTALL)


def extract_action_sequence(response: str) -> list[dict]:
    """Pull the action sequence out of a recorded LLM response."""
    m = _ACTIONS_TAG_RE.search(response)
    if m:
        body = m.group(1)
    else:
        m = _FENCE_RE.search(response)
        body = m.group(1) if m else response
    out: list[dict] = []
    for line in body.splitlines():
        a = parse_action_line(line)
        if a is not None:
            out.append(a)
    return out


# ---------------------------------------------------------------------------
# Replay one recorded response onto the native board (worker; runs in a child
# process under a forkserver pool -- must be picklable / import lazily).
# ---------------------------------------------------------------------------

def _replay_one(job: dict) -> dict:
    """Replay one ``sample_NN.response.txt`` and emit a PER_ROLLOUT_FIELDS row.

    Imports PCBWorld lazily so forkserver children pull a clean module/router.
    Leaves the DRC-scoring columns unfilled (blank; the runtime metric columns
    ``drv_errors_only_count``/``drc_violations`` come out NaN since env-DRC is
    off here) -- Stage 2 (eval/pipeline.py) fills the canonical DRC-scoring
    metrics against the now-native co-located ``.kicad_pro``.
    """
    from pcb_world.core.env import PCBWorld

    row = u.assemble_rollout_row(
        identity={
            "board_index": job["board_index"],
            "board_id": job["board_id"],
            "board_path": job["board_path"],
            "rollout_idx": job["rollout_idx"],
            "seed": job["seed"],
            "deterministic": True,
        },
        runtime={
            "engine_crash": False,
            "rollout_finish_time": time.time(),
        },
    )

    out_pcb = Path(job["out_pcb"])
    t0 = time.monotonic()
    try:
        actions = extract_action_sequence(Path(job["response_path"]).read_text())
    except OSError as exc:
        row.update(status="crashed", crashed=True,
                   error=f"read response: {exc}",
                   per_board_rollout_time=0.0)
        return row

    env = None
    try:
        # Native rules: load the board's companion .kicad_pro, do NOT fall back
        # to the configs/drc/default.yaml template.
        env = PCBWorld(
            board_path=job["board_path"],
            masking_rule="default",
            state_format="sexpr",
            max_steps=10_000,
            use_yaml_drc_fallback=False,
        )
        env.reset()
        u_0 = env._initial_unconnected
        steps_run = 0
        last_done = False
        for action in actions:
            if last_done:
                break
            try:
                _, _, terminated, truncated, _info = env.step(action)
                last_done = bool(terminated or truncated)
            except Exception:
                # Engine refusal / mask reject is logged inside env; keep going.
                pass
            steps_run += 1

        # Geometry-only snapshot (run_drc=False): Stage 2 owns canonical DRC.
        snap = env._engine.get_reward_snapshot(run_drc=False)
        u_t = snap.unrouted_count
        out_pcb.parent.mkdir(parents=True, exist_ok=True)
        env._engine.save(str(out_pcb))   # emits a native .kicad_pro alongside
        # Deterministic safety net: RLRouter.save's companion .kicad_pro write
        # can be dropped when many samples of the SAME board save concurrently
        # (its project-file write races across the pool). The saved .kicad_pcb
        # still embeds native (setup) rules, but co-locating the per-rollout
        # .kicad_pro keeps the cell convention intact, so backfill from the
        # source board's native .kicad_pro when the sidecar is missing.
        out_pro = out_pcb.with_suffix(".kicad_pro")
        if not out_pro.is_file():
            src_pro = Path(job["board_path"]).with_suffix(".kicad_pro")
            if src_pro.is_file():
                import shutil
                shutil.copy2(src_pro, out_pro)
        # Metric semantics via the shared derivation point. The info dict is
        # synthesized from the end-of-replay snapshot rather than the last
        # step's live info: the replayed action sequence may stop mid-episode
        # (or every step may be rejected), so only the snapshot is a reliable
        # terminal record. terminated=True — the plan is exhausted, so the
        # episode is over by construction.
        row.update(u.runtime_metrics_from_info(
            {
                "unrouted_count": int(u_t),
                "initial_unconnected": int(u_0),
                "wirelength": float(snap.total_wirelength),
                "via_count": int(snap.via_count),
                "track_count": int(snap.track_count),
            },
            terminated=True,
            truncated=False,
        ))
        row.update({
            "steps": int(steps_run),
            "initial_unrouted_edges": int(u_0),
            "unrouted_edges_remaining": int(u_t),
            "per_board_rollout_time": round(time.monotonic() - t0, 4),
            "artifact_path": str(out_pcb.resolve()),
        })
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb
        row.update(status="crashed", crashed=True,
                   error=f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}",
                   per_board_rollout_time=round(time.monotonic() - t0, 4))
    finally:
        if env is not None:
            try:
                env.close()   # singleton: drop router before next env in this worker
            except Exception:
                pass
        gc.collect()
    return row


# ---------------------------------------------------------------------------
# Job discovery + per-model driver
# ---------------------------------------------------------------------------

def _board_id_from_dir(name: str) -> str:
    """``test__board_00007`` -> ``board_00007`` (strip the split prefix)."""
    return re.sub(r"^[a-z]+__", "", name)


def _discover_jobs(responses_dir: Path, dataset: Path, cell: str, seed: int,
                   boards_dir: Path, limit: int | None,
                   sample_substr: list[str] | None) -> list[dict]:
    """Walk ``per_board/test__board_*/sample_*.response.txt`` -> per-sample jobs.

    ``board_index`` is assigned by sorted-unique ``board_id`` (one stable index
    per board), matching ``eval/pipeline.py:_rows_from_boards_dir``.
    """
    per_board = responses_dir / "per_board"
    if not per_board.is_dir():
        raise FileNotFoundError(f"missing per_board dir: {per_board}")

    found: list[tuple[str, Path, int, Path]] = []   # (board_id, ds_pcb, ridx, resp)
    missing_pcb: list[str] = []
    for d in sorted(per_board.iterdir()):
        if not d.is_dir():
            continue
        board_id = _board_id_from_dir(d.name)
        if sample_substr and not any(s in board_id for s in sample_substr):
            continue
        ds_pcb = dataset / f"{board_id}.kicad_pcb"
        if not ds_pcb.is_file():
            missing_pcb.append(board_id)
            continue
        for resp in sorted(d.glob("sample_*.response.txt")):
            m = re.search(r"sample_(\d+)\.response\.txt$", resp.name)
            if not m:
                continue
            found.append((board_id, ds_pcb, int(m.group(1)), resp))

    if missing_pcb:
        log.warning("skipping %d board(s) with no dataset .kicad_pcb under %s: %s",
                    len(missing_pcb), dataset,
                    ", ".join(missing_pcb[:8]) + (" ..." if len(missing_pcb) > 8 else ""))
    if limit is not None:
        keep = set(sorted({b for b, *_ in found})[:limit])
        found = [t for t in found if t[0] in keep]

    idx_of = {bid: i for i, bid in enumerate(sorted({t[0] for t in found}))}
    jobs = [
        dict(board_index=idx_of[bid], board_id=bid, board_path=str(ds_pcb),
             rollout_idx=ridx, seed=seed,
             response_path=str(resp),
             out_pcb=str(boards_dir / f"{bid}_{cell}_s{seed:02d}_r{ridx:02d}.kicad_pcb"))
        for (bid, ds_pcb, ridx, resp) in found
    ]
    jobs.sort(key=lambda j: (j["board_index"], j["rollout_idx"]))
    return jobs


def run_model(*, responses_dir: Path, dataset: Path, output_dir: Path, cell: str,
              seed: int, workers: int, limit: int | None,
              sample_substr: list[str] | None) -> tuple[int, int]:
    boards_dir = output_dir / "boards"
    boards_dir.mkdir(parents=True, exist_ok=True)
    jobs = _discover_jobs(responses_dir, dataset, cell, seed, boards_dir,
                          limit, sample_substr)
    if not jobs:
        log.warning("[%s] no jobs discovered", cell)
        return 0, 0

    n_boards = len({j["board_id"] for j in jobs})
    log.info("=== %s ===", cell)
    log.info("responses: %s", responses_dir)
    log.info("dataset:   %s", dataset)
    log.info("output:    %s", output_dir)
    log.info("boards: %d  samples: %d  workers: %d  seed: s%02d",
             n_boards, len(jobs), workers, seed)

    rows: list[dict] = []
    if workers <= 1:
        for j in jobs:
            rows.append(_replay_one(j))
            _emit_progress(rows[-1], cell, len(rows), len(jobs))
    else:
        ctx = mp.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_replay_one, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                done += 1
                rows.append(r)
                _emit_progress(r, cell, done, len(jobs))

    rows.sort(key=lambda r: (r["board_index"], r["rollout_idx"]))
    u.write_csv(output_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "args": {
            "generator": "rollout/plan_only.py",
            "stage": "rollout_only",
            "cell": cell,
            "responses_dir": str(responses_dir),
            "dataset": str(dataset),
            "seed": seed,
            "n_boards": n_boards,
            "n_samples": len(jobs),
            "use_yaml_drc_fallback": False,
            "design_rules": "native (board .kicad_pro)",
            "artifact_naming": "<board_id>_<cell>_s<SS>_r<RR>.kicad_pcb",
            "note": ("Replayed plan-only action sequences under NATIVE per-board "
                     "design rules (no DRC/aggregate). Score with `python "
                     "eval/pipeline.py --stages eval,aggregate --output-dir <this dir>`."),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str))

    ok = sum(1 for r in rows if r["status"] == "completed")
    log.info("[%s] done: %d/%d replayed (status=completed)", cell, ok, len(rows))
    return ok, len(rows)


def _emit_progress(row, cell, done, total):
    detail = (f"-> {Path(row['artifact_path']).name}"
              if row.get("artifact_path") else (row.get("error", "") or "")[:48])
    print(f"  [{done:4d}/{total}] {cell:22s} "
          f"{str(row['board_id'])[:14]:14s} r{int(row['rollout_idx']):02d} "
          f"{row['status']:10s} rats={float(row.get('ratsnest_reduction') or 0):.3f} "
          f"t={float(row.get('per_board_rollout_time') or 0):5.1f}s {detail}",
          flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses-root", type=Path, default=None,
                    help="root holding the apiseq_<split>_zs_<model>/ summary dirs. "
                         "Default: the 'plan_only_responses' dataset under "
                         "$CADAGENT_DATA_ROOT (configs/paths.yaml).")
    ap.add_argument("--split", default="synth", choices=("synth", "real"),
                    help="plan-only response split (D2=synth, D3=real).")
    ap.add_argument("--model", default=None,
                    help="single model tag, e.g. gpt-5.4. Mutually exclusive "
                         "with --models.")
    ap.add_argument("--models", default=None,
                    help="comma list of model tags, or 'all' for the synth set. "
                         "Each writes its own cell dir under --output-root.")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dir of native boards (<board_id>.kicad_pcb + .kicad_pro).")
    ap.add_argument("--cell", default=None,
                    help="cell name used in filenames (default: plan_only_<model>).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="single-cell mode: write boards/+per_rollout.csv here "
                         "(eval/pipeline.py --output-dir style). Requires --model.")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="multi-model mode: write <output-root>/plan_only_<model>/ "
                         "per model.")
    ap.add_argument("--seed", type=int, default=0, help="seed label (s<SS>).")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap to the first N boards (by sorted board_id).")
    ap.add_argument("--sample", action="append", default=None,
                    help="only boards whose id contains this substring (repeatable).")
    args = ap.parse_args()
    if args.responses_root is None:
        args.responses_root = resolve_dataset("plan_only_responses")

    if args.models:
        models = (SYNTH_MODELS if args.models.strip() == "all"
                  else [m.strip() for m in args.models.split(",") if m.strip()])
    elif args.model:
        models = [args.model]
    else:
        ap.error("pass --model <m> (with --output-dir) or --models <list> "
                 "(with --output-root).")

    if len(models) == 1 and args.output_dir is not None:
        targets = [(models[0], args.output_dir)]
    else:
        if args.output_root is None:
            ap.error("multi-model (or no --output-dir) requires --output-root.")
        targets = [(m, args.output_root / f"plan_only_{m}") for m in models]

    grand_ok = grand_total = 0
    for model, out_dir in targets:
        cell = args.cell if (args.cell and len(targets) == 1) else f"plan_only_{model}"
        # Response summary dirs are named apiseq_<split>_zs_<model>.
        responses_dir = args.responses_root / f"apiseq_{args.split}_zs_{model}"
        if not responses_dir.is_dir():
            log.warning("skip %s: no responses dir %s", model, responses_dir)
            continue
        ok, total = run_model(
            responses_dir=responses_dir, dataset=args.dataset,
            output_dir=Path(out_dir), cell=cell, seed=args.seed,
            workers=args.workers, limit=args.limit, sample_substr=args.sample)
        grand_ok += ok
        grand_total += total

    n_crashed = grand_total - grand_ok
    log.info("ALL done: %d/%d replayed across %d cell(s)%s.",
             grand_ok, grand_total, len(targets),
             f" ({n_crashed} crashed — see status=crashed rows)" if n_crashed else "")
    # A completed run (some boards may legitimately crash on malformed action
    # sequences — reported per-row) is success; only "nothing ran" is an error,
    # so a multi-cell driver loop under `set -e` isn't aborted by stray crashes.
    return 0 if grand_total else 2


if __name__ == "__main__":
    sys.exit(main())
