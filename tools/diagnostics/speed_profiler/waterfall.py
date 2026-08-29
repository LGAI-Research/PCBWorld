"""waterfall — renders a directory of prof_*.json files into a single wide
decomposition-table HTML (rule-based, no manual editing).

    python -m tools.diagnostics.speed_profiler.waterfall <data_dir> [-o out.html]
    python -m tools.diagnostics.speed_profiler.waterfall <dirA> <dirB> ... \
        [--labels "baseline,variant1,..."] [-o cmp.html]   # variant-comparison mode
    python scripts/profile.py ... --waterfall     # auto-generated into the out dir after a run

Fixed rules:
- Cell = auto-discovered by the filename convention
  ``prof_<ds>_e<NE>_b<BS>(__n27)?.json`` (extra experiment files with other
  suffixes are excluded automatically; an ``__n27`` clean-run file takes
  priority when present).
- eval column = per-set decomp from ``prof_eval*.json`` — same row skeleton (update is "—").
- Sum-closure assert at every level (rollout<3%, update<2%) — generation aborts on failure.
- The header's single-server serial-equivalent time = each JSON's measured
  ``run.total_wall_s`` (summed per GPU type; older JSONs without it fall back
  to an approximate sum of phase times).
- **The row scheme is fixed to a single ROWS list** — both the single-campaign
  table and variant comparison (2+ directories) use the same 20 rows. Don't
  build a separate one-off generator for an A/B table; use comparison mode.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

_CELL_PAT = re.compile(
    # ds may contain underscores (e.g. d3b_autoreg_v2) — backtracking still
    # anchors the trailing _e<NE>_b<BS> groups correctly.
    r"^prof_([a-z0-9_]+)_e(\d+)_b(\d+)(__n27)?(__t\d{6}-\d{6})?\.json$")

ROWS = [
    ("TOTAL", "total — train: ITER (sum of 4 phases) · eval: wall", "sum", ""),
    ("select_boards", "select_boards (board reload)", "child", ""),
    ("roll_sum", "rollout — total (train=collect phase · eval=wall)", "sum", ""),
    ("mask_ipc", "├ mask_ipc (4 barriers, inline)", "child", "i"),
    ("fw_sum", "├ forward — total (train 3-pass/eval 3-pass)", "child", ""),
    ("fw_walk", "│ ├ walk (CPU, main)", "child2", "w"),
    ("fw_gpu", "│ ├ GPU (cuda-event)", "child2", "g"),
    ("fw_resid", "│ └ launch/sampling/sync residual", "child2 resid", ""),
    ("step_bar", "├ step_barrier (engine+IPC, inline)", "child", "i"),
    ("advance", "├ obs advance", "child", ""),
    ("collector", "├ collector/bookkeeping (yield gap)", "child", ""),
    ("reset", "├ reset_batch (train only)", "child", ""),
    ("outloop", "└ out-of-loop residual (train: final_values · eval: wave·spawn·reload·scoring)", "child resid", ""),
    ("compute_targets", "compute_targets (GAE/buffer)", "child", ""),
    ("upd_sum", "update — total (absent in eval)", "sum", ""),
    ("up_entry_walk", "├ entry walk (uncached fallback, once per update)", "child", "w"),
    ("up_eval", "├ evaluate (fwd — total)", "child", ""),
    ("up_fwd", "│ · GPU fwd 2-pass (cuda-event)", "child2", "g"),
    ("up_h2d", "│ · h2d_encode/scatter (CPU→GPU)", "child2", "w"),
    ("up_bwd", "├ backward (wall≈GPU)", "child", "g"),
    ("up_clip", "├ clip + optimizer.step", "child", ""),
    ("up_sync", "├ sync_grads (DDP allreduce; —=single)", "child", "i"),
    ("up_bcast", "├ buffer bcast (DDP, once per iter; —=single)", "child", "i"),
    ("up_perm", "├ perm broadcast (DDP; —=single)", "child", "i"),
    ("up_resid", "└ residual: minibatch glue", "child resid", ""),
]
ROW_KEYS = [key for key, *_ in ROWS]
_UP_KEYS = tuple(k for k in ROW_KEYS if k.startswith("up_"))

# Shared "chip" CSS (injected into both single-campaign and comparison mode)
_CHIPS_CSS = """
.chips{display:flex;flex-wrap:wrap;gap:7px;margin:0 0 16px}
.chip{font-family:var(--mono);font-size:11px;color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 9px}
.chip b{color:var(--ink);font-weight:600}
"""


def _load_css(extra: str = "") -> str:
    css = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "waterfall.css")).read()
    return css.replace("</style>", _CHIPS_CSS + extra + "</style>")


def _discover_cells(data_dir: str):
    found = {}
    for pth in glob.glob(f"{data_dir}/prof_*.json"):
        m = _CELL_PAT.match(os.path.basename(pth))
        if m:
            found.setdefault((m.group(1), int(m.group(2)), int(m.group(3))), True)
    return sorted(found)


def _load_cell(data_dir, ds, ne, bs):
    # Candidates per cell, in priority order: __n27 (legacy, preferred) >
    # __t<YYMMDD-HHMMSS> (latest timestamped) > no suffix. When multiple
    # timestamped files exist (the filename records when the run was made),
    # the lexicographically-largest one = the latest run wins; files with no
    # suffix stay compatible as-is.
    n27 = f"{data_dir}/prof_{ds}_e{ne}_b{bs}__n27.json"
    if os.path.exists(n27):
        path = n27
    else:
        stamped = sorted(glob.glob(
            f"{data_dir}/prof_{ds}_e{ne}_b{bs}__t[0-9]*.json"))
        canon = f"{data_dir}/prof_{ds}_e{ne}_b{bs}.json"
        path = stamped[-1] if stamped else canon
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    d["_n27"] = path.endswith("__n27.json")
    return d


def _train_col(d):
    ph = d["phases"]["per_phase"]
    p = lambda k: (ph.get(k, {}) or {}).get("mean")
    rd = d.get("rollout_decomp", {}) or {}
    fs = rd.get("forward_split", {}) or {}
    ud = d.get("update_decomp", {}) or {}
    pc = ud.get("perf_counter_ms", {}) or {}
    ga = ud.get("gpu_active_ms", {}) or {}
    oom = bool(d["run"].get("update_oom"))
    up = p("update")
    ev_w = pc.get("evaluate"); fwd = ga.get("fwd_pass")
    entry_walk = ud.get("entry_walk_ms")   # uncached-fallback walk, once per update (0 when carried)
    ddp = ud.get("ddp_ms") or {}           # DDP-only; {} on single-GPU → children are None
    up_children = None
    if not oom and ev_w is not None and up:
        h2d = max(ev_w - (fwd or 0), 0.0)  # CPU→GPU inside evaluate (in-forward)
        clip = (pc.get("clip", 0) + pc.get("step", 0))
        ew = entry_walk or 0.0
        sync, perm, bcast = ddp.get("sync"), ddp.get("perm"), ddp.get("bcast")
        resid = (up - ew - ev_w - pc.get("backward", 0) - clip
                 - (sync or 0) - (perm or 0) - (bcast or 0))
        up_children = dict(
            up_entry_walk=entry_walk, up_eval=ev_w, up_fwd=fwd, up_h2d=h2d,
            up_bwd=pc.get("backward"), up_clip=clip,
            up_sync=sync, up_bcast=bcast, up_perm=perm, up_resid=resid)
    col = dict.fromkeys(ROW_KEYS)   # row scheme derives from the single ROWS source (prevents drift)
    col.update({
        "TOTAL": sum(x for x in (p("select_boards"), p("collect"),
                                 p("compute_targets"), up) if x),
        "select_boards": p("select_boards"),
        "roll_sum": p("collect"),
        "mask_ipc": rd.get("mask_ipc_ms"),
        "fw_sum": rd.get("forward_wall_ms"),
        "fw_walk": fs.get("walk_cpu_ms"),
        "fw_gpu": fs.get("gpu_event_ms"),
        "fw_resid": fs.get("launch_sync_resid_ms"),
        "step_bar": rd.get("step_barrier_ms"),
        "advance": rd.get("obs_advance_ms"),
        "collector": rd.get("collector_ms"),
        "reset": rd.get("reset_ms"),
        "outloop": rd.get("unbucketed_post_loop_ms"),
        "compute_targets": p("compute_targets"),
        "upd_sum": up,
        **{k: (up_children or {}).get(k) for k in _UP_KEYS},
    })
    col["_host"] = d["run"].get("host")
    col["_fp"] = d.get("fingerprint", {}).get("main", {})
    return col, oom


def _eval_col(r):
    dc = r["decomp_ms"]; fs = dc["forward_split"]
    wall = r["wall_s"] * 1000
    col = dict.fromkeys(ROW_KEYS)   # rows that don't apply here (e.g. update) derive as None ("—")
    col.update({
        "TOTAL": wall,
        "roll_sum": wall,
        "mask_ipc": dc["mask_ipc"], "fw_sum": dc["forward_wall"],
        "fw_walk": fs["walk_cpu"], "fw_gpu": fs["gpu_event"],
        "fw_resid": fs["launch_sync_resid"],
        "step_bar": dc["step_barrier"], "advance": dc["obs_advance"],
        "collector": dc["between_bookkeeping"],
        "outloop": dc["unbucketed"],
    })
    return col


def _run_wall_s(j):
    """Total elapsed time (s): prefers the measured run.total_wall_s, falling
    back to a phase-sum estimate. Returns (seconds, is_measured?)."""
    r = j.get("run", {})
    if "total_wall_s" in r:
        return float(r["total_wall_s"]), True
    s = float(r.get("spawn_setup_s", 0.0))
    ph = j.get("phases", {}).get("per_phase", {})
    iter_ms = sum(v.get("mean", 0) for v in ph.values())
    s += iter_ms / 1000 * (r.get("warmup_iters", 0) + r.get("measured_iters", 0))
    b = j.get("barrier", {})
    coll = ph.get("collect", {}).get("mean")
    if b and coll and r.get("n_steps"):
        s += b.get("n_steps", 0) * (coll / r["n_steps"]) / 1000
    s += sum(ev.get("wall_s", 0) for ev in j.get("eval", {}).values()
             if isinstance(ev, dict))
    return s, False


def _gpu_short(j):
    name = j.get("fingerprint", {}).get("main", {}).get("gpu_name", "?")
    return name.replace("NVIDIA ", "").split()[0]


def _campaign_header(data_dir, cols, hdr_jsons):
    import datetime as dt
    logs = glob.glob(f"{data_dir}/log_*.txt") + glob.glob(f"{data_dir}/node_*.log")
    profs = glob.glob(f"{data_dir}/prof_*.json")
    t0 = min(os.path.getmtime(f) for f in logs) if logs else None
    t1 = max(os.path.getmtime(f) for f in profs) if profs else None
    wall_min = (t1 - t0) / 60 if t0 and t1 else None
    serial = {}
    for j in hdr_jsons:
        sec, exact = _run_wall_s(j)
        g = serial.setdefault(_gpu_short(j), [0.0, False])
        g[0] += sec
        g[1] = g[1] or not exact
    serial_txt = " · ".join(
        f"{g} → {'≈' if est else ''}{s/60:.0f}min" for g, (s, est) in sorted(serial.items()))
    hosts = sorted({(c.get("_host") or "").split("-l40")[0].split("-a10")[0]
                    for _, c, _, _ in cols if c.get("_host")})
    fp = next((c.get("_fp") for _, c, _, _ in cols if c.get("_fp")), {})
    # iter workload (prevents a config difference from being misread as a
    # slowdown; shown as a range when cells differ)
    _ns = sorted({j["run"].get("n_steps") for j in hdr_jsons if j["run"].get("n_steps")})
    _ms = sorted({j["run"].get("max_steps") for j in hdr_jsons if j["run"].get("max_steps")})
    _rng = lambda v: (f"{v[0]}" if len(v) == 1 else f"{v[0]}~{v[-1]}") if v else "?"
    chips = [
        ("single-server serial equivalent", serial_txt or "?"),
        ("measurement campaign wall",
         f"{wall_min:.0f}min ({len(hosts) or '?'} nodes in parallel)" if wall_min else "?"),
        ("iter workload", f"n_steps {_rng(_ns)} · max_steps {_rng(_ms)}"),
        ("nodes", ", ".join(hosts) or "?"),
        ("GPU", fp.get("gpu_name", "?")),
        ("CPU", f"{fp.get('cores_physical','?')}C/{fp.get('cores_logical','?')}T"),
        ("torch", f"{fp.get('torch_version','?')} · matmul {fp.get('matmul_precision','?')}"
                  f" · TF32 {'on' if fp.get('cuda_matmul_allow_tf32') else 'off'}"),
        ("measured at", dt.datetime.fromtimestamp(t1).strftime("%Y-%m-%d %H:%M") if t1 else "?"),
    ]
    return '<div class="chips">' + "".join(
        f'<span class="chip">{k} <b>{v}</b></span>' for k, v in chips) + "</div>"


def _barrier_section(cells_raw):
    rows = []
    for label, d in cells_raw:
        b = d.get("barrier") or {}
        sb = dict(b.get("step_barrier") or {})
        if not sb:
            continue
        sb["_mask"] = round(sum(m.get("barrier_wall_ms", 0)
                                for m in (b.get("mask_barriers") or {}).values()), 1)
        sb["_wc"] = b.get("worker_compute_mean_ms") or b.get("worker_compute_median_ms")
        rows.append((label, sb))
    if not rows:
        return ""
    cols_h = "".join(f"<th>{l}</th>" for l, _ in rows)

    def tr(name, key, cls="child", pct=True):
        tds = ""
        for _, sb in rows:
            v = sb.get(key); w = sb.get("barrier_wall_ms") or 1
            if v is None:
                tds += "<td class=dimc>—</td>"
            else:
                p = f'<span class="pct">{100*v/w:.0f}%</span>' if pct else ""
                tds += f"<td>{v:,.1f}{p}</td>"
        return f'<tr class="{cls}"><td class="name">{name}</td>{tds}</tr>'

    body = "".join([
        tr("step_barrier wall = send + worker_MAX + unpickle", "barrier_wall_ms", "sum"),
        tr("├ send (main serial action dispatch)", "send_ms"),
        tr("├ worker_MAX = mean + idle_waste", "worker_max_ms"),
        tr("│ ├ worker mean (typical compute: engine+pickle)", "worker_mean_ms", "child2"),
        tr("│ └ idle_waste (max−mean: imbalance wait, recoverable via balancing/async)", "idle_waste_ms", "child2"),
        tr("└ unpickle (main serial state deserialization)", "unpickle_ms"),
        tr("ref: worker median (distribution)", "worker_median_ms", "child resid", pct=False),
        tr("ref: worker p90 (distribution)", "worker_p90_ms", "child resid", pct=False),
        tr("ref: straggler = max−median (tail diagnostic)", "straggler_ms", "child resid", pct=False),
        tr("ref: wcomp = pure in-worker env.step (mean)", "_wc", "child resid", pct=False),
        tr("ref: sum of the 4 mask barriers (separate wall)", "_mask", "child resid", pct=False),
    ])
    return ('<h2>step-barrier probe detail — ms/step (100-step fresh-state probe; '
            "lower than in-loop)</h2>"
            f'<div class="card"><div class="scroll"><table><thead><tr>'
            f'<th class="name">component (% is vs wall)</th>{cols_h}</tr></thead>'
            f'<tbody>{body}</tbody></table></div></div>')


def _util_section(cells_raw):
    """Per-phase GPU/CPU utilization — consumes UtilSampler's ``util.per_phase``.

    proctree (main+worker PID tree) is the authoritative number; syswide is
    only a shared-node contamination check. Light mode (``--no-util``) has no
    util block, so this section is omitted entirely."""
    rows = []
    for label, d in cells_raw:
        up = (d.get("util") or {}).get("per_phase") or {}
        if up:
            rows.append((label, up))
    if not rows:
        return ""
    cols_h = "".join(f"<th>{l}</th>" for l, _ in rows)

    def tr(name, ph, scope, res, cls="child", unit=1.0):
        tds = ""
        for _, up in rows:
            s = ((up.get(ph) or {}).get(scope) or {}).get(res) or {}
            if s.get("mean") is None:
                tds += "<td class=dimc>—</td>"
            else:
                tds += (f"<td>{s['mean'] * unit:,.1f}"
                        f'<span class="pct">max {s["max"] * unit:,.1f}</span></td>')
        return f'<tr class="{cls}"><td class="name">{name}</td>{tds}</tr>'

    body = "".join([
        tr("collect — GPU util %", "collect", "gpu", "gpu_util", "sum"),
        tr("├ CPU cores (proctree = main+workers)", "collect", "proctree", "cpu_cores"),
        tr("└ CPU cores (syswide — contamination watch)", "collect", "syswide", "cpu_cores", "child resid"),
        tr("update — GPU util %", "update", "gpu", "gpu_util", "sum"),
        tr("├ CPU cores (proctree)", "update", "proctree", "cpu_cores"),
        tr("└ GPU mem GB", "update", "gpu", "gpu_mem_mb", "child", 1 / 1024),
    ])
    return ('<h2>utilization — per-phase GPU/CPU (0.1s samples; value = mean, superscript = max)</h2>'
            f'<div class="card"><div class="scroll"><table><thead><tr>'
            f'<th class="name">item</th>{cols_h}</tr></thead>'
            f'<tbody>{body}</tbody></table></div></div>')


def _provenance_section(data_dir, cells_raw, eval_provenance):
    rows = []
    if cells_raw:
        tmpl = set()
        for _, d in cells_raw:
            r = d["run"]
            tmpl.add((r["n_steps"], r["max_steps"], r["warmup_iters"], r["measured_iters"]))
        specs = " ".join(f"{d['run']['dataset']}:{d['run']['n_envs']}:{d['run']['batch_size']}"
                         for _, d in cells_raw)
        t = sorted(tmpl)[0]
        rows.append(("① measure (per train cell)",
                     f"python scripts/profile.py --dataset &lt;ds&gt; --n-envs &lt;NE&gt; --batch-size &lt;BS&gt; "
                     f"--n-steps {t[0]} --max-steps {t[1]} --warmup-iters {t[2]} --measured-iters {t[3]} "
                     f"--gpu-index &lt;g&gt; --out $DIR/prof_&lt;ds&gt;_e&lt;NE&gt;_b&lt;BS&gt;.json<br>"
                     f"<span class=dimc>specs in this table = {specs}</span>"))
    for tag, evp, evj in eval_provenance:
        d = json.load(open(evp)); r = d["run"]
        any_set = next(iter(evj.values()))
        rows.append((f"② measure (eval {tag})",
                     f"python scripts/profile.py --dataset {r['dataset']} --n-envs {r['n_envs']} "
                     f"--n-steps {r['n_steps']} --max-steps {r['max_steps']} --warmup-iters {r['warmup_iters']} "
                     f"--measured-iters {r['measured_iters']} --no-barrier --eval "
                     f"--eval-rollouts {any_set['n_rollouts_per_board']} --eval-board-limit {r.get('eval_board_limit','8')} "
                     f"<span class=dimc>(JSON: {os.path.basename(evp)})</span>"))
    rows += [
        ("③ generate table (this page)",
         f"python -m tools.diagnostics.speed_profiler.waterfall {data_dir} -o waterfall.html"
         "<br><span class=dimc>or profile.py ... --waterfall (auto-generated after the run)</span>"),
        ("related files",
         "tool: tools/diagnostics/speed_profiler/ (tracked, scripts/profile.py front door) · "
         f"data: {data_dir}"),
        ("fixed-rule declaration",
         "The measured JSONs are unedited originals. Every cell/row/sum/% in this table is computed "
         "from the JSONs by waterfall.py, and generation aborts if a per-level sum-closure assert "
         "(rollout&lt;3%, update&lt;2%) fails. "
         "Cell/eval columns are auto-discovered by filename convention — drop a new measurement JSON "
         "into the data directory and it is picked up on the next generation. "
         "The header's single-server serial-equivalent time sums each JSON's measured run.total_wall_s "
         "(new runs) per GPU type; older JSONs without total_wall_s are estimated from the sum of "
         "recorded phases (marked ≈)."),
    ]
    body = "".join(f'<tr><td class="name" style="width:220px">{k}</td>'
                   f'<td style="text-align:left;font-family:var(--mono);font-size:10.5px">{v}</td></tr>'
                   for k, v in rows)
    return ('<h2>reproduction — run scripts/commands (reverse-generated from run config)</h2>'
            f'<div class="card"><div class="scroll"><table><tbody>{body}</tbody></table></div></div>')


def _cell_html(c, key):
    v = c.get(key)
    if v is None:
        return '<td class="dimc">—</td>'
    tot = c["TOTAL"] or 1
    return f'<td>{v/1000:,.1f}<span class="pct">{100*v/tot:.1f}%</span></td>'


def _assert_closures(cols):
    """Verify sum-closure at every level (failure aborts generation). Shared by
    single-campaign and comparison mode."""
    for label, c, oom, kind in cols:
        # --mode light (lightweight A/B): only phase timers exist, no detailed
        # decomposition, so every child row is None and closure can't hold —
        # skip it (only Collect/Update/ITER totals are shown).
        if c.get("fw_sum") is None and c.get("up_eval") is None:
            continue
        if c["roll_sum"]:
            kids = sum(c[k] or 0 for k in ("mask_ipc", "fw_sum", "step_bar", "advance",
                                           "collector", "reset", "outloop"))
            gap = 100 * (c["roll_sum"] - kids) / c["roll_sum"]
            assert abs(gap) < 3.0, f"{label} rollout closure gap {gap:.1f}%"
        if kind == "train" and not oom and c["upd_sum"]:
            # update children (sum = upd_sum): up_eval is itself the forward
            # wall, and up_fwd/up_h2d are its child2 rows, so they're excluded
            # from the parent sum (avoids double-counting).
            kids = sum(c[k] or 0 for k in ("up_entry_walk", "up_eval", "up_bwd",
                                           "up_clip", "up_sync", "up_bcast",
                                           "up_perm", "up_resid"))
            gap = 100 * (c["upd_sum"] - kids) / c["upd_sum"]
            assert abs(gap) < 2.0, f"{label} update closure gap {gap:.1f}%"
            # child2 closure: up_fwd + up_h2d == up_eval.
            if c.get("up_eval"):
                c2 = (c.get("up_fwd") or 0) + (c.get("up_h2d") or 0)
                g2 = 100 * (c["up_eval"] - c2) / c["up_eval"]
                assert abs(g2) < 2.0, f"{label} evaluate closure gap {g2:.1f}%"


def _cmp_cell_html(c, key, base):
    """Comparison-mode cell: value + %-of-total; variant columns also show Δ% vs the baseline."""
    if c is None:
        return '<td class="dimc">—</td>'
    v = c.get(key)
    if v is None:
        return '<td class="dimc">—</td>'
    tot = c["TOTAL"] or 1
    extra = ""
    bv = base.get(key) if (base is not None and base is not c) else None
    if bv:
        delta = 100 * (v - bv) / bv
        dcls = "dminus" if delta < 0 else "dplus"
        extra = f' · <span class="{dcls}">{delta:+.0f}%</span>'
    return f'<td>{v/1000:,.1f}<span class="pct">{100*v/tot:.1f}%{extra}</span></td>'


def generate_compare(dirs: list[str], out: str, labels: list[str] | None = None) -> str:
    """Variant comparison — one column per directory (= a code variant), one
    table per cell (ds, ne, bs).

    Row scheme, value extraction, and sum-closure are identical to single
    mode (reuses ROWS/_train_col/_assert_closures). Baseline = the first
    directory; Δ% is measured against it. Cell keys are the intersection
    across all directories (a variant comparison is only meaningful across
    cells that share the same protocol).
    """
    dirs = [d.rstrip("/") for d in dirs]
    labels = labels or [os.path.basename(d) for d in dirs]
    assert len(labels) == len(dirs), "--labels count differs from the number of directories"
    cell_sets = [set(_discover_cells(d)) for d in dirs]
    keys = sorted(set.intersection(*cell_sets))
    if not keys:
        raise SystemExit(f"no common prof_<ds>_e<N>_b<B>.json cells across {dirs}")

    css = _load_css("""
.dminus{color:var(--ipc);font-weight:600}
.dplus{color:var(--crit);font-weight:600}
""")

    sections, variant_rows = [], []
    fp_seen = {}
    for ds, ne, bs in keys:
        cols = []
        run_cfg = {}   # iter workload for the cell title (n_steps·max_steps — from the first loaded JSON)
        for lab, d in zip(labels, dirs):
            j = _load_cell(d, ds, ne, bs)
            if j is None:
                cols.append((lab, None, False, "train"))
                continue
            if not run_cfg:
                run_cfg = j.get("run", {})
            col, oom = _train_col(j)
            cols.append((lab, col, oom, "train"))
            fp_seen.setdefault(lab, (col.get("_host"), col.get("_fp") or {}))
        _assert_closures([(l, c, o, k) for l, c, o, k in cols if c])
        base = next((c for _, c, _, _ in cols if c), None)
        head = ('<tr><th class="name">component (s · %of total · Δ vs baseline)</th>'
                + "".join(f'<th>{lbl}{"<span class=badge>OOM</span>" if oom else ""}</th>'
                          for lbl, c, oom, _ in cols) + "</tr>")
        body = "".join(
            f'<tr class="{cls}"><td class="name"><span class="{cc}">{lab}</span></td>'
            + "".join(_cmp_cell_html(c, key, base) for _, c, _, _ in cols) + "</tr>"
            for key, lab, cls, cc in ROWS)
        # State the iter workload in the title — prevents a config difference
        # (e.g. n_steps×4) from being misread as "the code got slower".
        _work = (f" · n_steps {run_cfg['n_steps']} · max_steps {run_cfg['max_steps']}"
                 if run_cfg.get("n_steps") else "")
        sections.append(
            f"<h2>{ds} · n_envs {ne} · batch {bs}{_work}</h2>"
            f'<div class="card"><div class="scroll"><table><thead>{head}</thead>'
            f"<tbody>{body}</tbody></table></div></div>")

    for lab, d in zip(labels, dirs):
        host, fp = fp_seen.get(lab, (None, {}))
        variant_rows.append(
            f'<tr><td class="name" style="width:200px">{lab}</td>'
            f'<td style="text-align:left;font-family:var(--mono);font-size:10.5px">{d}'
            f'<br><span class="dimc">{host or "?"} · {fp.get("gpu_name", "?")} · '
            f'torch {fp.get("torch_version", "?")}</span></td></tr>')
    gpus = {fp.get("gpu_name") for _, fp in fp_seen.values()}
    warn = ("" if len(gpus) <= 1 else
            '<p class="sub" style="color:var(--crit)">⚠ GPUs differ across columns — compare absolute values with caution</p>')

    html = f"""<title>waterfall variant comparison — {" vs ".join(labels)}</title>{css}
<div class="wrap">
<h1>variant comparison — same cells, same row scheme (baseline = {labels[0]})</h1>
<p class="sub">train columns = s/iter (median of measured iters) · row/sum/closure rules identical to the single-campaign table
(waterfall.py fixed ROWS scheme) · Δ% = vs the first column · cells = intersection across all directories</p>
{warn}{"".join(sections)}
<h2>variant (column) definitions — data directories</h2>
<div class="card"><div class="scroll"><table><tbody>{"".join(variant_rows)}</tbody></table></div></div>
<p class="foot">regenerate: python -m tools.diagnostics.speed_profiler.waterfall {" ".join(dirs)}
 --labels "{",".join(labels)}" -o {os.path.basename(out)}<br>
measured JSONs unedited · only columns passing the sum-closure asserts (rollout&lt;3%, update&lt;2%) are shown ·
barrier probe detail/utilization/eval columns live in each directory's single-campaign waterfall.</p>
</div>"""
    with open(out, "w") as f:
        f.write(html)
    print(f"[waterfall] wrote {out} ({len(html)} bytes, {len(keys)} cells x {len(dirs)} variants)")
    return out


def generate(data_dir: str, out: str, partial_ok: bool = True) -> str:
    data_dir = data_dir.rstrip("/")
    cols, cells_raw, eval_provenance, eval_fulls = [], [], [], []
    for ds, ne, bs in _discover_cells(data_dir):
        d = _load_cell(data_dir, ds, ne, bs)
        if d is None:
            if partial_ok:
                continue
            raise SystemExit(f"missing cell {ds} e{ne} b{bs}")
        col, oom = _train_col(d)
        label = f"{ds}·e{ne}·b{bs}" + ("ⁿ²⁷" if d["_n27"] else "")
        cols.append((label, col, oom, "train"))
        cells_raw.append((label, d))

    eval_paths = sorted(glob.glob(f"{data_dir}/prof_eval*.json"))
    for evp in eval_paths:
        tag = os.path.basename(evp).replace("prof_eval", "").replace(".json", "").strip("_") or "eval"
        full = json.load(open(evp))
        eval_fulls.append(full)
        evj = full["eval"]
        for k in evj:
            lbl = f"eval·{k.replace('val_', '')}"
            if len(eval_paths) > 1:
                lbl += f"({tag})"
            cols.append((lbl, _eval_col(evj[k]), False, "eval"))
        eval_provenance.append((tag, evp, evj))

    if not cols:
        raise SystemExit(f"no prof_<ds>_e<N>_b<B>.json / prof_eval*.json cells in {data_dir}")

    _assert_closures(cols)

    css = _load_css()

    head = ('<tr><th class="name">component (s · %of total)</th>'
            + "".join(f'<th{" style=color:var(--ipc)" if kind=="eval" else ""}>{lbl}'
                      f'{"<span class=badge>OOM</span>" if oom else ""}</th>'
                      for lbl, c, oom, kind in cols) + "</tr>")
    body = "".join(
        f'<tr class="{cls}"><td class="name"><span class="{cc}">{lab}</span></td>'
        + "".join(_cell_html(c, key) for _, c, _, _ in cols) + "</tr>"
        for key, lab, cls, cc in ROWS)

    html = f"""<title>speed_profiler waterfall — {os.path.basename(data_dir)}</title>{css}
<div class="wrap">
<h1>train+eval full breakdown — one table, rows unified via inline instrumentation</h1>
<p class="sub">train columns = s/iter (median of measured iters) · eval columns = s/set · same row skeleton (update is — in eval) · sum-closure at every level (residuals explicit)</p>
{_campaign_header(data_dir, cols, [d for _, d in cells_raw] + eval_fulls)}
<div class="card"><div class="scroll"><table><thead>{head}</thead><tbody>{body}</tbody></table></div></div>
{_barrier_section(cells_raw)}
{_util_section(cells_raw)}
{_provenance_section(data_dir, cells_raw, eval_provenance)}
<p class="foot">
collect is <b>inline (iter_rollout mirror)</b>, not probe-estimated — the residual closes over mask_ipc/forward(walk·GPU·launch)/step_barrier/collector/reset/out-of-loop (verified gap&lt;3%).
The step_barrier row is the inline wall (the internal send/worker/straggler/unpickle breakdown is in the probe section). eval columns: rollout-only phase, so select_boards/compute_targets/update = —.
</p></div>"""
    with open(out, "w") as f:
        f.write(html)
    print(f"[waterfall] wrote {out} ({len(html)} bytes, {len(cols)} cols)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="waterfall", description=__doc__)
    p.add_argument("data_dir", nargs="+", help="prof_*.json directory (2+ = variant comparison mode)")
    p.add_argument("-o", "--out", default=None, help="output html (default: <data_dir>/waterfall.html)")
    p.add_argument("--labels", default=None,
                   help="comparison-mode column labels (comma-separated, default: directory basenames)")
    p.add_argument("--strict", action="store_true", help="treat missing cells as errors (default: use what exists)")
    a = p.parse_args()
    if len(a.data_dir) > 1:
        generate_compare(a.data_dir,
                         a.out or os.path.join(a.data_dir[0], "waterfall_compare.html"),
                         a.labels.split(",") if a.labels else None)
    else:
        generate(a.data_dir[0], a.out or os.path.join(a.data_dir[0], "waterfall.html"),
                 partial_ok=not a.strict)


if __name__ == "__main__":
    main()
