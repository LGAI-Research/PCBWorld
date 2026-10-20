"""Unified runner for the three rule-based routers (freerouting / krt / orthoroute).

Per board:
  1. Locate raw .kicad_pcb + .kicad_pro + sibling .dsn / .orp via _lib.datasets
  2. Drive the chosen baseline -> raw output (.ses / .ORS / .kicad_pcb)
  3. Convert raw -> .kicad_pcb if needed (SES/ORS via _converters)
  4. Run eval.metrics.evaluate_one against the routed .kicad_pcb
  5. Aggregate per-run summary.{json,txt}

Output: <output-root>/<dataset_tag>/<algorithm>/seed<N>/{raw,routed,eval,manifest.json}
"""

from __future__ import annotations

import argparse
import functools
import gc
import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_BASELINES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASELINES_DIR.parents[2]   # rule_based -> baselines -> methods -> repo root
_EXTERNAL = _PROJECT_ROOT / "external"      # vendored rule-based libs (OrthoRoute, freerouting)
for p in (_PROJECT_ROOT, _PROJECT_ROOT / "eval", _PROJECT_ROOT / "build_rl" / "pcbnew" / "python" / "rl"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from methods.baselines.rule_based._lib.datasets import (  # noqa: E402
    BoardPaths, D3_DIFFICULTIES, algorithm_tag, dataset_tag, describe_missing,
    select_boards_with_inputs,
)

log = logging.getLogger("run_rule_based_routers")

DEFAULT_OUTPUT_ROOT = Path("methods/baselines/rule_based") / "_run_outputs"
FREEROUTING_JAR = _EXTERNAL / "freerouting" / "freerouting-2.1.0.jar"

DEFAULT_TIMEOUTS = {"freerouting": 1800, "krt": 1800, "orthoroute": 1800}


@dataclass
class BoardResult:
    board_id: str
    seed: int
    status: str
    elapsed_routing_s: float
    routed_pcb: Optional[str] = None
    raw_path: Optional[str] = None
    routing_log: Optional[str] = None
    eval_log: Optional[str] = None
    error: str = ""
    eval_metrics: dict = field(default_factory=dict)
    # The net-class rules handed to KRT (krt only): its one clearance and via, and whether the
    # classes in use want other ones.
    krt_rules: Optional[dict] = None
    # This board's warnings: its .kicad_pro's (each names the file and class) and the rules a
    # router could not take as given.
    warnings: list = field(default_factory=list)


def _inject_snap_angle_90(src_dsn, dst_dsn):
    text = src_dsn.read_text(encoding="utf-8", errors="replace")
    if "snap_angle" in text:
        dst_dsn.write_text(text, encoding="utf-8")
        return
    text = re.sub(r"(\(structure\s*\n)",
                  r"\1    (snap_angle ninety_degree)\n",
                  text, count=1)
    dst_dsn.write_text(text, encoding="utf-8")


# The KRT tools/setup/fetch_baselines.sh pins (the KDD / CADET papers'); another tree is an
# error unless KRT_ALLOW_UNPINNED=1, and then it is recorded in each board's krt_run.json.
KRT_PINNED_COMMIT = "561f1584577ea86d99e223ab27a2cd1cdbfa8bb3"
# FreeRouting 2.1.0's jar is built for Java 21 (environment.yml: openjdk 21).
FREEROUTING_JAVA = 21


def _krt_unpinned_allowed():
    return os.environ.get("KRT_ALLOW_UNPINNED") == "1"


def _java_major():
    """The major version of the ``java`` on PATH, or None when there is none."""
    java = shutil.which("java")
    if java is None:
        return None
    try:
        out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunSetupError(f"cannot run {java} -version: {exc}") from exc
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out.stderr + out.stdout)
    if not m:
        raise RunSetupError(f"cannot read the version of {java}: {(out.stderr or out.stdout)[:200]}")
    major = int(m.group(1))
    return int(m.group(2) or 0) if major == 1 else major       # "1.8" is Java 8


def prepare_run(baseline):
    """Everything a run needs before its first board, cheapest first (the .kicad_pro format file,
    FreeRouting's jar and Java, the engine, KRT and its grid_router); a missing piece stops the
    run. Returns the KRT commit (looked up once) for krt, else None."""
    if baseline not in ("freerouting", "krt"):
        return None
    _kicad9_pro_format()
    if baseline == "freerouting":
        if not FREEROUTING_JAR.is_file():
            raise RunSetupError(f"{FREEROUTING_JAR} not found — run tools/setup/fetch_baselines.sh")
        major = _java_major()
        if major is None:
            raise RunSetupError(f"no java on PATH — FreeRouting needs Java {FREEROUTING_JAVA} "
                                f"(the pcbworld env's openjdk {FREEROUTING_JAVA}, environment.yml)")
        if major < FREEROUTING_JAVA:
            raise RunSetupError(f"FreeRouting needs Java {FREEROUTING_JAVA}, the java on PATH is "
                                f"Java {major} (install openjdk {FREEROUTING_JAVA}, environment.yml)")
    _require_engine()
    if baseline != "krt":
        return None
    root = _require_krt()
    commit = _krt_commit(str(root))
    if commit != KRT_PINNED_COMMIT:
        where = (f"the git commit of {root} cannot be read (not its own git checkout, or git "
                 "failed)" if commit is None else f"KRT tree is at {commit}")
        if not _krt_unpinned_allowed():
            raise RunSetupError(f"{where}, not the pinned {KRT_PINNED_COMMIT} (the papers' KRT; "
                                "tools/setup/fetch_baselines.sh) — set KRT_ALLOW_UNPINNED=1 to "
                                "run another tree anyway")
        log.warning("%s, not the pinned %s (KRT_ALLOW_UNPINNED=1): its results are not the "
                    "papers' baseline", where, KRT_PINNED_COMMIT)
    _require_krt(prepare=True)      # the grid_router build last: it may compile
    return commit


def _route_freerouting(board, seed, raw_dir, routed_dir, algo, timeout_s,
                       via_prohibit, angle_90, max_passes, extra_env):
    from methods.baselines.rule_based._converters import convert_ses
    try:
        _, _, warnings = _checked_dsn(board.dsn, board.pro)
    except DsnRuleError as exc:
        return BoardResult(board.board_id, seed, "error", 0.0, error=f"DSN: {exc}")
    ses_path = raw_dir / f"{board.board_id}.ses"
    log_path = raw_dir / f"{board.board_id}.routing.log"
    routed_path = routed_dir / f"{board.board_id}_{algo}.kicad_pcb"
    for stale in (ses_path, routed_path):   # an earlier run's output must not pass as this one's
        stale.unlink(missing_ok=True)
    tmp_dir = raw_dir / f"{board.board_id}.freerouting_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    patched_dsn = None
    if angle_90:
        patched_dsn = raw_dir / f"{board.board_id}._angle90.dsn"
        _inject_snap_angle_90(board.dsn, patched_dsn)
        dsn_in = patched_dsn
    else:
        dsn_in = board.dsn
    env = os.environ.copy()
    if via_prohibit:
        env["FREEROUTING__ROUTER__VIAS_ALLOWED"] = "false"
        env["FREEROUTING__ROUTER__SCORING__VIA_COSTS"] = "10000"
    env.update(extra_env)
    cmd = ["java", "-Djava.awt.headless=true",
           f"-Djava.io.tmpdir={tmp_dir}", "-Xmx4g",
           "-jar", str(FREEROUTING_JAR),
           "-de", str(dsn_in), "-do", str(ses_path),
           "-mp", str(max_passes)]
    t0 = time.time()
    log_text = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=env)
        log_text = ((proc.stdout or "") + (proc.stderr or "")
                    + f"\n[runner] freerouting exit code {proc.returncode}\n")
    except subprocess.TimeoutExpired as exc:
        log_text = f"TIMEOUT after {timeout_s}s\n"
        if exc.stdout:
            log_text += exc.stdout.decode("utf-8", errors="replace")
        log_path.write_text(log_text)
        return BoardResult(board.board_id, seed, "timeout",
                           round(time.time()-t0, 3),
                           raw_path=str(ses_path) if ses_path.exists() else None,
                           routing_log=str(log_path), warnings=warnings)
    except Exception as exc:
        log_text = f"ERROR: {exc}\n"
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           routing_log=str(log_path), error=str(exc), warnings=warnings)
    finally:
        log_path.write_text(log_text)
        if patched_dsn:                 # the SES conversion does not read the DSN
            patched_dsn.unlink(missing_ok=True)
    # The routing time ends here: the SES conversion and the checks below are not routing.
    elapsed = round(time.time() - t0, 3)
    if not ses_path.exists() or ses_path.stat().st_size == 0:
        return BoardResult(board.board_id, seed, "no_output", elapsed,
                           routing_log=str(log_path), warnings=warnings)
    try:
        stats = convert_ses(ses_path, board.raw_pcb, routed_path)
    except Exception as exc:
        return BoardResult(board.board_id, seed, "error", elapsed,
                           raw_path=str(ses_path), routing_log=str(log_path),
                           error=f"ses_to_kicadpcb: {exc}", warnings=warnings)
    issue = (f"SES nets not on the board: {stats['unmapped_nets'][:5]}" if stats["unmapped_nets"]
             else _routed_board_issue(board.raw_pcb, routed_path, board.pro))
    if issue:
        return BoardResult(board.board_id, seed, "error", elapsed,
                           raw_path=str(ses_path), routing_log=str(log_path), error=issue,
                           warnings=warnings)
    return BoardResult(board.board_id, seed, "ok", elapsed,
                       routed_pcb=str(routed_path), raw_path=str(ses_path),
                       routing_log=str(log_path), warnings=warnings)


class DsnRuleError(ValueError):
    """The DSN cannot hand a router the board's KiCad net-class rules exactly."""


# A DSN token: a quoted string, a paren, or a bare word (net names such as 3.3V or /5(**)).
# pcbnew quotes without escaping, so a quoted name runs to the next quote.
_DSN_TOKEN = re.compile(r'"[^"]*"|[()]|[^\s()"]+')
# pcbnew's via padstack name: Via[<top>-<bot>]_<diameter µm>:<drill µm>_um, both %.6g.
_DSN_VIA_NAME = re.compile(r"Via\[(\d+-\d+)\]_(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)_um")
_REGENERATE = "regenerate the DSN set with engine/pcbnew_prep"
# KiCad's netclass patterns (eda_pattern_match.cpp, CTX_NETCLASS): an anchored regex and an
# anchored wildcard ('*', '?'; these characters literal). Bus patterns (net_settings.cpp):
# NAME[0..7] or a {A B} group — a brace after ~ ^ _ is text markup.
_WILDCARD_LITERAL = set(".+^${}()|[]/\\")
_BUS_PATTERN = re.compile(r"\[\d+\.\.\d+\]|(?<![~^_])\{")

# The DSN / .kicad_pro reading below mirrors engine/pcbnew_prep/dsn_classes.py, which is
# GPL (it ships with the engine) and so is not imported into this BSD-3 runner.


def _dsn_block(text, head):
    """The first ``(head ...)`` block as nested lists of tokens, quotes stripped."""
    m = re.search(r"\(" + head + r"(?=[\s()])", text)
    if not m:
        raise DsnRuleError(f"DSN has no ({head} ...) block")
    stack = [[]]
    for t in _DSN_TOKEN.finditer(text, m.start()):
        tok = t.group()
        if tok == "(":
            stack.append([])
        elif tok == ")":
            done = stack.pop()
            stack[-1].append(done)
            if len(stack) == 1:
                return done
        else:
            stack[-1].append(tok[1:-1] if tok.startswith('"') else tok)
    raise DsnRuleError(f"DSN ({head} ...) block is not closed")


def _children(block, head):
    return [c for c in block if isinstance(c, list) and c and c[0] == head]


def _read_dsn_rules(dsn_path):
    """The DSN's length grid, signal layers, nets and net classes.

    Raises DsnRuleError for anything a router would otherwise have to guess.
    """
    try:
        text = Path(dsn_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise DsnRuleError(f"cannot read {dsn_path}: {exc}") from exc
    res = re.search(r"\(resolution\s+(\S+)\s+(\d+)\s*\)", text)
    if not res or res.group(1) != "um" or str(int(res.group(2))).rstrip("0") != "1":
        raise DsnRuleError("DSN header is not (resolution um <power of ten>)")
    structure = _dsn_block(text, "structure")
    layers = [layer[1] for layer in _children(structure, "layer")
              if len(layer) > 1 and ["type", "signal"] in layer[2:]]
    if not layers:
        raise DsnRuleError("DSN structure declares no signal layer")
    padstacks = [n for via in _children(structure, "via") for n in via[1:] if isinstance(n, str)]
    network = _dsn_block(text, "network")
    nets = [n[1] for n in _children(network, "net") if len(n) > 1 and isinstance(n[1], str)]
    classes = []
    for block in _children(network, "class"):
        if len(block) < 2 or not isinstance(block[1], str):
            raise DsnRuleError("DSN class block has no name")
        name = block[1]
        members = []
        for tok in block[2:]:
            if not isinstance(tok, str):
                break
            members.append(tok)
        rules = {}
        for rule in _children(block, "rule"):
            for r in rule[1:]:
                if isinstance(r, list) and len(r) == 2 and r[0] in ("width", "clearance"):
                    rules[r[0]] = float(r[1]) / 1000.0
        if set(rules) != {"width", "clearance"}:
            raise DsnRuleError(f"DSN class {name!r} has no (rule (width ...) (clearance ...)) of its own")
        use_via = [u[1] for c in _children(block, "circuit") for u in _children(c, "use_via")
                   if len(u) > 1]
        if len(use_via) != 1 or use_via[0] not in padstacks:
            raise DsnRuleError(f"DSN class {name!r} uses via {use_via}, not one padstack the "
                               f"structure declares ({padstacks}) — {_REGENERATE}")
        if not _DSN_VIA_NAME.fullmatch(use_via[0]):
            raise DsnRuleError(f"DSN class {name!r}: via name {use_via[0]!r} does not carry its size")
        classes.append({"name": name, "nets": members,
                        "width_mm": rules["width"], "clearance_mm": rules["clearance"],
                        "via": use_via[0]})
    if not classes:
        raise DsnRuleError("DSN network has no class block")
    return {"steps_per_um": int(res.group(2)), "layers": layers, "nets": nets, "classes": classes}


def _kicad_nm(mm):
    """``mm`` as KiCad holds it: whole nm, rounded half away from zero (KiRound)."""
    return int(math.floor(abs(mm) * 1e6 + 0.5)) * (1 if mm >= 0 else -1)


def _pcbnew_via_name(dia_mm, drill_mm, layer_span):
    """pcbnew's padstack name for a netclass via (specctra_export.cpp makeVia), with its own
    nm → µm arithmetic: scale() for the diameter, IU2um() for the drill, both printed %.6g."""
    dia_um = _kicad_nm(dia_mm) / (1e6 / 1000.0)
    drill_um = _kicad_nm(drill_mm) * (1000.0 / 1e6)
    return f"Via[{layer_span}]_{dia_um:.6g}:{drill_um:.6g}_um"


@functools.lru_cache(maxsize=None)
def _kicad9_pro_format():
    """The one .kicad_pro format accepted, shared with the DSN generators: the
    engine/pcbnew_prep/kicad9_pro_format.json facts file (data, not code — the GPL boundary)."""
    path = _PROJECT_ROOT / "engine" / "pcbnew_prep" / "kicad9_pro_format.json"
    try:
        fmt = json.loads(path.read_text(encoding="utf-8"))
        return fmt["project_meta_version"], fmt["net_settings_meta_version"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RunSetupError(f"cannot read {path} ({exc}) — check out the engine/ submodule") from exc


def _check_pro_format(pro, pro_name):
    found = ((pro.get("meta") or {}).get("version"),
             ((pro.get("net_settings") or {}).get("meta") or {}).get("version"))
    if found != _kicad9_pro_format():
        raise DsnRuleError(f"{pro_name}: .kicad_pro format (meta.version, net_settings.meta.version)"
                           f" = {found}, only the KiCad 9 format {_kicad9_pro_format()} is supported"
                           " — re-save the project with KiCad 9")


def _pro_net_classes(pro, pro_name, nets):
    """{net: class} as a KiCad 9 .kicad_pro assigns it, for the assignments KiCad and this runner
    read the same way: ``netclass_patterns`` entries that match exactly one net of the board (KiCad
    9 does not read a class's ``nets`` list; a listed net no pattern repeats is warned about).
    Anything wider (wildcard / regex matches, bus patterns, a net in two classes — KiCad merges
    those — or ``netclass_assignments``) is refused, not reimplemented."""
    settings = pro.get("net_settings") or {}
    if settings.get("netclass_assignments"):
        raise DsnRuleError(f"{pro_name} sets netclass_assignments — not supported")
    classes = settings.get("classes") or []
    defined = {c.get("name") for c in classes}
    entries = settings.get("netclass_patterns") or []
    repeated = {(e.get("pattern"), e.get("netclass")) for e in entries if isinstance(e, dict)}
    patterned = {e.get("pattern") for e in entries if isinstance(e, dict)}
    warnings = []
    for c in classes:
        # A Default entry no pattern names changes nothing by being ignored (the net is in Default).
        ignored = [n for n in c.get("nets") or [] if (n, c.get("name")) not in repeated
                   and not (c.get("name") == "Default" and n not in patterned)]
        if ignored:
            warnings.append(f"{pro_name}: class {c.get('name')!r} `nets` lists {ignored[:5]}"
                            f"{' ...' if len(ignored) > 5 else ''} ({len(ignored)}), which KiCad 9 "
                            "does not read (only netclass_patterns assign classes)")
    assigned = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue        # KiCad skips a non-object entry too (net_settings.cpp)
        pat, cls = entry.get("pattern"), entry.get("netclass")
        if not (pat and cls):
            continue
        if cls not in defined:
            raise DsnRuleError(f"{pro_name}: netclass pattern {pat!r} names undefined class {cls!r}")
        if _BUS_PATTERN.search(pat):
            raise DsnRuleError(f"{pro_name}: netclass pattern {pat!r} is a bus pattern — not supported")
        readings = [re.compile("^" + "".join(
            "." if c == "?" else ".*" if c == "*" else "\\" + c if c in _WILDCARD_LITERAL
            else re.escape(c) for c in pat) + "$")]
        rx = (pat if pat.startswith("^") else "^" + pat)
        try:
            readings.append(re.compile(rx if rx.endswith("$") else rx + "$"))
        except re.error:
            pass  # not a regex KiCad could compile either: the wildcard reading alone
        wider = sorted(n for n in nets if n != pat and any(r.match(n) for r in readings))
        if wider:
            raise DsnRuleError(f"{pro_name}: netclass pattern {pat!r} ({cls}) also matches "
                               f"{wider[:5]} as KiCad reads it — only exact net names are supported")
        assigned.append((pat, cls))
    out = {}
    for net in nets:
        found = {cls for name, cls in assigned if name == net}
        if len(found) > 1:
            raise DsnRuleError(f"{pro_name}: net {net!r} is in net classes {sorted(found)} — "
                               "KiCad merges them; not supported")
        out[net] = found.pop() if found else "Default"
    return out, warnings


_PRO_RULE_KEYS = ("track_width", "clearance", "via_diameter", "via_drill")


def _check_dsn_against_pro(dsn, pro_path):
    """Every DSN class carries its .kicad_pro net class exactly: the same nets, width and
    clearance on the DSN grid (not below, less than one grid step above), and the via padstack
    pcbnew names after the class's via size and drill. Returns (the .kicad_pro rules in mm per DSN
    class name and under None the .kicad_pro Default class's, the .kicad_pro warnings)."""
    pro_name = str(pro_path)       # every pcbench board's .kicad_pro has the same file name
    try:
        pro = json.loads(Path(pro_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DsnRuleError(f"cannot read the board's .kicad_pro {pro_path}: {exc}") from exc
    _check_pro_format(pro, pro_name)
    pro_classes = {c.get("name"): c for c in
                   ((pro.get("net_settings") or {}).get("classes") or [])}
    if "Default" not in pro_classes:
        raise DsnRuleError(f"{pro_name} has no Default net class")
    step_nm = 1000.0 / dsn["steps_per_um"]

    def rule(cls, key):
        # KiCad: a field the class leaves unset comes from Default.
        value = pro_classes[cls].get(key)
        value = pro_classes["Default"].get(key) if value is None else value
        if value is None:
            raise DsnRuleError(f"{pro_name}: net class {cls!r} has no {key}")
        return float(value)

    net_class, warnings = _pro_net_classes(pro, pro_name, dsn["nets"])
    listed = {}
    rules = {}
    for c in dsn["classes"]:
        cls = "Default" if c["name"] == "kicad_default" else c["name"]
        if cls not in pro_classes:
            raise DsnRuleError(f"DSN class {c['name']!r} is not a net class of {pro_name}")
        for net in c["nets"]:
            if net in listed:
                raise DsnRuleError(f"DSN lists net {net!r} in classes {listed[net]!r} and {cls!r}")
            listed[net] = cls
        for got, key in ((c["width_mm"], "track_width"), (c["clearance_mm"], "clearance")):
            want = rule(cls, key)
            if not -1e-6 <= got * 1e6 - _kicad_nm(want) < step_nm - 1e-6:
                raise DsnRuleError(f"DSN class {c['name']!r} {key} {got:g} mm is not the "
                                   f".kicad_pro rule {want:g} mm — {_REGENERATE}")
        span = _DSN_VIA_NAME.fullmatch(c["via"]).group(1)
        want_via = _pcbnew_via_name(rule(cls, "via_diameter"), rule(cls, "via_drill"), span)
        if c["via"] != want_via:
            raise DsnRuleError(f"DSN class {c['name']!r} uses via {c['via']!r}, the .kicad_pro "
                               f"class has {want_via!r} — {_REGENERATE}")
        rules[c["name"]] = {key: rule(cls, key) for key in _PRO_RULE_KEYS}
    rules[None] = {key: rule("Default", key) for key in _PRO_RULE_KEYS}   # the .kicad_pro Default
    moved = sorted(n for n in dsn["nets"] if listed.get(n) != net_class[n])
    if moved:
        raise DsnRuleError(f"DSN puts nets in other classes than {pro_name} does, e.g. "
                           + ", ".join(f"{n!r}: DSN {listed.get(n)!r} / pro {net_class[n]!r}"
                                       for n in moved[:3]) + f" — {_REGENERATE}")
    return rules, warnings


def _checked_dsn(dsn_path, pro_path):
    """The DSN's rules, checked against the .kicad_pro: (dsn, .kicad_pro rules per class, warnings).

    Any malformed input — not only the checks' own refusals — becomes a DsnRuleError, so one bad
    board is an error row and never ends the run."""
    try:
        dsn = _read_dsn_rules(dsn_path)
        return (dsn, *_check_dsn_against_pro(dsn, pro_path))
    except DsnRuleError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
        raise DsnRuleError(f"malformed DSN or .kicad_pro ({type(exc).__name__}: {exc})") from exc


def _canonical_copper_layers(dsn_layers, pcb_path):
    """The DSN's signal layers in canonical KiCad names: a DSN carries the board's user aliases
    (Top / Bottom / Front ...), KRT and KiCad need F.Cu / B.Cu / In<n>.Cu."""
    from methods.baselines.rule_based._converters.ses_to_kicadpcb import (
        canonical_copper_layer, pcb_layer_names)
    try:
        names = pcb_layer_names(Path(pcb_path).read_text(encoding="utf-8", errors="replace"))
        return [canonical_copper_layer(names, layer) for layer in dsn_layers]
    except (OSError, ValueError) as exc:
        raise DsnRuleError(f"{Path(pcb_path).name}: DSN layers {dsn_layers}: {exc}") from exc


def _parse_dsn_constraints(dsn_path, pro_path, pcb_path):
    """The KRT command-line rules of a board: its DSN classes, checked against its .kicad_pro, with
    the exact .kicad_pro values (KRT takes mm, so the DSN's grid rounding is not passed on) and its
    layers in canonical names. The pinned KRT (561f158) takes one clearance for every net and one
    via size: both the Default class's; whether other classes in use differ is recorded."""
    dsn, pro_rules, warnings = _checked_dsn(dsn_path, pro_path)
    in_use = [c for c in dsn["classes"] if c["nets"]]
    if not in_use:
        raise DsnRuleError("DSN classes list no nets")
    base = pro_rules[None]          # the .kicad_pro Default
    net_rule = {net: pro_rules[c["name"]] for c in in_use for net in c["nets"]}
    non_default = {net: round(r["track_width"], 6) for net, r in net_rule.items()
                   if abs(r["track_width"] - base["track_width"]) >= 1e-9}
    class_clearances = {c["name"]: round(pro_rules[c["name"]]["clearance"], 6) for c in in_use}
    vias = {(r["via_diameter"], r["via_drill"]) for r in net_rule.values()}
    return {"layers": _canonical_copper_layers(dsn["layers"], pcb_path),
            "track_width": round(base["track_width"], 6),
            "clearance": round(base["clearance"], 6),
            "class_clearances": class_clearances,
            "class_clearances_differ": set(class_clearances.values()) != {round(base["clearance"], 6)},
            "via_size": round(base["via_diameter"], 6),
            "via_drill": round(base["via_drill"], 6),
            "class_vias_differ": vias != {(base["via_diameter"], base["via_drill"])},
            "non_default_net_widths": non_default,
            "warnings": warnings}


def _krt_width_issue(log_text, constraints):
    """KRT's log (561f158 prints "Power net width assignments (N nets)") shows fewer per-net widths
    matched than were passed: a net KRT did not find on the board."""
    want = len(constraints["non_default_net_widths"])
    if not want:
        return None
    got = re.search(r"Power net width assignments \((\d+) nets\)", log_text)
    if not got or int(got.group(1)) != want:
        return f"KRT matched {got.group(1) if got else 0} of {want} per-net widths"
    return None


_COPPER_ITEM = re.compile(r"^\s*\((?:segment|arc|via)\b", re.MULTILINE)


def _require_engine():
    """The PCBWorld engine checks every routed board's KiCad load, so before any board it must be
    built, from this tree's C++ (the provenance guard's message names the reason)."""
    from pcb_world.engine import engine_available, ensure_router_provenance
    from pcb_world.engine.provenance import RouterProvenanceError
    if not engine_available():
        raise RunSetupError("the PCBWorld engine (kicad_rl_router) is not built — "
                            "bash engine/build_rl_router.sh")
    try:
        ensure_router_provenance()
    except RouterProvenanceError as exc:
        raise RunSetupError(f"the PCBWorld engine cannot be used: {exc}") from exc


def _kicad_load_issue(pcb_path, pro_path):
    """Why KiCad (through the PCBWorld engine) cannot load a routed board, or None."""
    from pcb_world.engine import KiCadEngine
    from pcb_world.engine.provenance import RouterProvenanceError
    from pcb_world.engine.router_client import EngineServerCrashed
    for attempt in (1, 2):
        try:
            eng = KiCadEngine.for_board_file(str(pcb_path), project_path=str(pro_path))
        except RouterProvenanceError as exc:
            raise RunSetupError(f"the PCBWorld engine build does not match its sources: {exc}") from exc
        except EngineServerCrashed as exc:
            if attempt == 1:
                continue    # a server start can fail under load: one retry before blaming the board
            return f"KiCad cannot load {Path(pcb_path).name}: engine server crashed twice: {exc}"
        except Exception as exc:    # any other load failure is the board's result, not the run's
            return f"KiCad cannot load {Path(pcb_path).name}: {type(exc).__name__}: {str(exc)[:300]}"
        eng.close()
        gc.collect()
        return None


def _routed_board_issue(raw_pcb, routed_pcb, pro_path):
    """Why a router's output is not a routed board: it added no copper, or KiCad cannot load it."""
    try:
        added = (len(_COPPER_ITEM.findall(Path(routed_pcb).read_text(encoding="utf-8", errors="replace")))
                 - len(_COPPER_ITEM.findall(Path(raw_pcb).read_text(encoding="utf-8", errors="replace"))))
    except OSError as exc:
        return f"cannot read the routed board: {exc}"
    if added <= 0:
        return "the router added no copper (0 nets routed)"
    return _kicad_load_issue(routed_pcb, pro_path)


def log_pro_warnings(warnings, where):
    """The run summary of the boards' warnings: the first and last 20, and where the full list is
    (each board's result)."""
    if not warnings:
        return
    log.warning("board warnings (%d; all of them in %s):", len(warnings), where)
    shown = warnings if len(warnings) <= 40 else [*warnings[:20], f"... {len(warnings) - 40} more ...",
                                                    *warnings[-20:]]
    for w in shown:
        log.warning("  %s", w)


def _krt_net_pattern(name):
    """A net name as a KRT ``--power-nets`` entry: KRT matches the entries with fnmatch, and
    argparse would read a leading '-' as an option — escape both so the entry is that net only."""
    pattern = glob.escape(name)
    return "[-]" + pattern[1:] if pattern.startswith("-") else pattern


# KRT routing hyperparameters of the KDD / CADET papers' runs, passed explicitly (each equals
# the pinned commit's default — tools/setup/fetch_baselines.sh pins the papers' KRT, 561f158).
KRT_PAPER_ARGS = (
    "--grid-step", "0.1",
    "--max-iterations", "200000",
    "--max-probe-iterations", "5000",
    "--max-ripup", "3",
    "--heuristic-weight", "1.9",
    "--ordering", "mps",
    "--via-cost", "50",
    "--via-proximity-cost", "10",
    "--turn-cost", "1000",
    "--direction-preference-cost", "50",
    "--hole-to-hole-clearance", "0.2",
    "--board-edge-clearance", "0.0",       # 561f158: 0 = use the track clearance
)
_KRT_MISSING = ("KRT route.py not found under {root} — run tools/setup/fetch_baselines.sh "
                "(pinned checkout at external/KiCadRoutingTools) or set KRT_ROOT")


class RunSetupError(RuntimeError):
    """A setup error that stops the run before any board, not a per-board result."""


class KrtNotFound(RunSetupError):
    """No usable KRT checkout to run."""


def _require_krt(prepare=False):
    """The KRT tree the run uses: KRT_ROOT, else the pinned external/KiCadRoutingTools. With
    ``prepare`` (once per run, before any board) also krt_runner's grid_router build, so a tree
    it cannot build for stops the run rather than failing every board."""
    try:
        from krt_runner import get_krt_root
    except ImportError as exc:
        raise KrtNotFound("krt_runner is not importable — "
                          "pip install -e methods/baselines/rule_based/krt") from exc
    root = get_krt_root()
    if not (root / "route.py").is_file():
        raise KrtNotFound(_KRT_MISSING.format(root=root))
    if prepare:
        try:
            from krt_runner import _ensure_grid_router_so
            _ensure_grid_router_so()
        except (ImportError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            raise KrtNotFound(f"KRT grid_router is not usable: {exc}") from exc
    return root


@functools.lru_cache(maxsize=None)
def _krt_commit(krt_root):
    """The KRT commit a run used: ``git rev-parse HEAD`` of its tree, with ``+dirty`` when tracked
    files carry local changes; None when it cannot be read (not the tree's own git checkout, or
    git failed)."""
    # -c safe.directory: read a checkout another user owns without touching any git config.
    git = ["git", "-c", f"safe.directory={krt_root}", "-C", str(krt_root)]
    try:
        top = subprocess.run([*git, "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                             timeout=30)
        # The KRT tree's own checkout only: git -C walks up to an enclosing repository.
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != Path(krt_root).resolve():
            return None
        sha = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        if sha.returncode != 0:
            return None
        dirty = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=120)
        if dirty.returncode != 0:
            return None             # the tree's state is unknown: not reported as clean
    except (OSError, subprocess.TimeoutExpired):
        return None
    return sha.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def _route_krt(board, seed, raw_dir, routed_dir, algo, timeout_s, *, krt_commit):
    """Route one board with KRT. ``krt_commit``: the run's _krt_commit, looked up once per run."""
    from krt_runner import run_route

    krt_root = _require_krt()
    routed_path = routed_dir / f"{board.board_id}_{algo}.kicad_pcb"
    routed_path.unlink(missing_ok=True)     # an earlier run's output must not pass as this one's
    run_path = raw_dir / f"{board.board_id}.krt_run.json"
    log_path = raw_dir / f"{board.board_id}.routing.log"
    try:
        constraints = _parse_dsn_constraints(board.dsn, board.pro, board.raw_pcb)
    except DsnRuleError as exc:
        return BoardResult(board.board_id, seed, "error", 0.0, error=f"DSN: {exc}")
    warnings = list(constraints["warnings"])
    for differ, what in (("class_clearances_differ", "clearances"), ("class_vias_differ", "vias")):
        if constraints[differ]:
            warnings.append(f"{board.board_id}: net classes in use have other {what} than the Default "
                            "class's; the pinned KRT takes one value, so they get the Default's")
    krt_record = dict(warnings=warnings, krt_rules={
        "clearance_mm": constraints["clearance"],
        "class_clearances_mm": constraints["class_clearances"],
        "class_clearances_differ": constraints["class_clearances_differ"],
        "via_size_mm": constraints["via_size"], "via_drill_mm": constraints["via_drill"],
        "class_vias_differ": constraints["class_vias_differ"]})
    args = [str(board.raw_pcb.resolve()), str(routed_path.resolve()),
            "--layers", *constraints["layers"],
            "--track-width", str(constraints["track_width"]),
            "--clearance", str(constraints["clearance"]),
            "--via-size", str(constraints["via_size"]),
            "--via-drill", str(constraints["via_drill"]),
            *KRT_PAPER_ARGS]
    nd = constraints["non_default_net_widths"]
    if nd:
        names = [_krt_net_pattern(n) for n in nd]; widths = [str(v) for v in nd.values()]
        args += ["--power-nets", *names, "--power-nets-widths", *widths]
    run_record = {"args": args, "krt_root": str(krt_root),
                  "krt_commit": krt_commit, "krt_pinned_commit": KRT_PINNED_COMMIT,
                  "krt_unpinned_allowed": krt_commit != KRT_PINNED_COMMIT and _krt_unpinned_allowed(),
                  "returncode": None, "wall_s": None, "status": None}

    t_end = None

    def finish(status, t0, **fields):
        # The runner's own record of the KRT call (KRT writes no metrics file of its own); the
        # wall time is KRT's run, not the checks of its output.
        run_record.update(returncode=fields.pop("returncode", None), status=status,
                          wall_s=round((t_end or time.time()) - t0, 3))
        run_path.write_text(json.dumps(run_record, indent=1))
        return BoardResult(board.board_id, seed, status, run_record["wall_s"],
                           routing_log=str(log_path), **fields, **krt_record)

    t0 = time.time()
    log_text = ""
    try:
        proc = run_route(args, capture_output=True, text=True, timeout=timeout_s)
        t_end = time.time()
        log_text = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        log_text = f"TIMEOUT after {timeout_s}s\n"
        for out in (exc.stdout, exc.stderr):    # KRT warns on both streams
            if out:
                log_text += out.decode("utf-8", errors="replace") if isinstance(out, bytes) else out
        return finish("timeout", t0)
    except Exception as exc:
        log_text = f"ERROR: {exc}\n"
        return finish("error", t0, error=str(exc))
    finally:
        log_path.write_text(log_text)
    if rc != 0 or not routed_path.exists():
        return finish("error", t0, returncode=rc, error=f"krt rc={rc}")
    issue = (_krt_width_issue(log_text, constraints)
             or _routed_board_issue(board.raw_pcb, routed_path, board.pro))
    if issue:
        return finish("error", t0, returncode=rc, error=issue)
    return finish("ok", t0, returncode=rc, routed_pcb=str(routed_path))


def _route_orthoroute(board, seed, raw_dir, routed_dir, algo, timeout_s, use_gpu=True):
    from methods.baselines.rule_based._converters import convert_ors
    main_script = _EXTERNAL / "OrthoRoute" / "main.py"
    ors_path = raw_dir / f"{board.board_id}.ORS"
    log_path = raw_dir / f"{board.board_id}.routing.log"
    routed_path = routed_dir / f"{board.board_id}_{algo}.kicad_pcb"
    if not board.orp.is_file():
        log_path.write_text(f"ERROR: missing .orp at {board.orp}\n")
        return BoardResult(board.board_id, seed, "error", 0.0,
                           routing_log=str(log_path),
                           error=f"no ORP: {board.orp}")
    cmd = [sys.executable, str(main_script),
           "headless", str(board.orp), "-o", str(ors_path)]
    if use_gpu:
        cmd.append("--use-gpu")
    else:
        cmd.append("--cpu-only")
    t0 = time.time()
    log_text = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        log_text = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        log_text = f"TIMEOUT after {timeout_s}s\n"
        if exc.stdout:
            log_text += exc.stdout.decode("utf-8", errors="replace")
        log_path.write_text(log_text)
        return BoardResult(board.board_id, seed, "timeout",
                           round(time.time()-t0, 3),
                           raw_path=str(ors_path) if ors_path.exists() else None,
                           routing_log=str(log_path))
    except Exception as exc:
        log_text = f"ERROR: {exc}\n"
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           routing_log=str(log_path), error=str(exc))
    finally:
        log_path.write_text(log_text)
    if rc != 0 or not ors_path.exists() or ors_path.stat().st_size == 0:
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           raw_path=str(ors_path) if ors_path.exists() else None,
                           routing_log=str(log_path),
                           error=f"orthoroute headless rc={rc}")
    try:
        convert_ors(ors_path, board.raw_pcb, routed_path)
    except Exception as exc:
        return BoardResult(board.board_id, seed, "error",
                           round(time.time()-t0, 3),
                           raw_path=str(ors_path),
                           routing_log=str(log_path),
                           error=f"ors_to_kicadpcb: {exc}")
    return BoardResult(board.board_id, seed, "ok",
                       round(time.time()-t0, 3),
                       routed_pcb=str(routed_path),
                       raw_path=str(ors_path),
                       routing_log=str(log_path))


def route_board(baseline, board, seed, raw_dir, routed_dir, algo, timeout_s, *, via_prohibit,
                angle_90, max_passes, extra_env, use_gpu, krt_commit):
    """Route one board with ``baseline`` (shared by this runner and eval/rollout/rule_based.py)."""
    if baseline == "freerouting":
        return _route_freerouting(board, seed, raw_dir, routed_dir, algo, timeout_s,
                                  via_prohibit, angle_90, max_passes, extra_env)
    if baseline == "krt":
        return _route_krt(board, seed, raw_dir, routed_dir, algo, timeout_s, krt_commit=krt_commit)
    if baseline == "orthoroute":
        return _route_orthoroute(board, seed, raw_dir, routed_dir, algo, timeout_s, use_gpu=use_gpu)
    return BoardResult(board.board_id, seed, "error", 0.0, error=f"unknown baseline {baseline!r}")


def _run_eval(board, routed_pcb, eval_logs_dir, reward_config, check_angle, algo):
    from eval import metrics as eval_metrics
    log_path = eval_logs_dir / f"{board.board_id}_{algo}.json"
    m = eval_metrics.evaluate_one(
        str(routed_pcb), str(board.pro),
        reward_config_name=reward_config, check_angle=check_angle)
    log_path.write_text(json.dumps(m, indent=2, default=str))
    return m, log_path


def _process_one_board(args):
    board = args["board"]; seed = args["seed"]; baseline = args["baseline"]
    raw_dir = args["raw_dir"]; routed_dir = args["routed_dir"]
    eval_logs_dir = args["eval_logs_dir"]; algo = args["algo"]
    timeout_s = args["timeout_s"]; do_eval = args["do_eval"]
    reward_config = args["reward_config"]; check_angle = args["check_angle"]
    via_prohibit = args["via_prohibit"]; angle_90 = args["angle_90"]
    max_passes = args["max_passes"]; extra_env = args["extra_env"]
    use_gpu = args["use_gpu"]
    try:
        result = route_board(baseline, board, seed, raw_dir, routed_dir, algo, timeout_s,
                             via_prohibit=via_prohibit, angle_90=angle_90, max_passes=max_passes,
                             extra_env=extra_env, use_gpu=use_gpu, krt_commit=args["krt_commit"])
    except RunSetupError:
        raise
    except Exception as exc:    # one board's own failure is its row, not the end of the run
        result = BoardResult(board.board_id, seed, "error", 0.0,
                             error=f"{type(exc).__name__}: {exc}")
    if do_eval and result.status == "ok" and result.routed_pcb:
        try:
            m, ev_log = _run_eval(board, Path(result.routed_pcb), eval_logs_dir,
                                  reward_config=reward_config,
                                  check_angle=check_angle, algo=algo)
            result.eval_log = str(ev_log)
            result.eval_metrics = {
                "success": m.get("success"),
                "routability": m.get("routability"),
                "track_count": m.get("track_count"),
                "via_count": m.get("via_count"),
                "wirelength_mm": m.get("wirelength_mm"),
                "drv_errors_only_count": m.get("drv_errors_only_count"),
                "drv_errors_and_promoted_count": m.get("drv_errors_and_promoted_count"),
                "final_potential": m.get("final_potential")}
        except Exception as exc:
            result.status = "eval_error"
            result.error = f"eval: {exc}"
    gc.collect()
    return asdict(result)


def _aggregate_summary(results, seed_dir):
    """summary.json over this run's boards: a board scored now enters with its
    eval log, every other one (routing or scoring failed) as a failure."""
    from eval import aggregation
    per_sample = []
    for r in results:
        if r["status"] == "ok" and r.get("eval_log"):
            per_sample.append(json.loads(Path(r["eval_log"]).read_text()))
        else:
            per_sample.append({"board_id": r["board_id"], "error": r.get("error") or r["status"]})
    if not per_sample:
        log.warning("aggregate: no boards in this run under %s", seed_dir)
        return
    agg = aggregation.aggregate(per_sample)
    (seed_dir / "eval" / "summary.json").write_text(
        json.dumps(agg, indent=2, default=str))
    # NOTE: _format_summary_txt was never part of the eval API (pre-existing
    # broken reference); summary.txt is skipped rather than crashing.


def _emit_progress(r, baseline):
    em = r.get("eval_metrics", {}) or {}
    print(
        f"  {baseline:11s} {r['board_id'][:34]:34s} {r['status']:10s} "
        f"t={r['elapsed_routing_s']:6.1f}s "
        f"r={em.get('routability', '-')!s:6} "
        f"DRV[ep]={em.get('drv_errors_and_promoted_count', '-')!s} "
        f"Phi={em.get('final_potential', '-')!s}")


def run_one_seed(boards, *, baseline, dataset, grid, seed, seed_dir,
                 timeout_s, workers, do_eval, reward_config, check_angle,
                 via_prohibit, angle_90, max_passes, extra_env, use_gpu):
    krt_commit = prepare_run(baseline)       # a missing piece stops the run before any board
    raw_dir = seed_dir / "raw"
    routed_dir = seed_dir / "routed"
    eval_logs_dir = seed_dir / "eval" / "logs"
    for d in (raw_dir, routed_dir, eval_logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    # The eval logs and summary describe this run only: a board that fails now, or is
    # not in this run, must not keep a previous run's score.
    (seed_dir / "eval" / "summary.json").unlink(missing_ok=True)
    for stale in eval_logs_dir.glob("*.json"):
        stale.unlink()
    algo = algorithm_tag(baseline, dataset, grid=grid)
    work = [
        dict(board=b, seed=seed, baseline=baseline, raw_dir=raw_dir,
             routed_dir=routed_dir, eval_logs_dir=eval_logs_dir, algo=algo,
             timeout_s=timeout_s, do_eval=do_eval, reward_config=reward_config,
             check_angle=check_angle, via_prohibit=via_prohibit,
             angle_90=angle_90, max_passes=max_passes, extra_env=extra_env,
             use_gpu=use_gpu, krt_commit=krt_commit)
        for b in boards]
    results = []
    if workers <= 1:
        for w in work:
            r = _process_one_board(w)
            _emit_progress(r, baseline)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process_one_board, w): w["board"].board_id for w in work}
            for fut in as_completed(futs):
                r = fut.result()
                _emit_progress(r, baseline)
                results.append(r)
    log_pro_warnings([w for r in results for w in r.get("warnings") or []],
                     where="manifest.json results[].warnings")
    if do_eval:
        _aggregate_summary(results, seed_dir)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from configs.loader.cli import StrictArgumentParser
    ap = StrictArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True,
                    choices=("freerouting", "krt", "orthoroute"))
    ap.add_argument("--dataset", required=True,
                    choices=("pcbench", "synthetic_1l", "synthetic_2l"))
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--difficulty", choices=D3_DIFFICULTIES, default=None,
                    help="pcbench: route one D3 tier of configs/datasets/d3.json "
                         "(easy = D3-A, medium = D3-B, hard = D3-C) at --split. "
                         "--dataset pcbench needs this or --all-corpus.")
    ap.add_argument("--all-corpus", action="store_true",
                    help="pcbench: route every corpus folder instead of a D3 tier "
                         "(includes the boards excluded from D3; output tree "
                         "PCBench_corpus).")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", action="append", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=None,
                    help=f"Per-board timeout (s). Defaults: {DEFAULT_TIMEOUTS}.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--reward-config", default="pcbworld_reward")
    ap.add_argument("--check-angle", type=int, choices=(45, 90), default=45)
    ap.add_argument("--max-passes", type=int, default=10)
    gpu_grp = ap.add_mutually_exclusive_group()
    gpu_grp.add_argument("--use-gpu", dest="use_gpu", action="store_true",
                         help="OrthoRoute: enable GPU. Faster but non-deterministic; "
                              "small per-board DRV wobble (~+1 segment) can occur "
                              "vs CPU/paper-canonical numbers.")
    gpu_grp.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                         default=False, help="OrthoRoute: force CPU-only (default). "
                                             "Bit-perfect reproducible.")
    args = ap.parse_args()
    via_prohibit = False
    angle_90 = False
    if args.dataset == "synthetic_1l":
        if args.baseline != "freerouting":
            ap.error("synthetic_1l only supports --baseline freerouting "
                     "(via prohibited, 90-deg routing only).")
        if args.grid is None:
            ap.error("synthetic_1l requires --grid {10,50,100,200,500}.")
        via_prohibit = True
        angle_90 = True
        if args.check_angle == 45:
            args.check_angle = 90
    timeout_s = args.timeout if args.timeout is not None else DEFAULT_TIMEOUTS[args.baseline]
    try:
        boards, valid, missing = select_boards_with_inputs(
            args.dataset, args.baseline, grid=args.grid, split=args.split,
            limit=args.limit, sample_substrings=args.sample,
            difficulty=args.difficulty, all_corpus=args.all_corpus)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    if not boards:
        ap.error("no boards matched the selection.")
    if missing:
        log.warning("Skipping %d board(s) with missing inputs:\n%s",
                    len(missing), describe_missing(missing))
    if not valid:
        ap.error("no boards remain after dropping incomplete inputs.")
    ds_tag = dataset_tag(args.dataset, grid=args.grid,
                         difficulty=args.difficulty, split=args.split,
                         all_corpus=args.all_corpus)
    algo_tag = algorithm_tag(args.baseline, args.dataset, grid=args.grid)
    run_root = Path(args.output_root) / ds_tag / algo_tag
    log.info("=== %s / %s ===", ds_tag, algo_tag)
    log.info("boards: %d  workers: %d  timeout: %ds  seeds: %d  gpu: %s",
             len(valid), args.workers, timeout_s, args.seeds, args.use_gpu)
    log.info("output: %s", run_root)
    seed_count = args.seeds if args.baseline == "freerouting" else 1
    if seed_count != args.seeds and args.seeds > 1:
        log.info("--seeds %d collapsed to 1 (%s is deterministic).",
                 args.seeds, args.baseline)
    all_results = {}
    for seed in range(seed_count):
        seed_dir = run_root / f"seed{seed}"
        log.info("--- seed %d -> %s", seed, seed_dir)
        results = run_one_seed(
            valid, baseline=args.baseline, dataset=args.dataset,
            grid=args.grid,
            seed=seed, seed_dir=seed_dir, timeout_s=timeout_s,
            workers=args.workers, do_eval=not args.no_eval,
            reward_config=args.reward_config, check_angle=args.check_angle,
            via_prohibit=via_prohibit, angle_90=angle_90,
            max_passes=args.max_passes, extra_env={}, use_gpu=args.use_gpu)
        all_results[seed] = results
        (seed_dir / "manifest.json").write_text(json.dumps({
            "dataset": args.dataset, "dataset_tag": ds_tag,
            "baseline": args.baseline, "algorithm_tag": algo_tag,
            "grid": args.grid, "split": args.split, "seed": seed,
            "difficulty": args.difficulty, "all_corpus": args.all_corpus,
            "n_boards": len(valid),
            "boards": [b.board_id for b in valid],
            "timeout_s": timeout_s,
            "via_prohibit": via_prohibit, "angle_90": angle_90,
            "max_passes": args.max_passes, "use_gpu": args.use_gpu,
            "reward_config": args.reward_config,
            "check_angle": args.check_angle, "results": results,
        }, indent=2, default=str))
    ok = sum(1 for rrs in all_results.values() for r in rrs if r["status"] == "ok")
    total = sum(len(r) for r in all_results.values())
    log.info("done: %d/%d ok across %d seed(s).", ok, total, seed_count)
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
