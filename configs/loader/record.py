"""Run record — what a run is configured with and where every value came from, written at start.

:func:`write_run_record` writes ``<run dir>/run_record.yaml`` at the start of every Tier-1
entry point (PPO/GRPO training — plus ``run_record.val.yaml``, the validation env —, the
eval pipeline, LLM eval, each async-val checkpoint), before any env, engine or model is
built, so the file survives a crash. One file holds:

* ``config`` — per key the resolved value, its source (``defaults`` | ``mode-default`` |
  ``ckpt`` | ``legacy:*`` | ``derived`` | ``recipe:<file>`` | ``cli`` | ``code:<file>`` | ``cell``: a re-scoring run's record), where it
  was set, the chain of every layer that set it (:class:`configs.loader.resolve.Entry`) and
  ``effective_reason`` — when or why no sink readback fills the value: the field's declared
  ``readback_reason``, :data:`AT_ENV_CREATION` for an env field of a run that reads its env
  back, else :data:`NO_READBACK`;
* the rule files the config names (masking / reward / DRC) with their content hash;
* the board list (:func:`board_record`): where it came from with that file's hash, its size,
  a hash of the path list, and for a short list the paths and their ``.kicad_pro`` set hash;
* the router build (its ``ENGINE_CPP_HASH`` stamp, or why there is none), the repo version and
  git commit/dirty;
* the ``PCBWORLD_*`` / ``KICAD_ENGINE_*`` environment variables;
* ``effective`` — appended by :func:`fill_effective` once the run builds its env: per key the
  value read back from the sink that consumes it (``PCBWorld.effective_config``).

:func:`record_tool_run` writes the same record for a tool that makes experiment inputs or
scores (data generators, dataset builders, scorers): its command line as the config and, per
input, the hashes of the run records that made it (:func:`input_record`).

:func:`collect_provenance` is the code identity a checkpoint stamps.

Light at import: stdlib + PyYAML; the rule and router lookups import ``pcb_world`` lazily, so
``--dump-config-only`` works without a GPU or a router build.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[2]

RUN_RECORD = "run_record.yaml"
#: ``effective_reason`` of a key nothing reads back (not an env field, or no fill follows).
NO_READBACK = "no readback yet"
#: ``effective_reason`` of an env field read back once the run builds its env.
AT_ENV_CREATION = "read back when the env is built"
#: Environment variables a run record keeps (prefixes).
ENV_PREFIXES = ("PCBWORLD_", "KICAD_ENGINE_")
#: Config keys that name a rule file: key -> rule kind (masking / reward / a plain path).
RULE_KEYS = {"masking_rule": "masking", "reward_rule": "reward", "reward_config": "reward",
             "drc_config_path": "path"}
#: Board lists up to this size record their paths and ``.kicad_pro`` set hash; a longer one
#: only its source and path-list hash — hashing every project file would make the start
#: O(boards) (100k training boards on NFS: minutes).
BOARD_PATHS_MAX = 64


def _read_version() -> str:
    """README ``<!--VERSION-->`` marker (the version single source of truth)."""
    try:
        # [^<] (not [\d.]) so personal-line suffixes (v0.26.0+b16) survive.
        m = re.search(
            r"<!--VERSION-->v?([^<\s]+)<!--/VERSION-->",
            (_REPO / "README.md").read_text(),
        )
        return f"v{m.group(1)}" if m else "unknown"
    except OSError:
        return "unknown"


def collect_provenance() -> dict:
    """Best-effort code identity for ckpt stamping: repo version + git HEAD.

    Purely informational (loaders print it, never enforce it) — failures
    outside a git checkout degrade to "unknown" instead of blocking training.
    Single definition shared with the trainer's ckpt stamp (loop.py) so the
    version source stays the README marker above.
    """
    prov: dict = {
        "repo_version": _read_version().lstrip("v"),
        "git_commit": "unknown", "git_dirty": None,
    }
    try:
        prov["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=_REPO, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        prov["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"],
            cwd=_REPO, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return prov


def run_record_filename(run: Any) -> str:
    """``run_record.yaml``; a resumed training run (``run.resume``) keeps its own
    ``run_record.resume_iter<N>.yaml`` beside it (N = the checkpoint's iteration + 1, from the
    checkpoint load resolve made — ``run.ckpt_meta``)."""
    return RUN_RECORD if not run.resume else f"run_record.resume_iter{run.ckpt_meta['iteration'] + 1}.yaml"


def _sha256(path: str | Path) -> str:
    """Content sha256, read in 1 MiB chunks (an input may be a multi-GB checkpoint)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rule_files(values: Mapping[str, Any]) -> dict[str, Any]:
    """The rule files the config names, each with its resolved path and content hash."""
    from pcb_world.core.masking import masking_rule_path
    from pcb_world.core.reward_config import reward_rule_path

    find = {"masking": masking_rule_path, "reward": reward_rule_path, "path": Path}
    out: dict[str, Any] = {}
    for key, kind in RULE_KEYS.items():
        name = values.get(key)   # only the rule keys this config has
        if name is None:
            continue
        path = find[kind](str(name))
        out[key] = {"name": str(name), "path": None if path is None else str(path),
                    "sha256": None if path is None else _sha256(path)}
        if path is None:
            out[key]["reason"] = "registered in-process (no rule file)"
    return out


def board_record(paths: Sequence[str], **sources: Any) -> dict[str, Any]:
    """The board list a run resolved: its count, the options that named it (``sources``, e.g.
    ``boards_json=<split>``; None values are dropped) with each file's content hash, the hash of
    the path list and — for at most :data:`BOARD_PATHS_MAX` boards — the paths and the hash of
    the ``.kicad_pro`` set beside them (sorted, content)."""
    out: dict[str, Any] = {
        "count": len(paths),
        "sources": {k: {"path": str(v), "sha256": _sha256(v) if Path(v).is_file() else None}
                    for k, v in sources.items() if v is not None},
        "paths_sha256": hashlib.sha256("\n".join(map(str, paths)).encode()).hexdigest(),
    }
    if len(paths) > BOARD_PATHS_MAX:
        out["kicad_pro"] = {"sha256": None, "reason": f"over {BOARD_PATHS_MAX} boards: not hashed "
                            "at start (sources + paths_sha256 identify the list)"}
        return out
    h = hashlib.sha256()
    found = 0
    for pro in sorted({str(Path(p).with_suffix(".kicad_pro")) for p in paths}):
        if os.path.isfile(pro):
            found += 1
            h.update(pro.encode() + b"\0" + _sha256(pro).encode() + b"\0")
    out["paths"] = [str(p) for p in paths]
    out["kicad_pro"] = {"count": found, "sha256": h.hexdigest() if found else None}
    return out


def _router_build() -> dict[str, Any]:
    """The router this process loads: its ``ENGINE_CPP_HASH`` stamp, after the provenance
    guard (``pcb_world.engine.provenance.router_cpp_hash``). A router of other C++ — or none
    built for this tree — is recorded with the guard's message, not raised: the record is
    written before any engine exists (``--dump-config-only`` in an unbuilt worktree)."""
    from pcb_world.engine import router_lib_dir
    from pcb_world.engine.provenance import STAMP_NAME, RouterProvenanceError, router_cpp_hash

    try:
        stamp, lib_dir = router_cpp_hash(), router_lib_dir()
    except RouterProvenanceError as e:
        return {"lib_dir": None, STAMP_NAME: None, "reason": f"router provenance check failed: {e}"}
    out = {"lib_dir": lib_dir, STAMP_NAME: stamp}
    if stamp is None:
        out["reason"] = "no stamped router there (none built, or built before the stamp)"
    return out


def _env_vars() -> dict[str, str]:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}


def run_record(run: Any, *, boards: Mapping[str, Any] | None, readback: bool = False,
               extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The run record of the :class:`~configs.loader.resolve.RunConfig` ``run`` as a dict
    (``boards``: :func:`board_record`; ``readback``: a :func:`fill_effective` follows)."""
    reasons = _effective_reasons(readback)
    return {
        "schema": 2,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": run.resolve_mode,
        "code": collect_provenance(),
        "router_build": _router_build(),
        "env_vars": _env_vars(),
        "rules": _rule_files(run.values),
        "boards": None if boards is None else dict(boards),
        **({} if extra is None else {"extra": dict(extra)}),
        "config": {k: {"value": e.value, "source": e.source, "where": e.where,
                       "chain": [dict(c) for c in e.chain],
                       "effective_reason": reasons.get(k, NO_READBACK)}
                   for k, e in run.provenance.items()},
    }


def _effective_reasons(readback: bool) -> dict[str, str]:
    """``effective_reason`` per ``EnvConfig`` field — and per inverted dest of one
    (``no_drc_tokens``, the key a training run has for ``emit_drc_tokens``): its declared
    ``readback_reason``, else — when the run reads its env back (``readback``) —
    :data:`AT_ENV_CREATION`."""
    from dataclasses import fields

    from configs.loader.schema import INVERTED_DESTS, EnvConfig

    out = {}
    for f in fields(EnvConfig):
        if "readback_reason" in f.metadata:
            out[f.name] = f.metadata["readback_reason"]
        elif readback:
            out[f.name] = AT_ENV_CREATION
    out.update({dest: out[name] for dest, name in INVERTED_DESTS.items() if name in out})
    return out


def fill_effective(path: str | Path, run: Any, effective: Mapping[str, Any]) -> None:
    """Append the ``effective`` section to the run record at ``path``: of a constructed env's
    ``PCBWorld.effective_config()``, the keys ``run`` has — a field ``run`` holds under its
    inverted dest (``INVERTED_DESTS``) under that dest, negated. Appended — the record written
    at start is not read back or rewritten."""
    import yaml

    from configs.loader.schema import INVERTED_DESTS

    dest = {name: d for d, name in INVERTED_DESTS.items()}
    section = {"effective": {}}
    for k, v in effective.items():
        if k in run.provenance:
            section["effective"][k] = _plain(v)
        elif dest.get(k) in run.provenance:
            section["effective"][dest[k]] = not v
    with open(path, "a") as f:
        f.write(yaml.safe_dump(section, sort_keys=False, allow_unicode=True))


def write_run_record(path: str | Path, run: Any, *, boards: Mapping[str, Any] | None,
                     readback: bool = False, extra: Mapping[str, Any] | None = None,
                     echo: bool = True) -> Path:
    """Write :func:`run_record` to ``path`` (a directory gets :data:`RUN_RECORD`) and return it.
    ``boards=None`` records that this entry point resolves no board list; ``readback``: this
    process appends the ``effective`` section later (:func:`fill_effective`). ``echo`` prints
    the provenance table (it lands at the head of a launch log)."""
    import yaml

    path = Path(path)
    if path.suffix != ".yaml":
        path = path / RUN_RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_plain(run_record(run, boards=boards, readback=readback, extra=extra)),
                                   sort_keys=False, allow_unicode=True))
    if echo:
        from configs.loader.explain import format_table
        print(format_table(run.provenance))
    print(f"[config] run record -> {path}")
    return path


# ---------------------------------------------------------------------------
# Tier-2 tools: the tools that make experiment inputs or scores
# ---------------------------------------------------------------------------
# A data generator, a dataset builder or a scorer writes the same record into its output: its
# command line resolved over the parser defaults (no schema — every option is an experiment
# key), its argv, and per input the run records that made that input (the provenance chain).

#: A tool's record beside a file output: ``<file>`` + this suffix.
FILE_RECORD_SUFFIX = ".run_record.yaml"
#: The records that describe a directory: every file directly in it matching this pattern
#: (``run_record.yaml``, ``run_record.val.yaml``, ``run_record_<tag>.yaml``, ``<file>.run_record.yaml``, and
#: the ``<stem>.<k>.yaml`` of a later run that appended to the same output).
RECORD_GLOB = "*run_record*.yaml"
_FILE_RECORD_RE = re.compile(r"(.+)\.run_record(?:\.\d+)?\.yaml")


def _file_record_owner(name: str) -> str | None:
    """The file a record named ``<file>.run_record[.<k>].yaml`` sits beside, else None."""
    m = _FILE_RECORD_RE.fullmatch(name)
    return m.group(1) if m else None


def input_record(path: str | Path, *, skip: Path | None = None) -> dict[str, Any]:
    """What a tool read at ``path``: ``records`` = the run records that made it, each by content
    sha256 (``skip`` left out) — a directory: every :data:`RECORD_GLOB` file directly in it; a
    file: its own ``<file>`` :data:`FILE_RECORD_SUFFIX` (and the ``.<k>`` ones of later appending
    runs), else its own ``sha256`` plus the directory records of its folder (a checkpoint beside
    its training run's ``run_record.yaml``; ``records_from: folder`` — they describe the run, not
    which of its files was read). A directory no record describes is recorded with the reason.
    A missing path raises."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tool input {p} does not exist")
    out: dict[str, Any] = {"path": os.path.abspath(p)}
    if p.is_dir():
        found = [q for q in sorted(p.glob(RECORD_GLOB)) if q != skip]
    else:
        beside = [q for q in sorted(p.parent.glob(RECORD_GLOB)) if q != skip]
        found = [q for q in beside if _file_record_owner(q.name) == p.name]
        if not found:
            out["sha256"] = _sha256(p)
            found = [q for q in beside if _file_record_owner(q.name) is None]
            if found:
                out["records_from"] = "folder"
    out["records"] = {q.name: _sha256(q) for q in found}
    if not found and p.is_dir():
        out["reason"] = "no run record in it (made before tools recorded their runs, or by hand)"
    return out


def _free_record_path(path: Path) -> Path:
    """Create and return ``path`` if it does not exist, else the first free ``<stem>.<k>.yaml``
    (k = 1, 2, ...). Created with ``open(..., "x")``, so two runs appending at once never take
    the same name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    k = 0
    while True:
        free = path if k == 0 else path.with_name(f"{path.stem}.{k}.yaml")
        try:
            with open(free, "x"):
                return free
        except FileExistsError:
            k += 1


def record_tool_run(parser: Any, args: Any, *, out_dir: str | Path | None = None,
                    out_file: str | Path | None = None, name: str = RUN_RECORD,
                    inputs: Sequence[str | Path | None] = (), append: bool = False) -> Path:
    """Write the run record of a tool parse (``parser``: a
    :class:`~configs.loader.cli.StrictArgumentParser` that parsed ``args``) and return its path:
    ``<out_dir>/<name>``, or beside ``out_file`` as ``<out_file>`` :data:`FILE_RECORD_SUFFIX`
    (exactly one of the two); ``append`` (a tool that appends to its output, e.g. a resumed shard):
    an existing record there is kept and this one takes the first free ``<stem>.<k>.yaml``. The config is :func:`configs.loader.resolve.resolve` of the parser
    defaults < the command line (:func:`configs.loader.resolve.cli_layer`) < ``code:<prog>`` = the
    other dests whose value the tool changed after parsing; ``extra`` holds ``argv`` and
    ``inputs`` (:func:`input_record` per non-None path). Called when the outputs are known,
    before the work, so a crash still leaves it."""
    import sys

    from configs.loader.knob import Policy
    from configs.loader.resolve import Layer, cli_layer, parser_defaults, resolve

    if (out_dir is None) == (out_file is None):
        raise ValueError("record_tool_run takes exactly one of out_dir / out_file")
    path = (Path(out_dir) / name if out_file is None
            else Path(out_file).with_name(Path(out_file).name + FILE_RECORD_SUFFIX))
    defaults = parser_defaults(parser, {})
    policies = {k: Policy("experiment", False, False, f"{parser.prog} {k}") for k in defaults.values}
    cli = cli_layer(parser, args)
    changed = {k: getattr(args, k) for k, v in defaults.values.items()
               if k not in cli.values and getattr(args, k) != v}
    user = [cli] + ([Layer(f"code:{parser.prog}", changed)] if changed else [])
    run = resolve("fresh", policies=policies, defaults=[defaults], user=user)
    # An appending run's own record does not exist yet (earlier runs' records are inputs); its
    # name is reserved only now, so a failure above leaves no empty record behind.
    extra = {"argv": list(sys.argv),
             "inputs": [input_record(p, skip=None if append else path) for p in inputs if p is not None]}
    if append:
        path = _free_record_path(path)
    return write_run_record(path, run, boards=None, extra=extra, echo=False)


def _plain(x: Any) -> Any:
    """Tuples, mapping proxies and paths as plain YAML types."""
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


# ---------------------------------------------------------------------------
# Train vs validation env (``--expect-env-diff``)
# ---------------------------------------------------------------------------
# The trainer records the validation env as a run record of its own (VAL_RUN_RECORD: the
# resolved eval-mode config, its seed and policy_net_select under ``extra``) and compares it
# with the training env key by key. A difference is not judged here; it is reported, and the
# launch declares which differences are intended — the 2026-08-20 audit found five knobs on
# which the two silently differed.

#: The validation env's run record, beside the training run's.
VAL_RUN_RECORD = "run_record.val.yaml"


def env_diff(train: Mapping[str, Any], val: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{key: {"train": ..., "val": ...}}`` for every key of two env records
    (``RLEnvConfig.to_record`` form, one key set) whose values differ."""
    return {k: {"train": train[k], "val": val[k]} for k in sorted(train) if train[k] != val[k]}


def check_expected_env_diff(diff: Mapping[str, Mapping[str, Any]], expected: str) -> None:
    """Halt unless every train/val difference was declared at launch.

    Intent belongs to the experiment, not to a list in the source tree: a run
    may legitimately want validation to differ (a deliberately harder training
    reward, say), and a global "these keys may differ" table would either
    forbid that or quietly bless a genuine mistake. So the declaration travels
    with the launch command, and the error message is the line to paste.
    """
    declared = {k.strip() for k in expected.split(",") if k.strip()}
    unexplained = sorted(set(diff) - declared)
    if not unexplained:
        return
    rows = "\n".join(
        f"    {k:<28} train={diff[k]['train']!r}  val={diff[k]['val']!r}"
        for k in unexplained
    )
    raise SystemExit(
        "\n[env-contract] the training env and the validation env are "
        "configured differently:\n"
        f"{rows}\n\n"
        "  If the difference is intended, paste this into the launch command:\n"
        f"    --expect-env-diff {','.join(sorted(set(unexplained) | declared))}\n"
        "  If it is not, start by finding where the two paths diverge.\n"
    )
