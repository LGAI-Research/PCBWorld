"""Print a run's config provenance table: every key, its value, where it came from.

    python -m configs.loader.explain <run_dir | run_record.yaml | checkpoint.pt> [--all]

A run directory or record file is read from its ``run_record.yaml`` (a resumed run's
``run_record.resume_iter<N>.yaml`` records are listed under the table); a directory without one
from its newest other run record (a tool's output folder), else — written before run records
existed — from its newest checkpoint (named). A checkpoint is read through
the checkpoint loader — its schema-2 ``config`` with the stored sources, or an older ``args``
with the keys the loader filled. By default keys still at their schema default are folded into
a count; ``--all`` lists them too. ``--dump-config-only`` of the trainers prints the same table.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from configs.loader.record import RECORD_GLOB, RUN_RECORD


def _fmt(value: Any, width: int = 48) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def format_table(provenance: Mapping[str, Any], *, all_keys: bool = False) -> str:
    """The provenance table: ``key  value  source  where`` per key (``provenance`` maps a key to
    a :class:`~configs.loader.resolve.Entry` or to its run-record dict). Keys whose source is
    ``defaults`` are folded into a count unless ``all_keys``."""
    rows, folded = [], 0
    for key in sorted(provenance):
        e = provenance[key]
        value, source, where = ((e["value"], e["source"], e["where"]) if isinstance(e, Mapping)
                                else (e.value, e.source, e.where))
        if source == "defaults" and not all_keys:
            folded += 1
            continue
        rows.append((key, _fmt(value), source, where))
    width = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(("key", "value", "source"))]
    lines = [f"{'key':<{width[0]}}  {'value':<{width[1]}}  {'source':<{width[2]}}  where"]
    lines += [f"{k:<{width[0]}}  {v:<{width[1]}}  {s:<{width[2]}}  {w}" for k, v, s, w in rows]
    if folded:
        lines.append(f"({folded} more keys at their schema default; --all lists them)")
    return "\n".join(lines)


def load_provenance(target: str | Path) -> Mapping[str, Any]:
    """``{key: {value, source, where}}`` of a run directory, run record or checkpoint (module
    docstring); a file without a ``config`` mapping raises ``ValueError``."""
    target = _target(Path(target))
    if target.suffix == ".yaml":
        from configs.loader import load_config
        record = load_config(target)
        if not isinstance(record, Mapping) or not isinstance(record.get("config"), Mapping):
            raise ValueError(f"{target} is not a run record (no `config` mapping)")
        return record["config"]
    import torch

    from configs.loader.ckpt import load_ckpt_config

    payload = torch.load(target, map_location="cpu", weights_only=False)
    values, sources = load_ckpt_config(payload, strict=False)
    schema2 = "config" in payload
    stored = payload["config"]["sources"] if schema2 else {}
    where = "checkpoint config" if schema2 else "checkpoint args (pre-schema-2)"
    # The writer's sources (cli / recipe / defaults ...) unless the loader derived or filled the key.
    src = {**{k: "ckpt" for k in values}, **stored, **{k: s for k, s in sources.items() if s != "ckpt"}}
    return {k: {"value": v, "source": src[k], "where": where} for k, v in values.items()}


def _target(target: Path) -> Path:
    """A run directory's ``run_record.yaml``, else its newest other run record
    (:data:`~configs.loader.record.RECORD_GLOB`: a tool's output folder), else (a run before run
    records) its newest checkpoint — either named on stderr; any other target as given."""
    if not target.is_dir():
        return target
    if (target / RUN_RECORD).is_file():
        return target / RUN_RECORD
    for kind, pattern in (("run record", RECORD_GLOB), ("checkpoint", "*.pt")):
        found = sorted(target.glob(pattern), key=lambda p: p.stat().st_mtime)
        if found:
            others = [p.name for p in found[:-1]]
            if len(others) > 5:   # a long-lived tool folder can hold hundreds
                others = [*others[:3], "...", *others[-2:], f"{len(found) - 1} in all"]
            print(f"[explain] {target} has no {RUN_RECORD}: reading its newest {kind} {found[-1].name}"
                  + (f" (also there: {', '.join(others)})" if others else ""),
                  file=sys.stderr)
            return found[-1]
    raise FileNotFoundError(f"{target} has neither a run record ({RECORD_GLOB}) nor a checkpoint (*.pt)")


def main(argv: list[str] | None = None) -> int:
    from configs.loader.cli import StrictArgumentParser

    p = StrictArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("target", help="run directory, run_record.yaml or checkpoint .pt")
    p.add_argument("--all", action="store_true", help="also list keys at their schema default")
    args = p.parse_args(argv)
    print(format_table(load_provenance(args.target), all_keys=args.all))
    target = Path(args.target)
    def by_iteration(p: Path) -> tuple[bool, int, str]:   # a non-integer stem (a hand copy) sorts last
        n = p.stem.removeprefix("run_record.resume_iter")
        return (not n.isdigit(), int(n) if n.isdigit() else 0, p.name)

    resumed = sorted((target if target.is_dir() else target.parent).glob("run_record.resume_iter*.yaml"),
                     key=by_iteration)
    if resumed and target.suffix != ".pt":
        print("resumed runs keep their own record (explain each): " + ", ".join(p.name for p in resumed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
