"""Golden configs of real launcher command lines: what each recipe's trainer (or LLM rollout) resolves.

A fixture (``tests/fixtures/recipe_argv/*.json``; private recipes in the private test tree's own
``fixtures/recipe_argv/``) is one argv a launcher printed (captured with python replaced
by an argv recorder): ``kind`` ``train`` (``algo`` ``ppo`` | ``grpo``) or ``llm``, and ``argv`` with
placeholders — ``<REPO>`` (the checkout), ``<OUT>`` (the capture's output root) and
``<CKPT:name>`` (the checkpoint of ``ckpt_real`` fixture ``name``, :mod:`tests.helpers.ckpt_golden`;
a resume names the run's own checkpoint). :func:`resolved` parses and resolves it — no env, engine or model — into every
non-runtime value (a training recipe) or the LLM env config + prompt version, paths back to their
placeholders. ``expected`` is written by ``python -m tests.helpers.recipe_golden <ckpt_real dir>
<fixture.json>...``.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

from tests.helpers.ckpt_golden import jsonable, train_run, write_golden
from tests.helpers.ckpt_payload import save_ckpt

REPO = Path(__file__).resolve().parents[2]


def _substitute(arg: str, subs: dict[str, str]) -> str:
    for k, v in subs.items():
        arg = arg.replace(k, v)
    return arg


def resolved(fx: dict[str, Any], ckpt_dir: Path, tmp_path: Path) -> dict[str, Any]:
    """The resolved values of recipe fixture ``fx`` (module docstring); ``<CKPT:name>`` comes from
    ``ckpt_dir``, checkpoints are written under ``tmp_path``."""
    subs = {"<REPO>": str(REPO), "<OUT>": str(tmp_path / "out")}
    for a in fx["argv"]:
        if a.startswith("<CKPT:"):
            name = a[len("<CKPT:"):-1]
            subs[a] = str(save_ckpt(tmp_path / f"{name}.pt", json.loads((ckpt_dir / f"{name}.json").read_text())))
    argv = [_substitute(a, subs) for a in fx["argv"]]
    if fx["kind"] == "llm":
        from dataclasses import asdict

        from eval.args import build_eval_parser, llm_env_config

        args = build_eval_parser().parse_args(argv)
        values = {**asdict(llm_env_config(args)), "prompt_version": args.prompt_version}
    else:
        from configs.loader.schema import GRPOConfig, PPOConfig
        from methods.rl_agent.training.args import train_policies

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run = train_run(fx["algo"], argv)
        pol = train_policies(GRPOConfig() if fx["algo"] == "grpo" else PPOConfig())
        values = {k: v for k, v in vars(run).items() if pol[k].scope != "runtime"}
    back = {v: k for k, v in subs.items()}
    out = json.dumps(dict(sorted(values.items())), default=repr)
    for real, placeholder in sorted(back.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(real, placeholder)
    return jsonable(json.loads(out))


def check(path: Path, ckpt_dir: Path, tmp_path: Path) -> None:
    """Recipe fixture ``path`` resolves to its ``expected``."""
    fx = json.loads(path.read_text())
    assert resolved(fx, ckpt_dir, tmp_path) == fx["expected"]


def main(ckpt_dir: str, paths: list[str]) -> None:
    import tempfile

    for p in map(Path, paths):
        fx = json.loads(p.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            fx["expected"] = resolved(fx, Path(ckpt_dir), Path(tmp))
        write_golden(p, fx)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
