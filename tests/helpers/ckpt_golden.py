"""Golden configs of real checkpoints: what the loader and the trainers resolve from each one.

A fixture (``tests/fixtures/ckpt_real/*.json``; private runs in the private test tree's own
``fixtures/ckpt_real/``) is a :func:`tests.helpers.ckpt_payload.ckpt_payload` fixture dict cut from one
real checkpoint: its stored ``args``, ``provenance``, ``trainer`` marker, ``obs_probe`` presence and the
shapes of the weights the loader derives the token layout from (``tokenizer.vocab.*`` and one
``critic_head.*``; no tensors); ``source`` names the checkpoint, ``algo`` the trainer that wrote it
(``ppo`` | ``grpo`` | ``bc``). Scrubbed on extraction, so no machine or person is named: an absolute
path becomes ``<abs>/<its last two parts>``, a relative path under the old split-config folder (``configs/`` + ``splits/``) becomes
``<rel>/<file>``, every ``wandb_*`` string ``<wandb>``, an ``ssh<N>`` host tag ``<host>``.
:func:`resolved` computes, from the fixture alone:

* ``load`` — :func:`configs.loader.ckpt.load_ckpt_config`: the checkpoint fields (+ the renamed training
  switches) and every non-``ckpt`` source (what an eval reads);
* ``resume`` — the trainer's ``--resume`` config: every non-runtime key and every non-``ckpt`` source;
* ``init_ckpt`` — the PPO ``--init-ckpt`` config's arch keys (GRPO has no ``--init-ckpt``).

``expected`` is :func:`resolved` written by ``python -m tests.helpers.ckpt_golden <fixture.json>...``
— rerun it when a deliberate change moves a value, and read the diff.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

from tests.helpers.ckpt_payload import ckpt_payload, save_ckpt

BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"


def golden_params(directory: Path) -> list[Any]:
    """One ``pytest.param`` per golden fixture in ``directory``, named by its stem."""
    import pytest

    return [pytest.param(p, id=p.stem) for p in sorted(directory.glob("*.json"))]


def write_golden(path: Path, fx: dict[str, Any]) -> None:
    path.write_text(json.dumps(fx, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")


def jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=repr))


def train_run(algo: str, argv: list[str]) -> Any:
    """The trainer config of ``argv``, resolved by ``algo``'s entry point."""
    from configs.loader.schema import GRPOConfig, PPOConfig
    from methods.rl_agent.training import train_grpo, train_ppo
    from methods.rl_agent.training.args import resolve_train_args

    mod, cfg = (train_grpo, GRPOConfig()) if algo == "grpo" else (train_ppo, PPOConfig())
    return resolve_train_args(mod.build_arg_parser(), cfg, argv)


def _run_golden(run: Any, keep: Any) -> dict[str, Any]:
    return jsonable({"values": {k: v for k, v in sorted(vars(run).items()) if keep(k)},
                     "sources": {k: e.source for k, e in sorted(run.provenance.items())
                                 if keep(k) and e.source not in ("ckpt", "defaults")}})


def resolved(fx: dict[str, Any], ckpt_path: Path) -> dict[str, Any]:
    """``load`` / ``resume`` / ``init_ckpt`` of fixture ``fx`` (module docstring); ``ckpt_path`` is
    where the fixture's checkpoint is written for the trainer to read."""
    from configs.loader.ckpt import CKPT_FIELDS, INVERTED_RENAMES, load_ckpt_config
    from configs.loader.schema import GRPOConfig, PPOConfig
    from methods.rl_agent.training.args import train_policies

    save_ckpt(ckpt_path, fx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values, sources = load_ckpt_config(ckpt_payload(fx), strict=False)
        keep = CKPT_FIELDS | set(INVERTED_RENAMES.values())
        out = {"load": jsonable({"values": {k: v for k, v in sorted(values.items()) if k in keep},
                                 "sources": {k: s for k, s in sorted(sources.items()) if s != "ckpt"}})}
        if fx["algo"] != "bc":
            pol = train_policies(GRPOConfig() if fx["algo"] == "grpo" else PPOConfig())
            run = train_run(fx["algo"], ["--board", BOARD, "--resume", str(ckpt_path)])
            out["resume"] = _run_golden(run, lambda k: pol[k].scope != "runtime")
        if fx["algo"] != "grpo":
            pol = train_policies(PPOConfig())
            run = train_run("ppo", ["--board", BOARD, "--init-ckpt", str(ckpt_path)])
            out["init_ckpt"] = _run_golden(run, lambda k: pol[k].scope == "arch")
    return out


def check(path: Path, tmp_path: Path) -> None:
    """Fixture ``path`` resolves to its ``expected``; its resume names no dropped checkpoint key;
    the checkpoint the resumed run writes reads the same through ``args`` (an older loader's path)
    as through ``config`` — unlike test_config_ckpt.py's round trip, with a real run's legacy values."""
    from configs.loader.ckpt import ckpt_entries, load_ckpt_config

    fx = json.loads(path.read_text())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = resolved(fx, tmp_path / "ckpt.pt")
    assert got == fx["expected"]
    assert not [w for w in caught if "not options of this trainer" in str(w.message)]
    if fx["algo"] == "bc":
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = train_run(fx["algo"], ["--board", BOARD, "--resume", str(tmp_path / "ckpt.pt")])
        entries = ckpt_entries(run.ckpt_config())
        state = ckpt_payload(fx)["policy_state_dict"]
        via_config = load_ckpt_config({"config": entries["config"], "policy_state_dict": state}, strict=True)[0]
        via_args = load_ckpt_config({"args": entries["args"], "policy_state_dict": state}, strict=False)[0]
    assert dict(via_args) == dict(via_config)


def main(paths: list[str]) -> None:
    import tempfile

    for p in map(Path, paths):
        fx = json.loads(p.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            fx["expected"] = resolved(fx, Path(tmp) / "ckpt.pt")
        write_golden(p, fx)


if __name__ == "__main__":
    main(sys.argv[1:])
