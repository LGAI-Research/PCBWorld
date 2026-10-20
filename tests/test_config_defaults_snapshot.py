"""No resolved default changes by accident.

The snapshot (``tests/fixtures/config_defaults_snapshot.json``) holds the defaults of every
schema dataclass (``LLMTrainConfig`` also as ``from_verl_config`` builds it from an empty config) and, per key,
the value and source of every program's resolved config: RL train ppo / grpo fresh, ppo
init-ckpt (GRPO has no ``--init-ckpt``), the async-val / queue watcher's eval-mode resolution of
a checkpoint, LLM eval — and the RL eval pipeline's parser defaults. A key the snapshot lacks (a
new knob) passes; a changed or removed one fails. After a deliberate change, regenerate with
``python -m tests.test_config_defaults_snapshot --accept KEY ...``: it prints the diff and writes
only when every changed or removed key is named (``KEY`` = ``n_heads`` for every entry, or
``RLPolicyConfig.n_heads`` for one).
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest import mock

SNAPSHOT = Path(__file__).parent / "fixtures" / "config_defaults_snapshot.json"
_REPO = str(Path(__file__).resolve().parent.parent)


def _program(run: Any) -> dict[str, Any]:
    """``{key: {value, source}}`` of a resolved :class:`~configs.loader.resolve.RunConfig`."""
    return {k: {"value": v, "source": run.provenance[k].source} for k, v in vars(run).items()}


def _pipeline_parser_defaults() -> dict[str, Any]:
    from eval import pipeline

    with mock.patch.object(sys, "argv", ["pipeline", "--skip-rollout", "--output-dir", "out"]):
        parser, _ = pipeline._parse_args()
    return {a.dest: parser.get_default(a.dest) for a in parser._actions if a.dest != "help"}


def defaults_snapshot() -> dict[str, Any]:
    """``{entry: {key: default}}`` over the schema dataclasses and every program's resolved
    config given no flag; this checkout's root is spelled ``<repo>``."""
    from configs.loader import schema
    from eval.args import resolve_eval_args
    from methods.rl_agent.training import train_grpo, train_ppo
    from methods.rl_agent.training.args import resolve_ckpt_train_config, resolve_train_args
    from tests.helpers.ckpt_payload import save_ckpt

    out: dict[str, Any] = {
        name: asdict(getattr(schema, name)())
        for name in ("EnvConfig", "RLEnvConfig", "RLPolicyConfig", "RLTrainConfig", "PPOConfig",
                     "GRPOConfig", "RLEvalConfig", "LLMConfig", "LLMTrainConfig")
    }
    out["LLMTrainConfig/from_verl_config"] = asdict(
        schema.LLMTrainConfig.from_verl_config({}))   # verl config without env.pcbworld
    out["FINETUNE_DEFAULTS"] = dict(schema.FINETUNE_DEFAULTS)
    board = ["--board", "board.kicad_pcb"]   # the one required training option
    for algo, module, config in (("ppo", train_ppo, schema.PPOConfig()),
                                 ("grpo", train_grpo, schema.GRPOConfig())):
        out[f"program/train_{algo}"] = _program(
            resolve_train_args(module.build_arg_parser(), config, board))
    out["program/train_ppo_init_ckpt"] = _program(resolve_train_args(
        train_ppo.build_arg_parser(), schema.PPOConfig(), [*board, "--init-ckpt", "init.pt"],
        load_ckpt=False))
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the fixture checkpoint's legacy-fill notes
        ckpt = save_ckpt(Path(tmp) / "policy_iter_1.pt")
        out["program/async_val_watcher"] = _program(resolve_ckpt_train_config(ckpt, strict=False))
    out["program/eval_rl_pipeline"] = _pipeline_parser_defaults()
    out["program/eval_llm"] = _program(resolve_eval_args([]))
    return json.loads(json.dumps(out, sort_keys=True, default=str).replace(_REPO, "<repo>"))


def changes(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[list[str], list[str]]:
    """``(changed or removed, added)`` as ``entry.key: old -> new`` lines."""
    changed, added = [], []
    for name in sorted(set(expected) | set(actual)):
        old, new = expected.get(name, {}), actual.get(name, {})
        for key in sorted(set(old) | set(new)):
            if key not in old:
                added.append(f"{name}.{key}: + {new[key]!r}")
            elif old[key] != new.get(key, "<removed>"):
                changed.append(f"{name}.{key}: {old[key]!r} -> {new.get(key, '<removed>')!r}")
    return changed, added


def test_resolved_defaults_match_snapshot():
    changed, _ = changes(json.loads(SNAPSHOT.read_text()), defaults_snapshot())
    assert not changed, ("resolved defaults changed (if deliberate: `python -m "
                         "tests.test_config_defaults_snapshot --accept KEY ...`):\n" + "\n".join(changed))


def test_defaults_section_in_two_files_raises(monkeypatch):
    import pytest

    from configs.loader import schema

    monkeypatch.setattr(schema, "_DEFAULTS_FILES", ("env.yaml", "rl.yaml", "env.yaml"))
    with pytest.raises(ValueError, match=r"configs/env.yaml: sections \['env'\]"):
        schema._load_defaults()

def main(argv: list[str] | None = None) -> int:
    from configs.loader.cli import StrictArgumentParser

    p = StrictArgumentParser(description="Regenerate the resolved-defaults snapshot.")
    p.add_argument("--accept", nargs="+", default=[], metavar="KEY",
                   help="changed or removed keys to accept: `key` (every entry) or `entry.key`")
    accept = set(p.parse_args(argv).accept)
    actual = defaults_snapshot()
    changed, added = changes(json.loads(SNAPSHOT.read_text()), actual)
    print("\n".join(changed + added) or "no change")
    qualified = [c.split(": ", 1)[0] for c in changed]
    unaccepted = [q for q in qualified if not {q, q.rsplit(".", 1)[1]} & accept]
    if unaccepted:
        print(f"refusing to write: changed/removed keys not named by --accept: {unaccepted}")
        return 1
    SNAPSHOT.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
    print(f"wrote {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
