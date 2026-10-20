"""Every argparse parser is a ``StrictArgumentParser`` (``configs/loader/cli.py``): exact option names
only, each dest at most once, and the dests the command line gave recorded on the parser.

With argparse's defaults, ``--clip 0.3`` silently means ``--clip-eps 0.3`` and ``--lr 1e-4 ... --lr 3e-4``
silently keeps the last value — a launcher whose fixed flags and pass-through flags overlap trains with
something other than what either line says. ``engine/`` is a standalone bundle that does not import
``configs``: its parsers only carry ``allow_abbrev=False``.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from configs.loader.cli import StrictArgumentParser, add_dataclass_args
from tests.helpers.tracked_files import REPO, tracked_py


def _names_argument_parser(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "ArgumentParser"
            or isinstance(node, ast.Name) and node.id == "ArgumentParser")


def _parser_offenders(path: Path, *, engine: bool) -> list[int]:
    """Lines building a parser the wrong way: anywhere, an ``allow_abbrev=`` other than the constant
    ``False``; outside ``engine/`` a plain ``ArgumentParser`` (also as ``add_subparsers(parser_class=...)``);
    inside it, a parser without ``allow_abbrev=False``."""
    bad = []
    for node in ast.walk(ast.parse(path.read_bytes())):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        abbrev = next((k.value for k in node.keywords if k.arg == "allow_abbrev"), None)
        if abbrev is not None and not (isinstance(abbrev, ast.Constant) and abbrev.value is False):
            bad.append(node.lineno)
        elif not engine:
            if _names_argument_parser(func) or any(
                    k.arg == "parser_class" and _names_argument_parser(k.value) for k in node.keywords):
                bad.append(node.lineno)
        elif name in ("ArgumentParser", "add_parser") and abbrev is None:
            bad.append(node.lineno)
    return bad


def test_every_parser_is_strict():
    # The class itself forwards its own allow_abbrev to argparse.
    offenders = [f"{p.relative_to(REPO)}:{line}" for p in tracked_py() if p != REPO / "configs/loader/cli.py"
                 for line in _parser_offenders(p, engine=p.relative_to(REPO).parts[0] == "engine")]
    assert not offenders, (
        "argparse parser that is not configs.loader.cli.StrictArgumentParser (engine/: without "
        "allow_abbrev=False), or an allow_abbrev= other than False — shortened or repeated flags "
        f"would be silently accepted: {offenders}")


def test_scan_sees_plain_and_engine_parsers(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import argparse\nfrom argparse import ArgumentParser\n"
                   "p = argparse.ArgumentParser()\nq = ArgumentParser()\n"
                   "s = StrictArgumentParser().add_subparsers(parser_class=argparse.ArgumentParser)\n"
                   "ok = StrictArgumentParser().add_subparsers().add_parser('x')\n"
                   "t = StrictArgumentParser(allow_abbrev=True)\nu = make(allow_abbrev=flag)\n"
                   "v = StrictArgumentParser(allow_abbrev=False)\n")
    assert _parser_offenders(bad, engine=False) == [3, 4, 5, 7, 8]
    eng = tmp_path / "eng.py"
    eng.write_text("import argparse\np = argparse.ArgumentParser(allow_abbrev=False)\n"
                   "s = p.add_subparsers().add_parser('x', allow_abbrev=True)\n"
                   "q = argparse.ArgumentParser(allow_abbrev=0)\nr = argparse.ArgumentParser()\n")
    assert _parser_offenders(eng, engine=True) == [3, 4, 5]


# --------------------------------------------------------------------------- the parser class
@dataclasses.dataclass
class _BoolConfig:
    bootstrap: bool = True        # add_dataclass_args: the --bootstrap / --no-bootstrap pair, one dest


def _parser() -> StrictArgumentParser:
    p = StrictArgumentParser(prog="t")
    p.add_argument("-n", "--n-envs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--extra", action="extend", nargs="+", default=[])
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--const", action="append_const", const=1, dest="consts")
    p.add_argument("rest", nargs="*")
    add_dataclass_args(p, _BoolConfig())
    return p


def _error(parser: StrictArgumentParser, argv: list[str], capsys) -> str:
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("argv,needles", [
    (["--lr", "1e-4", "--lr", "3e-4"], ["argument --lr: given more than once"]),
    (["--lr=1e-4", "--lr", "3e-4"], ["argument --lr: given more than once"]),
    (["-n", "8", "--n-envs", "16"], ["argument -n/--n-envs: given more than once"]),
    (["--bootstrap", "--no-bootstrap"], ["argument --no-bootstrap: conflicts with --bootstrap", "'bootstrap'"]),
    (["--no-bootstrap", "--bootstrap"], ["argument --bootstrap: conflicts with --no-bootstrap"]),
])
def test_one_dest_given_twice_is_an_error_naming_both(argv, needles, capsys):
    err = _error(_parser(), argv, capsys)
    for needle in needles:
        assert needle in err


def test_accumulating_actions_repeat_freely():
    p = _parser()
    a = p.parse_args(["--tag", "a", "--tag", "b", "--extra", "x", "--extra", "y", "z",
                      "-vv", "--verbose", "--const", "--const"])
    assert (a.tag, a.extra, a.verbose, a.consts) == (["a", "b"], ["x", "y", "z"], 3, [1, 1])


def test_explicit_dests_are_what_the_command_line_gave():
    p = _parser()
    a = p.parse_args(["--lr=1e-4", "--no-bootstrap", "pos", "--", "--lr", "2"])
    assert a.lr == 1e-4 and a.bootstrap is False and a.rest == ["pos", "--lr", "2"]
    assert p.explicit_dests == {"lr", "bootstrap"}   # after `--` is positional; defaults are not given
    p.parse_args(["-n", "2"])                          # each parse starts over
    assert p.explicit_dests == {"n_envs"}


def test_intermixed_parse_records_the_whole_command_line(capsys):
    # parse_intermixed_args runs two internal passes (on Python 3.9 both through parse_known_args):
    # the record must still hold every option given, and a repeat must still be caught.
    p = StrictArgumentParser(prog="t")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("files", nargs="*")
    a = p.parse_intermixed_args(["a", "--lr", "1e-4", "b", "--seed", "3", "c"])
    assert (a.files, a.lr, a.seed) == (["a", "b", "c"], 1e-4, 3)
    assert p.explicit_dests == {"lr", "seed"}
    with pytest.raises(SystemExit):
        p.parse_intermixed_args(["--lr", "1", "a", "--lr", "2"])
    assert "argument --lr: given more than once" in capsys.readouterr().err


def test_explicit_dests_before_a_parse_is_an_error():
    with pytest.raises(RuntimeError, match="before the parser parsed"):
        _parser().explicit_dests


def test_parse_known_args_checks_known_options_and_passes_the_rest(capsys):
    p = _parser()
    ns, rest = p.parse_known_args(["--lr", "1e-4", "--unknown=1"])
    assert ns.lr == 1e-4 and rest == ["--unknown=1"] and p.explicit_dests == {"lr"}
    with pytest.raises(SystemExit):
        p.parse_known_args(["--lr", "1e-4", "--unknown", "--lr", "2e-4"])


def test_subcommands_are_strict_and_record_on_their_own_parser(capsys):
    top = StrictArgumentParser(prog="t")
    top.add_argument("--dry-run", action="store_true")
    sub = top.add_subparsers(dest="cmd", required=True).add_parser("run")
    sub.add_argument("--seed", type=int, default=0)
    assert isinstance(sub, StrictArgumentParser)
    a = top.parse_args(["--dry-run", "run", "--seed", "3"])
    assert (a.cmd, a.seed) == ("run", 3)
    assert top.explicit_dests == {"dry_run"} and sub.explicit_dests == {"seed"}
    assert "argument --seed: given more than once" in _error(top, ["run", "--seed", "1", "--seed", "2"], capsys)
    assert "unrecognized arguments: --se" in _error(top, ["run", "--se", "1"], capsys)


# --------------------------------------------------------------------------- the real entry points
def test_training_parser_rejects_a_shortened_or_repeated_flag():
    from methods.rl_agent.training import train_ppo

    parser = train_ppo.build_arg_parser()
    assert parser.parse_args(["--board", "b", "--clip-eps", "0.3"]).clip_eps == 0.3
    for argv in (["--clip", "0.3"], ["--lr", "1e-4", "--lr", "3e-4"],
                 ["--truncation-bootstrap", "--no-truncation-bootstrap"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["--board", "b", *argv])


def test_eval_parser_rejects_a_shortened_or_repeated_flag():
    from eval.args import build_eval_parser

    parser = build_eval_parser()
    assert parser.parse_args(["--prompt_version", "v5"]).prompt_version == "v5"
    for argv in (["--prompt", "v5"], ["--prompt_version", "v5", "--prompt_version", "v6"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_a_removed_option_names_its_replacement(capsys):
    # A v1.0.1 launch line with an option this version dropped fails with what replaces it,
    # not argparse's bare "unrecognized arguments".
    from methods.rl_agent.training import train_grpo, train_ppo

    for build in (train_ppo.build_arg_parser, train_grpo.build_arg_parser):
        out = _error(build(), ["--board", "b", "--use-yaml-drc-fallback", "--engine-seed=3"], capsys)
        assert "unrecognized arguments: --use-yaml-drc-fallback --engine-seed=3" in out
        assert "--engine-seed: no flag any more: set env." + "engine_seed in configs/env.yaml" in out
        assert "--use-yaml-drc-fallback: use --rules-source board_or_yaml" in out
    assert "--finetune-lr: drop it" in _error(train_ppo.build_arg_parser(),
                                              ["--board", "b", "--finetune-lr", "1e-5"], capsys)
    # GRPO has no --init-ckpt, so the fine-tune hint would point at nothing
    out = _error(train_grpo.build_arg_parser(), ["--board", "b", "--finetune-lr", "1e-5"], capsys)
    assert out.strip().endswith("unrecognized arguments: --finetune-lr 1e-5")


def test_a_hint_is_given_only_by_a_parser_that_had_the_option(capsys):
    from eval.args import build_eval_parser

    out = _error(build_eval_parser(), ["--no-use_yaml_drc_fallback"], capsys)
    assert "--no-use_yaml_drc_fallback: use --rules_source board" in out
    out = _error(build_eval_parser(), ["--use-yaml-drc-fallback"], capsys)  # the dash spelling: never its option
    assert out.strip().endswith("unrecognized arguments: --use-yaml-drc-fallback")
    out = _error(StrictArgumentParser(), ["--engine-seed", "3"], capsys)
    assert out.strip().endswith("unrecognized arguments: --engine-seed 3")
    with pytest.raises(KeyError, match="not in REMOVED_OPTIONS"):
        StrictArgumentParser(removed=("--never-was",))


def test_the_d3_builder_names_the_dropped_glob(monkeypatch, capsys):
    import sys

    from experiments.kdd.d3_dataset import build

    monkeypatch.setattr(sys, "argv", ["build", "--csv", "c", "--sorted-dir", "s", "--out", "o",
                                      "--easy-train-only-glob", "0096_*"])
    with pytest.raises(SystemExit):
        build.parse_args()
    assert "--easy-train-only-glob: drop it" in capsys.readouterr().err


def test_the_engine_knob_hints_name_keys_of_the_env_defaults():
    import yaml

    from configs.loader.cli import REMOVED_OPTIONS

    env = yaml.safe_load((REPO / "configs" / "env.yaml").read_text())["env"]
    named = [text.split("env.", 1)[1].split()[0] for text in REMOVED_OPTIONS.values() if "set env." in text]
    assert named and all(key in env for key in named)


def test_a_subcommand_that_had_the_option_gives_the_hint(capsys):
    top = StrictArgumentParser()
    subs = top.add_subparsers(dest="cmd")
    subs.add_parser("run", removed=("--use-yaml-drc-fallback",))
    out = _error(top, ["run", "--use-yaml-drc-fallback"], capsys)
    assert "--use-yaml-drc-fallback: use --rules-source board_or_yaml" in out


def test_a_parser_without_exit_on_error_raises_the_hinted_message():
    import argparse

    parser = StrictArgumentParser(exit_on_error=False, removed=("--engine-seed",))
    with pytest.raises(argparse.ArgumentError, match="--engine-seed: no flag any more"):
        parser.parse_args(["--engine-seed", "3"])
