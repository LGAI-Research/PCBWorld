"""Shared argparse builders for pcbworld LLM-env eval scripts.

Centralises CLI argument definitions so ``methods/llm_agent/rollout/pcbworld.py`` (vLLM / API /
fixed rollout) and ``interactive_eval.py`` (REPL) stay in sync on common flags.
Each helper attaches a named argument group to an existing parser so individual
entry points can compose only the pieces they need.

The flag list is *derived from the schema* (:class:`configs.loader.schema.LLMConfig`,
one nested sub-config per group) via :func:`configs.loader.cli.add_dataclass_args`, so
the dataclasses are the single visible list of every LLM-eval variable and the
CLI cannot drift from them. This mirrors the RL side (methods/rl_agent/training/args.py); the
only difference is ``style="underscore"`` (snake_case flags) to preserve the
``args.<name>`` surface existing shell scripts depend on.

The ``env`` group is the shared :class:`~configs.loader.schema.EnvConfig` (the
same env RL builds, LLM defaults from ``llm.env``) plus the LLM-only
:class:`~configs.loader.schema.LLMEnvConfig`; :func:`llm_env_config` reads the
parsed values back into one ``EnvConfig`` — the batch rollout and the REPL build
their env from it.

A small bespoke tail per group keeps the flags whose *form* is custom — the
required runtime input ``--board_path``, the inverted ``--no_drc_tokens``, the
degrees ``--corner_mode`` (45 / 90), the repo-abs-resolved ``--boards_json``, and the
dynamic-choices ``--prompt_version`` — but every such variable is still a schema
field (``cli_skip``), visible in ``configs/loader/schema.py``.

Grouping (see :func:`build_eval_parser` for the full-compose shortcut):
    add_env_args            board + env-build knobs (board_path, max_steps, ...)
    add_boards_args         multi-board selection (boards_order, ...)
    add_prompt_args         --prompt_version (PCBWorldPromptBundle selector)
    add_rollout_args        common rollout knobs (env_num, seed, ...)
    add_llm_args            vLLM-specific (model_path, tensor_parallel_size, ...)
    add_api_args            provider-API-specific (api_provider, api_model, ...)
    add_verbose_arg         --verbose / --silent (output verbosity, not config)
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Iterable, Optional

from configs.loader.cli import StrictArgumentParser, add_dataclass_args
from configs.loader.knob import Policy, runtime
from configs.loader.resolve import RunConfig, add_run_args
from configs.loader.schema import _ENV_FLAT_KEYS, INVERTED_DESTS, EnvConfig, LLMConfig
from methods.llm_agent.wrappers.prompts import PCBWORLD_PROMPT_VERSIONS

# LLM-eval defaults (single source: configs/llm.yaml (llm:), via the nested
# LLMConfig). Used as the argparse defaults below; consumers still read args.<name>.
_L = LLMConfig()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
# eval/ -> <repo-root>/
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
DEFAULT_BOARD = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "simple_routing_board.kicad_pcb",
)


# ---------------------------------------------------------------------------
# Argument groups
# ---------------------------------------------------------------------------

def add_env_args(
    parser: argparse.ArgumentParser,
    *,
    default_max_steps: int = _L.env.max_steps,
    default_reward_rule: str = _L.env.reward_rule,
    default_board: Optional[str] = DEFAULT_BOARD,
) -> argparse._ArgumentGroup:
    """Attach PCBWorld build knobs (shared between batch eval + REPL).

    ``default_board=None`` allows callers (e.g. the interactive REPL) to start in
    a blank state with no board selected until the user issues ``:load <path>``.
    The ``default_max_steps`` / ``default_reward_rule`` overrides flow through
    :func:`dataclasses.replace` so the generated flags keep the caller's defaults.
    """
    g = parser.add_argument_group("env")
    # --board_path: required runtime input, bespoke (computed / caller-overridden
    # default), so it has no schema field.
    g.add_argument(
        "--board_path",
        default=default_board,
        help=(
            "Path to .kicad_pcb"
            + (" (default: tests fixture)." if default_board else
               " (optional — REPL starts blank and loads via :load).")
        ),
    )
    env_cfg = dataclasses.replace(
        _L.env, max_steps=default_max_steps, reward_rule=default_reward_rule,
    )
    add_dataclass_args(g, env_cfg, style="underscore")
    add_dataclass_args(g, _L.llm_env, style="underscore")
    # cli_skip EnvConfig fields with a bespoke flag form.
    g.add_argument("--corner_mode", type=int, default=env_cfg.corner_mode, choices=[45, 90],
                   help="Routing corner angle in degrees: 45 (default, diagonals OK) or 90 (no diagonals).")
    g.add_argument("--no_drc_tokens", action="store_true",
                   help="Disable DRC tokens in observation (default: emit).")
    return g


# The flat EnvConfig keys the ``env`` group registers as dests — what
# :func:`llm_env_config` reads back.
ENV_ARG_KEYS = _ENV_FLAT_KEYS & {a.dest for a in add_env_args(StrictArgumentParser())._group_actions}


def llm_env_config(args: argparse.Namespace) -> EnvConfig:
    """The :class:`EnvConfig` of an LLM run, read back from the ``env`` group's
    parsed flags (every flat EnvConfig key the group registers, plus the inverted
    ``--no_drc_tokens``). ``args`` from a parser without the group raises."""
    return _L.env.with_flat({k: getattr(args, k) for k in ENV_ARG_KEYS}
                            | {"emit_drc_tokens": not args.no_drc_tokens})


def add_boards_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Attach multi-board selection flags (mirrors the RL trainer's board flags).

    ``--boards_order=single`` (default) preserves legacy behaviour — only
    ``--board_path`` is consulted. ``round_robin`` consults ``--boards_json`` to
    build a list of boards that eval iterates over (one reload per board).
    """
    g = parser.add_argument_group("boards")
    add_dataclass_args(g, _L.boards, style="underscore")
    # boards_json is cli_skip and has no default — multi-board mode must set
    # it explicitly.
    g.add_argument(
        "--boards_json",
        default=None,
        help="JSON split file; required when --boards_order != single "
             "(see the configs/datasets/ registry).",
    )
    # Per-split dataset directories now come from the JSON's top-level
    # ``dataset_dirs[split]`` (see configs/datasets/*.json). No CLI fallback.
    return g


def add_prompt_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Attach ``--prompt_version`` (selects a PCBWorldPromptBundle).

    Bespoke because its ``choices`` are dynamic (``PCBWORLD_PROMPT_VERSIONS``);
    the variable is the cli_skip ``LLMPromptConfig.prompt_version`` field.
    """
    g = parser.add_argument_group("prompt")
    g.add_argument("--prompt_version", default=_L.prompt.prompt_version,
                   choices=list(PCBWORLD_PROMPT_VERSIONS),
                   help="PCBWorld prompt version: latest (edited in place) or a "
                        "frozen legacy v1..v7. Its action set must match "
                        "--masking_rule (checked when the env is built).")
    return g


def add_rollout_args(
    parser: argparse.ArgumentParser,
    *,
    default_env_num: int = _L.rollout.env_num,
    default_rollout_episodes: int = _L.rollout.rollout_episodes,
) -> argparse._ArgumentGroup:
    """Attach rollout orchestration knobs (batch-eval only)."""
    g = parser.add_argument_group("rollout")
    roll_cfg = dataclasses.replace(
        _L.rollout,
        env_num=default_env_num,
        rollout_episodes=default_rollout_episodes,
    )
    add_dataclass_args(g, roll_cfg, style="underscore")
    return g


def add_llm_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Attach vLLM-mode knobs."""
    g = parser.add_argument_group("llm")
    add_dataclass_args(g, _L.vllm, style="underscore")
    return g


def add_api_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Attach provider-API-mode knobs."""
    g = parser.add_argument_group("api")
    add_dataclass_args(g, _L.api, style="underscore")
    return g


def add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true",
                        help="Print full prompts every step.")
    parser.add_argument("--silent", action="store_true",
                        help="Suppress per-step long content (system/user "
                             "prompts AND model response bodies). The one-line "
                             "[step N][env i] STATUS (...tokens) tracker plus "
                             "board / episode / rollout summaries still print, "
                             "so progress is visible without dumping prompts.")


# ---------------------------------------------------------------------------
# Pre-composed parser
# ---------------------------------------------------------------------------

def build_eval_parser(
    *,
    description: str = "pcbworld environment eval",
    modes: Optional[Iterable[str]] = ("fixed", "llm", "api"),
) -> argparse.ArgumentParser:
    """Return a parser wired with all groups used by ``methods/llm_agent/rollout/pcbworld.py``."""
    parser = StrictArgumentParser(description=description,
                                  removed=("--use_yaml_drc_fallback", "--no-use_yaml_drc_fallback"))
    if modes is not None:
        parser.add_argument(
            "--mode", default="fixed", choices=list(modes),
            help="fixed: one-step smoke test | llm: local vLLM rollout | "
                 "api: API-based rollout (default: fixed).",
        )
    add_env_args(parser)
    add_boards_args(parser)
    add_prompt_args(parser)
    add_rollout_args(parser)
    add_llm_args(parser)
    add_api_args(parser)
    add_verbose_arg(parser)
    add_run_args(parser)
    # Legacy aliases (accepted but ignored)
    parser.add_argument("--num_gpus", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu_devices", type=str, default=None, help=argparse.SUPPRESS)
    return parser


#: Options of :func:`build_eval_parser` that are not schema fields, with their policy.
OPTION_POLICIES = {
    "board_path": Policy("experiment", False, False, "--board_path"),
    "mode": Policy("experiment", False, False, "--mode"),
    **{dest: runtime(f"--{dest}") for dest in ("verbose", "silent", "num_gpus", "gpu_devices")},
}


def resolve_eval_args(argv: Optional[list[str]] = None, **parser_kw) -> RunConfig:
    """Parse an LLM eval command line (:func:`build_eval_parser`) and resolve it once (mode
    ``fresh``: no checkpoint) into the run's frozen config."""
    from configs.loader.resolve import dest_policies, resolve_parsed

    parser = build_eval_parser(**parser_kw)
    args = parser.parse_args(argv)
    return resolve_parsed(parser, args, "fresh", policies=dest_policies(
        LLMConfig(), inverted=INVERTED_DESTS, extra=OPTION_POLICIES))


__all__ = [
    "DEFAULT_BOARD",
    "add_env_args",
    "add_boards_args",
    "add_prompt_args",
    "add_rollout_args",
    "add_llm_args",
    "add_api_args",
    "add_verbose_arg",
    "build_eval_parser",
    "llm_env_config",
    "resolve_eval_args",
]
