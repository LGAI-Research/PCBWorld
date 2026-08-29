"""Shared argparse builders for cadagent LLM-env eval scripts.

Centralises CLI argument definitions so ``methods/llm_agent/rollout/cadagent.py`` (vLLM / API /
fixed rollout) and ``interactive_eval.py`` (REPL) stay in sync on common flags.
Each helper attaches a named argument group to an existing parser so individual
entry points can compose only the pieces they need.

The flag list is *derived from the schema* (:class:`configs.loader.schema.LLMConfig`,
one nested sub-config per group) via :func:`configs.loader.cli.add_dataclass_args`, so
the dataclasses are the single visible list of every LLM-eval variable and the
CLI cannot drift from them. This mirrors the RL side (methods/rl_agent/training/args.py); the
only difference is ``style="underscore"`` (snake_case flags), which keeps the
``args.<name>`` surface the shell scripts depend on.

A small bespoke tail per group keeps the flags whose *form* is custom — the
required runtime input ``--board_path``, the inverted ``--no_drc_tokens``, the
repo-abs-resolved ``--boards_json``, and the dynamic-choices ``--prompt_version``
— but every such variable is still a schema field (``cli_skip``), visible in
``configs/loader/schema.py``.

Grouping (see :func:`build_eval_parser` for the full-compose shortcut):
    add_env_args            board + env-build knobs (board_path, max_steps, ...)
    add_boards_args         multi-board selection (boards_order, ...)
    add_prompt_args         --prompt_version (CadagentPromptBundle selector)
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

from configs.loader.cli import add_dataclass_args
from configs.loader.schema import LLMConfig
from methods.llm_agent.wrappers.prompts import CADAGENT_PROMPT_VERSIONS

# LLM-eval defaults (single source: configs/defaults/llm.yaml, via the nested
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
    # emit_drc_tokens is cli_skip; its inverted form (default: emit) is here.
    g.add_argument("--no_drc_tokens", action="store_true",
                   help="Disable DRC tokens in observation (default: emit).")
    return g


def add_boards_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Attach multi-board selection flags (mirrors the RL trainer's board flags).

    ``--boards_order=single`` (default) consults only ``--board_path``.
    ``round_robin`` consults ``--boards_json`` to build a list of boards that
    eval iterates over (one reload per board).
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
    # Per-split dataset directories come from the JSON's top-level
    # ``dataset_dirs[split]`` (see configs/datasets/*.json). No CLI fallback.
    return g


def add_prompt_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Attach ``--prompt_version`` (selects a CadagentPromptBundle).

    Bespoke because its ``choices`` are dynamic (``CADAGENT_PROMPT_VERSIONS``);
    the variable is the cli_skip ``LLMPromptConfig.prompt_version`` field.
    """
    g = parser.add_argument_group("prompt")
    g.add_argument("--prompt_version", default=_L.prompt.prompt_version,
                   choices=list(CADAGENT_PROMPT_VERSIONS),
                   help="Cadagent prompt template version (default: v1).")
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
    description: str = "cadagent environment eval",
    modes: Optional[Iterable[str]] = ("fixed", "llm", "api"),
) -> argparse.ArgumentParser:
    """Return a parser wired with all groups used by ``methods/llm_agent/rollout/cadagent.py``."""
    parser = argparse.ArgumentParser(description=description)
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
    # Accepted and ignored; hidden from --help.
    parser.add_argument("--num_gpus", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu_devices", type=str, default=None, help=argparse.SUPPRESS)
    return parser


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
]
