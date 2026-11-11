"""PCBWorld ``latest`` prompt — generated from ``glossary.yaml`` and the run's EnvConfig.

Every sentence comes from the reviewed glossary; :func:`build` picks the entries the env's
rules call for (masking rule, ``net_reselect``, ``action_target``, the constraint knobs, the
reward rule's coefficients, the obs knobs) and fills their placeholders. A config value the
glossary has no sentence for raises, so a new rule cannot reach the agent undescribed.
:data:`ENV_KNOBS` says, for every EnvConfig field, how the agent learns of it
(tests/test_llm_env_knobs.py keeps it complete). The observation it describes is the one the
RL tokenizer reads (state_converter.OBS_ELEMENTS, tests/test_llm_obs_parity.py).
When results from a ``latest`` run are cited, freeze the generated text as legacy/v<N+1>.py;
the run record carries the prompt content hash.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

GLOSSARY_PATH = Path(__file__).with_name("glossary.yaml")

#: How the agent learns of each EnvConfig field: ``prompt`` (a glossary sentence the
#: generator picks or fills from it), ``obs`` (it shapes the observation or the candidate
#: list), ``engine`` (router / DRC internals the agent meets only through outcomes) or
#: ``rl_only`` (RL transport). The second item says which entry or why.
ENV_KNOBS: Dict[str, Tuple[str, str]] = {
    "max_steps": ("prompt", "terms.step_budget"),
    "masking_rule": ("prompt", "action_lines (the actions it grants; net_end variant)"),
    "net_reselect": ("prompt", "action_lines.net_end.variants"),
    "reward_rule": ("prompt", "objective.score + drc.severity"),
    "reward": ("prompt", "objective.score (per-run reward overrides)"),
    "drc_promoted_codes": ("prompt", "drc.severity (which warnings count)"),
    "keep_routing_fraction": ("prompt", "terms.kept_routing"),
    "force_walkaround": ("prompt", "modes.forced_walkaround"),
    "mask_start_point": ("prompt", "rules.free.same_point (candidates: the list omits it)"),
    "offboard_mask": ("prompt", "rules.free.off_board (candidates: the list omits them)"),
    "action_target": ("prompt", "rules.<action_target> + observation candidates"),
    "net_constraint_obs": ("prompt", "design_rules + the per-net (constraints ...)"),
    "emit_drc_tokens": ("prompt", "drc + (drc_violations ...)"),
    "outline_obs": ("obs", "the (boardlines ...) entries (edge / arc)"),
    "simplify_outline": ("obs", "the (boardlines ...) entries"),
    "action_history_len": ("obs", "RL action-history tokens; the LLM History block comes from "
                                  "the rollout memory (history_length)"),
    "directional_candidates": ("obs", "the (candidates ...) points (candidates mode)"),
    "connectivity_filter": ("obs", "the (candidates ...) points (candidates mode)"),
    "pad_graze_margin_mm": ("obs", "the (candidates ...) points (candidates mode)"),
    "corner_mode": ("engine", "the router's corner angle; the drawn tracks show it"),
    "reject_if_stuck": ("engine", "make_line outcome on a blocked path (v7 action text kept)"),
    "trim_at_contact": ("engine", "commit-time copper cleanup"),
    "shove_iter_limit": ("engine", "router iteration bound"),
    "followbranch_iter_limit": ("engine", "router iteration bound"),
    "engine_seed": ("engine", "router determinism"),
    "drc_incremental": ("engine", "DRC speed; equal results"),
    "rules_source": ("engine", "where the board's design rules come from"),
    "drc_config_path": ("engine", "where the board's design rules come from"),
    "reward_noise_std": ("rl_only", "training reward noise"),
    "reward_noise_clip_sigma": ("rl_only", "training reward noise"),
    "obs_format": ("rl_only", "RL obs transport (LLM obs are json)"),
}

_PROMOTED_CODES = {"errors_and_promoted": [12, 13, 37]}   # the codes drc.severity names
# reward-rule potentials a score line folds in (no line of their own) and the value they must hold
_FOLDED = {"net_bonus_size_linear_scale": None, "wirelength_bbox_normalize": True,
           "wire_via_emission": None, "drc_shape": "log_per_net", "drc_log_scale": None,
           "drc_log_agg_scale": None, "drc_log_offset": None, "drc_severity_mode": None,
           "truncation_mode": "none"}
# PotentialReward arguments no score line names, each with the condition under which it has
# no effect; outside it (and any other unnamed argument whenever it is truthy) it stops the build
_INERT = {"completion_bonus": lambda p: p["completion_bonus_padgroup_scale"] is not None,
          "clean_completion_bonus": lambda p: p["clean_completion_bonus_padgroup_scale"] is not None,
          "drc_penalty": lambda p: p["drc_shape"] != "linear",
          **dict.fromkeys(("drc_aggregate_scale", "drc_per_net_scale", "drc_saturation_offset"),
                          lambda p: p["drc_shape"] != "saturating")}
_SCORED = ("unconnected_penalty", "net_completion_bonus", "net_clean_bonus",
           "completion_bonus_padgroup_scale", "clean_completion_bonus_padgroup_scale",
           "wirelength_penalty", "via_penalty", "step_penalty")
_ENV_PENALTIES = ("invalid_action_penalty", "mask_reject_penalty", "parse_fail_penalty")


@lru_cache(maxsize=None)
def glossary() -> Dict[str, Any]:
    return yaml.safe_load(GLOSSARY_PATH.read_text(encoding="utf-8"))


def _text(entry: Any) -> str:
    return entry["text"] if isinstance(entry, dict) else entry


def _bullets(lines: List[str]) -> str:
    return "\n".join(f"- {ln}" for ln in lines)


@dataclass(frozen=True)
class LatestPrompt:
    """The ``latest`` prompt built for one EnvConfig."""

    actions: frozenset
    state_format_desc: str
    system: str
    user_no_his: str
    user: str
    template_no_his: str
    template: str
    action_lines: Dict[str, str]
    mode_lines: Tuple[str, ...]

    def format_valid_actions(self, mask: List[bool]) -> str:
        """The valid actions of ``mask`` (ACTION_NAMES order; idle never shown)."""
        from methods.llm_agent.wrappers.state_converter import format_valid_actions

        return format_valid_actions(mask, self.action_lines, self.mode_lines)


def _net_end_variant(rule, net_reselect: bool) -> str:
    when = rule.conditions["net_end"]
    if when.get("net_fully_connected") is True:
        return "after_complete"
    if "net_fully_connected" not in when:
        return "anytime_reselect" if net_reselect else "anytime"
    raise ValueError(f"masking rule {rule.name!r}: no glossary net_end sentence for the "
                     f"condition {dict(when)}")


def action_lines(masking_rule: str, net_reselect: bool) -> Tuple[frozenset, Dict[str, str]]:
    """The routing actions ``masking_rule`` grants and each one's valid-action line; raises
    when the glossary has no sentence for one of them."""
    from pcb_world.core.masking import get_masking_rule

    g = glossary()["action_lines"]
    rule = get_masking_rule(masking_rule)
    actions = frozenset(rule.actions) - {"idle"}
    missing = sorted(actions - set(g))
    if missing:
        raise ValueError(f"masking rule {rule.name!r} grants {missing}, which the glossary "
                         "does not describe (methods/llm_agent/wrappers/prompts/glossary.yaml)")
    lines = {}
    for name in actions:
        e = g[name]
        text = (e["variants"][_net_end_variant(rule, net_reselect)]["text"]
                if name == "net_end" else e["text"])
        lines[name] = f"{text}  e.g. {e['example']}"
    return actions, lines


@lru_cache(maxsize=None)
def _potential_defaults() -> Dict[str, Any]:
    """:class:`PotentialReward` argument defaults — what a reward rule that leaves an
    argument out trains with (as ``reward_config.phi_diff`` reads them)."""
    import inspect

    from pcb_world.core.reward import PotentialReward

    return {k: p.default for k, p in inspect.signature(PotentialReward.__init__).parameters.items()
            if p.default is not inspect.Parameter.empty}


def _score_lines(env, cfg) -> List[str]:
    g = glossary()["objective"]["score"]
    pot = {**_potential_defaults(), **cfg.potential}
    described = {*_FOLDED, *_SCORED, *_ENV_PENALTIES}
    unknown = sorted(k for k, v in pot.items() if k not in described and v
                     and not (k in _INERT and _INERT[k](pot)))
    if unknown:
        raise ValueError(f"reward rule {env.reward_rule!r} sets {unknown} (or leaves them at "
                         "their PotentialReward default), which no glossary score line describes")
    for k, need in _FOLDED.items():
        if k == "wirelength_bbox_normalize" and not pot["wirelength_penalty"]:
            continue
        if need is not None and pot[k] != need:
            raise ValueError(f"reward rule {env.reward_rule!r}: {k}={pot[k]!r} has no glossary "
                             f"score line (described: {need!r})")
    coef = {**pot, "step_penalty": cfg.step_penalty,
            **{k: getattr(cfg, k) for k in _ENV_PENALTIES}}
    out = []
    for key in (*_SCORED, *_ENV_PENALTIES):
        if key == "wirelength_penalty" and (pot.get("drc_log_scale")
                                            or pot.get("drc_log_agg_scale")):
            out.append(g["drc_log_per_net"])   # the DRC line sits before the cost lines
        if not coef.get(key):
            continue
        if key == "net_completion_bonus" and pot.get("net_bonus_size_linear_scale"):
            out.append(g["net_completion_bonus_sized"].format(
                coef=coef[key], size_scale=pot["net_bonus_size_linear_scale"]))
        else:
            out.append(_text(g[key]).format(coef=coef[key]))
    return out


def _severity_line(env, cfg) -> str:
    mode = cfg.potential.get("drc_severity_mode")
    sev = glossary()["drc"]["severity"]
    if mode not in sev or list(env.drc_promoted_codes) != _PROMOTED_CODES.get(mode):
        raise ValueError(f"drc_severity_mode {mode!r} with drc_promoted_codes "
                         f"{list(env.drc_promoted_codes)}: no glossary severity sentence")
    return sev[mode]


def _forced_mode_lines(env) -> Tuple[str, ...]:
    """The forced-mode sentence when every routing action runs in one mode
    (``force_walkaround`` or ``PCBWORLD_FORCE_ROUTING_MODE``); raises for a forced mode
    the glossary has no sentence for."""
    from pcb_world.core.action_schema import MODE_WALKAROUND
    from pcb_world.core.constraints import forced_routing_mode

    forced = forced_routing_mode(env.force_walkaround)
    if forced is None:
        return ()
    if forced != MODE_WALKAROUND:
        raise ValueError(f"forced routing mode {forced} (PCBWORLD_FORCE_ROUTING_MODE): no "
                         "glossary sentence (described: Walkaround)")
    return (glossary()["modes"]["forced_walkaround"]["text"],)


def _state_format_desc(env, cfg) -> str:
    from methods.llm_agent.wrappers.state_converter import _rendered
    from pcb_world.core.env import DRC_OBS_K

    g = glossary()
    obs = g["observation"]
    shown = set(_rendered("latest", env.net_constraint_obs, env.emit_drc_tokens))
    if env.action_target == "candidates":
        shown.add("candidates")

    def lines(section: str) -> List[str]:
        return [f"  {e['text']}" for e in obs[section] if e["element"] in shown]

    header = obs["header"]["variants"]["candidates" if env.action_target == "candidates"
                                         else "rounded"]["text"]
    parts = [header, "",
             "## board_static — static board context (unchanged during episode)",
             *lines("board_static"), "",
             "## routing_geometry — dynamic per-net routing state (updated every step)",
             "  (net CODE", *[f"  {ln}" for ln in lines("routing_geometry")]]
    dynamic = lines("dynamic")
    if dynamic:
        parts += ["", "## dynamic state (updated every step)", *dynamic]
    parts += ["", "## router_head — current router cursor state", *lines("router_head")]
    candidates = lines("candidates")
    if candidates:   # the block the worker appends after the state
        parts += ["", "## candidates — allowed targets (updated every step)", *candidates]
    if env.emit_drc_tokens:
        d = g["drc"]
        parts += ["", "## DRC", d["intro"]["text"].format(k=DRC_OBS_K),
                  _bullets([v for k, v in d["types"].items() if k != "origin"]),
                  _severity_line(env, cfg)]
    return "\n".join(parts)


def _system_blocks(env, cfg, *, guidelines: bool) -> str:
    g = glossary()
    terms = [_text(e).format(max_steps=env.max_steps) for k, e in g["terms"].items()
             if not (isinstance(e, dict) and e.get("when") == "keep_routing_fraction"
                     and env.keep_routing_fraction is None)]
    obj = g["objective"]
    blocks = [obj["role"]["text"],
              "\n".join([obj["priority"]["text"], obj["success"]["text"],
                         obj["score"]["intro"], _bullets(_score_lines(env, cfg))]),
              "# Terms\n" + _bullets(terms)]
    if env.net_constraint_obs:
        d = g["design_rules"]
        blocks.append("# Design rules\n" + d["intro"]["text"] + "\n" + _bullets(
            [d[k]["text"] for k in ("clearance", "track_width", "via_diameter")]))
    rules = g["rules"]
    if env.action_target == "candidates":
        rule_lines = [rules["candidates"]["listed"]["text"]]
    elif env.action_target == "free":
        rule_lines = [e["text"] for e in rules["free"].values() if getattr(env, e["when"])]
    else:
        raise ValueError(f"action_target {env.action_target!r}: no glossary rules sentence")
    modes = g["modes"]
    mode_lines = [modes[m] for m in ("m", "w", "p")] + list(_forced_mode_lines(env))
    blocks.append("# Action rules\n" + _bullets(rule_lines) + "\n\n# Routing modes\n"
                  + _bullets(mode_lines))
    if guidelines:
        gl = g["guidelines"]
        blocks.append("# Routing Guidelines\n" + _bullets(gl["items"]) + "\n\n"
                      + gl["detour_pattern"] + "\n\n" + gl["wrong_layer"])
    return "\n\n".join(blocks)


def build(env, *, guidelines: bool) -> LatestPrompt:
    """The ``latest`` prompt for the EnvConfig ``env``; ``guidelines`` adds the strategy hints."""
    from pcb_world.core.reward_config import effective_reward_config

    g = glossary()
    actions, lines = action_lines(env.masking_rule, env.net_reselect)
    cfg = effective_reward_config(env.reward_rule, vars(env.reward))
    core = _system_blocks(env, cfg, guidelines=guidelines)
    fb = g["feedback"]
    response = "\n".join(["# Response Format", g["response_format"]["text"],
                          fb["no_effect"]["text"], fb["rejected"]["text"]])
    mode_lines = (g["modes"]["line"], *_forced_mode_lines(env))
    desc = _state_format_desc(env, cfg)
    c = lambda s: s.replace("{", "{{").replace("}", "}}")   # noqa: E731 — literal text in a template
    system = f"{c(core)}\n\n{c(response)}\n\n# Board State\n{{state_format_desc}}\n\n{{board_static}}"
    user_no_his = "{dynamic_observation}\n\n# Valid Actions\nChoose exactly one from:\n{valid_actions}"
    user = ("## Step {current_step}\n{dynamic_observation}\n\n"
            "# History (step {current_step}, {valid_step_count} valid actions taken)\n"
            "{action_history}{rejected_attempts}\n\n"
            "# Valid Actions\nChoose exactly one from:\n{valid_actions}")
    template_no_his = (f"\n{c(core)}\n\n# Board State\n{{state_format_desc}}\n\n"
                       f"{{current_observation}}\n\n# Valid Actions\nChoose exactly one from:\n"
                       f"{{valid_actions}}\n\n{c(response)}\n")
    template = (f"\n{c(core)}\n\n# Board State\n{{state_format_desc}}\n\n## Step {{current_step}}\n"
                f"{{current_observation}}\n\n# History (step {{current_step}}, "
                f"{{valid_step_count}} valid actions taken)\n{{action_history}}{{rejected_attempts}}"
                f"\n\n# Valid Actions\nChoose exactly one from:\n{{valid_actions}}\n\n{c(response)}\n")
    return LatestPrompt(actions=actions, state_format_desc=desc, system=system,
                        user_no_his=user_no_his, user=user, template_no_his=template_no_his,
                        template=template, action_lines=lines, mode_lines=mode_lines)
