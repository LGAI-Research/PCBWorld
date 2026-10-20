"""Reward configuration system with YAML-based config loading.

Mirrors the masking rule pattern: schema + YAML loader + registry + auto-discovery.

YAML format (validated by :class:`RewardRuleSchema`; every key below except the
optional ``PotentialReward`` arguments is required):
    name: default
    mode: terminal     # "terminal" or "per_step"
    potential:         # PotentialReward arguments (absent ones keep its defaults) + the four penalties
      completion_bonus: 5.0
      unconnected_penalty: 1.0
      drc_penalty: 0.5
      truncation_mode: none
      step_penalty: 0.01
      invalid_action_penalty: 0.01
      parse_fail_penalty: 0.2
      mask_reject_penalty: 0.09

Usage:
    cfg = get_reward_config("default")
    reward = cfg.build_reward()
"""

from __future__ import annotations

import copy
import typing
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from pcb_world.core.reward import FinalOnlyReward, PotentialReward, RewardFunction


# ---------------------------------------------------------------------------
# Reward config protocol
# ---------------------------------------------------------------------------

VALID_MODES = ("terminal", "per_step")


class RewardConfig(Protocol):
    """Interface for reward configurations."""

    @property
    def name(self) -> str: ...

    @property
    def mode(self) -> str:
        """Reward mode: 'terminal' or 'per_step'."""
        ...

    @property
    def step_penalty(self) -> float:
        """Fixed penalty applied every step (positive value, negated at use)."""
        ...

    @property
    def invalid_action_penalty(self) -> float:
        """Penalty applied when the engine rejects an action that passed the
        action-type mask (e.g. make_line to an unreachable point). Positive
        value, subtracted from step reward when ``info["action_success"]``
        is False."""
        ...

    @property
    def parse_fail_penalty(self) -> float:
        """Penalty applied when the agent's output is unparseable (LLM env
        only). The LLM projection layer flags the action dict with
        ``_parse_invalid=True`` so PCBWorld can apply this penalty as the
        single source of truth for invalid-action handling."""
        ...

    @property
    def mask_reject_penalty(self) -> float:
        """Penalty applied when the agent picks a state-illegal action
        (action-mask rejects before engine dispatch). Positive value,
        subtracted from step reward — cousin of ``invalid_action_penalty``
        (which fires post-dispatch) and ``parse_fail_penalty`` (output
        unparseable). Kept as a separate knob since mask-reject signals a
        higher-level policy error than empty/dispatch-fail."""
        ...

    def build_reward(self) -> PotentialReward:
        """Build unified reward function from config."""
        ...

    def with_overrides(self, **overrides: Any) -> "RewardConfig":
        """Return a copy with non-None potential-field overrides applied.

        ``None`` values mean "keep the config value". Implementations must
        NOT mutate ``self``: instances handed out by
        :func:`get_reward_config` are cached and shared process-wide.
        """
        ...

    # Deprecated — kept for backward compatibility
    def build_step_reward(self) -> RewardFunction:
        """Deprecated: use build_reward() instead."""
        ...

    def build_final_reward(self) -> FinalOnlyReward:
        """Deprecated: use build_reward() instead."""
        ...


# ---------------------------------------------------------------------------
# YAML-based reward config
# ---------------------------------------------------------------------------

# Mode names of older rules, read as the paper terms.
_MODE_MIGRATIONS = {
    "final_only": "terminal",
    "sparse": "terminal",
    "combined": "per_step",
    "step": "per_step",
    "dense": "per_step",
}

# Penalties every rule states under ``potential:``. The env applies them; only
# step_penalty is also a PotentialReward argument.
_RULE_PENALTIES = ("step_penalty", "invalid_action_penalty", "parse_fail_penalty", "mask_reject_penalty")


def is_old_reward_format(raw: Any) -> bool:
    """A reward rule in the format before ``potential:`` — the old loader read any mapping without
    it (top-level ``step_reward:`` / ``final_reward:`` / penalties, or only ``name`` / ``mode``).
    Converted for a rule outside the tracked tree (``configs.loader.load_rule_file``), refused in
    a tracked one."""
    return isinstance(raw, Mapping) and "potential" not in raw


@dataclass(frozen=True)
class RewardRuleSchema:
    """Schema of one reward rule YAML — see the module docstring for the layout.

    ``potential`` holds the :class:`PotentialReward` arguments the rule sets plus the
    four penalties of ``_RULE_PENALTIES`` (required). Build it with :meth:`from_dict`.
    """

    name: str
    mode: str
    potential: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Any, source: str | Path) -> "RewardRuleSchema":
        """Validate a parsed rule YAML; any unknown, missing or mistyped key raises ``ValueError`` naming *source*."""
        from configs.loader import check_mapping

        where = f"reward rule {source}"
        if is_old_reward_format(raw):
            raise ValueError(
                f"{where}: old reward format (no `potential:`; top-level keys {sorted(raw)}) is read only for "
                f"a rule outside the tracked tree — rewrite it as a `potential:` rule."
            )
        top = check_mapping(raw, _SCHEMA_TYPES, where=where)
        check_mapping(top["potential"], _POTENTIAL_TYPES, required=_RULE_PENALTIES, where=f"{where} potential")
        mode = top["mode"]
        if mode in _MODE_MIGRATIONS:
            warnings.warn(
                f"Reward mode {mode!r} is deprecated, treating as {_MODE_MIGRATIONS[mode]!r} ({source}).",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = _MODE_MIGRATIONS[mode]
        if mode not in VALID_MODES:
            raise ValueError(f"{where}: invalid mode={mode!r}. Must be one of {VALID_MODES}.")
        return cls(name=top["name"], mode=mode, potential=MappingProxyType(dict(top["potential"])))


_SCHEMA_TYPES = typing.get_type_hints(RewardRuleSchema)
# ``potential:`` keys: the PotentialReward arguments + the env penalties.
_POTENTIAL_TYPES = {
    **{k: v for k, v in typing.get_type_hints(PotentialReward.__init__).items() if k != "return"},
    **dict.fromkeys(_RULE_PENALTIES, float),
}


class YamlRewardConfig:
    """Reward configuration built from a validated :class:`RewardRuleSchema`."""

    def __init__(self, rule: RewardRuleSchema) -> None:
        self.rule = rule
        self._name: str = rule.name
        self._mode: str = rule.mode
        # PotentialReward arguments; the three env-only penalties are split off.
        self._potential_cfg: dict[str, Any] = dict(rule.potential)
        self._step_penalty: float = self._potential_cfg["step_penalty"]
        self._invalid_action_penalty: float = self._potential_cfg.pop("invalid_action_penalty")
        self._parse_fail_penalty: float = self._potential_cfg.pop("parse_fail_penalty")
        self._mask_reject_penalty: float = self._potential_cfg.pop("mask_reject_penalty")

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def step_penalty(self) -> float:
        return self._step_penalty

    @property
    def invalid_action_penalty(self) -> float:
        return self._invalid_action_penalty

    @property
    def parse_fail_penalty(self) -> float:
        return self._parse_fail_penalty

    @property
    def mask_reject_penalty(self) -> float:
        return self._mask_reject_penalty

    def build_reward(self) -> PotentialReward:
        """Build a unified PotentialReward from config."""
        return PotentialReward(**self._potential_cfg)

    def with_overrides(self, **overrides: Any) -> "YamlRewardConfig":
        """Return a copy with non-None potential-field overrides applied.

        Keys must be :class:`PotentialReward` constructor arguments — unknown
        keys or invalid values fail loudly at the next :meth:`build_reward`.
        Always copies so the shared, cached instances handed out by
        :func:`get_reward_config` stay pristine.
        """
        applied = {k: v for k, v in overrides.items() if v is not None}
        if not applied:
            return self
        new = copy.copy(self)
        new._potential_cfg = {**self._potential_cfg, **applied}
        if "step_penalty" in applied:
            new._step_penalty = applied["step_penalty"]
        return new

    # -- Deprecated backward-compatible methods --

    def build_step_reward(self) -> RewardFunction:
        """Deprecated: use build_reward() instead."""
        warnings.warn(
            "build_step_reward() is deprecated, use build_reward().",
            DeprecationWarning,
            stacklevel=2,
        )
        return RewardFunction()  # type: ignore[return-value]

    def build_final_reward(self) -> FinalOnlyReward:
        """Deprecated: use build_reward() instead."""
        warnings.warn(
            "build_final_reward() is deprecated, use build_reward().",
            DeprecationWarning,
            stacklevel=2,
        )
        return FinalOnlyReward()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Loading & registry
# ---------------------------------------------------------------------------

_REWARD_CONFIG_CACHE: dict[str, RewardConfig] = {}

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Name lookup order: the live rules in configs/reward first, then the KDD-paper
# rules kept with their reproduction recipes under experiments/kdd/configs/reward.
# The second dir exists for the paper checkpoints and recipes, which store rule
# NAMES (not paths); new work should not add rules there.
_REWARD_SEARCH_DIRS = (
    _REPO_ROOT / "configs" / "reward",
    _REPO_ROOT / "experiments" / "kdd" / "configs" / "reward",
)

# Backward-compatible name mapping (old names → new canonical names).
_NAME_COMPAT: dict[str, str] = {
    "shaped_log_all": "drc_only_dense",
    "step_drc": "drc_linear",
    # drc_sparse_promoted was renamed to drc_sparse_promoted_grpo when the
    # PPO variant (drc_sparse_promoted_ppo, truncation_mode=none) was split
    # off — the truncation_mode=full version is GRPO-only because GRPO has
    # no critic bootstrap to absorb terminal Φ. Legacy v51/v56 scripts still
    # reference the bare name; alias keeps them resolving.
    "drc_sparse_promoted": "drc_sparse_promoted_grpo",
    # The shipped rule was renamed when it became the one public reward; checkpoints store the name.
    "drc_log_ladder_linear_w7a": "pcbworld_reward",
}


def register_reward_config(name: str, config: RewardConfig) -> None:
    """Register a reward config instance in the cache."""
    _REWARD_CONFIG_CACHE[name] = config


def load_reward_config(path: str | Path) -> YamlRewardConfig:
    """Load a YamlRewardConfig from a YAML config file.

    Args:
        path: Path to a reward config YAML file.

    Returns:
        Instantiated YamlRewardConfig.
    """
    from configs.loader import load_rule_file

    rule = RewardRuleSchema.from_dict(load_rule_file(path, "reward"), path)
    _warn_linear_ladder(rule, path)
    return YamlRewardConfig(rule)


# Rules already reported by _warn_linear_ladder in this process (resolved path).
_LINEAR_LADDER_WARNED: set[str] = set()


def _warn_linear_ladder(rule: RewardRuleSchema, path: str | Path) -> None:
    """Name the rule when it turns the linear per-net ladder weighting on.

    Once per distinct rule file and process (path-loaded rules are not cached,
    so eval would otherwise repeat it per board).
    """
    # Optional key: absent means the PotentialReward default (0.0, off).
    scale = rule.potential.get("net_bonus_size_linear_scale")
    if scale is None or not scale > 0:
        return
    key = str(Path(path).resolve())
    if key not in _LINEAR_LADDER_WARNED:
        _LINEAR_LADDER_WARNED.add(key)
        warnings.warn(
            f"reward rule {rule.name!r} ({key}): net_bonus_size_linear_scale="
            f"{scale} > 0 weights the per-net ladder by pad groups "
            f"(w_i = scale * pg_i); configs that reproduce runs before "
            f"2026-10-09 use 0.0.",
            UserWarning,
            stacklevel=3,
        )


def reward_rule_path(name: str) -> Path | None:
    """The YAML file the reward rule *name* resolves to (an explicit path, else ``{name}.yaml``
    under the search dirs after the name mapping); None when no file has that name (a rule
    registered in-process only)."""
    if name.endswith((".yaml", ".yml")):
        return Path(name)
    name = _NAME_COMPAT.get(name, name)
    return next((d / f"{name}.yaml" for d in _REWARD_SEARCH_DIRS if (d / f"{name}.yaml").exists()), None)


#: The per-run reward overrides a run trains with — ``RewardOverrides`` field (= the flat
#: checkpoint / run-record key) -> :class:`PotentialReward` argument; ``None`` keeps the rule's.
REWARD_OVERRIDE_ARGS: Mapping[str, str] = MappingProxyType({
    "reward_step_penalty": "step_penalty", "via_penalty": "via_penalty",
    "wirelength_penalty": "wirelength_penalty", "drc_penalty": "drc_penalty",
    "drc_log_scale": "drc_log_scale", "drc_log_agg_scale": "drc_log_agg_scale",
    "drc_log_offset": "drc_log_offset", "wire_via_emission": "wire_via_emission",
})
#: :class:`PotentialReward` arguments that shape the reward but not Φ itself.
_NON_PHI_ARGS = ("step_penalty", "truncation_mode", "wire_via_emission", "jumanji_connector_dense",
                 "jumanji_connector_dense_flat", "jumanji_connector_wirelength_dense")


def effective_reward_config(rule: str, overrides: Mapping[str, Any]) -> RewardConfig:
    """The reward a run trains with: the rule *rule* with the run's overrides (*overrides* holds
    every :data:`REWARD_OVERRIDE_ARGS` key, e.g. ``RewardOverrides`` as a dict or a checkpoint's
    args)."""
    return get_reward_config(rule).with_overrides(
        **{arg: overrides[key] for key, arg in REWARD_OVERRIDE_ARGS.items()})


def phi_diff(a: YamlRewardConfig, b: YamlRewardConfig) -> dict[str, tuple[Any, Any]]:
    """``{argument: (a's value, b's value)}`` of the :class:`PotentialReward` arguments that define
    Φ where the reward configs *a* and *b* differ — an argument a config leaves out read as its
    default; names, ``mode``, the env penalties and :data:`_NON_PHI_ARGS` are not Φ. Empty = the
    same Φ."""
    import inspect

    defaults = {k: p.default for k, p in inspect.signature(PotentialReward.__init__).parameters.items()
                if p.default is not inspect.Parameter.empty}

    def phi(cfg: YamlRewardConfig) -> dict[str, Any]:
        return {k: v for k, v in {**defaults, **cfg._potential_cfg}.items() if k not in _NON_PHI_ARGS}

    pa, pb = phi(a), phi(b)
    return {k: (pa[k], pb[k]) for k in pa if pa[k] != pb[k]}


def get_reward_config(name: str = "default") -> RewardConfig:
    """Get a reward config by name or YAML path.

    Resolution order:
        1. If *name* ends with .yaml/.yml -> load directly from that path.
        2. Apply backward-compatible name mapping.
        3. If *name* is already cached -> return cached instance.
        4. Look for {name}.yaml in configs/reward/, then in
           experiments/kdd/configs/reward/ (paper-era rules) -> load and cache.
        5. Raise KeyError if not found.
    """
    # 1. Explicit YAML path
    if name.endswith((".yaml", ".yml")):
        return load_reward_config(name)

    # 2. Backward-compatible name mapping
    name = _NAME_COMPAT.get(name, name)

    # 3. Cache hit
    if name in _REWARD_CONFIG_CACHE:
        return _REWARD_CONFIG_CACHE[name]

    # 4. Auto-discover {name}.yaml from the search dirs
    yaml_path = reward_rule_path(name)
    if yaml_path is not None:
        config = load_reward_config(yaml_path)
        _REWARD_CONFIG_CACHE[name] = config
        return config

    raise KeyError(
        f"Unknown reward config '{name}'. No cached config and no {name}.yaml under "
        + " or ".join(str(d) for d in _REWARD_SEARCH_DIRS) + "."
    )
