"""Typed config schemas — the single place to read/edit run config + defaults.

One module holding the canonical, sink-agnostic config dataclasses shared across
``envs`` / ``training`` / ``eval``. argparse Namespaces, checkpoint ``args``
dicts, and YAML are *adapters* that load into these — the field names and
defaults are defined once, here, not re-declared per entry point.

Lives in the existing ``configs`` package (alongside the YAML files + loaders in
``configs/__init__.py``) so there is one ``configs/`` home, not a separate
``config/`` vs ``configs/`` split.

Contents:
  * :class:`EnvConfig`   — env-core (what ``PCBWorld`` is built from), shared
    RL + LLM; ``with_overrides()`` for convenience construction.
  * :class:`RLEnvConfig` — env-core + RL-wrapper knobs (what ``make_decoder_env``
    is built from); ``from_namespace`` / ``from_checkpoint`` adapters.
  * :class:`RLPolicyConfig` — ``KiCadRLModel`` construction; ``build()`` +
    ``from_namespace`` / ``from_checkpoint``.

The ``from_checkpoint`` adapters take the resolved mapping of
:func:`configs.loader.ckpt.load_ckpt_config` (the only reader of a checkpoint's
``args``), never the raw args.

Scope note — ``corner_mode`` vs ``check_angle``: ``corner_mode`` here is the
routing angle in degrees (45 or 90); ``pcb_world.core.env.corner_mode_code`` turns
it into the engine code at the env boundary. ``check_angle`` (the post-hoc
track-angle DRC check in :func:`eval.metrics.compute_metrics`) is a *scoring*
concern, NOT an env construction param; eval derives it from the final angle.

Heavy deps (``torch`` / ``KiCadRLModel``) are imported lazily inside
:meth:`RLPolicyConfig.build`, so importing this module stays torch-free.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from configs.loader import load_config
from configs.loader.knob import knob

# Default *values* live in YAML (the single place to read/edit them); the
# dataclasses below define the *schema* (fields, types, adapters) and pull their
# field defaults from here. One file per program, one section per surface (section: class):
#   configs/env.yaml  env: EnvConfig (+ EnvConfig.reward) — shared by the RL and LLM programs
#   configs/rl.yaml   rl_train: RLTrainConfig/PPOConfig/GRPOConfig · rl_policy: RLPolicyConfig ·
#                     rl_eval: RLEvalConfig
#   configs/llm.yaml  llm: LLMConfig (nested groups) · llm_train: LLMTrainConfig
# Edit a value there to change the default everywhere; an experiment sets other values with
# --recipe <yaml> instead.
_CONFIGS_DIR = Path(__file__).resolve().parent.parent
_DEFAULTS_FILES = ("env.yaml", "rl.yaml", "llm.yaml")


def _load_defaults() -> dict[str, Any]:
    """The sections of every :data:`_DEFAULTS_FILES` file; a section two files define raises."""
    out: dict[str, Any] = {}
    for name in _DEFAULTS_FILES:
        sections = load_config(_CONFIGS_DIR / name)
        dup = sorted(set(out) & set(sections))
        if dup:
            raise ValueError(f"configs/{name}: sections {dup} are already defined by another "
                             f"defaults file ({', '.join(_DEFAULTS_FILES)})")
        out.update(sections)
    return out


_DEFAULTS = _load_defaults()


def _none_if_off(value: Any) -> Any:
    """CLI spelling of ``directional_candidates`` None (the default 8-direction 0.5 mm ring):
    ``none`` / ``off`` / empty -> None."""
    if isinstance(value, str) and value.strip().lower() in ("none", "off", ""):
        return None
    return value


_ENV = _DEFAULTS["env"]
_REWARD = _ENV["reward"]
_POLICY = _DEFAULTS["rl_policy"]


# ============================================================================
# Env construction
# ============================================================================


@dataclass(frozen=True)
class RewardOverrides:
    """Per-run overrides of the selected ``reward_rule`` YAML.

    Each field is ``None`` by default = "use the reward rule's own value"; setting
    one overrides that term of the potential (see
    ``pcb_world.core.reward_config`` / ``PCBWorld``). Lives under
    :attr:`EnvConfig.reward`; ``reward_rule`` (which YAML) and ``reward_noise_std``
    stay on :class:`EnvConfig`.
    """

    via_penalty: float | None = knob(_REWARD["via_penalty"], "experiment", eval_override=True,
        help="Override potential's via_penalty (Φ -= w * via_count). "
             "None = use reward-rule YAML default (0).")
    wirelength_penalty: float | None = knob(_REWARD["wirelength_penalty"], "experiment", eval_override=True,
        help="Override potential's wirelength_penalty (Φ -= w * wirelength_mm). "
             "None = YAML default.")
    drc_penalty: float | None = knob(_REWARD["drc_penalty"], "experiment", eval_override=True,
        help="Override potential's drc_penalty (linear-shape DRC weight). "
             "None = YAML default.")
    drc_log_scale: float | None = knob(_REWARD["drc_log_scale"], "experiment", eval_override=True,
        help="Override drc_log_scale for log_per_net DRC shape (per-net depth "
             "term). None = YAML default.")
    drc_log_agg_scale: float | None = knob(_REWARD["drc_log_agg_scale"], "experiment", eval_override=True,
        help="Override drc_log_agg_scale for log_per_net DRC shape (aggregate "
             "breadth term; 0 disables). None = YAML default.")
    drc_log_offset: float | None = knob(_REWARD["drc_log_offset"], "experiment",
        help="Override drc_log_offset (log-curve knee) for log_per_net DRC "
             "shape. None = YAML default.")
    reward_step_penalty: float | None = knob(_REWARD["reward_step_penalty"], "experiment", eval_override=True,
        help="Override potential step_penalty. Used by the Jumanji Connector "
             "dense reward to scale grid sizes.")
    wire_via_emission: str | None = knob(_REWARD["wire_via_emission"], "experiment",
        choices=["per_step", "on_net_end"],
        help="When to emit wire + via penalties in per_step mode: 'per_step' "
             "(every env step, default) or 'on_net_end' (accumulated and "
             "emitted only on ACT_NET_END steps). Terminal mode unaffected.")


#: ``EnvConfig.rules_source`` values (``pcb_world.engine.drc_config.apply_default_drc_if_fallback``).
RULES_SOURCES: tuple[str, ...] = ("board", "board_or_yaml")

#: ``readback_reason`` of a field the env (or the engine) keeps as the attribute it was
#: given: reading that attribute back would compare the config with its own copy.
_HELD_AS_GIVEN = "held as given; no deeper sink"
#: ``readback_reason`` of a rule name: resolved into an object that keeps no name.
_RULE_RESOLVED = ("a rule name or YAML path resolved (aliases included) into the {} once; "
                  "it keeps no name to compare")


@dataclass(frozen=True)
class EnvConfig:
    """Canonical env-core config — the ``PCBWorld`` keyword surface.

    The reward-rule penalty overrides are grouped under :attr:`reward`
    (:class:`RewardOverrides`); :meth:`to_env_kwargs` flattens them back so the
    KiCad env still receives a flat kwargs dict.

    Every field declares its readback in ``metadata["readback"]``: the attribute path,
    on the constructed ``PCBWorld``, of a sink that holds the value in its own form
    (a C++ getter, a structure built from it — ``PCBWorld.effective_config`` reads it
    and the constructor raises when it differs from the config), or ``None`` with a
    ``readback_reason``.
    """

    max_steps: int = knob(_ENV["max_steps"], "experiment", eval_override=True,
        readback=None, readback_reason=_HELD_AS_GIVEN,
        help="Per-episode time-limit (env truncation cap)")
    masking_rule: str = knob(_ENV["masking_rule"], "experiment", eval_override=True,
        readback=None, readback_reason=_RULE_RESOLVED.format("rule object"))
    # One knob, two consumers: the env reopens a closed net on net_select (and
    # keeps the all-nets-closed termination), the RL wrapper's net-valid mask
    # reads PCBWorld.net_reselect to keep such nets selectable.
    net_reselect: bool = knob(_ENV["net_reselect"], "experiment",
        readback=None, readback_reason=_HELD_AS_GIVEN,
        help="Let a net closed by net_end while it still has ratsnest be "
             "selected again: it stays in the net-valid mask and net_select "
             "on it reopens it (it leaves closed_nets until its next "
             "net_end). Episode termination is unchanged. Off by default "
             "(net_end retires the net for the episode); stored in the "
             "checkpoint and restored when it is evaluated.")
    reward_rule: str = knob(_ENV["reward_rule"], "experiment", eval_override=True,
        readback=None, readback_reason=_RULE_RESOLVED.format("reward config"))
    reward_noise_std: float = knob(_ENV["reward_noise_std"], "experiment",
        readback=None, readback_reason=_HELD_AS_GIVEN,
        help="Gaussian noise stddev applied to env reward (default: 0).")
    # Result-affecting constants promoted to fields (NO_FLAG_FIELDS): no CLI flag — no run
    # sets one — but carried, recorded and pinned in the checkpoint like every field.
    reward_noise_clip_sigma: float = knob(_ENV["reward_noise_clip_sigma"], "experiment",
        readback=None, readback_reason=_HELD_AS_GIVEN,
        cli_skip=True,
        help="The reward noise is clipped to +-this x reward_noise_std (default 4).")
    drc_incremental: bool = knob(_ENV["drc_incremental"], "experiment",
        readback="_drc_incremental",
        cli_skip=True,
        help="Per-step DRC runs the engine's incremental path (re-checks changed items "
             "only; tested equal to a full DRC). false = a full DRC every step. The "
             "episode-end DRC is full either way.")
    drc_promoted_codes: list[int] = knob(scope="experiment",
        default_factory=lambda: list(_ENV["drc_promoted_codes"]),
        readback="_engine.drc_helper.promoted_codes",
        cli_skip=True,
        help="KiCad PCB_DRC_CODE values of the warnings the errors_and_promoted DRC severity "
             "mode counts as errors (reward and DRC tokens).")
    # emit_drc_tokens / corner_mode: variables live here (visible), but their CLI
    # forms are bespoke — emit_drc_tokens is the inverted --no-drc-tokens pair and
    # corner_mode is each entrypoint's own --corner-mode — so the generator skips
    # them and each entrypoint adds the explicit flag.
    emit_drc_tokens: bool = knob(_ENV["emit_drc_tokens"], "experiment", eval_override=True, cli_skip=True,
        readback=None, readback_reason=_HELD_AS_GIVEN)
    # degrees, 45 or 90 (see module docstring)
    corner_mode: int = knob(_ENV["corner_mode"], "experiment", eval_override=True, cli_skip=True,
        readback=None, readback_reason=_HELD_AS_GIVEN)
    rules_source: str = knob(_ENV["rules_source"], "experiment",
        readback=None, readback_reason=_HELD_AS_GIVEN,
        choices=list(RULES_SOURCES),
        help="Where a board's design rules come from. 'board': its .kicad_pro "
             "or legacy (setup ...) rules; a board with neither is refused. "
             "'board_or_yaml': such a board gets the global minima of the YAML "
             "at --drc-config-path (required then) instead, its Default net "
             "class sizes raised to them, with a UserWarning; a board with "
             "rules of its own is unaffected.")
    drc_config_path: str | None = knob(_ENV["drc_config_path"], "experiment",
        readback=None, readback_reason=_HELD_AS_GIVEN,
        help="YAML DRC config used with --rules-source board_or_yaml. "
             "Required then — there is no implicit default "
             "(e.g. configs/drc/pcbworld_drc.yaml).")
    # Engine knobs (ENGINE_KNOBS): no CLI flag — but carried to
    # KiCadEngine and recorded in the checkpoint like every other field. reject_if_stuck is the
    # exception: it gets the generated flag pair (training --[no-]reject-if-stuck, LLM --[no-]reject_if_stuck;
    # the KDD recipes pin the paper's false).
    engine_seed: int = knob(_ENV["engine_seed"], "experiment",
        readback="_engine.engine_seed",
        cli_skip=True,
        help="Seed for KiCad's KIID/UUID generator, set once at engine init. "
             "Makes routing + UUID-keyed DRC reproducible across runs/processes "
             "for a fixed action sequence (default 77). Global generator → one "
             "env per process for clean determinism.")
    shove_iter_limit: int = knob(_ENV["shove_iter_limit"], "experiment",
        readback="_engine.shove_iter_limit",
        cli_skip=True,
        help="Max PNS shove iterations (default 250). The shove loop is bounded "
             "by this count rather than by a wallclock timeout, so runs "
             "reproduce.")
    followbranch_iter_limit: int = knob(_ENV["followbranch_iter_limit"], "experiment",
        readback="_engine.followbranch_iter_limit",
        cli_skip=True,
        help="Max TOPOLOGY::followBranch DFS pops (default 1,000,000). Same rule "
             "as the shove bound: an iteration count, not a wallclock timeout.")
    reject_if_stuck: bool = knob(_ENV["reject_if_stuck"], "experiment",
        readback=None, readback_reason=_HELD_AS_GIVEN,
        help="make_line aborts (draws nothing, action fails) when the walkaround "
             "cannot reach the target, instead of committing a partial dangling stub "
             "(default true); false = legacy commit-partial behavior (checkpoints before 2026-07-10, the KDD paper).")
    trim_at_contact: bool = knob(_ENV["trim_at_contact"], "experiment",
        readback=None, readback_reason=_HELD_AS_GIVEN,
        cli_skip=True,
        help="Just before each make_line/make_via/finish commit, on every net the commit touches "
             "(including nets shove moved): drop copper lying on same-net copper and "
             "split tracks at T/X junctions and vias so they are joints. The route "
             "is not recomputed; vias are kept (default false).")
    simplify_outline: bool = knob(_ENV["simplify_outline"], "experiment",
        readback="_engine.simplify_outline",
        help="Rewrite tessellated micro-segment outline chains (Edge.Cuts/"
             "Margin) into native arcs/merged lines at board load "
             "(pcb_world/engine/outline_simplify.py). Kills the PNS walkaround "
             "cluster blowup on baked-curve boards. Off by default; changes obs "
             "EDGE token counts, so keep one setting within a campaign.")
    obs_format: str = knob(_ENV["obs_format"], "runtime",
        readback=None, readback_reason=_HELD_AS_GIVEN,
        choices=["json", "indexed"],
        cli_skip=True,  # Not exposed on the public CLI — the RL training
        # path is fixed to indexed; LLM/eval uses json. scripts/profile.py
        # injects args.obs_format programmatically, for diagnostic A/B only.
        help="Observation format the env emits. 'json' = legacy nested "
             "dict (LLM/eval path); 'indexed' = indexed_v1 array tables "
             "(pcb_world/core/indexed_obs.py) — fixed value for the RL "
             "training path, tokens bit-identical.")
    outline_obs: str = knob(_ENV["outline_obs"], "arch",
        readback=None,
        readback_reason="consumed once by parse_pcb_file at construction; the parsed "
                        "edges carry no mode tag",
        choices=["poly16", "tess", "arc"],
        help="Edge.Cuts representation in the obs boardlines. 'tess' = "
             "C++ error-bounded tessellation into segments (0.005mm, "
             "the default); 'poly16' = fixed 16-per-90deg re-tessellation "
             "(coarse A/B reference); 'arc' = one entry per "
             "arc/circle carrying the on-arc midpoint (requires an arc-capable "
             "policy — legacy 2-point checkpoints refuse arc obs).")
    # CLI form is owned by the policy group's --action-history-len (one knob
    # drives both env window and tokenizer K); from_namespace copies it here.
    action_history_len: int = knob(_ENV["action_history_len"], "experiment",
        readback="_action_history.maxlen",
        cli_skip=True,
        help="obs['action_history'] window: how many recent action records "
             "(type/pointer/net/success, newest first) the env keeps and "
             "emits (default 1 = the single-step prev-action window; history "
             "ablations raise it). K=1 checkpoints consume entry 0 only.")
    net_constraint_obs: bool = knob(_ENV["net_constraint_obs"], "arch",
        readback=None, readback_reason=_HELD_AS_GIVEN,
        help="Per-net DRC constraint observation: every board_static net "
             "carries its resolved netclass values (track_width/clearance/"
             "via_diameter/via_drill; KiCad inherit→Default fallback — the "
             "sizes the engine routes that net with), feeding "
             "the NET-token tw/cl/vd channels (constant 0 when off). "
             "Obs-drift: retrain; the ckpt stores the setting so eval "
             "follows automatically.")
    # CLI form is the bespoke two-float --keep-routing-fraction LO HI (args.py);
    # the generator can't express nargs=2.
    keep_routing_fraction: list[float] | None = knob(_ENV["keep_routing_fraction"], "experiment",
        readback="_keep_routing_fraction",
        cli_skip=True,
        help="Keep-routing augmentation (train-only): (lo, hi) — each "
             "board load samples f~U[lo,hi] and keeps that fraction of "
             "the nets' file routing as the episode's initial state "
             "(uniform random nets; rest re-routed from scratch). "
             "Requires boards whose file carries complete routing. "
             "None = off; eval always off.")
    reward: RewardOverrides = field(default_factory=RewardOverrides, metadata={
        "readback": None,
        "readback_reason": "folded with reward_rule into one reward config (None = the "
                           "rule's value), whose build validates the values"})

    def to_env_kwargs(self) -> dict[str, Any]:
        """Flat dict of every field, the ``reward`` sub-object's penalty keys at top level."""
        d = asdict(self)
        d.update(d.pop("reward"))
        return d

    def with_flat(self, flat: Mapping[str, Any]) -> "EnvConfig":
        """Copy with ``flat`` applied. Keys are the flat surface of :meth:`to_env_kwargs`
        (EnvConfig fields, plus the :class:`RewardOverrides` fields at top level);
        an unknown key raises ``ValueError``."""
        unknown = sorted(set(flat) - _ENV_FLAT_KEYS)
        if unknown:
            raise ValueError(f"not EnvConfig fields: {unknown}")
        reward_keys = {f.name for f in fields(RewardOverrides)}
        top = {k: v for k, v in flat.items() if k not in reward_keys}
        reward = {k: v for k, v in flat.items() if k in reward_keys}
        return replace(self, **top, reward=replace(self.reward, **reward))

    @classmethod
    def with_overrides(cls, **kw: Any) -> "EnvConfig":
        """The YAML defaults with ``kw`` applied — convenience construction for tests,
        viz and diagnostics, which have no run config to resolve. An unknown key
        raises ``TypeError``."""
        return replace(cls(), **kw)

    def engine_kwargs(self) -> dict[str, Any]:
        """``KiCadEngine`` construction kwargs: the :data:`ENGINE_CTOR_FIELDS`."""
        return {k: getattr(self, k) for k in ENGINE_CTOR_FIELDS}


_ENV_FLAT_KEYS = frozenset(EnvConfig().to_env_kwargs())

#: EnvConfig fields consumed by the C++ router through ``KiCadEngine``.
ENGINE_KNOBS = ("engine_seed", "shove_iter_limit", "followbranch_iter_limit",
                "reject_if_stuck", "trim_at_contact")

#: EnvConfig fields without a CLI flag: the training parser puts their YAML values on the
#: namespace (so the run record and the checkpoint record them) and
#: :meth:`RLEnvConfig.from_namespace` reads them from there.
NO_FLAG_FIELDS = (*(k for k in ENGINE_KNOBS if k != "reject_if_stuck"),
                  "reward_noise_clip_sigma", "drc_incremental", "drc_promoted_codes")

#: EnvConfig fields ``KiCadEngine`` takes at construction (:meth:`EnvConfig.engine_kwargs`).
ENGINE_CTOR_FIELDS = (*ENGINE_KNOBS, "simplify_outline", "drc_promoted_codes")


#: Config fields a checkpoint does not carry: runtime emission format and
#: train-only augmentation, pinned by :meth:`RLEnvConfig.from_checkpoint`.
RUNTIME_FIELDS = frozenset({
    "obs_format", "reward_noise_std", "reward_noise_clip_sigma", "keep_routing_fraction",
    "slot_perm", "aug_bbox_shifted", "aug_flip", "aug_rotate", "aug_trans", "aug_zoom",
})


#: Fields only the training env sets (their defaults = off; the noise clip acts only with
#: noise on); validation and eval build without them (:meth:`RLEnvConfig.for_training`).
TRAIN_LAYER = ("reward_noise_std", "reward_noise_clip_sigma", "aug_bbox_shifted", "aug_flip",
               "aug_rotate", "aug_trans", "aug_zoom")

#: Train-only fields an eval env takes from the defaults (off), never from the training config:
#: the RUNTIME_FIELDS but ``obs_format`` (each path emits its own format, bit-identical tokens)
#: and ``slot_perm`` — inline validation keeps a slot_perm run's permutation, as it always has
#: (``from_checkpoint`` pins it off for a checkpoint eval).
EVAL_PINNED = RUNTIME_FIELDS - {"obs_format", "slot_perm"}

#: No env-side DRC (eval ``step_drc=False``): no DRC tokens, zero DRC reward terms — the
#: engine is then never asked for DRC. Flat keys of :meth:`RLEnvConfig.to_record`.
DRC_OFF = MappingProxyType({"emit_drc_tokens": False, "drc_penalty": 0.0,
                            "drc_log_scale": 0.0, "drc_log_agg_scale": 0.0})

#: Parser dests that are a field's inverted flag (``--no-x`` stores ``no_x`` = not ``x``).
INVERTED_DESTS = MappingProxyType({"no_drc_tokens": "emit_drc_tokens",
                                   "no_mask_start_point": "mask_start_point",
                                   "no_vecenv": "vecenv", "no_eval_greedy": "eval_greedy"})

#: ``RLPolicyConfig`` fields only a checkpoint's weights set (an older token layout, derived by
#: ``configs.loader.ckpt``): a training run has no option for them — a fresh run builds the
#: current layout, ``--resume`` / ``--init-ckpt`` take the checkpoint's (arch keys, pinned).
CHECKPOINT_LAYOUT_FIELDS = ("legacy_pad_layer_encoding", "legacy_net_encoding", "legacy_action_history")


def _field_values(args: Any) -> dict[str, Any]:
    """The values of a resolved config ``args`` by field name: an :data:`INVERTED_DESTS` dest is
    read as its field."""
    values = dict(vars(args))
    values.update({name: not values[dest] for dest, name in INVERTED_DESTS.items() if dest in values})
    return values


@dataclass(frozen=True)
class RLEnvConfig:
    """Env-core + the RL-wrapper knobs — what ``make_decoder_env`` builds from."""

    env: EnvConfig = field(default_factory=EnvConfig)
    force_walkaround: bool = knob(False, "experiment",
        help="Override routing_mode to 2 (Walkaround) for all "
             "make_line/make_via/finish actions, matching the MLP trainer's "
             "hardcoded mode. Use for fair comparison.")
    # mask_start_point: variable lives here (visible) but its CLI form is the
    # inverted --no-mask-start-point, so the generator skips it.
    mask_start_point: bool = knob(True, "experiment", cli_skip=True)
    slot_perm: bool = knob(False, "experiment",
        help="v1: per-episode random permutation of the net slot embedding "
             "table (net-exchangeability augmentation). v0 (default) uses "
             "identity mapping sorted-net-k -> slot-k.")
    directional_candidates: str | None = knob(_ENV["directional_candidates"], "arch",
        help="Directional candidate generation mode: a preset name "
             "(e.g. 'multi_resolution' = 8 dirs × 0.2/1.0/5.0/25.0mm; see "
             "candidate_pool.DIRECTIONAL_DISTANCE_PRESETS) or 'grid<N>' "
             "(1-layer Grid mode, e.g. grid200). None/'none' = the default "
             "8-direction 0.5 mm ring.")
    # connectivity_filter: variable lives here (visible) but its CLI form is the
    # --connectivity-filter / --no-connectivity-filter pair, so the generator
    # skips it. It trims already-connected existing copper from the candidate
    # pool = it changes the ACTION SPACE, so train and eval must agree.
    connectivity_filter: bool = knob(True, "experiment", cli_skip=True)
    # Pad-graze guard width (mm; 0 = off). Drops DIRECTIONAL
    # candidates landing in the sliver annulus outside a same-net pad (copper
    # the track cleaner deletes) — changes the CANDIDATE SET, so like
    # connectivity_filter it round-trips through checkpoints. CLI form is the
    # bespoke --pad-graze-margin-mm in the args module (richer help text), so
    # the generator skips it; that flag's default reads THIS field.
    pad_graze_margin_mm: float = knob(0.0, "experiment", cli_skip=True)
    # Off-board pointer mask (act time, every cand row): blocks DIRECTIONAL
    # candidates whose (x, y) falls outside the board bbox — the long rungs of
    # a ladder preset (mres8: 25 / 50 mm) overshoot most boards. Leaves the
    # candidate SET (pool / pointer index space) untouched, but a policy trained
    # with it never learned to avoid off-board targets, so it round-trips
    # through checkpoints like the other wrapper knobs. Default off = existing
    # checkpoints keep their exact behaviour.
    offboard_mask: bool = knob(_ENV["offboard_mask"], "arch",
        help="Hard-mask DIRECTIONAL candidates whose (x, y) falls outside "
             "the board bbox at act time (every cand row: neither make_line "
             "nor make_via can target them). Off by default; stored in the "
             "checkpoint and restored when it is evaluated.")
    # Per-episode augmentation toggles (5 independent boolean axes). Train-only
    # (TRAIN_LAYER, applied by for_training); validation and eval never enable them.
    aug_bbox_shifted: bool = knob(False, "experiment",
        help="Per-episode per-axis virtual scale of the board edges around a "
             "random interior centre (scale_x, scale_y ~ U[0.7, 1.3]). Pads "
             "stay physically fixed.")
    aug_flip: bool = knob(False, "experiment",
        help="Per-episode independent x/y sign reflection (prob 0.5 each).")
    aug_rotate: bool = knob(False, "experiment",
        help="Per-episode x/y axis swap (prob 0.5).")
    aug_trans: bool = knob(False, "experiment",
        help="Per-episode feature-space translation (nn_dx, nn_dy ~ U[-0.2, 0.2]).")
    aug_zoom: bool = knob(False, "experiment",
        help="Per-episode feature-space uniform scale jitter "
             "(nn_zoom ~ U[0.9, 1.1]): the whole normalized scene — "
             "positions and dims together — is zoomed around the bbox "
             "centre, so the board long axis spans [-zoom, +zoom] instead "
             "of exactly [-1, 1].")

    def wrapper_kwargs(self) -> dict[str, Any]:
        """``KiCadRLWrapper`` construction kwargs: every field but ``env``."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "env"}

    def to_record(self) -> dict[str, Any]:
        """Flat dict of every leaf field (env and reward fields at top level) — the
        form run records, the train/val env check and the eval manifest store and compare."""
        return {**self.env.to_env_kwargs(), **self.wrapper_kwargs()}

    def with_flat(self, flat: Mapping[str, Any]) -> "RLEnvConfig":
        """Copy with ``flat`` applied; keys are those of :meth:`to_record`. An unknown
        key raises (``ValueError`` for an env key, ``TypeError`` otherwise)."""
        return replace(
            self, env=self.env.with_flat({k: v for k, v in flat.items() if k in _ENV_FLAT_KEYS}),
            **{k: v for k, v in flat.items() if k not in _ENV_FLAT_KEYS},
        )

    @classmethod
    def from_namespace(cls, args: Any) -> "RLEnvConfig":
        """Build from a training run's resolved config (a ``RunConfig``, or a namespace holding
        every training dest) — the env validation builds: each :meth:`to_record` key is read
        from it (an :data:`INVERTED_DESTS` dest as its field; a missing key raises, no default
        fills it), except the training-only layer (:data:`TRAIN_LAYER`, left off —
        :meth:`for_training` adds it)."""
        v = _field_values(args)
        return cls().with_flat({**{k: v[k] for k in cls().to_record() if k not in TRAIN_LAYER},
                                "directional_candidates": _none_if_off(v["directional_candidates"])})

    @classmethod
    def for_training(cls, args: Any) -> "RLEnvConfig":
        """The training env: :meth:`from_namespace` + the namespace's
        :data:`TRAIN_LAYER` values (reward noise, per-episode augmentation)."""
        return cls.from_namespace(args).with_flat({k: getattr(args, k) for k in TRAIN_LAYER})

    @classmethod
    def from_checkpoint(cls, ckpt: Any, max_steps: int) -> "RLEnvConfig":
        """Build from a checkpoint's resolved config (eval path).

        ``ckpt`` is the ``values`` mapping of
        :func:`configs.loader.ckpt.load_ckpt_config` — complete for every field
        read here (stored, renamed, derived or pre-knob legacy value), so no key
        falls back to a current default. ``max_steps`` is the caller's episode
        cap (an eval override or ``ckpt["max_steps"]``).

        TRAIN-only augmentations are pinned off (``slot_perm`` False,
        ``keep_routing_fraction`` None, ``reward_noise_std`` 0): eval scores the
        full from-scratch board deterministically. ``obs_format`` keeps the YAML
        default (the eval/LLM path's json; the tokens are bit-identical to the
        trainer's indexed format).
        """
        explicit = RUNTIME_FIELDS | {"env", "max_steps", "reward", "directional_candidates"}
        env = EnvConfig(
            max_steps=max_steps,
            keep_routing_fraction=None,
            reward=RewardOverrides(**{f.name: ckpt[f.name] for f in fields(RewardOverrides)}),
            **{f.name: ckpt[f.name] for f in fields(EnvConfig) if f.name not in explicit},
        )
        return cls(
            env=env,
            directional_candidates=_none_if_off(ckpt["directional_candidates"]),
            **{f.name: ckpt[f.name] for f in fields(cls) if f.name not in explicit},
        )


# ============================================================================
# Policy construction
# ============================================================================


@dataclass(frozen=True)
class RLPolicyConfig:
    """Canonical ``KiCadRLModel`` construction config.

    Field ``metadata`` drives :func:`configs.loader.cli.add_dataclass_args` (help /
    choices). ``cli_skip`` marks fields NOT emitted as plain CLI flags by the
    generator: ``use_critic`` / ``legacy_pad_layer_encoding`` are set by the
    trainer/checkpoint (not flags); the bool toggles (``detach_critic`` etc.)
    are added explicitly by the trainer entrypoints where they belong.
    """

    d_model: int = knob(_POLICY["d_model"], "arch")
    n_heads: int = knob(_POLICY["n_heads"], "arch")
    n_layers: int = knob(_POLICY["n_layers"], "arch",
        help="Transformer layer count (4 for small boards; increase for larger)")
    d_ff: int = knob(_POLICY["d_ff"], "arch")
    max_seq_len: int = knob(_POLICY["max_seq_len"], "arch")
    n_freq: int = knob(_POLICY["n_freq"], "arch")
    n_max_slots: int = knob(_POLICY["n_max_slots"], "arch",
        help="Slot-table size = per-board max net count the tokenizer "
             "accepts (rows of slot_emb_table — checkpoints record it; "
             "old ckpts without the key load as 64). Default 512 gives "
             "headroom for real-board sets (d3b max 109 nets). "
             "Incompatible with --slot-perm unless 64.")
    use_critic: bool = knob(_POLICY["use_critic"], "arch", cli_skip=True)
    detach_critic: bool = knob(_POLICY["detach_critic"], "experiment", cli_skip=True)
    coord_encoding: str = knob(_POLICY["coord_encoding"], "arch",
        choices=["fourier", "mlp", "linear"],
        help="Continuous-value encoding: 'fourier' (default, sin/cos "
             "multi-freq), 'mlp' (raw values through a 2-layer MLP with "
             "LayerNorm), or 'linear'.")
    mlp_hidden: int = knob(_POLICY["mlp_hidden"], "arch",
        help="Hidden size of the coord MLP (only used when --coord-encoding mlp).")
    disable_slot_emb: bool = knob(_POLICY["disable_slot_emb"], "arch",
        help="Drop the input-side slot embedding (no per-net fingerprint added to token "
             "embeddings; embed_ln still applies). The slot table then only bounds the "
             "per-board net count, which lets eval lift it (--override-n-max-slots).")
    policy_net_select: bool = knob(_POLICY["policy_net_select"], "arch",
        help="Learn net ordering: the policy's net_select pointer is propagated to the env "
             "and contributes to log-prob / entropy; invalid (already-routed) nets are "
             "masked out. Off: the env picks a random unfinished net and the pointer is "
             "kept out of log-prob / entropy.")
    same_net_bias: bool = knob(_POLICY["same_net_bias"], "arch", cli_skip=True)
    legacy_pad_layer_encoding: bool = knob(_POLICY["legacy_pad_layer_encoding"], "arch", cli_skip=True)
    legacy_net_encoding: bool = knob(_POLICY["legacy_net_encoding"], "arch", cli_skip=True)
    legacy_edge_encoding: bool = knob(_POLICY["legacy_edge_encoding"], "arch", cli_skip=True)
    time_feature: str = knob(_POLICY["time_feature"], "arch",
        choices=["step_ratio", "log_remaining", "sin_remaining", "none"],
        help="HEAD-token time scalar: 'step_ratio' (step/max_steps, legacy), "
             "'log_remaining' (log1p(steps_remaining)/log1p(cap), "
             "max_steps-invariant across per-board step budgets), "
             "'sin_remaining' (steps_remaining/cap through a dedicated "
             "sinusoidal ladder anchored to step units — resolves ±1 step "
             "at any horizon, transformer-PE style), or 'none' (constant 0 "
             "— time-blind ablation; the policy cannot observe episode "
             "progress. All modes share one Fourier slot with zero new "
             "weights, so checkpoints stay weight-compatible).")
    time_feature_cap: int = knob(_POLICY["time_feature_cap"], "arch",
        help="Fixed normalization ceiling for --time-feature "
             "log_remaining/sin_remaining (largest plausible max_steps; "
             "for sin_remaining also the longest ladder period).")
    action_history_len: int = knob(_POLICY["action_history_len"], "experiment",
        help="ACTION_HISTORY window K: recent action entries tokenized per "
             "obs (3 tokens each, newest first; idle-sentinel padded). Age "
             "uses Fourier(age/MAX_HISTORY)->proj — no K-shaped weights, so "
             "checkpoints stay load-compatible across K.")
    legacy_action_history: bool = knob(_POLICY["legacy_action_history"], "arch", cli_skip=True)
    obstacle_obs: bool = knob(_POLICY["obstacle_obs"], "arch",
        help="Emit OBSTACLE tokens for netless immovable blockers (NPTH "
             "mounting holes/slots + net-less NC pads; rule-area keepout "
             "zones stay engine-only). Off (default) keeps the token "
             "stream byte-identical to pre-knob checkpoints.")
    shape_obs: bool = knob(_POLICY["shape_obs"], "arch",
        help="Add the boundary-shape channel (6-bucket categorical: "
             "rect/roundrect/circle/oval/other/unknown) to PAD and "
             "OBSTACLE tokens via an additive shape_embed. Off (default) "
             "adds no weights — old checkpoints round-trip unchanged.")

    def build(self, device: Any | None = None) -> Any:
        """Construct the policy (optionally moved to ``device``)."""
        from methods.rl_agent.models.v1.net import KiCadRLModel

        policy = KiCadRLModel(**asdict(self))
        return policy if device is None else policy.to(device)

    @classmethod
    def from_namespace(cls, args: Any, *, use_critic: bool) -> "RLPolicyConfig":
        """Build from a training run's resolved config (like :meth:`RLEnvConfig.from_namespace`;
        a missing field raises). ``use_critic`` is supplied by the trainer (per-algorithm class
        attribute, PPO=True / GRPO=False — not an argparse flag); the
        :data:`CHECKPOINT_LAYOUT_FIELDS` are the run's (the checkpoint's on a resume / init)."""
        v = _field_values(args)
        return cls(use_critic=use_critic, **{f.name: v[f.name] for f in fields(cls) if f.name != "use_critic"})

    @classmethod
    def from_checkpoint(cls, ckpt: Any) -> "RLPolicyConfig":
        """Build from a checkpoint's resolved config (eval path): the ``values``
        mapping of :func:`configs.loader.ckpt.load_ckpt_config`, where the
        weight-shaped fields (``use_critic``, ``n_max_slots``, ``shape_obs``,
        ``legacy_*``) are already derived from the state_dict and pre-knob keys
        carry their legacy values."""
        return cls(**{f.name: ckpt[f.name] for f in fields(cls)})


# ============================================================================
# RL training orchestration (train-only argparse defaults)
# ============================================================================
#
# These hold the *default values* for the train-only CLI flags. The argparse
# builders (methods/rl_agent/training/args.py + the PPO/GRPO entrypoints) read defaults from
# here; the trainer still consumes the parsed argparse Namespace, so these are
# the single place to see/edit the default numbers, not a new consumption path.

_TRAIN = _DEFAULTS["rl_train"]
_TRAIN_SHARED = _TRAIN["shared"]

#: The mode defaults of an init-ckpt run (a BC-init fine-tune; configs.loader.resolve): its
#: optimizer starts from configs/rl.yaml ``rl_train.finetune`` instead of ``shared``.
FINETUNE_DEFAULTS = MappingProxyType(dict(_TRAIN["finetune"]))


@dataclass(frozen=True)
class RLTrainConfig:
    """Shared (PPO+GRPO) training-loop / optim / logging / board defaults.

    Emitted as CLI flags by :func:`configs.loader.cli.add_dataclass_args` (the few
    with non-trivial help/choices carry ``metadata``).
    """

    iterations: int = knob(_TRAIN_SHARED["iterations"], "experiment", resume_mutable=True)
    # optimization
    lr: float = knob(_TRAIN_SHARED["lr"], "experiment")
    clip_eps: float = knob(_TRAIN_SHARED["clip_eps"], "experiment")
    entropy_coef: float = knob(_TRAIN_SHARED["entropy_coef"], "experiment",
        help="Entropy bonus coefficient")
    entropy_norm: bool = knob(_TRAIN_SHARED["entropy_norm"], "experiment",
        help="Normalize each sample's joint entropy by its max achievable "
             "entropy ln(N_valid) (summed per used head over the masked "
             "action/pointer/mode distributions) before applying "
             "--entropy-coef. Makes the bonus invariant to action-space "
             "size (e.g. mres candidate ladders); the logged 'entropy' "
             "metric becomes this [0,1] relative entropy.")
    batch_size: int = knob(_TRAIN_SHARED["batch_size"], "experiment")
    max_grad_norm: float = knob(_TRAIN_SHARED["max_grad_norm"], "experiment",
        help="Gradient clipping")
    warmup_iters: int = knob(_TRAIN_SHARED["warmup_iters"], "experiment",
        help="Linear LR warmup iterations (0 = disabled)")
    # NOTE: update-time OOM is handled automatically by policy_update_loop
    # (sorted 1/4-peel gradient-accumulation, no budget/knob) —
    # see methods/rl_agent/algorithms/_common.py.
    mem_budget: bool = knob(_TRAIN_SHARED["mem_budget"], "runtime",
        help="Preemptive peak-VRAM budget planner: calibrate peak(B,L) on "
             "the first buffer, then pre-split update minibatches and "
             "rollout forwards to fit free VRAM (exact; OOM peel stays as "
             "backstop). See methods/rl_agent/training/mem_budget.py.")
    expect_env_diff: str = knob("", "runtime",
        help="Comma-separated env kwargs that are ALLOWED to differ between "
             "the training envs and the validation envs. The trainer records "
             "the validation env in <save-dir>/run_record.val.yaml, compares "
             "it with the training env and halts on any undeclared "
             "difference, printing the line to paste here. Intent lives with "
             "the run, not in a global table, so a deliberately different "
             "validation setup stays expressible. Differences the harness "
             "creates itself (the val seed, which comes from "
             "--eval-base-seed) are not compared.")
    # logging / checkpointing
    save_freq: int = knob(_TRAIN_SHARED["save_freq"], "runtime")
    log_every: int = knob(_TRAIN_SHARED["log_every"], "runtime",
        help="Print iter summary every N iterations (default: 1 = every iter).")
    seed: int = knob(_TRAIN_SHARED["seed"], "experiment")
    device: str = knob(_TRAIN_SHARED["device"], "runtime")
    # inline held-out eval cadence
    eval_n_rollouts: int = knob(_TRAIN_SHARED["eval_n_rollouts"], "experiment", resume_mutable=True,
        help="Stochastic rollouts per test board during inline eval.")
    eval_base_seed: int = knob(_TRAIN_SHARED["eval_base_seed"], "experiment", resume_mutable=True,
        help="Base seed for inline eval rollouts.")
    # board / dataset selection
    boards_order: str = knob(_TRAIN_SHARED["boards_order"], "experiment",
        choices=["single", "round_robin", "per_env_random", "per_env_epoch"],
        help="single: train on --board only (default). round_robin: rotate "
             "through boards in --boards-json, sorted ascending by pad count "
             "(gentle curriculum), one board/iter replicated across n_envs. "
             "per_env_random: each env samples a random board per iteration "
             "(sticky within iter, WITH replacement). per_env_epoch: shuffle "
             "the full list, consume n_envs boards/iter WITHOUT replacement; "
             "reshuffle on exhaustion (~ceil(N/n_envs) iters/epoch).")
    boards_json: str | None = knob(_TRAIN_SHARED["boards_json"], "experiment", resume_mutable=True,
        help="JSON split file — required when --boards-order != single "
             "(configs/datasets/ registry).")
    boards_difficulty: str = knob(_TRAIN_SHARED["boards_difficulty"], "experiment",
                                   choices=["easy", "medium", "hard"])
    boards_split: str = knob(_TRAIN_SHARED["boards_split"], "experiment",
        help="Key under <difficulty> to use (e.g. 'train', 'train_small', 'test').")
    boards_dataset_dir: str | None = knob(_TRAIN_SHARED["boards_dataset_dir"], "experiment", resume_mutable=True,
        help="Deprecated compatibility override: if set, used instead of the "
             "split JSON's top-level dataset_dirs entry for both train and "
             "eval board resolution.")
    # held-out validation (val/*, policy_best.pt)
    eval_split: str | None = knob(_TRAIN_SHARED["eval_split"], "experiment", resume_mutable=True,
        off_flag=True,
        help="The --boards-json split to validate on (val/*, policy_best.pt); needs a "
             "multi-board --boards-order. A single-board run skips the default (logged); "
             "an --eval-split given there is an error. --no-eval-split: no validation.")
    eval_every: int | None = knob(_TRAIN_SHARED["eval_every"], "experiment", resume_mutable=True,
        help="Run held-out eval every N iterations. If unset, falls back to "
             "once per full pass of the train board list.")
    eval_boards_per_batch: int | None = knob(_TRAIN_SHARED["eval_boards_per_batch"], "runtime",
        help="Boards to pack into one eval pool batch. None = auto "
             "(n_envs // eval_n_rollouts).")
    eval_board_limit: int | None = knob(_TRAIN_SHARED["eval_board_limit"], "experiment", resume_mutable=True,
        help="Optional cap on inline held-out eval boards. None evaluates the "
             "full split.")
    # secondary (diagnostic) inline eval set — does NOT drive best-ckpt
    eval2_boards: str | None = knob(_TRAIN_SHARED["eval2_boards"], "experiment", resume_mutable=True,
        help="Secondary held-out eval board-list file (e.g. real d3-b). "
             "Evaluated each --eval-every under --eval2-prefix; diagnostic "
             "only (best-ckpt stays on the primary --eval-split).")
    eval2_prefix: str = knob(_TRAIN_SHARED["eval2_prefix"], "runtime",
        help="Dashboard prefix for the --eval2-boards set (default val_d3b).")
    eval3_boards: str | None = knob(_TRAIN_SHARED["eval3_boards"], "experiment", resume_mutable=True,
        help="Third held-out eval board-list file (e.g. real d3-a). Same "
             "semantics as --eval2-boards; logged under --eval3-prefix.")
    eval3_prefix: str = knob(_TRAIN_SHARED["eval3_prefix"], "runtime",
        help="Dashboard prefix for the --eval3-boards set (default val_d3a).")
    eval4_boards: str | None = knob(_TRAIN_SHARED["eval4_boards"], "experiment", resume_mutable=True,
        help="Fourth held-out eval board-list file. Same semantics as "
             "--eval2-boards; logged under --eval4-prefix.")
    eval4_prefix: str = knob(_TRAIN_SHARED["eval4_prefix"], "runtime",
        help="Dashboard prefix for the --eval4-boards set (default val_d2b).")
    eval5_boards: str | None = knob(_TRAIN_SHARED["eval5_boards"], "experiment", resume_mutable=True,
        help="Fifth held-out eval board-list file. Same semantics as "
             "--eval2-boards; logged under --eval5-prefix.")
    eval5_prefix: str = knob(_TRAIN_SHARED["eval5_prefix"], "runtime",
        help="Dashboard prefix for the --eval5-boards set (default val_d2a).")
    eval_diag_max_steps: int | None = knob(_TRAIN_SHARED["eval_diag_max_steps"], "experiment", resume_mutable=True,
            help="Pin the diagnostic eval2..eval5 sets to this max_steps "
                 "(env cap + rollout loop), regardless of the run's "
                 "--max-steps — keeps the diagnostic protocol comparable "
                 "across cells that vary the train horizon. None (default) "
                 "= inherit the train protocol. Primary val / val_greedy "
                 "stay native.")
    eval_diag_masking_rule: str | None = knob(_TRAIN_SHARED["eval_diag_masking_rule"], "experiment", resume_mutable=True,
            help="Pin the diagnostic eval2..eval5 sets to this masking "
                 "rule (path or name), regardless of the run's "
                 "--masking-rule. None (default) = inherit. Primary val / "
                 "val_greedy stay native.")
    eval_at_init: bool = knob(_TRAIN_SHARED["eval_at_init"], "runtime",
        help="Eval the initial (untrained/zero-shot) policy once at iter 0 "
             "before training; logged but excluded from best-ckpt.")
    eval_greedy: bool = knob(_TRAIN_SHARED["eval_greedy"], "runtime",
        cli_skip=True,  # default-True bool -> inverted flag --no-eval-greedy (args.py tail)
        help="Greedy (argmax) 1-rollout pass over the primary val set each "
             "cadence, logged under val_greedy/* (mean==max per board by "
             "construction). Separates exploration noise from genuine "
             "uncertainty next to the sampled val/*; diagnostic only — "
             "best-ckpt stays on val/*.")
    async_val: bool = knob(_TRAIN_SHARED["async_val"], "runtime",
        help="Detach validation from training: at each eval cadence, queue "
             "the policy into <save-dir>/val_queue/ for an external watcher "
             "process (methods.rl_agent.training.async_val) instead of "
             "evaluating inline. Results are logged into the trainer's own "
             "TensorBoard/W&B run as they arrive. Requires --eval-split; "
             "default off = inline eval, unchanged.")
    # Weights & Biases (optional; default off — telemetry is opt-in via --wandb).
    # `wandb` is the positive master toggle; its CLI form is added explicitly by
    # the args tail, so it is cli_skip. The wandb_* string knobs are emitted
    # directly.
    wandb: bool = knob(_TRAIN_SHARED["wandb"], "runtime", cli_skip=True)
    wandb_project: str | None = knob(_TRAIN_SHARED["wandb_project"], "runtime",
        help="W&B project name. Falls back to $WANDB_PROJECT or "
             "'pcbworld'.")
    wandb_entity: str | None = knob(_TRAIN_SHARED["wandb_entity"], "runtime",
        help="W&B entity (team/org). Falls back to $WANDB_ENTITY.")
    wandb_group: str | None = knob(_TRAIN_SHARED["wandb_group"], "runtime",
        help="W&B run group (clusters ablation runs). Falls back to "
             "$WANDB_RUN_GROUP.")
    wandb_run_name: str | None = knob(_TRAIN_SHARED["wandb_run_name"], "runtime",
        help="W&B run display name. Defaults to the --log-dir basename.")
    wandb_tags: str | None = knob(_TRAIN_SHARED["wandb_tags"], "runtime",
        help="Comma-separated list of W&B tags (e.g. 'stage1,ablation').")
    # vecenv: positive master toggle (inverted CLI --no-vecenv), so cli_skip.
    vecenv: bool = knob(_TRAIN_SHARED["vecenv"], "runtime", cli_skip=True)
    # resume: variable lives here (visible); its CLI form (--resume, type=str) is
    # added by each entrypoint, so cli_skip.
    resume: str | None = knob(_TRAIN_SHARED["resume"], "runtime", cli_skip=True)


@dataclass(frozen=True)
class PPOConfig:
    """PPO-specific training defaults (rollout + GAE + per-algo dirs).

    ``n_epochs`` / ``log_dir`` / ``save_dir`` are ``cli_skip`` — they feed
    :func:`methods.rl_agent.training.args.add_shared_args` as per-algorithm defaults (the
    flags are emitted there), not as entrypoint-local flags.
    """

    n_epochs: int = knob(_TRAIN["ppo"]["n_epochs"], "experiment", cli_skip=True)
    # n_envs is `experiment`, not the `runtime` the scope table lists for pool sizes: it sets the
    # rollout batch (samples per update; with eval_n_rollouts the inline-eval packing), so it is
    # recorded and pinned — resume_mutable / eval_override (the watcher sizes its own pool) free it.
    n_envs: int = knob(_TRAIN["ppo"]["n_envs"], "experiment", resume_mutable=True, eval_override=True,
        help="Number of parallel environments")
    n_steps: int = knob(_TRAIN["ppo"]["n_steps"], "experiment",
        help="Steps per env per PPO iteration")
    gamma: float = knob(_TRAIN["ppo"]["gamma"], "experiment",
        help="Discount factor (default 1.0 — episodic + terminal-reward)")
    gae_lambda: float = knob(_TRAIN["ppo"]["gae_lambda"], "experiment")
    vf_coef: float = knob(_TRAIN["ppo"]["vf_coef"], "experiment")
    norm_reward_clip: float = knob(_TRAIN["ppo"]["norm_reward_clip"], "experiment",
        help="Symmetric clipping bound for normalized rewards")
    # --- BC-initialised fine-tuning (methods/rl_agent/training/finetune.py, loop.py) ---
    init_ckpt: str | None = knob(_TRAIN["ppo"]["init_ckpt"], "experiment",
        help="Initialise the policy WEIGHTS from this checkpoint (e.g. a BC "
             "policy_step_*.pt); optimizer, iteration counter and reward "
             "normalizer start fresh (unlike --resume). A missing critic_head "
             "is newly initialised; obs-affecting args are cross-checked.")
    critic_warmup_iters: int = knob(_TRAIN["ppo"]["critic_warmup_iters"], "experiment",
        help="Iterations 1..N train ONLY the critic head: actor + backbone "
             "frozen (detach_critic forced on, policy/entropy/KL terms "
             "dropped) at --critic-warmup-lr; the actor's --warmup-iters LR "
             "ramp starts after them.")
    critic_warmup_lr: float = knob(_TRAIN["ppo"]["critic_warmup_lr"], "experiment",
        help="Learning rate during --critic-warmup-iters")
    critic_warmup_min_ev: float = knob(_TRAIN["ppo"]["critic_warmup_min_ev"], "experiment",
        help="EV gate: after --critic-warmup-iters keep the critic-only phase until "
             "train/explained_variance >= this for --critic-warmup-ev-streak consecutive "
             "iterations (0 = fixed-length warm-up). Scratch runs here reach 0.9 by ~iter 20.")
    critic_warmup_ev_streak: int = knob(_TRAIN["ppo"]["critic_warmup_ev_streak"], "experiment",
        help="Consecutive iterations the EV gate must hold")
    critic_warmup_max_iters: int = knob(_TRAIN["ppo"]["critic_warmup_max_iters"], "experiment",
        help="Hard cap on the EV-gated warm-up (0 = 3 x --critic-warmup-iters); the actor "
             "phase starts here even if the gate never passed (logged loudly)")
    kl_ref_coef: float = knob(_TRAIN["ppo"]["kl_ref_coef"], "experiment",
        help="beta of the KL(pi || pi_ref) penalty toward a frozen reference "
             "policy (k3 estimator on the rollout actions); 0 = off. Not "
             "supported with --update-gpus > 1.")
    kl_ref_ckpt: str | None = knob(_TRAIN["ppo"]["kl_ref_ckpt"], "experiment",
        help="Reference policy checkpoint for --kl-ref-coef (default: --init-ckpt)")
    kl_ref_seen_drc_types: str = knob(_TRAIN["ppo"]["kl_ref_seen_drc_types"], "experiment",
        help="Comma list of DRC token type_ids the reference policy was trained "
             "on (pcb_world.engine.drc taxonomy: 2=unconnected 4=via 6=other are "
             "the connectivity-only tokens DRC-clean BC demos carry). With "
             "--kl-ref-mask-unseen-drc, a sample whose observation holds any "
             "other type_id gets KL weight 0.")
    kl_ref_mask_unseen_drc: bool = knob(_TRAIN["ppo"]["kl_ref_mask_unseen_drc"], "experiment", cli_skip=True)
    normalize_adv: bool = knob(_TRAIN["ppo"]["normalize_adv"], "experiment",
        help="Per-minibatch advantage normalization (SB3 default)")
    truncation_bootstrap: bool = knob(_TRAIN["ppo"]["truncation_bootstrap"], "experiment",
        help="V(s_next) bootstrap at truncation boundaries (off: treat truncation as termination with V=0)")
    norm_reward: bool = knob(_TRAIN["ppo"]["norm_reward"], "experiment",
        help="SB3-style discounted-return reward normalization")
    log_dir: str = knob(_TRAIN["ppo"]["log_dir"], "runtime", cli_skip=True)
    save_dir: str = knob(_TRAIN["ppo"]["save_dir"], "runtime", cli_skip=True)


@dataclass(frozen=True)
class GRPOConfig:
    """GRPO-specific training defaults (group baseline + per-algo dirs).

    ``n_epochs`` / ``log_dir`` / ``save_dir`` are ``cli_skip`` (see PPOConfig).
    """

    n_epochs: int = knob(_TRAIN["grpo"]["n_epochs"], "experiment", cli_skip=True)
    # experiment + resume_mutable / eval_override: see PPOConfig.n_envs.
    n_envs: int = knob(_TRAIN["grpo"]["n_envs"], "experiment", resume_mutable=True, eval_override=True,
        help="Number of parallel environments (GRPO groups)")
    group_size: int = knob(_TRAIN["grpo"]["group_size"], "experiment",
        help="Rollouts per GRPO group")
    log_dir: str = knob(_TRAIN["grpo"]["log_dir"], "runtime", cli_skip=True)
    save_dir: str = knob(_TRAIN["grpo"]["save_dir"], "runtime", cli_skip=True)


# ============================================================================
# RL eval orchestration
# ============================================================================
#
# Eval-pipeline knobs (env/policy come from the checkpoint at eval time). The
# `_EVAL` values live in configs/rl.yaml (rl_eval:); the validation enums + manifest version
# below are code constants (argparse `choices=` / on-disk schema tag), not
# tunable values, so they stay here rather than in YAML.

# Manifest/summary schema version, shared by every eval stage entrypoint
# (core 3-stage, rollout.rule-based) — defined exactly once.
SCHEMA_VERSION: int = 1

ROLLOUT_MODES: tuple[str, ...] = ("serial", "parallel")
DRC_SWITCHES: tuple[str, ...] = ("on", "off")
SELECTION_METHODS: tuple[str, ...] = ("final_potential", "posthoc_drc_aware", "none")
CHECK_ANGLES: tuple[int, ...] = (45, 90)

_EVAL = _DEFAULTS["rl_eval"]


@dataclass(frozen=True)
class RLEvalConfig:
    """Canonical eval-pipeline defaults."""

    rollout_mode: str = knob(_EVAL["rollout_mode"], "runtime")
    n_envs: int = knob(_EVAL["n_envs"], "runtime")
    # None = inherit the ckpt's training emit_drc_tokens; "on"/"off" force it.
    env_drc: str | None = knob(_EVAL["env_drc"], "experiment")
    inline_drc: str = knob(_EVAL["inline_drc"], "experiment")
    # DRC scoring reward (configs/reward/*.yaml). Distinct from the training
    # reward_rule on EnvConfig: you can train with one rule and score with another.
    reward_config: str = knob(_EVAL["reward_config"], "experiment")
    # check_angle CLI flag defaults to None = follow the final env corner_mode (a
    # re-scoring: the cell's record); this value is the scoring helpers' default.
    check_angle: int = knob(_EVAL["check_angle"], "experiment")
    selection_method: str = knob(_EVAL["selection_method"], "experiment")
    # Intentionally lifts the ckpt's trained slot cap at inference; None = inherit.
    override_n_max_slots: int | None = knob(_EVAL["override_n_max_slots"], "experiment")
    save_artifacts: str = knob(_EVAL["save_artifacts"], "runtime")
    output_root: str = knob(_EVAL["output_root"], "runtime")
    device: str = knob(_EVAL["device"], "runtime")
    early_stop_finish_no_progress: int = knob(_EVAL["early_stop_finish_no_progress"], "experiment")
    early_stop_no_geometry_progress: int = knob(_EVAL["early_stop_no_geometry_progress"], "experiment")
    # Best-checkpoint selection — the single canonical criterion shared across
    # branches: RL (PPOTrainer/GRPOTrainer inline best-ckpt via on_validation)
    # AND LLM (verl trainer.best_metric_key in run_pcbworld*.sh) both select the
    # checkpoint maximizing this val tag ("largest fp gain is best").
    best_metric_key: str = knob(_EVAL["best_metric_key"], "experiment")
    best_metric_mode: str = knob(_EVAL["best_metric_mode"], "experiment")


# Back-compat instance name used as function/argparse defaults across eval/*.
DEFAULTS = RLEvalConfig()


# ============================================================================
# LLM eval orchestration
# ============================================================================
#
# The LLM eval program's defaults (eval/args.py reads these as argparse
# defaults). Structured into one nested sub-config per eval/args.py CLI group
# (env / boards / prompt / rollout / vllm / api), so the schema is the single
# *complete* visible list of every LLM-eval variable and each group function is a
# one-line `add_dataclass_args(parser, _L.<group>, style="underscore")`.
#
# `cli_skip` marks variables whose CLI *form* is bespoke (the inverted
# --no_drc_tokens, the degrees --corner_mode (45/90), the dynamic-choices
# --prompt_version, the repo-abs-resolved --boards_json) — the variable stays
# visible; only the flag is added explicitly.

_LLM = _DEFAULTS["llm"]


@dataclass(frozen=True)
class LLMEnvConfig:
    """LLM-only env fields: how the env's observation becomes prompt text. The env
    itself is an :class:`EnvConfig` (:attr:`LLMConfig.env`)."""

    state_format: str = knob(_LLM["env"]["state_format"], "experiment",
        choices=["sexpr", "xml"],
        help="Observation serialization format (default: sexpr).")


def _llm_env_config(base: EnvConfig, section: Mapping[str, Any], own: Any, where: str) -> EnvConfig:
    """``base`` with the EnvConfig keys of an LLM config section applied; every other
    key of the section must be a field of the dataclass ``own``."""
    unknown = sorted(set(section) - _ENV_FLAT_KEYS - {f.name for f in fields(own)})
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown} — not {own.__name__} fields "
                         "nor EnvConfig keys (configs/loader/schema.py)")
    return base.with_flat({k: v for k, v in section.items() if k in _ENV_FLAT_KEYS})


@dataclass(frozen=True)
class LLMBoardsConfig:
    """LLM-eval multi-board selection (eval/args.py ``boards`` group)."""

    boards_order: str = knob(_LLM["boards"]["boards_order"], "experiment",
        choices=["single", "round_robin"],
        help="single (default): eval only on --board_path. round_robin: "
             "iterate through boards in --boards_json (sorted ascending by pad "
             "count), running --rollout_episodes episodes per board.")
    # boards_json has no default (None) — multi-board mode must set it
    # explicitly; it stays cli_skip because its flag form is bespoke
    # (eval/args.py defines it directly).
    boards_json: str | None = knob(_LLM["boards"]["boards_json"], "experiment", cli_skip=True)
    boards_difficulty: str = knob(_LLM["boards"]["boards_difficulty"], "experiment",
                                   choices=["easy", "medium", "hard"])
    boards_split: str = knob(_LLM["boards"]["boards_split"], "experiment",
        help="Key under <difficulty> to use (e.g. 'train', 'test', "
             "'train_small'). Eval default is 'test'.")


@dataclass(frozen=True)
class LLMPromptConfig:
    """LLM-eval prompt selection (eval/args.py ``prompt`` group).

    ``prompt_version``'s choices are dynamic (``PCBWORLD_PROMPT_VERSIONS``), so the
    flag is added explicitly by ``add_prompt_args``; the variable is visible here.
    """

    prompt_version: str = knob(_LLM["prompt"]["prompt_version"], "experiment", cli_skip=True)


@dataclass(frozen=True)
class LLMRolloutConfig:
    """LLM-eval rollout orchestration (eval/args.py ``rollout`` group)."""

    env_num: int = knob(_LLM["rollout"]["env_num"], "runtime",
        help="Number of parallel envs in the vectorised rollout.")
    seed: int = knob(_LLM["rollout"]["seed"], "experiment")
    rollout_episodes: int = knob(_LLM["rollout"]["rollout_episodes"], "experiment")
    num_cpus_per_worker: float = knob(_LLM["rollout"]["num_cpus_per_worker"], "runtime")
    history_length: int = knob(_LLM["rollout"]["history_length"], "experiment",
        help="Number of recent (obs, action) pairs in prompt (0 = no history).")
    temperature: float = knob(_LLM["rollout"]["temperature"], "experiment")
    max_new_tokens: int = knob(_LLM["rollout"]["max_new_tokens"], "experiment")
    dump_dir: str | None = knob(_LLM["rollout"]["dump_dir"], "runtime",
        help="Directory for per-episode rollout JSON (one file per episode).")
    early_stop_no_progress: int = knob(_LLM["rollout"]["early_stop_no_progress"], "experiment",
        help="Early-stop an env after this many consecutive non-progress steps "
             "(parse-error / mask-rejected / idle / empty_action). 0 disables "
             "(the default). --early_stop_no_geometry_progress does not cover "
             "these: board-state measures are stamped on valid steps only, so a "
             "streak that never reaches the engine leaves that guard untouched.")
    early_stop_no_geometry_progress: int = knob(_LLM["rollout"]["early_stop_no_geometry_progress"], "experiment",
            help="Early-stop an env after this many consecutive steps that "
                 "changed no board-state measure (track/via count, "
                 "wirelength, ratsnest_reduction) — the shared "
                 "NoGeometryProgressGuard the RL rollout uses. 0 disables.")


@dataclass(frozen=True)
class LLMVLLMConfig:
    """LLM-eval vLLM-provider knobs (eval/args.py ``llm`` group)."""

    model_path: str = knob(_LLM["vllm"]["model_path"], "experiment",
        help="HuggingFace model name or local path.")
    tensor_parallel_size: int = knob(_LLM["vllm"]["tensor_parallel_size"], "runtime",
        help="Number of GPUs for tensor parallelism. Use CUDA_VISIBLE_DEVICES "
             "to select specific GPUs.")
    gpu_memory_utilization: float = knob(_LLM["vllm"]["gpu_memory_utilization"], "runtime",
        help="Fraction of GPU memory for vLLM KV cache (default: 0.95).")
    max_model_len: int = knob(_LLM["vllm"]["max_model_len"], "experiment",
        help="Maximum sequence length (prompt + generation).")
    enable_prefix_caching: bool = knob(_LLM["vllm"]["enable_prefix_caching"], "runtime",
        help="Enable vLLM automatic prefix caching.")
    enable_chunked_prefill: bool = knob(_LLM["vllm"]["enable_chunked_prefill"], "runtime",
        help="Enable vLLM chunked prefill.")
    guided_grammar: str | None = knob(_LLM["vllm"]["guided_grammar"], "experiment",
        help="Name of a grammar registered in methods.llm_agent.wrappers.grammar GRAMMARS "
             "(e.g. 'pcbworld_v1'). When set, generation is constrained via vLLM "
             "GuidedDecodingParams (regex, llguidance backend).")


@dataclass(frozen=True)
class LLMAPIConfig:
    """LLM-eval provider-API knobs (eval/args.py ``api`` group)."""

    api_provider: str = knob(_LLM["api"]["api_provider"], "experiment",
        choices=["openai", "anthropic", "google", "together"],
        help="API provider (default: openai). 'together' uses Together AI "
             "(OpenAI-compatible) for hosted open-weight models like Qwen.")
    api_model: str | None = knob(_LLM["api"]["api_model"], "experiment",
        help="API model name (e.g. 'gpt-4o', 'claude-sonnet-4-20250514', "
             "'gemini-2.0-flash'). If unset, uses a per-provider default.")
    api_key: str | None = knob(_LLM["api"]["api_key"], "runtime",
        help="API key. Overrides OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY.")
    api_rpm: int | None = knob(_LLM["api"]["api_rpm"], "runtime",
        help="Per-minute request cap. None disables (default).")
    api_itpm: int | None = knob(_LLM["api"]["api_itpm"], "runtime",
        help="Per-minute input-token cap (incl. prompt-cache reads/creations).")
    api_otpm: int | None = knob(_LLM["api"]["api_otpm"], "runtime",
        help="Per-minute output-token cap.")
    api_max_retries: int = knob(_LLM["api"]["api_max_retries"], "runtime",
        help="Provider-SDK retry count for 429 / transient errors (default 8).")
    api_prompt_cache: bool = knob(_LLM["api"]["api_prompt_cache"], "runtime",
        help="Anthropic only: tag the system prompt with cache_control (5-min "
             "ephemeral cache, ~90%% input cost reduction on hits).")
    api_reasoning_effort: str | None = knob(_LLM["api"]["api_reasoning_effort"], "experiment",
        choices=["minimal", "low", "medium", "high"],
        help="OpenAI reasoning models (o1/o3/o4/gpt-5/gpt-6). 'low' shrinks "
             "chain-of-thought so it fits inside max_new_tokens. Accepted "
             "values are model-specific ('minimal' is gpt-5-only; gpt-6 "
             "takes low|medium|high) — an unsupported value returns a "
             "provider 400. Ignored by non-OpenAI providers.")
    api_disable_thinking: bool = knob(_LLM["api"]["api_disable_thinking"], "experiment",
        help="Together (Qwen3 thinking models): pass "
             "chat_template_kwargs={enable_thinking: false} so the server skips "
             "chain-of-thought entirely.")


@dataclass(frozen=True)
class LLMConfig:
    """Canonical LLM-eval defaults — the complete, grouped variable list.

    One nested sub-config per eval/args.py CLI group; each is generated via
    :func:`configs.loader.cli.add_dataclass_args` (``style='underscore'``). Defaults live
    in configs/llm.yaml (llm:). :attr:`env` is the same :class:`EnvConfig` RL
    builds (``env:`` defaults; ``llm.env`` sets the few LLM-specific values) and
    the ``env`` CLI group carries both it and :attr:`llm_env`.
    """

    env: EnvConfig = field(default_factory=lambda: _llm_env_config(
        EnvConfig(), _LLM["env"], LLMEnvConfig, "configs/llm.yaml llm.env"))
    llm_env: LLMEnvConfig = field(default_factory=LLMEnvConfig)
    boards: LLMBoardsConfig = field(default_factory=LLMBoardsConfig)
    prompt: LLMPromptConfig = field(default_factory=LLMPromptConfig)
    rollout: LLMRolloutConfig = field(default_factory=LLMRolloutConfig)
    vllm: LLMVLLMConfig = field(default_factory=LLMVLLMConfig)
    api: LLMAPIConfig = field(default_factory=LLMAPIConfig)


# ============================================================================
# LLM (verl) training orchestration
# ============================================================================
#
# The pcbworld-specific slice of the verl Hydra config (the ``env.pcbworld.*``
# block + the ``env.*`` knobs pcbworld reads). Distinct from LLMConfig, which is
# the LLM *eval* program's argparse defaults: this is the LLM *training* program
# (verl) — there is no argparse, the verl config supplies per-run overrides and
# this schema is the single defaults source for the ``config.env.pcbworld.*``
# fallbacks read by the verl-agent patch + methods/llm_agent/training/manager.py. Flat, to
# mirror the flat ``env.pcbworld`` namespace.

_LLM_TRAIN = _DEFAULTS["llm_train"]


@dataclass(frozen=True)
class LLMTrainConfig:
    """pcbworld LLM(verl)-training config — the ``env.pcbworld.*`` slice.

    Built from a verl omegaconf config with :meth:`from_verl_config`; :attr:`env` is
    the whole :class:`EnvConfig` the workers are built with (the shared ``env:``
    defaults + the ``llm_train:`` EnvConfig keys + the run's overrides).
    Defaults = configs/llm.yaml (llm_train:).
    """

    env: EnvConfig = field(default_factory=lambda: _llm_env_config(
        EnvConfig(), _LLM_TRAIN, LLMTrainConfig, "configs/llm.yaml llm_train"))
    state_format: str = knob(_LLM_TRAIN["state_format"], "experiment")
    # prompt (methods/llm_agent/training/manager.py)
    prompt_version: str = knob(_LLM_TRAIN["prompt_version"], "experiment")
    history_length: int = knob(_LLM_TRAIN["history_length"], "experiment")
    # board scheduling — train
    boards_order: str = knob(_LLM_TRAIN["boards_order"], "experiment")
    boards_json: str | None = knob(_LLM_TRAIN["boards_json"], "experiment")
    boards_difficulty: str = knob(_LLM_TRAIN["boards_difficulty"], "experiment")
    boards_split: str = knob(_LLM_TRAIN["boards_split"], "experiment")
    # board scheduling — val (val_boards_split None = inherit boards_split)
    val_boards_order: str = knob(_LLM_TRAIN["val_boards_order"], "experiment")
    val_boards_split: str | None = knob(_LLM_TRAIN["val_boards_split"], "experiment")
    # episode scoring + guided decoding
    score_train_episodes: bool = knob(_LLM_TRAIN["score_train_episodes"], "experiment")
    guided_decoding_grammar: str | None = knob(_LLM_TRAIN["guided_decoding_grammar"], "experiment")
    # val episode-end DRC scoring config — DELIBERATELY sourced from the shared
    # eval config (RLEvalConfig / configs/rl.yaml rl_eval: = DEFAULTS), NOT a separate
    # llm_train default, so LLM val and the RL inline-eval path score with the
    # SAME ruler (W5 unification). Override per-run via env.pcbworld.eval_*.
    eval_reward_config: str = knob(DEFAULTS.reward_config, "experiment")
    # None = the env corner_mode, as eval does; an explicit angle may only restate it.
    eval_check_angle: int | None = knob(None, "experiment")
    # misc
    seed: int = knob(_LLM_TRAIN["seed"], "experiment")
    # required runtime input (env.pcbworld.board_path); no meaningful default
    board_path: str | None = knob(_LLM_TRAIN["board_path"], "experiment")

    @property
    def resolved_val_boards_split(self) -> str:
        """Val split, inheriting the train ``boards_split`` when unset."""
        return self.val_boards_split if self.val_boards_split is not None else self.boards_split

    @staticmethod
    def _cfg_get(node: Any, key: str, default: Any) -> Any:
        """Read ``key`` from an OmegaConf node (``.get``) or plain object/dict.

        Mirrors ``methods.llm_agent.training.manager.KiCadLLMRolloutManager._cfg_get`` so the
        adapter works with omegaconf configs AND test doubles (SimpleNamespace /
        dict) alike.
        """
        if node is None:
            return default
        if hasattr(node, "get"):
            try:
                return node.get(key, default)
            except TypeError:
                pass
        return getattr(node, key, default)

    @classmethod
    def from_verl_config(cls, config: Any) -> "LLMTrainConfig":
        """Build from a verl Hydra/omegaconf config.

        Every ``env.pcbworld.<key>`` is a field of this class or a flat EnvConfig key
        (:meth:`EnvConfig.with_flat` — e.g. ``masking_rule``, ``via_penalty``); any
        other key raises ``ValueError``. ``max_steps`` / ``history_length`` / ``seed``
        come from the general ``env.*`` block (where pcbworld consumes them) — under
        ``env.pcbworld`` they raise. A key the config omits keeps the YAML default
        (``cls()`` = configs/llm.yaml (llm_train:)).
        """
        d = cls()  # shared YAML defaults
        g = cls._cfg_get
        env = g(config, "env", None)
        cad = g(env, "pcbworld", None)
        cad = {} if cad is None else dict(cad.items() if hasattr(cad, "items") else vars(cad).items())
        misplaced = sorted(set(cad) & {"env", "max_steps", "history_length", "seed"})
        if misplaced:
            raise ValueError(f"env.pcbworld: {misplaced} not allowed — max_steps / "
                             "history_length / seed are read from env.max_steps / "
                             "env.history_length / env.seed")
        return replace(
            d,
            env=_llm_env_config(d.env, {**cad, "max_steps": g(env, "max_steps", d.env.max_steps)},
                                cls, "env.pcbworld"),
            history_length=int(g(env, "history_length", d.history_length)),
            seed=int(g(env, "seed", d.seed)),
            **{k: v for k, v in cad.items() if k not in _ENV_FLAT_KEYS},
        )
