"""Typed config schemas — the single place to read/edit run config + defaults.

One module holding the canonical, sink-agnostic config dataclasses shared across
``envs`` / ``training`` / ``eval``. argparse Namespaces, checkpoint ``args``
dicts, and YAML are *adapters* that load into these — the field names and
defaults are defined once, here, not re-declared per entry point.

Lives in the existing ``configs`` package (alongside the YAML files + loaders in
``configs/__init__.py``) so there is one ``configs/`` home, not a separate
``config/`` vs ``configs/`` split.

Contents:
  * :func:`corner_mode_to_code` — 45/90 sugar -> engine corner code.
  * :class:`EnvConfig`   — env-core (``PCBWorld`` keyword surface), shared
    RL + LLM; ``to_env_kwargs()``.
  * :class:`RLEnvConfig` — env-core + RL-wrapper knobs (the ``make_decoder_env``
    surface); ``to_pool_kwargs()`` + ``from_namespace`` / ``from_checkpoint``
    adapters.
  * :class:`RLPolicyConfig` — ``KiCadRLModel`` construction; ``build()`` +
    ``from_namespace`` / ``from_checkpoint``.

Scope note — ``corner_mode`` vs ``check_angle``: ``corner_mode`` here is the
**engine** routing-geometry mode (canonical = engine code ``0..3``;
``0=MITERED_45``, ``2=MITERED_90``). ``check_angle`` (the post-hoc track-angle DRC
check in :func:`eval.metrics.compute_metrics`) is a *scoring* concern, NOT an env
construction param, and is deliberately not modelled here.

Heavy deps (``torch`` / ``KiCadRLModel``) are imported lazily inside
:meth:`RLPolicyConfig.build`, so importing this module stays torch-free.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from configs.loader import load_config

# Default *values* live in YAML (the single place to read/edit them); the
# dataclasses below define the *schema* (fields, types, adapters) and pull their
# field defaults from here. Edit configs/defaults/*.yaml to change a default.
_DEFAULTS_DIR = Path(__file__).resolve().parent.parent / "defaults"
_ENV = load_config(_DEFAULTS_DIR / "env.yaml")
_REWARD = _ENV["reward"]
_POLICY = load_config(_DEFAULTS_DIR / "rl_policy.yaml")


def corner_mode_to_code(value: Any) -> int:
    """Map a user-facing corner-mode value to the engine's int code.

    ``45 -> 0`` (MITERED_45), ``90 -> 2`` (MITERED_90); any other int passes
    through (a checkpoint may store the raw engine code ``0..3`` directly), and
    an unparsable value falls back to ``0``.

    This is the lenient **checkpoint-side** resolver — a stored value may be
    legacy 45/90 sugar *or* a raw engine code. The strict user-facing *degrees*
    entry (CLI ``--corner-mode``) is :func:`corner_deg_to_code`.
    """
    try:
        mode = int(value)
    except (TypeError, ValueError):
        return 0
    if mode == 45:
        return 0
    if mode == 90:
        return 2
    return mode


def corner_deg_to_code(deg: int) -> int:
    """User-facing corner *angle* (degrees) → engine corner code, STRICT.

    The two documented angles only: ``45 -> 0`` (MITERED_45), ``90 -> 2``
    (MITERED_90). This is the degrees entry point (CLI ``--corner-mode`` already
    enforces ``choices=[45, 90]``); anything else — a raw engine code, another
    angle, or a wrong type — is a caller bug, so **assert** rather than silently
    coercing it (the old inline ``0 if ==45 else 2`` turned any non-45 value,
    including raw codes 0/1/3, into 2). Raw engine codes from a checkpoint go
    through the lenient :func:`corner_mode_to_code`; the final code is validated
    against ``{0,1,2,3}`` at ``PCBWorld`` construction.
    """
    assert deg in (45, 90), f"corner angle must be 45 or 90 degrees, got {deg!r}"
    return 0 if deg == 45 else 2


# Env/reward/policy param keys that were RENAMED or REMOVED. A checkpoint's
# stored training args (``vars(args)``) may still carry the old name; the current
# ``from_checkpoint`` no longer reads it, so its value would be silently dropped
# and eval would fall back to the current default. We can't error (old
# checkpoints must stay loadable), but we surface the drift with a warning
# instead of letting it vanish. Register the OLD key here whenever you rename or
# remove an env/reward/policy param (same discipline as ``DEAD_PATHS`` in
# tools/docs/check_docs.py). Empty = no known drift yet — training-only args
# (lr, batch_size, …) are NOT listed here and never warn.
_RENAMED_CKPT_KEYS: dict[str, str] = {
    # "old_key": "renamed to <new_key>" | "removed (since v<X.Y>)",
    "allow_incomplete_net_end": "removed (v0.11.26 — the net_end precondition "
                                "is owned by the masking rule alone)",
}


def _warn_renamed_ckpt_keys(ckpt_args: dict[str, Any]) -> None:
    """Warn if a checkpoint carries a renamed/removed param whose value is now
    dropped. See ``_RENAMED_CKPT_KEYS``."""
    stale = [k for k in _RENAMED_CKPT_KEYS if k in ckpt_args]
    if stale:
        import warnings
        details = "; ".join(f"{k} ({_RENAMED_CKPT_KEYS[k]})" for k in stale)
        warnings.warn(
            f"checkpoint args carry renamed/removed params whose stored values "
            f"are ignored at eval — falling back to current defaults: {details}",
            stacklevel=3,
        )


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

    via_penalty: float | None = field(default=_REWARD["via_penalty"], metadata={
        "help": "Override potential's via_penalty (Φ -= w * via_count). "
                "None = use reward-rule YAML default (0)."})
    wirelength_penalty: float | None = field(default=_REWARD["wirelength_penalty"], metadata={
        "help": "Override potential's wirelength_penalty (Φ -= w * wirelength_mm). "
                "None = YAML default."})
    drc_penalty: float | None = field(default=_REWARD["drc_penalty"], metadata={
        "help": "Override potential's drc_penalty (linear-shape DRC weight). "
                "None = YAML default."})
    drc_log_scale: float | None = field(default=_REWARD["drc_log_scale"], metadata={
        "help": "Override drc_log_scale for log_per_net DRC shape (per-net depth "
                "term). None = YAML default."})
    drc_log_agg_scale: float | None = field(default=_REWARD["drc_log_agg_scale"], metadata={
        "help": "Override drc_log_agg_scale for log_per_net DRC shape (aggregate "
                "breadth term; 0 disables). None = YAML default."})
    drc_log_offset: float | None = field(default=_REWARD["drc_log_offset"], metadata={
        "help": "Override drc_log_offset (log-curve knee) for log_per_net DRC "
                "shape. None = YAML default."})
    reward_step_penalty: float | None = field(default=_REWARD["reward_step_penalty"], metadata={
        "help": "Override potential step_penalty. Used by the Jumanji Connector "
                "dense reward to scale grid sizes."})
    wire_via_emission: str | None = field(default=_REWARD["wire_via_emission"], metadata={
        "choices": ["per_step", "on_net_end"],
        "help": "When to emit wire + via penalties in per_step mode: 'per_step' "
                "(every env step, default) or 'on_net_end' (accumulated and "
                "emitted only on ACT_NET_END steps). Terminal mode unaffected."})


@dataclass(frozen=True)
class EnvConfig:
    """Canonical env-core config — the ``PCBWorld`` keyword surface.

    The reward-rule penalty overrides are grouped under :attr:`reward`
    (:class:`RewardOverrides`); :meth:`to_env_kwargs` flattens them back so the
    KiCad env still receives a flat kwargs dict.
    """

    max_steps: int = field(default=_ENV["max_steps"], metadata={
        "help": "Per-episode time-limit (env truncation cap)"})
    masking_rule: str = _ENV["masking_rule"]
    reward_rule: str = _ENV["reward_rule"]
    reward_noise_std: float = field(default=_ENV["reward_noise_std"], metadata={
        "help": "Gaussian noise stddev applied to env reward (default: 0)."})
    # emit_drc_tokens / corner_mode: variables live here (visible), but their CLI
    # forms are bespoke — emit_drc_tokens is the inverted --no-drc-tokens pair and
    # corner_mode is the engine code behind the 45/90 --corner-mode sugar — so the
    # generator skips them and each entrypoint adds the explicit flag.
    emit_drc_tokens: bool = field(default=_ENV["emit_drc_tokens"], metadata={"cli_skip": True})
    corner_mode: int = field(default=_ENV["corner_mode"], metadata={"cli_skip": True})  # engine code 0..3 (see module docstring)
    use_yaml_drc_fallback: bool = field(default=_ENV["use_yaml_drc_fallback"], metadata={
        "help": "If a board has neither a companion .kicad_pro nor legacy "
                "(setup ...) tokens, substitute the YAML at --drc-config-path's "
                "global minima into BDS (with a UserWarning). Off by default; "
                "when on, --drc-config-path is required."})
    drc_config_path: str | None = field(default=_ENV["drc_config_path"], metadata={
        "help": "YAML DRC config used together with --use-yaml-drc-fallback. "
                "Required when that flag is set — there is no implicit default "
                "(e.g. configs/drc/default.yaml)."})
    engine_seed: int = field(default=_ENV["engine_seed"], metadata={
        "help": "Seed for KiCad's KIID/UUID generator, set once at engine init. "
                "Makes routing + UUID-keyed DRC reproducible across runs/processes "
                "for a fixed action sequence (default 77). Global generator → one "
                "env per process for clean determinism."})
    shove_iter_limit: int = field(default=_ENV["shove_iter_limit"], metadata={
        "help": "Max PNS shove iterations (default 250). The shove loop is bounded "
                "by this count rather than by a wallclock timeout, so runs "
                "reproduce."})
    followbranch_iter_limit: int = field(default=_ENV["followbranch_iter_limit"], metadata={
        "help": "Max TOPOLOGY::followBranch DFS pops (default 1,000,000). Same rule "
                "as the shove bound: an iteration count, not a wallclock timeout."})
    reject_if_stuck: bool = field(default=_ENV["reject_if_stuck"], metadata={
        "cli_skip": True,
        "help": "make_line aborts (draws nothing, action fails) when the walkaround "
                "cannot reach the target, instead of committing a partial dangling stub "
                "(default true). false = legacy commit-partial behavior."})
    simplify_outline: bool = field(default=_ENV["simplify_outline"], metadata={
        "help": "Rewrite tessellated micro-segment outline chains (Edge.Cuts/"
                "Margin) into native arcs/merged lines at board load "
                "(pcb_world/engine/outline_simplify.py). Kills the PNS walkaround "
                "cluster blowup on baked-curve boards. Off by default; changes obs "
                "EDGE token counts, so keep one setting within a campaign."})
    obs_format: str = field(default=_ENV["obs_format"], metadata={
        "choices": ["json", "indexed"],
        "cli_skip": True,  # Not exposed on the public CLI — the RL training
        # path is fixed to indexed; LLM/eval uses json. scripts/profile.py
        # injects args.obs_format programmatically, for diagnostic A/B only.
        "help": "Observation format the env emits. 'json' = legacy nested "
                "dict (LLM/eval path); 'indexed' = indexed_v1 array tables "
                "(pcb_world/core/indexed_obs.py) — fixed value for the RL "
                "training path, tokens bit-identical."})
    outline_obs: str = field(default=_ENV["outline_obs"], metadata={
        "choices": ["poly16", "tess", "arc"],
        "help": "Edge.Cuts representation in the obs boardlines. 'tess' = "
                "C++ error-bounded tessellation into segments (0.005mm, "
                "the default); 'poly16' = fixed 16-per-90deg re-tessellation "
                "(coarse A/B reference); 'arc' = one entry per "
                "arc/circle carrying the on-arc midpoint (requires an arc-capable "
                "policy — legacy 2-point checkpoints refuse arc obs)."})
    # CLI form is owned by the policy group's --action-history-len (one knob
    # drives both env window and tokenizer K); from_namespace copies it here.
    action_history_len: int = field(default=_ENV["action_history_len"], metadata={
        "cli_skip": True,
        "help": "obs['action_history'] window: how many recent action records "
                "(type/pointer/net/success, newest first) the env keeps and "
                "emits (default 1 = the single-step prev-action window; history "
                "ablations raise it). K=1 checkpoints consume entry 0 only."})
    net_constraint_obs: bool = field(default=_ENV["net_constraint_obs"], metadata={
        "help": "Per-net DRC constraint observation: every board_static net "
                "carries its resolved netclass values (track_width/clearance/"
                "via_diameter/via_drill; KiCad inherit→Default fallback, BDS-min "
                "clamped — identical to the engine push on net_select), feeding "
                "the NET-token tw/cl/vd channels (constant 0 when off). "
                "Obs-drift: retrain; the ckpt stores the setting so eval "
                "follows automatically."})
    # CLI form is the bespoke two-float --keep-routing-fraction LO HI (args.py);
    # the generator can't express nargs=2.
    keep_routing_fraction: list[float] | None = field(
        default=_ENV["keep_routing_fraction"], metadata={
            "cli_skip": True,
            "help": "Keep-routing augmentation (train-only): (lo, hi) — each "
                    "board load samples f~U[lo,hi] and keeps that fraction of "
                    "the nets' file routing as the episode's initial state "
                    "(uniform random nets; rest re-routed from scratch). "
                    "Requires boards whose file carries complete routing. "
                    "None = off; eval always off."})
    reward: RewardOverrides = field(default_factory=RewardOverrides)

    def to_env_kwargs(self) -> dict[str, Any]:
        """Flat kwargs for ``PCBWorld(board_path=..., **cfg.to_env_kwargs())``.

        The ``reward`` sub-object is flattened back to top-level penalty keys so
        the KiCad env keeps its flat keyword surface.
        """
        d = asdict(self)
        d.update(d.pop("reward"))
        return d


@dataclass(frozen=True)
class RLEnvConfig:
    """Env-core + the RL-wrapper knobs — the ``make_decoder_env`` surface."""

    env: EnvConfig = field(default_factory=EnvConfig)
    force_walkaround: bool = field(default=False, metadata={
        "help": "Override routing_mode to 2 (Walkaround) for all "
                "make_line/make_via/finish actions, matching the MLP trainer's "
                "hardcoded mode. Use for fair comparison."})
    # mask_start_point: variable lives here (visible) but its CLI form is the
    # inverted --no-mask-start-point, so the generator skips it.
    mask_start_point: bool = field(default=True, metadata={"cli_skip": True})
    slot_perm: bool = field(default=False, metadata={
        "help": "v1: per-episode random permutation of the net slot embedding "
                "table (net-exchangeability augmentation). v0 (default) uses "
                "identity mapping sorted-net-k -> slot-k."})
    directional_candidates: str | None = field(default=None, metadata={
        "help": "Directional candidate generation mode: a preset name "
                "(e.g. 'multi_resolution' = 8 dirs × 0.2/1.0/5.0/25.0mm; see "
                "candidate_pool.DIRECTIONAL_DISTANCE_PRESETS) or 'grid<N>' "
                "(1-layer Grid mode, e.g. grid200). Default (None) uses "
                "the 8-direction 0.5mm path."})
    # connectivity_filter: variable lives here (visible) but its CLI form is the
    # --connectivity-filter / --no-connectivity-filter pair, so the generator
    # skips it. It trims already-connected existing copper from the candidate
    # pool = it changes the ACTION SPACE, so train and eval must agree.
    connectivity_filter: bool = field(default=True, metadata={"cli_skip": True})
    # Pad-graze guard width (mm; 0 = off). Drops DIRECTIONAL
    # candidates landing in the sliver annulus outside a same-net pad (copper
    # the track cleaner deletes) — changes the CANDIDATE SET, so like
    # connectivity_filter it round-trips through checkpoints. CLI form is the
    # bespoke --pad-graze-margin-mm in the args module (richer help text), so
    # the generator skips it; that flag's default reads THIS field.
    pad_graze_margin_mm: float = field(default=0.0, metadata={"cli_skip": True})
    # Off-board pointer mask (act time, every cand row): blocks DIRECTIONAL
    # candidates whose (x, y) falls outside the board bbox — the long rungs of
    # a ladder preset (mres8: 25 / 50 mm) overshoot most boards. Leaves the
    # candidate SET (pool / pointer index space) untouched, but a policy trained
    # with it never learned to avoid off-board targets, so it round-trips
    # through checkpoints like the other wrapper knobs. Default off = existing
    # checkpoints keep their exact behaviour.
    offboard_mask: bool = field(default=False, metadata={
        "help": "Hard-mask DIRECTIONAL candidates whose (x, y) falls outside "
                "the board bbox at act time (every cand row: neither make_line "
                "nor make_via can target them). Off by default; stored in the "
                "checkpoint and restored when it is evaluated."})
    # Per-episode augmentation toggles (5 independent boolean axes). Consumed by
    # the trainer via ``args.aug_*`` and passed to the RL wrapper factory; eval
    # never enables them (RLEnvConfig stays default-False, so to_pool_kwargs need
    # not surface them — like reward_noise_std).
    aug_bbox_shifted: bool = field(default=False, metadata={
        "help": "Per-episode per-axis virtual scale of the board edges around a "
                "random interior centre (scale_x, scale_y ~ U[0.7, 1.3]). Pads "
                "stay physically fixed."})
    aug_flip: bool = field(default=False, metadata={
        "help": "Per-episode independent x/y sign reflection (prob 0.5 each)."})
    aug_rotate: bool = field(default=False, metadata={
        "help": "Per-episode x/y axis swap (prob 0.5)."})
    aug_trans: bool = field(default=False, metadata={
        "help": "Per-episode feature-space translation (nn_dx, nn_dy ~ U[-0.2, 0.2])."})
    aug_zoom: bool = field(default=False, metadata={
        "help": "Per-episode feature-space uniform scale jitter "
                "(nn_zoom ~ U[0.9, 1.1]): the whole normalized scene — "
                "positions and dims together — is zoomed around the bbox "
                "centre, so the board long axis spans [-zoom, +zoom] instead "
                "of exactly [-1, 1]."})

    def to_pool_kwargs(self) -> dict[str, Any]:
        """Kwargs for ``make_decoder_env`` / ``make_decoder_env_pool``.

        The field set is exactly the factory's keyword surface; it carries no
        ``reward_noise_std`` — the pool path leaves that at the factory default.
        """
        e = self.env
        r = e.reward
        return {
            "max_steps": e.max_steps,
            "masking_rule": e.masking_rule,
            "reward_rule": e.reward_rule,
            "force_walkaround": self.force_walkaround,
            "mask_start_point": self.mask_start_point,
            "slot_perm": self.slot_perm,
            "emit_drc_tokens": e.emit_drc_tokens,
            "via_penalty": r.via_penalty,
            "wirelength_penalty": r.wirelength_penalty,
            "drc_penalty": r.drc_penalty,
            "drc_log_scale": r.drc_log_scale,
            "drc_log_agg_scale": r.drc_log_agg_scale,
            "drc_log_offset": r.drc_log_offset,
            "reward_step_penalty": r.reward_step_penalty,
            "wire_via_emission": r.wire_via_emission,
            "corner_mode": e.corner_mode,
            "directional_candidates": self.directional_candidates,
            "connectivity_filter": self.connectivity_filter,
            "pad_graze_margin_mm": self.pad_graze_margin_mm,
            "offboard_mask": self.offboard_mask,
            "use_yaml_drc_fallback": e.use_yaml_drc_fallback,
            "drc_config_path": e.drc_config_path,
            "simplify_outline": e.simplify_outline,
            "obs_format": e.obs_format,
            "outline_obs": e.outline_obs,
            "action_history_len": e.action_history_len,
            "net_constraint_obs": e.net_constraint_obs,
            "keep_routing_fraction": e.keep_routing_fraction,
        }

    @classmethod
    def from_namespace(cls, args: Any) -> "RLEnvConfig":
        """Build from a training-time argparse Namespace.

        ``--corner-mode`` is the CLI's 45/90 sugar (``45 -> 0``, anything else
        -> ``2``); ``no_mask_start_point`` / ``no_drc_tokens`` invert to their
        positive fields. ``reward_noise_std`` is intentionally left at the eval
        default of 0.0.
        """
        env = EnvConfig(
            max_steps=int(args.max_steps),
            masking_rule=str(args.masking_rule),
            reward_rule=str(args.reward_rule),
            emit_drc_tokens=not bool(getattr(args, "no_drc_tokens", False)),
            corner_mode=corner_deg_to_code(getattr(args, "corner_mode", 45)),
            use_yaml_drc_fallback=bool(getattr(args, "use_yaml_drc_fallback", False)),
            drc_config_path=getattr(args, "drc_config_path", None),
            simplify_outline=bool(getattr(args, "simplify_outline", False)),
            # RL training namespace default = indexed (not on the CLI;
            # profile.py injects args.obs_format programmatically, A/B only).
            obs_format=str(getattr(args, "obs_format", "indexed")),
            outline_obs=str(getattr(args, "outline_obs", _ENV["outline_obs"])),
            # Keep the env's recorded window in lockstep with the model's
            # --action-history-len so the tokenizer never sees a shorter
            # history than its K (which would silently pad with sentinels).
            action_history_len=int(getattr(
                args, "action_history_len", _ENV["action_history_len"],
            )),
            net_constraint_obs=bool(getattr(
                args, "net_constraint_obs", _ENV["net_constraint_obs"],
            )),
            keep_routing_fraction=getattr(
                args, "keep_routing_fraction", _ENV["keep_routing_fraction"],
            ),
            reward=RewardOverrides(
                via_penalty=getattr(args, "via_penalty", None),
                wirelength_penalty=getattr(args, "wirelength_penalty", None),
                drc_penalty=getattr(args, "drc_penalty", None),
                drc_log_scale=getattr(args, "drc_log_scale", None),
                drc_log_agg_scale=getattr(args, "drc_log_agg_scale", None),
                drc_log_offset=getattr(args, "drc_log_offset", None),
                reward_step_penalty=getattr(args, "reward_step_penalty", None),
                wire_via_emission=getattr(args, "wire_via_emission", None),
            ),
        )
        return cls(
            env=env,
            force_walkaround=bool(getattr(args, "force_walkaround", False)),
            mask_start_point=not bool(getattr(args, "no_mask_start_point", False)),
            slot_perm=bool(getattr(args, "slot_perm", False)),
            directional_candidates=getattr(args, "directional_candidates", None),
            connectivity_filter=bool(getattr(args, "connectivity_filter", True)),
            pad_graze_margin_mm=float(getattr(args, "pad_graze_margin_mm", 0.0)),
            offboard_mask=bool(getattr(args, "offboard_mask", False)),
        )

    @classmethod
    def from_checkpoint(cls, ckpt_args: dict[str, Any], max_steps: int) -> "RLEnvConfig":
        """Build from a checkpoint's stored ``args`` dict (eval path).

        ``slot_perm`` is forced ``False``: it is a TRAIN-only augmentation
        (random per-rollout net-slot permutation); applying it at eval would make
        eval nondeterministic. Every eval entry point uses ``slot_perm=False``.

        Every key falls back to the shared YAML default (``EnvConfig()`` =
        configs/defaults/env.yaml) when the checkpoint lacks it, so an old
        checkpoint evals with the same defaults training would have used.
        """
        _warn_renamed_ckpt_keys(ckpt_args)
        d = EnvConfig()  # shared YAML defaults (= training defaults)
        dr = d.reward
        env = EnvConfig(
            max_steps=max_steps,
            masking_rule=str(ckpt_args.get("masking_rule", d.masking_rule)),
            reward_rule=str(ckpt_args.get("reward_rule", d.reward_rule)),
            emit_drc_tokens=not bool(
                ckpt_args.get("no_drc_tokens", not d.emit_drc_tokens)
            ),
            corner_mode=corner_mode_to_code(ckpt_args.get("corner_mode", d.corner_mode)),
            use_yaml_drc_fallback=bool(
                ckpt_args.get("use_yaml_drc_fallback", d.use_yaml_drc_fallback)
            ),
            drc_config_path=ckpt_args.get("drc_config_path", d.drc_config_path),
            simplify_outline=bool(
                ckpt_args.get("simplify_outline", d.simplify_outline)
            ),
            # Fallback is the literal "tess", NOT the YAML default: every
            # checkpoint saved before the flag existed was trained on tessellated
            # outlines, and a later YAML default flip must not silently change
            # how old checkpoints eval.
            outline_obs=str(ckpt_args.get("outline_obs", "tess")),
            # Old checkpoints (no key) eval with the YAML default window — the
            # legacy K=1 policy reads entry 0 only, extra entries are inert.
            action_history_len=int(
                ckpt_args.get("action_history_len", d.action_history_len)
            ),
            # Fallback is the literal False, NOT the YAML default: every
            # checkpoint saved before the knob existed was trained on all-zero
            # NET constraint channels, and a later YAML default flip must not
            # silently change how old checkpoints eval (obs-drift).
            net_constraint_obs=bool(
                ckpt_args.get("net_constraint_obs", False)
            ),
            # TRAIN-only augmentation (like slot_perm): eval always scores the
            # full from-scratch board, so the ckpt's stored value is never
            # applied — pinned None rather than YAML default on purpose.
            keep_routing_fraction=None,
            reward=RewardOverrides(
                via_penalty=ckpt_args.get("via_penalty", dr.via_penalty),
                wirelength_penalty=ckpt_args.get("wirelength_penalty", dr.wirelength_penalty),
                drc_penalty=ckpt_args.get("drc_penalty", dr.drc_penalty),
                drc_log_scale=ckpt_args.get("drc_log_scale", dr.drc_log_scale),
                drc_log_agg_scale=ckpt_args.get("drc_log_agg_scale", dr.drc_log_agg_scale),
                drc_log_offset=ckpt_args.get("drc_log_offset", dr.drc_log_offset),
                reward_step_penalty=ckpt_args.get("reward_step_penalty", dr.reward_step_penalty),
                wire_via_emission=ckpt_args.get("wire_via_emission", dr.wire_via_emission),
            ),
        )
        return cls(
            env=env,
            force_walkaround=bool(ckpt_args.get("force_walkaround", False)),
            mask_start_point=not bool(ckpt_args.get("no_mask_start_point", False)),
            slot_perm=False,
            # A checkpoint that carries the int directional_grid_size maps to
            # the equivalent "grid<N>" mode, so a grid-mode checkpoint evaluates
            # with the candidate set it was trained on.
            directional_candidates=(
                ckpt_args.get("directional_candidates")
                or (f"grid{int(g)}"
                    if (g := ckpt_args.get("directional_grid_size")) is not None
                    else None)
            ),
            # Fallback is False, NOT the flag's default True (same rule as
            # outline_obs above): the filter changes which existing-copper
            # candidates exist, i.e. the pointer index space. A checkpoint saved
            # before the flag existed was trained with every candidate
            # selectable, so evaluating it with the filter ON would hand the
            # policy a candidate set it never saw.
            connectivity_filter=bool(ckpt_args.get("connectivity_filter", False)),
            # Same candidate-set rule: pre-knob checkpoints trained without the
            # graze guard — fallback 0.0 (off).
            pad_graze_margin_mm=float(ckpt_args.get("pad_graze_margin_mm", 0.0)),
            # Pre-knob checkpoints trained with every directional candidate
            # selectable — fallback False keeps their behaviour exact.
            offboard_mask=bool(ckpt_args.get("offboard_mask", False)),
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

    d_model: int = _POLICY["d_model"]
    n_heads: int = _POLICY["n_heads"]
    n_layers: int = field(default=_POLICY["n_layers"], metadata={
        "help": "Transformer layer count (4 for small boards; increase for larger)"})
    d_ff: int = _POLICY["d_ff"]
    max_seq_len: int = _POLICY["max_seq_len"]
    n_freq: int = _POLICY["n_freq"]
    n_max_slots: int = field(default=_POLICY["n_max_slots"], metadata={
        "help": "Slot-table size = per-board max net count the tokenizer "
                "accepts (rows of slot_emb_table — checkpoints record it; "
                "old ckpts without the key load as 64). Default 512 gives "
                "headroom for real-board sets (d3b max 109 nets). "
                "Incompatible with --slot-perm unless 64."})
    use_critic: bool = field(default=_POLICY["use_critic"], metadata={"cli_skip": True})
    detach_critic: bool = field(default=_POLICY["detach_critic"], metadata={"cli_skip": True})
    coord_encoding: str = field(default=_POLICY["coord_encoding"], metadata={
        "choices": ["fourier", "mlp", "linear"],
        "help": "Continuous-value encoding: 'fourier' (default, sin/cos "
                "multi-freq), 'mlp' (raw values through a 2-layer MLP with "
                "LayerNorm), or 'linear'."})
    mlp_hidden: int = field(default=_POLICY["mlp_hidden"], metadata={
        "help": "Hidden size of the coord MLP (only used when --coord-encoding mlp)."})
    disable_slot_emb: bool = field(default=_POLICY["disable_slot_emb"], metadata={"cli_skip": True})
    policy_net_select: bool = field(default=_POLICY["policy_net_select"], metadata={"cli_skip": True})
    same_net_bias: bool = field(default=_POLICY["same_net_bias"], metadata={"cli_skip": True})
    legacy_pad_layer_encoding: bool = field(default=_POLICY["legacy_pad_layer_encoding"],
                                            metadata={"cli_skip": True})
    legacy_net_encoding: bool = field(default=_POLICY["legacy_net_encoding"],
                                      metadata={"cli_skip": True})
    legacy_edge_encoding: bool = field(default=_POLICY["legacy_edge_encoding"],
                                       metadata={"cli_skip": True})
    time_feature: str = field(default=_POLICY["time_feature"], metadata={
        "choices": ["step_ratio", "log_remaining", "sin_remaining", "none"],
        "help": "HEAD-token time scalar: 'step_ratio' (step/max_steps, legacy), "
                "'log_remaining' (log1p(steps_remaining)/log1p(cap), "
                "max_steps-invariant across per-board step budgets), "
                "'sin_remaining' (steps_remaining/cap through a dedicated "
                "sinusoidal ladder anchored to step units — resolves ±1 step "
                "at any horizon, transformer-PE style), or 'none' (constant 0 "
                "— time-blind ablation; the policy cannot observe episode "
                "progress. All modes share one Fourier slot with zero new "
                "weights, so checkpoints stay weight-compatible)."})
    time_feature_cap: int = field(default=_POLICY["time_feature_cap"], metadata={
        "help": "Fixed normalization ceiling for --time-feature "
                "log_remaining/sin_remaining (largest plausible max_steps; "
                "for sin_remaining also the longest ladder period)."})
    action_history_len: int = field(default=_POLICY["action_history_len"], metadata={
        "help": "ACTION_HISTORY window K: recent action entries tokenized per "
                "obs (3 tokens each, newest first; idle-sentinel padded). Age "
                "uses Fourier(age/MAX_HISTORY)->proj — no K-shaped weights, so "
                "checkpoints stay load-compatible across K."})
    legacy_action_history: bool = field(default=_POLICY["legacy_action_history"],
                                        metadata={"cli_skip": True})
    obstacle_obs: bool = field(default=_POLICY["obstacle_obs"], metadata={
        "help": "Emit OBSTACLE tokens for netless immovable blockers (NPTH "
                "mounting holes/slots + net-less NC pads; rule-area keepout "
                "zones stay engine-only). Off (default) keeps the token "
                "stream byte-identical to pre-knob checkpoints."})
    shape_obs: bool = field(default=_POLICY["shape_obs"], metadata={
        "help": "Add the boundary-shape channel (6-bucket categorical: "
                "rect/roundrect/circle/oval/other/unknown) to PAD and "
                "OBSTACLE tokens via an additive shape_embed. Off (default) "
                "adds no weights — old checkpoints round-trip unchanged."})

    def build(self, device: Any | None = None) -> Any:
        """Construct the policy (optionally moved to ``device``)."""
        from methods.rl_agent.models.v1.net import KiCadRLModel

        policy = KiCadRLModel(**asdict(self))
        return policy if device is None else policy.to(device)

    @classmethod
    def from_namespace(cls, args: Any, *, use_critic: bool) -> "RLPolicyConfig":
        """Build from a training-time argparse Namespace.

        ``use_critic`` is supplied by the trainer (per-algorithm class attribute,
        PPO=True / GRPO=False — not an argparse flag). The training path never sets
        ``legacy_pad_layer_encoding`` (a checkpoint-compat concern); it stays False.
        """
        return cls(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            n_freq=args.n_freq,
            use_critic=use_critic,
            detach_critic=getattr(args, "detach_critic", False),
            coord_encoding=args.coord_encoding,
            mlp_hidden=args.mlp_hidden,
            disable_slot_emb=args.disable_slot_emb,
            policy_net_select=args.policy_net_select,
            same_net_bias=args.same_net_bias,
            time_feature=args.time_feature,
            time_feature_cap=args.time_feature_cap,
            n_max_slots=getattr(args, "n_max_slots", cls().n_max_slots),
            action_history_len=getattr(
                args, "action_history_len", cls().action_history_len,
            ),
            obstacle_obs=getattr(args, "obstacle_obs", cls().obstacle_obs),
            shape_obs=getattr(args, "shape_obs", cls().shape_obs),
            legacy_edge_encoding=getattr(
                args, "legacy_edge_encoding", cls().legacy_edge_encoding),
        )

    @classmethod
    def from_checkpoint(cls, args: dict[str, Any]) -> "RLPolicyConfig":
        """Build from a checkpoint's ``args`` dict (eval path).

        ``use_critic`` / ``legacy_pad_layer_encoding`` may have been overwritten on
        ``args`` by the loader (from the saved ``policy_state_dict`` shape) before
        this is called; they are read back here.

        Every key falls back to the shared YAML default (``cls()`` =
        configs/defaults/rl_policy.yaml) when the checkpoint lacks it.
        """
        d = cls()  # shared YAML defaults (= training defaults)
        return cls(
            d_model=int(args.get("d_model", d.d_model)),
            n_heads=int(args.get("n_heads", d.n_heads)),
            n_layers=int(args.get("n_layers", d.n_layers)),
            d_ff=int(args.get("d_ff", d.d_ff)),
            max_seq_len=int(args.get("max_seq_len", d.max_seq_len)),
            n_freq=int(args.get("n_freq", d.n_freq)),
            use_critic=bool(args.get("use_critic", d.use_critic)),
            detach_critic=bool(args.get("detach_critic", d.detach_critic)),
            coord_encoding=str(args.get("coord_encoding", d.coord_encoding)),
            mlp_hidden=int(args.get("mlp_hidden", d.mlp_hidden)),
            disable_slot_emb=bool(args.get("disable_slot_emb", d.disable_slot_emb)),
            # Fallback is the literal 64, NOT the YAML default: every checkpoint
            # saved before the flag existed has a 64-row slot_emb_table, and a
            # later YAML default bump (64 -> 512) must not break how old
            # checkpoints load.
            n_max_slots=int(args.get("n_max_slots", 64)),
            policy_net_select=bool(args.get("policy_net_select", d.policy_net_select)),
            same_net_bias=bool(args.get("same_net_bias", d.same_net_bias)),
            legacy_pad_layer_encoding=bool(
                args.get("legacy_pad_layer_encoding", d.legacy_pad_layer_encoding)
            ),
            legacy_net_encoding=bool(
                args.get("legacy_net_encoding", d.legacy_net_encoding)
            ),
            legacy_edge_encoding=bool(
                args.get("legacy_edge_encoding", d.legacy_edge_encoding)
            ),
            time_feature=str(args.get("time_feature", d.time_feature)),
            time_feature_cap=int(args.get("time_feature_cap", d.time_feature_cap)),
            # Fallback 1, NOT the YAML default: checkpoints saved before the
            # history window existed encode a single prev-action entry; the
            # loader also sets legacy_action_history from the state_dict.
            action_history_len=int(args.get("action_history_len", 1)),
            legacy_action_history=bool(
                args.get("legacy_action_history", d.legacy_action_history)
            ),
            # Fallback literal False, NOT the YAML default: checkpoints saved
            # before these obs knobs existed were trained without the tokens/
            # channel, and a later YAML default flip must not change how they
            # re-tokenize (obs-drift discipline — the net_constraint_obs
            # precedent). shape_obs is additionally presence-detected from the
            # state_dict by the loader (shape_embed module).
            obstacle_obs=bool(args.get("obstacle_obs", False)),
            shape_obs=bool(args.get("shape_obs", False)),
        )


# ============================================================================
# RL training orchestration (train-only argparse defaults)
# ============================================================================
#
# These hold the *default values* for the train-only CLI flags. The argparse
# builders (methods/rl_agent/training/args.py + the PPO/GRPO entrypoints) read defaults from
# here; the trainer still consumes the parsed argparse Namespace, so these are
# the single place to see/edit the default numbers, not a new consumption path.

_TRAIN = load_config(_DEFAULTS_DIR / "rl_train.yaml")
_TRAIN_SHARED = _TRAIN["shared"]


@dataclass(frozen=True)
class RLTrainConfig:
    """Shared (PPO+GRPO) training-loop / optim / logging / board defaults.

    Emitted as CLI flags by :func:`configs.loader.cli.add_dataclass_args` (the few
    with non-trivial help/choices carry ``metadata``).
    """

    iterations: int = _TRAIN_SHARED["iterations"]
    # optimization
    lr: float = _TRAIN_SHARED["lr"]
    clip_eps: float = _TRAIN_SHARED["clip_eps"]
    entropy_coef: float = field(default=_TRAIN_SHARED["entropy_coef"],
                                metadata={"help": "Entropy bonus coefficient"})
    entropy_norm: bool = field(default=_TRAIN_SHARED["entropy_norm"], metadata={
        "help": "Normalize each sample's joint entropy by its max achievable "
                "entropy ln(N_valid) (summed per used head over the masked "
                "action/pointer/mode distributions) before applying "
                "--entropy-coef. Makes the bonus invariant to action-space "
                "size (e.g. mres candidate ladders); the logged 'entropy' "
                "metric becomes this [0,1] relative entropy."})
    batch_size: int = _TRAIN_SHARED["batch_size"]
    max_grad_norm: float = field(default=_TRAIN_SHARED["max_grad_norm"],
                                 metadata={"help": "Gradient clipping"})
    warmup_iters: int = field(default=_TRAIN_SHARED["warmup_iters"],
                              metadata={"help": "Linear LR warmup iterations (0 = disabled)"})
    # NOTE: update-time OOM is handled automatically by policy_update_loop
    # (sorted 1/4-peel gradient-accumulation, no budget/knob) —
    # see methods/rl_agent/algorithms/_common.py.
    mem_budget: bool = field(default=_TRAIN_SHARED["mem_budget"], metadata={
        "help": "Preemptive peak-VRAM budget planner: calibrate peak(B,L) on "
                "the first buffer, then pre-split update minibatches and "
                "rollout forwards to fit free VRAM (exact; OOM peel stays as "
                "backstop). See methods/rl_agent/training/mem_budget.py."})
    expect_env_diff: str = field(default="", metadata={
        "help": "Comma-separated env kwargs that are ALLOWED to differ between "
                "the training envs and the validation envs. The trainer dumps "
                "what each side was actually built with to "
                "<save-dir>/env_records/ and halts on any undeclared "
                "difference, printing the line to paste here. Intent lives with "
                "the run, not in a global table, so a deliberately different "
                "validation setup stays expressible. Differences the harness "
                "creates itself (the val seed, which comes from "
                "--eval-base-seed) need no declaration."})
    # logging / checkpointing
    save_freq: int = _TRAIN_SHARED["save_freq"]
    log_every: int = field(default=_TRAIN_SHARED["log_every"], metadata={
        "help": "Print iter summary every N iterations (default: 1 = every iter)."})
    seed: int = _TRAIN_SHARED["seed"]
    device: str = _TRAIN_SHARED["device"]
    # inline held-out eval cadence
    eval_n_rollouts: int = field(default=_TRAIN_SHARED["eval_n_rollouts"], metadata={
        "help": "Stochastic rollouts per test board during inline eval."})
    eval_base_seed: int = field(default=_TRAIN_SHARED["eval_base_seed"], metadata={
        "help": "Base seed for inline eval rollouts."})
    # board / dataset selection
    boards_order: str = field(default=_TRAIN_SHARED["boards_order"], metadata={
        "choices": ["single", "round_robin", "per_env_random", "per_env_epoch"],
        "help": "single: train on --board only (default). round_robin: rotate "
                "through boards in --boards-json, sorted ascending by pad count "
                "(gentle curriculum), one board/iter replicated across n_envs. "
                "per_env_random: each env samples a random board per iteration "
                "(sticky within iter, WITH replacement). per_env_epoch: shuffle "
                "the full list, consume n_envs boards/iter WITHOUT replacement; "
                "reshuffle on exhaustion (~ceil(N/n_envs) iters/epoch)."})
    boards_json: str | None = field(default=_TRAIN_SHARED["boards_json"], metadata={
        "help": "JSON split file — required when --boards-order != single "
                "(configs/datasets/ registry)."})
    boards_difficulty: str = field(default=_TRAIN_SHARED["boards_difficulty"],
                                   metadata={"choices": ["easy", "medium", "hard"]})
    boards_split: str = field(default=_TRAIN_SHARED["boards_split"], metadata={
        "help": "Key under <difficulty> to use (e.g. 'train', 'train_small', 'test')."})
    boards_dataset_dir: str | None = field(default=_TRAIN_SHARED["boards_dataset_dir"], metadata={
        "help": "Deprecated compatibility override: if set, used instead of the "
                "split JSON's top-level dataset_dirs entry for both train and "
                "eval board resolution."})
    # inline held-out eval cadence (None = auto / once per round-robin pass)
    eval_split: str | None = field(default=_TRAIN_SHARED["eval_split"], metadata={
        "help": "If set (e.g. 'test'), run held-out eval after each full "
                "round-robin pass over the train boards. Requires "
                "--boards-order=round_robin."})
    eval_every: int | None = field(default=_TRAIN_SHARED["eval_every"], metadata={
        "help": "Run held-out eval every N iterations. If unset, falls back to "
                "once per full pass of the train board list."})
    eval_boards_per_batch: int | None = field(default=_TRAIN_SHARED["eval_boards_per_batch"], metadata={
        "help": "Boards to pack into one eval pool batch. None = auto "
                "(n_envs // eval_n_rollouts)."})
    eval_board_limit: int | None = field(default=_TRAIN_SHARED["eval_board_limit"], metadata={
        "help": "Optional cap on inline held-out eval boards. None evaluates the "
                "full split."})
    # secondary (diagnostic) inline eval set — does NOT drive best-ckpt
    eval2_boards: str | None = field(default=_TRAIN_SHARED["eval2_boards"], metadata={
        "help": "Secondary held-out eval board-list file (e.g. real d3-b). "
                "Evaluated each --eval-every under --eval2-prefix; diagnostic "
                "only (best-ckpt stays on the primary --eval-split)."})
    eval2_prefix: str = field(default=_TRAIN_SHARED["eval2_prefix"], metadata={
        "help": "Dashboard prefix for the --eval2-boards set (default val_d3b)."})
    eval3_boards: str | None = field(default=_TRAIN_SHARED["eval3_boards"], metadata={
        "help": "Third held-out eval board-list file (e.g. real d3-a). Same "
                "semantics as --eval2-boards; logged under --eval3-prefix."})
    eval3_prefix: str = field(default=_TRAIN_SHARED["eval3_prefix"], metadata={
        "help": "Dashboard prefix for the --eval3-boards set (default val_d3a)."})
    eval4_boards: str | None = field(default=_TRAIN_SHARED["eval4_boards"], metadata={
        "help": "Fourth held-out eval board-list file. Same semantics as "
                "--eval2-boards; logged under --eval4-prefix."})
    eval4_prefix: str = field(default=_TRAIN_SHARED["eval4_prefix"], metadata={
        "help": "Dashboard prefix for the --eval4-boards set (default val_d2b)."})
    eval5_boards: str | None = field(default=_TRAIN_SHARED["eval5_boards"], metadata={
        "help": "Fifth held-out eval board-list file. Same semantics as "
                "--eval2-boards; logged under --eval5-prefix."})
    eval5_prefix: str = field(default=_TRAIN_SHARED["eval5_prefix"], metadata={
        "help": "Dashboard prefix for the --eval5-boards set (default val_d2a)."})
    eval_diag_max_steps: int | None = field(
        default=_TRAIN_SHARED["eval_diag_max_steps"], metadata={
            "help": "Pin the diagnostic eval2..eval5 sets to this max_steps "
                    "(env cap + rollout loop), regardless of the run's "
                    "--max-steps — keeps the diagnostic protocol comparable "
                    "across cells that vary the train horizon. None (default) "
                    "= inherit the train protocol. Primary val / val_greedy "
                    "stay native."})
    eval_diag_masking_rule: str | None = field(
        default=_TRAIN_SHARED["eval_diag_masking_rule"], metadata={
            "help": "Pin the diagnostic eval2..eval5 sets to this masking "
                    "rule (path or name), regardless of the run's "
                    "--masking-rule. None (default) = inherit. Primary val / "
                    "val_greedy stay native."})
    eval_at_init: bool = field(default=_TRAIN_SHARED["eval_at_init"], metadata={
        "help": "Eval the initial (untrained/zero-shot) policy once at iter 0 "
                "before training; logged but excluded from best-ckpt."})
    eval_greedy: bool = field(default=_TRAIN_SHARED["eval_greedy"], metadata={
        "cli_skip": True,  # default-True bool -> inverted flag --no-eval-greedy (args.py tail)
        "help": "Greedy (argmax) 1-rollout pass over the primary val set each "
                "cadence, logged under val_greedy/* (mean==max per board by "
                "construction). Separates exploration noise from genuine "
                "uncertainty next to the sampled val/*; diagnostic only — "
                "best-ckpt stays on val/*."})
    async_val: bool = field(default=_TRAIN_SHARED["async_val"], metadata={
        "help": "Detach validation from training: at each eval cadence, queue "
                "the policy into <save-dir>/val_queue/ for an external watcher "
                "process (methods.rl_agent.training.async_val) instead of "
                "evaluating inline. Results are logged into the trainer's own "
                "TensorBoard/W&B run as they arrive. Requires --eval-split; "
                "default off = inline eval, unchanged."})
    # Weights & Biases (optional; default off — telemetry is opt-in via --wandb).
    # `wandb` is the positive master toggle; its CLI form is added explicitly by
    # the args tail, so it is cli_skip. The wandb_* string knobs are emitted
    # directly.
    wandb: bool = field(default=_TRAIN_SHARED["wandb"], metadata={"cli_skip": True})
    wandb_project: str | None = field(default=_TRAIN_SHARED["wandb_project"], metadata={
        "help": "W&B project name. Falls back to $WANDB_PROJECT or "
                "'pcbworld'."})
    wandb_entity: str | None = field(default=_TRAIN_SHARED["wandb_entity"], metadata={
        "help": "W&B entity (team/org). Falls back to $WANDB_ENTITY."})
    wandb_group: str | None = field(default=_TRAIN_SHARED["wandb_group"], metadata={
        "help": "W&B run group (clusters ablation runs). Falls back to "
                "$WANDB_RUN_GROUP."})
    wandb_run_name: str | None = field(default=_TRAIN_SHARED["wandb_run_name"], metadata={
        "help": "W&B run display name. Defaults to the --log-dir basename."})
    wandb_tags: str | None = field(default=_TRAIN_SHARED["wandb_tags"], metadata={
        "help": "Comma-separated list of W&B tags (e.g. 'stage1,ablation')."})
    # vecenv: positive master toggle (inverted CLI --no-vecenv), so cli_skip.
    vecenv: bool = field(default=_TRAIN_SHARED["vecenv"], metadata={"cli_skip": True})
    # resume: variable lives here (visible); its CLI form (--resume, type=str) is
    # added by each entrypoint, so cli_skip.
    resume: str | None = field(default=_TRAIN_SHARED["resume"], metadata={"cli_skip": True})


@dataclass(frozen=True)
class PPOConfig:
    """PPO-specific training defaults (rollout + GAE + per-algo dirs).

    ``n_epochs`` / ``log_dir`` / ``save_dir`` are ``cli_skip`` — they feed
    :func:`methods.rl_agent.training.args.add_shared_args` as per-algorithm defaults (the
    flags are emitted there), not as entrypoint-local flags.
    """

    n_epochs: int = field(default=_TRAIN["ppo"]["n_epochs"], metadata={"cli_skip": True})
    n_envs: int = field(default=_TRAIN["ppo"]["n_envs"],
                        metadata={"help": "Number of parallel environments"})
    n_steps: int = field(default=_TRAIN["ppo"]["n_steps"],
                         metadata={"help": "Steps per env per PPO iteration"})
    gamma: float = field(default=_TRAIN["ppo"]["gamma"], metadata={
        "help": "Discount factor (default 1.0 — episodic + terminal-reward)"})
    gae_lambda: float = _TRAIN["ppo"]["gae_lambda"]
    vf_coef: float = _TRAIN["ppo"]["vf_coef"]
    norm_reward_clip: float = field(default=_TRAIN["ppo"]["norm_reward_clip"], metadata={
        "help": "Symmetric clipping bound for normalized rewards"})
    # Positive master toggles whose CLI forms are inverted (--no-normalize-adv,
    # --no-truncation-bootstrap, --no-norm-reward), so they are cli_skip — the
    # variables stay visible here; each entrypoint adds the explicit --no-* flag.
    normalize_adv: bool = field(default=_TRAIN["ppo"]["normalize_adv"], metadata={"cli_skip": True})
    truncation_bootstrap: bool = field(default=_TRAIN["ppo"]["truncation_bootstrap"],
                                       metadata={"cli_skip": True})
    norm_reward: bool = field(default=_TRAIN["ppo"]["norm_reward"], metadata={"cli_skip": True})
    log_dir: str = field(default=_TRAIN["ppo"]["log_dir"], metadata={"cli_skip": True})
    save_dir: str = field(default=_TRAIN["ppo"]["save_dir"], metadata={"cli_skip": True})


@dataclass(frozen=True)
class GRPOConfig:
    """GRPO-specific training defaults (group baseline + per-algo dirs).

    ``n_epochs`` / ``log_dir`` / ``save_dir`` are ``cli_skip`` (see PPOConfig).
    """

    n_epochs: int = field(default=_TRAIN["grpo"]["n_epochs"], metadata={"cli_skip": True})
    n_envs: int = field(default=_TRAIN["grpo"]["n_envs"],
                        metadata={"help": "Number of parallel environments (GRPO groups)"})
    group_size: int = field(default=_TRAIN["grpo"]["group_size"],
                            metadata={"help": "Rollouts per GRPO group"})
    log_dir: str = field(default=_TRAIN["grpo"]["log_dir"], metadata={"cli_skip": True})
    save_dir: str = field(default=_TRAIN["grpo"]["save_dir"], metadata={"cli_skip": True})


# ============================================================================
# RL eval orchestration
# ============================================================================
#
# Eval-pipeline knobs (env/policy come from the checkpoint at eval time). The
# `_EVAL` values live in rl_eval.yaml; the validation enums + manifest version
# below are code constants (argparse `choices=` / on-disk schema tag), not
# tunable values, so they stay here rather than in YAML.

# Manifest/summary schema version, shared by every eval stage entrypoint
# (core 3-stage, rollout.plan_only, rollout.rule-based) — defined exactly once.
SCHEMA_VERSION: int = 1

ROLLOUT_MODES: tuple[str, ...] = ("serial", "parallel")
DRC_SWITCHES: tuple[str, ...] = ("on", "off")
SELECTION_METHODS: tuple[str, ...] = ("final_potential", "posthoc_drc_aware", "none")
CHECK_ANGLES: tuple[int, ...] = (45, 90)

_EVAL = load_config(_DEFAULTS_DIR / "rl_eval.yaml")


@dataclass(frozen=True)
class RLEvalConfig:
    """Canonical eval-pipeline defaults."""

    rollout_mode: str = _EVAL["rollout_mode"]
    n_envs: int = _EVAL["n_envs"]
    # None = inherit the ckpt's training emit_drc_tokens; "on"/"off" force it.
    env_drc: str | None = _EVAL["env_drc"]
    inline_drc: str = _EVAL["inline_drc"]
    # DRC scoring reward (configs/reward/*.yaml). Distinct from the training
    # reward_rule on EnvConfig: you can train with one rule and score with another.
    reward_config: str = _EVAL["reward_config"]
    # check_angle CLI flag defaults to None = inherit the ckpt's corner_mode;
    # this value is only the no-ckpt fallback (e.g. --skip-rollout).
    check_angle: int = _EVAL["check_angle"]
    selection_method: str = _EVAL["selection_method"]
    # Intentionally lifts the ckpt's trained slot cap at inference; None = inherit.
    override_n_max_slots: int | None = _EVAL["override_n_max_slots"]
    save_artifacts: str = _EVAL["save_artifacts"]
    output_root: str = _EVAL["output_root"]
    device: str = _EVAL["device"]
    early_stop_finish_no_progress: int = _EVAL["early_stop_finish_no_progress"]
    early_stop_no_geometry_progress: int = _EVAL["early_stop_no_geometry_progress"]
    # Best-checkpoint selection — the single canonical criterion shared across
    # branches: RL (PPOTrainer/GRPOTrainer inline best-ckpt via on_validation)
    # AND LLM (verl trainer.best_metric_key in run_cadagent*.sh) both select the
    # checkpoint maximizing this val tag ("largest fp gain is best").
    best_metric_key: str = _EVAL["best_metric_key"]
    best_metric_mode: str = _EVAL["best_metric_mode"]


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
# --no_drc_tokens, the dynamic-choices --prompt_version, the repo-abs-resolved
# --boards_json) — the variable stays visible; only the flag is added explicitly.

_LLM = load_config(_DEFAULTS_DIR / "llm.yaml")


@dataclass(frozen=True)
class LLMEnvConfig:
    """LLM-eval env-core knobs (eval/args.py ``env`` group)."""

    max_steps: int = _LLM["env"]["max_steps"]
    masking_rule: str = field(default=_LLM["env"]["masking_rule"], metadata={
        "choices": ["strict", "relaxed", "default", "default_no_finish",
                    "default_no_via", "default_no_finish_no_via"]})
    reward_rule: str = field(default=_LLM["env"]["reward_rule"],
                            metadata={"choices": ["default", "shaped", "grpo_final"]})
    corner_mode: int = field(default=_LLM["env"]["corner_mode"], metadata={
        "choices": [0, 1, 2, 3],
        "help": "0=MITERED_45 (default), 2=MITERED_90 (no diagonals)."})
    state_format: str = field(default=_LLM["env"]["state_format"], metadata={
        "choices": ["sexpr", "xml"],
        "help": "Observation serialization format (default: sexpr)."})
    via_penalty: float | None = field(default=_LLM["env"]["via_penalty"], metadata={
        "help": "Override reward-config via_penalty (default: use config)."})
    reward_noise_std: float = field(default=_LLM["env"]["reward_noise_std"], metadata={
        "help": "Gaussian noise stddev applied to env reward (default: 0)."})
    # emit_drc_tokens: variable visible here; CLI form is the inverted
    # --no_drc_tokens (default: emit), so cli_skip.
    emit_drc_tokens: bool = field(default=_LLM["env"]["emit_drc_tokens"],
                                  metadata={"cli_skip": True})
    use_yaml_drc_fallback: bool = field(default=_LLM["env"]["use_yaml_drc_fallback"], metadata={
        "help": "When a board has no companion .kicad_pro and no legacy (setup ...) "
                "tokens, substitute the YAML at --drc_config_path instead of "
                "raising (default: strict)."})
    drc_config_path: str | None = field(default=_LLM["env"]["drc_config_path"], metadata={
        "help": "YAML path supplying the global minima used by "
                "--use_yaml_drc_fallback. Required when that flag is set — "
                "there is no implicit default (e.g. configs/drc/default.yaml)."})


@dataclass(frozen=True)
class LLMBoardsConfig:
    """LLM-eval multi-board selection (eval/args.py ``boards`` group)."""

    boards_order: str = field(default=_LLM["boards"]["boards_order"], metadata={
        "choices": ["single", "round_robin"],
        "help": "single (default): eval only on --board_path. round_robin: "
                "iterate through boards in --boards_json (sorted ascending by pad "
                "count), running --rollout_episodes episodes per board."})
    # boards_json has no default (None) — multi-board mode must set it
    # explicitly; it stays cli_skip because its flag form is bespoke
    # (eval/args.py defines it directly).
    boards_json: str | None = field(default=_LLM["boards"]["boards_json"],
                                    metadata={"cli_skip": True})
    boards_difficulty: str = field(default=_LLM["boards"]["boards_difficulty"],
                                   metadata={"choices": ["easy", "medium", "hard"]})
    boards_split: str = field(default=_LLM["boards"]["boards_split"], metadata={
        "help": "Key under <difficulty> to use (e.g. 'train', 'test', "
                "'train_small'). Eval default is 'test'."})


@dataclass(frozen=True)
class LLMPromptConfig:
    """LLM-eval prompt selection (eval/args.py ``prompt`` group).

    ``prompt_version``'s choices are dynamic (``CADAGENT_PROMPT_VERSIONS``), so the
    flag is added explicitly by ``add_prompt_args``; the variable is visible here.
    """

    prompt_version: str = field(default=_LLM["prompt"]["prompt_version"],
                               metadata={"cli_skip": True})


@dataclass(frozen=True)
class LLMRolloutConfig:
    """LLM-eval rollout orchestration (eval/args.py ``rollout`` group)."""

    env_num: int = field(default=_LLM["rollout"]["env_num"], metadata={
        "help": "Number of parallel envs in the vectorised rollout."})
    seed: int = _LLM["rollout"]["seed"]
    rollout_episodes: int = _LLM["rollout"]["rollout_episodes"]
    num_cpus_per_worker: float = _LLM["rollout"]["num_cpus_per_worker"]
    history_length: int = field(default=_LLM["rollout"]["history_length"], metadata={
        "help": "Number of recent (obs, action) pairs in prompt (0 = no history)."})
    temperature: float = _LLM["rollout"]["temperature"]
    max_new_tokens: int = _LLM["rollout"]["max_new_tokens"]
    dump_dir: str | None = field(default=_LLM["rollout"]["dump_dir"], metadata={
        "help": "Directory for per-episode rollout JSON (one file per episode)."})
    early_stop_no_progress: int = field(default=_LLM["rollout"]["early_stop_no_progress"], metadata={
        "help": "Early-stop an env after this many consecutive non-progress steps "
                "(parse-error / mask-rejected / empty_action). 0 disables (default)."})


@dataclass(frozen=True)
class LLMVLLMConfig:
    """LLM-eval vLLM-provider knobs (eval/args.py ``llm`` group)."""

    model_path: str = field(default=_LLM["vllm"]["model_path"], metadata={
        "help": "HuggingFace model name or local path."})
    tensor_parallel_size: int = field(default=_LLM["vllm"]["tensor_parallel_size"], metadata={
        "help": "Number of GPUs for tensor parallelism. Use CUDA_VISIBLE_DEVICES "
                "to select specific GPUs."})
    gpu_memory_utilization: float = field(default=_LLM["vllm"]["gpu_memory_utilization"], metadata={
        "help": "Fraction of GPU memory for vLLM KV cache (default: 0.95)."})
    max_model_len: int = field(default=_LLM["vllm"]["max_model_len"], metadata={
        "help": "Maximum sequence length (prompt + generation)."})
    enable_prefix_caching: bool = field(default=_LLM["vllm"]["enable_prefix_caching"], metadata={
        "help": "Enable vLLM automatic prefix caching."})
    enable_chunked_prefill: bool = field(default=_LLM["vllm"]["enable_chunked_prefill"], metadata={
        "help": "Enable vLLM chunked prefill."})
    guided_grammar: str | None = field(default=_LLM["vllm"]["guided_grammar"], metadata={
        "help": "Name of a grammar registered in methods.llm_agent.wrappers.grammar GRAMMARS "
                "(e.g. 'cadagent_v1'). When set, generation is constrained via vLLM "
                "GuidedDecodingParams (regex, llguidance backend)."})


@dataclass(frozen=True)
class LLMAPIConfig:
    """LLM-eval provider-API knobs (eval/args.py ``api`` group)."""

    api_provider: str = field(default=_LLM["api"]["api_provider"], metadata={
        "choices": ["openai", "anthropic", "google", "together"],
        "help": "API provider (default: openai). 'together' uses Together AI "
                "(OpenAI-compatible) for hosted open-weight models like Qwen."})
    api_model: str | None = field(default=_LLM["api"]["api_model"], metadata={
        "help": "API model name (e.g. 'gpt-4o', 'claude-sonnet-4-20250514', "
                "'gemini-2.0-flash'). If unset, uses a per-provider default."})
    api_key: str | None = field(default=_LLM["api"]["api_key"], metadata={
        "help": "API key. Overrides OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY."})
    api_rpm: int | None = field(default=_LLM["api"]["api_rpm"], metadata={
        "help": "Per-minute request cap. None disables (default)."})
    api_itpm: int | None = field(default=_LLM["api"]["api_itpm"], metadata={
        "help": "Per-minute input-token cap (incl. prompt-cache reads/creations)."})
    api_otpm: int | None = field(default=_LLM["api"]["api_otpm"], metadata={
        "help": "Per-minute output-token cap."})
    api_max_retries: int = field(default=_LLM["api"]["api_max_retries"], metadata={
        "help": "Provider-SDK retry count for 429 / transient errors (default 8)."})
    api_prompt_cache: bool = field(default=_LLM["api"]["api_prompt_cache"], metadata={
        "help": "Anthropic only: tag the system prompt with cache_control (5-min "
                "ephemeral cache, ~90%% input cost reduction on hits)."})
    api_reasoning_effort: str | None = field(default=_LLM["api"]["api_reasoning_effort"], metadata={
        "choices": ["minimal", "low", "medium", "high"],
        "help": "Reasoning models (Qwen3 thinking on Together, gpt-5.x). 'low' "
                "shrinks chain-of-thought so it fits inside max_new_tokens."})
    api_disable_thinking: bool = field(default=_LLM["api"]["api_disable_thinking"], metadata={
        "help": "Together (Qwen3 thinking models): pass "
                "chat_template_kwargs={enable_thinking: false} so the server skips "
                "chain-of-thought entirely."})


@dataclass(frozen=True)
class LLMConfig:
    """Canonical LLM-eval defaults — the complete, grouped variable list.

    One nested sub-config per eval/args.py CLI group; each is generated via
    :func:`configs.loader.cli.add_dataclass_args` (``style='underscore'``). Defaults live
    in configs/defaults/llm.yaml.
    """

    env: LLMEnvConfig = field(default_factory=LLMEnvConfig)
    boards: LLMBoardsConfig = field(default_factory=LLMBoardsConfig)
    prompt: LLMPromptConfig = field(default_factory=LLMPromptConfig)
    rollout: LLMRolloutConfig = field(default_factory=LLMRolloutConfig)
    vllm: LLMVLLMConfig = field(default_factory=LLMVLLMConfig)
    api: LLMAPIConfig = field(default_factory=LLMAPIConfig)


# ============================================================================
# LLM (verl) training orchestration
# ============================================================================
#
# The cadagent-specific slice of the verl Hydra config (the ``env.cadagent.*``
# block + the ``env.*`` knobs cadagent reads). Distinct from LLMConfig, which is
# the LLM *eval* program's argparse defaults: this is the LLM *training* program
# (verl) — there is no argparse, the verl config supplies per-run overrides and
# this schema is the single defaults source for the ``config.env.cadagent.*``
# fallbacks read by the verl-agent patch + methods/llm_agent/training/manager.py. Flat, to
# mirror the flat ``env.cadagent`` namespace.

_LLM_TRAIN = load_config(_DEFAULTS_DIR / "llm_train.yaml")


@dataclass(frozen=True)
class LLMTrainConfig:
    """cadagent LLM(verl)-training config — the ``env.cadagent.*`` slice.

    Built from a verl omegaconf config with :meth:`from_verl_config`; the
    env-build kwargs come back via :meth:`to_env_kwargs` (the dict fed to
    ``build_cadagent_envs``). Defaults = configs/defaults/llm_train.yaml.
    """

    # env-core (PCBWorld build knobs → to_env_kwargs)
    max_steps: int = _LLM_TRAIN["max_steps"]
    masking_rule: str = _LLM_TRAIN["masking_rule"]
    reward_rule: str = _LLM_TRAIN["reward_rule"]
    state_format: str = _LLM_TRAIN["state_format"]
    corner_mode: int = _LLM_TRAIN["corner_mode"]  # engine code 0..3
    via_penalty: float | None = _LLM_TRAIN["via_penalty"]
    reward_noise_std: float = _LLM_TRAIN["reward_noise_std"]
    emit_drc_tokens: bool = _LLM_TRAIN["emit_drc_tokens"]
    # prompt (methods/llm_agent/training/manager.py)
    prompt_version: str = _LLM_TRAIN["prompt_version"]
    history_length: int = _LLM_TRAIN["history_length"]
    # board scheduling — train
    boards_order: str = _LLM_TRAIN["boards_order"]
    boards_json: str | None = _LLM_TRAIN["boards_json"]
    boards_difficulty: str = _LLM_TRAIN["boards_difficulty"]
    boards_split: str = _LLM_TRAIN["boards_split"]
    # board scheduling — val (val_boards_split None = inherit boards_split)
    val_boards_order: str = _LLM_TRAIN["val_boards_order"]
    val_boards_split: str | None = _LLM_TRAIN["val_boards_split"]
    # episode scoring + guided decoding
    score_train_episodes: bool = _LLM_TRAIN["score_train_episodes"]
    guided_decoding_grammar: str | None = _LLM_TRAIN["guided_decoding_grammar"]
    # val episode-end DRC scoring config — DELIBERATELY sourced from the shared
    # eval config (RLEvalConfig / rl_eval.yaml = DEFAULTS), NOT a separate
    # llm_train default, so LLM val and the RL inline-eval path score with the
    # SAME ruler (W5 unification). Override per-run via env.cadagent.eval_*.
    eval_reward_config: str = DEFAULTS.reward_config
    eval_check_angle: int = DEFAULTS.check_angle
    # misc
    seed: int = _LLM_TRAIN["seed"]
    # required runtime input (env.cadagent.board_path); no meaningful default
    board_path: str | None = _LLM_TRAIN["board_path"]

    @property
    def resolved_val_boards_split(self) -> str:
        """Val split, inheriting the train ``boards_split`` when unset."""
        return self.val_boards_split if self.val_boards_split is not None else self.boards_split

    def to_env_kwargs(self) -> dict[str, Any]:
        """Flat env-build kwargs for ``build_cadagent_envs(env_kwargs=...)``.

        Exactly the dict the verl-agent patch built by hand (PCBWorld keyword
        surface) — board scheduling / prompt / scoring are handled separately.
        """
        return {
            "max_steps": self.max_steps,
            "masking_rule": self.masking_rule,
            "reward_rule": self.reward_rule,
            "state_format": self.state_format,
            "corner_mode": self.corner_mode,
            "via_penalty": self.via_penalty,
            "reward_noise_std": self.reward_noise_std,
            "emit_drc_tokens": self.emit_drc_tokens,
        }

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

        env-core / board / prompt / scoring knobs read from ``env.cadagent.*``;
        ``max_steps`` / ``history_length`` / ``seed`` read from the general
        ``env.*`` block (where cadagent consumes them). Any key the config omits
        falls back to the shared YAML default (``cls()`` =
        configs/defaults/llm_train.yaml).
        """
        d = cls()  # shared YAML defaults
        g = cls._cfg_get
        env = g(config, "env", None)
        cad = g(env, "cadagent", None)
        return cls(
            max_steps=int(g(env, "max_steps", d.max_steps)),
            masking_rule=str(g(cad, "masking_rule", d.masking_rule)),
            reward_rule=str(g(cad, "reward_rule", d.reward_rule)),
            state_format=str(g(cad, "state_format", d.state_format)),
            corner_mode=int(g(cad, "corner_mode", d.corner_mode)),
            via_penalty=g(cad, "via_penalty", d.via_penalty),
            reward_noise_std=float(g(cad, "reward_noise_std", d.reward_noise_std)),
            emit_drc_tokens=bool(g(cad, "emit_drc_tokens", d.emit_drc_tokens)),
            prompt_version=str(g(cad, "prompt_version", d.prompt_version)),
            history_length=int(g(env, "history_length", d.history_length)),
            boards_order=str(g(cad, "boards_order", d.boards_order)),
            boards_json=g(cad, "boards_json", d.boards_json),
            boards_difficulty=str(g(cad, "boards_difficulty", d.boards_difficulty)),
            boards_split=str(g(cad, "boards_split", d.boards_split)),
            val_boards_order=str(g(cad, "val_boards_order", d.val_boards_order)),
            val_boards_split=g(cad, "val_boards_split", d.val_boards_split),
            score_train_episodes=bool(g(cad, "score_train_episodes", d.score_train_episodes)),
            guided_decoding_grammar=g(cad, "guided_decoding_grammar", d.guided_decoding_grammar),
            eval_reward_config=str(g(cad, "eval_reward_config", d.eval_reward_config)),
            eval_check_angle=int(g(cad, "eval_check_angle", d.eval_check_angle)),
            seed=int(g(env, "seed", d.seed)),
            board_path=g(cad, "board_path", d.board_path),
        )
