"""Checkpoint -> policy + env-kwargs builders for the eval pipeline.

The canonical "load a trained ``KiCadRLModel`` from a ``.pt`` checkpoint"
routine, shared by the eval rollout path
(``methods.rl_agent.rollout.transformer.load_policy_from_ckpt``) and per-board eval scripts,
plus the ``env_kwargs_from_*`` builders. ``torch`` / ``KiCadRLModel`` are
imported lazily inside the functions so importing this module stays free of an
eager torch dependency.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from configs.loader.schema import RLEnvConfig, corner_mode_to_code


# ============================================================================
# Env-kwargs builders (thin adapters onto the canonical config.env.RLEnvConfig)
# ============================================================================


# Alias: the canonical corner-mode sugar mapper lives in configs.loader.schema;
# kept under this name because eval.pipeline + tests import it from here.
_corner_mode_code = corner_mode_to_code


def env_kwargs_from_checkpoint(
    ckpt_args: dict[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    """Build ``make_decoder_env_pool`` kwargs from a checkpoint's ``args`` dict."""
    return RLEnvConfig.from_checkpoint(ckpt_args, max_steps).to_pool_kwargs()


def env_kwargs_from_training_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build pool kwargs from a training-time argparse Namespace."""
    return RLEnvConfig.from_namespace(args).to_pool_kwargs()


def apply_drc_off(env_kwargs: dict[str, Any]) -> None:
    """Silence env-side DRC: skip DRC engine calls and zero the DRC reward
    terms. Mutates ``env_kwargs`` in place."""
    env_kwargs["emit_drc_tokens"] = False
    env_kwargs["drc_penalty"] = 0.0
    env_kwargs["drc_log_scale"] = 0.0
    env_kwargs["drc_log_agg_scale"] = 0.0


# ============================================================================
# Checkpoint -> policy reconstruction
# ============================================================================


def _tokenizer_fenc_dim(args: dict[str, Any]) -> int:
    coord_encoding = str(args.get("coord_encoding", "fourier"))
    if coord_encoding == "fourier":
        return 4 * int(args.get("n_freq", 32))
    return int(args.get("d_model", 128))


def _policy_args_for_checkpoint(
    args: dict[str, Any],
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Return policy args compatible with this checkpoint's tokenizer shape."""
    compat_args = dict(args)
    # shape_obs (boundary-shape channel): the additive shape_embed module is
    # constructed only when the knob is on, so its presence in the state_dict
    # is physical evidence — weights cannot lie, saved args could. A ckpt
    # whose args CONTRADICT its weights (hand-edited ckpt, tampered save) is
    # refused rather than silently resolved; only an ABSENT key (pre-knob
    # checkpoint — the knob didn't exist, weights are the sole record) falls
    # back to the weights verdict. Sits above the pad_proj early-return so
    # every state_dict gets a verdict.
    weights_shape_obs = "tokenizer.vocab.shape_embed.weight" in state_dict
    saved_shape_obs = args.get("shape_obs")
    if saved_shape_obs is not None and bool(saved_shape_obs) != weights_shape_obs:
        raise RuntimeError(
            "Checkpoint args/weights contradiction: args say "
            f"shape_obs={bool(saved_shape_obs)} but the state_dict "
            f"{'contains' if weights_shape_obs else 'lacks'} shape_embed "
            "weights — refusing to guess which is the training truth."
        )
    if saved_shape_obs is None:
        print(
            "[ckpt] legacy checkpoint without shape_obs arg — using the "
            f"weights verdict shape_obs={weights_shape_obs}"
        )
    compat_args["shape_obs"] = weights_shape_obs
    pad_weight = state_dict.get("tokenizer.vocab.pad_proj.weight")
    if pad_weight is None:
        return compat_args

    f = _tokenizer_fenc_dim(compat_args)
    legacy_pad_in = 5 * f + 1
    current_pad_in = 6 * f + 1
    pad_in = int(pad_weight.shape[1])
    if pad_in == legacy_pad_in:
        compat_args["legacy_pad_layer_encoding"] = True
    elif pad_in == current_pad_in:
        compat_args["legacy_pad_layer_encoding"] = False
    else:
        raise RuntimeError(
            "Unsupported pad tokenizer input dimension in checkpoint: "
            f"got {pad_in}, expected {legacy_pad_in} (legacy) or "
            f"{current_pad_in} (current)."
        )

    net_weight = state_dict.get("tokenizer.vocab.net_proj.weight")
    if net_weight is not None:
        net_in = int(net_weight.shape[1])
        if net_in == 3 * f:      # old: track_width + clearance + via_diameter
            compat_args["legacy_net_encoding"] = True
        elif net_in == 4 * f:    # current: + per-episode `closed` flag
            compat_args["legacy_net_encoding"] = False
        else:
            raise RuntimeError(
                "Unsupported net tokenizer input dimension in checkpoint: "
                f"got {net_in}, expected {3 * f} (legacy) or {4 * f} (current)."
            )

    # EDGE token: 2-point (old) vs 3-point + midpoint projection (current).
    # The shared proj shapes are identical either way, so the signal is the
    # presence of the edge_mid_proj module, not a weight shape.
    compat_args["legacy_edge_encoding"] = (
        "tokenizer.vocab.edge_mid_proj.weight" not in state_dict
    )

    # ACTION_HISTORY: single prev-action entry (old) vs K-entry history with
    # the age projection (current). The shared at/pt/mode weights are
    # identical either way, so the signal is the presence of the
    # history_age_proj module. Legacy => force K=1 (the historical layout;
    # bit-identical encoding, slot/age-free).
    if "tokenizer.vocab.history_age_proj.weight" not in state_dict:
        compat_args["legacy_action_history"] = True
        compat_args["action_history_len"] = 1
    else:
        compat_args["legacy_action_history"] = False

    # obstacle_obs adds no weights (OBSTACLE ties pad_proj; the entity-type
    # row lives in the always-current table) — the saved arg, defaulted False
    # by from_checkpoint for pre-knob checkpoints, is the only signal.
    # (shape_obs is presence-detected at the top of this function.)
    return compat_args


def pad_legacy_entity_type_rows(
    state_dict: dict[str, Any],
    model: Any,
) -> None:
    """Grow a pre-OBSTACLE checkpoint's entity-type table in place.

    ``EntityType.OBSTACLE`` (2026-08) extended the type-embedding table by one
    row. ``load_state_dict(strict=False)`` hard-errors on size mismatch, so a
    legacy table is padded with the freshly initialized rows of *model* before
    the load. Knob-off policies never index the padded row, keeping their
    token stream byte-identical; any other row-count mismatch is a real drift
    and stays loud.
    """
    key = "entity_type_embed.weight"
    full_key = next(
        (k for k in model.state_dict() if k.endswith(key)), None,
    )
    if full_key is None or full_key not in state_dict:
        return
    ckpt_rows = int(state_dict[full_key].shape[0])
    model_rows = int(model.state_dict()[full_key].shape[0])
    if ckpt_rows == model_rows:
        return
    if not ckpt_rows < model_rows:
        raise RuntimeError(
            f"Checkpoint entity-type table has {ckpt_rows} rows but the model "
            f"expects {model_rows} — a checkpoint from a NEWER EntityType "
            "enum cannot be down-loaded."
        )
    fresh = model.state_dict()[full_key]
    import torch

    state_dict[full_key] = torch.cat(
        [state_dict[full_key].to(fresh.device, fresh.dtype),
         fresh[ckpt_rows:].clone()],
        dim=0,
    )


def pad_legacy_optimizer_state(
    optimizer_state: dict[str, Any],
    model: Any,
) -> None:
    """Trainer-resume twin of :func:`pad_legacy_entity_type_rows`.

    A pre-OBSTACLE checkpoint's ``optimizer_state_dict`` carries 14-row Adam
    moments (``exp_avg``/``exp_avg_sq``/``max_exp_avg_sq``) for
    ``entity_type_embed.weight``; torch's ``Optimizer.load_state_dict`` does
    no shape validation, so the mismatch would surface as a RuntimeError at
    the FIRST ``optimizer.step()`` instead of at load. Zero-pad the moment
    rows — zeros are exactly the moments a never-updated fresh row would
    have, so a knob-off resume stays behavior-identical.
    """
    import torch

    key = "entity_type_embed.weight"
    named = [n for n, _ in model.named_parameters()]
    try:
        idx = next(i for i, n in enumerate(named) if n.endswith(key))
    except StopIteration:
        return
    target = dict(model.named_parameters())[named[idx]]
    model_rows, d_model = int(target.shape[0]), int(target.shape[1])
    # The saved per-param state is keyed by the param's position in
    # model.parameters() (single AdamW group over parameters(), loop.py; and
    # the pre/post-change param ORDER is identical — knob-off adds no
    # params), so the index lookup is exact. Shape-check before touching:
    # a same-height table means a current-format checkpoint (no-op).
    p_state = (optimizer_state.get("state") or {}).get(idx)
    if not p_state:
        return
    for mkey, t in list(p_state.items()):
        if not torch.is_tensor(t) or t.dim() != 2:
            continue
        rows, cols = int(t.shape[0]), int(t.shape[1])
        if (rows, cols) == (model_rows, d_model):
            continue  # current format
        if cols != d_model or rows >= model_rows:
            raise RuntimeError(
                f"optimizer moment {mkey!r} for {named[idx]} has shape "
                f"{tuple(t.shape)} — expected ({model_rows}, {d_model}) or a "
                f"shorter legacy table; refusing to guess."
            )
        p_state[mkey] = torch.cat(
            [t, torch.zeros(model_rows - rows, cols,
                            dtype=t.dtype, device=t.device)],
            dim=0,
        )


def _build_policy(args: dict[str, Any], device: Any) -> Any:
    from configs.loader.schema import RLPolicyConfig

    return RLPolicyConfig.from_checkpoint(args).build(device)


def _checkpoint_iteration(path: Path) -> int:
    import torch

    path = Path(path)
    try:
        iteration = int(path.stem.removeprefix("policy_iter_"))
        if path.stem.startswith("policy_iter_"):
            return iteration
    except ValueError:
        pass
    ckpt = torch.load(path, map_location="cpu")
    if "iteration" not in ckpt:
        raise KeyError(f"Checkpoint {path} has no iteration field")
    return int(ckpt["iteration"])


def _print_provenance(ckpt: dict, path: Path) -> None:
    """Informational only — commit mismatch is NOT enforcement (it is almost
    always true and almost always harmless); the obs probe below is the
    enforced check."""
    prov = ckpt.get("provenance")
    if prov:
        dirty = "+dirty" if prov.get("git_dirty") else ""
        print(
            f"[ckpt] {path.name}: trained at "
            f"{prov.get('git_commit', '?')}{dirty} "
            f"(v{prov.get('repo_version', '?')})"
        )


def _check_obs_probe(ckpt: dict, policy: Any, path: Path) -> None:
    """Enforce obs-semantics compatibility via the ckpt's embedded probe.

    Re-encodes the probe obs STORED IN THE CKPT with the current code and
    compares digests — "the obs semantics this policy was trained under still
    hold". Mismatch (or failure to encode the stored probe at all) means the
    current code would silently misread this policy's inputs → hard error.
    ``CADAGENT_ALLOW_OBS_MISMATCH=1`` downgrades to a loud warning and stamps
    ``policy.obs_schema_mismatch`` so result writers can mark the output.
    Rationale + probe contents: methods/rl_agent/models/v1/obs_probe.py.
    """
    import os

    from methods.rl_agent.models.v1.obs_probe import probe_digest

    policy.obs_schema_mismatch = None
    rec = ckpt.get("obs_probe")
    if not rec:
        print(
            f"[ckpt] {path.name}: no obs probe (checkpoint predates the "
            "obs-schema guard) — semantics check skipped; eval such "
            "checkpoints on their training-era code."
        )
        return
    try:
        current = probe_digest(policy.tokenizer, rec["obs"])
        if current == rec["digest"]:
            return
        reason = (
            f"walk digest {rec['digest'][:12]}… (training code) != "
            f"{current[:12]}… (current code)"
        )
    except Exception as e:  # noqa: BLE001 — any encode failure IS the evidence
        reason = f"stored probe no longer encodes: {type(e).__name__}: {e}"
    msg = (
        f"Obs-semantics mismatch for {path}: {reason}. The checkpoint was "
        "trained under different obs encoding semantics; evaluating it with "
        "the current code would silently misread its inputs. Re-train, or "
        "eval with the training-era code. Set CADAGENT_ALLOW_OBS_MISMATCH=1 "
        "to proceed anyway (results are stamped obs_schema_mismatch)."
    )
    if os.environ.get("CADAGENT_ALLOW_OBS_MISMATCH") == "1":
        print(f"[ckpt][WARNING] {msg}")
        policy.obs_schema_mismatch = reason
    else:
        raise RuntimeError(msg)


def _load_policy(
    checkpoint_path: Path,
    device: Any,
) -> tuple[Any, dict[str, Any], int]:
    """Load a trained policy + its args + iteration from a .pt checkpoint."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    ckpt_args = _policy_args_for_checkpoint(
        ckpt_args, ckpt["policy_state_dict"],
    )
    ckpt_args["use_critic"] = any(
        key.startswith("critic_head.") for key in ckpt["policy_state_dict"]
    )
    policy = _build_policy(ckpt_args, device)
    pad_legacy_entity_type_rows(ckpt["policy_state_dict"], policy)
    missing, unexpected = policy.load_state_dict(
        ckpt["policy_state_dict"], strict=False,
    )
    allowed_missing = {"prev_action", "drc"}
    hard_missing = [
        k for k in missing
        if not any(tag in k.lower() for tag in allowed_missing)
    ]
    if hard_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch for {checkpoint_path}: "
            f"missing={hard_missing}, unexpected={unexpected}"
        )
    # Stamp the training-time net_constraint_obs on the policy so the eval
    # driver (eval.rollout.rl.eval_transformer) can verify the env it is given
    # matches — a hand-assembled env_kwargs that skips from_checkpoint would
    # otherwise silently eval a constraint-trained policy on zeroed NET
    # channels (or vice versa). Fallback False = the from_checkpoint semantics
    # (pre-knob checkpoints trained on all-zero constraint channels).
    policy.net_constraint_obs = bool(ckpt_args.get("net_constraint_obs", False))
    _print_provenance(ckpt, checkpoint_path)
    _check_obs_probe(ckpt, policy, checkpoint_path)
    policy.eval()
    iteration = int(ckpt.get("iteration", _checkpoint_iteration(checkpoint_path)))
    return policy, ckpt_args, iteration
