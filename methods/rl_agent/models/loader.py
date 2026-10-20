"""Checkpoint -> policy builder for the eval pipeline.

The canonical "load a trained ``KiCadRLModel`` from a ``.pt`` checkpoint"
routine, shared by the eval rollout path
(``methods.rl_agent.rollout.transformer.load_policy_from_ckpt``) and per-board eval
scripts. The env is built from ``RLEnvConfig.from_checkpoint``. ``torch`` / ``KiCadRLModel`` are
imported lazily inside the functions so importing this module stays free of an
eager torch dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


# ============================================================================
# Checkpoint -> policy reconstruction
# ============================================================================


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


def _build_policy(ckpt: Mapping[str, Any], device: Any) -> Any:
    from configs.loader.schema import RLPolicyConfig

    return RLPolicyConfig.from_checkpoint(ckpt).build(device)


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
    ``PCBWORLD_ALLOW_OBS_MISMATCH=1`` downgrades to a loud warning and stamps
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
        "eval with the training-era code. Set PCBWORLD_ALLOW_OBS_MISMATCH=1 "
        "to proceed anyway (results are stamped obs_schema_mismatch)."
    )
    if os.environ.get("PCBWORLD_ALLOW_OBS_MISMATCH") == "1":
        print(f"[ckpt][WARNING] {msg}")
        policy.obs_schema_mismatch = reason
    else:
        raise RuntimeError(msg)


def _load_policy(
    checkpoint_path: Path,
    device: Any,
    *,
    strict: bool,
) -> tuple[Any, Mapping[str, Any], int]:
    """Load a trained policy + its resolved config + iteration from a .pt checkpoint.

    The config is :func:`configs.loader.ckpt.load_ckpt_config`'s ``values``
    (``strict`` = ``--strict-ckpt``): every stored arg plus every config field,
    weight-shaped ones derived from the state_dict."""
    import torch

    from configs.loader.ckpt import load_ckpt_config

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args, sources = load_ckpt_config(ckpt, strict=strict)
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
    # channels (or vice versa).
    policy.net_constraint_obs = bool(ckpt_args["net_constraint_obs"])
    # The loader's per-key sources (ckpt / legacy:* / derived): the provenance of the CKPT
    # layer when an eval resolves its env config over this checkpoint (resolve_eval_env).
    policy.ckpt_sources = sources
    _print_provenance(ckpt, checkpoint_path)
    _check_obs_probe(ckpt, policy, checkpoint_path)
    policy.eval()
    iteration = int(ckpt.get("iteration", _checkpoint_iteration(checkpoint_path)))
    return policy, ckpt_args, iteration
