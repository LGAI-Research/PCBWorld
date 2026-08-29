"""Asynchronous (off-node) validation over the regular training checkpoints.

``--async-val`` detaches the in-training 3-set validation (``val`` +
``eval2``..``eval5``) from the training process. There is NO separate
"validation checkpoint" — the watcher evaluates the ordinary periodic
checkpoints (``<save_dir>/policy_iter_<N>.pt``), so checkpoint saving stays a
single code path (``_train_ckpt_payload``/``_save_ckpt``). The trainer only
guarantees a checkpoint exists at every eval cadence, which is enforced by
``eval_every % save_freq == 0`` (plus an initial ``policy_iter_0.pt`` when
``--eval-at-init``); a **watcher** process — typically on another, cheaper GPU
node sharing the filesystem — picks cadence checkpoints up, runs the SAME
evaluators (:func:`methods.rl_agent.training.loop.build_evaluators`), and
writes a result json back. The trainer polls results every iteration and logs
them through its OWN TensorBoard/W&B sinks, so train and val land in one run
with a single writer (W&B cannot log to past steps from a second process — see
``MultiLogger.add_scalars_async``). Best-ckpt selection also stays in the
trainer, fed by the returned ``overall`` metrics.

On-disk layout::

    <save_dir>/policy_iter_<N>.pt        the regular ckpts (evaluated in place,
                                         never deleted by the async machinery)
    <save_dir>/async_val/results/iter_{N:06d}.json   watcher result
    <save_dir>/async_val/results/*.json.done         consumed by the trainer
    <save_dir>/async_val/train_done      trainer finished; watcher exits once
                                         no cadence ckpt is left unevaluated

Work discovery is stateless: a cadence checkpoint (iter 0 or a multiple of
``eval_every``) with no result file is pending — for the watcher (what to
evaluate next) and for the trainer's teardown drain (what to wait for) alike.
Result writes are atomic (tmp + ``os.replace``); either process can restart
and re-derive its state from the files. Validation lags training by however
long one eval pass takes, but each result is logged at its true iteration, so
the curves are identical to inline eval up to eval-time nondeterminism (the
watcher runs the policy fp32/eager — ``--bf16``/``--compile-*`` are
training-process knobs).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ASYNC_DIRNAME = "async_val"
_TRAIN_DONE = "train_done"
# teardown drain: how long the trainer waits for the watcher to finish the
# last queued validations before abandoning them (files stay on disk).
DRAIN_TIMEOUT_S = 4 * 3600.0


def _res_name(iteration: int) -> str:
    return f"iter_{iteration:06d}.json"


def _ckpt_iter(ckpt_path: Path) -> int:
    return int(ckpt_path.stem.removeprefix("policy_iter_"))


def load_recorded_val_env(save_dir: str | Path) -> dict | None:
    """Recorded val-env kwargs from ``<save_dir>/env_records/val_env.yaml``.

    The trainer records the resolved val kwargs at startup; the watcher hands
    them to ``build_evaluators`` as ``expect_env_kwargs`` so the async path
    gets the same record-vs-actual cross-check the inline path has
    (``assert_env_kwargs_as_recorded``). Returns None — with a loud note, the
    check is then skipped — only for a save_dir predating the env-records
    feature.
    """
    import yaml
    from methods._shared.config_dump import ENV_RECORDS_DIR

    path = Path(save_dir) / ENV_RECORDS_DIR / "val_env.yaml"
    if not path.is_file():
        print(
            f"[watcher] {path} missing — skipping the env-kwargs cross-check "
            "(save_dir predates env_records?)", flush=True,
        )
        return None
    return yaml.safe_load(path.read_text())


def pending_evals(save_dir: str | Path, eval_every: int) -> list[tuple[int, Path]]:
    """Cadence checkpoints with no result yet, ascending by iteration.

    The single work-discovery rule, shared by the watcher (next request) and
    the trainer's teardown drain (what is still outstanding): a
    ``policy_iter_<N>.pt`` is pending iff N is on the eval cadence (iter 0 —
    which every run snapshots — or a multiple of ``eval_every``) and
    ``results/iter_N.json[.done]`` is absent.
    """
    save_dir = Path(save_dir)
    evaluated = {
        p.name.split(".")[0]  # iter_000123(.json|.json.done) -> iter_000123
        for p in (save_dir / ASYNC_DIRNAME / "results").glob("iter_*.json*")
    }
    out = []
    for p in save_dir.glob("policy_iter_*.pt"):
        try:
            n = _ckpt_iter(p)
        except ValueError:
            continue
        if (n == 0 or n % eval_every == 0) and f"iter_{n:06d}" not in evaluated:
            out.append((n, p))
    return sorted(out)


class AsyncValResults:
    """Trainer-side handle for the results channel (rank 0 / main process)."""

    def __init__(self, save_dir: str | Path) -> None:
        self.save_dir = Path(save_dir)
        self.dir = self.save_dir / ASYNC_DIRNAME
        self.results_dir = self.dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Stale marker from a previous run of this save_dir (--resume) would
        # make a fresh watcher exit immediately.
        (self.dir / _TRAIN_DONE).unlink(missing_ok=True)

    def ckpt_path(self, iteration: int) -> Path:
        """The regular periodic ckpt the watcher evaluated for this result."""
        return self.save_dir / f"policy_iter_{iteration}.pt"

    def poll_results(self) -> list[dict]:
        """Unconsumed watcher results, oldest iteration first."""
        return [
            json.loads(p.read_text())
            for p in sorted(self.results_dir.glob("iter_*.json"))
        ]

    def consume(self, iteration: int) -> None:
        """Mark a result as logged (kept as ``.done`` audit; ckpt untouched)."""
        res = self.results_dir / _res_name(iteration)
        os.replace(res, res.with_name(res.name + ".done"))

    def n_pending(self, eval_every: int) -> int:
        return len(pending_evals(self.save_dir, eval_every))

    def mark_train_done(self) -> None:
        (self.dir / _TRAIN_DONE).touch()


# ============================================================================
# Watcher side
# ============================================================================


def watcher_main(
    save_dir: str | Path,
    *,
    device: str = "auto",
    n_envs: int | None = None,
    poll_interval: float = 10.0,
) -> None:
    """Evaluate cadence checkpoints until the trainer signals ``train_done``.

    Bootstraps the policy, the evaluators, and ``eval_every`` ONCE from the
    first checkpoint's saved training args (later checkpoints only swap the
    ``state_dict``), so the per-request cost is one ``load_state_dict`` plus
    the eval rollouts. ``n_envs`` overrides the training-time value for the
    eval pool — size it to THIS node's cores/VRAM.
    """
    import gc

    import torch

    from methods.rl_agent.models.loader import pad_legacy_entity_type_rows
    from methods.rl_agent.training.utils import auto_device

    save_dir = Path(save_dir)
    results_dir = save_dir / ASYNC_DIRNAME / "results"
    dev = auto_device(device)
    agent = None
    eval_sets = None  # [(prefix, Evaluator)] — "val" first
    eval_every = None

    print(f"[watcher] polling {save_dir} (device={dev})", flush=True)
    while True:
        if eval_every is None:
            # Bootstrap from any regular ckpt (args are identical across iters).
            ckpts = sorted(save_dir.glob("policy_iter_*.pt"), key=_ckpt_iter)
            if not ckpts:
                time.sleep(poll_interval)
                continue
            from types import SimpleNamespace

            from methods.rl_agent.policy.agent import KiCadRLAgent
            from methods.rl_agent.training.loop import (
                build_evaluators, load_eval_boards,
            )

            agent = KiCadRLAgent.from_checkpoint(ckpts[0], dev)
            args = SimpleNamespace(**torch.load(ckpts[0], map_location="cpu")["args"])
            if args.eval_every is None:
                raise RuntimeError(
                    "--async-val checkpoints carry eval_every=None; the "
                    "trainer should have rejected this (explicit --eval-every "
                    "is required)"
                )
            eval_every = int(args.eval_every)
            if n_envs is not None:
                args.n_envs = n_envs
            primary, extras = build_evaluators(
                args, agent, dev, load_eval_boards(args),
                expect_env_kwargs=load_recorded_val_env(save_dir),
            )
            eval_sets = [("val", primary)] + list(extras)
            print(
                f"[watcher] evaluators ready: "
                f"{', '.join(pfx for pfx, _ in eval_sets)} "
                f"(eval_every={eval_every}, n_envs={args.n_envs})",
                flush=True,
            )

        pending = pending_evals(save_dir, eval_every)
        if not pending:
            if (save_dir / ASYNC_DIRNAME / _TRAIN_DONE).exists():
                print("[watcher] train_done + nothing pending -> exit", flush=True)
                return
            time.sleep(poll_interval)
            continue

        iteration, ckpt = pending[0]
        state = torch.load(ckpt, map_location="cpu")["policy_state_dict"]
        # Watcher on new code consuming a pre-OBSTACLE trainer's checkpoints:
        # pad the 14-row entity-type table (strict load hard-errors on size
        # mismatch; the bootstrap from_checkpoint path is already padded).
        pad_legacy_entity_type_rows(state, agent.model)
        agent.model.load_state_dict(state)

        scalars: dict[str, float] = {}
        overall = None
        for pfx, ev in eval_sets:
            t0 = time.time()
            summary = ev.run()
            # Same wall-clock fold as Trainer.validate -> <prefix>/eval_time_sec.
            summary.overall["eval_time_sec"] = time.time() - t0
            scalars.update(
                summary.to_logger_dict(prefix=pfx, include_per_board=False)
            )
            if pfx == "val":
                overall = summary.overall

        from eval.metrics import _jsonable

        result = {
            "iteration": iteration,
            "scalars": scalars,
            "overall": _jsonable(overall),
        }
        # CADAGENT_ALLOW_OBS_MISMATCH runs: stamp the mismatch into every
        # result so aggregation can refuse/flag it — the load-time warning
        # alone is volatile (scrolls away), the artifact is not.
        mismatch = getattr(agent.model, "obs_schema_mismatch", None)
        if mismatch:
            result["obs_schema_mismatch"] = mismatch
        res_path = results_dir / _res_name(iteration)
        tmp = res_path.with_name(res_path.name + ".tmp")
        tmp.write_text(json.dumps(result))
        os.replace(tmp, res_path)
        print(
            f"[watcher] iter {iteration}: {len(scalars)} scalars "
            f"(val/eval_time_sec={overall.get('eval_time_sec', 0):.1f}s)",
            flush=True,
        )
        # Same residual-VRAM release as the inline path (_run_dual_eval).
        gc.collect()
        torch.cuda.empty_cache()
