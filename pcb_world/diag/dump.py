"""Guard context dumps — full payload to disk right before a fail-fast raise.

Exception-path only: zero steady-state cost. torch is imported lazily so this
module (and pcb_world.diag) stays importable in torch-free environments
(eval pool, rule-based baselines); without torch the payload falls back to
pickle (.pkl instead of .pt).
"""
from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, NoReturn

from pcb_world.diag import default_log_dir, diag_enabled

#: payloads above this rough estimate drop their largest keys (recorded in the
#: .json summary) — a runaway obs batch must not fill the NFS volume.
MAX_DUMP_BYTES = 512 * 2**20


def dump_context(tag: str, *, log_dir: str | None = None, **payload: Any) -> str | None:
    """Serialize ``payload`` to ``<log_dir>/<ts>_<tag>_pid<pid>.pt`` (+ ``.json``).

    Returns the data-file path, or None when diagnostics are disabled or the
    dump fails — callers embed the result in their exception message and must
    raise regardless.
    """
    if not diag_enabled():
        return None
    try:
        return _dump(tag, log_dir, payload)
    except Exception:
        return None


def guard_fail(tag: str, message: str, /, **payload: Any) -> NoReturn:
    """Impossible-state guard exit: dump the context, then fail fast.

    Serializes ``payload`` via :func:`dump_context` and raises RuntimeError
    with the dump path appended to ``message`` (``dump=None`` when diagnostics
    are disabled or the dump failed — the guard raises regardless).
    """
    raise RuntimeError(f"{message} dump={dump_context(tag, **payload)}")


def _dump(tag: str, log_dir: str | None, payload: dict[str, Any]) -> str:
    try:
        import torch
    except ImportError:
        torch = None

    payload = {k: _sanitize(v, torch) for k, v in payload.items()}
    dropped = _enforce_size_cap(payload, torch)

    stem = os.path.join(
        log_dir or default_log_dir(),
        f"{time.strftime('%y%m%d_%H%M%S')}_{tag}_pid{os.getpid()}",
    )
    if torch is not None:
        data_path = stem + ".pt"
        torch.save(payload, data_path)
    else:
        import pickle

        data_path = stem + ".pkl"
        with open(data_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "tag": tag,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "role": os.environ.get("KICAD_DIAG_ROLE"),
        "data_file": os.path.basename(data_path),
        "keys": {k: _describe(v) for k, v in payload.items()},
        "dropped_keys": dropped,
    }
    with open(stem + ".json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return data_path


def _sanitize(obj: Any, torch) -> Any:
    """Detach/CPU tensors recursively; leave numpy & plain Python as-is."""
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _sanitize(v, torch) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize(v, torch) for v in obj)
    return obj


def _nbytes(obj: Any, torch) -> int:
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if hasattr(obj, "nbytes"):  # numpy
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_nbytes(v, torch) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v, torch) for v in obj)
    return 64  # scalars/strings — order of magnitude is enough for the cap


def _enforce_size_cap(payload: dict[str, Any], torch) -> list[str]:
    dropped: list[str] = []
    sizes = {k: _nbytes(v, torch) for k, v in payload.items()}
    while sum(sizes.values()) > MAX_DUMP_BYTES and sizes:
        biggest = max(sizes, key=sizes.get)
        payload[biggest] = f"<dropped: ~{sizes.pop(biggest)} bytes>"
        dropped.append(biggest)
    return dropped


def _describe(obj: Any) -> str:
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return f"{type(obj).__name__}{tuple(obj.shape)} {obj.dtype}"
    if isinstance(obj, (dict, list, tuple)):
        return f"{type(obj).__name__}[{len(obj)}]"
    return repr(obj)[:120]
