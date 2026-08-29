"""Fatal-signal crash logs: native C++ backtrace + Python stacks, per process.

Extracted from tests/conftest.py so training/eval processes get the same
coverage as pytest. On SIGSEGV/SIGABRT/SIGBUS/SIGILL/SIGFPE the dlopen'd
crashtrace.so appends the native backtrace to the log file, then chains to
Python's faulthandler for the Python stacks (see crashtrace.c). The log is
opened "a" (O_APPEND) so faulthandler's writes land after the native part.

A clean exit removes the (empty) log; a crashed process never gets there, so
its log survives. Callers in multiprocessing workers must pass
``register_atexit=False`` and call :func:`remove_log_if_empty` on their clean
exit path themselves — children exit via os._exit and skip atexit.
"""
from __future__ import annotations

import atexit
import ctypes
import faulthandler
import os
import subprocess
from typing import TextIO

from pcb_world.diag import ENV_ROLE, ENV_STEM, artifact_stem, default_log_dir, diag_enabled

_installed: tuple[str, TextIO] | None = None


def install_crash_handler(
    role: str,
    log_dir: str | None = None,
    *,
    stem: str | None = None,
    with_host: bool = True,
    register_atexit: bool = True,
) -> tuple[str, TextIO] | None:
    """Install per-process fatal-signal logging; returns (log_path, file).

    Idempotent per process (second call returns the first result). Fail-soft:
    without gcc the native backtrace is skipped and faulthandler alone writes
    the Python stacks; if even that fails, returns None. Never raises.
    """
    global _installed
    if not diag_enabled():
        return None
    if _installed is not None:
        return _installed
    try:
        log_dir = log_dir or default_log_dir()
        stem = stem or artifact_stem(role, with_host=with_host)
        # Overwrite, never setdefault: children inherit the parent's exported
        # stem via environ, and a per-process stem must win over that lie.
        os.environ[ENV_STEM] = stem
        os.environ[ENV_ROLE] = role
        path = os.path.join(log_dir, stem + ".log")
        log = open(path, "a")
        faulthandler.enable(file=log, all_threads=True)
        _installed = (path, log)
    except Exception:
        return None
    try:  # native part — degrade silently to faulthandler-only (e.g. no gcc)
        so = _compile_crashtrace(log_dir)
        os.environ["CRASHTRACE_FILE"] = path
        ctypes.CDLL(so)  # constructor installs handlers, chaining to faulthandler
    except Exception:
        pass
    if register_atexit:
        atexit.register(remove_log_if_empty, path, log)
    return _installed


def remove_log_if_empty(path: str, log: TextIO) -> None:
    """Clean-exit teardown: no crash happened, leave no trace."""
    try:
        faulthandler.disable()
        if not log.closed:
            log.close()
        if os.path.getsize(path) == 0:
            os.remove(path)
    except OSError:
        pass


def _compile_crashtrace(log_dir: str) -> str:
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crashtrace.c")
    so = os.path.join(log_dir, "crashtrace.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        tmp = f"{so}.{os.getpid()}.tmp"  # processes may race-compile: build + atomic replace
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O1", "-o", tmp, src],
            check=True, capture_output=True,
        )
        os.replace(tmp, so)
    return so
