"""Forkserver children must survive numpy's first import (``pcb_world.vec.get_mp_context``).

On macOS a forkserver child that imports numpy itself fails at random with
``RuntimeError: Could not parse python long as longdouble`` and the vec-env
worker dies before building its env. Preloading numpy in the forkserver avoids it.
"""
from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
import textwrap

import pytest

from pcb_world.vec import (
    FORKSERVER_PRELOAD,
    add_forkserver_preload,
    get_mp_context,
)


def test_preload_stays_free_of_gpl_and_kicad_modules():
    assert "numpy" in FORKSERVER_PRELOAD
    for name in FORKSERVER_PRELOAD:
        assert "kicad" not in name and "engine" not in name


@pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods(), reason="no forkserver"
)
def test_forkserver_children_import_numpy_reliably():
    # A fresh interpreter, so this test owns the (process-wide) forkserver.
    script = textwrap.dedent(
        """
        import importlib

        from pcb_world.vec import get_mp_context

        if __name__ == "__main__":
            ctx = get_mp_context("forkserver")
            bad = 0
            for _ in range(5):
                ps = [ctx.Process(
                    target=importlib.import_module, args=("numpy",), daemon=True) for _ in range(4)]
                for p in ps:
                    p.start()
                for p in ps:
                    p.join(60)
                bad += sum(p.exitcode != 0 for p in ps)
            print("BAD", bad)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "BAD 0" in out.stdout, out.stdout + out.stderr[-2000:]


def test_spawn_context_is_untouched():
    assert get_mp_context("spawn").get_start_method() == "spawn"


@pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods(), reason="no forkserver"
)
def test_preload_merges_with_what_is_already_registered():
    """Ours is added to the global list, not swapped in for it.

    The speed profiler installs its worker shim through the same list
    (``tools/diagnostics/speed_profiler/driver.py``), before the trainer builds
    the vec env; replacing the list there would silently drop worker timing.
    """
    from multiprocessing import forkserver

    saved = list(forkserver._forkserver._preload_modules)
    shim = "tools.diagnostics.speed_profiler.worker_shim"
    try:
        mp.set_forkserver_preload([shim])
        get_mp_context("forkserver")
        now = list(forkserver._forkserver._preload_modules)
        assert shim in now, now
        assert "numpy" in now, now
    finally:
        mp.set_forkserver_preload(saved)


@pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods(), reason="no forkserver"
)
def test_preload_does_not_carry_cpythons_main_default():
    """With nobody else registered the list is exactly ours.

    CPython seeds ``_preload_modules`` with ``__main__``; carrying that over
    would make the forkserver import the entry script's module body -- the
    opposite of the "pure-Python, no KiCad state" rule the list exists for.
    """
    from multiprocessing import forkserver

    saved = list(forkserver._forkserver._preload_modules)
    try:
        mp.set_forkserver_preload(["__main__"])     # CPython's own default
        get_mp_context("forkserver")
        assert list(forkserver._forkserver._preload_modules) == list(FORKSERVER_PRELOAD)
    finally:
        mp.set_forkserver_preload(saved)


@pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods(), reason="no forkserver"
)
def test_add_forkserver_preload_is_the_merging_register():
    """What the speed profiler calls: neither side drops the other."""
    from multiprocessing import forkserver

    saved = list(forkserver._forkserver._preload_modules)
    shim = "tools.diagnostics.speed_profiler.worker_shim"
    try:
        mp.set_forkserver_preload(["__main__"])
        add_forkserver_preload(shim)
        get_mp_context("forkserver")
        now = list(forkserver._forkserver._preload_modules)
        assert now == [shim, *FORKSERVER_PRELOAD], now
    finally:
        mp.set_forkserver_preload(saved)
