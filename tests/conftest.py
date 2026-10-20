"""Pytest configuration: ensure kicad_rl_router is importable."""

import os
import sys

import pytest

# Cap BLAS/OpenMP threads to 1 for every test process (serial and xdist workers).
# torch/numpy default to one thread per core at import; on this suite's tiny tensors
# the per-op fork/join overhead makes that ~10x slower than single-threaded, and
# under xdist 16 workers would spawn 16x64 threads and oversubscribe the host.
# Must run before torch/numpy are first imported — conftest loads before any test
# module. setdefault keeps an explicit OMP_NUM_THREADS=<n> from the caller intact.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

# Crash-diagnostics isolation: several tests deliberately crash/kill env workers
# (recovery, postmortem paths), and without this every suite run would litter
# the repo's var/crashlogs with their artifacts. Route all pcb_world.diag
# consumers (CrashLogger, worker installs, dumps, postmortems) to a throwaway
# dir instead; the controller sets it before xdist workers spawn (they inherit),
# and removes it at exit. The pytest processes' OWN fatal-signal logs still go
# to var/crashlogs — pytest_configure passes that log_dir explicitly — so real
# suite segfaults keep leaving evidence.
if "KICAD_CRASH_LOG_DIR" not in os.environ:
    import atexit
    import shutil
    import tempfile

    _diag_tmp = tempfile.mkdtemp(prefix="pcbworld_test_crashlogs_")
    os.environ["KICAD_CRASH_LOG_DIR"] = _diag_tmp
    atexit.register(shutil.rmtree, _diag_tmp, True)

# --- Duration-aware file scheduling (xdist --dist loadfile) -----------------
# Wall-clock long poles (inductor compile ~8-20s, board-reload ~6s) must start
# early: with --no-loadscope-reorder (pyproject addopts) xdist hands out
# file-scopes in collection order — one per worker, then, in worker order, a
# second one to each worker whose first file has at most 2 tests left
# (xdist's _reschedule), then each next file to whichever worker drains first.
# Reorder from measured durations: longest first, with the next n files (n =
# worker count) reversed so gw0, holding the longest file, is pre-given the
# shortest of them when that file has at most 2 tests (today
# test_flex_attn_compiled.py, one test). Two duration sources, freshest wins per file:
#   durations.json   — tracked seed per test tree (the nearest one above each
#                      file, e.g. tests/durations.json), so fresh clones and
#                      worktrees order well from run 1. Refresh with
#                      `pytest <paths> --update-durations` (merges the files that
#                      ran into their own tree's seed; prunes deleted ones).
#   pytest cache     — this host's latest measurements, recorded every run.
# Workers read the same sources at collection, so xdist sees identical orderings.
_FILE_DURATIONS_KEY = "pcbworld/file_durations"
_SEED_NAME = "durations.json"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_this_run_durations: dict[str, float] = {}


def pytest_addoption(parser):
    parser.addoption(
        "--update-durations", action="store_true", dest="update_durations",
        help="merge this run's per-file wall times into each test tree's durations.json",
    )


def _seed_for(rel_file: str, root: str = _REPO_ROOT) -> str | None:
    """The durations seed of the test tree ``rel_file`` (repo-relative) belongs to:
    the nearest ``durations.json`` in its directory or above, else None."""
    d = os.path.dirname(rel_file)
    while d:
        path = os.path.join(root, d, _SEED_NAME)
        if os.path.isfile(path):
            return path
        d = os.path.dirname(d)
    return None


def _load_seed(path: str) -> dict:
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _schedule_order(files: list[str], recorded: dict, n_workers: int) -> list[str]:
    """Longest recorded file first; the second block of ``n_workers`` reversed."""
    order = sorted(files, key=lambda k: -recorded.get(k, 0.0))
    order[n_workers:2 * n_workers] = order[n_workers:2 * n_workers][::-1]
    return order


def pytest_collection_modifyitems(config, items):
    files, groups = [], {}
    for it in items:
        key = it.location[0]
        if key not in groups:
            groups[key] = []
            files.append(key)
        groups[key].append(it)
    recorded = {}
    for seed in sorted({s for s in map(_seed_for, files) if s}):
        recorded.update(_load_seed(seed))
    cache = getattr(config, "cache", None)
    if cache is not None:
        recorded.update(cache.get(_FILE_DURATIONS_KEY, {}))
    if not recorded:
        return
    n = getattr(config, "workerinput", {}).get("workercount", 0)
    items[:] = [it for key in _schedule_order(files, recorded, n) for it in groups[key]]


def pytest_runtest_logreport(report):
    # On the xdist controller this receives every worker's reports (setup/call/
    # teardown), summing to per-file wall time; workers accumulate but never write.
    key = report.location[0]
    _this_run_durations[key] = _this_run_durations.get(key, 0.0) + report.duration


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if hasattr(config, "workerinput"):   # xdist worker — controller persists
        return
    if not _this_run_durations:
        return
    cache = getattr(config, "cache", None)
    if cache is not None:
        merged = cache.get(_FILE_DURATIONS_KEY, {})
        merged.update({k: round(v, 3) for k, v in _this_run_durations.items()})
        cache.set(_FILE_DURATIONS_KEY, merged)
    if config.getoption("update_durations", default=False):
        import json
        by_seed: dict[str, dict] = {}
        for k, v in _this_run_durations.items():
            seed = _seed_for(k)
            if seed is None:
                config.pluginmanager.get_plugin("terminalreporter").write_line(
                    f"--update-durations: no {_SEED_NAME} above {k} — not recorded", yellow=True)
                continue
            by_seed.setdefault(seed, {})[k] = round(v, 1)
        for seed, ran in by_seed.items():
            merged = _load_seed(seed)
            merged.update(ran)
            merged = {k: v for k, v in merged.items()
                      if os.path.exists(os.path.join(_REPO_ROOT, k))}
            with open(seed, "w") as f:
                json.dump(merged, f, indent=1, sort_keys=True)
                f.write("\n")


# --- Crash logging (always-on fatal-signal diagnostics) ----------------------
# The C++ router intermittently segfaults a worker (~0.4%/full run as of 2607),
# and xdist swallows the worker's stderr, leaving no trace. So every pytest
# process (controller, xdist worker, -n 0) logs fatal signals to
# var/crashlogs/<ts>_<worker>_<pid>.log via the shared pcb_world.diag module
# (native C++ backtrace from crashtrace.c chained to Python's faulthandler).
# A clean session removes its own (empty) log at exit; a crashed process never
# gets there, so its log survives for post-hoc analysis.

@pytest.hookimpl(trylast=True)  # after pytest's own faulthandler plugin, so we chain onto it
def pytest_configure(config):
    try:
        from pcb_world.diag import install_crash_handler

        config._crashlog = install_crash_handler(
            role=os.environ.get("PYTEST_XDIST_WORKER", "main"),
            log_dir=os.path.join(_project_root, "var", "crashlogs"),
            with_host=False,       # keep the legacy <ts>_<worker>_<pid> naming
            register_atexit=False,  # pytest_unconfigure is the teardown
        )
    except Exception:
        pass  # diagnostics must never break the suite


def pytest_unconfigure(config):
    entry = getattr(config, "_crashlog", None)
    if entry is None:
        return
    from pcb_world.diag import remove_log_if_empty

    remove_log_if_empty(*entry)


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_test_dir = os.path.dirname(os.path.abspath(__file__))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

# The engine build the tests import: the ONE resolver the runtime uses (explicit variable, this
# tree's build_rl, else a discovered build of this tree's C++) — it puts the lib dir on sys.path.
from pcb_world.engine import router_lib_dir  # noqa: E402  (after sys.path above)

router_lib_dir()


@pytest.fixture
def pool_kwargs():
    """Factory for COMPLETE ``make_decoder_env_pool`` kwargs (``env_config``, ``seed``,
    ``policy_net_select`` — :func:`tests.helpers.env_kwargs.full_env_kwargs`); override
    only what the test cares about. The pool's ``advance_rng_on_reload`` is passed
    explicitly::

        pool = make_decoder_env_pool(board, n_envs=2, advance_rng_on_reload=False,
                                     **pool_kwargs(max_steps=20))
    """
    from tests.helpers.env_kwargs import full_env_kwargs

    def _make(**overrides):
        return full_env_kwargs(**overrides)

    return _make


@pytest.fixture(params=["ipc", "in_process"])
def engine_mode(request, monkeypatch):
    """Run the test over both engine transports: the IPC engine server (the default)
    and in-process (``KICAD_ENGINE_IPC=0``)."""
    monkeypatch.setenv("KICAD_ENGINE_IPC", "1" if request.param == "ipc" else "0")
    return request.param
