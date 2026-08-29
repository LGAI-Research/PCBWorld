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

    _diag_tmp = tempfile.mkdtemp(prefix="cadagent_test_crashlogs_")
    os.environ["KICAD_CRASH_LOG_DIR"] = _diag_tmp
    atexit.register(shutil.rmtree, _diag_tmp, True)

# --- Duration-aware file scheduling (xdist --dist loadfile) -----------------
# Wall-clock long poles (inductor compile ~7-13s, board-reload ~9s) must start
# early: xdist hands out file-scopes in collection order and pre-assigns two
# consecutive scopes per worker, so alphabetical order both starts them late and
# can chain two slow files onto one worker. Reorder from measured durations:
# zipper longest/shortest so each worker's initial pair is one long + one short
# file. Two duration sources, freshest wins per file:
#   tests/durations.json  — tracked seed, so fresh clones order well from run 1.
#                           Refresh with `pytest <paths> --update-durations`
#                           (merges just the files that ran; prunes deleted ones).
#                           check_docs.py fails on missing/stale entries, so new
#                           test files get an entry before push.
#   pytest cache          — this host's latest measurements, recorded every run.
# Workers read the same sources at collection, so xdist sees identical orderings.
_FILE_DURATIONS_KEY = "cadagent/file_durations"
_DURATIONS_SEED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "durations.json")
_this_run_durations: dict[str, float] = {}


def pytest_addoption(parser):
    parser.addoption(
        "--update-durations", action="store_true", dest="update_durations",
        help="merge this run's per-file wall times into tests/durations.json",
    )


def _load_durations_seed() -> dict:
    import json
    try:
        with open(_DURATIONS_SEED) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def pytest_collection_modifyitems(config, items):
    recorded = _load_durations_seed()
    cache = getattr(config, "cache", None)
    if cache is not None:
        recorded.update(cache.get(_FILE_DURATIONS_KEY, {}))
    if not recorded:
        return
    files, groups = [], {}
    for it in items:
        key = it.location[0]
        if key not in groups:
            groups[key] = []
            files.append(key)
        groups[key].append(it)
    files.sort(key=lambda k: -recorded.get(k, 0.0))
    woven, i, j = [], 0, len(files) - 1
    while i <= j:
        woven.append(files[i])
        i += 1
        if i <= j:
            woven.append(files[j])
            j -= 1
    items[:] = [it for key in woven for it in groups[key]]


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
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        seed = _load_durations_seed()
        seed.update({k: round(v, 1) for k, v in _this_run_durations.items()})
        seed = {k: v for k, v in seed.items() if os.path.exists(os.path.join(root, k))}
        with open(_DURATIONS_SEED, "w") as f:
            json.dump(seed, f, indent=1, sort_keys=True)
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
_rl_lib_path = os.path.join(_project_root, "build_rl", "pcbnew", "python", "rl")

if _rl_lib_path not in sys.path:
    sys.path.insert(0, _rl_lib_path)
_test_dir = os.path.dirname(os.path.abspath(__file__))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)


@pytest.fixture
def pool_kwargs():
    """Factory for a COMPLETE ``make_decoder_env_pool`` kwarg surface.

    The pool factory has no signature defaults for env-contract knobs (a
    missing one raises TypeError — see factory._REQ), so tests build the whole
    surface from ``RLEnvConfig.to_pool_kwargs()`` and override only what the
    test cares about (``seed`` is part of the bundle — override it via
    ``pool_kwargs(seed=...)``, never as a separate kwarg)::

        pool = make_decoder_env_pool(board, n_envs=2,
                                     **pool_kwargs(max_steps=20))
    """
    from tests.helpers.env_kwargs import full_env_kwargs

    def _make(**overrides):
        return full_env_kwargs(**overrides)

    return _make
