"""pcb_world.engine.provenance — the router a process is about to load must have been built
from this tree's engine/kicad-patches/ (ENGINE_CPP_HASH stamp == source hash).

Pure unit tests on a fake engine root (a copy of the real hash script + a small patch tree) and
a fake router directory; no C++ is loaded. Covers: match (silent), mismatch (error), mismatch
with the override (warning), a stampless router (one warning), an engine bundle without the
script (one warning, then no-op), a script that fails (error carrying its stderr), a missing
router dir (no-op), once-per-process memoisation, the build resolver, and the loader's one
resolver/one guard (``pcb_world.engine.router_lib_dir`` / ``ensure_router_provenance``).
"""
from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import pytest

from pcb_world.engine import provenance
from pcb_world.engine.router_client import engine_home

REAL_TOOL = Path(engine_home()) / "tools" / "cpp_content_hash.sh"


@pytest.fixture
def engine_root(tmp_path: Path) -> Path:
    root = tmp_path / "engine"
    (root / "tools").mkdir(parents=True)
    shutil.copy2(REAL_TOOL, root / "tools" / "cpp_content_hash.sh")
    (root / "kicad-patches" / "rl").mkdir(parents=True)
    (root / "kicad-patches" / "rl" / "x.cpp").write_text("int x = 1;\n", encoding="utf-8")
    (root / "kicad-patches" / "ENGINE_VERSION").write_text("1.4\n", encoding="utf-8")
    return root


@pytest.fixture
def lib_dir(tmp_path: Path) -> Path:
    d = tmp_path / "build" / "pcbnew" / "python" / "rl"
    d.mkdir(parents=True)
    (d / "kicad_rl_router.so").write_bytes(b"not really\n")
    return d


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    """Every test is 'a new process': no verified pair, no cached hash, no cached main root."""
    monkeypatch.setattr(provenance, "_checked", {})
    monkeypatch.setattr(provenance, "_source_hashes", {})
    monkeypatch.setattr(provenance, "_main_roots", {})
    monkeypatch.delenv(provenance.ALLOW_ENV, raising=False)


def _stamp(lib_dir: Path, value: str) -> None:
    (lib_dir / provenance.STAMP_NAME).write_text(value + "\n", encoding="utf-8")


def test_matching_stamp_is_silent(engine_root, lib_dir):
    _stamp(lib_dir, provenance.source_hash(str(engine_root)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        provenance.check_router_matches_sources(str(lib_dir), str(engine_root))


def test_mismatch_raises_with_both_hashes(engine_root, lib_dir):
    _stamp(lib_dir, "deadbeef")
    with pytest.raises(provenance.RouterProvenanceError) as ei:
        provenance.check_router_matches_sources(str(lib_dir), str(engine_root))
    msg = str(ei.value)
    assert "deadbeef" in msg and provenance.source_hash(str(engine_root)) in msg
    assert "build_rl_router.sh" in msg


def test_editing_cpp_after_the_build_is_a_mismatch(engine_root, lib_dir):
    _stamp(lib_dir, provenance.source_hash(str(engine_root)))
    (engine_root / "kicad-patches" / "rl" / "x.cpp").write_text("int x = 2;\n", encoding="utf-8")
    provenance._source_hashes.clear()      # the hash is per process: the edit is seen by a NEW one
    with pytest.raises(provenance.RouterProvenanceError):
        provenance.check_router_matches_sources(str(lib_dir), str(engine_root))


def test_source_hash_is_computed_once_per_process(engine_root, monkeypatch):
    """A refused guard re-raises on retry without re-hashing (the tree a running process sees
    is the one it started from)."""
    import subprocess as sp
    first = provenance.source_hash(str(engine_root))
    monkeypatch.setattr(sp, "run", lambda *a, **k: pytest.fail("hash script re-run in the same process"))
    assert provenance.source_hash(str(engine_root)) == first


def test_override_downgrades_to_a_warning(engine_root, lib_dir, monkeypatch):
    _stamp(lib_dir, "deadbeef")
    monkeypatch.setenv(provenance.ALLOW_ENV, "1")
    with pytest.warns(RuntimeWarning, match="deadbeef"):
        provenance.check_router_matches_sources(str(lib_dir), str(engine_root))


def test_stampless_router_warns_once_and_passes(engine_root, lib_dir):
    with pytest.warns(RuntimeWarning, match="carries no ENGINE_CPP_HASH"):
        provenance.check_router_matches_sources(str(lib_dir), str(engine_root))
    with warnings.catch_warnings():                       # memoised: second call is silent
        warnings.simplefilter("error")
        provenance.check_router_matches_sources(str(lib_dir), str(engine_root))


def test_bundle_without_the_script_warns_once_then_passes(tmp_path, lib_dir):
    """An older engine bundle: the guard cannot run — said once per process, never silently."""
    _stamp(lib_dir, "deadbeef")
    old_root = tmp_path / "old_engine"
    old_root.mkdir()
    with pytest.warns(RuntimeWarning, match="no tools/cpp_content_hash.sh"):
        provenance.check_router_matches_sources(str(lib_dir), str(old_root))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        provenance.check_router_matches_sources(str(lib_dir), str(old_root))


def test_failing_hash_script_is_an_error_carrying_its_stderr(tmp_path, lib_dir):
    """A script that fails must not fold into 'no script' (which would switch the guard off)."""
    _stamp(lib_dir, "deadbeef")
    root = tmp_path / "broken_engine"
    (root / "tools").mkdir(parents=True)
    tool = root / "tools" / "cpp_content_hash.sh"
    tool.write_text("#!/usr/bin/env bash\necho 'sha256sum: not found here' >&2\nexit 1\n", encoding="utf-8")
    with pytest.raises(provenance.RouterProvenanceError, match="sha256sum: not found here"):
        provenance.source_hash(str(root))
    with pytest.raises(provenance.RouterProvenanceError, match="cannot be hashed"):
        provenance.check_router_matches_sources(str(lib_dir), str(root))
    shutil.copy2(REAL_TOOL, tool)                          # a fixed script is picked up: failures are not memoised
    (root / "kicad-patches").mkdir()
    assert provenance.source_hash(str(root))


def test_missing_router_dir_is_a_noop(engine_root, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        provenance.check_router_matches_sources(str(tmp_path / "nowhere"), str(engine_root))


def test_the_real_tree_and_its_router_agree():
    """Integration: whatever build this tree loads must match this tree. A router without the
    stamp cannot pass the suite: the runtime tolerates one (a warning, for old builds in the
    field), the suite is the gate that asks for the rebuild."""
    from pcb_world.engine import engine_available, router_lib_dir
    if not engine_available():
        pytest.skip("no router build reachable from this tree")
    lib = router_lib_dir()
    if not os.path.isfile(os.path.join(lib, provenance.STAMP_NAME)):
        pytest.fail(f"the router at {lib} carries no {provenance.STAMP_NAME} — {provenance.REBUILD_HINT}")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        provenance.check_router_matches_sources(lib, engine_home())


# --------------------------------------------------------------------------- resolver

def _fake_build(root: Path, name: str, stamp: str | None) -> Path:
    d = root / name / "pcbnew" / "python" / "rl"
    d.mkdir(parents=True)
    (d / "kicad_rl_router.so").write_bytes(b"x\n")
    if stamp is not None:
        (d / provenance.STAMP_NAME).write_text(stamp + "\n", encoding="utf-8")
    return root / name


def test_resolver_prefers_the_stamp_that_matches_this_tree(engine_root, tmp_path, monkeypatch):
    project = tmp_path / "wt"
    project.mkdir()
    monkeypatch.setattr(provenance, "main_root", lambda root: str(project))   # no git here
    mine = provenance.source_hash(str(engine_root))
    (project / "var" / "builds").mkdir(parents=True)
    _fake_build(project / "var" / "builds", "1.4-aaaaaaaa", "deadbeef")       # other C++
    good = _fake_build(project / "var" / "builds", "1.4-bbbbbbbb", mine)        # ours
    found, why = provenance.resolve_build_dir(str(project), str(engine_root))
    assert found == str(good) and why is None


def test_resolver_never_takes_a_build_of_other_cpp(engine_root, tmp_path, monkeypatch):
    project = tmp_path / "wt"
    project.mkdir()
    monkeypatch.setattr(provenance, "main_root", lambda root: str(project))
    _fake_build(project, "build_rl", "deadbeef")
    found, why = provenance.resolve_build_dir(str(project), str(engine_root))
    assert found is None and "deadbeef" in why and "build_rl_router.sh" in why


def test_resolver_falls_back_to_a_stampless_build_only_without_a_match(engine_root, tmp_path, monkeypatch):
    project = tmp_path / "wt"
    project.mkdir()
    monkeypatch.setattr(provenance, "main_root", lambda root: str(project))
    legacy = _fake_build(project, "build_rl", None)
    found, why = provenance.resolve_build_dir(str(project), str(engine_root))
    assert found == str(legacy) and why is None            # usable (the guard warns once about the missing stamp)
    (project / "var" / "builds").mkdir(parents=True)
    good = _fake_build(project / "var" / "builds", "1.4-cccccccc", provenance.source_hash(str(engine_root)))
    assert provenance.resolve_build_dir(str(project), str(engine_root))[0] == str(good)


def test_resolver_searches_the_main_clone_from_a_worktree(engine_root, tmp_path, monkeypatch):
    main, wt = tmp_path / "main", tmp_path / "wt"
    main.mkdir(); wt.mkdir()
    monkeypatch.setattr(provenance, "main_root", lambda root: str(main))
    good = _fake_build(main, "build_rl", provenance.source_hash(str(engine_root)))
    found, _ = provenance.resolve_build_dir(str(wt), str(engine_root))
    assert found == str(good)


def test_resolver_with_nothing_built_has_nothing_to_say(engine_root, tmp_path, monkeypatch):
    """No candidate anywhere: (None, None) — the ordinary "no router" failure, not a refusal."""
    monkeypatch.setattr(provenance, "main_root", lambda root: root)
    assert provenance.resolve_build_dir(str(tmp_path), str(engine_root)) == (None, None)


def test_main_root_of_a_real_worktree_is_the_main_clone(tmp_path):
    """git plumbing: a worktree's common dir points at the main clone."""
    import subprocess as sp
    main = tmp_path / "main"
    main.mkdir()
    g = lambda *a, cwd=main: sp.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],
                                    cwd=cwd, check=True, capture_output=True)
    g("init", "-q"); (main / "f").write_text("x\n"); g("add", "-A"); g("commit", "-qm", "i")
    g("worktree", "add", "-q", str(tmp_path / "wt"), "--detach", "HEAD")
    assert os.path.realpath(provenance.main_root(str(tmp_path / "wt"))) == os.path.realpath(str(main))
    assert os.path.realpath(provenance.main_root(str(main))) == os.path.realpath(str(main))


# --------------------------------------------------------------------------- the loader
# pcb_world.engine: ONE resolver (router_lib_dir) and ONE guard (ensure_router_provenance), on a
# fake project tree — the module's root and memo are pointed at it for the test.

@pytest.fixture
def loader(tmp_path, engine_root, monkeypatch):
    import sys
    import pcb_world.engine as eng
    project = tmp_path / "wt"
    project.mkdir()
    # The resolver puts the lib dir it lands on onto sys.path: work on a copy that monkeypatch
    # restores, or a fake router dir would shadow the real one for later tests in this process.
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(provenance, "main_root", lambda root: str(project))   # no git on a tmp dir
    monkeypatch.setattr(eng, "_PROJECT_ROOT", str(project))
    monkeypatch.setattr(eng, "_OWN_BUILD_DIR", str(project / "build_rl"))
    monkeypatch.setattr(eng, "_resolved", None)
    monkeypatch.setattr(eng, "_guarded", False)
    monkeypatch.setenv("PCBWORLD_ENGINE_HOME", str(engine_root))
    monkeypatch.delenv("PCBWORLD_KICAD_RL_BUILD_DIR", raising=False)
    monkeypatch.delenv("PCBWORLD_KICAD_RL_MODULE_DIR", raising=False)
    return eng, project


def test_loader_discovers_the_build_of_this_cpp_and_the_guard_is_silent(loader, engine_root):
    eng, project = loader
    (project / "var" / "builds").mkdir(parents=True)
    _fake_build(project / "var" / "builds", "1.4-aaaaaaaa", "deadbeef")
    good = _fake_build(project / "var" / "builds", "1.4-bbbbbbbb", provenance.source_hash(str(engine_root)))
    assert eng.router_lib_dir() == provenance.lib_dir_of(str(good))
    assert eng.router_build_dir() == str(good)
    assert eng.engine_available()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eng.ensure_router_provenance()


def test_resolver_reason_reaches_the_error_when_every_build_is_of_other_cpp(loader, engine_root):
    eng, project = loader
    (project / "var" / "builds").mkdir(parents=True)
    _fake_build(project / "var" / "builds", "1.4-aaaaaaaa", "deadbeef")
    assert not eng.engine_available()
    with pytest.raises(provenance.RouterProvenanceError) as ei:
        eng.ensure_router_provenance()
    msg = str(ei.value)
    assert "every known build was made from other C++" in msg and "deadbeef" in msg
    assert provenance.source_hash(str(engine_root)) in msg
    with pytest.raises(provenance.RouterProvenanceError):      # a retry refuses again
        eng.ensure_router_provenance()


def test_loader_with_nothing_built_leaves_the_ordinary_no_router_failure(loader):
    eng, project = loader
    assert eng.router_lib_dir() == provenance.lib_dir_of(str(project / "build_rl"))
    assert not eng.engine_available()
    eng.ensure_router_provenance()                              # nothing to refuse: the import will say so


def test_bundle_without_the_script_lets_discovery_use_a_build_unverified(loader, engine_root, tmp_path, monkeypatch):
    """No hash script in the bundle: nothing can be verified, so discovery takes the first build
    and the guard lets it through — the same tolerance the guard has for such a bundle."""
    eng, project = loader
    bare = tmp_path / "bare_bundle"
    (bare / "kicad-patches").mkdir(parents=True)
    monkeypatch.setenv("PCBWORLD_ENGINE_HOME", str(bare))
    (project / "var" / "builds").mkdir(parents=True)
    only = _fake_build(project / "var" / "builds", "1.4-aaaaaaaa", "deadbeef")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert eng.router_lib_dir() == provenance.lib_dir_of(str(only))
        assert eng.engine_available()
        eng.ensure_router_provenance()                          # no raise: unverifiable, not mismatched
    assert any("cpp_content_hash" in str(w.message) for w in seen)     # but said once


def test_a_failing_hash_script_leaves_the_probe_false_and_the_guard_loud(loader, tmp_path, monkeypatch):
    """The bundle's hash script fails: discovery cannot verify anything, so the probe answers
    False with a warning (a pytest session is not aborted at collection) and the first engine
    use raises with the script's stderr."""
    eng, project = loader
    broken = tmp_path / "broken_bundle"
    (broken / "tools").mkdir(parents=True)
    (broken / "kicad-patches").mkdir()
    (broken / "tools" / "cpp_content_hash.sh").write_text(
        "#!/usr/bin/env bash\necho 'hash tool broken on purpose' >&2\nexit 1\n", encoding="utf-8")
    monkeypatch.setenv("PCBWORLD_ENGINE_HOME", str(broken))
    (project / "var" / "builds").mkdir(parents=True)
    _fake_build(project / "var" / "builds", "1.4-aaaaaaaa", "deadbeef")
    with pytest.warns(RuntimeWarning, match="discovery skipped"):
        assert not eng.engine_available()
    with pytest.raises(provenance.RouterProvenanceError, match="hash tool broken on purpose"):
        eng.ensure_router_provenance()


def test_an_explicit_module_dir_is_used_but_guarded(loader, engine_root, tmp_path, monkeypatch):
    eng, _ = loader
    bad = _fake_build(tmp_path, "elsewhere", "deadbeef")
    monkeypatch.setenv("PCBWORLD_KICAD_RL_MODULE_DIR", provenance.lib_dir_of(str(bad)))
    assert eng.router_lib_dir() == provenance.lib_dir_of(str(bad))
    with pytest.raises(provenance.RouterProvenanceError, match="deadbeef"):
        eng.ensure_router_provenance()


# --------------------------------------------------------------------------- end to end
# The two load sites and the import-time resolver, exercised with the real router: a copy of
# the loaded module directory with a tampered stamp must be refused by BOTH paths, and a git
# worktree of this repo (no build_rl of its own, no variable) must find this tree's build.

import subprocess
import sys


def _router_copy_with_stamp(tmp_path: Path, stamp: str) -> Path:
    """A build dir holding a copy of the loaded module files, stamped *stamp*."""
    from pcb_world.engine import engine_available, router_lib_dir
    if not engine_available():
        pytest.skip("no router build reachable from this tree")
    dst = tmp_path / "build" / "pcbnew" / "python" / "rl"
    dst.mkdir(parents=True)
    for f in Path(router_lib_dir()).iterdir():
        if f.is_file() and not f.name.endswith(".a"):
            shutil.copy2(f, dst / f.name)
    (dst / provenance.STAMP_NAME).write_text(stamp + "\n", encoding="utf-8")
    return tmp_path / "build"


def _construct_engine(build_dir: Path, ipc: str, via: str = "build",
                      extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Construct an engine in a child process pointed at *build_dir* through the build-dir
    variable (``via="build"``) or the module-dir variable (``via="module"``)."""
    env = {**os.environ, "KICAD_ENGINE_IPC": ipc, "PYTHONPATH": str(Path(engine_home()).parent)}
    env.pop("PCBWORLD_KICAD_RL_MODULE_DIR", None)
    env.pop("PCBWORLD_KICAD_RL_BUILD_DIR", None)
    env.pop(provenance.ALLOW_ENV, None)
    if via == "module":
        env["PCBWORLD_KICAD_RL_MODULE_DIR"] = provenance.lib_dir_of(str(build_dir))
    else:
        env["PCBWORLD_KICAD_RL_BUILD_DIR"] = str(build_dir)
    env.update(extra_env or {})
    code = ("from configs.loader.schema import EnvConfig\n"
            "from pcb_world.engine import KiCadEngine\n"
            "KiCadEngine('does-not-exist.kicad_pcb', **EnvConfig().engine_kwargs())\n")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          cwd=Path(engine_home()).parent, timeout=120)


@pytest.mark.parametrize("via", ["build", "module"], ids=["BUILD_DIR", "MODULE_DIR"])
@pytest.mark.parametrize("ipc", ["1", "0"], ids=["ipc-server-spawn", "in-process-import"])
def test_both_load_sites_refuse_a_router_from_other_cpp(tmp_path, ipc, via):
    bad = _router_copy_with_stamp(tmp_path, "deadbeef")
    proc = _construct_engine(bad, ipc, via)
    assert proc.returncode != 0
    assert "RouterProvenanceError" in proc.stderr and "deadbeef" in proc.stderr, proc.stderr[-800:]
    assert "does-not-exist" not in proc.stderr          # refused BEFORE any board/server work


def test_a_refused_ipc_spawn_leaves_no_tempdir_behind(tmp_path):
    """The guard runs before the server's krl_ipc_* dir and stderr file exist."""
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    bad = _router_copy_with_stamp(tmp_path, "deadbeef")
    proc = _construct_engine(bad, "1", extra_env={"TMPDIR": str(tmp)})   # the child's tempfile.gettempdir()
    assert "RouterProvenanceError" in proc.stderr, proc.stderr[-800:]
    assert list(tmp.glob("krl_ipc_*")) == []


def test_a_matching_stamp_gets_past_the_guard(tmp_path):
    good = _router_copy_with_stamp(tmp_path, provenance.source_hash(engine_home()))
    proc = _construct_engine(good, "1")
    assert "RouterProvenanceError" not in proc.stderr    # the failure, if any, is the missing board
    assert proc.returncode != 0 and "does-not-exist" in proc.stderr


def test_a_failing_hash_script_refuses_the_load_loudly(tmp_path):
    """PCBWORLD_ENGINE_HOME at a bundle whose hash script exits 1: the load site raises with the
    script's stderr instead of running unverified."""
    good = _router_copy_with_stamp(tmp_path, provenance.source_hash(engine_home()))
    bundle = tmp_path / "bundle"
    (bundle / "tools").mkdir(parents=True)
    (bundle / "tools" / "cpp_content_hash.sh").write_text(
        "#!/usr/bin/env bash\necho 'hash tool broken on purpose' >&2\nexit 1\n", encoding="utf-8")
    proc = _construct_engine(good, "0", extra_env={"PCBWORLD_ENGINE_HOME": str(bundle)})
    assert proc.returncode != 0
    assert "RouterProvenanceError" in proc.stderr and "hash tool broken on purpose" in proc.stderr, proc.stderr[-800:]


def test_plain_import_spawns_no_subprocess():
    """Discovery (git + the hash script) is deferred to first use: importing the package must
    not exec anything, in a tree with its own build (eager, cheap) or without (deferred)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PCBWORLD_KICAD_RL_BUILD_DIR", "PCBWORLD_KICAD_RL_MODULE_DIR")}
    env["PYTHONPATH"] = str(Path(engine_home()).parent)
    code = ("import subprocess\n"
            "def boom(*a, **k): raise AssertionError('subprocess at import: %r' % (a[:1],))\n"
            "subprocess.run = subprocess.Popen = subprocess.check_output = boom\n"
            "import pcb_world.engine\n"
            "print('imported')\n")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          cwd=Path(engine_home()).parent, timeout=120)
    assert proc.returncode == 0 and proc.stdout.strip() == "imported", proc.stderr[-800:]


def test_a_worktree_finds_this_trees_build_without_any_variable(tmp_path):
    """The deferred resolver: a fresh git worktree has no build_rl; the loader must land on a
    build whose stamp matches its C++ — here the same sources, so this tree's build."""
    from pcb_world.engine import engine_available, router_build_dir
    if not engine_available():
        pytest.skip("no router build reachable from this tree")
    repo = Path(__file__).resolve().parents[2]                  # THIS repository, whatever PCBWORLD_ENGINE_HOME says
    wt = tmp_path / "wt"
    add = subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", str(wt), "--detach", "HEAD"],
                         capture_output=True, text=True)
    if add.returncode != 0:
        pytest.skip(f"cannot add a worktree here: {add.stderr.strip()[:120]}")
    try:
        env = {k: v for k, v in os.environ.items()
               if k not in ("PCBWORLD_KICAD_RL_BUILD_DIR", "PCBWORLD_KICAD_RL_MODULE_DIR", "PCBWORLD_ENGINE_HOME")}
        env["PYTHONPATH"] = str(wt)
        code = "import os, pcb_world.engine as e; print(os.path.realpath(e.router_build_dir())); print(e.engine_available())"
        proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=wt, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr[-800:]
        found, available = proc.stdout.strip().splitlines()[-2:]
        assert available == "True"
        assert not found.startswith(str(wt))                                   # not a build of its own
        assert found == os.path.realpath(router_build_dir())                    # this tree's build, by hash
    finally:
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)], capture_output=True)
