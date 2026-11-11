"""Engine build entry points: where the build lands, and no build through a symlink."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _fake_conda(tmp_path: Path) -> dict:
    """PATH with a `conda` that reports the BUILD_DIR it would build into, instead of building."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    conda = bin_dir / "conda"
    conda.write_text('#!/usr/bin/env bash\n# conda run -n ENV env BUILD_DIR=... bash SCRIPT\n'
                     'for a in "$@"; do case "$a" in BUILD_DIR=*) echo "$a";; esac; done\n')
    conda.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


def test_build_engine_defaults_to_the_tree_s_build_rl(tmp_path):
    env = _fake_conda(tmp_path)
    env.pop("BUILD_DIR", None)
    proc = subprocess.run(["bash", str(REPO / "tools/setup/build_engine.sh")], env=env,
                          capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == f"BUILD_DIR={REPO}/build_rl"


def test_build_engine_passes_a_caller_s_build_dir_through(tmp_path):
    env = {**_fake_conda(tmp_path), "BUILD_DIR": str(tmp_path / "elsewhere")}
    proc = subprocess.run(["bash", str(REPO / "tools/setup/build_engine.sh")], env=env,
                          capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == f"BUILD_DIR={tmp_path / 'elsewhere'}"


def test_engine_build_refuses_a_symlinked_build_dir(tmp_path):
    # the guard runs before any dependency check, so this needs no toolchain
    other = tmp_path / "other_checkout_build_rl"
    other.mkdir()
    link = tmp_path / "build_rl"
    link.symlink_to(other)
    proc = subprocess.run(["bash", str(REPO / "engine/build_rl_router.sh")],
                          env={**os.environ, "BUILD_DIR": str(link)}, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "is a symlink" in proc.stdout and f'BUILD_DIR="{other}"' in proc.stdout
    assert not any(other.iterdir())        # nothing was written through the link


def _mac_build(tmp_path: Path, *, env_has_clang: bool = True, cache: str | None = None, flags: tuple = (),
               darwin: bool = True, env_has_deps: bool = False, conda: bool = True) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run engine/build_rl_router.sh as on macOS inside a conda env, with fake tools; return (proc, cmake args)."""
    engine = tmp_path / "engine"
    engine.mkdir()
    shutil.copy2(REPO / "engine/build_rl_router.sh", engine)
    shutil.copytree(REPO / "engine/tools", engine / "tools")
    shutil.copytree(REPO / "engine/kicad-patches", engine / "kicad-patches")
    src = engine / "kicad-python/kicad"
    src.mkdir(parents=True)
    (src / "CMakeLists.txt").write_text("")
    for dest in re.findall(r'"\$KICAD_SRC/([^"]*)"', (REPO / "engine/build_rl_router.sh").read_text()):
        (src / (dest if dest.endswith("/") else str(Path(dest).parent))).mkdir(parents=True, exist_ok=True)

    prefix = tmp_path / "prefix"                                         # the conda env: OCC, optionally the cmake configs
    (prefix / "include/opencascade").mkdir(parents=True)
    (prefix / "include/opencascade/Standard_Handle.hxx").write_text("")
    if env_has_deps:
        for dep in ("absl", "protobuf", "utf8_range"):
            (prefix / "lib/cmake" / dep).mkdir(parents=True)
    (prefix / "bin").mkdir()
    # the env's python3, answering only the queries the script makes (PATH puts prefix/bin first on macOS)
    (prefix / "bin/python3").write_text(
        f"""#!/bin/sh\ncase "$*" in --version) echo Python 3.12.0;; *"'include'"*) echo {prefix}/include/python3.12;;
  *LIBDIR*) echo {prefix}/lib;; *LDLIBRARY*) echo libpython3.12.dylib;; *__version__*) echo 2.13.6;; esac\n""")
    (prefix / "bin/python3").chmod(0o755)
    (prefix / "lib").mkdir(exist_ok=True)
    (prefix / "lib/libpython3.12.dylib").write_text("")
    build = tmp_path / "build"
    log = tmp_path / "cmake_args"
    fakes = {
        "uname": 'echo Darwin' if darwin else 'echo Linux',
        "cmake": f'[ "$1" = --version ] && {{ echo "cmake version 3.31"; exit 0; }}; printf "%s\\n" "$@" > {log}',
        "ninja": f'[ "$1" = --version ] && {{ echo 1.12; exit 0; }}; mkdir -p {build}/pcbnew/python/rl; touch {build}/pcbnew/python/rl/kicad_rl_router.so',
        "wx-config": 'echo 3.2.8',
        "curl": 'while [ $# -gt 0 ]; do [ "$1" = -o ] && echo header > "$2"; shift; done',
        "shasum": 'cat > /dev/null',             # the real header is not fetched offline
    }
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in fakes.items():
        (bin_dir / name).write_text(f"#!/usr/bin/env bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    if env_has_clang:
        for cc in ("clang", "clang++"):
            (prefix / "bin" / cc).write_text("#!/bin/sh\n")
            (prefix / "bin" / cc).chmod(0o755)
    if cache is not None:
        build.mkdir()
        (build / "CMakeCache.txt").write_text(cache)
    env = {"PATH": f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": str(tmp_path),
           "CONDA_PREFIX": str(prefix), "BUILD_DIR": str(build), "CXXFLAGS": "-O2"}
    if not conda:
        del env["CONDA_PREFIX"]
    proc = subprocess.run(["bash", str(engine / "build_rl_router.sh"), *flags], env=env, capture_output=True, text=True)
    return proc, (log.read_text().splitlines() if log.exists() else [])


def test_macos_conda_build_takes_everything_from_the_env(tmp_path):
    proc, args = _mac_build(tmp_path, env_has_deps=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    prefix = tmp_path / "prefix"
    assert {f"-DCMAKE_C_COMPILER={prefix}/bin/clang", f"-DCMAKE_CXX_COMPILER={prefix}/bin/clang++",
            "-DCMAKE_CXX_FLAGS=-O2 -include cstdlib", f"-DCMAKE_PREFIX_PATH={prefix}",
            "-DCMAKE_SYSTEM_IGNORE_PREFIX_PATH=/opt/homebrew;/usr/local",
            f"-DOCC_INCLUDE_DIR={prefix}/include/opencascade", f"-Dabsl_DIR={prefix}/lib/cmake/absl",
            f"-DProtobuf_DIR={prefix}/lib/cmake/protobuf", f"-Dutf8_range_DIR={prefix}/lib/cmake/utf8_range",
            f"-DNGSPICE_ROOT_DIR={tmp_path}/build/macos_deps", f"-DPYTHON_EXECUTABLE={prefix}/bin/python3",
            f"-DPYTHON_INCLUDE_DIR={prefix}/include/python3.12",
            f"-DPYTHON_LIBRARY={prefix}/lib/libpython3.12.dylib"} <= set(args)
    assert (tmp_path / "build/macos_deps/include/ngspice/sharedspice.h").is_file()
    assert not any("/opt/homebrew" in a for a in args if not a.startswith("-DCMAKE_SYSTEM_IGNORE_PREFIX_PATH="))


def test_macos_without_the_conda_toolchain_is_refused(tmp_path):
    # an env without clang++ (an auto-activated base) or an explicit --conda there: no Homebrew fallback
    proc, args = _mac_build(tmp_path, env_has_clang=False)
    assert proc.returncode == 1 and "has no bin/clang++" in proc.stdout and not args


def test_macos_without_a_conda_env_is_refused(tmp_path):
    proc, args = _mac_build(tmp_path, conda=False)
    assert proc.returncode == 1 and "no conda env is active" in proc.stdout and not args


def test_macos_tree_from_the_removed_toolchain_file_is_refused(tmp_path):
    proc, args = _mac_build(tmp_path, cache="CMAKE_TOOLCHAIN_FILE:FILEPATH=/x/tools/setup/macos_conda_toolchain.cmake\n")
    assert proc.returncode == 1 and "rm -rf" in proc.stdout and not args


def test_a_trailing_slash_does_not_hide_a_symlinked_build_dir(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "build_rl").symlink_to(other)
    proc = subprocess.run(["bash", str(REPO / "engine/build_rl_router.sh")],
                          env={**os.environ, "BUILD_DIR": f"{tmp_path}/build_rl/"}, capture_output=True, text=True)
    assert proc.returncode == 1 and "is a symlink" in proc.stdout


def test_build_engine_resolves_a_relative_build_dir_against_the_caller(tmp_path):
    env = {**_fake_conda(tmp_path), "BUILD_DIR": "rel"}
    proc = subprocess.run(["bash", str(REPO / "tools/setup/build_engine.sh")], env=env, cwd=tmp_path,
                          capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == f"BUILD_DIR={tmp_path}/rel"


def _setup_all(tmp_path: Path, **extra: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"                      # a conda that fails, so nothing real runs either way
    bin_dir.mkdir()
    (bin_dir / "conda").write_text("#!/usr/bin/env bash\nexit 7\n")
    (bin_dir / "conda").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **extra}
    return subprocess.run(["bash", str(REPO / "tools/setup/setup_all.sh")], env=env, capture_output=True, text=True)


def test_setup_all_refuses_any_build_dir_the_shell_carries(tmp_path):
    proc = _setup_all(tmp_path, BUILD_DIR="build_rl")            # even its own: it always builds build_rl/
    assert proc.returncode == 1 and "BUILD_DIR=build_rl is set" in proc.stderr and "build_engine.sh" in proc.stderr


def test_setup_all_without_build_dir_passes_the_check(tmp_path):
    assert "is set" not in _setup_all(tmp_path).stderr


def test_linux_takes_the_dependencies_from_the_active_conda_env(tmp_path):
    proc, args = _mac_build(tmp_path, darwin=False, env_has_deps=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    prefix = tmp_path / "prefix"
    assert {f"-Dabsl_DIR={prefix}/lib/cmake/absl", f"-DProtobuf_DIR={prefix}/lib/cmake/protobuf",
            f"-Dutf8_range_DIR={prefix}/lib/cmake/utf8_range",
            f"-DOCC_INCLUDE_DIR={prefix}/include/opencascade"} <= set(args)
    assert not any(a.startswith(("-DCMAKE_C_COMPILER=", "-DCMAKE_SYSTEM_IGNORE_PREFIX_PATH=")) for a in args)
