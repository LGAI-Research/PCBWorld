"""setup_all.sh refuses a conda whose classic solver never finishes, and prefers base's mamba."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(tmp_path: Path, solver: str | None, *, base_mamba: bool = False, path_mamba: bool = False,
         env_exists: bool = False, **extra: str) -> subprocess.CompletedProcess:
    """setup_all.sh on a fake Linux host with fake conda/mamba that log the call and stop at env creation."""
    base = tmp_path / "base"
    (base / "etc/profile.d").mkdir(parents=True)
    (base / "etc/profile.d/conda.sh").write_text("")
    (base / "bin").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    show = f'echo "solver: {solver}"' if solver else 'echo "invalid parameter: solver" >&2; exit 1'
    fakes = {bin_dir / "conda": f'''case "$1" in
  info) echo {base};;
  --version) echo "conda 22.9.0";;
  config) {show};;
  env) [ "$2" = list ] && {{ {'echo "pcbworld  /envs/pcbworld"' if env_exists else 'true'}; exit 0; }}; echo "conda $*"; exit 3;;
  *) exit 5;;
esac''',
             bin_dir / "uname": '[ "$1" = -m ] && echo x86_64 || echo Linux',
             # answers `git -C <dir> rev-parse --show-toplevel`; anything else (step 2) stops the run
             bin_dir / "git": f'[ "$3" = rev-parse ] && {{ echo {REPO}; exit 0; }}; echo "git $*"; exit 4'}
    if base_mamba:
        fakes[base / "bin/mamba"] = 'echo "mamba $*"; exit 3'
    if path_mamba:                                                # a mamba with its own root prefix
        fakes[bin_dir / "mamba"] = 'echo "foreign-mamba $*"; exit 3'
    for path, body in fakes.items():
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), **extra}
    return subprocess.run(["bash", str(REPO / "tools/setup/setup_all.sh")], env=env,
                          capture_output=True, text=True)


def test_a_conda_without_the_solver_option_is_refused(tmp_path):
    proc = _run(tmp_path, None)                                   # conda 22.9: no `solver` key
    assert proc.returncode == 1
    assert "classic solver" in proc.stderr and "env create" not in proc.stdout


def test_the_classic_solver_is_refused(tmp_path):
    proc = _run(tmp_path, "classic")                              # conda 22.11–23.9 default
    assert proc.returncode == 1 and "conda-libmamba-solver" in proc.stderr and "mamba" in proc.stderr
    assert "SKIP_ENV" not in proc.stderr                          # no env to keep


def test_the_refusal_offers_skip_env_when_the_env_exists(tmp_path):
    proc = _run(tmp_path, "classic", env_exists=True)
    assert proc.returncode == 1 and "SKIP_ENV=1 keeps the existing env" in proc.stderr


def test_conda_with_libmamba_creates_the_env(tmp_path):
    proc = _run(tmp_path, "libmamba")
    assert proc.returncode == 3 and "conda env create -n pcbworld -f environment.yml" in proc.stdout


def test_base_mamba_is_used_when_present(tmp_path):
    proc = _run(tmp_path, None, base_mamba=True)
    assert proc.returncode == 3 and "mamba env create -n pcbworld -f environment.yml" in proc.stdout


def test_a_mamba_outside_conda_s_base_is_not_used(tmp_path):
    proc = _run(tmp_path, None, path_mamba=True)
    assert proc.returncode == 1 and "foreign-mamba" not in proc.stdout


def test_skip_env_keeps_the_env_without_solving(tmp_path):
    proc = _run(tmp_path, None, env_exists=True, SKIP_ENV="1")
    assert proc.returncode == 4                                   # stopped by the fake git at step 2
    assert "kept as is" in proc.stdout and "env update" not in proc.stdout and "git submodule update" in proc.stdout


def test_skip_env_without_an_env_is_refused(tmp_path):
    proc = _run(tmp_path, None, SKIP_ENV="1")
    assert proc.returncode == 1 and "no conda env 'pcbworld'" in proc.stderr
