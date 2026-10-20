"""krt_runner's launcher runs KRT's route.py as `python route.py` would: route.py's folder first
on sys.path (KRT imports its flat modules from there) and the cached grid_router pre-loaded."""
import os
import subprocess
import sys
from pathlib import Path

LAUNCHER = (Path(__file__).resolve().parents[1] / "methods" / "baselines" / "rule_based" / "krt"
            / "src" / "krt_runner" / "_launcher.py")


def test_launcher_imports_route_py_siblings(tmp_path):
    krt, so_dir, out = tmp_path / "krt", tmp_path / "so", tmp_path / "out.txt"
    krt.mkdir()
    so_dir.mkdir()
    (so_dir / "grid_router.py").write_text("")
    (krt / "env_knobs.py").write_text("VALUE = 'sibling'\n")
    (krt / "route.py").write_text(
        "import sys, env_knobs, grid_router\n"
        "open(sys.argv[2], 'w').write(env_knobs.VALUE + ' ' + sys.argv[1])\n")
    env = {**os.environ, "KRT_RUST_SO_DIR": str(so_dir), "KRT_ROUTE_PY": str(krt / "route.py")}
    proc = subprocess.run([sys.executable, str(LAUNCHER), "arg1", str(out)], env=env,
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert out.read_text() == "sibling arg1"
