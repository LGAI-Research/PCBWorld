"""configs/datasets/d3.json and its builder — D3-A is 99 boards, the excluded boards stay out.

The shipped split must keep D3-A as one 99-board list (easy.train == easy.test) and keep every
board of the builder's ``EXCLUDED_BOARDS`` out of every tier. The builder tests run on a
synthetic CSV + corpus under tmp_path (no data root needed). The KDD LLM baseline scripts
resolve real boards through d3a_boards.py (D3-A of the json; a full run with a missing board
fails), and prepare_pcbench.sh reports the D3 counts from the json, not the folder count.
"""
import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
D3 = REPO / "configs" / "datasets" / "d3.json"
BUILD = REPO / "experiments" / "kdd" / "d3_dataset" / "build.py"
LLM_DIR = REPO / "experiments/kdd/table1_llm/baselines"
D3A_BOARDS = LLM_DIR / "d3a_boards.py"
LLM_SCRIPTS = (
    LLM_DIR / "run_engine_free_llm_v3_standalone.sh",
    LLM_DIR / "run_plan_only_llm_v8_standalone.sh",
    LLM_DIR / "run_qwen_together_sweep.sh",
)
PREPARE = REPO / "tools/quickstart/prepare_pcbench.sh"


def _load_builder():
    spec = importlib.util.spec_from_file_location("_d3_build_under_test", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def d3():
    return json.loads(D3.read_text())


def test_d3a_is_99_boards_same_in_train_and_test(d3):
    assert len(d3["easy"]["test"]) == 99
    assert d3["easy"]["train"] == d3["easy"]["test"]


def test_excluded_boards_absent_from_every_tier(d3):
    excluded = set(_load_builder().EXCLUDED_BOARDS)
    assert {b.split("_", 1)[0] for b in excluded} == {"0096", "0170"}
    for tier in ("easy", "medium", "hard"):
        for split, boards in d3[tier].items():
            assert not excluded & set(boards), (tier, split)
            assert not any(b.startswith(("0096_", "0170_")) for b in boards), (tier, split)


def test_d3_pool_size_and_test_subset(d3):
    pool = [b for tier in ("easy", "medium", "hard") for b in d3[tier]["train"]]
    assert len(pool) == len(set(pool)) == 677
    for tier in ("easy", "medium", "hard"):
        assert set(d3[tier]["test"]) <= set(d3[tier]["train"]), tier


# --- builder on a synthetic corpus -----------------------------------------------------

def _corpus(tmp_path, n_easy=4, n_medium=12, n_hard=12):
    """CSV sorted by pins + matching NNNN_<sample>/ folders; all boards 2-layer."""
    rows = []
    for i in range(n_easy):
        rows.append(dict(sample=f"e{i}", nets=5, components=5, pins=10 + i, layers=2))
    for i in range(n_medium):
        rows.append(dict(sample=f"m{i}", nets=20, components=20, pins=50 + i, layers=2))
    for i in range(n_hard):
        rows.append(dict(sample=f"h{i}", nets=50, components=50, pins=200 + i, layers=2))
    root = tmp_path / "exacad_sorted"
    root.mkdir()
    for k, r in enumerate(rows, start=1):
        (root / f"{k:04d}_{r['sample']}").mkdir()
    csv_path = root / "pcb_characteristics_exacad_sorted.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    args = argparse.Namespace(
        csv=csv_path, sorted_dir=root, easy_rows=n_easy, medium_pin_threshold=100,
        medium_n=2, hard_n=2, hard_trim_frac=0.0)
    return args


def test_builder_drops_excluded_boards_after_test_picks(tmp_path, monkeypatch):
    mod = _load_builder()
    args = _corpus(tmp_path)
    monkeypatch.setattr(mod, "EXCLUDED_BOARDS", ())
    ref = mod.build(args)
    # Exclude one easy board and one medium board that is not a test pick.
    medium_spare = next(b for b in ref["medium"]["train"] if b not in ref["medium"]["test"])
    monkeypatch.setattr(mod, "EXCLUDED_BOARDS", ("0002_e1", medium_spare))
    out = mod.build(args)
    assert "0002_e1" not in out["easy"]["train"]
    assert out["easy"]["test"] == out["easy"]["train"]
    assert len(out["easy"]["train"]) == 3
    assert medium_spare not in out["medium"]["train"]
    assert out["medium"]["test"] == ref["medium"]["test"]
    assert out["hard"] == ref["hard"]


def test_builder_fails_when_excluded_board_is_a_test_pick(tmp_path, monkeypatch):
    mod = _load_builder()
    args = _corpus(tmp_path)
    monkeypatch.setattr(mod, "EXCLUDED_BOARDS", ())
    pick = mod.build(args)["hard"]["test"][0]
    monkeypatch.setattr(mod, "EXCLUDED_BOARDS", (pick,))
    with pytest.raises(SystemExit, match="picked for a test set"):
        mod.build(args)


def test_builder_fails_when_excluded_board_is_missing(tmp_path, monkeypatch):
    mod = _load_builder()
    args = _corpus(tmp_path)
    monkeypatch.setattr(mod, "EXCLUDED_BOARDS", ("9999_not_there",))
    with pytest.raises(SystemExit, match="not found in the corpus"):
        mod.build(args)


# --- KDD LLM baseline scripts: default real-board set = D3-A of d3.json ---------------

def _d3a(tmp_path, bids, *limit_args):
    exacad = tmp_path / "exacad"
    for bid in bids:
        (exacad / bid).mkdir(parents=True, exist_ok=True)
        (exacad / bid / "board.kicad_pcb").write_text("(kicad_pcb)")
    return subprocess.run([sys.executable, str(D3A_BOARDS), str(exacad), "board.kicad_pcb", *limit_args],
                          capture_output=True, text=True)


def test_d3a_boards_lists_the_99_boards_in_order(tmp_path, d3):
    res = _d3a(tmp_path, d3["easy"]["test"] + ["0096_karabas-nano_wifi_revA"], "0")
    assert res.returncode == 0, res.stderr
    assert [Path(p).parent.name for p in res.stdout.splitlines()] == d3["easy"]["test"]


def test_d3a_boards_full_run_fails_on_a_missing_board(tmp_path, d3):
    res = _d3a(tmp_path, d3["easy"]["test"][1:], "0")
    assert res.returncode == 1
    assert d3["easy"]["test"][0] in res.stderr


def test_d3a_boards_limited_run_skips_missing_boards(tmp_path, d3):
    res = _d3a(tmp_path, d3["easy"]["test"][1:], "2")
    assert res.returncode == 0 and "[WARN]" in res.stderr
    assert [Path(p).parent.name for p in res.stdout.splitlines()] == d3["easy"]["test"][1:3]


@pytest.mark.parametrize("limit", ("99", "100"))
def test_d3a_boards_limit_covering_d3a_is_a_full_run(tmp_path, d3, limit):
    res = _d3a(tmp_path, d3["easy"]["test"][1:], limit)
    assert res.returncode == 1 and "[ERROR]" in res.stderr


def test_d3a_boards_filename_may_be_a_glob(tmp_path, d3):
    exacad = tmp_path / "exacad"
    for bid in d3["easy"]["test"]:
        (exacad / bid).mkdir(parents=True)
        (exacad / bid / "processed_v9_guide_v3.kicad_pcb").write_text("(kicad_pcb)")
    res = subprocess.run([sys.executable, str(D3A_BOARDS), str(exacad), "processed_*.kicad_pcb", "0"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert [Path(p).name for p in res.stdout.splitlines()] == ["processed_v9_guide_v3.kicad_pcb"] * 99


@pytest.mark.parametrize("limit", ("all", "--5", "²", "-1"))
def test_d3a_boards_rejects_an_invalid_limit(tmp_path, d3, limit):
    res = _d3a(tmp_path, d3["easy"]["test"], limit)
    assert res.returncode == 2 and "Traceback" not in res.stderr


@pytest.mark.parametrize("script", LLM_SCRIPTS, ids=lambda p: p.name)
def test_llm_baselines_resolve_real_boards_through_d3a_boards(script):
    text = script.read_text()
    assert '"$SCRIPT_DIR/d3a_boards.py"' in text
    assert "range(1, 101)" not in text and "| sort | head -n" not in text
    # A failing resolver must stop the script (exit status checked, not lost in <(...)).
    assert '|| { echo "[error] could not resolve the D3-A board list' in text


def test_qwen_sweep_resolver_failure_ends_only_that_scenario():
    # run_one's caller logs "failed; continuing" — the resolver must return, not exit the sweep.
    text = (LLM_DIR / "run_qwen_together_sweep.sh").read_text()
    assert '(see above)" >&2; return 1; }' in text and '(see above)" >&2; exit 1; }' not in text


def test_prepare_pcbench_reports_d3_counts_from_the_split(tmp_path, d3):
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY\n", PREPARE.read_text(), flags=re.S)
    assert len(blocks) == 1
    for bid in d3["easy"]["test"][:3]:
        (tmp_path / bid).mkdir()
    out = subprocess.run([sys.executable, "-c", blocks[0], str(tmp_path)], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    assert "D3 = 677 boards" in out and "3 present here" in out
    assert "D3-A 99 / D3-B 10 / D3-C 10" in out
