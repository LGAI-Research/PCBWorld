"""Rule-based runner board selection — `--difficulty` picks one D3 tier of configs/datasets/d3.json.

`--dataset pcbench` needs `--difficulty` (exactly the d3.json tier/split) or the explicit
`--all-corpus` (every corpus folder, boards excluded from D3 included). A whole tier run with a
board missing its inputs stops with an error, and so does a `--sample` that matches no board of
the tier. `--difficulty` on a synthetic dataset is rejected.
PCBENCH_*_ROOT point at tmp_path, so no data root is needed.
"""
import json
import sys
from pathlib import Path

import pytest

from methods.baselines.rule_based._lib import datasets as ds

REPO = Path(__file__).resolve().parents[1]
D3 = json.loads((REPO / "configs" / "datasets" / "d3.json").read_text())


@pytest.fixture
def pcb_root(tmp_path, monkeypatch):
    root = tmp_path / "pcb"
    for name in ("0001_a", "0096_karabas-nano_wifi_revA", "0170_x"):
        (root / name).mkdir(parents=True)
    monkeypatch.setenv("PCBENCH_PCB_ROOT", str(root))
    monkeypatch.setenv("PCBENCH_DSN_ROOT", str(tmp_path / "dsn"))
    return root


@pytest.mark.parametrize("difficulty", ds.D3_DIFFICULTIES)
def test_difficulty_selects_the_d3_test_tier(pcb_root, difficulty):
    boards = ds.select_boards("pcbench", split="test", difficulty=difficulty)
    assert [b.board_id for b in boards] == D3[difficulty]["test"]
    assert all(b.raw_pcb.parent.parent == pcb_root for b in boards)


def test_d3a_tier_is_99_boards_without_0096(pcb_root):
    ids = [b.board_id for b in ds.select_boards("pcbench", difficulty="easy")]
    assert len(ids) == 99 and "0096_karabas-nano_wifi_revA" not in ids


def test_difficulty_with_train_split_and_sample_filter(pcb_root):
    boards = ds.select_boards("pcbench", split="train", difficulty="medium",
                              sample_substrings=[D3["medium"]["train"][3]])
    assert [b.board_id for b in boards] == [D3["medium"]["train"][3]]


def test_pcbench_needs_a_tier_or_all_corpus(pcb_root):
    with pytest.raises(ValueError, match="needs --difficulty"):
        ds.select_boards("pcbench")
    with pytest.raises(ValueError, match="mutually exclusive"):
        ds.select_boards("pcbench", difficulty="easy", all_corpus=True)
    with pytest.raises(ValueError, match="needs --difficulty"):
        ds.dataset_tag("pcbench")


def test_all_corpus_walks_every_folder(pcb_root):
    ids = [b.board_id for b in ds.select_boards("pcbench", all_corpus=True)]
    assert ids == ["0001_a", "0096_karabas-nano_wifi_revA", "0170_x"]


def test_all_corpus_rejects_a_split(pcb_root):
    with pytest.raises(ValueError, match="--split train selects within a D3 tier"):
        ds.select_boards("pcbench", all_corpus=True, split="train")


def test_corrupt_d3_json_is_not_a_selection_error(tmp_path, monkeypatch, pcb_root):
    bad = tmp_path / "d3.json"
    bad.write_text("<<<<<<< HEAD\n{}")
    monkeypatch.setattr(ds, "D3_SPLIT_JSON", bad)
    monkeypatch.setattr(ds, "_d3_split", ds._d3_split.__wrapped__)  # bypass the cached real file
    with pytest.raises(RuntimeError, match="d3.json is not valid JSON"):
        ds.select_boards("pcbench", difficulty="easy")
    for entry, main in _entry_points().items():  # the CLIs surface it, not as a usage error
        monkeypatch.setattr(sys, "argv", [entry, "--baseline", "krt", "--dataset", "pcbench",
                                          "--difficulty", "easy"])
        with pytest.raises(RuntimeError, match="d3.json is not valid JSON"):
            main()


def test_difficulty_rejects_unknown_split_and_tier(pcb_root):
    with pytest.raises(ValueError, match="no easy.val"):
        ds.select_boards("pcbench", split="val", difficulty="easy")
    with pytest.raises(ValueError, match="--difficulty must be one of"):
        ds.select_boards("pcbench", difficulty="d3a")


@pytest.mark.parametrize("dataset", ("synthetic_2l", "synthetic_1l"))
@pytest.mark.parametrize("flags", ({"difficulty": "easy"}, {"all_corpus": True}))
def test_pcbench_flags_rejected_for_synthetic_sets(dataset, flags):
    with pytest.raises(ValueError, match="only with --dataset pcbench"):
        ds.select_boards(dataset, grid=10, **flags)


@pytest.mark.parametrize("baseline,sibling", (("krt", "dsn"), ("freerouting", "dsn"), ("orthoroute", "orp")))
def test_partition_by_inputs_checks_the_baselines_sibling(tmp_path, baseline, sibling):
    full = ds.BoardPaths("full", *(tmp_path / f"full.{k}" for k in ("pcb", "pro", "dsn", "orp")))
    gap = ds.BoardPaths("gap", *(tmp_path / f"gap.{k}" for k in ("pcb", "pro", "dsn", "orp")))
    for b in (full, gap):
        for k in ("raw_pcb", "pro", "dsn", "orp"):
            if not (b is gap and k == sibling):
                getattr(b, k).write_text("x")
    valid, missing = ds.partition_by_inputs([full, gap], baseline)
    assert valid == [full]
    assert missing == [("gap", f"{sibling}={getattr(gap, sibling)}")]


def test_sample_outside_the_tier_is_an_error(pcb_root):
    # 0096 is a corpus folder but not part of D3-A: asking for it must not silently shrink the run.
    with pytest.raises(ValueError, match=r"--sample matched no board of D3 tier easy.test: \['0096_karabas"):
        ds.select_boards("pcbench", difficulty="easy",
                         sample_substrings=[D3["easy"]["test"][0], "0096_karabas"])


def test_tier_with_a_missing_pcb_root_names_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PCBENCH_PCB_ROOT", str(tmp_path / "nope"))
    monkeypatch.setenv("PCBENCH_DSN_ROOT", str(tmp_path / "dsn"))
    with pytest.raises(FileNotFoundError, match="PCBENCH_PCB_ROOT=.*nope does not exist"):
        ds.select_boards("pcbench", difficulty="easy")


def test_dataset_tag_separates_tiers():
    assert ds.dataset_tag("pcbench", all_corpus=True) == "PCBench_corpus"
    assert ds.dataset_tag("pcbench", difficulty="easy", split="test") == "PCBench_easy_test"
    assert ds.dataset_tag("pcbench", difficulty="hard", split="train") == "PCBench_hard_train"


def _entry_points():
    from eval.rollout import rule_based as rollout  # puts the runner dir on sys.path
    import run_rule_based_routers
    return {"run_rule_based_routers": run_rule_based_routers.main, "rollout_rule_based": rollout.main}


@pytest.mark.parametrize("entry", ("run_rule_based_routers", "rollout_rule_based"))
def test_cli_rejects_difficulty_for_synthetic(entry, monkeypatch, capsys):
    main = _entry_points()[entry]
    monkeypatch.setattr(sys, "argv", [entry, "--baseline", "krt", "--dataset", "synthetic_2l",
                                      "--difficulty", "easy"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "only with --dataset pcbench" in capsys.readouterr().err


def _run_main(entry, monkeypatch, capsys, *argv):
    main = _entry_points()[entry]
    monkeypatch.setattr(sys, "argv", [entry, "--baseline", "krt", *argv])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code, capsys.readouterr().err


@pytest.mark.parametrize("entry", ("run_rule_based_routers", "rollout_rule_based"))
def test_cli_requires_difficulty_for_pcbench(entry, monkeypatch, capsys):
    code, err = _run_main(entry, monkeypatch, capsys, "--dataset", "pcbench")
    assert code == 2 and "needs --difficulty" in err


@pytest.mark.parametrize("entry", ("run_rule_based_routers", "rollout_rule_based"))
@pytest.mark.parametrize("extra", (
    (),
    ("--limit", "100"),  # a limit that keeps every tier board is still the whole tier
    tuple(a for bid in D3["easy"]["test"] for a in ("--sample", bid)),  # so are samples covering it
), ids=("plain", "limit-100", "sample-all"))
def test_cli_whole_tier_with_missing_inputs_is_an_error(entry, extra, pcb_root, monkeypatch, capsys):
    # pcb_root holds 0001_a only as an empty folder: every D3-A board lacks its inputs.
    code, err = _run_main(entry, monkeypatch, capsys, "--dataset", "pcbench", "--difficulty", "easy",
                          *extra)
    assert code == 2
    assert "D3 tier easy.test is incomplete: 99 of 99 board(s)" in err
    assert "... and 89 more" in err


@pytest.mark.parametrize("entry", ("run_rule_based_routers", "rollout_rule_based"))
@pytest.mark.parametrize("argv,needle", (
    (("--split", "val"), "has no easy.val"),
    (("--limit", "-1"), "--limit must be >= 1"),
    (("--limit", "0"), "--limit must be >= 1"),
    (("--sample", "0096_karabas"), "--sample matched no board of D3 tier easy.test"),
))
def test_cli_selection_errors_are_usage_errors(entry, argv, needle, pcb_root, monkeypatch, capsys):
    code, err = _run_main(entry, monkeypatch, capsys, "--dataset", "pcbench", "--difficulty", "easy",
                          *argv)
    assert code == 2 and needle in err


def test_runner_scores_a_tier_run_under_its_tier_tag(pcb_root, tmp_path, monkeypatch):
    # The runner's built-in scoring (on unless --no-eval) writes the summary under the tier's tag.
    _entry_points()
    import run_rule_based_routers as rb
    bid = D3["easy"]["test"][0]
    board = ds.select_boards("pcbench", difficulty="easy", sample_substrings=[bid])[0]
    for p in (board.raw_pcb, board.pro, board.dsn):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    labels = []
    monkeypatch.setattr(rb, "prepare_run", lambda baseline: None)   # no KRT/engine install needed
    monkeypatch.setattr(rb, "_process_one_board", lambda w: dict(
        board_id=w["board"].board_id, seed=w["seed"], status="ok", elapsed_routing_s=0.0,
        eval_metrics={}))
    monkeypatch.setattr(rb, "_aggregate_summary",
                        lambda results, seed_dir: labels.append(seed_dir.parent.parent.name))
    monkeypatch.setattr(sys, "argv", ["run_rule_based_routers", "--baseline", "krt", "--dataset", "pcbench",
                                      "--difficulty", "easy", "--sample", bid,
                                      "--output-root", str(tmp_path / "out")])
    assert rb.main() == 0
    assert labels == ["PCBench_easy_test"]
    (manifest,) = (tmp_path / "out" / "PCBench_easy_test").glob("*/seed0/manifest.json")
    assert json.loads(manifest.read_text())["boards"] == [bid]


@pytest.mark.parametrize("entry", ("run_rule_based_routers", "rollout_rule_based"))
def test_cli_limited_tier_run_only_skips_missing_boards(entry, pcb_root, monkeypatch, capsys):
    code, err = _run_main(entry, monkeypatch, capsys, "--dataset", "pcbench", "--difficulty", "easy",
                          "--limit", "2")
    assert code == 2
    assert "D3 tier" not in err and "no boards remain" in err


def test_orthoroute_pipeline_tier_with_missing_boards_is_an_error(tmp_path, monkeypatch, capsys):
    from methods.baselines.rule_based import run_orthoroute_pipeline as orp
    d3a = D3["easy"]["test"]
    for bid in d3a[:2]:
        (tmp_path / "data" / bid).mkdir(parents=True)
        (tmp_path / "data" / bid / f"{orp.DEFAULT_STEM}_unrouted.kicad_pcb").write_text("(kicad_pcb)")
    base = ["run_orthoroute_pipeline.py", "--data-root", str(tmp_path / "data"),
            "--out-root", str(tmp_path / "out"), "--difficulty", "easy"]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit) as exc:
        orp.main()
    assert exc.value.code == 2 and "97 of 99 board(s)" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()  # a rejected selection leaves no output tree
    monkeypatch.setattr(sys, "argv", base + ["--boards", d3a[0]])
    with pytest.raises(SystemExit) as exc:
        orp.main()
    assert exc.value.code == 2 and "mutually exclusive" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", base + ["--split", "val"])
    with pytest.raises(SystemExit) as exc:
        orp.main()
    assert exc.value.code == 2 and "has no easy.val" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", base + ["--limit", "99"])
    with pytest.raises(SystemExit) as exc:
        orp.main()
    assert exc.value.code == 2 and "97 of 99 board(s)" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [a for a in base if a not in ("--difficulty", "easy")]
                        + ["--split", "train"])
    with pytest.raises(SystemExit) as exc:
        orp.main()
    assert exc.value.code == 2 and "needs --difficulty" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", base + ["--limit", "-1"])
    with pytest.raises(SystemExit) as exc:
        orp.main()
    assert exc.value.code == 2 and "--limit must be >= 0" in capsys.readouterr().err


def test_orthoroute_pipeline_limited_tier_warns_and_keeps_d3_order(tmp_path, monkeypatch, capsys):
    from methods.baselines.rule_based import run_orthoroute_pipeline as orp
    d3a = D3["easy"]["test"]
    data = tmp_path / "data"
    for bid in (d3a[2], d3a[0]):
        (data / bid).mkdir(parents=True)
        (data / bid / f"{orp.DEFAULT_STEM}_unrouted.kicad_pcb").write_text("(kicad_pcb)")
    assert [b[0] for b in orp.discover_boards(data, orp.DEFAULT_STEM, d3a)] == [d3a[0], d3a[2]]

    class _Stop(Exception):
        pass

    def _stop():
        raise _Stop

    monkeypatch.setattr(orp.kicad_tools, "pcbnew_python", _stop)  # end the run right after selection
    monkeypatch.setattr(sys, "argv", ["run_orthoroute_pipeline.py", "--data-root", str(data),
                                      "--out-root", str(tmp_path / "out"), "--difficulty", "easy",
                                      "--limit", "2"])
    with pytest.raises(_Stop):
        orp.main()
    out = capsys.readouterr()
    assert "[warn] D3 tier easy.test: 97 of 99 board(s)" in out.err
    assert "[pipeline] 2 boards" in out.out
