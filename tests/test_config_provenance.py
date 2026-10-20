"""Config provenance: the run record, ``--dump-config-only``, the explain CLI and the batch preflight.

1. **Run record** (``configs.loader.record``) — per key value / source / where / chain and the
   ``effective`` hook, plus rules, boards, router build, code identity, env vars; a resumed run
   keeps a separate record.
2. **The trainer writes it before any env exists** — a setup that crashes right after
   still leaves the record.
3. **``train_ppo --dump-config-only`` subprocess** — prints the table, writes the record and
   exits 0 WITHOUT the engine (PYTHONPATH/LD_LIBRARY_PATH stripped).
4. **``python -m configs.loader.explain``** — the table of a run dir / record.
5. **``tools/experiments/preflight_diff.py``** exit-code contract over run records.
6. **Tier-2 tools** (``record_tool_run``) — the command line as the config, the record beside a
   file output, and per input the hash of the run record that made it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from configs.loader.record import (
    AT_ENV_CREATION, BOARD_PATHS_MAX, NO_READBACK, _sha256, board_record, fill_effective, run_record,
    run_record_filename, write_run_record,
)

REPO = Path(__file__).resolve().parent.parent
PREFLIGHT = REPO / "tools" / "experiments" / "preflight_diff.py"
BOARD = "tests/fixtures/simple_routing_board.kicad_pcb"


def _run(*argv: str):
    from methods.rl_agent.training.train_ppo import parse_train_args

    return parse_train_args(["--board", BOARD, *argv])


# ===================================================================
# 1. the run record
# ===================================================================
class TestRunRecord:
    def test_config_rows_carry_provenance_and_the_effective_reason(self, tmp_path, capsys):
        run = _run("--lr", "1e-3", "--masking-rule", "pcbworld_masking")
        path = write_run_record(tmp_path, run, boards=board_record([str(REPO / BOARD)], board=BOARD),
                                readback=True)
        assert path == tmp_path / "run_record.yaml"
        rec = yaml.safe_load(path.read_text())

        lr = rec["config"]["lr"]
        assert (lr["value"], lr["source"], lr["where"]) == (1e-3, "cli", "--lr 0.001")
        assert [c["source"] for c in lr["chain"]] == ["defaults", "cli"]
        assert rec["config"]["gamma"]["source"] == "defaults"
        assert rec["config"]["lr"]["effective_reason"] == NO_READBACK                 # not an env field
        assert rec["config"]["engine_seed"]["effective_reason"] == AT_ENV_CREATION   # read off the env
        assert rec["config"]["max_steps"]["effective_reason"] == "held as given; no deeper sink"
        # an inverted dest carries its field's reason (was 'no readback yet')
        assert rec["config"]["no_drc_tokens"]["effective_reason"] == "held as given; no deeper sink"
        assert set(rec["config"]) == set(vars(run)) and "effective" not in rec
        assert rec["mode"] == "fresh"
        assert {"repo_version", "git_commit", "git_dirty"} <= set(rec["code"])
        assert "ENGINE_CPP_HASH" in rec["router_build"]
        assert all(k.startswith(("PCBWORLD_", "KICAD_ENGINE_")) for k in rec["env_vars"])
        masking = rec["rules"]["masking_rule"]
        assert masking["path"].endswith("configs/masking/pcbworld_masking.yaml")
        assert len(masking["sha256"]) == 64
        boards = rec["boards"]
        assert boards["count"] == 1 and boards["kicad_pro"]["count"] in (0, 1)
        assert boards["sources"]["board"]["path"] == BOARD and len(boards["sources"]["board"]["sha256"]) == 64
        assert "--lr 0.001" in capsys.readouterr().out          # the table is echoed

    def test_a_record_no_readback_follows_names_no_env_readback(self):
        """eval / LLM / async-val records are never filled: an env field there reads
        NO_READBACK unless it declares its own reason (was 'read back when the env is built')."""
        rec = run_record(_run(), boards=None)
        assert rec["config"]["engine_seed"]["effective_reason"] == NO_READBACK
        assert rec["config"]["max_steps"]["effective_reason"] == "held as given; no deeper sink"
        assert rec["boards"] is None

    def test_a_long_board_list_is_recorded_by_its_source_without_reading_each_board(self, tmp_path,
                                                                                    monkeypatch):
        split = tmp_path / "split.json"
        split.write_text("{}")
        paths = [f"/data/b{i}.kicad_pcb" for i in range(BOARD_PATHS_MAX + 1)]
        monkeypatch.setattr("os.path.isfile", lambda p: pytest.fail(f"board file touched: {p}"))
        rec = board_record(paths, boards_json=str(split), boards_dataset_dir=None)
        assert rec["count"] == BOARD_PATHS_MAX + 1 and "paths" not in rec
        assert rec["sources"] == {"boards_json": {"path": str(split), "sha256": rec["sources"]["boards_json"]["sha256"]}}
        assert len(rec["sources"]["boards_json"]["sha256"]) == 64 and len(rec["paths_sha256"]) == 64
        assert rec["kicad_pro"]["sha256"] is None and "over 64 boards" in rec["kicad_pro"]["reason"]
        assert board_record(paths[::-1])["paths_sha256"] != rec["paths_sha256"]

    def test_an_unverifiable_router_is_recorded_not_raised(self, monkeypatch):
        """--dump-config-only in a worktree without a matching build must still write a record."""
        from pcb_world.engine import provenance

        def refuse():
            raise provenance.RouterProvenanceError("no build of this tree's C++")

        monkeypatch.setattr(provenance, "router_cpp_hash", refuse)
        build = run_record(_run(), boards=None)["router_build"]
        assert build[provenance.STAMP_NAME] is None
        assert build["reason"] == "router provenance check failed: no build of this tree's C++"

    def test_trainer_appends_effective_from_the_built_env(self, tmp_path):
        """After the envs are built, the record holds what the first env's sinks read back —
        appended below the record written at start, which is not rewritten."""
        from types import SimpleNamespace

        from methods.rl_agent.training.loop import PPOTrainer

        t = PPOTrainer(_run("--save-dir", str(tmp_path)))
        t._run_record_path = write_run_record(tmp_path, t.args, boards=None, readback=True, echo=False)
        start = t._run_record_path.read_text()
        t.envs = [SimpleNamespace(effective_config=lambda: {
            "engine_seed": 77, "keep_routing_fraction": (0.1, 0.2), "emit_drc_tokens": True,
            "not_a_key": 1})]
        t._record_effective()
        text = t._run_record_path.read_text()
        assert text.startswith(start)
        # the run holds emit_drc_tokens as its inverted dest no_drc_tokens: read back under it
        assert yaml.safe_load(text)["effective"] == {"engine_seed": 77, "keep_routing_fraction": [0.1, 0.2],
                                                     "no_drc_tokens": False}

    def test_resumed_run_keeps_its_own_record(self):
        from types import SimpleNamespace

        assert run_record_filename(SimpleNamespace(resume=None)) == "run_record.yaml"
        resumed = SimpleNamespace(resume="p.pt", ckpt_meta={"iteration": 42})
        assert run_record_filename(resumed) == "run_record.resume_iter43.yaml"

    @pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
    def test_dump_config_only_of_a_resume_keeps_the_run_start_record(self, tmp_path, monkeypatch):
        """PREFLIGHT=1 on a resume command pointed at the run's own save dir: the dump goes to
        the resume record, not over the record the run wrote at its start."""
        from methods.rl_agent.training import train_ppo
        from tests.helpers.ckpt_payload import save_ckpt

        start = tmp_path / "run_record.yaml"
        start.write_text("written at the run's start\n")
        ckpt = save_ckpt(tmp_path / "policy_iter_7.pt")
        monkeypatch.setattr("sys.argv", ["train_ppo", "--board", BOARD, "--save-dir", str(tmp_path),
                                         "--resume", str(ckpt), "--dump-config-only"])
        train_ppo.main()
        assert start.read_text() == "written at the run's start\n"
        assert (tmp_path / "run_record.resume_iter1.yaml").is_file()   # the fixture has no iteration


# ===================================================================
# 2. written before any env is built
# ===================================================================
def test_trainer_writes_the_record_before_building_envs(tmp_path, monkeypatch):
    """setup() writes run_record.yaml first: an env build that crashes still leaves it."""
    from methods.rl_agent.training.loop import PPOTrainer

    run = _run("--save-dir", str(tmp_path), "--device", "cpu")
    trainer = PPOTrainer(run)

    def crash():
        assert (tmp_path / "run_record.yaml").is_file()
        raise RuntimeError("env build crashed")

    monkeypatch.setattr(trainer, "_build_envs", crash)
    with pytest.raises(RuntimeError, match="env build crashed"):
        trainer.setup()
    rec = yaml.safe_load((tmp_path / "run_record.yaml").read_text())
    assert rec["config"]["save_dir"]["value"] == str(tmp_path)
    assert rec["boards"]["paths"][0].endswith("simple_routing_board.kicad_pcb")
    assert rec["config"]["engine_seed"]["effective_reason"] == AT_ENV_CREATION   # the trainer fills it
    assert rec["extra"] == {"trainer": "ppo"}



def test_trainer_board_record_names_board_only_for_a_single_board_run(tmp_path):
    """--board is required but ignored by a multi-board run: its record names the options that
    chose the boards, not the --board path."""
    from methods.rl_agent.training.loop import PPOTrainer

    for order, named in (("single", True), ("per_env_random", False)):
        t = PPOTrainer.__new__(PPOTrainer)
        t.args, t.boards = _run("--save-dir", str(tmp_path / order), "--boards-order", order), [BOARD]
        t._write_run_record()
        rec = yaml.safe_load(t._run_record_path.read_text())
        assert ("board" in rec["boards"]["sources"]) is named

# ===================================================================
# 3. --dump-config-only subprocess (no GPU / no engine)
# ===================================================================
def _engine_free_env() -> dict:
    """Strip build_rl paths: importing kicad_rl_router must be impossible."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    return env


def test_dump_config_only_writes_the_record_and_exits_zero(tmp_path):
    res = subprocess.run(
        [sys.executable, "-m", "methods.rl_agent.training.train_ppo",
         "--board", BOARD, "--save-dir", str(tmp_path), "--dump-config-only"],
        cwd=REPO, env=_engine_free_env(), capture_output=True, text=True, timeout=180,
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    rec = yaml.safe_load((tmp_path / "run_record.yaml").read_text())
    assert (rec["config"]["board"]["value"], rec["config"]["board"]["source"]) == (BOARD, "cli")
    assert rec["config"]["dump_config_only"]["value"] is True
    assert rec["boards"] is None
    # the provenance table heads the launch log
    assert f"--board '{BOARD}'" in res.stdout and "run record ->" in res.stdout


# ===================================================================
# 4. explain
# ===================================================================
def _explain(capsys, *argv: str) -> tuple[list[str], str]:
    from configs.loader.explain import main

    capsys.readouterr()
    assert main([*argv]) == 0
    out = capsys.readouterr()
    return out.out.splitlines(), out.err


def test_explain_prints_the_table_of_a_run_dir(tmp_path, capsys):
    write_run_record(tmp_path, _run("--gamma", "0.9"), boards=None, echo=False)
    for n in (11, "2.copy", 2):   # a hand copy's non-integer stem sorts last instead of raising
        write_run_record(tmp_path / f"run_record.resume_iter{n}.yaml", _run("--gamma", "0.9"), boards=None,
                         echo=False)
    lines, _ = _explain(capsys, str(tmp_path))
    assert lines[0].split() == ["key", "value", "source", "where"]
    assert any(line.split()[:3] == ["gamma", "0.9", "cli"] for line in lines)
    assert "more keys at their schema default" in lines[-2]
    # resumed runs are named, in iteration order (not 11 before 2)
    assert lines[-1].endswith("run_record.resume_iter2.yaml, run_record.resume_iter11.yaml, "
                              "run_record.resume_iter2.copy.yaml")
    lines, _ = _explain(capsys, str(tmp_path), "--all")
    assert any(line.split()[:1] == ["lr"] for line in lines)


def test_explain_reads_a_tool_folder_from_its_run_record(tmp_path, capsys):
    """A tool's output folder has a named record (``run_record_<tag>.yaml``) and no
    ``run_record.yaml``: explain reads the newest record, not a checkpoint lying there."""
    import os

    from tests.helpers.ckpt_payload import save_ckpt

    old = write_run_record(tmp_path / "run_record_0_8.yaml", _run("--gamma", "0.8"), boards=None, echo=False)
    write_run_record(tmp_path / "run_record_8_8.yaml", _run("--gamma", "0.9"), boards=None, echo=False)
    os.utime(old, (1, 1))
    save_ckpt(tmp_path / "policy_iter_10.pt", gamma=0.97)
    lines, err = _explain(capsys, str(tmp_path))
    assert "reading its newest run record run_record_8_8.yaml (also there: run_record_0_8.yaml)" in err
    assert any(line.split()[:3] == ["gamma", "0.9", "cli"] for line in lines)


def test_explain_summarises_a_long_list_of_other_records(tmp_path, capsys):
    """A tool folder with hundreds of records: the stderr note names the first 3 and the last 2
    of the others and their count, not all of them."""
    import os

    from configs.loader.explain import _target

    for i in range(300):
        rec = tmp_path / f"run_record_{i:03d}.yaml"
        rec.write_text("schema: 2\n")
        os.utime(rec, (i + 1, i + 1))
    assert _target(tmp_path).name == "run_record_299.yaml"
    err = capsys.readouterr().err
    assert ("(also there: run_record_000.yaml, run_record_001.yaml, run_record_002.yaml, ..., "
            "run_record_297.yaml, run_record_298.yaml, 299 in all)") in err
    assert "run_record_150.yaml" not in err


@pytest.mark.filterwarnings("ignore:pre-schema-2 checkpoint")
def test_explain_reads_an_old_run_dir_from_its_newest_checkpoint(tmp_path, capsys):
    """A run dir written before run records has only checkpoints: explain reads the newest
    one and names it (it raised on the missing run_record.yaml)."""
    import os

    from tests.helpers.ckpt_payload import save_ckpt

    old, new = save_ckpt(tmp_path / "policy_iter_10.pt", gamma=0.97), save_ckpt(tmp_path / "policy_iter_20.pt", gamma=0.95)
    os.utime(old, (1, 1))
    lines, err = _explain(capsys, str(tmp_path), "--all")
    assert "reading its newest checkpoint policy_iter_20.pt" in err
    assert any(line.split()[:2] == ["gamma", "0.95"] for line in lines)


def test_explain_of_a_schema_2_checkpoint_keeps_the_writer_sources(tmp_path, capsys):
    from configs.loader.explain import load_provenance
    from tests.helpers.ckpt_payload import ckpt_payload

    torch = pytest.importorskip("torch")
    payload = ckpt_payload()
    payload["policy_state_dict"] = {k: torch.zeros(v.shape) for k, v in payload["policy_state_dict"].items()}
    run = _run("--gamma", "0.9", "--d-model", "256", "--n-layers", "6", "--d-ff", "1024")
    payload = {**{k: v for k, v in payload.items() if k != "args"}, "config": run.ckpt_config(), "trainer": "ppo"}
    torch.save(payload, tmp_path / "p.pt")
    prov = load_provenance(tmp_path / "p.pt")
    assert (prov["gamma"]["value"], prov["gamma"]["source"]) == (0.9, "cli")
    assert prov["use_critic"]["source"] == "derived"                  # the loader's own fill


# ===================================================================
# 5. preflight_diff exit-code contract
# ===================================================================
_BASE_CFG = {"gamma": 0.99, "masking_rule": "hybrid", "n_envs": 8}


def _write_cfg(tmp_path: Path, case: str, *, created: str = "2026-07-06", **overrides) -> Path:
    values = dict(_BASE_CFG, **overrides)
    path = tmp_path / case / "run_record.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({
        "created": created,
        "config": {k: {"value": v, "source": "cli", "where": f"--{k}"} for k, v in values.items()}}))
    return path


def _run_preflight(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PREFLIGHT), *argv],
        capture_output=True, text=True, timeout=60,
    )


class TestPreflightDiff:
    def test_identical_configs_exit_0_meta_ignored(self, tmp_path):
        a = _write_cfg(tmp_path, "caseA")
        b = _write_cfg(tmp_path, "caseB", created="2026-07-07")
        res = _run_preflight(str(a), str(b))
        assert res.returncode == 0, res.stdout + res.stderr
        assert "identical" in res.stdout

    def test_unexpected_diff_exit_1(self, tmp_path):
        a = _write_cfg(tmp_path, "caseA")
        b = _write_cfg(tmp_path, "caseB", gamma=1.0)
        res = _run_preflight(str(a), str(b))
        assert res.returncode == 1
        assert "gamma" in res.stdout
        assert "FAIL" in res.stdout

    def test_expected_diff_exit_0_with_matrix(self, tmp_path):
        a = _write_cfg(tmp_path, "caseA")
        b = _write_cfg(tmp_path, "caseB", gamma=1.0)
        res = _run_preflight(str(a), str(b), "--expect", "gamma")
        assert res.returncode == 0, res.stdout + res.stderr
        assert "gamma" in res.stdout          # matrix row
        assert "0.99" in res.stdout and "1.0" in res.stdout

    def test_absent_key_is_unexpected_diff(self, tmp_path):
        a = _write_cfg(tmp_path, "caseA")
        b = _write_cfg(tmp_path, "caseB", time_feature="sin_remaining")
        res = _run_preflight(str(a), str(b))
        assert res.returncode == 1
        assert "time_feature" in res.stdout

        res = _run_preflight(str(a), str(b), "--expect", "time_feature")
        assert res.returncode == 0
        assert "<absent>" in res.stdout

    def test_glob_input_and_expected_axis_warning(self, tmp_path):
        _write_cfg(tmp_path, "caseA")
        _write_cfg(tmp_path, "caseB")
        res = _run_preflight(
            str(tmp_path / "*" / "run_record.yaml"),
            "--expect", "gamma,masking_rule",
        )
        # nothing actually differs -> exit 0 but warn the axes are silent
        assert res.returncode == 0
        assert "WARNING" in res.stdout

    def test_a_file_that_is_not_a_run_record_is_refused(self, tmp_path):
        a = _write_cfg(tmp_path, "caseA")
        bad = tmp_path / "old.yaml"
        bad.write_text(yaml.safe_dump({"gamma": 0.99}))
        res = _run_preflight(str(a), str(bad))
        assert res.returncode != 0 and "not a run record" in (res.stdout + res.stderr)


# ===================================================================
# 6. tier-2 tools: record_tool_run and the provenance chain
# ===================================================================
def _tool_parse(argv):
    from configs.loader.cli import StrictArgumentParser

    ap = StrictArgumentParser(prog="tool.py")
    ap.add_argument("src")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--task", default="a")
    ap.add_argument("--out", type=Path, default=None)
    return ap, ap.parse_args(argv)


class TestToolRecord:
    def test_config_is_defaults_cli_and_what_the_tool_changed_after_parsing(self, tmp_path):
        from configs.loader.record import record_tool_run

        ap, args = _tool_parse(["data.txt", "--n", "5"])
        args.task = "b"   # a tool deriving a value after parsing
        rec = yaml.safe_load(record_tool_run(ap, args, out_dir=tmp_path).read_text())
        cfg = rec["config"]
        assert (cfg["src"]["value"], cfg["src"]["source"]) == ("data.txt", "cli")   # a positional is given
        assert (cfg["n"]["value"], cfg["n"]["source"]) == (5, "cli")
        assert (cfg["task"]["value"], cfg["task"]["source"]) == ("b", "code:tool.py")
        assert (cfg["out"]["value"], cfg["out"]["source"]) == (None, "defaults")
        assert rec["extra"]["inputs"] == [] and rec["extra"]["argv"] == sys.argv

    def test_a_file_output_gets_its_record_beside_it(self, tmp_path):
        from configs.loader.record import FILE_RECORD_SUFFIX, RUN_RECORD, record_tool_run

        ap, args = _tool_parse(["x"])
        out = tmp_path / "rows.jsonl"
        assert record_tool_run(ap, args, out_file=out) == tmp_path / ("rows.jsonl" + FILE_RECORD_SUFFIX)
        assert record_tool_run(ap, args, out_dir=tmp_path, name="run_record.x.yaml") == tmp_path / "run_record.x.yaml"
        assert record_tool_run(ap, args, out_dir=tmp_path / "d") == tmp_path / "d" / RUN_RECORD
        with pytest.raises(ValueError, match="exactly one"):
            record_tool_run(ap, args, out_dir=tmp_path, out_file=out)

    def test_a_consumer_records_the_hash_of_the_producer_record(self, tmp_path):
        from configs.loader.record import record_tool_run

        ap, args = _tool_parse(["x"])
        produced = tmp_path / "produced.json"
        produced.write_text("{}")
        producer = record_tool_run(ap, args, out_file=produced)
        prod_dir = tmp_path / "gen"
        dir_rec = record_tool_run(ap, args, out_dir=prod_dir, name="run_record_0_16.yaml")
        plain = tmp_path / "plain.txt"
        plain.write_text("no record\n")
        bare_dir = tmp_path / "bare"
        bare_dir.mkdir()
        rec = yaml.safe_load(record_tool_run(
            ap, args, out_dir=tmp_path / "out", inputs=[produced, prod_dir, plain, bare_dir, None]).read_text())
        by_path = {Path(i["path"]).name: i for i in rec["extra"]["inputs"]}
        assert by_path["produced.json"]["records"] == {producer.name: _sha256(producer)}
        assert "sha256" not in by_path["produced.json"]    # a recorded input is identified by its record
        assert by_path["gen"]["records"] == {dir_rec.name: _sha256(dir_rec)}
        assert by_path["plain.txt"] == {"path": str(plain), "records": {}, "sha256": _sha256(plain)}
        assert by_path["bare"]["records"] == {} and "reason" in by_path["bare"]

    def test_a_record_written_into_its_input_dir_leaves_its_own_record_out(self, tmp_path):
        from configs.loader.record import record_tool_run

        ap, args = _tool_parse(["x"])
        rollout = record_tool_run(ap, args, out_dir=tmp_path)
        again = record_tool_run(ap, args, out_dir=tmp_path, name="run_record.score.yaml", inputs=[tmp_path])
        record_tool_run(ap, args, out_dir=tmp_path, name="run_record.score.yaml", inputs=[tmp_path])
        inputs = yaml.safe_load(again.read_text())["extra"]["inputs"]
        assert list(inputs[0]["records"]) == [rollout.name]

    def test_a_missing_input_raises(self, tmp_path):
        from configs.loader.record import record_tool_run

        ap, args = _tool_parse(["x"])
        with pytest.raises(FileNotFoundError, match="does not exist"):
            record_tool_run(ap, args, out_dir=tmp_path, inputs=[tmp_path / "nope"])

    def test_an_appending_run_keeps_the_earlier_records(self, tmp_path):
        """A tool that appends rows to its output (a resumed shard) writes ``<stem>.<k>.yaml``
        instead of overwriting the record of the run that wrote the earlier rows; a consumer of
        the output names every one of them."""
        from configs.loader.record import record_tool_run

        out = tmp_path / "shard.jsonl"
        out.write_text("{}\n")
        first = record_tool_run(*_tool_parse(["x", "--n", "1"]), out_file=out, append=True)
        before = first.read_text()
        second = record_tool_run(*_tool_parse(["x", "--n", "2"]), out_file=out, append=True)
        third = record_tool_run(*_tool_parse(["x", "--n", "3"]), out_file=out, append=True)
        assert [p.name for p in (first, second, third)] == [
            "shard.jsonl.run_record.yaml", "shard.jsonl.run_record.1.yaml", "shard.jsonl.run_record.2.yaml"]
        assert first.read_text() == before
        assert yaml.safe_load(third.read_text())["config"]["n"]["value"] == 3
        gen = tmp_path / "gen"
        tagged = [record_tool_run(*_tool_parse(["x"]), out_dir=gen, name="run_record_0_8.yaml", append=True)
                  for _ in range(2)]
        assert [p.name for p in tagged] == ["run_record_0_8.yaml", "run_record_0_8.1.yaml"]
        ap, args = _tool_parse(["x"])
        (inp,) = yaml.safe_load(record_tool_run(ap, args, out_dir=tmp_path / "c", inputs=[out])
                                .read_text())["extra"]["inputs"]
        assert inp["records"] == {p.name: _sha256(p) for p in (first, second, third)}

    def test_a_file_without_its_own_record_names_its_folder_records(self, tmp_path):
        """A checkpoint has no ``<file>.run_record.yaml``: its input entry names the directory
        records of the run folder it sits in (not another file's record), marked as such."""
        from configs.loader.record import input_record

        run = tmp_path / "run"
        run.mkdir()
        train = write_run_record(run, _run(), boards=None, echo=False)
        val = write_run_record(run / "run_record.val.yaml", _run(), boards=None, echo=False)
        (run / "rows.jsonl.run_record.yaml").write_text("schema: 2\n")   # another file's record
        ckpt = run / "policy_iter_10.pt"
        ckpt.write_bytes(b"weights")
        # the folder records name the run, not which of its checkpoints was read: its own hash too
        assert input_record(ckpt) == {"path": str(ckpt), "sha256": _sha256(ckpt), "records_from": "folder",
                                      "records": {train.name: _sha256(train), val.name: _sha256(val)}}
        lone = tmp_path / "lone.pt"
        lone.write_bytes(b"weights")
        assert input_record(lone) == {"path": str(lone), "records": {}, "sha256": _sha256(lone)}

    def test_a_missing_input_leaves_no_empty_record(self, tmp_path):
        """An appending run that fails before writing (a missing input) reserves no record name:
        an empty ``<stem>.<k>.yaml`` left behind would be hashed as a producer by the next run."""
        from configs.loader.record import record_tool_run

        out = tmp_path / "shard.jsonl"
        out.write_text("{}\n")
        first = record_tool_run(*_tool_parse(["x"]), out_file=out, append=True)
        for _ in range(2):
            with pytest.raises(FileNotFoundError, match="does not exist"):
                record_tool_run(*_tool_parse(["x"]), out_file=out, append=True, inputs=[tmp_path / "nope"])
        assert sorted(tmp_path.glob("*run_record*.yaml")) == [first]
        with pytest.raises(FileNotFoundError, match="does not exist"):
            record_tool_run(*_tool_parse(["x"]), out_dir=tmp_path / "gen", append=True, inputs=[tmp_path / "nope"])
        assert not (tmp_path / "gen").exists()

    def test_a_free_record_name_is_taken_atomically(self, tmp_path):
        """The free ``<stem>.<k>.yaml`` is created when it is chosen (``open(..., "x")``), so a
        second run appending at once cannot pick it too; a name already taken is skipped."""
        from configs.loader.record import _free_record_path

        base = tmp_path / "new" / "run_record.yaml"
        assert _free_record_path(base) == base and base.is_file()
        (tmp_path / "new" / "run_record.1.yaml").write_text("taken\n")
        second = _free_record_path(base)
        assert second.name == "run_record.2.yaml" and second.is_file()
        assert _free_record_path(base).name == "run_record.3.yaml"
        assert (tmp_path / "new" / "run_record.1.yaml").read_text() == "taken\n"

    def test_sha256_hashes_a_file_larger_than_one_read(self, tmp_path):
        import hashlib

        big = tmp_path / "big.bin"
        big.write_bytes(os.urandom((1 << 20) * 2 + 7))
        assert _sha256(big) == hashlib.sha256(big.read_bytes()).hexdigest()
