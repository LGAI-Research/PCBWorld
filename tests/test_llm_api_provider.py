"""API provider error handling and per-board token accounting (no network).

Errors no retry can fix (credit, key, permission, model) end the run instead of
being retried and replaced by idle steps — classified on the real SDK
exception classes (openai / anthropic / google-genai, pinned in
requirements.txt), whose calls are never made. The rollout writes one
``token_usage.json`` per board, summed from the per-episode usage each episode
dump carries, so it stays exact across resumes (the layout the d3b_astra
recipe's ``summarize.py`` / ``price.py`` read). ``test_llm_api_wiring.py``
covers the request shape of ``_call_openai``.
"""
from __future__ import annotations

import csv
import importlib
import json
import os
from types import SimpleNamespace

import httpx
import pytest

from configs.loader.schema import LLMConfig
from methods.llm_agent.policy import model_provider as MP
from methods.llm_agent.rollout import pcbworld as R
from methods.llm_agent.wrappers.prompts import get_pcbworld_prompts


def _provider(caller, monkeypatch, **over) -> MP.APIProvider:
    monkeypatch.setitem(MP._API_CALLERS, "openai", caller)
    args = dict(api_provider="openai", api_model="gpt-6-astra", temperature=0.7,
                max_new_tokens=64, api_key=None, api_reasoning_effort="medium")
    args.update(over)
    return MP.APIProvider(SimpleNamespace(**args))


# The SDK errors are built at test time, not import time: importing the three SDKs
# costs ~4 s, which every xdist worker would otherwise pay while collecting.
def _sdk(cls_path: str, status: int, message: str):
    """An openai / anthropic SDK status error as the client raises it
    (``cls_path`` = ``"<sdk module>.<error class>"``)."""
    module, name = cls_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    req = httpx.Request("POST", "https://api.example.invalid/v1/x")
    return cls(message, response=httpx.Response(status, request=req), body=None)


def _google(status: int, state: str, message: str, reason: str | None = None):
    from google.genai import errors as genai_errors
    err = {"code": status, "message": message, "status": state}
    if reason:
        err["details"] = [{"reason": reason}]
    return genai_errors.ClientError(status, {"error": err})


def _out_of_credit():
    return _sdk(
        "openai.RateLimitError", 429,
        "Error code: 429 - {'error': {'message': 'You exceeded your current quota', "
        "'code': 'insufficient_quota'}}")


_FATAL = {
    "openai_out_of_credit": _out_of_credit,
    "openai_bad_key": lambda: _sdk("openai.AuthenticationError", 401,
                                   "Incorrect API key provided"),
    "openai_no_model": lambda: _sdk("openai.NotFoundError", 404, "The model does not exist"),
    "anthropic_credit": lambda: _sdk(
        "anthropic.BadRequestError", 400,
        "Your credit balance is too low to access the Anthropic API."),
    "anthropic_no_model": lambda: _sdk("anthropic.NotFoundError", 404,
                                       "not_found_error: model: claude-x"),
    "anthropic_no_permission": lambda: _sdk("anthropic.PermissionDeniedError", 403,
                                            "permission_error"),
    "google_bad_key": lambda: _google(400, "INVALID_ARGUMENT", "API key not valid.",
                                      reason="API_KEY_INVALID"),
    "google_no_model": lambda: _google(404, "NOT_FOUND", "models/gemini-x is not found"),
    "google_no_permission": lambda: _google(403, "PERMISSION_DENIED", "Permission denied"),
}

_TRANSIENT = {
    "openai_rate_limit": lambda: _sdk("openai.RateLimitError", 429,
                                      "Rate limit reached for requests"),
    # Same words as OpenAI's out-of-credit message, but a per-minute limit.
    "google_per_minute_quota": lambda: _google(
        429, "RESOURCE_EXHAUSTED",
        "You exceeded your current quota, please check your plan. "
        "Please retry in 31s."),
}


@pytest.mark.parametrize("make_exc", _FATAL.values(), ids=_FATAL.keys())
def test_fatal_error_raises_without_retry(monkeypatch, make_exc):
    exc = make_exc()
    calls = []

    def caller(*_a, **_kw):
        calls.append(1)
        raise exc

    with pytest.raises(MP.FatalAPIError):
        _provider(caller, monkeypatch).generate([("sys", "usr")])
    assert len(calls) == 1          # no retry budget burnt, no idle step


@pytest.mark.parametrize("make_exc", _TRANSIENT.values(), ids=_TRANSIENT.keys())
def test_transient_429_is_retried(monkeypatch, make_exc):
    exc = make_exc()
    monkeypatch.setattr(MP.time, "sleep", lambda _s: None)
    outcomes = [exc, ("<think>t</think><action>net_end</action>", (10, 5), {})]

    def caller(*_a, **_kw):
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    assert not MP._is_fatal_api_error(exc)
    responses, _ = _provider(caller, monkeypatch).generate([("sys", "usr")])
    assert responses == ["<think>t</think><action>net_end</action>"]


def test_bad_request_falls_back_to_idle_and_counts_as_failed(monkeypatch):
    def caller(*_a, **_kw):
        raise _sdk("openai.BadRequestError", 400, "maximum context length exceeded")

    p = _provider(caller, monkeypatch)
    responses, _ = p.generate([("sys", "usr")])
    assert "<action>" not in responses[0]          # idle fallback
    assert (p.usage_totals["calls"], p.usage_totals["failed_calls"]) == (1, 1)


def test_unreported_cache_write_counts_as_zero():
    usage = SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(cached_tokens=900),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=12))
    assert MP._usage_details(usage) == {
        "cached_input_tokens": 900, "cache_write_tokens": 0, "reasoning_tokens": 12}


# Key order the recipe's token_usage.json carries.
_TOKEN_USAGE_KEYS = ["calls", "failed_calls", "input_tokens", "cached_input_tokens",
                     "cache_write_tokens", "output_tokens", "reasoning_tokens",
                     "uncached_input_tokens", "model", "reasoning_effort", "wall_time_s"]


def _usage(calls: int) -> dict:
    return {"calls": calls, "failed_calls": 0, "input_tokens": 1000,
            "cached_input_tokens": 600, "cache_write_tokens": 300,
            "output_tokens": 40, "reasoning_tokens": 5, "wall_time_s": 12.34}


def test_write_token_usage_layout_and_names(tmp_path):
    kw = dict(model="gpt-6-astra", reasoning_effort="medium")
    path = R._write_token_usage(str(tmp_path), "0225", 1, _usage(2), **kw)
    assert path == str(tmp_path / "token_usage.json")
    doc = json.loads((tmp_path / "token_usage.json").read_text())
    assert list(doc) == _TOKEN_USAGE_KEYS
    assert doc["uncached_input_tokens"] == 400      # input - cache reads (writes stay in)
    assert doc["wall_time_s"] == 12.3
    # several boards in one dump dir: one file per board
    assert R._write_token_usage(str(tmp_path), "0225", 3, _usage(2), **kw) == str(
        tmp_path / "0225_token_usage.json")


def test_write_token_usage_marks_a_board_without_calls_done(tmp_path):
    """A complete board that made no call still gets its file, so run.sh and
    summarize.py read the cell as finished rather than re-entering it."""
    R._write_token_usage(str(tmp_path), "0225", 1, _usage(0), model=None,
                         reasoning_effort=None)
    assert json.loads((tmp_path / "token_usage.json").read_text())["calls"] == 0


# ---------------------------------------------------------------------------
# _run_rollout + the real _run_one_episode over stub envs / manager: every
# step is one provider call, an episode ends after ``steps[board]`` steps.
# ---------------------------------------------------------------------------

class _Envs:
    num_processes = 1
    env_config = LLMConfig().env
    state_format = "sexpr"

    def __init__(self, board_path):
        self.board = os.path.splitext(os.path.basename(board_path))[0]
        self.rebuilt = 0

    def reload_boards(self, paths):
        self.board = os.path.splitext(os.path.basename(paths[0]))[0]

    def rebuild_workers(self):
        self.rebuilt += 1

    def get_last_obs_dicts(self):
        return [{}]

    def save_boards(self, paths):
        for p in paths:
            open(p, "w").close()
        return paths


_SUBSTITUTION = {"0101": {"drc_config_path": "d.yaml", "minima": {"min_clearance_mm": 0.2}}}


class _Mgr:
    _prompts = get_pcbworld_prompts("v7")
    effective_max_steps = 1000
    _eval_check_angle = 45

    def __init__(self, envs, steps, crash_board=None):
        self.envs, self.steps, self.crash_board = envs, steps, crash_board
        envs.mgr = self
        self.closed = False
        self.episode_scored = [None]
        self._last_actions = [{}]
        self._last_valids = [True]

    @staticmethod
    def _obs():
        return {"anchor": ["obs"], "system": ["sys"], "text": ["usr"]}

    def _rules(self):   # PCBWorld's reset and terminal info: the rules the board ran under
        return {"rules_digest": f"digest-{self.envs.board}",
                "rules_substitution": _SUBSTITUTION.get(self.envs.board)}

    def reset(self):
        self.t = 0
        return self._obs(), [self._rules()]

    def step(self, _outputs):
        if self.envs.board == self.crash_board:
            raise RuntimeError("worker died")
        self.t += 1
        done = self.t >= self.steps[self.envs.board]
        info = {"action_success": True, "track_count": self.t}
        if done:
            info.update(self._rules())
        return self._obs(), [0.0], [done], [info]

    def mark_episode_done(self, idx, infos):
        pass

    def finalize_episode(self, infos):
        pass

    def build_text_obs(self, obs, action_masks, init):
        return ["sys"], ["final"]

    def close(self):
        self.closed = True


def _caller(fail_at=None):
    """Provider stub: 100 input / 10 output tokens per call; raises the
    out-of-credit error on call number ``fail_at`` (1-based)."""
    n = [0]

    def caller(*_a, **_kw):
        n[0] += 1
        if n[0] == fail_at:
            raise _out_of_credit()
        return "<think>t</think><action>net_end</action>", (100, 10), {
            "cached_input_tokens": 60, "cache_write_tokens": 30, "reasoning_tokens": 4}
    return caller


def _rollout(monkeypatch, tmp_path, provider, steps, *, episodes=1, crash_board=None):
    """Run ``_run_rollout``; returns (stub envs, the FatalAPIError or None)."""
    boards = [(b, f"{b}.kicad_pcb") for b in steps]
    holder: dict = {}

    def build_envs(_a, board_path=None):
        holder["envs"] = _Envs(board_path)
        return holder["envs"]

    monkeypatch.setattr(R, "_resolve_eval_boards", lambda _a: boards)
    monkeypatch.setattr(R, "_build_envs", build_envs)
    monkeypatch.setattr(R, "_make_manager",
                        lambda envs, _a: _Mgr(envs, steps, crash_board))
    args = SimpleNamespace(dump_dir=str(tmp_path), env_num=1, rollout_episodes=episodes,
                           max_steps=1000, verbose=False, state_format="sexpr",
                           early_stop_no_progress=0, early_stop_no_geometry_progress=0)
    try:
        R._run_rollout(args, provider)
    except MP.FatalAPIError as e:
        return holder["envs"], e
    return holder["envs"], None


def test_rollout_writes_one_token_usage_per_board(monkeypatch, tmp_path):
    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch),
             {"0101": 2, "0102": 3})
    a = json.loads((tmp_path / "0101_token_usage.json").read_text())
    b = json.loads((tmp_path / "0102_token_usage.json").read_text())
    assert list(a) == _TOKEN_USAGE_KEYS
    assert (a["calls"], a["input_tokens"], a["cache_write_tokens"]) == (2, 200, 60)
    assert (b["calls"], b["cached_input_tokens"], b["reasoning_tokens"]) == (3, 180, 12)
    assert (b["model"], b["reasoning_effort"]) == ("gpt-6-astra", "medium")
    assert not (tmp_path / "token_usage.json").exists()


def test_single_board_rollout_writes_token_usage_json(monkeypatch, tmp_path):
    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch), {"0101": 2})
    assert json.loads((tmp_path / "token_usage.json").read_text())["calls"] == 2


def test_rollout_manifest_records_the_env_config_and_prompt_hash(monkeypatch, tmp_path):
    """The eval output says which env and which prompt text produced it (the env as
    the flat PCBWorld kwargs, the shape the RL eval records)."""
    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch), {"0101": 2})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["env_kwargs"] == json.loads(json.dumps(LLMConfig().env.to_env_kwargs()))
    assert manifest["args"] == {"prompt_version": "v7", "state_format": "sexpr",
                                "prompt_content_hash": get_pcbworld_prompts("v7").content_hash}


def test_token_usage_spans_a_fatal_stop_and_its_resume(monkeypatch, tmp_path):
    """Run 1: episode 0 records 11 calls, episode 1 dies on its 3rd call.
    Run 2: resumes episode 0, runs episode 1. The board's file = 11 + 11."""
    envs, err = _rollout(monkeypatch, tmp_path,
                         _provider(_caller(fail_at=14), monkeypatch),
                         {"0101": 11}, episodes=2)
    assert isinstance(err, MP.FatalAPIError)
    assert envs.rebuilt == 0                              # not an episode crash
    assert envs.mgr.closed                                # workers shut down
    assert not (tmp_path / "token_usage.json").exists()   # board not complete

    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch),
             {"0101": 11}, episodes=2)
    doc = json.loads((tmp_path / "token_usage.json").read_text())
    assert (doc["calls"], doc["input_tokens"], doc["output_tokens"]) == (22, 2200, 220)


def test_fully_resumed_board_rebuilds_the_same_totals(monkeypatch, tmp_path):
    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch), {"0101": 4})
    first = json.loads((tmp_path / "token_usage.json").read_text())
    _, err = _rollout(monkeypatch, tmp_path, _provider(_caller(fail_at=1), monkeypatch),
                      {"0101": 4})                         # would fail on any call
    assert err is None
    again = json.loads((tmp_path / "token_usage.json").read_text())
    assert again == first


def test_zero_episodes_write_no_token_usage(monkeypatch, tmp_path):
    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch), {"0101": 2},
             episodes=0)
    assert not (tmp_path / "token_usage.json").exists()


def test_crashed_episode_leaves_no_token_usage(monkeypatch, tmp_path):
    envs, _ = _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch),
                       {"0101": 3}, crash_board="0101")
    assert envs.rebuilt == 1
    assert not (tmp_path / "token_usage.json").exists()


def test_rerun_episode_removes_the_stale_token_usage(monkeypatch, tmp_path, capsys):
    """A finished cell whose dumps were deleted to re-measure: the old file goes
    as soon as an episode re-runs, so a crash cannot leave the cell 'finished'."""
    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch), {"0101": 2})
    assert (tmp_path / "token_usage.json").exists()
    for dump in tmp_path.glob("0101_episode_*"):
        dump.unlink()
    _rollout(monkeypatch, tmp_path, _provider(_caller(fail_at=2), monkeypatch),
             {"0101": 2})
    assert not (tmp_path / "token_usage.json").exists()
    out = capsys.readouterr().out
    assert "removed" in out
    assert ("0101 episode 1: 1 calls (0 failed with no usage reported; 100 input / "
            "10 output tokens) are not in any token_usage file") in out   # the lost spend


class _LocalProvider:
    """A provider with no billing counters (vLLM / remote)."""

    def generate(self, prompt_pairs):
        return ["<think>t</think><action>net_end</action>"] * len(prompt_pairs), [
            (1, 1, 1)] * len(prompt_pairs)


def test_non_api_rerun_also_removes_the_stale_token_usage(monkeypatch, tmp_path):
    (tmp_path / "token_usage.json").write_text('{"calls": 7}\n')   # an earlier API run
    envs, _ = _rollout(monkeypatch, tmp_path, _LocalProvider(), {"0101": 2})
    assert envs.rebuilt == 0                                # no crash path
    assert (tmp_path / "0101_episode_00_env_00.json").exists()   # the episode was dumped
    assert not (tmp_path / "token_usage.json").exists()


def test_all_crashed_board_says_why_no_token_usage(monkeypatch, tmp_path, capsys):
    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch),
             {"0101": 3}, episodes=2, crash_board="0101")
    assert "not written — 2 crashed episode(s) unrecorded" in capsys.readouterr().out


def test_dumps_without_episode_usage_keep_the_existing_file(monkeypatch, tmp_path):
    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch), {"0101": 2})
    dump = tmp_path / "0101_episode_00_env_00.json"
    doc = json.loads(dump.read_text())
    del doc["token_usage"]                     # a dump from before usage was recorded
    dump.write_text(json.dumps(doc))
    earlier = '{"calls": 7}\n'
    (tmp_path / "token_usage.json").write_text(earlier)
    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch), {"0101": 2})
    assert (tmp_path / "token_usage.json").read_text() == earlier


# ---------------------------------------------------------------------------
# API failures that fell back to idle stay in the episode but are counted:
# provider record -> per_rollout ``api_failed_calls`` -> cell summary + WARN.
# ---------------------------------------------------------------------------

def _failing_caller(fail_calls):
    """Provider stub that raises a non-retried bad request on the given 1-based
    call numbers (each becomes an idle step) and answers normally otherwise."""
    n = [0]
    good = _caller()

    def caller(*a, **kw):
        n[0] += 1
        if n[0] in fail_calls:
            raise _sdk("openai.BadRequestError", 400, "maximum context length exceeded")
        return good(*a, **kw)
    return caller


def test_provider_records_which_pair_fell_back_to_idle(monkeypatch):
    p = _provider(_failing_caller({2}), monkeypatch)
    p.generate([("s", "u"), ("s", "u"), ("s", "u")])
    assert p.last_call_failed == [False, True, False]
    p.generate([("s", "u")])
    assert p.last_call_failed == [False]          # reset per call


def test_failed_api_steps_reach_rows_summary_and_warning(monkeypatch, tmp_path, capsys):
    # board 0101: 3 steps, calls 1 and 3 fail; board 0102: 2 steps, none fail.
    _rollout(monkeypatch, tmp_path, _provider(_failing_caller({1, 3}), monkeypatch),
             {"0101": 3, "0102": 2})
    rows = {r["board_id"]: r for r in csv.DictReader(open(tmp_path / "per_rollout.csv"))}
    assert (rows["0101"]["api_failed_calls"], rows["0102"]["api_failed_calls"]) == ("2", "0")
    assert int(rows["0101"]["steps"]) == 3                # the idle steps stay in the episode
    overall = json.loads((tmp_path / "eval_overall_summary.json").read_text())["overall"]
    assert (overall["api_failed_calls"], overall["api_failed_boards"]) == (2, 1)
    per_board = {r["board_id"]: r for r in csv.DictReader(open(tmp_path / "per_board_avg.csv"))}
    assert (per_board["0101"]["api_failed_calls"], per_board["0102"]["api_failed_calls"]) == (
        "2", "0")
    assert ("[WARN] 2 LLM step(s) on 1 board(s) fell back to idle after the provider "
            "call failed") in capsys.readouterr().out
    # a resume rebuilds the same count from the episode dumps
    _rollout(monkeypatch, tmp_path, _provider(_caller(), monkeypatch), {"0101": 3, "0102": 2})
    rows = {r["board_id"]: r for r in csv.DictReader(open(tmp_path / "per_rollout.csv"))}
    assert rows["0101"]["api_failed_calls"] == "2"


def test_provider_without_failure_record_leaves_the_column_blank(monkeypatch, tmp_path,
                                                                 capsys):
    _rollout(monkeypatch, tmp_path, _LocalProvider(), {"0101": 2})
    (row,) = csv.DictReader(open(tmp_path / "per_rollout.csv"))
    assert row["api_failed_calls"] == ""
    (board,) = csv.DictReader(open(tmp_path / "per_board_avg.csv"))
    assert board["api_failed_calls"] == ""
    overall = json.loads((tmp_path / "eval_overall_summary.json").read_text())["overall"]
    assert "api_failed_calls" not in overall
    assert "fell back to idle" not in capsys.readouterr().out


class _TwoEnvMgr:
    """Two envs on one board: env 0 finishes after step 1, env 1 after step 3."""
    _prompts = SimpleNamespace(version="v7")
    effective_max_steps = 1000

    def __init__(self):
        self.t = 0
        self.episode_scored = [None, None]
        self._last_actions = [{}, {}]
        self._last_valids = [True, True]

    @staticmethod
    def _obs():
        return {"anchor": ["obs"] * 2, "system": ["sys"] * 2, "text": ["usr"] * 2}

    def reset(self):
        self.t = 0
        return self._obs(), [{}, {}]

    def step(self, _outputs):
        self.t += 1
        infos = [{"action_success": True, "track_count": self.t} for _ in range(2)]
        return self._obs(), [0.0, 0.0], [self.t >= 1, self.t >= 3], infos

    def mark_episode_done(self, idx, infos):
        pass

    def finalize_episode(self, infos):
        pass

    def build_text_obs(self, obs, action_masks, init):
        return ["sys"] * 2, ["final"] * 2


def test_failed_calls_count_per_env_and_skip_finished_envs(monkeypatch):
    """One call per env per step (calls 1-2 = step 1, 3-4 = step 2, 5-6 = step 3).
    Calls 3 and 5 are env 0's after it finished; call 4 is env 1's step 2."""
    args = SimpleNamespace(dump_dir=None, env_num=2, max_steps=1000, verbose=False,
                           state_format="sexpr", early_stop_no_progress=0,
                           early_stop_no_geometry_progress=0)
    results = R._run_one_episode(
        args, _provider(_failing_caller({3, 4, 5}), monkeypatch), _TwoEnvMgr(), None,
        "0101", "", 0, "v7", lambda text: text, lambda info: False)
    assert [r["steps"] for r in results] == [3, 3]
    assert [r["api_failed_calls"] for r in results] == [0, 1]


def test_crashed_episode_keeps_the_column_as_not_recorded():
    res = R._crashed_result("0101", 0)
    assert "api_failed_calls" in res and res["api_failed_calls"] is None
    row = R._result_to_canonical_row(res, board_index=0, board_path="", rollout_idx=0, check_angle=45)
    assert row["api_failed_calls"] == ""


def test_llm_rows_carry_the_rules_the_board_ran_under(monkeypatch, tmp_path):
    """per_rollout.csv of an LLM eval records the rules each board ran under (the env's
    reset info), and a resume rebuilds them from the episode dumps."""
    for _ in range(2):                                   # run, then a full resume
        _rollout(monkeypatch, tmp_path, _LocalProvider(), {"0101": 2, "0102": 1})
        rows = {r["board_id"]: r for r in csv.DictReader(open(tmp_path / "per_rollout.csv"))}
        assert (rows["0101"]["rules_digest"], rows["0102"]["rules_digest"]) == (
            "digest-0101", "digest-0102")
        assert json.loads(rows["0101"]["rules_substitution"]) == _SUBSTITUTION["0101"]
        assert rows["0102"]["rules_substitution"] == ""


def test_llm_rows_record_the_angle_the_episode_was_scored_at(monkeypatch, tmp_path):
    """A scored row's ``check_angle`` is the manager's scoring angle (the env corner_mode),
    the value Stage 2 re-scoring reads back — not the 45 default."""
    class _ScoredMgr(_Mgr):
        _eval_check_angle = 90

        def finalize_episode(self, infos):
            self.episode_scored = [{"success": True, "routability": 1.0,
                                    "initial_potential": None, "potential_gain": None}]

    monkeypatch.setattr(R, "_make_manager",
                        lambda envs, _a: _ScoredMgr(envs, {"0101": 1}))
    monkeypatch.setattr(R, "_resolve_eval_boards", lambda _a: [("0101", "0101.kicad_pcb")])
    monkeypatch.setattr(R, "_build_envs", lambda _a, board_path=None: _Envs(board_path))
    args = SimpleNamespace(dump_dir=str(tmp_path), env_num=1, rollout_episodes=1,
                           max_steps=1000, verbose=False, state_format="sexpr",
                           early_stop_no_progress=0, early_stop_no_geometry_progress=0)
    R._run_rollout(args, _LocalProvider())
    (row,) = csv.DictReader(open(tmp_path / "per_rollout.csv"))
    assert (row["eval_status"], row["check_angle"]) == ("ok", "90")


def test_eval_manager_scores_at_the_env_corner_mode():
    """The LLM eval manager (flat CLI args, no ``env.pcbworld`` block) takes the
    workers' corner_mode as its scoring angle."""
    envs = SimpleNamespace(prompt_version="v5", state_format="sexpr",
                           env_config=LLMConfig().env.with_flat({"corner_mode": 90}))
    mgr = R._EvalRolloutManager(envs, SimpleNamespace(max_steps=10, history_length=0))
    assert mgr._eval_check_angle == 90
