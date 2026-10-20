"""LLM rollout early stops and the ``early_stop_reason`` they record.

``_run_one_episode`` is driven with a stub manager (no engine, no API): a board
that never changes trips ``--early_stop_no_geometry_progress``, a streak of
rejected steps trips ``--early_stop_no_progress``, and the reason reaches the
canonical row, the episode dump and a resume of that dump.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

from methods._shared.early_stop import NoGeometryProgressGuard
from methods.llm_agent.rollout import pcbworld as R

_STALL = {"track_count": 4, "via_count": 1, "wirelength": 12.5,
          "ratsnest_reduction": 0.5}
# PCBWorld's reset info carries the rules the board is routed under.
_RULES = {"rules_digest": "digest-b0",
          "rules_substitution": {"drc_config_path": "d.yaml", "minima": {"min_clearance_mm": 0.2}}}


def test_guard_counts_unchanged_steps_and_resets_on_change():
    g = NoGeometryProgressGuard(threshold=3)
    assert g.update(info=_STALL) is False          # first sighting: baseline
    assert g.update(info=_STALL) is False and g.count == 1
    assert g.update(info={**_STALL, "wirelength": 13.0}) is False and g.count == 0
    assert g.update(info={**_STALL, "wirelength": 13.0}) is False
    assert g.update(info={**_STALL, "wirelength": 13.0}) is False
    assert g.update(info={**_STALL, "wirelength": 13.0}) is True and g.count == 3


class _Manager:
    """Stub KiCadLLMRolloutManager: the board never changes.

    ``valid=True`` = every step is accepted (geometry stall only);
    ``valid=False`` = every step is rejected and carries no board measures.
    """

    def __init__(self, max_steps, valid=True):
        self.effective_max_steps = max_steps
        self.episode_scored = [None]
        self.marked_done = []
        self._last_actions = [{}]
        self._last_valids = [valid]
        self._valid = valid

    @staticmethod
    def _obs():
        return {"anchor": ["obs"], "system": ["sys"], "text": ["usr"]}

    def reset(self):
        return self._obs(), [{**_STALL, **_RULES}]

    def step(self, _outputs):
        info = ({**_STALL, "action_success": True} if self._valid
                else {"action_success": False})
        return self._obs(), [0.0], [False], [info]

    def mark_episode_done(self, idx, infos):
        self.marked_done.extend(idx)

    def finalize_episode(self, infos):
        pass

    def build_text_obs(self, obs, action_masks, init):
        return ["sys"], ["final"]


class _Provider:
    def generate(self, prompt_pairs):
        return ["<think>t</think><action>make_line 1 1 w</action>"], [(1, 1, 1)]


class _Envs:
    """Dump-side stub: obs dicts and a board save that just touches the file."""

    def get_last_obs_dicts(self):
        return [{}]

    def save_boards(self, paths):
        for p in paths:
            open(p, "w").close()
        return paths


def _run(*, geometry=0, no_progress=0, max_steps=20, valid=True, dump_dir=None):
    args = SimpleNamespace(env_num=1, max_steps=max_steps, verbose=False,
                           dump_dir=dump_dir, state_format="sexpr",
                           early_stop_no_progress=no_progress,
                           early_stop_no_geometry_progress=geometry)
    mgr = _Manager(max_steps, valid=valid)
    (result,) = R._run_one_episode(
        args, _Provider(), mgr, _Envs(), "b0", "b0.kicad_pcb", 0, "v7",
        extract_think_action=lambda _t: None, is_no_progress=lambda _info: False,
    )
    return result, mgr


def test_stalled_geometry_stops_with_its_reason():
    result, mgr = _run(geometry=3)
    # step 0 sets the baseline, steps 1..3 are unchanged -> stop after 4 steps
    assert result["steps"] == 4
    assert result["early_stopped"] is True
    assert result["early_stop_reason"] == "no_geometry_progress"
    assert mgr.marked_done == [0]


def test_rejected_streak_stops_with_its_reason():
    result, _ = _run(geometry=3, no_progress=5, valid=False)
    assert result["steps"] == 5
    assert result["early_stop_reason"] == "no_progress"


def test_guards_off_run_to_max_steps():
    result, _ = _run(max_steps=6)
    assert result["steps"] == 6 and result["early_stopped"] is False
    assert result["early_stop_reason"] == ""


def test_early_stop_reason_reaches_canonical_row():
    result, _ = _run(geometry=3)
    row = R._result_to_canonical_row(result, board_index=0, board_path="b0.kicad_pcb",
                                     rollout_idx=0, check_angle=45)
    assert row["early_stop"] is True
    assert row["early_stop_reason"] == "no_geometry_progress"


def test_early_stopped_row_keeps_the_rules_from_the_reset_info(tmp_path):
    """The early stop ends the episode before any terminal info, so the row's
    rules come from the reset info — in the canonical row and through a resume."""
    result, _ = _run(geometry=3, dump_dir=str(tmp_path))
    assert result["early_stop_reason"] == "no_geometry_progress"
    row = R._result_to_canonical_row(result, board_index=0, board_path="b0.kicad_pcb",
                                     rollout_idx=0, check_angle=45)
    (resumed,) = R._load_episode_results(str(tmp_path), "b0", 0, 1)
    for r in (row, resumed):
        assert r["rules_digest"] == "digest-b0"
        assert json.loads(r["rules_substitution"]) == _RULES["rules_substitution"]


def test_manager_rows_of_an_episode_ended_outside_the_env_keep_the_rules():
    """``mark_episode_done`` on a step info (the eval CLI's early stop) still
    buffers the rules the reset info carried."""
    from methods.llm_agent.training.manager import KiCadLLMRolloutManager
    from methods.llm_agent.wrappers.memory import SimpleMemory

    class _ScoringEnvs:
        def reset(self):
            return ["obs"], None, [dict(_RULES)]

        def apply_adaptive_max_steps(self, base):
            return base

        def env_method(self, _name, indices, **_kw):
            return [None] * len(indices)

    m = KiCadLLMRolloutManager.__new__(KiCadLLMRolloutManager)
    m.envs, m._board_scheduler, m._base_max_steps = _ScoringEnvs(), None, 10
    m.memory, m._state_format, m.score_episodes = SimpleMemory(), "sexpr", True
    m._eval_reward_config, m._eval_check_angle = "default", 0
    m.reset_state()
    m.mark_episode_done([0], infos=[dict(_STALL)])
    (row,) = m.pop_episode_rows()
    assert row["rules_digest"] == "digest-b0"
    assert json.loads(row["rules_substitution"]) == _RULES["rules_substitution"]


def test_early_stop_reason_survives_dump_and_resume(tmp_path):
    _run(geometry=3, dump_dir=str(tmp_path))
    assert os.path.isfile(tmp_path / "b0_episode_00_env_00.kicad_pcb")
    (row,) = R._load_episode_results(str(tmp_path), "b0", 0, 1)
    assert row["early_stop_reason"] == "no_geometry_progress"


def test_crashed_result_has_empty_reason():
    assert R._crashed_result("b0", 0)["early_stop_reason"] == ""
