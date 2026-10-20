"""Give-up diagnostic: per-rollout record + ``giveup_*`` aggregation.

A give-up = a successful ``net_end`` on a net with ratsnest edges left. The
rollout records the events and the end-state unrouted nets per rollout
(``eval_utils.GIVEUP_FIELDS``); ``eval.aggregation.giveup_quality`` turns them
into ``giveup_count`` / ``giveup_precision`` / ``giveup_fp_delta`` on both the
inline (training val) and the Stage-3 path, without touching existing metrics.
"""
from __future__ import annotations

import csv
import json
import math
import types

import numpy as np
import pytest

from eval import eval_utils as u
from eval.aggregation import aggregate_boards, aggregate_inline, giveup_log, giveup_quality
from methods._shared.board_loader import BoardSpec
from pcb_world.core.action_schema import ACT_MAKE_LINE, ACT_NET_END

GIVEUP_KEYS = {"giveup_read_rollouts", "giveup_unread_rollouts", "giveup_count",
               "giveup_judged", "giveup_precision", "giveup_fp_delta", "giveup_fp_delta_boards"}


def _rec(board_id, rollout_idx, events, unrouted, fp, board_index=0):
    """An in-memory rollout row carrying the give-up record."""
    return {"board_index": board_index, "board_id": board_id, "rollout_idx": rollout_idx,
            "status": "completed", "final_potential": fp, "success": not unrouted,
            **u.giveup_columns(events, unrouted)}


def _toy_rows():
    return [
        # board a, 3 rollouts. r0 gives up net 5, but r1 finishes it -> imprecise.
        _rec("a", 0, [[5, 10, 2]], [5, 7], 1.0),
        _rec("a", 1, [], [7], 3.0),
        # r2 gives up net 7 (twice: one judged (rollout, net)); no other rollout
        # finished 7 -> precise.
        _rec("a", 2, [[7, 4, 1], [7, 9, 1]], [7], 2.0),
        # board b, one rollout: nothing to compare against -> not judged.
        _rec("b", 0, [[1, 3, 1]], [1], 0.0, board_index=1),
        # board c: the give-up rollout's end state was not read (None) -> left out
        # of every giveup_* key (count, precision, Φ difference).
        _rec("c", 0, [[2, 5, 3]], None, 9.0, board_index=2),
        _rec("c", 1, [], [], 4.0, board_index=2),
    ]


# ---------------------------------------------------------------------------
# unit: event detection, record columns, the precision rule
# ---------------------------------------------------------------------------


def test_giveup_event_is_a_successful_net_end_on_an_unfinished_net():
    from methods.rl_agent.rollout.transformer import giveup_event

    def info(action_type, success=True, remaining=2):
        return {"action_type": action_type, "action_success": success, "step": 12,
                "dispatch_info": {"net_id": 4, "remaining": remaining}}

    assert giveup_event(info(ACT_NET_END)) == [4, 12, 2]
    assert giveup_event(info(ACT_NET_END, remaining=0)) is None     # normal close
    assert giveup_event(info(ACT_NET_END, success=False)) is None
    assert giveup_event(info(ACT_MAKE_LINE)) is None
    assert giveup_event({"engine_crash": True}) is None


def test_giveup_columns_round_trip_through_the_csv(tmp_path):
    row = {f: "" for f in u.PER_ROLLOUT_FIELDS}
    row.update(u.giveup_columns([[5, 10, 2]], {7, 5}))
    assert row["giveup_count"] == 1
    assert row["giveup_events"] == "[[5,10,2]]" and row["unrouted_nets"] == "[5,7]"
    path = tmp_path / "per_rollout.csv"
    u.write_csv(path, [row], u.PER_ROLLOUT_FIELDS)
    (back,) = u.read_csv(path)
    assert u.parse_json_list(back["giveup_events"]) == [[5, 10, 2]]
    assert u.parse_json_list(back["unrouted_nets"]) == [5, 7]
    assert u.giveup_columns([], None)["unrouted_nets"] == ""
    assert u.parse_json_list("") is None and u.parse_json_list("oops") is None
    # a one-shot iterable of end-state nets is read once, for the filter and the column
    once = u.giveup_columns([[5, 1, 1]], (n for n in [7, 5]))
    assert once["unrouted_nets"] == "[5,7]" and once["giveup_count"] == 1


def test_giveup_quality_on_toy_rollouts():
    q = giveup_quality(_toy_rows(), board_key="board_id")
    assert set(q) == GIVEUP_KEYS
    assert (q["giveup_read_rollouts"], q["giveup_unread_rollouts"]) == (5, 1)   # board c r0
    assert q["giveup_count"] == 4               # every event of a read rollout, judged or not
    assert q["giveup_judged"] == 2              # (a r0, net 5) + (a r2, net 7)
    assert q["giveup_precision"] == pytest.approx(0.5)
    # board a only: mean(1.0, 2.0) with a give-up - 3.0 without
    assert q["giveup_fp_delta"] == pytest.approx(-1.5)
    assert q["giveup_fp_delta_boards"] == 1


def test_giveup_quality_without_judgeable_events():
    q = giveup_quality([_rec("a", 0, [], [], 1.0), _rec("b", 0, [[3, 1, 1]], [3], 0.0, 1)],
                       board_key="board_id")
    assert q["giveup_count"] == 1 and q["giveup_judged"] == 0
    assert math.isnan(q["giveup_precision"]) and math.isnan(q["giveup_fp_delta"])


def test_a_giveup_the_rollout_undid_is_not_counted():
    """net_reselect: net 7 given up, then reopened and finished by the same
    rollout (not in its end-state unrouted nets) -> dropped from the row itself,
    so the row and the aggregate agree."""
    row = u.giveup_columns([[7, 3, 2], [5, 4, 1]], [5])
    assert row["giveup_count"] == 1 and row["giveup_events"] == "[[5,4,1]]"
    assert u.giveup_columns([[7, 3, 2]], None)["giveup_count"] == 1   # unread: kept as recorded
    q = giveup_quality([_rec("a", 0, [[7, 3, 2]], [], 1.0), _rec("a", 1, [], [7], 0.0)],
                       board_key="board_id")
    assert q["giveup_count"] == 0 and q["giveup_judged"] == 0
    assert math.isnan(q["giveup_fp_delta"])


def test_a_rollout_whose_end_state_was_not_read_adds_no_giveups():
    q = giveup_quality([_rec("a", 0, [[2, 5, 3]], None, 1.0)], board_key="board_id")
    assert q["giveup_count"] == 0 and q["giveup_judged"] == 0
    # "nothing read", not "never gave up"
    assert (q["giveup_read_rollouts"], q["giveup_unread_rollouts"]) == (0, 1)


def test_giveup_quality_takes_the_board_key_explicitly():
    """Inline keys boards by board_index, Stage 3 by board_id: no default that
    silently picks one when a watcher log is recomputed."""
    with pytest.raises(TypeError):
        giveup_quality(_toy_rows())


def test_inline_giveup_keys_boards_by_index_not_by_stem():
    """Two boards sharing a file stem are still two boards inline: one rollout
    each, so nothing is judged (keyed by board_id they would be pooled)."""
    rows = [_rec("board", 0, [[5, 1, 1]], [5], 1.0, board_index=0),
            _rec("board", 0, [], [], 2.0, board_index=1)]
    boards = [BoardSpec(0, "board", ""), BoardSpec(1, "board", "")]
    _, overall = aggregate_inline(rows, boards, selection_mode="final_potential")
    assert overall["giveup_count"] == 1 and overall["giveup_judged"] == 0


def test_finalize_keeps_the_artifact_when_the_giveup_read_kills_the_worker(tmp_path):
    """Stock eval path (artifacts saved): the end-state read is diagnostic and
    runs after save_pcb, so a worker it takes down cannot cost the artifact."""
    from methods.rl_agent.rollout.transformer import _finalize_slot

    class DyingPool:
        dead = False

        def env_method(self, name, *args, indices=None, **kwargs):
            if self.dead:
                raise BrokenPipeError("worker is gone")
            if name == "unrouted_net_ids":
                self.dead = True
                raise EOFError
            return [None]

    src = tmp_path / "a.kicad_pcb"
    src.write_text("")
    row = _finalize_slot(
        pool=DyingPool(), slot=0, board=BoardSpec(0, "a", str(src)), rollout_idx=0,
        seed=0, deterministic=False,
        info={"final_potential": -1.0, "initial_potential": -5.0, "unrouted_count": 1},
        reward=0.0, steps=3, terminated=True, truncated=False, save_artifacts=True,
        artifacts_dir=tmp_path / "boards", final_drc=False, reward_config="default",
        check_angle=45, per_board_rollout_time=0.0, giveups=[[5, 1, 1]],
    )
    assert row["artifact_path"]
    assert row["giveup_count"] == 1 and row["unrouted_nets"] == ""


def test_rows_without_a_giveup_record_add_no_keys():
    assert giveup_quality([{"board_id": "a", "final_potential": 1.0}], board_key="board_id") == {}
    assert giveup_quality([{f: "" for f in u.PER_ROLLOUT_FIELDS}], board_key="board_id") == {}
    assert giveup_log([{"board_id": "a"}]) == []


def test_giveup_log_is_compact_and_json_safe():
    log = giveup_log(_toy_rows())
    assert len(log) == 6
    assert log[0] == {"board_id": "a", "board_index": 0, "rollout_idx": 0, "seed": "", "ckpt": "",
                      "events": [[5, 10, 2]], "unrouted_nets": [5, 7], "final_potential": 1.0}
    assert log[4]["unrouted_nets"] is None
    json.dumps(log, allow_nan=False)
    # the log alone reproduces the metric (pooling with a paired run's log)
    as_rows = [{**r, "giveup_events": r["events"], "unrouted_nets": r["unrouted_nets"]}
               for r in log]
    assert giveup_quality(as_rows, board_key="board_index") == pytest.approx(
        giveup_quality(_toy_rows(), board_key="board_index"), nan_ok=True)


# ---------------------------------------------------------------------------
# aggregation entry points: new keys added, existing values untouched
# ---------------------------------------------------------------------------


def test_aggregate_inline_adds_giveup_keys_and_keeps_every_other_value():
    rows = _toy_rows()
    boards = [BoardSpec(0, "a", ""), BoardSpec(1, "b", ""), BoardSpec(2, "c", "")]
    stripped = [{k: v for k, v in r.items() if k not in u.GIVEUP_FIELDS} for r in rows]

    _, overall = aggregate_inline(rows, boards, selection_mode="final_potential")
    _, before = aggregate_inline(stripped, boards, selection_mode="final_potential")

    assert set(overall) - set(before) == GIVEUP_KEYS
    assert {k: overall[k] for k in before} == pytest.approx(before, nan_ok=True)
    assert overall["giveup_precision"] == pytest.approx(0.5)


def test_stage3_summary_carries_giveup_keys(tmp_path):
    cell_dir = tmp_path / "cell"
    (cell_dir / "boards").mkdir(parents=True)
    rows = []
    for r in _toy_rows():
        row = {f: "" for f in u.PER_ROLLOUT_FIELDS}
        row.update(r)
        row["artifact_path"] = f"{r['board_id']}_cell_s00_r{r['rollout_idx']:02d}.kicad_pcb"
        (cell_dir / "boards" / row["artifact_path"]).write_text("")
        rows.append(row)
    u.write_csv(cell_dir / "per_rollout.csv", rows, u.PER_ROLLOUT_FIELDS)

    aggregate_boards(cell_dir)

    with open(cell_dir / "per_boards_summary.csv") as f:
        (summary,) = list(csv.DictReader(f))
    assert float(summary["giveup_count"]) == 4
    assert int(summary["giveup_read_rollouts"]) == 5 and int(summary["giveup_unread_rollouts"]) == 1
    assert float(summary["giveup_precision"]) == pytest.approx(0.5)
    assert float(summary["giveup_fp_delta"]) == pytest.approx(-1.5)


# ---------------------------------------------------------------------------
# integration: Evaluator(eval_transformer) -> rows, overall, logger scalars
# ---------------------------------------------------------------------------


class _Pool:
    """Fake env pool: the end state of every slot leaves net 5 unrouted."""

    def __init__(self, n):
        self.n = n
        self.calls = []

    def reload_boards(self, paths):
        self.paths = list(paths)

    def reset_batch(self, indices, seeds=None):
        obs = {"board_static": {"nets": {i: {} for i in range(3)}}}
        return [(obs, {"initial_potential": -5.0}) for _ in indices]

    def env_method(self, name, *args, indices=None, **kwargs):
        self.calls.append((name, list(indices)))
        assert name == "unrouted_net_ids"
        return [[5] for _ in indices]

    def close(self):
        pass


def _scripted_rollout(pool, policy, device, obs_by_slot, active, done, **kw):
    """Two steps per slot: a give-up of net 5 (2 edges left), then the
    all-nets-closed termination. Slot 0 alone also closes a finished net."""
    live = [s for s in active if not done[s]]
    infos = [{"action_type": ACT_NET_END, "action_success": True, "step": 1,
              "action_class": "valid_effective",
              "dispatch_info": {"net_id": 5, "remaining": 2 if s else 0}} for s in live]
    yield types.SimpleNamespace(
        actions=np.zeros((len(live), 1), dtype=np.int64), live=live,
        rewards=[0.0] * len(live), terminateds=[False] * len(live),
        truncateds=[False] * len(live), infos=infos)
    final = {"action_type": ACT_NET_END, "action_success": True, "step": 2,
             "action_class": "valid_effective", "final_potential": -2.0,
             "initial_potential": -5.0, "unrouted_count": 2, "initial_unconnected": 4,
             "dispatch_info": {"net_id": 6, "remaining": 0}}
    yield types.SimpleNamespace(
        actions=np.zeros((len(live), 1), dtype=np.int64), live=live,
        rewards=[1.0] * len(live), terminateds=[True] * len(live),
        truncateds=[False] * len(live), infos=[dict(final) for _ in live])


def test_evaluator_reports_giveup_metrics_next_to_the_headline(monkeypatch):
    import eval.rollout.rl as rl
    import methods.rl_agent.rollout.transformer as transformer
    import methods.rl_agent.wrappers.factory as factory
    from eval.evaluator import Evaluator

    pools = []

    def make_pool(path, n, cfg, **kw):
        pools.append(_Pool(n))
        return pools[-1]

    monkeypatch.setattr(factory, "make_decoder_env_pool", make_pool)
    monkeypatch.setattr(transformer, "iter_rollout", _scripted_rollout)
    policy = types.SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    boards = [BoardSpec(0, "a", "/b/a.kicad_pcb")]

    from configs.loader.schema import RLEnvConfig

    def rollout_fn(bs):
        return rl.eval_transformer(policy, None, bs, env_config=RLEnvConfig(), n_rollouts=2, n_envs=2,
                                   base_seed=0, max_steps=10,
                                   selection_mode="final_potential")

    summary = Evaluator(rollout_fn, boards).run()

    rows = sorted(summary.per_rollout, key=lambda r: r["rollout_idx"])
    # slot 0 closed a finished net (no give-up); slot 1 gave net 5 up
    assert [r["giveup_count"] for r in rows] == [0, 1]
    assert rows[1]["giveup_events"] == "[[5,1,2]]"
    assert [r["unrouted_nets"] for r in rows] == ["[5]", "[5]"]
    assert sorted(c[1] for c in pools[0].calls) == [[0], [1]]
    # the other rollout left net 5 unrouted too -> a precise give-up
    assert summary.overall["giveup_precision"] == 1.0
    assert summary.overall["giveup_count"] == 1
    assert summary.overall["giveup_fp_delta"] == 0.0
    scalars = summary.to_logger_dict(prefix="val", include_per_board=False)
    assert scalars["val/giveup_precision"] == 1.0
    assert "val/fp_mean_of_means" in scalars
