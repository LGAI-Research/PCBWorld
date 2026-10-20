"""Linear-ladder knobs (added for W7a): pad-group-proportional net weights + board bonuses.

The log-shaped twins (``net_bonus_size_log_scale`` /
``completion_bonus_log_scale``) are covered by ``test_ladder_reward.py`` and
``test_reward_parity.py``; this file pins the LINEAR shape added for W7a and
the mutual-exclusion guards, so a future refactor cannot silently collapse the
two shapes into one. It also checks that the linear weights reach the
potential itself (not only the resolved weight maps) and that the shipped
``pcbworld_reward`` rule keeps the flat per-net ladder every run used.
"""
import importlib.util
import math
import sys
import warnings
from pathlib import Path

import pytest

import pcb_world.core.reward_config as rc
from pcb_world.core.reward import PotentialReward, RewardState
from pcb_world.core.reward_config import get_reward_config, load_reward_config
from pcb_world.engine.containers import BoardMeta, RewardSnapshot
from tests._rule_helpers import RULE_PENALTIES, yaml_lines

PAD_GROUPS = {1: 2, 2: 3, 3: 5}
NET_NAMES = {1: "a", 2: "b", 3: "c"}
ROUTABLE = frozenset(PAD_GROUPS)
TOTAL_PG = sum(PAD_GROUPS.values())  # 10


def _bind(**kwargs) -> PotentialReward:
    pr = PotentialReward(**kwargs)
    pr.bind_board(net_count=len(PAD_GROUPS), bbox_w=40.0, bbox_h=20.0)
    pr.bind_board(
        pad_groups=PAD_GROUPS, net_names=NET_NAMES, routable_nets=ROUTABLE
    )
    return pr


def test_linear_net_weights_are_proportional_to_pad_groups():
    pr = _bind(net_bonus_size_linear_scale=0.5)
    assert pr.net_size_weights_by_code == {c: 0.5 * pg for c, pg in PAD_GROUPS.items()}
    assert pr.net_size_weights_by_name == {
        NET_NAMES[c]: 0.5 * pg for c, pg in PAD_GROUPS.items()
    }


def test_log_shape_still_compresses():
    """The two shapes must stay distinguishable (linear > log for pg >= 2)."""
    lin = _bind(net_bonus_size_linear_scale=1.0).net_size_weights_by_code
    log = _bind(net_bonus_size_log_scale=1.0).net_size_weights_by_code
    assert log == {c: math.log1p(pg) for c, pg in PAD_GROUPS.items()}
    assert all(lin[c] > log[c] for c in PAD_GROUPS)


def test_board_bonuses_scale_with_total_pad_groups():
    pr = _bind(
        completion_bonus_padgroup_scale=0.5,
        clean_completion_bonus_padgroup_scale=0.25,
    )
    assert pr.completion_bonus == 0.5 * TOTAL_PG
    assert pr.clean_completion_bonus == 0.25 * TOTAL_PG


def test_board_bonus_rebinds_per_reset():
    """Keep-routing makes pad groups episode-dependent — a rebind must move
    the board bonus, not keep the first episode's value."""
    pr = _bind(completion_bonus_padgroup_scale=1.0)
    assert pr.completion_bonus == TOTAL_PG
    smaller = {1: 2, 2: 2, 3: 2}
    pr.bind_board(
        pad_groups=smaller, net_names=NET_NAMES, routable_nets=ROUTABLE
    )
    assert pr.completion_bonus == 6


def test_unresolved_pad_groups_fail_loudly():
    pr = PotentialReward(net_bonus_size_linear_scale=0.5)
    with pytest.raises(RuntimeError, match="no pad groups"):
        pr.bind_board(
            pad_groups={1: 2}, net_names=NET_NAMES, routable_nets=ROUTABLE
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(net_bonus_size_log_scale=1.0, net_bonus_size_linear_scale=0.5),
        dict(completion_bonus_log_scale=2.0, completion_bonus_padgroup_scale=0.5),
        dict(
            clean_completion_bonus_log_scale=2.0,
            clean_completion_bonus_padgroup_scale=0.5,
        ),
    ],
)
def test_shapes_are_mutually_exclusive(kwargs):
    with pytest.raises(ValueError, match="mutually exclusive"):
        PotentialReward(**kwargs)


def test_defaults_are_unchanged_without_the_knobs():
    """A config that never sets the linear knobs must behave exactly as before
    (the W-series campaigns depend on this)."""
    pr = _bind(net_bonus_size_log_scale=1.0, completion_bonus_log_scale=2.0)
    assert pr.completion_bonus == pytest.approx(2.0 * math.log1p(len(PAD_GROUPS)))
    assert pr.net_size_weights_by_code == {
        c: math.log1p(pg) for c, pg in PAD_GROUPS.items()
    }


# --- The linear weights reach the potential ---------------------------------

def _partial_state(**overrides) -> RewardState:
    """Nets a, b connected, net c (5 pad groups) open by 4 edges; net a dirty."""
    kw = dict(
        unconnected=4, drc_violations=1, wirelength=0.0, track_count=0,
        drc_violations_per_net={"a": 1}, drc_errors=1,
        drc_errors_per_net={"a": 1}, connected_net_count=2,
        target_net_count=3, unconnected_net_codes=frozenset({3}),
    )
    kw.update(overrides)
    return RewardState(**kw)


def _net_ladder(linear_scale: float) -> PotentialReward:
    """Edge term + per-net ladder only, so Φ isolates the ladder weights."""
    return _bind(
        completion_bonus=0.0, unconnected_penalty=1.0, drc_shape="linear",
        drc_penalty=0.0, step_penalty=0.0, net_completion_bonus=1.0,
        net_clean_bonus=1.0, net_bonus_size_linear_scale=linear_scale,
    )


def test_linear_weights_drive_the_potential():
    st = _partial_state()
    w = {c: 0.5 * pg for c, pg in PAD_GROUPS.items()}  # a=1.0, b=1.5, c=2.5
    expected = (
        -4.0                                  # 4 open edges
        + (w[1] + w[2])                       # connected nets a, b
        + (sum(w.values()) - w[1])            # clean nets = all but a
    )
    assert _net_ladder(0.5).potential(st) == pytest.approx(expected)  # 2.5
    # Flat ladder on the same state: 2 connected nets + 2 clean nets.
    assert _net_ladder(0.0).potential(st) == pytest.approx(-4.0 + 2 + 2)


def test_closing_a_net_pays_its_linear_weight():
    pr = _net_ladder(0.5)
    before = _partial_state()
    after = _partial_state(
        unconnected=0, connected_net_count=3, unconnected_net_codes=frozenset(),
    )
    # 4 edges closed + net c's weight 0.5 * 5.
    assert pr.potential(after) - pr.potential(before) == pytest.approx(4 + 2.5)


def test_linear_ladder_unresolved_weights_fail_loudly():
    pr = PotentialReward(net_completion_bonus=1.0, net_bonus_size_linear_scale=0.5)
    with pytest.raises(ValueError, match="unresolved"):
        pr.potential(_partial_state())


def _rule_yaml(path: Path, name: str, linear: float) -> Path:
    path.write_text(
        f"name: {name}\nmode: per_step\npotential:\n"
        f"  net_completion_bonus: 1.0\n"
        f"  net_bonus_size_linear_scale: {linear}\n"
        + yaml_lines(RULE_PENALTIES)
    )
    return path


def test_linear_rule_warns_once_per_rule_naming_it(tmp_path):
    a = _rule_yaml(tmp_path / "a.yaml", "rule_a", 0.5)
    b = _rule_yaml(tmp_path / "b.yaml", "rule_b", 0.25)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_reward_config(a)
        load_reward_config(a)          # same rule again: no repeat
        load_reward_config(b)          # distinct rule: its own warning
    msgs = [str(w.message) for w in caught
            if "net_bonus_size_linear_scale" in str(w.message)]
    assert len(msgs) == 2
    assert "'rule_a'" in msgs[0] and str(a.resolve()) in msgs[0]
    assert "'rule_b'" in msgs[1] and str(b.resolve()) in msgs[1]
    assert "w_i = scale * pg_i" in msgs[0]


def test_flat_rule_does_not_warn(tmp_path):
    flat = _rule_yaml(tmp_path / "flat.yaml", "rule_flat", 0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_reward_config(flat)
        get_reward_config("pcbworld_reward")
    assert not [w for w in caught
                if "net_bonus_size_linear_scale" in str(w.message)]


def test_flat_rule_load_skips_the_path_resolve(tmp_path, monkeypatch):
    """Path-loaded rules are re-read per board in eval; a flat rule must not
    pay the warning's path resolve (an NFS stat chain) on every load."""
    flat = _rule_yaml(tmp_path / "flat.yaml", "rule_flat", 0.0)
    linear = _rule_yaml(tmp_path / "linear.yaml", "rule_linear", 0.5)

    class _NoResolvePath(type(Path())):
        def resolve(self, *a, **k):
            raise AssertionError("rule resolved its path")

    monkeypatch.setattr(rc, "Path", _NoResolvePath)
    assert load_reward_config(flat).name == "rule_flat"
    with pytest.raises(AssertionError, match="resolved its path"):
        load_reward_config(linear)     # positive control: the patch intercepts


# --- Flat clean ladder keeps the net-name guard ------------------------------

def _flat_clean() -> PotentialReward:
    return PotentialReward(net_clean_bonus=1.0)


def test_flat_clean_ladder_rejects_duplicate_net_names():
    pr = _flat_clean()
    with pytest.raises(RuntimeError, match=r"share a net name \{'a': \[1, 2\]\}"):
        pr.bind_board(
            pad_groups=PAD_GROUPS, net_names={1: "a", 2: "a", 3: "c"},
            routable_nets=ROUTABLE,
        )


def test_name_guard_lists_every_shared_name_and_code():
    with pytest.raises(RuntimeError, match=r"\{'a': \[1, 2, 3\]\}"):
        _flat_clean().bind_board(
            pad_groups=PAD_GROUPS, net_names={1: "a", 2: "a", 3: "a"},
            routable_nets=ROUTABLE,
        )
    pg4 = {1: 2, 2: 2, 3: 2, 4: 2}
    with pytest.raises(RuntimeError, match=r"\{'a': \[1, 2\], 'b': \[3, 4\]\}"):
        _flat_clean().bind_board(
            pad_groups=pg4, net_names={1: "a", 2: "a", 3: "b", 4: "b"},
            routable_nets=frozenset(pg4),
        )


def test_flat_clean_ladder_rejects_unnamed_routable_net():
    pr = _flat_clean()
    with pytest.raises(RuntimeError, match="no net name"):
        pr.bind_board(
            pad_groups=PAD_GROUPS, net_names={1: "a", 2: "b"},
            routable_nets=ROUTABLE,
        )


def test_name_guard_ignores_non_routable_nets_and_ladder_off():
    names = {1: "a", 2: "b", 3: "c", 4: "a"}       # net 4 is not routable
    _flat_clean().bind_board(
        pad_groups=PAD_GROUPS, net_names=names, routable_nets=ROUTABLE,
    )
    PotentialReward().bind_board(                    # no per-net ladder term
        pad_groups=PAD_GROUPS, net_names={1: "a", 2: "a", 3: "a"},
        routable_nets=ROUTABLE,
    )


# --- Shipped rule: same Φ as every recorded run -----------------------------

def _shipped() -> PotentialReward:
    pr = get_reward_config("pcbworld_reward").build_reward()
    pr.bind_board(net_count=len(PAD_GROUPS), bbox_w=40.0, bbox_h=20.0)
    pr.bind_board(
        pad_groups=PAD_GROUPS, net_names=NET_NAMES, routable_nets=ROUTABLE
    )
    return pr


def test_shipped_rule_keeps_the_flat_net_ladder():
    """Regression guard: pcbworld_reward prices this state exactly as the
    pre-fix code did (flat per-net counts, 0.5·Σpg board bonuses). The
    expected value is written out term by term from the yaml's coefficients."""
    pr = _shipped()
    assert pr.net_size_weights_by_code is None
    st = RewardState(
        unconnected=4, drc_violations=3, wirelength=30.0, track_count=5,
        via_count=2, drc_violations_per_net={"a": 2, "b": 1},
        drc_errors=3, drc_errors_per_net={"a": 2, "b": 1},
        drc_promoted=3, drc_promoted_per_net={"a": 2, "b": 1},
        connected_net_count=2, target_net_count=3,
        unconnected_net_codes=frozenset({3}),
    )
    expected = (
        -1.0 * 4                                          # open edges
        - 3.0 * math.log1p(2 / 2.0)                       # DRC aggregate (2 nets)
        - (math.log1p(2 / 2.0) + math.log1p(1 / 2.0))     # DRC per net
        + 1.0 * 2                                         # connected nets, flat
        + 1.0 * (3 - 2)                                   # clean nets, flat
        - 0.5 * 30.0 / 40.0                               # WL / max(bbox)
        - 0.2 * 2                                         # vias
    )
    assert pr.potential(st) == pytest.approx(expected)

    done = RewardState(
        unconnected=0, drc_violations=0, wirelength=30.0, track_count=5,
        via_count=2, connected_net_count=3, target_net_count=3,
        unconnected_net_codes=frozenset(),
    )
    expected_done = (
        0.5 * TOTAL_PG            # board completion
        + 3 + 3                   # flat net completion + clean
        + 0.5 * TOTAL_PG          # board clean
        - 0.5 * 30.0 / 40.0 - 0.2 * 2
    )
    assert pr.potential(done) == pytest.approx(expected_done)


# --- Offline scorer records the ladder knobs ---------------------------------

_METRICS_PATH = Path(__file__).resolve().parents[2] / "eval" / "metrics.py"
_spec = importlib.util.spec_from_file_location("_eval_metrics_linear", _METRICS_PATH)
_metrics = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _metrics
_spec.loader.exec_module(_metrics)


class _BareEngine:
    """Bare 3-net board (no copper) for ``compute_metrics``."""

    def get_reward_snapshot(self, run_drc: bool) -> RewardSnapshot:
        return RewardSnapshot(
            unrouted_count=7, connected_net_count=0, target_net_count=3,
            unconnected_net_codes=frozenset(PAD_GROUPS),
        )

    def get_board_meta(self) -> BoardMeta:
        return BoardMeta(net_count=3, bbox_w=40.0, bbox_h=20.0)

    def get_pad_groups(self):
        return dict(PAD_GROUPS)

    def get_net_names(self):
        return dict(NET_NAMES)

    def get_routable_nets(self):
        return ROUTABLE

    def get_tracks(self):
        return []

    def get_vias(self):
        return []

    def get_ratsnest(self):
        return []

    def get_pads(self):
        return []

    def get_drc_violations(self):
        return []

    def is_routing(self):
        return False

    def get_track_count(self):
        return 0

    def get_via_count(self):
        return 0

    def build_connectivity(self):
        pass


def test_phi_weights_record_linear_scale_and_flat_weights():
    m = _metrics.compute_metrics(
        _BareEngine(), u_0=None, reward_config_name="pcbworld_reward",
        check_angle=45,
    )
    w = m["phi_weights"]
    assert w["net_bonus_size_linear_scale"] == 0.0
    assert w["net_bonus_size_log_scale"] == 0.0
    assert w["net_size_weights"] is None          # flat per-net ladder
    assert w["completion_bonus"] == pytest.approx(0.5 * TOTAL_PG)
