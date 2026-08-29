"""Engine-IPC wire schema codec tests (stdlib-only — no router needed).

Round-trips every registered mirror type through ``to_wire`` → pickle →
``from_wire`` and checks the codec's structural rules (tuple tagging,
unknown-type fail-loud, registry field order).
"""

import pickle

import pytest

from pcb_world.engine.containers import (
    KRL_CONSTANT_NAMES,
    KRL_FIELDS,
    BoardEdge,
    BoardOutlineShape,
    BoundingBox,
    CleanupItem,
    CleanupResult,
    ClusterPoint,
    DesignRules,
    DRCViolation,
    FootprintInfo,
    GraphicShape,
    NetClassInfo,
    PadInfo,
    RatsnestEdge,
    TrackInfo,
    ViaInfo,
    ZoneInfo,
    from_wire,
    to_wire,
)

# One representative instance per registered wire type, with non-default
# values in every field so a swapped/missed field cannot round-trip clean.
SAMPLES = {
    "TrackInfo": TrackInfo(1.0, 2.0, 3.0, 4.0, 0.25, 0, 3, "GND", "aa-bb"),
    "ViaInfo": ViaInfo(1.5, 2.5, 0.8, 0.4, 0, 3, 7, "VCC", "cc-dd"),
    "PadInfo": PadInfo(0.1, 0.2, 1.0, 2.0, 0, 5, "NET5", "1", "U1",
                       "thru_hole", "circle"),
    "RatsnestEdge": RatsnestEdge(0.0, 1.0, 2.0, 3.0, 4, 1, 2),
    "ClusterPoint": ClusterPoint(9.0, 8.0, 1),
    "ZoneInfo": ZoneInfo([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], 0,
                         True, False, True, "keepout1"),
    "BoardEdge": BoardEdge(0.0, 0.0, 10.0, 0.0, 0.05),
    "BoardOutlineShape": BoardOutlineShape("arc", 0.0, 1.0, 2.0, 3.0,
                                           4.0, 5.0, 0.1),
    "GraphicShape": GraphicShape(3, "segment", 0, 1, 2, 3, 4, 5, 100000),
    "DRCViolation": DRCViolation(2, "clearance", "too close", 1.0, 2.0, 0,
                                 ["GND", "VCC"], 0x20, "uuid-a", "uuid-b"),
    "BoundingBox": BoundingBox(0.0, 0.0, 100.0, 80.0),
    "FootprintInfo": FootprintInfo(
        "U1", "LM358", "Lib:Pkg", 12.5, 30.0, 90.0, True, 0,
        [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], [(5.0, 5.0), (6.0, 5.0)]]),
    "CleanupItem": CleanupItem(1, "merge_tracks", "uuid-1", "uuid-2"),
    "NetClassInfo": NetClassInfo("Default", 0.2, 0.25, 0.8, 0.4, 0.3, 0.2),
    "DesignRules": DesignRules(
        0.2, 0.25, 0.8, 0.3, 0.15, 0.25, 0.3, 0.2, 0.5,
        [0.25, 0.5], [(0.8, 0.4)],
        NetClassInfo("Default", 0.2, 0.25, 0.8, 0.4, 0.3, 0.2),
        [NetClassInfo("Power", 0.3, 0.5, 1.0, 0.5, 0.4, 0.3)]),
    "CleanupResult": CleanupResult(
        True, "", [CleanupItem(1, "merge_tracks", "uuid-1", "uuid-2")],
        ["uuid-3"], ["uuid-4"]),
}


def test_samples_cover_registry():
    assert set(SAMPLES) == set(KRL_FIELDS)


@pytest.mark.parametrize("tname", sorted(SAMPLES))
def test_roundtrip_lossless(tname):
    obj = SAMPLES[tname]
    wire = to_wire(obj)
    # The wire form must survive pickling (the actual transport).
    back = from_wire(pickle.loads(pickle.dumps(wire)))
    assert type(back).__name__ == tname
    for f in KRL_FIELDS[tname]:
        orig, dec = getattr(obj, f), getattr(back, f)
        # Nested mirrors compare by fields; everything else by equality.
        if tname == "CleanupResult" and f == "items":
            assert [tuple(i) for i in dec] == [tuple(i) for i in orig]
        elif tname == "DesignRules" and f in ("default_netclass", "netclasses"):
            assert dec == orig  # dataclass __eq__
        else:
            assert dec == orig, f"{tname}.{f}: {dec!r} != {orig!r}"


def test_registry_field_order_and_purity():
    # NamedTuple registry entries = _fields; dataclass entries exclude
    # properties/methods (CleanupResult.changed / .counts must NOT be fields).
    assert KRL_FIELDS["CleanupResult"] == (
        "ran", "reject_reason", "items", "removed", "modified")
    assert KRL_FIELDS["CleanupItem"] == ("code", "code_name", "item_a", "item_b")
    assert KRL_FIELDS["FootprintInfo"] == (
        "ref", "value", "fpid", "x_mm", "y_mm", "orientation_deg",
        "flipped", "layer", "courtyard")
    for fields in KRL_FIELDS.values():
        assert all(isinstance(f, str) for f in fields)


def test_plain_tuple_does_not_alias_mirror_tag():
    # A plain tuple round-trips as a tuple, not as a mirror or bare list.
    v = (1, 2, (3, "x"))
    assert from_wire(to_wire(v)) == v
    assert isinstance(from_wire(to_wire(v)), tuple)


def test_containers_recursed():
    v = {"tracks": [SAMPLES["TrackInfo"]], "n": 3, "flags": (True, None)}
    back = from_wire(to_wire(v))
    assert back["n"] == 3
    assert back["flags"] == (True, None)
    assert isinstance(back["tracks"][0], TrackInfo)
    assert back["tracks"][0] == SAMPLES["TrackInfo"]


def test_unknown_type_raises():
    class NotRegistered:
        pass

    with pytest.raises(TypeError, match="KRL_FIELDS"):
        to_wire(NotRegistered())


def test_constants_list_is_names_only():
    assert all(isinstance(c, str) for c in KRL_CONSTANT_NAMES)
    assert len(set(KRL_CONSTANT_NAMES)) == len(KRL_CONSTANT_NAMES)
