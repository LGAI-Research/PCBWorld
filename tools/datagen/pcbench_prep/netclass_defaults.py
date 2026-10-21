"""Net-class fields the KiCad 5 -> 9 round trip fills with KiCad defaults, replaced by the designer's values.

PCBench boards are KiCad 5 files whose ``(net_class ...)`` blocks mostly carry no ``diff_pair_width`` /
``diff_pair_gap``. Loading and saving them with KiCad 9 (``convert_v9.py``) writes KiCad's defaults
(0.2 mm / 0.25 mm) into the ``.kicad_pro``. Those values are not inert: KiCad's DRC engine builds
implicit net-class rules for nets it detects as a differential pair by name (``D+``/``D-``,
``X_P1``/``X_N1``), so the router's ``ImportSizes`` routes such a net at ``diff_pair_width`` instead
of the class track width, and a ``diff_pair_gap`` below the class clearance lowers the P<->N
clearance. A diff-pair field therefore follows its class — ``diff_pair_width`` = ``track_width``,
``diff_pair_gap`` = ``clearance`` — unless the designer set it to a value of its own, which is kept.

What counts as the designer's value: KiCad 5 writes the two diff-pair fields together, and only when
either differs from its default (``pcbnew/netclass.cpp`` ``NETCLASS::Format``, 5.1.12), so a written
value equal to the KiCad default may be one the designer never touched — it follows too, as does a
value equal to the class's own (a guide class ``make_guide.py`` derives at another width then moves
with it; ``make_guide.py`` applies the same rule through the field values).
"""
from __future__ import annotations

import math
import re

# .kicad_pro field -> (.kicad_pro field it follows, KiCad 5 net_class key, KiCad 5 key of the followed field)
UNSET_DIFF_PAIR_FIELDS = {
    "diff_pair_width": ("track_width", "diff_pair_width", "trace_width"),
    "diff_pair_gap": ("clearance", "diff_pair_gap", "clearance"),
}
# KiCad 5 DEFAULT_DIFF_PAIR_WIDTH / DEFAULT_DIFF_PAIR_GAP (mm)
KICAD5_DEFAULTS = {"diff_pair_width": 0.2, "diff_pair_gap": 0.25}
# KiCad 5 net_class keys make_guide.py carries into guide classes; the parser below reads LEGACY_KEYS
LEGACY_PRESERVED_KEYS = ("clearance", "trace_width", "via_dia", "via_drill", "uvia_dia", "uvia_drill")
LEGACY_KEYS = LEGACY_PRESERVED_KEYS + ("diff_pair_width", "diff_pair_gap")


def balanced_block_end(content: str, start: int) -> int:
    """Exclusive end index of the block from '(' to its matching ')'."""
    depth = 0
    for j in range(start, len(content)):
        if content[j] == "(":
            depth += 1
        elif content[j] == ")":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(content)


def parse_legacy_net_classes(content: str) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """The ``(net_class ...)`` blocks of a legacy (KiCad 5/6) board: ``({class: {key: value}}, {net: class})``
    with the KiCad 5 keys of ``LEGACY_KEYS`` that the block declares."""
    class_params: dict[str, dict[str, float]] = {}
    net_to_class: dict[str, str] = {}
    idx = 0
    while True:
        start = content.find("(net_class", idx)
        if start == -1:
            break
        end = balanced_block_end(content, start)
        block = content[start:end]
        m = re.match(r'\(net_class\s+("([^"]+)"|(\S+))', block)
        fields = re.sub(r'"(?:[^"\\]|\\.)*"', '""', block)   # values only: a description cannot shadow a field
        if m:
            cname = (m.group(2) or m.group(3)).strip('"')
            params = {}
            for key in LEGACY_KEYS:
                pm = re.search(rf"\({key}\s+([^()\s]+)\)", fields)
                if pm:
                    params[key] = finite_mm(pm.group(1), f"net_class {cname} {key}")
            class_params[cname] = params
            for mn in re.finditer(r'\(add_net\s+("([^"]+)"|([^)"]+))\)', block):
                net_to_class[(mn.group(2) or mn.group(3)).strip()] = cname
        idx = end
    return class_params, net_to_class


def finite_mm(text: str, what: str) -> float:
    """A finite non-negative size or ValueError naming ``what`` (such a value fails the board, never a silent skip)."""
    try:
        v = float(text)
    except ValueError:
        raise ValueError(f"{what}: {text!r} is not a number") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{what}: {text!r} is not a finite non-negative size")
    return v


def designer_set(legacy_class: dict[str, float], field: str) -> bool:
    """True when the legacy class gives ``field`` a value of its own: written, not the KiCad default
    (which KiCad 5 writes alongside a changed partner field) and not the class value it follows."""
    _, legacy_key, legacy_source = UNSET_DIFF_PAIR_FIELDS[field]
    if legacy_key not in legacy_class or legacy_class[legacy_key] == KICAD5_DEFAULTS[field]:
        return False
    return legacy_source not in legacy_class or legacy_class[legacy_key] != legacy_class[legacy_source]


def fill_unset_diff_pair(pro: dict, legacy: dict[str, dict[str, float]]) -> list[str]:
    """Make every net class's diff-pair field follow the class unless the legacy board set it to a value
    of its own; returns the ``<class>.<field>`` names it changed. A class with no legacy block of its own
    (the guide classes ``make_guide.py`` derives from Default) is judged by the legacy Default block."""
    ns = pro.get("net_settings") if isinstance(pro, dict) else None
    classes = ns.get("classes") if isinstance(ns, dict) else None
    if not isinstance(classes, list) or not all(isinstance(c, dict) and isinstance(c.get("name"), str) for c in classes):
        raise ValueError("the .kicad_pro has no net_settings.classes list of named objects")
    filled = []
    default = legacy.get("Default", {})
    for cls in classes:
        legacy_class = legacy.get(cls.get("name"), default)
        for field, (source, _, _) in UNSET_DIFF_PAIR_FIELDS.items():
            if designer_set(legacy_class, field) or cls.get(source) is None:
                continue
            if cls.get(field) != cls[source]:
                cls[field] = cls[source]
                filled.append(f"{cls.get('name')}.{field}")
    return filled
