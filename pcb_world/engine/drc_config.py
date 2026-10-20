"""Default DRC config loader and applier.

Loads a YAML file describing global DRC minima and applies it to a
``KiCadEngine`` **only when** the engine indicates the board had no
authoritative design rules — i.e. neither a companion ``.kicad_pro`` nor
legacy (pre-KiCad 6) ``(setup ...)`` tokens.

This keeps boards that already carry real rules untouched while giving
bare ``.kicad_pcb`` files from the training dataset a sensible baseline
for DRC evaluation.
"""

from __future__ import annotations

import math
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pcb_world.engine.kicad_engine import KiCadEngine


_REPO_ROOT = Path(__file__).resolve().parents[2]
# Canonical shipped baseline. NOT an implicit fallback: callers that want
# it must pass it explicitly (``drc_config_path`` / ``--drc-config-path``).
# ``apply_default_drc_if_fallback`` never resolves to this on its own.
DEFAULT_DRC_CONFIG_PATH = _REPO_ROOT / "configs" / "drc" / "pcbworld_drc.yaml"


# YAML key → DesignRules field name. Kept explicit (rather than auto-
# prefixing with ``min_`` / suffixing with ``_mm``) so the YAML file stays
# readable and the mapping is easy to audit.
_FIELD_MAP: dict[str, str] = {
    "clearance":             "min_clearance_mm",
    "track_width":           "min_track_width_mm",
    "via_diameter":          "min_via_diameter_mm",
    "through_hole_drill":    "min_through_hole_mm",
    "via_annular_width":     "min_via_annular_width_mm",
    "hole_to_hole":          "min_hole_to_hole_mm",
    "uvia_diameter":         "min_uvia_diameter_mm",
    "uvia_drill":            "min_uvia_drill_mm",
    "copper_edge_clearance": "copper_edge_clearance_mm",
}


@dataclass(frozen=True)
class DrcRuleSchema:
    """Schema of one DRC YAML: every ``global_minima`` key of ``_FIELD_MAP`` (mm) is
    required; ``name`` / ``description`` may document the file. Build it with :meth:`from_dict`."""

    global_minima: Mapping[str, float]

    @classmethod
    def from_dict(cls, raw: Any, source: str | Path) -> "DrcRuleSchema":
        """Validate a parsed DRC YAML; any unknown, missing or mistyped key raises ``ValueError`` naming *source*.

        A minimum may be a number or a string that parses as one (YAML 1.1 reads ``1e-3``
        as a string); it is stored as ``float``.
        """
        from configs.loader import check_mapping

        where = f"DRC config {source}"
        top = check_mapping(raw, _SCHEMA_TYPES, required=("global_minima",), where=where)
        where = f"{where} global_minima"
        minima = {key: _parse_float(value, where, key) for key, value in top["global_minima"].items()}
        check_mapping(minima, dict.fromkeys(_FIELD_MAP, float), where=where)
        return cls(global_minima=MappingProxyType({key: float(value) for key, value in minima.items()}))


# Top-level keys: the schema field + documentation keys nothing reads.
_SCHEMA_TYPES = {**typing.get_type_hints(DrcRuleSchema), "name": str, "description": str}


def _parse_float(value: Any, where: str, key: str) -> Any:
    """Return *value* parsed as ``float`` when it is a string; a string that is not a number, or
    a non-finite number, raises ``ValueError`` naming *where* and *key*. Other types pass through to the schema check."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"{where}: key {key!r} must be a number, got {value!r}") from None
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: key {key!r} must be a finite number, got {value!r}")
    return value


def load_drc_config(path: str | Path) -> dict[str, float]:
    """Load a DRC YAML file (validated by :class:`DrcRuleSchema`) and return {DesignRules field → mm value}.

    ``path`` is required — there is no implicit default YAML. Callers that
    want the shipped baseline must pass ``DEFAULT_DRC_CONFIG_PATH``
    (``configs/drc/pcbworld_drc.yaml``) explicitly.
    """
    from configs.loader import load_config

    if path is None:
        raise ValueError("load_drc_config requires an explicit YAML path")
    rule = DrcRuleSchema.from_dict(load_config(path), path)
    return {_FIELD_MAP[key]: value for key, value in rule.global_minima.items()}


# Output field → (BDS global minimum field, NetClassInfo field).
# The global minimum is the DRC-enforced floor across every net; the
# netclass field is the per-class default used when routing that class.
# The effective "hardest" rule is the MAX of both, so both must be
# considered.
_HARDEST_FIELD_MAP: dict[str, tuple[str, str]] = {
    "clearance_mm":     ("min_clearance_mm",     "clearance_mm"),
    "track_width_mm":   ("min_track_width_mm",   "track_width_mm"),
    "via_diameter_mm":  ("min_via_diameter_mm",  "via_diameter_mm"),
    # BDS stores the via drill minimum as "through hole" (name carries
    # over from pre-KiCad 6). Netclass side is just "via_drill".
    "via_drill_mm":     ("min_through_hole_mm",  "via_drill_mm"),
    "uvia_diameter_mm": ("min_uvia_diameter_mm", "uvia_diameter_mm"),
    "uvia_drill_mm":    ("min_uvia_drill_mm",    "uvia_drill_mm"),
}


def compute_hardest_per_netclass(rules) -> dict[str, float]:
    """For each design-rule field, return the strictest effective value
    across the board.

    DRC enforces ``max(BDS_global_min, netclass_default)`` for every net,
    so the "hardest" constraint any route must satisfy is:

        max(global_min, max over all netclasses of (class field))

    Both the BDS global minimum and every netclass (Default + non-default)
    are folded into the same MAX. Values unset in KiCad (``std::optional``
    without a value, exposed as ``-1.0``) are skipped — unset means
    "inherit / no constraint", not "zero".

    In practice:
      - Basic fields (clearance / track_width / via_*) are virtually
        always set by either BDS or the Default netclass, so the returned
        value is > 0.
      - uvia fields stay at ``-1.0`` on boards that don't use microvias;
        callers should treat ``-1.0`` as "no applicable constraint" and
        omit it from prompts / logs rather than printing a -1.

    This is a pure computation: the result is **not** pushed back into
    the engine or BDS. Callers hold the returned dict themselves.
    """
    out: dict[str, float] = {f: -1.0 for f in _HARDEST_FIELD_MAP}

    for out_field, (global_field, nc_field) in _HARDEST_FIELD_MAP.items():
        best = -1.0

        # BDS global minimum — DRC-enforced floor for every net.
        global_val = float(getattr(rules, global_field, -1.0))
        if global_val >= 0:
            best = global_val

        # Per-netclass defaults (Default + overrides).
        for nc in [rules.default_netclass, *rules.netclasses]:
            v = float(getattr(nc, nc_field))
            if v < 0:
                continue
            if best < 0 or v > best:
                best = v

        out[out_field] = best

    return out


def apply_default_drc_if_fallback(
    engine: "KiCadEngine",
    config_path: str | Path | None,
    *,
    rules_source: str,
) -> dict[str, Any] | None:
    """Verify the board carries authoritative rules, or substitute the YAML's.

    No-op (returns None) when either ``was_project_loaded_from_file()``
    or ``was_legacy_design_settings_loaded()`` is True — i.e. BDS already
    reflects real rules from a companion ``.kicad_pro`` or legacy
    ``(setup ...)`` tokens.

    When **both** are False the board has no authoritative rules. Every
    dataset we ship should carry one or the other (synthetic v2 emits a
    sibling ``.kicad_pro``; real boards have legacy setup), so reaching
    this path almost always signals a load failure. ``rules_source``
    (``EnvConfig.rules_source``) decides:
      - ``"board"``: raise ``ValueError``. Forces callers to confirm a
        misloaded board rather than silently route on KiCad compile-time
        defaults.
      - ``"board_or_yaml"``: substitute the YAML at ``config_path``. This
        is **required** — there is no implicit default YAML, so a
        ``None`` ``config_path`` here also raises ``ValueError``. Its
        global minima are pushed into BDS — and every Default net class
        size below its minimum is raised to it — with a ``UserWarning``
        naming the file and every raised size.

    Returns the substitution record when the engine's rules were changed —
    ``{"drc_config_path", "minima", "default_class_raised": {field: [old, new]}}``
    (sizes in mm) — else None.
    """
    from configs.loader.schema import RULES_SOURCES

    if rules_source not in RULES_SOURCES:
        raise ValueError(f"rules_source must be one of {RULES_SOURCES}, got {rules_source!r}")
    if engine.was_project_loaded_from_file():
        return None
    if engine.was_legacy_design_settings_loaded():
        return None

    msg = (
        "DRC fallback triggered: board has no companion .kicad_pro and "
        "no legacy (setup ...) tokens. Every dataset we ship should "
        "have one or the other; check the board source if this is "
        "unexpected."
    )
    if rules_source == "board":
        raise ValueError(
            msg + " rules_source is 'board'; pass rules_source='board_or_yaml' (or "
            "--rules-source board_or_yaml) together with an explicit "
            "drc_config_path to opt in to a YAML substitute (e.g. "
            "--drc-config-path configs/drc/pcbworld_drc.yaml)."
        )

    if config_path is None:
        raise ValueError(
            msg + " rules_source is 'board_or_yaml' but no drc_config_path was "
            "given, and there is no implicit default YAML. Pass an explicit "
            "path (e.g. --drc-config-path configs/drc/pcbworld_drc.yaml)."
        )

    overrides = load_drc_config(config_path)
    rules = engine.get_design_rules()
    for field, value in overrides.items():
        setattr(rules, field, value)
    # The board has no rules of its own, so its Default class is KiCad's built-in
    # one (0.2 / 0.2 / 0.6 / 0.3 mm). Raise each size to the substituted minimum:
    # the router, the observation and get_design_rules() then all hold the same
    # Default value, and verify_netclass_rules passes without an exemption.
    default = rules.default_netclass
    raised: dict[str, list[float]] = {}
    for nc_attr, floor_attr, _pro_key in NETCLASS_RULE_FIELDS:
        floor = getattr(rules, floor_attr)
        if floor >= 0 and getattr(default, nc_attr) < floor:
            raised[nc_attr] = [getattr(default, nc_attr), floor]
            setattr(default, nc_attr, floor)
    engine.set_design_rules(rules)

    import warnings
    warnings.warn(
        f"{msg} Substituted DRC minima from {config_path}; Default net class sizes "
        f"raised to them: {raised or 'none'}.",
        UserWarning, stacklevel=2,
    )
    return {"drc_config_path": str(config_path), "minima": overrides,
            "default_class_raised": raised}


def design_rules_digest(rules: Any) -> str:
    """16-hex sha256 of a ``DesignRules`` snapshot in its wire form (``to_wire``): the
    board minima, the size presets and every net class's sizes, Default included —
    equal digests mean the router sized and spaced routes from the same values.

    Not covered (``DesignRules`` does not carry them): which net belongs to which
    class (net-class patterns / assignments) and custom ``.kicad_dru`` rules."""
    import hashlib
    import json

    from pcb_world.engine.wire import to_wire

    blob = json.dumps(to_wire(rules), separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# (net class attr, board-minimum attr on RLDesignRules, key under
# board.design_settings.rules in the .kicad_pro) of every size the router takes
# from the net class at start_route.
NETCLASS_RULE_FIELDS = (
    ("track_width_mm", "min_track_width_mm", "min_track_width"),
    ("clearance_mm", "min_clearance_mm", "min_clearance"),
    ("via_diameter_mm", "min_via_diameter_mm", "min_via_diameter"),
    ("via_drill_mm", "min_through_hole_mm", "min_through_hole_diameter"),
)


def verify_netclass_rules(engine: "KiCadEngine", rules: Any) -> None:
    """Fail on a board whose net class values the router would not use as written.

    ``rules`` is ``engine.get_design_rules()`` (the caller's one fetch). Checked are the
    four sizes a route takes from its class (:data:`NETCLASS_RULE_FIELDS`: track width,
    clearance, via diameter, via drill) — not the microvia sizes, which the router never
    uses. ``set_design_rules`` (C++) refuses a 0 in the same four Default class sizes.

    The router sizes every route from the start net's class, floored at the board
    minima (pcbnew's ``ImportSizes``). The floor must never be what decides a size
    here, so these cases stop the load with ``ValueError``:

    * a Default class size that is unset or 0 (``set_design_rules`` refuses a 0
      Default size the same way);
    * a net class value below the board minimum of the same quantity — the
      project contradicts itself, and either value silently wins something;
    * a project file that declares no board minimum for a quantity — KiCad then
      substitutes its compile-time default (0.5 mm via, 0.3 mm drill), which may
      be larger than the net class value.

    A net class field left unset inherits the Default class (KiCad's own
    resolution) — reported with a ``UserWarning``, not an error. Boards without
    a project file (legacy ``(setup ...)`` rules) skip the second check.
    """
    import json
    import warnings

    default = rules.default_netclass
    problems: list[str] = []
    declared: dict[str, Any] | None = None
    if engine.was_project_loaded_from_file():
        pro = json.loads(Path(engine.get_project_path()).read_text())
        declared = ((pro.get("board") or {}).get("design_settings") or {}).get("rules") or {}
    for nc in [default, *rules.netclasses]:
        for attr, floor_attr, pro_key in NETCLASS_RULE_FIELDS:
            v = getattr(nc, attr)
            if v < 0:
                if nc is default:
                    problems.append(f"Default net class leaves {attr} unset")
                else:
                    warnings.warn(
                        f"net class {nc.name!r} leaves {attr} unset — inherits the "
                        f"Default class value {getattr(default, attr)} "
                        f"({engine.get_project_path()})", UserWarning, stacklevel=2)
                continue
            if v == 0 and nc is default:   # set_design_rules refuses it the same way
                problems.append(f"Default net class {attr} is 0")
                continue
            floor = getattr(rules, floor_attr)
            if floor > 0 and v < floor - 1e-9:
                problems.append(f"net class {nc.name!r} {attr}={v} is below the board minimum {floor_attr}={floor}")
    if declared is not None:
        for _attr, _floor_attr, pro_key in NETCLASS_RULE_FIELDS:
            if pro_key not in declared:
                problems.append(f"project file declares no board minimum {pro_key!r} (KiCad would use its built-in default)")
    if problems:
        raise ValueError(
            "inconsistent design rules — refusing to route with silently adjusted sizes "
            f"({engine.get_project_path()}): " + "; ".join(problems))
