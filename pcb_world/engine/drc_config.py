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

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pcb_world.engine.kicad_engine import KiCadEngine


_REPO_ROOT = Path(__file__).resolve().parents[2]
# Canonical shipped baseline. NOT an implicit fallback: callers that want
# it must pass it explicitly (``drc_config_path`` / ``--drc-config-path``).
# ``apply_default_drc_if_fallback`` never resolves to this on its own.
DEFAULT_DRC_CONFIG_PATH = _REPO_ROOT / "configs" / "drc" / "default.yaml"


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


def load_drc_config(path: str | Path) -> dict[str, float]:
    """Load a DRC YAML file and return {DesignRules field → mm value}.

    ``path`` is required — there is no implicit default YAML. Callers that
    want the shipped baseline must pass ``DEFAULT_DRC_CONFIG_PATH``
    (``configs/drc/default.yaml``) explicitly.

    Unknown keys in the YAML are silently ignored so the file format can
    grow backwards-compatibly. Missing keys map to no override (negative
    sentinel ``-1.0`` so ``setDesignRules`` leaves the field untouched).
    """
    import yaml

    if path is None:
        raise ValueError("load_drc_config requires an explicit YAML path")
    resolved = Path(path)
    with open(resolved) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    minima: dict[str, Any] = raw.get("global_minima") or {}
    out: dict[str, float] = {field: -1.0 for field in _FIELD_MAP.values()}
    for yaml_key, field in _FIELD_MAP.items():
        if yaml_key in minima:
            out[field] = float(minima[yaml_key])
    return out


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
    config_path: str | Path | None = None,
    *,
    use_yaml: bool = True,
) -> bool:
    """Verify the board carries authoritative rules, or opt-in to YAML.

    No-op (returns False) when either ``was_project_loaded_from_file()``
    or ``was_legacy_design_settings_loaded()`` is True — i.e. BDS already
    reflects real rules from a companion ``.kicad_pro`` or legacy
    ``(setup ...)`` tokens.

    When **both** are False the board has no authoritative rules. Every
    dataset we ship should carry one or the other (synthetic v2 emits a
    sibling ``.kicad_pro``; real boards have legacy setup), so reaching
    this path almost always signals a load failure. Behaviour:
      - ``use_yaml=False`` (default at the env entry point): raise
        ``ValueError``. Forces callers to confirm a misloaded board
        rather than silently route on KiCad compile-time defaults.
      - ``use_yaml=True``: substitute the YAML at ``config_path``. This
        is **required** — there is no implicit default YAML, so a
        ``None`` ``config_path`` here also raises ``ValueError``. When a
        path is given, its global minima are pushed into BDS with a
        ``UserWarning`` so the trace is visible; opt-in for legacy
        datasets that genuinely need the substitution.

    Returns True iff the engine's BDS was mutated.
    """
    if engine.was_project_loaded_from_file():
        return False
    if engine.was_legacy_design_settings_loaded():
        return False

    msg = (
        "DRC fallback triggered: board has no companion .kicad_pro and "
        "no legacy (setup ...) tokens. Every dataset we ship should "
        "have one or the other; check the board source if this is "
        "unexpected."
    )
    if not use_yaml:
        raise ValueError(
            msg + " Pass use_yaml_drc_fallback=True (or "
            "--use-yaml-drc-fallback) together with an explicit "
            "drc_config_path to opt in to a YAML substitute (e.g. "
            "--drc-config-path configs/drc/default.yaml)."
        )

    if config_path is None:
        raise ValueError(
            msg + " use_yaml_drc_fallback is on but no drc_config_path was "
            "given, and there is no implicit default YAML. Pass an explicit "
            "path (e.g. --drc-config-path configs/drc/default.yaml)."
        )

    import warnings
    warnings.warn(
        f"{msg} Substituting DRC from {config_path}.",
        UserWarning, stacklevel=2,
    )
    overrides = load_drc_config(config_path)
    rules = engine.get_design_rules()
    for field, value in overrides.items():
        setattr(rules, field, value)
    engine.set_design_rules(rules)
    return True
