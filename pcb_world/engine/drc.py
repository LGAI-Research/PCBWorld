from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


# Canonical ordered DRC violation-keyword taxonomy — the single owner of the
# keyword *sequence* (order + membership). Consumed here as error_type → type_id
# (each keyword's position IS its id) and by pcb_world/core/reward.py, which derives
# its keyword→weight-category map from this list so the two can't drift.
# Substring match on lowercased error_type.
VIOLATION_KEYWORDS: Tuple[str, ...] = (
    "clearance",
    "track width",
    "unconnected",
    "short",
    "via",
    "copper edge",
)
_VIOLATION_KEYWORD_TO_ID: List[Tuple[str, int]] = [
    (kw, i) for i, kw in enumerate(VIOLATION_KEYWORDS)
]
# DEFAULT bucket (== len) catches anything that doesn't match; +1 for it.
DRC_TYPE_ID_DEFAULT = len(VIOLATION_KEYWORDS)      # 6
DRC_NUM_TYPE_BUCKETS = DRC_TYPE_ID_DEFAULT + 1     # 7

# Canonical KiCad RPT_SEVERITY_* codes (import-safe — the C++ router is imported
# lazily). Consumed directly by the codec (models/v1/tokenizer.py imports it);
# eval/metrics.py keeps a local mirror instead so its import stays cheap (that
# module keeps KiCad imports function-local). The mirror is pinned to this by
# tests/test_constant_consistency.py.
DRC_SEVERITY_ERROR = 0x20
DRC_SEVERITY_WARNING = 0x10

# Phantom per-net key grouping violations with no net attribution (see
# get_violation_counts_by_net docstring). Consumers that count "dirty nets"
# must exclude it; board-level zero-DRV checks must not.
ORPHAN_NET_KEY = "<orphan>"

# KiCad PCB_DRC_CODE enum values for the warning types we optionally promote
# into the reward/state DRC set. Source of truth:
# engine/kicad-python/kicad/pcbnew/drc/drc_item.h (DRCE_FIRST == 1, ordered).
# If KiCad enum ordering ever changes these must be re-synced.
DRCE_DANGLING_VIA = 12
DRCE_DANGLING_TRACK = 13
DRCE_NET_CONFLICT = 37
# Warnings that are effectively router-induced critical issues and should be
# counted alongside errors in the ``errors_and_promoted`` mode.
_PROMOTED_ERROR_CODES = frozenset({
    DRCE_DANGLING_VIA,
    DRCE_DANGLING_TRACK,
    DRCE_NET_CONFLICT,
})

# Single source of truth for "which DRC violations count" across reward and
# state. Selected once in the reward yaml (``drc_severity_mode``) and then
# threaded into both ``PotentialReward._drc_penalty`` and
# ``PCBWorld._get_obs`` so state tokens mirror the reward's view.
DRC_SEVERITY_MODE_ERRORS_ONLY = "errors_only"
DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED = "errors_and_promoted"
DRC_SEVERITY_MODE_ERRORS_AND_WARNINGS = "errors_and_warnings"
DRC_SEVERITY_MODES = (
    DRC_SEVERITY_MODE_ERRORS_ONLY,
    DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED,
    DRC_SEVERITY_MODE_ERRORS_AND_WARNINGS,
)


def violation_matches_severity_mode(v: Any, mode: str) -> bool:
    """Return True if violation *v* should be counted under *mode*.

    - ``errors_only``: severity == ERROR.
    - ``errors_and_promoted``: ERROR, plus router-induced warnings promoted to
      effective-errors: ``track_dangling``, ``via_dangling``, ``net_conflict``
      (regardless of their stock KiCad severity).
    - ``errors_and_warnings``: any violation (ERROR or WARNING).
    """
    sev = int(v.severity)
    if mode == DRC_SEVERITY_MODE_ERRORS_AND_WARNINGS:
        return True
    if sev == DRC_SEVERITY_ERROR:
        return True
    if mode == DRC_SEVERITY_MODE_ERRORS_AND_PROMOTED:
        return int(getattr(v, "error_code", 0)) in _PROMOTED_ERROR_CODES
    return False


def classify_violation_type_id(error_type: str) -> int:
    """Map KiCad error_type string to taxonomy id 0..6.

    Substring match on lowercased text; falls back to DEFAULT bucket.
    Note: 'copper edge' is checked before 'clearance' so that
    'Copper edge clearance' maps to bucket 5 rather than 0.
    """
    s = (error_type or "").lower()
    if "copper edge" in s:
        return 5
    for needle, tid in _VIOLATION_KEYWORD_TO_ID:
        if needle in s:
            return tid
    return DRC_TYPE_ID_DEFAULT


class DRCUtils:
    """Helper class that caches and post-processes DRC (Design Rule Check) results
    at the Python level.

    Design rationale:
    1. Independent Python-side caching: the C++ RLRouter's internal DRC caching
       mechanism is hard to inspect or extend from Python. This class holds DRC
       results in Python memory so they can be freely manipulated and reused
       without a heavy recomputation.
    2. Purpose-built helper methods: provides the various shapes of DRC data
       callers (e.g. api.py) need — grouping violations by net, filtering by
       severity, etc. — derived from the cached results held here.
    """

    def __init__(self) -> None:
        self._cached_violations: List[Any] = []
        self._cached_violations_dict: List[Any] = []
        # Net-subset (partial routing) DRC filter. When set, only violations
        # involving at least one target net (by net name) are cached — so
        # reward/state/per-net views ignore violations wholly among non-target
        # nets. None = keep every violation (legacy). The C++ m_drcViolations
        # stays complete regardless (incremental-DRC invariant): this filters
        # only the Python cache.
        self._target_net_names: Optional[frozenset] = None

    def set_target_net_names(self, names) -> None:
        """Restrict the cache to violations touching one of ``names`` (net-name
        strings). ``None`` disables the filter (whole-board, legacy).

        Rule (D3): keep a violation iff its ``net_names`` intersects the target
        set. Orphan violations (empty ``net_names`` — e.g. a board-edge item
        with no net) touch no target net and are therefore dropped under the
        filter. Applies from the next :meth:`update`.
        """
        self._target_net_names = (
            frozenset(str(n) for n in names) if names is not None else None
        )

    def _keep(self, v: Any) -> bool:
        """Filter predicate: True unless a net-subset filter excludes ``v``."""
        if self._target_net_names is None:
            return True
        return any(n in self._target_net_names for n in v.net_names)

    def update(self, violations: List[Any]) -> None:
        """Refresh the cache with the latest DRC results from the C++ router.

        Args:
            violations: list of DRCViolation objects received via pybind11.

        Under a net-subset filter (:meth:`set_target_net_names`) violations that
        involve no target net are dropped here — every downstream getter reads
        the filtered cache, so reward/state/per-net views stay consistent.
        """
        if self._target_net_names is not None:
            violations = [v for v in violations if self._keep(v)]
        self._cached_violations = violations
        self._cached_violations_dict = [self.drc_violation_to_dict(v) for v in violations]

    def clear(self) -> None:
        """Reset the cached DRC results."""
        self._cached_violations = []
        self._cached_violations_dict = []

    def get_violation_count(self) -> int:
        """Return the total number of cached violations."""
        return len(self._cached_violations)

    def get_violations(self) -> List[Any]:
        """Return the list of cached per-violation detail objects."""
        return self._cached_violations

    def get_violations_by_net(self) -> Dict[str, List[str]]:
        """Group the cached violations by net and list their error types.

        Returns:
            Dict[str, List[str]]: net name -> list of error types seen on it.
            e.g. {"NET1": ["clearance", "track width"], "NET_OBSTACLE": ["clearance"]}
        """
        result_set: Dict[str, set] = {}
        for v in self._cached_violations:
            for net_name in v.net_names:
                if net_name not in result_set:
                    result_set[net_name] = set()
                result_set[net_name].add(v.error_type)

        return {net: list(errors) for net, errors in result_set.items()}

    def get_violation_counts_by_net(self) -> Dict[str, int]:
        """Aggregate cached violation counts per net.

        A violation touching multiple nets (e.g. clearance between A and B)
        adds +1 to every net it touches, so the sum Σx_i can exceed the total
        violation count.

        Orphan violations (empty net_names — not attached to any net) are all
        pooled under the single phantom net `"<orphan>"`; the aggregate view
        sees it as one net, while its count is the total number of orphan
        violations.

        Returns:
            Dict[str, int]: net name -> number of violations touching it.
            The phantom key `"<orphan>"` holds the total of violations with no net.
        """
        counts: Dict[str, int] = {}
        for v in self._cached_violations:
            if not v.net_names:
                counts[ORPHAN_NET_KEY] = counts.get(ORPHAN_NET_KEY, 0) + 1
                continue
            for net_name in v.net_names:
                counts[net_name] = counts.get(net_name, 0) + 1
        return counts

    def get_sorted(
        self,
        head_xy: Optional[Tuple[float, float]] = None,
        k: int = 32,
        severity_mode: Optional[str] = None,
    ) -> List[dict]:
        """Return violations sorted by (severity desc, distance to head asc), cap to k.

        Output dicts carry the fields consumed by the state tokenizer:
        x_mm, y_mm, layer, error_type, type_id, severity, net_names.

        If *severity_mode* is given it must be one of ``DRC_SEVERITY_MODES``
        and violations that would not be counted by that mode (see
        ``violation_matches_severity_mode``) are dropped before sorting so
        state DRC tokens mirror the reward's view exactly.
        """
        hx, hy = (head_xy if head_xy is not None else (0.0, 0.0))
        has_head = head_xy is not None

        def _dist2(v: Any) -> float:
            if not has_head:
                return 0.0
            dx = float(v.x_mm) - hx
            dy = float(v.y_mm) - hy
            return dx * dx + dy * dy

        source = self._cached_violations
        if severity_mode is not None:
            source = [
                v for v in source
                if violation_matches_severity_mode(v, severity_mode)
            ]

        items = [
            {
                "x_mm": float(v.x_mm),
                "y_mm": float(v.y_mm),
                "layer": int(v.layer),
                "error_type": v.error_type,
                "type_id": classify_violation_type_id(v.error_type),
                "severity": int(v.severity),
                "net_names": list(v.net_names),
                "_d2": _dist2(v),
            }
            for v in source
        ]
        items.sort(key=lambda d: (-d["severity"], d["_d2"]))
        trimmed = items[: max(0, int(k))]
        for d in trimmed:
            d.pop("_d2", None)
        return trimmed

    def get_error_count(self) -> int:
        """Error-severity (0x20) violation count only."""
        return sum(
            1 for v in self._cached_violations
            if int(v.severity) == DRC_SEVERITY_ERROR
        )

    def get_warning_count(self) -> int:
        """Warning-severity (0x10) violation count only."""
        return sum(
            1 for v in self._cached_violations
            if int(v.severity) == DRC_SEVERITY_WARNING
        )

    def get_error_counts_by_net(self) -> Dict[str, int]:
        """Like ``get_violation_counts_by_net`` but errors only."""
        counts: Dict[str, int] = {}
        for v in self._cached_violations:
            if int(v.severity) != DRC_SEVERITY_ERROR:
                continue
            if not v.net_names:
                counts[ORPHAN_NET_KEY] = counts.get(ORPHAN_NET_KEY, 0) + 1
                continue
            for net_name in v.net_names:
                counts[net_name] = counts.get(net_name, 0) + 1
        return counts

    def get_warning_counts_by_net(self) -> Dict[str, int]:
        """Like ``get_violation_counts_by_net`` but warnings only."""
        counts: Dict[str, int] = {}
        for v in self._cached_violations:
            if int(v.severity) != DRC_SEVERITY_WARNING:
                continue
            if not v.net_names:
                counts[ORPHAN_NET_KEY] = counts.get(ORPHAN_NET_KEY, 0) + 1
                continue
            for net_name in v.net_names:
                counts[net_name] = counts.get(net_name, 0) + 1
        return counts

    def get_count_by_severity_mode(self, mode: str) -> int:
        """Violation count under a severity mode (see ``DRC_SEVERITY_MODES``).

        Shared by reward penalty and info logging — single source of truth so
        reward and state DRC-token views never diverge.
        """
        return sum(
            1 for v in self._cached_violations
            if violation_matches_severity_mode(v, mode)
        )

    def get_counts_by_net_by_severity_mode(self, mode: str) -> Dict[str, int]:
        """Per-net violation counts restricted to the severity mode."""
        counts: Dict[str, int] = {}
        for v in self._cached_violations:
            if not violation_matches_severity_mode(v, mode):
                continue
            if not v.net_names:
                counts[ORPHAN_NET_KEY] = counts.get(ORPHAN_NET_KEY, 0) + 1
                continue
            for net_name in v.net_names:
                counts[net_name] = counts.get(net_name, 0) + 1
        return counts

    def get_filtered_by_severity(self, severity_code: int) -> List[Any]:
        """Return only the violations matching a given severity code (e.g. ERROR=0x20, WARNING=0x10)."""
        return [v for v in self._cached_violations if v.severity == severity_code]

    @staticmethod
    def drc_violation_to_dict(v) -> dict:
        """Convert a DRCViolation object into a plain Python dict."""
        return {
            "error_code": v.error_code,
            "error_type": v.error_type,
            "message": v.message,
            "x_mm": v.x_mm,
            "y_mm": v.y_mm,
            "layer": v.layer,
            "net_names": v.net_names,  # list[str]
            "severity": v.severity
        }

