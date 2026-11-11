"""The two pip locks (Linux x86_64 `requirements.txt`, macOS arm64 `requirements-macos.txt`)."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _pins(lock: str) -> dict[str, str]:
    pins = {}
    for ln in (REPO / lock).read_text().splitlines():
        m = re.match(r"([A-Za-z0-9_.\-]+)==(\S+)", ln)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def test_both_locks_carry_the_rule_based_baseline_deps():
    # KRT's route.py imports shapely and scipy; on Linux scipy would otherwise arrive only via vllm.
    for lock in ("requirements.txt", "requirements-macos.txt"):
        assert {"shapely", "scipy"} <= _pins(lock).keys(), lock


def test_the_macos_lock_shares_every_linux_version():
    linux, mac = _pins("requirements.txt"), _pins("requirements-macos.txt")
    differ = {name: (linux[name], v) for name, v in mac.items() if name in linux and linux[name] != v}
    assert not differ


def _conda_specs(env_file: str) -> dict[str, str]:
    """`- name<op>version` conda specs of an env file (unversioned entries map to "")."""
    specs = {}
    for ln in (REPO / env_file).read_text().splitlines():
        m = re.match(r"  - ([A-Za-z0-9_.\-]+)\s*([<>=!].*?)?\s*(#.*)?$", ln)
        if m:
            specs[m.group(1)] = (m.group(2) or "").strip()
    return specs


def test_the_pcbnew_build_deps_stay_constrained():
    # unconstrained, conda-forge resolves OCCT 8, SWIG 4.5 and NNG 2, which KiCad 9.0.8 cannot build against
    specs = _conda_specs("environment.yml")
    assert all(specs[name] for name in ("occt", "swig", "nng")), specs


def test_the_two_env_files_pin_shared_packages_alike():
    linux, mac = _conda_specs("environment.yml"), _conda_specs("environment.macos.yml")
    differ = {n: (linux[n], mac[n]) for n in linux.keys() & mac.keys() if linux[n] != mac[n]}
    assert not differ
