"""The LLM eval's deviation report names what set each deviating env key; a key with no
provenance is a resolver bug and raises (no ``?`` placeholder)."""
from __future__ import annotations

import pytest


def test_deviation_names_the_cli_as_its_source(capsys):
    from eval.args import resolve_eval_args
    resolve_eval_args(["--masking_rule", "strict"])
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "masking_rule" in ln)
    assert "DEVIATION" in line and "?" not in line


def test_a_key_without_provenance_raises(monkeypatch):
    import eval.args as eval_args
    monkeypatch.setattr(eval_args, "report_env_deviations",
                        lambda env, source_of: source_of("not_a_resolved_key"))
    with pytest.raises(KeyError):
        eval_args.resolve_eval_args([])
