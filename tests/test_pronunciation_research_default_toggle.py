from __future__ import annotations

from mathula_tv.cli import parser


def test_skip_pronunciation_research_defaults_off() -> None:
    parsed = parser().parse_args(["native-translate", "job-123", "--candidate-pool"])
    assert parsed.skip_pronunciation_research is False


def test_skip_pronunciation_research_env_toggle_flips_the_default(monkeypatch) -> None:
    # Real scenario, 2026-09-10: web-research pronunciation lookups run on a
    # separate Azure deployment/quota from the main translation model, and
    # that deployment ran out of tokens -- a deterministic, zero-cost disable
    # (rather than letting every call fail and gracefully degrade on its own)
    # needs to be settable once, systemically, not remembered as a CLI flag
    # on every native-translate invocation.
    monkeypatch.setenv("MATHULA_TV_SKIP_PRONUNCIATION_RESEARCH", "1")
    parsed = parser().parse_args(["native-translate", "job-123", "--candidate-pool"])
    assert parsed.skip_pronunciation_research is True


def test_explicit_flag_still_works_without_the_env_var() -> None:
    parsed = parser().parse_args(
        ["native-translate", "job-123", "--candidate-pool", "--skip-pronunciation-research"]
    )
    assert parsed.skip_pronunciation_research is True
