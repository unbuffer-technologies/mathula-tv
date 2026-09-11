"""Regression test for a real policy change: undersized ("expand" direction)
translations no longer trigger Grok wording repair or a hard pipeline failure.
Mathula now accepts a naturally shorter rendering and plays it at its own
natural speed, exactly symmetric with how "compress" (overlong) has always been
handled -- a warning, never a crash or a forced rewrite. The removed rhetorical-
expansion repair mechanism (tautology/periphrasis/macrology techniques) was the
exact root cause of a real production bug where an island's expansion repair
restated most of a sibling island's content to hit a syllable floor.
"""
from pathlib import Path

from mathula_tv.native_dub import (
    MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS,
    _timing_recast_direction,
)

_SOURCE = Path(__file__).parents[1] / "src" / "mathula_tv" / "native_dub.py"


def test_removed_repair_machinery_is_not_present_in_source():
    text = _SOURCE.read_text(encoding="utf-8")
    assert "Rhetorical expansion rescue" not in text
    assert "DEFAULT_RHETORICAL_EXPANSION_RESCUE_ROUNDS" not in text
    assert "will not accept materially short isiZulu by leaving dead air" not in text
    assert "TIMING_REPAIR_SYSTEM_PROMPT" not in text
    assert "_contextual_expansion_plan" not in text
    assert "_rhetorical_strategy_plan" not in text


def test_expand_direction_still_classified_but_no_longer_repaired():
    # _timing_recast_direction itself is unchanged -- it's still the classifier
    # used for logging/diagnostics and for deciding whether the cheap trailing-
    # silence pad applies to a tiny residual. What changed is what happens AFTER
    # classification: nothing in the source calls a Grok repair provider for it
    # anymore (confirmed by the source-text checks above).
    direction = _timing_recast_direction(
        700, 1000, max_speed_percent=12, mouth_close_early_tolerance_ms=40, mouth_close_late_tolerance_ms=0,
    )
    assert direction == "expand"


def test_tiny_residual_pad_threshold_still_exists_for_the_cheap_silence_case():
    # A genuinely tiny gap (a fraction of a syllable) still gets closed with real
    # trailing silence rather than left as an abrupt cut -- that mechanism is
    # local/cheap (no Grok call) and is unrelated to the removed repair policy.
    assert MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS > 0
