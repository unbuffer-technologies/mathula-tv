"""The 'natural speed' episode policy: no speed-up exists, so any overrun is a failure."""
from __future__ import annotations

from mathula_tv.dub_workbench import Workbench
from mathula_tv.episode_scoring import _score_group

_BASE = {"group_id": "u", "source_text": "x", "spoken_text": "y", "start_ms": 0, "end_ms": 1000,
         "measured_ms": 1100, "direction": None, "required_speed_percent": 5.0, "variant_id": "agent"}


def _score(policy: str, speed_needed: float):
    # The pipeline's own `fit`/direction forgives up to ~10% speed-up: direction is None here on purpose.
    return _score_group({**_BASE, "required_speed_percent": speed_needed, "fits_naturally": True},
                        known_tokens=set(), labels=[], rush_flag_pct=10.0, speed_policy=policy)


def test_natural_policy_flags_any_unit_that_needs_a_speed_up_even_if_the_pipeline_forgives_it():
    assert "overrun" in _score("natural", 5.0)["flags"]
    assert "overrun" in _score("natural", 0.4)["flags"]
    assert "overrun" not in _score("natural", 0.0)["flags"]


def test_pipeline_policy_still_tolerates_the_renderers_bounded_speed_up():
    flags = _score("pipeline", 5.0)["flags"]
    assert "rushed" not in flags and "overrun" not in flags   # 5% is under the 10% ceiling
    assert "rushed" in _score("pipeline", 12.0)["flags"]


def test_workbench_shows_overrun_and_does_not_forgive_a_speed_up():
    wb = Workbench(store=None, tts_factory=lambda: None)
    wb.episode = {"speed_policy": "natural"}
    shown = wb._present({"required_speed_percent": 7.5, "speed_fit_mode": "x", "rank": [1], "audio_path": "p",
                         "measured_ms": 9, "direction": None, "fits_naturally": True})
    assert shown == {"overrun_pct": 7.5, "measured_ms": 9, "direction": None, "fits_naturally": False}
    ok = wb._present({"required_speed_percent": 0.0, "measured_ms": 9, "direction": None, "fits_naturally": True})
    assert ok["fits_naturally"] is True
    short = wb._present({"required_speed_percent": 0.0, "measured_ms": 9, "direction": "expand", "fits_naturally": False})
    assert short["fits_naturally"] is False
    wb.episode = {"speed_policy": "pipeline"}
    assert "required_speed_percent" in wb._present({"required_speed_percent": 7.5, "audio_path": "p"})
