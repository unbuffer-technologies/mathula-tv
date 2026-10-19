"""The 'natural speed' episode policy: no speed-up exists, so an overrun is a failure."""
from __future__ import annotations

from mathula_tv.dub_workbench import Workbench
from mathula_tv.episode_scoring import _score_group

_BASE = {"group_id": "u", "source_text": "x", "spoken_text": "y", "start_ms": 0, "end_ms": 1000,
         "measured_ms": 1100, "direction": "compress", "required_speed_percent": 5.0, "variant_id": "agent"}


def _score(policy: str, fits: bool | None):
    return _score_group({**_BASE, "fits_naturally": fits}, known_tokens=set(), labels=[], rush_flag_pct=10.0,
                        speed_policy=policy)


def test_natural_policy_flags_any_unit_that_does_not_fit_without_speed_up():
    assert "overrun" in _score("natural", False)["flags"]
    assert "overrun" in _score("natural", None)["flags"]      # pool records carry no fits flag: fall back to speed need
    assert "overrun" not in _score("natural", True)["flags"]


def test_pipeline_policy_still_tolerates_the_renderers_bounded_speed_up():
    flags = _score("pipeline", False)["flags"]
    assert "rushed" not in flags and "overrun" not in flags   # 5% is under the 10% ceiling


def test_workbench_shows_overrun_not_required_speed_under_the_natural_policy():
    wb = Workbench(store=None, tts_factory=lambda: None)
    wb.episode = {"speed_policy": "natural"}
    shown = wb._present({"required_speed_percent": 7.5, "speed_fit_mode": "x", "rank": [1], "audio_path": "p",
                         "measured_ms": 9})
    assert shown == {"overrun_pct": 7.5, "measured_ms": 9}
    wb.episode = {"speed_policy": "pipeline"}
    assert "required_speed_percent" in wb._present({"required_speed_percent": 7.5, "audio_path": "p"})
