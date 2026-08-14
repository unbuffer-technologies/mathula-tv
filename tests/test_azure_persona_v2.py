from __future__ import annotations

from mathula_tv.voice_prosody import voice_and_base_prosody
from mathula_tv.voice_resolver import (
    PERSONA_V2_VERSION,
    _spread_family_personas,
)


def assignment(
    speaker_id: str,
    *,
    family: str = "feminine",
    pitch: int = 0,
    rate: int = 0,
    f0: float | None = None,
    source_rate: float | None = None,
):
    features = {}
    if f0 is not None:
        features["median_f0_hz"] = f0
    if source_rate is not None:
        features["source_speech_rate_wps"] = source_rate
    return {
        "speaker_id": speaker_id,
        "anchor_voice_family": family,
        "selected_voice": (
            "zu-ZA-ThandoNeural"
            if family == "feminine"
            else "zu-ZA-ThembaNeural"
        ),
        "base_prosody": {
            "pitch_percent": pitch,
            "rate_percent": rate,
            "volume_percent": 0,
        },
        "source_features": features,
    }


def test_two_similar_thandos_are_audibly_spread():
    result = _spread_family_personas(
        [
            assignment("SPEAKER_A", f0=210.0, source_rate=2.2),
            assignment("SPEAKER_B", f0=212.0, source_rate=2.2),
        ]
    )
    by_id = {item["speaker_id"]: item for item in result}
    first = by_id["SPEAKER_A"]["base_prosody"]
    second = by_id["SPEAKER_B"]["base_prosody"]

    assert first["persona_version"] == PERSONA_V2_VERSION
    assert second["persona_version"] == PERSONA_V2_VERSION
    assert first["persona_slot"] != second["persona_slot"]
    assert abs(first["pitch_percent"] - second["pitch_percent"]) >= 8
    assert first["rate_percent"] != second["rate_percent"]


def test_measured_low_and_high_speakers_keep_natural_ordering():
    result = _spread_family_personas(
        [
            assignment("HIGH", family="masculine", f0=170.0, source_rate=2.8),
            assignment("LOW", family="masculine", f0=105.0, source_rate=1.8),
        ]
    )
    by_id = {item["speaker_id"]: item for item in result}
    assert (
        by_id["LOW"]["base_prosody"]["pitch_percent"]
        < by_id["HIGH"]["base_prosody"]["pitch_percent"]
    )


def test_one_speaker_is_not_artificially_shifted():
    result = _spread_family_personas(
        [assignment("ONLY", pitch=4, rate=-3, f0=200.0, source_rate=2.0)]
    )
    prosody = result[0]["base_prosody"]
    assert prosody["pitch_percent"] == 4
    assert prosody["rate_percent"] == -3
    assert prosody["separation_adjusted"] is False


def test_voice_prosody_accepts_wider_persona_v2_bounds():
    voice, rate, pitch, volume = voice_and_base_prosody(
        {
            "selected_voice": "zu-ZA-ThandoNeural",
            "base_prosody": {
                "rate_percent": 8,
                "pitch_percent": 11,
                "volume_percent": 2,
            },
        },
        "fallback",
    )
    assert voice == "zu-ZA-ThandoNeural"
    assert rate == 8
    assert pitch == 11
    assert volume == 2
