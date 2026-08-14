"""Focused tests for anchor-relative Azure speaker delivery."""

from mathula_tv.voice_prosody import (
    combined_azure_rate,
    voice_and_base_prosody,
)
from mathula_tv.azure_tts import AzureTTSRequest


def test_assignment_maps_to_safe_anchor_voice_and_prosody():
    assignment = {
        "selected_voice": "zu-ZA-ThembaNeural",
        "base_prosody": {
            "rate_percent": 4,
            "pitch_percent": -7,
            "volume_percent": 1,
        },
    }
    assert voice_and_base_prosody(assignment, "fallback") == (
        "zu-ZA-ThembaNeural",
        4,
        -7,
        1,
    )


def test_timing_rate_is_relative_to_speaker_base_and_bounded():
    assert combined_azure_rate(4, 3, minimum=-12, maximum=15) == 7
    assert combined_azure_rate(6, 15, minimum=-12, maximum=15) == 15
    assert combined_azure_rate(-6, -12, minimum=-12, maximum=15) == -12


def test_with_rate_preserves_speaker_pitch_and_volume():
    request = AzureTTSRequest(
        turn_id="unit_1",
        speaker_id="SPEAKER_00",
        text="Sawubona",
        preferred_duration_ms=1000,
        maximum_duration_ms=1500,
        voice="zu-ZA-ThembaNeural",
        rate_percent=-3,
        pitch_percent=5,
        volume_percent=1,
    )
    adjusted = request.with_rate(4)
    assert adjusted.rate_percent == 4
    assert adjusted.pitch_percent == 5
    assert adjusted.volume_percent == 1
    assert adjusted.voice == "zu-ZA-ThembaNeural"
