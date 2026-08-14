from __future__ import annotations

from mathula_tv.direct_azure_persona import (
    DIRECT_AZURE_PERSONA_VERSION,
    apply_direct_azure_personas,
    collect_source_features,
)


def _assignment(
    speaker_id: str,
    *,
    family: str = "feminine",
    f0: float | None = None,
    rate: float | None = None,
    resolution_status: str = "mapped_voice_family",
):
    features = {}
    if f0 is not None:
        features["median_f0_hz"] = f0
    if rate is not None:
        features["source_speech_rate_wps"] = rate
    return {
        "speaker_id": speaker_id,
        "voice_family": family,
        "selected_voice": (
            "zu-ZA-ThandoNeural"
            if family == "feminine"
            else "zu-ZA-ThembaNeural"
        ),
        "resolution_status": resolution_status,
        "base_prosody": {
            "rate_percent": 0,
            "pitch_percent": 0,
            "volume_percent": 0,
        },
        "source_features": features,
    }


def test_two_similar_thandos_receive_audible_direct_personas():
    result = apply_direct_azure_personas(
        {
            "A": _assignment("A", f0=210.0, rate=2.2),
            "B": _assignment("B", f0=212.0, rate=2.2),
        }
    )
    first = result["A"]["base_prosody"]
    second = result["B"]["base_prosody"]
    assert first["persona_version"] == DIRECT_AZURE_PERSONA_VERSION
    assert first["persona_slot"] != second["persona_slot"]
    assert abs(first["pitch_percent"] - second["pitch_percent"]) >= 8
    assert first["rate_percent"] != second["rate_percent"]
    assert -10 <= first["pitch_percent"] <= 10
    assert -10 <= second["pitch_percent"] <= 10
    assert first["volume_percent"] == 0
    assert second["volume_percent"] == 0


def test_source_pitch_order_is_preserved():
    result = apply_direct_azure_personas(
        {
            "HIGH": _assignment("HIGH", family="masculine", f0=170.0, rate=2.8),
            "LOW": _assignment("LOW", family="masculine", f0=105.0, rate=1.8),
        }
    )
    assert (
        result["LOW"]["base_prosody"]["pitch_percent"]
        < result["HIGH"]["base_prosody"]["pitch_percent"]
    )


def test_one_speaker_is_not_artificially_spread():
    item = _assignment("ONLY", f0=210.0, rate=2.2)
    item["base_prosody"] = {
        "rate_percent": 3,
        "pitch_percent": -2,
        "volume_percent": 0,
    }
    result = apply_direct_azure_personas({"ONLY": item})
    persona = result["ONLY"]["base_prosody"]
    assert persona["family_persona_count"] == 1
    assert persona["separation_adjusted"] is False


def test_explicit_voice_map_remains_exact():
    item = _assignment(
        "MANUAL",
        resolution_status="explicit_voice_map",
    )
    item["base_prosody"] = {
        "rate_percent": 7,
        "pitch_percent": -4,
        "volume_percent": 1,
    }
    result = apply_direct_azure_personas({"MANUAL": item})
    persona = result["MANUAL"]["base_prosody"]
    assert persona["rate_percent"] == 7
    assert persona["pitch_percent"] == -4
    assert persona["volume_percent"] == 0
    assert persona["requested_volume_percent"] == 1
    assert persona["volume_policy"] == "dialogue_bus_loudness_normalization"
    assert persona["strategy"] == "explicit_voice_map"


def test_collects_profile_delivery_before_acoustic_fallback():
    features = collect_source_features(
        {
            "source_delivery": {
                "median_f0_hz": 220.0,
                "source_speech_rate_wps": 2.4,
            }
        },
        {"median_f0_hz": 180.0},
    )
    assert features["median_f0_hz"] == 220.0
    assert features["source_speech_rate_wps"] == 2.4
