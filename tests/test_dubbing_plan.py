from mathula_tv.dubbing_plan import (
    build_dubbing_plan,
    normalize_translation_units,
    upgrade_legacy_translation_artifact,
)
from mathula_tv.pronunciation import PronunciationDictionary
import pytest


UNITS = [
    {
        "schema_version": "1.0",
        "unit_id": "unit_0001",
        "speaker_id": "AZURE_1",
        "source_phrase_ids": ["seg-1"],
        "source_word_ids": ["w1"],
        "source_text": "A long opening",
        "start_ms": 0,
        "preferred_end_ms": 1000,
        "hard_end_ms": 1100,
        "preferred_duration_ms": 1000,
        "maximum_duration_ms": 1100,
        "preceding_silence_ms": 0,
        "following_silence_ms": 0,
        "has_overlap": False,
        "overlap_group_id": None,
    },
    {
        "schema_version": "1.0",
        "unit_id": "unit_0002",
        "speaker_id": "AZURE_1",
        "source_phrase_ids": ["seg-1"],
        "source_word_ids": ["w2"],
        "source_text": "that continues here",
        "start_ms": 1000,
        "preferred_end_ms": 2000,
        "hard_end_ms": 2100,
        "preferred_duration_ms": 1000,
        "maximum_duration_ms": 1100,
        "preceding_silence_ms": 0,
        "following_silence_ms": 0,
        "has_overlap": False,
        "overlap_group_id": None,
    },
]


def test_legacy_upgrade_preserves_all_approved_words_without_ai():
    legacy = {
        "schema_version": "translation-v1",
        "turns": [{"segment_id": "seg-1", "translated_text": "Lena inkulumo egunyaziwe ayishintshi nhlobo"}],
    }
    result = upgrade_legacy_translation_artifact(legacy, UNITS)
    combined = " ".join(item["spoken_text"] for item in result["units"])
    assert combined == legacy["turns"][0]["translated_text"]
    assert result["ai_called"] is False and result["approved_content_preserved"] is True
    assert all(item["spoken_text"] == item["tts_text"] for item in result["units"])


def test_plan_records_distinct_source_audio_and_pending_conversion(tmp_path):
    source = tmp_path / "source.mp4"
    analysis = tmp_path / "analysis.wav"
    mix = tmp_path / "mix.wav"
    source.write_bytes(b"video")
    analysis.write_bytes(b"analysis")
    mix.write_bytes(b"mix")
    legacy = {
        "schema_version": "translation-v1",
        "turns": [{"segment_id": "seg-1", "translated_text": "Lena inkulumo egunyaziwe ayishintshi nhlobo"}],
    }
    artifact = {
        "schema_version": "dubbing-units-v1",
        "source_schema_version": "transcript-v1",
        "source_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
        "units": UNITS,
    }
    plan = build_dubbing_plan(
        job={"job_id": "job-1"},
        source_media=source,
        analysis_audio=analysis,
        mix_source_audio=mix,
        unit_artifact=artifact,
        translation_artifact=legacy,
        output_path=tmp_path / "dubbing_plan.json",
    )
    assert plan["analysis_audio"]["sha256"] != plan["mix_source_audio"]["sha256"]
    assert plan["compatibility"]["state"] == "pending"
    assert {item["openvoice"]["status"] for item in plan["units"]} == {"pending"}
    assert plan["translation_identity"]["ai_called_during_upgrade"] is False


def test_timing_aware_spoken_rewrite_is_recorded_and_unreviewed_tts_proposal_is_not_applied():
    translated = {
        "schema_version": "claude-translation-v1",
        "units": [
            {
                "unit_id": "unit_0001",
                "faithful_translation": "Lona umusho ophelele obalulekile.",
                "spoken_text": "Lona umusho obalulekile.",
                "tts_text": "Loh-na umusho obalulekile.",
                "timing_strategy": "bounded contraction",
                "omitted_or_compressed_detail": ["No fact omitted"],
                "protected_entities_found": [],
                "human_review_flags": ["Review contraction"],
            },
            {
                "unit_id": "unit_0002",
                "faithful_translation": "Uyaqhubeka lapha.",
                "spoken_text": "Uyaqhubeka lapha.",
                "tts_text": "Uyaqhubeka lapha.",
                "protected_entities_found": [],
            },
        ],
    }
    result = normalize_translation_units(
        translated,
        UNITS,
        pronunciation_dictionary=PronunciationDictionary(dictionary_version="v1"),
    )
    unit = result["units"][0]
    assert unit["text_changes"][0]["layer"] == "faithful_to_spoken"
    assert unit["tts_text"] == unit["spoken_text"]
    assert unit["tts_proposal_review"]["applied"] is False


@pytest.mark.parametrize(
    "left,right",
    [
        ("I-Big", "W West Side."),
        ("Lokhu ayikwazi ngisho", "nokuthumela i-email."),
        ("Uyaphuza bese", "kungazelelwe uyawa."),
        ("Kuvele", "ukuthi uyahamba."),
    ],
)
def test_unsafe_target_splits_are_rejected(left, right):
    translated = {
        "units": [
            {"unit_id": "unit_0001", "faithful_translation": left, "spoken_text": left, "tts_text": left},
            {"unit_id": "unit_0002", "faithful_translation": right, "spoken_text": right, "tts_text": right},
        ]
    }
    with pytest.raises(ValueError, match="Unsafe target split"):
        normalize_translation_units(translated, UNITS)
