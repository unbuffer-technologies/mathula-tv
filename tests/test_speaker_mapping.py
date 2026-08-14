from __future__ import annotations

from mathula_tv.speaker_mapping import (
    build_speaker_id_mapping,
    canonicalize_voice_family_evidence,
)


def test_prefers_explicit_raw_speaker_mapping() -> None:
    transcript = {
        "segments": [
            {
                "speaker_id": "SPEAKER_00",
                "raw_speaker": "AZURE_2",
                "start": 0.0,
                "end": 3.0,
            },
            {
                "speaker_id": "SPEAKER_01",
                "raw_speaker": "AZURE_1",
                "start": 3.0,
                "end": 8.0,
            },
        ]
    }
    diarization = {
        "turns": [
            {"speaker": "AZURE_2", "start": 0.0, "end": 3.0},
            {"speaker": "AZURE_1", "start": 3.0, "end": 8.0},
        ]
    }

    result = build_speaker_id_mapping(transcript, diarization)

    assert result["raw_to_canonical"] == {
        "AZURE_1": "SPEAKER_01",
        "AZURE_2": "SPEAKER_00",
    }
    assert {item["source"] for item in result["mappings"]} == {
        "transcript_raw_speaker"
    }


def test_uses_timeline_overlap_for_older_transcript() -> None:
    transcript = {
        "segments": [
            {"speaker_id": "SPEAKER_00", "start": 0.0, "end": 9.0},
            {"speaker_id": "SPEAKER_01", "start": 10.0, "end": 20.0},
        ]
    }
    diarization = {
        "turns": [
            {"speaker": "AZURE_7", "start": 0.2, "end": 8.8},
            {"speaker": "AZURE_3", "start": 10.1, "end": 19.8},
        ]
    }

    result = build_speaker_id_mapping(transcript, diarization)

    assert result["raw_to_canonical"] == {
        "AZURE_3": "SPEAKER_01",
        "AZURE_7": "SPEAKER_00",
    }
    assert {item["source"] for item in result["mappings"]} == {
        "timeline_overlap"
    }


def test_contextual_split_preserves_one_raw_id_for_both_canonical_speakers() -> None:
    transcript = {
        "segments": [
            {
                "speaker_id": "SPEAKER_00",
                "raw_speaker": "AZURE_1",
                "start": 0.0,
                "end": 40.0,
            },
            {
                "speaker_id": "SPEAKER_01",
                "raw_speaker": "AZURE_1",
                "start": 44.0,
                "end": 90.0,
                "speaker_assignment_method": "contextual_broadcast_handoff_repair",
            },
        ]
    }
    diarization = {
        "turns": [{"speaker": "AZURE_1", "start": 0.0, "end": 90.0}]
    }

    result = build_speaker_id_mapping(transcript, diarization)

    # The legacy one-to-one map chooses the dominant duration, while the
    # reverse audit map retains the intentional contextual split.
    assert result["raw_to_canonical"] == {"AZURE_1": "SPEAKER_01"}
    assert result["canonical_to_raw"] == {
        "SPEAKER_00": ["AZURE_1"],
        "SPEAKER_01": ["AZURE_1"],
    }


def test_canonicalizes_raw_classifier_results() -> None:
    mapping = {
        "raw_to_canonical": {
            "AZURE_1": "SPEAKER_00",
            "AZURE_2": "SPEAKER_01",
        }
    }
    acoustic = {
        "speakers": {
            "AZURE_1": {
                "speaker_id": "AZURE_1",
                "voice_family": "feminine",
                "confidence": 0.94,
                "evidence": {"method": "ecapa_voice_family_classifier"},
            },
            "AZURE_2": {
                "speaker_id": "AZURE_2",
                "voice_family": "masculine",
                "confidence": 0.91,
                "evidence": {"method": "ecapa_voice_family_classifier"},
            },
        }
    }

    result = canonicalize_voice_family_evidence(acoustic, mapping)

    assert result["SPEAKER_00"]["voice_family"] == "feminine"
    assert result["SPEAKER_00"]["source_speaker_id"] == "AZURE_1"
    assert result["SPEAKER_01"]["voice_family"] == "masculine"
    assert result["SPEAKER_01"]["source_speaker_id"] == "AZURE_2"
