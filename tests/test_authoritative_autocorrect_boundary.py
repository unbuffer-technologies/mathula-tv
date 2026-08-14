import json

import pytest

from mathula_tv.autocorrect_stage import assert_no_configured_autocorrect_confusions
from mathula_tv.dubbing_units import build_dubbing_units


def write_hints(root):
    path = root / "config/autocorrect_hints.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "pattern": r"\bhorse[\s-]+curry\b",
                        "replacement": "hoshkhari",
                        "flags": ["IGNORECASE"],
                        "auto_apply": True,
                    }
                ],
                "review_patterns": [],
            }
        ),
        encoding="utf-8",
    )


def test_language_ai_guard_rejects_unresolved_curated_confusion(tmp_path):
    write_hints(tmp_path)
    transcript = {
        "segments": [
            {
                "segment_id": "segment_1",
                "source_text": "The horse curry dons arrived.",
            }
        ]
    }
    with pytest.raises(ValueError, match="configured auto-apply Azure STT confusion"):
        assert_no_configured_autocorrect_confusions(transcript, project_root=tmp_path)


def test_language_ai_guard_accepts_corrected_authoritative_text(tmp_path):
    write_hints(tmp_path)
    transcript = {
        "segments": [
            {
                "segment_id": "segment_1",
                "source_text": "The hoshkhari dons arrived.",
            }
        ]
    }
    assert_no_configured_autocorrect_confusions(transcript, project_root=tmp_path)


def corrected_transcript(source_text):
    return {
        "schema_version": "transcript-v1",
        "duration_ms": 2_000,
        "segments": [
            {
                "segment_id": "segment_1",
                "speaker": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 2_000,
                "source_text": source_text,
                "words": [
                    {"word_id": "w1", "text": "The", "start_ms": 0, "end_ms": 300},
                    {"word_id": "w2", "text": "horse", "start_ms": 310, "end_ms": 700},
                    {"word_id": "w3", "text": "curry", "start_ms": 710, "end_ms": 1_100},
                    {"word_id": "w4", "text": "dons", "start_ms": 1_110, "end_ms": 1_500},
                    {"word_id": "w5", "text": "arrived.", "start_ms": 1_510, "end_ms": 2_000},
                ],
            }
        ],
        "autocorrect": {
            "schema_version": "mathula-autocorrect-v1",
            "overlays": [
                {
                    "item_id": "segment_1",
                    "azure_text": "horse curry",
                    "corrected_text": "hoshkhari",
                    "start_ms": 310,
                    "end_ms": 1_100,
                }
            ],
        },
    }


def test_corrected_phrase_replaces_raw_timed_words_in_units():
    artifact = build_dubbing_units(corrected_transcript("The hoshkhari dons arrived."))
    text = " ".join(unit["source_text"] for unit in artifact["units"]).casefold()
    assert "hoshkhari" in text
    assert "horse curry" not in text


def test_overlay_claim_without_effective_correction_is_rejected():
    with pytest.raises(ValueError, match="Corrected autocorrect text did not propagate"):
        build_dubbing_units(corrected_transcript("The horse curry dons arrived."))
