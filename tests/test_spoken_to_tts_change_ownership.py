from mathula_tv.dubbing_plan import normalize_translation_units
from mathula_tv.pronunciation import PronunciationDictionary, PronunciationEntry


def test_model_spoken_to_tts_span_is_rebuilt_by_application_dictionary() -> None:
    dictionary = PronunciationDictionary(
        dictionary_version="v1",
        entries=(
            PronunciationEntry(
                display_text="W",
                spoken_text="W",
                tts_text="double-you",
                language="zu-ZA",
                source="test",
                confidence=1.0,
                kind="english_code_switch",
            ),
        ),
    )
    translation = {
        "schema_version": "claude-translation-v2",
        "units": [
            {
                "unit_id": "unit_0008",
                "faithful_translation": "Big W",
                "spoken_text": "Big W",
                "tts_text": "Big double-you",
                "text_changes": [
                    {
                        "schema_version": "text-change-v1",
                        "layer": "spoken_to_tts",
                        "change_type": "code_switch",
                        "source_start": 0,
                        "source_end": 5,
                        "before": "Big W",
                        "after": "Big double-you",
                        "reason": "Model-proposed broad TTS alias",
                        "protected_terms": [],
                        "human_review_required": False,
                    }
                ],
            }
        ],
    }

    result = normalize_translation_units(
        translation,
        [{"unit_id": "unit_0008"}],
        pronunciation_dictionary=dictionary,
    )

    unit = result["units"][0]
    assert unit["spoken_text"] == "Big W"
    assert unit["tts_text"] == "Big double-you"
    assert len(unit["text_changes"]) == 1
    assert unit["text_changes"][0]["source_start"] == 4
    assert unit["text_changes"][0]["source_end"] == 5
    assert unit["text_changes"][0]["before"] == "W"
    assert unit["text_changes"][0]["after"] == "double-you"
    assert unit["tts_proposal_review"]["proposal_matches_application"] is True
    assert unit["tts_proposal_review"]["discarded_model_spoken_to_tts_change_count"] == 1
    assert unit["tts_proposal_review"]["human_review_required"] is False


def test_faithful_to_spoken_change_is_not_discarded() -> None:
    translation = {
        "schema_version": "claude-translation-v2",
        "units": [
            {
                "unit_id": "unit_0001",
                "faithful_translation": "Long wording",
                "spoken_text": "Short",
                "tts_text": "Short",
                "text_changes": [
                    {
                        "schema_version": "text-change-v1",
                        "layer": "faithful_to_spoken",
                        "change_type": "contraction",
                        "source_start": 0,
                        "source_end": 12,
                        "before": "Long wording",
                        "after": "Short",
                        "reason": "Reviewed timing contraction",
                        "protected_terms": [],
                        "human_review_required": True,
                    }
                ],
            }
        ],
    }

    result = normalize_translation_units(translation, [{"unit_id": "unit_0001"}])
    unit = result["units"][0]
    assert unit["spoken_text"] == "Short"
    assert unit["tts_text"] == "Short"
    assert len(unit["text_changes"]) == 1
    assert unit["text_changes"][0]["layer"] == "faithful_to_spoken"
