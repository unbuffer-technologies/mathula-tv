import json
from types import SimpleNamespace

from mathula_tv.ai_provider import (
    AnthropicClaudeProvider,
    AnthropicConfig,
    build_full_clip_payload,
)
from mathula_tv.models import JobManifest
from mathula_tv.production_pipeline import ProductionDubbingPipeline
from mathula_tv.pronunciation import PronunciationDictionary
from mathula_tv.target_text import build_target_text


class Response:
    status_code = 200
    headers = {}

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class Session:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(
            {
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": json.dumps(self.output)}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        )


def test_one_call_translation_preserves_cross_unit_wordplay_fragments():
    payload = build_full_clip_payload(
        transcript={"segments": []},
        dubbing_units={
            "units": [
                {"unit_id": "unit_0007", "source_text": "He called it the Big"},
                {"unit_id": "unit_0008", "source_text": "W, West Side. Woolworths side."},
            ]
        },
    )
    output = {
        "units": [
            {
                "unit_id": "unit_0007",
                "faithful_translation": "Wayibiza nge-Big",
                "spoken_text": "Wayibiza nge-Big",
                "tts_text": "Wayibiza nge-Big",
                "language_features": [
                    {
                        "source_text": "Big",
                        "policy": "preserve_verbatim",
                        "display_text": "Big",
                        "tts_text": "Big",
                        "reason": "Part of cross-unit wordplay",
                    }
                ],
                "human_review_flags": [],
            },
            {
                "unit_id": "unit_0008",
                "faithful_translation": "W, West Side. Woolworths side.",
                "spoken_text": "W, West Side. Woolworths side.",
                "tts_text": "double-you, West Side. Woolworths side.",
                "language_features": [
                    {
                        "source_text": "W",
                        "policy": "spell_out",
                        "display_text": "W",
                        "tts_text": "whatever the model guessed",
                        "reason": "Letter name in Big W wordplay",
                    },
                    {
                        "source_text": "West Side",
                        "policy": "preserve_verbatim",
                        "display_text": "West Side",
                        "tts_text": "West Side",
                        "reason": "Wordplay and code-switching",
                    },
                ],
                "human_review_flags": [],
            },
        ]
    }
    session = Session(output)
    provider = AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=1, max_repairs=0),
        session=session,
        sleep=lambda _delay: None,
    )

    result = provider.translate_full_clip(payload)

    assert len(session.calls) == 1
    assert [unit["unit_id"] for unit in result.data["units"]] == ["unit_0007", "unit_0008"]
    assert result.data["units"][0]["language_features"][0]["display_text"] == "Big"
    assert result.data["units"][1]["language_features"][0]["tts_text"] == "double-you"
    assert result.data["units"][1]["spoken_text"].startswith("W, West Side")


def test_application_builds_tts_alias_without_changing_subtitle_text(tmp_path):
    pipeline = ProductionDubbingPipeline.__new__(ProductionDubbingPipeline)
    pipeline._pronunciation_dictionary = lambda _job: PronunciationDictionary(
        dictionary_version="v1",
        language="zu-ZA",
    )
    dictionary_path = tmp_path / "pronunciation_dictionary.json"
    pipeline.paths = lambda _job_id: SimpleNamespace(pronunciation_dictionary=dictionary_path)
    job = JobManifest(
        job_id="job1",
        source_filename="clip.mp4",
        local_source_path=str(tmp_path / "clip.mp4"),
        source_checksum="abc",
        source_duration=10.0,
        target_language="zu-ZA",
    )

    dictionary = pipeline._translation_pronunciation_dictionary(
        job,
        [
            {
                "unit_id": "unit_0008",
                "spoken_text": "W, West Side.",
                "language_features": [
                    {
                        "source_text": "W",
                        "policy": "spell_out",
                        "display_text": "W",
                        "tts_text": "double-you",
                        "reason": "letter name",
                    }
                ],
            }
        ],
    )

    entry = dictionary.job_overrides[0]
    assert entry.spoken_text == "W"
    assert entry.tts_text == "double-you"
    target = build_target_text("W, West Side.", dictionary=dictionary)
    assert target.spoken_text == "W, West Side."
    assert target.tts_text == "double-you, West Side."
    assert dictionary_path.is_file()


def test_one_call_uses_plain_json_mode_and_normalizes_minimal_mapping_response():
    payload = build_full_clip_payload(
        transcript={"segments": []},
        dubbing_units={
            "units": [
                {"unit_id": "unit_0001", "source_text": "Hello Big"},
                {"unit_id": "unit_0002", "source_text": "W, West Side."},
            ]
        },
    )

    class PlainSession:
        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            text = '''Here is the JSON:\n{
              "translations": {
                "unit_0001": {"translation": "Sawubona Big"},
                "unit_0002": {
                  "translation": "W, West Side.",
                  "language_features": [{
                    "source_text": "W",
                    "policy": "spell_out",
                    "display_text": "W"
                  }]
                }
              }
            }'''
            return Response(
                {
                    "model": "claude-opus-4-8",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                }
            )

    session = PlainSession()
    provider = AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=1, max_repairs=0),
        session=session,
        sleep=lambda _delay: None,
    )

    result = provider.translate_full_clip(payload)

    request_body = session.calls[0][1]["json"]
    assert "format" not in request_body["output_config"]
    assert request_body["messages"][0]["content"]
    assert [unit["unit_id"] for unit in result.data["units"]] == ["unit_0001", "unit_0002"]
    assert result.data["units"][0]["spoken_text"] == "Sawubona Big"
    assert result.data["units"][1]["language_features"][0]["tts_text"] == "double-you"


def test_ordinary_place_phrase_is_translatable_while_name_remains_protected():
    payload = build_full_clip_payload(
        transcript={"segments": []},
        dubbing_units={
            "units": [
                {
                    "unit_id": "unit_0001",
                    "source_text": "First, we had a malanga ward.",
                }
            ]
        },
    )
    output = {
        "units": [
            {
                "unit_id": "unit_0001",
                "faithful_translation": "Okokuqala, besinewadi yaseMalanga.",
                "spoken_text": "Okokuqala, besinewadi yaseMalanga.",
                "tts_text": "Okokuqala, besinewadi yaseMalanga.",
                "language_features": [
                    {
                        "source_text": "malanga ward",
                        "type": "place_reference",
                        "reason": "Refers to a ward named Malanga",
                    }
                ],
                "protected_entities_found": ["Malanga"],
                "protected_entities_preserved": ["Malanga"],
            }
        ]
    }
    session = Session(output)
    provider = AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=1, max_repairs=0),
        session=session,
        sleep=lambda _delay: None,
    )

    result = provider.translate_full_clip(payload)

    feature = result.data["units"][0]["language_features"][0]
    assert feature["source_text"] == "malanga ward"
    assert feature["policy"] == "translate"
    assert result.data["units"][0]["spoken_text"] == "Okokuqala, besinewadi yaseMalanga."


def test_protected_wordplay_allows_punctuation_without_losing_words():
    payload = build_full_clip_payload(
        transcript={"segments": []},
        dubbing_units={
            "units": [
                {
                    "unit_id": "unit_0009",
                    "source_text": "The Big W West Side is Woolworths side.",
                }
            ]
        },
    )
    output = {
        "units": [
            {
                "unit_id": "unit_0009",
                "faithful_translation": "The Big W, West Side, yi-Woolworths side.",
                "spoken_text": "The Big W, West Side, yi-Woolworths side.",
                "tts_text": "The Big double-you, West Side, yi-Woolworths side.",
                "language_features": [
                    {
                        "source_text": "The Big W West Side",
                        "policy": "preserve_and_flag",
                        "reason": "English wordplay",
                    }
                ],
            }
        ]
    }
    session = Session(output)
    provider = AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=1, max_repairs=0),
        session=session,
        sleep=lambda _delay: None,
    )

    result = provider.translate_full_clip(payload)

    assert result.data["units"][0]["spoken_text"] == (
        "The Big W, West Side, yi-Woolworths side."
    )
    assert result.data["units"][0]["language_features"][0]["display_text"] == (
        "The Big W West Side"
    )
