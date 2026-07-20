import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import mathula_tv.azure_stt as stt
from mathula_tv.azure_stt import AzureFastTranscriptionBackend, AzureSpeechError, derive_endpoint, normalize, request_definition
from mathula_tv.cli import create_speech_backend


RAW = {
    "durationMilliseconds": 2500,
    "combinedPhrases": [{"channel": 0, "text": "Hello South Africa."}],
    "phrases": [
        {"offsetMilliseconds": 100, "durationMilliseconds": 900, "text": "Hello", "locale": "en-ZA", "confidence": 0.91, "speaker": 1, "words": [{"text": "Hello", "offsetMilliseconds": 100, "durationMilliseconds": 500}]},
        {"offsetMilliseconds": 1100, "durationMilliseconds": 1200, "text": "South Africa.", "locale": "en-ZA", "confidence": 0.87, "speaker": 2, "words": [{"text": "South", "offsetMilliseconds": 1100, "durationMilliseconds": 400}, {"text": "Africa", "offsetMilliseconds": 1550, "durationMilliseconds": 500}]},
    ],
}


class Response:
    def __init__(self, status=200, body=RAW, text="", headers=None):
        self.status_code, self.body, self.text, self.headers = status, body, text, headers or {}
    def json(self):
        if isinstance(self.body, Exception): raise self.body
        return self.body


class Session:
    def __init__(self, responses): self.responses, self.calls = list(responses), []
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs)); response = self.responses.pop(0)
        if isinstance(response, Exception): raise response
        return response


@pytest.fixture
def audio(tmp_path, monkeypatch):
    path = tmp_path / "source_audio.wav"; path.write_bytes(b"RIFF" + b"x" * 100)
    monkeypatch.setattr(stt, "probe", lambda path: {"duration": 2.5})
    return path


def backend(session, sleeps=None):
    sink = sleeps if sleeps is not None else []
    return AzureFastTranscriptionBackend("https://eastus.api.cognitive.microsoft.com", "not-a-real-key", session=session, sleep=sink.append)


def test_endpoint_derivation_and_explicit_override(monkeypatch):
    assert derive_endpoint("SouthAfricaNorth") == "https://southafricanorth.api.cognitive.microsoft.com"
    monkeypatch.setenv("AZURE_SPEECH_KEY", "not-a-real-key")
    settings = SimpleNamespace(azure_speech_endpoint="https://custom.cognitiveservices.azure.com", azure_speech_region="ignored", azure_speech_api_version="2025-10-15", azure_speech_max_speakers=10, azure_speech_timeout_seconds=600)
    assert create_speech_backend(settings).endpoint == "https://custom.cognitiveservices.azure.com"


def test_missing_cli_configuration(monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    with pytest.raises(ValueError, match="AZURE_SPEECH_KEY is missing"):
        create_speech_backend(SimpleNamespace())


def test_multipart_locale_diarization_and_word_timestamp_contract(audio):
    session = Session([Response()]); result = backend(session).transcribe(audio, "en-ZA")
    assert result == RAW
    url, kwargs = session.calls[0]
    definition = json.loads(kwargs["data"]["definition"])
    assert definition == {"locales": ["en-ZA"], "diarization": {"enabled": True, "maxSpeakers": 10}}
    assert "audio" in kwargs["files"] and kwargs["timeout"] == (15, 600)
    assert "wordLevelTimestampsEnabled" not in definition  # Fast API returns them without the batch-only switch.


def test_normalization_multiple_and_unknown_speakers_and_words():
    result = normalize(RAW, raw_reference="gs://bucket/raw.json")
    assert result["complete_text"] == "Hello South Africa."
    assert result["phrases"][0]["start"] == .1 and result["words"][2]["end"] == 2.05
    assert {t["speaker"] for t in result["turns"]} == {"AZURE_1", "AZURE_2"}
    assert result["raw_response_reference"] == "gs://bucket/raw.json"
    unknown = dict(RAW); unknown["phrases"] = [dict(RAW["phrases"][0])]; unknown["phrases"][0].pop("speaker")
    assert normalize(unknown)["phrases"][0]["speaker"] == "UNKNOWN"


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_not_retried(audio, status):
    session = Session([Response(status, {"error": {"message": "secret detail"}})])
    with pytest.raises(AzureSpeechError, match="authentication/authorization") as error: backend(session).transcribe(audio)
    assert not error.value.retryable and len(session.calls) == 1 and "not-a-real-key" not in str(error.value)


def test_429_respects_retry_after(audio):
    sleeps=[]; session=Session([Response(429, {"error":{"message":"slow"}}, headers={"Retry-After":"3"}), Response()])
    assert backend(session, sleeps).transcribe(audio) == RAW and sleeps == [3.0]


@pytest.mark.parametrize("status", [500, 503])
def test_server_errors_retry(audio, status):
    session=Session([Response(status, text="temporary", body=ValueError()), Response()])
    assert backend(session).transcribe(audio) == RAW and len(session.calls) == 2


def test_timeout_retry_and_safe_failure(audio):
    session=Session([requests.Timeout(), requests.Timeout(), requests.Timeout()])
    with pytest.raises(AzureSpeechError, match="timed out") as error: backend(session).transcribe(audio)
    assert error.value.retryable and len(session.calls) == 3


def test_malformed_and_empty_response(audio):
    with pytest.raises(AzureSpeechError, match="malformed JSON"): backend(Session([Response(body=ValueError())])).transcribe(audio)
    empty={"durationMilliseconds": 10, "combinedPhrases": [], "phrases": []}
    with pytest.raises(AzureSpeechError, match="empty transcription"): backend(Session([Response(body=empty)])).transcribe(audio)
