from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv import stt_merge as merge
from mathula_tv.azure_stt import AzureSpeechError, normalize
from mathula_tv.cloud_stt import CloudSTTBackend


def w(text, start, end, p=0.9):
    return {"text": text, "start": start, "end": end, "probability": p}


TURNS = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 4.0}, {"speaker": "SPEAKER_00", "start": 4.2, "end": 9.0}]
WORDS = [w(" Good", 0.2, 0.5), w(" morning,", 0.5, 1.0), w(" Chair.", 1.1, 1.6), w(" Please", 4.3, 4.7), w(" state", 4.7, 5.0), w(" your", 5.0, 5.2), w(" name.", 5.2, 5.8)]


def test_speakers_are_numbered_by_who_speaks_first_and_each_word_takes_the_turn_it_overlaps():
    assert merge.speaker_numbers(TURNS) == {"SPEAKER_01": 1, "SPEAKER_00": 2}
    assert merge.assign_speakers(WORDS, TURNS) == [1, 1, 1, 2, 2, 2, 2]


def test_a_word_in_a_silence_takes_the_near_turn_and_a_far_one_the_speaker_before_it():
    words = [w("a", 0.1, 0.3), w("b", 4.05, 4.15), w("c", 20.0, 20.2)]          # b falls in the 4.0-4.2 gap; c is far from every turn
    assert merge.assign_speakers(words, TURNS)[0] == 1
    assert merge.assign_speakers(words, TURNS)[1] in (1, 2)                       # equally near both turns: either is a fair answer, but it must be one of them
    assert merge.assign_speakers(words, TURNS)[2] == merge.assign_speakers(words, TURNS)[1]          # beyond NEAR_S of every turn: it keeps the speaker before it
    assert merge.assign_speakers([], TURNS) == [] and merge.assign_speakers([w("x", 1, 2)], []) == [None]


def test_phrases_are_sentences_of_one_speaker_in_azures_shape():
    raw = merge.to_azure_raw(WORDS, TURNS, duration_s=10.0, locale="en-ZA")
    assert raw["durationMilliseconds"] == 10_000 and [p["speaker"] for p in raw["phrases"]] == [1, 2]
    first = raw["phrases"][0]
    assert first["text"] == "Good morning, Chair." and first["offsetMilliseconds"] == 200 and first["durationMilliseconds"] == 1400 and first["locale"] == "en-ZA"
    assert [x["text"] for x in first["words"]] == ["Good", "morning,", "Chair."] and first["words"][1]["offsetMilliseconds"] == 500            # words keep their punctuation, as Azure's do
    assert raw["combinedPhrases"][0]["text"] == "Good morning, Chair. Please state your name."


def test_a_long_silence_and_a_closed_sentence_split_phrases_but_an_abbreviation_does_not():
    words = [w("Mr.", 0.0, 0.3), w("Smith", 0.35, 0.8), w("spoke.", 0.85, 1.2), w("Then", 1.3, 1.6), w("quiet", 3.5, 3.9)]
    assert [p["text"] for p in merge.build_phrases(words, [1] * 5)] == ["Mr. Smith spoke.", "Then", "quiet"]          # the sentence ends at "spoke."; the 1.9 s silence ends "Then"
    assert [p["text"] for p in merge.build_phrases([w("Really?", 0, 0.5), w("yes", 0.55, 0.9)], [1, 1])] == ["Really?", "yes"]          # a question mark ends it whatever follows


def test_without_turns_the_phrases_carry_no_speaker():
    raw = merge.to_azure_raw(WORDS, [], duration_s=6.0, locale="en-ZA")
    assert all("speaker" not in p for p in raw["phrases"])


def test_the_response_is_read_by_the_existing_normaliser_unchanged():
    normalised = normalize(merge.to_azure_raw(WORDS, TURNS, duration_s=10.0, locale="en-ZA"), provider="whisper-pyannote-cloud-run")
    assert [p["speaker"] for p in normalised["phrases"]] == ["AZURE_1", "AZURE_2"] and len(normalised["words"]) == 7 and normalised["turns"][1]["start"] == 4.3
    assert normalised["complete_text"].startswith("Good morning, Chair.")


def test_the_service_carries_an_identical_copy_of_the_merge_module():
    root = Path(__file__).resolve().parents[1]
    assert (root / "gcp" / "stt_service" / "stt_merge.py").read_text(encoding="utf-8") == (root / "src" / "mathula_tv" / "stt_merge.py").read_text(encoding="utf-8")


# ---- the client ----------------------------------------------------------------------------------------------------------------

class FakeSession:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def post(self, url, headers=None, files=None, data=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": json.loads(data["params"]), "file": files["audio"][0]})
        return self.replies.pop(0)


def reply(status=200, body=None):
    return SimpleNamespace(status_code=status, text=json.dumps(body or {}), json=lambda: body if body is not None else (_ for _ in ()).throw(ValueError("bad")))


def backend(session, **kw):
    return CloudSTTBackend(url="https://stt.example/", session=session, token=lambda url: "tok", sleep=lambda s: None, **kw)


def audio(tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    path.write_bytes(b"RIFFxxxx")
    monkeypatch.setattr("mathula_tv.cloud_stt.opus_16k_mono", lambda source, target: (target.write_bytes(b"ogg"), target)[1])
    return path


GOOD = merge.to_azure_raw(WORDS, TURNS, duration_s=10.0, locale="en-ZA")


def test_english_goes_to_whisper_with_speakers_and_the_phrase_hints_and_the_token(tmp_path, monkeypatch):
    session = FakeSession([reply(200, GOOD)])
    stt = backend(session, phrase_list=("Madlanga", " "))
    assert stt.transcribe(audio(tmp_path, monkeypatch), "en-ZA") == GOOD
    call = session.calls[0]
    assert call["url"] == "https://stt.example/v1/transcribe" and call["headers"]["Authorization"] == "Bearer tok"
    assert call["params"] == {"locale": "en-ZA", "diarize": True, "max_speakers": 10, "phrases": ["Madlanga"]} and call["file"] == "audio.ogg"
    assert stt.request_duration_seconds is not None


def test_another_language_is_refused_visibly_and_nothing_is_sent(tmp_path, monkeypatch):
    session = FakeSession([])
    with pytest.raises(AzureSpeechError, match="English only"):
        backend(session).transcribe(audio(tmp_path, monkeypatch), "zu-ZA")
    assert session.calls == []


def test_a_busy_service_is_retried_and_a_refusal_is_not(tmp_path, monkeypatch):
    path = audio(tmp_path, monkeypatch)
    session = FakeSession([reply(503), reply(200, GOOD)])
    assert backend(session).transcribe(path) == GOOD and len(session.calls) == 2
    refused = FakeSession([reply(403)])
    with pytest.raises(AzureSpeechError) as error:
        backend(refused).transcribe(path)
    assert error.value.status_code == 403 and not error.value.retryable and len(refused.calls) == 1


def test_an_empty_transcription_is_an_error_and_the_url_must_be_configured(tmp_path, monkeypatch):
    with pytest.raises(AzureSpeechError):
        backend(FakeSession([reply(200, {"phrases": [], "combinedPhrases": []})])).transcribe(audio(tmp_path, monkeypatch))
    with pytest.raises(ValueError):
        CloudSTTBackend.from_environment({})
    assert CloudSTTBackend.from_environment({"STT_URL": "https://x/"}).url == "https://x"


def test_the_provider_switch_chooses_the_cloud_backend(monkeypatch):
    from mathula_tv.cli import create_speech_backend

    monkeypatch.setenv("MATHULA_STT_PROVIDER", "whisper")
    monkeypatch.setenv("STT_URL", "https://stt.example")
    chosen = create_speech_backend(SimpleNamespace(azure_speech_max_speakers=6))
    assert isinstance(chosen, CloudSTTBackend) and chosen.max_speakers == 6
