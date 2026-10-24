import wave

import numpy as np

from mathula_tv.source_gaps import find_gaps, find_missing_speech, insertion
from mathula_tv.source_suspects import apply_corrections


def write_wav(path, seconds_loud, total=40, rate=16000):
    """Quiet background everywhere, 'speech' (loud noise) in the listed [start, end) second ranges."""
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 30, total * rate)                         # quiet floor
    for start, end in seconds_loud:
        lo, hi = int(round(start * rate)), int(round(end * rate))
        signal[lo:hi] = rng.normal(0, 4000, hi - lo)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.clip(signal, -32768, 32767).astype(np.int16).tobytes())


WORDS = [{"text": "Leopard", "start": 10.0, "end": 10.4}, {"text": "Creek", "start": 10.4, "end": 10.9},
         {"text": "Rupert", "start": 20.8, "end": 21.3}, {"text": "is", "start": 21.6, "end": 22.1},
         {"text": "We", "start": 26.1, "end": 26.3}, {"text": "don't", "start": 26.3, "end": 26.6}]
SEGMENTS = [{"segment_id": "seg-10", "start_ms": 19000, "end_ms": 22400, "restored_text": "But Leopard Creek will pay. Rupert is—"},
            {"segment_id": "seg-11", "start_ms": 26100, "end_ms": 30000, "restored_text": "We don't care about Rupert."}]


def test_only_pauses_of_a_second_or_more_are_gaps():
    gaps = [(a["text"], b["text"]) for a, b in find_gaps(WORDS)]
    assert gaps == [("Creek", "Rupert"), ("is", "We")]


def test_the_missing_words_are_inserted_after_the_word_the_speaker_was_in_the_middle_of():
    change = insertion("But Leopard Creek will pay. Rupert is—", "is", "going to pay for water and electricity")
    assert change == {"token": "is—", "replacement": "is going to pay for water and electricity."}
    assert insertion("nothing here", "is", "x") is None


def test_speech_in_a_transcript_gap_is_listened_to_again_and_added_but_silence_and_noise_are_not(tmp_path):
    audio = tmp_path / "a.wav"
    write_wav(audio, [(8, 11), (20, 22.2), (22.4, 26.0), (26, 30)])      # loud 22.4-26: speech the transcript missed; 11-20 is silence
    asked = []

    def transcribe(path, start, end):
        asked.append((round(start), round(end)))
        return [{"text": "going", "start": 1.4}, {"text": "to", "start": 1.7}, {"text": "pay", "start": 2.0}, {"text": "for", "start": 2.3},
                {"text": "water", "start": 2.6}, {"text": "and", "start": 2.9}, {"text": "electricity", "start": 3.2}]

    found = find_missing_speech(SEGMENTS, WORDS, audio, transcribe, progress=lambda message: None)
    assert asked == [(21, 27)]                                            # the quiet 11-20 s gap was never re-transcribed
    assert len(found) == 1 and found[0]["segment_id"] == "seg-10"
    pass1 = {"segments": [dict(s, source_text=s["restored_text"]) for s in SEGMENTS]}
    apply_corrections(pass1, found)
    assert pass1["segments"][0]["restored_text"].endswith("Rupert is going to pay for water and electricity.")


def test_a_loud_gap_that_yields_no_words_adds_nothing(tmp_path):
    audio = tmp_path / "a.wav"
    write_wav(audio, [(22.4, 26.0)])
    assert find_missing_speech(SEGMENTS, WORDS, audio, lambda p, s, e: [{"text": "uh", "start": 1.0}], progress=lambda m: None) == []
