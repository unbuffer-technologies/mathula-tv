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
    change = insertion("But Leopard Creek will pay. Rupert is—", "is", "going to pay for water and electricity.")
    assert change == {"token": "is—", "replacement": "is going to pay for water and electricity."}       # no double full stop
    assert insertion("A cleaner is a worker.", "worker", "to be guaranteed that if you become sick")["replacement"] == \
        "worker to be guaranteed that if you become sick."                                              # a lowercase continuation
    assert insertion("We need education.", "education", "Science and technology.")["replacement"] == \
        "education. Science and technology."                                                           # a sentence of its own
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
    assert asked == [(21, 27), (22, 26)]                                  # the gap, with context and tight; the quiet 11-20 s gap was never re-transcribed
    assert len(found) == 1 and found[0]["segment_id"] == "seg-10"
    pass1 = {"segments": [dict(s, source_text=s["restored_text"]) for s in SEGMENTS]}
    apply_corrections(pass1, found)
    assert pass1["segments"][0]["restored_text"].endswith("Rupert is going to pay for water and electricity.")
    # the words are spoken in the gap, so the segment's window reaches them (it ended at 22.4 s, the last word starts at 24.3 s)
    assert 24300 < found[0]["extends_to_ms"] <= 24500
    assert pass1["segments"][0]["end_ms"] == found[0]["extends_to_ms"]


def test_a_recovered_window_never_runs_into_the_next_segment_and_never_shrinks():
    segments = [{"segment_id": "a", "start_ms": 0, "end_ms": 5000, "restored_text": "one two."},
                {"segment_id": "b", "start_ms": 9000, "end_ms": 12000, "restored_text": "three."}]
    far = [{"segment_id": "a", "token": "two.", "replacement": "two four.", "extends_to_ms": 20000}]
    apply_corrections({"segments": segments}, far)
    assert segments[0]["end_ms"] == 8950                                   # stops 50 ms short of the next segment
    near = [{"segment_id": "a", "token": "four.", "replacement": "four five.", "extends_to_ms": 4000}]
    apply_corrections({"segments": segments}, near)
    assert segments[0]["end_ms"] == 8950


def test_the_second_pass_keeps_only_the_middle_of_each_chunk_and_reports_absolute_times():
    from mathula_tv.source_gaps import second_pass_words

    spoken = [(2.0, "alpha"), (27.0, "bravo"), (29.0, "edge-of-first"), (33.5, "charlie"), (58.0, "delta")]   # (absolute start, word)

    def transcribe(audio, start, end):
        return [{"text": t, "start": s - start, "end": s - start + 0.3} for s, t in spoken if start <= s and s + 0.3 <= end]

    heard = second_pass_words(60.0, transcribe, None)
    assert [w["text"] for w in heard] == ["alpha", "bravo", "edge-of-first", "charlie", "delta"]       # each word exactly once
    assert [round(w["start"], 1) for w in heard] == [2.0, 27.0, 29.0, 33.5, 58.0]


def test_stretches_are_what_the_second_pass_hears_where_the_first_pass_has_nothing():
    from mathula_tv.source_gaps import uncovered_stretches

    first = [{"text": "one", "start": 1.0, "end": 1.4}, {"text": "two", "start": 1.5, "end": 1.9}, {"text": "six", "start": 9.0, "end": 9.4}]
    second = [{"text": "one", "start": 1.05, "end": 1.4}, {"text": "two", "start": 1.55, "end": 1.9},
              {"text": "three", "start": 3.0, "end": 3.4}, {"text": "four", "start": 3.6, "end": 4.0}, {"text": "five", "start": 4.2, "end": 4.6},
              {"text": "lone", "start": 6.5, "end": 6.8}, {"text": "six", "start": 9.1, "end": 9.4}]
    assert uncovered_stretches(second, first) == [(3.0, 4.6)]               # a single stray word (6.5 s) is not enough


def test_speech_only_the_second_pass_hears_is_listened_to_again_and_added_even_inside_a_short_pause(tmp_path):
    audio = tmp_path / "a.wav"
    write_wav(audio, [(19, 30)])
    words = [{"text": "Rupert", "start": 20.0, "end": 20.4}, {"text": "is", "start": 20.5, "end": 20.9}, {"text": "We", "start": 21.95, "end": 22.2}]
    segments = [{"segment_id": "s", "start_ms": 19500, "end_ms": 21000, "restored_text": "Rupert is—"},
                {"segment_id": "t", "start_ms": 21950, "end_ms": 25000, "restored_text": "We care."}]
    second = [{"text": "going", "start": 21.05, "end": 21.3}, {"text": "to", "start": 21.35, "end": 21.45}, {"text": "pay", "start": 21.5, "end": 21.75}]
    asked = []

    def transcribe(path, start, end):
        asked.append((round(start, 1), round(end, 1)))
        return [{**w, "start": w["start"] - start, "end": w["end"] - start} for w in second]

    assert find_missing_speech(segments, words, audio, transcribe, progress=lambda m: None) == []        # a 1.05 s pause: no first-pass candidate
    found = find_missing_speech(segments, words, audio, transcribe, progress=lambda m: None, second_pass=second)
    assert len(found) == 1 and found[0]["replacement"] == "is going to pay."
    assert 21500 < found[0]["extends_to_ms"] <= 21850


def test_the_second_passs_own_words_win_when_they_are_the_fullest_listen(tmp_path):
    audio = tmp_path / "a.wav"
    write_wav(audio, [(22.4, 26.0)])
    second = [{"text": "going", "start": 22.5, "end": 22.8}, {"text": "to", "start": 22.9, "end": 23.0}, {"text": "pay", "start": 23.1, "end": 23.4},
              {"text": "cells.", "start": 23.5, "end": 24.0}]
    short = lambda path, start, end: [{"text": "going", "start": 22.5 - start, "end": 22.8 - start}, {"text": "to", "start": 22.9 - start, "end": 23.0 - start},
                                      {"text": "pay", "start": 23.1 - start, "end": 23.4 - start}]
    found = find_missing_speech(SEGMENTS, WORDS, audio, short, progress=lambda m: None, second_pass=second)
    assert found[0]["replacement"].endswith("going to pay cells.")


def test_recovered_speech_that_is_not_english_is_left_out_and_a_failed_check_keeps_everything():
    from types import SimpleNamespace

    from mathula_tv.source_suspects import vet_recovered

    recovered = [{"segment_id": "seg-10", "heard": "Science and technology.", "replacement": "x"},
                 {"segment_id": "seg-10", "heard": "Nalay Yaban Tuanasia in Yusa", "replacement": "y"}]

    class Provider:
        def complete_json(self, **kwargs):
            assert kwargs["payload"]["items"][1]["heard"] == "Nalay Yaban Tuanasia in Yusa"
            return SimpleNamespace(data={"verdicts": [{"id": 0, "verdict": "english_speech"}, {"id": 1, "verdict": "not_english"}]})

    messages = []
    kept = vet_recovered(Provider(), recovered, SEGMENTS, messages.append)
    assert [k["heard"] for k in kept] == ["Science and technology."]
    assert any("Nalay" in m for m in messages)

    class Broken:
        def complete_json(self, **kwargs):
            raise RuntimeError("down")

    assert vet_recovered(Broken(), recovered, SEGMENTS, lambda m: None) == recovered
    assert vet_recovered(Broken(), [], SEGMENTS, lambda m: None) == []


def test_a_window_azure_cannot_transcribe_does_not_stop_the_check(tmp_path):
    audio = tmp_path / "a.wav"
    write_wav(audio, [(22.4, 26.0)])

    def transcribe(path, start, end):
        if end - start < 5:                                  # the tight window: "empty transcription"
            raise RuntimeError("Azure Speech returned an empty transcription")
        return [{"text": "going", "start": 1.4}, {"text": "to", "start": 1.7}, {"text": "pay", "start": 2.0}]

    found = find_missing_speech(SEGMENTS, WORDS, audio, transcribe, progress=lambda m: None)
    assert len(found) == 1 and "going to pay" in found[0]["replacement"]


def test_a_loud_gap_that_yields_no_words_adds_nothing(tmp_path):
    audio = tmp_path / "a.wav"
    write_wav(audio, [(22.4, 26.0)])
    assert find_missing_speech(SEGMENTS, WORDS, audio, lambda p, s, e: [{"text": "uh", "start": 1.0}], progress=lambda m: None) == []
