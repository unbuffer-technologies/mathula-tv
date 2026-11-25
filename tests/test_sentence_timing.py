import wave
from pathlib import Path

import numpy as np

from mathula_tv import sentence_timing as st

RATE = 16000


def tone(seconds):
    return (4000 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * RATE)) / RATE)).astype(np.int16)


def quiet(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.int16)


def write(path, parts):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1), out.setsampwidth(2), out.setframerate(RATE), out.writeframes(np.concatenate(parts).tobytes())
    return path


def words(spans):
    """Source words: (text, start, end) in seconds from the unit's start at 100 s; a word ending in a full stop closes a sentence."""
    return [{"text": t, "start": 100.0 + a, "end": 100.0 + b} for t, a, b in spans]


ITEM = {"item_id": "u1", "start_ms": 100000, "end_ms": 120000, "speaker_id": "S", "zulu_text": "Isiqeshana sokuqala sinde kakhulu. Isiqeshana sesibili. Isiqeshana sesithathu."}


def test_a_sentence_that_stays_behind_the_speakers_after_the_renderer_has_done_its_best_is_reported(tmp_path):
    # Zulu: sentence 1 speaks 8 s, then a 0.92 s full stop, sentence 2 speaks 4 s, full stop, sentence 3 speaks 4 s
    audio = write(tmp_path / "u1.wav", [tone(8.0), quiet(0.92), tone(4.0), quiet(0.92), tone(4.0)])
    # the speaker's sentences begin at 5.0 s and 10.0 s (each held 0.3 s), over 15 s of speech
    spoken = words([("one", 0.0, 4.9), ("stop.", 4.9, 5.0), ("two", 5.0, 9.9), ("stop.", 9.9, 10.0), ("three", 10.0, 15.0)])
    found = st.check_unit({**ITEM}, audio, spoken, shortfall_ms=0, chars_per_s=12.0)
    assert found is not None and found["behind_the_mouth"][0]["sentence"] == 2
    assert found["behind_the_mouth"][0]["take_out_about_chars"] > 12 and found["behind_the_mouth"][0]["opens"].startswith("Isiqeshana sesibili")


def test_a_unit_whose_sentences_start_with_the_speakers_has_nothing_to_report(tmp_path):
    audio = write(tmp_path / "u1.wav", [tone(5.0), quiet(0.92), tone(4.0), quiet(0.92), tone(4.0)])
    spoken = words([("one", 0.0, 4.9), ("stop.", 4.9, 5.0), ("two", 5.0, 9.9), ("stop.", 9.9, 10.0), ("three", 10.0, 15.0)])
    assert st.check_unit({**ITEM}, audio, spoken, chars_per_s=12.0) is None
    assert st.check_unit({**ITEM}, tmp_path / "missing.wav", spoken) is None


def test_openings_name_the_sentences_after_the_first():
    assert st.sentence_openings("Okokuqala kwenzekile. Okwesibili ngoba kunjalo. ~pause900 Okwesithathu nje.") == ["Okwesibili ngoba kunjalo.", "Okwesithathu nje."]
