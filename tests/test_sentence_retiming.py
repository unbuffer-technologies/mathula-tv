import wave
from pathlib import Path

import numpy as np

from mathula_tv import delivery, sentence_retiming as sr

RATE = 16000


def _unit(path: Path, parts):
    """parts: ('speech', seconds) or ('silence', seconds) -> a mono 16-bit wav."""
    chunks = []
    t = 0
    for kind, seconds in parts:
        n = int(seconds * RATE)
        if kind == "speech":
            chunks.append((6000 * np.sin(2 * np.pi * 220 * (np.arange(n) + t) / RATE)).astype(np.int16))
        else:
            chunks.append(np.zeros(n, dtype=np.int16))
        t += n
    with wave.open(str(path), "wb") as w:
        w.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        w.writeframes(np.concatenate(chunks).tobytes())
    return path


def _read(path: Path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def _onsets(x):
    runs = sr._boundaries(x, RATE, min_s=0.4)                       # after the move a gap may be shorter than the 0.8 s that marks a sentence end in fresh speech
    return [b / RATE for _, b in runs]


WORDS = [{"text": "one.", "start": 10.0, "end": 12.0}, {"text": "two.", "start": 12.6, "end": 14.5}, {"text": "three", "start": 15.5, "end": 16.7}]          # a unit window 10.0 - 17.0 s


def test_sentence_gaps_move_toward_the_source_onsets_and_the_length_and_the_speech_are_kept(tmp_path):
    src = _unit(tmp_path / "unit.wav", [("speech", 2.0), ("silence", 0.95), ("speech", 2.0), ("silence", 0.95), ("speech", 1.5), ("silence", 0.3)])
    out = tmp_path / "retimed.wav"
    report = sr.retime(src, out, WORDS, 10000, 17000)
    assert report["applied"] and report["matched"] == 2
    before, after = _read(src), _read(out)
    assert len(before) == len(after)                                                    # the unit keeps exactly its length
    assert np.count_nonzero(before) == np.count_nonzero(after)                         # not one sample of speech is cut, added or changed
    assert report["onset_error_s"]["after"] < report["onset_error_s"]["before"]
    targets = [2.6, 5.5]                                                                # the source's sentence starts, from the unit's start
    now = _onsets(after)
    assert abs(now[0] - targets[0]) < abs(_onsets(before)[0] - targets[0])
    assert all(0.45 - 0.02 <= (b - a) / RATE <= 2.0 for a, b in sr._boundaries(after, RATE, min_s=0.4))        # a sentence end stays a sentence end


def test_a_unit_that_ends_before_its_window_spends_the_shortfall_at_the_boundaries_never_beyond_the_window(tmp_path):
    words = [{"text": "one.", "start": 10.0, "end": 12.0}, {"text": "two.", "start": 13.5, "end": 15.5}, {"text": "three", "start": 17.4, "end": 18.9}]          # window 10.0 - 19.0 s
    src = _unit(tmp_path / "g.wav", [("speech", 2.0), ("silence", 0.95), ("speech", 2.0), ("silence", 0.95), ("speech", 1.5), ("silence", 0.3)])
    out = tmp_path / "g2.wav"
    report = sr.retime(src, out, words, 10000, 19000, extra_ms=1300)             # the unit is 7.7 s in a 9 s window
    assert report["applied"]
    grown = (len(_read(out)) - len(_read(src))) / RATE
    assert 0.5 < grown <= 1.3 + 0.001                                              # silence is added where the speaker paused, and never past the window
    assert report["onset_error_s"]["after"] < report["onset_error_s"]["before"]
    assert np.count_nonzero(_read(src)) == np.count_nonzero(_read(out))
    shorter = sr.retime(src, tmp_path / "g3.wav", words, 10000, 19000, extra_ms=0)           # with no room it can only move silence about, not add any
    assert not shorter["applied"] or len(_read(tmp_path / "g3.wav")) == len(_read(src))


def test_a_unit_already_where_the_speaker_began_each_sentence_and_a_single_sentence_are_left_alone(tmp_path):
    exact = [{"text": "one.", "start": 10.0, "end": 12.0}, {"text": "two.", "start": 12.95, "end": 14.9}, {"text": "three", "start": 15.85, "end": 17.0}]
    src = _unit(tmp_path / "a.wav", [("speech", 2.0), ("silence", 0.95), ("speech", 2.0), ("silence", 0.95), ("speech", 1.5)])
    assert not sr.retime(src, tmp_path / "a2.wav", exact, 10000, 17000)["applied"] and not (tmp_path / "a2.wav").exists()
    one = _unit(tmp_path / "b.wav", [("speech", 3.0)])
    assert not sr.retime(one, tmp_path / "b2.wav", WORDS, 10000, 17000)["applied"]


def test_a_sentence_that_matches_nothing_cannot_move_a_gap(tmp_path):
    far = [{"text": "one.", "start": 10.0, "end": 10.4}, {"text": "two", "start": 10.5, "end": 16.9}]                 # the only source onset is 0.5 s in: nowhere near either Zulu sentence start
    src = _unit(tmp_path / "c.wav", [("speech", 2.0), ("silence", 0.95), ("speech", 2.0), ("silence", 0.95), ("speech", 1.5)])
    assert not sr.retime(src, tmp_path / "c2.wav", far, 10000, 17000)["applied"]


def test_a_marked_pause_means_the_total_silence_so_a_stop_before_it_is_not_added_to_it():
    pieces = delivery.split_pauses("Masemola. ~pause720 Ake usichazele")
    assert pieces == ["Masemola. Ake usichazele"]                                           # 0.92 s of full stop already is more than the 0.72 s asked for: no break, one text, its space kept ("Masemola.Ake" would lose the stop's silence)
    longer = delivery.split_pauses("Masemola. ~pause1500 Ake")
    assert [p for p in longer if isinstance(p, int)] == [1500 - 920 - 110]                                          # the stop's 0.92 s counts toward the 1.5 s
    after_word = delivery.split_pauses("kodwa ~pause900 Ake")
    assert [p for p in after_word if isinstance(p, int)] == [900 - 110]
    after_comma = delivery.split_pauses("Masemola, ~pause720 Ake")
    assert [p for p in after_comma if isinstance(p, int)] == [720 - 150 - 110]
    assert delivery.strip_pauses("Masemola. ~pause720 Ake") == "Masemola. Ake"


def test_a_gap_is_never_made_longer_than_the_voices_own_to_reach_a_later_source_onset(tmp_path):
    # the source speaker's sentences begin later than the Zulu's (3.5 s and 6.5 s in) but they left only 0.5 s between sentences: lengthening the Zulu gaps would only make dead air
    words = [{"text": "one.", "start": 10.0, "end": 13.0}, {"text": "two.", "start": 13.5, "end": 15.9}, {"text": "three", "start": 16.5, "end": 17.9}]
    src = _unit(tmp_path / "d.wav", [("speech", 2.0), ("silence", 0.95), ("speech", 2.0), ("silence", 0.95), ("speech", 1.5), ("silence", 0.3)])
    out = tmp_path / "d2.wav"
    report = sr.retime(src, out, words, 10000, 19000, extra_ms=1300)
    assert not report["applied"] or all(b - a <= 0.95 * RATE + 2 for a, b in sr._boundaries(_read(out), RATE))           # no gap grew past the voice's 0.95 s
    assert not report["applied"] or len(_read(out)) == len(_read(src))


def test_a_gap_may_grow_up_to_the_speakers_own_longer_pause(tmp_path):
    words = [{"text": "one.", "start": 10.0, "end": 12.0}, {"text": "two.", "start": 13.5, "end": 15.5}, {"text": "three", "start": 17.4, "end": 18.9}]          # the speaker paused 1.5 s and 1.9 s
    src = _unit(tmp_path / "e.wav", [("speech", 2.0), ("silence", 0.95), ("speech", 2.0), ("silence", 0.95), ("speech", 1.5), ("silence", 0.3)])
    report = sr.retime(src, tmp_path / "e2.wav", words, 10000, 19000, extra_ms=1300)
    assert report["applied"] and max(report["gap_changes_s"]) > 0.3
