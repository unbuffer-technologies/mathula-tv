import json
from pathlib import Path

from mathula_tv import media
from mathula_tv.pronunciation_evidence import consensus, heard_word, speaker_sounds, word_spans, word_windows
from mathula_tv.pronunciation_search import looks_isizulu

WORDS = [
    {"text": "The", "start": 1.0, "end": 1.2},
    {"text": "Etegini", "start": 1.2, "end": 1.8},
    {"text": "Municipality,", "start": 2.0, "end": 2.9},
    {"text": "Moses", "start": 3.0, "end": 3.4},
    {"text": "municipality", "start": 8.0, "end": 8.8},
]


class FakeGop:
    def __init__(self):
        self.calls = []

    def choose_expected(self, clips, group):
        self.calls.append((len(clips), group))
        return {"chosen": group + "!", "base": group, "chosen_excess": 0.1, "base_excess": 0.4}


def make_job(tmp_path: Path) -> Path:
    (tmp_path / "analysis").mkdir()
    (tmp_path / "audio").mkdir()
    (tmp_path / "analysis" / "azure_diarization.json").write_text(json.dumps({"words": WORDS}), encoding="utf-8")
    (tmp_path / "audio" / "analysis_mono.wav").write_bytes(b"RIFF")
    return tmp_path


def test_a_word_is_found_on_its_own_even_when_the_whole_name_is_not_in_the_transcript():
    assert word_spans(WORDS, "Municipality") == [(2.0, 2.9), (8.0, 8.8)]
    assert word_spans(WORDS, "Mabhida") == []


def test_occurrences_are_spread_across_the_recording_when_there_are_many():
    many = [{"text": "party", "start": float(i), "end": i + 0.4} for i in range(10)]
    assert word_spans(many, "Party", limit=3) == [(0.0, 0.4), (4.0, 4.4), (9.0, 9.4)]


def test_the_speakers_own_occurrences_decide_each_english_word_and_zulu_words_keep_their_form(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "extract_clip", lambda source, start, end, out, **kw: out)
    gop = FakeGop()
    heard = speaker_sounds(make_job(tmp_path), "Moses Mabhida", "moses mabida", gop, tmp_path / "w", looks_isizulu=looks_isizulu)
    assert heard["words"] == ["moses!", "mabida"] and heard["missing"] == []
    assert gop.calls == [(1, "moses")]


def test_a_name_with_an_english_word_the_speaker_never_says_has_no_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "extract_clip", lambda source, start, end, out, **kw: out)
    heard = speaker_sounds(make_job(tmp_path), "Floyd Shivambu", "floid shivambu", FakeGop(), tmp_path / "w", looks_isizulu=looks_isizulu)
    assert heard["words"] is None and heard["missing"] == ["Floyd"]


def test_a_word_may_reach_before_its_start_but_never_into_the_word_before_it():
    words = [{"text": "Good", "start": 0.0, "end": 0.5}, {"text": "OK", "start": 0.62, "end": 0.9}, {"text": "sir", "start": 0.95, "end": 1.2},
             {"text": "so", "start": 5.0, "end": 5.2}, {"text": "OK", "start": 5.5, "end": 5.8}, {"text": "now", "start": 7.0, "end": 7.2}]
    first, second = word_windows(words, "OK")
    assert first == (0.62, 0.9, 0.5, 0.95)                 # the previous word's end and the next word's start bound it
    assert second[2] == 5.5 - 0.20 and second[3] == 5.8 + 0.12


def test_the_first_sounds_of_a_word_that_the_transcript_starts_late_are_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "extract_clip", lambda source, start, end, out, **kw: out)
    words = [{"text": "Good", "start": 0.0, "end": 0.5}, {"text": "OK", "start": 1.0, "end": 1.3}, {"text": "sir", "start": 1.6, "end": 1.9}]

    class Gop:
        def decode_timed(self, clip):          # the clip starts 0.25 s before the span; times are in the clip
            return [("g", 0.0, 0.06), ("oʊ", 0.12, 0.2), ("k", 0.33, 0.4), ("eɪ", 0.42, 0.55), ("s", 0.8, 0.9)]

    heard = heard_word(tmp_path / "a.wav", words, "OK", Gop(), tmp_path / "w")
    assert heard["phones"] == ["oʊ", "k", "eɪ"]            # "oʊ" sits 0.09 s before the transcript's start and is kept; Good and sir are not


def test_a_truncated_decoding_does_not_win_the_consensus_over_fuller_ones():
    heard = [["aɪ", "d", "iː"], ["d", "iː"], ["h", "eː", "l", "d", "i"]]
    assert consensus(heard) == ["aɪ", "d", "iː"]
    assert consensus([["p", "ɑː", "t", "iː"]]) == ["p", "ɑː", "t", "iː"]


def test_sounds_that_have_no_letter_in_the_word_are_not_a_reference_for_it():
    from mathula_tv.pronunciation_evidence import reference_fits_spelling

    assert "do not fit its letters" in reference_fits_spelling("High", ["p", "aː", "k"])           # prod-51: the transcript's time for High was decoded as "park"
    for word, phones in (("Heidi", ["aɪ", "d", "iː", "x"]), ("Party", ["p", "ɑː", "t", "iː", "æ"]), ("Court", ["k", "oː", "t", "d"]), ("SG", ["ɪ", "s", "tʃ", "iː", "ɪ"]),
                         ("Western", ["ɪ", "s", "ə", "n"]), ("Cyril", ["s", "ø", "r", "r", "a"]), ("Litchfield", ["ɪ", "tʃ", "f", "iː", "l", "d", "æ"])):
        assert reference_fits_spelling(word, phones) is None, word
    assert reference_fits_spelling("Hi", ["h", "aɪ"]) is None and reference_fits_spelling("X", ["p"]) is None      # too few consonants to judge
