import json
from pathlib import Path

from mathula_tv import media
from mathula_tv.pronunciation_evidence import speaker_sounds, word_spans
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
