from mathula_tv.pronunciation_lean import lean_reading, verdict
from mathula_tv.pronunciation_search import looks_isizulu

TOKENS = {"dʒeɪkob": ["dʒ", "eɪ", "k", "o", "b"], "zuma": ["z", "u", "m", "a"], "məʊzəz": ["m", "əʊ", "z", "ə", "z"]}


def test_the_lean_reading_leaves_isizulu_words_and_acronyms_alone_and_spells_the_rest_by_rule():
    reading = lean_reading("Jacob Zuma", ["dʒeɪkob", "zuma"], ["Jeyikab", "Zuma"], lambda ipa: TOKENS[ipa], looks_isizulu=looks_isizulu)
    assert [w["lean"] for w in reading] == ["Jeykhob", "Zuma"]
    assert reading[1]["why"].startswith("isiZulu shape")
    acronym = lean_reading("SABC", ["ɛs"], ["Es Ey Bii Sii"], lambda ipa: ["ɛ", "s"], looks_isizulu=looks_isizulu)
    assert acronym[0]["lean"] == "Es Ey Bii Sii" and acronym[0]["why"].startswith("acronym")


def test_the_lean_reading_does_not_use_the_research_respelling_for_an_english_word():
    reading = lean_reading("Moses", ["məʊzəz"], ["Mowziz"], lambda ipa: TOKENS[ipa], looks_isizulu=looks_isizulu)
    assert reading[0]["lean"] != "Mowziz" and reading[0]["why"] == "rule from the speaker's sounds"


def test_a_verdict_needs_a_clear_gap():
    assert verdict(0.3, 0.5) == "lean better" and verdict(0.5, 0.3) == "current better"
    assert verdict(0.40, 0.45) == "about equal"


def test_the_names_lexicon_is_written_from_the_speakers_sounds_with_no_scoring(tmp_path, monkeypatch):
    import json

    from mathula_tv import pronunciation_evidence as pe
    from mathula_tv.pronunciation_lean import speaker_names

    (tmp_path / "native_dub").mkdir()
    (tmp_path / "native_dub" / "pronunciation_research.json").write_text(json.dumps({"accepted_corrections": [
        {"canonical_text": "Floyd Shivambu", "pronunciation_ipa": "/flɔɪd ʃivambu/", "entity_type": "person",
         "pronunciation_tts_text": "Floyid Shivambu"}]}), encoding="utf-8")

    class Gop:
        _vocab = {"f": 1, "l": 2, "ɔɪ": 3, "d": 4, "ʃ": 5, "i": 6, "v": 7, "a": 8, "m": 9, "b": 10, "u": 11, "eɪ": 12, "<pad>": 0}

        def _load(self):
            pass

    monkeypatch.setattr(pe, "speaker_sounds", lambda *a, **k: {"words": ["fleɪd", "ʃivambu"], "missing": [], "evidence": []})
    path = speaker_names(tmp_path, gop=Gop(), progress=lambda message: None)
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    assert [(e["display_text"], e["tts_text"], e["kind"]) for e in entries] == [("Floyd Shivambu", "Fleyd Shivambu", "personal_name")]
