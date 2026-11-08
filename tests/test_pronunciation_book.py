import json

import pytest

from mathula_tv import job_pronunciations as jp
from mathula_tv import pronunciation_book as book
from mathula_tv.english_sounds import EnglishSounds
from mathula_tv.interpreter import Interpreter

THANDO, THEMBA = "zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"


def test_a_committed_spelling_is_kept_per_voice_or_for_both_and_can_be_taken_back(tmp_path):
    path = tmp_path / "book.json"
    book.decide(path, "Jacob", "jeykob", kind="approved", rule_spelling="jeykob", job="dbaf917d")
    book.decide(path, "Heidi", "haydi", kind="approved", voice=THANDO)
    book.decide(path, "Heidi", "haidi", kind="custom", voice=THEMBA, note="Themba says it flatter")
    entries = book.load(path)["entries"]
    assert book.spelling_for(entries, "JACOB", THANDO) == "jeykob" and book.spelling_for(entries, "jacob", THEMBA) == "jeykob"
    assert book.spelling_for(entries, "Heidi", THANDO) == "haydi" and book.spelling_for(entries, "Heidi", THEMBA) == "haidi"
    assert book.spelling_for(entries, "Nobody", THANDO) is None
    assert entries["heidi"]["kind"] == "custom" and entries["jacob"]["jobs"] == ["dbaf917d"]
    assert book.revoke(path, "Jacob") is True and book.revoke(path, "Jacob") is False
    assert [row["action"] for row in book.load(path)["log"]] == ["approved", "approved", "custom", "revoked"]


def test_a_spelling_must_be_plain_letters_and_the_kind_must_be_known(tmp_path):
    for bad in ("", "<script>", "a" * 200, "x;y"):
        with pytest.raises(ValueError):
            book.decide(tmp_path / "b.json", "Jacob", bad, kind="approved")
    with pytest.raises(ValueError):
        book.decide(tmp_path / "b.json", "Jacob", "jeykob", kind="maybe")
    assert not (tmp_path / "b.json").exists()


def test_feedback_names_the_rules_on_which_the_listeners_spelling_departs_and_the_tally_counts_them(tmp_path):
    from mathula_tv import zulu_rules as rules

    path = tmp_path / "book.json"
    found = rules.respell_word(["dʒ", "eɪ", "k", "ə", "b"], [None, 1, None, 0, None], limit=4, written="Jacob")
    chosen = next(r for r in found if r["spelling"] == "jeykab")
    differences = rules.choice_differences(found[0]["choice"], chosen["choice"])
    book.record_feedback(path, word="Jacob", rule_spelling=found[0]["spelling"], chosen="jeykab", differences=differences, job="j1", note="o is not right", english="dʒ eɪ k ə b", reading="word", derivation=["the reduced vowel follows the letter"])
    book.record_feedback(path, word="Cyril", rule_spelling="siril", chosen="siral", differences=differences)
    book.decide(path, "Heidi", "haydi", kind="approved", rule_spelling="haydi")
    book.decide(path, "Cape", "kaphe", kind="custom", rule_spelling="kheyp")
    tally = book.rule_tally(book.load(path))
    assert tally["rules"]["letter_vowel"]["wanted"] == {"0": 2} and sorted(tally["rules"]["letter_vowel"]["words"]["0"]) == ["Cyril", "Jacob"]
    assert tally["approved_as_ruled"] == 1 and tally["custom"] == 1
    with pytest.raises(ValueError):
        book.record_feedback(path, word="x", rule_spelling="a", chosen="<b>")
    row = book.load(path)["feedback"][0]
    assert row["english"] == "dʒ eɪ k ə b" and row["derivation"] and row["diff"][0]["pattern"] == "o→a (inside)"


def _interpreter(tmp_path, dictionary):
    (tmp_path / "cmudict.dict").write_text(dictionary, encoding="utf-8")
    return Interpreter(EnglishSounds.load(tmp_path / "cmudict.dict"), {}, None, None)


def test_the_alternatives_are_the_rules_own_variants_with_the_choice_that_made_each(tmp_path):
    ours = _interpreter(tmp_path, "court K AO1 R T\n")
    found = ours.alternatives("Court", limit=5)
    assert found[0]["spelling"] == "khoot" == ours.rules_spelling("Court")["spelling"]
    assert len({a["spelling"] for a in found}) == len(found) == 5 and all(a["derivation"] and a["choice"] is not None for a in found)
    assert ours.alternatives("Nkandla") == []
    assert [a["spelling"] for a in ours.alternatives("AI", limit=3)][0] == "eyi-aayi"


def test_a_job_lexicon_speaks_what_the_listener_committed_over_what_the_rules_write(tmp_path):
    ours = _interpreter(tmp_path, "court K AO1 R T\nheidi HH AY1 D IY0\n")
    job = tmp_path / "work" / "jobs" / "j1"
    (job / "native_dub").mkdir(parents=True)
    path = tmp_path / "work" / "pronunciation_book.json"
    book.decide(path, "Court", "koort", kind="custom", rule_spelling="khoot")
    book.decide(path, "Heidi", "haydi", kind="approved", voice=THANDO, rule_spelling="haydi")
    spelled = [{"word": w, **ours.rules_spelling(w)} for w in ("Court", "Heidi")]
    done = jp.apply_book(spelled, book.load(path)["entries"])
    assert done[0]["by_voice"] == {THANDO: "koort", THEMBA: "koort"} and done[0]["source"] == "ear" and done[0]["decision"] == "custom" and done[0]["rule_spelling"] == "khoot"
    assert done[1]["by_voice"] == {THANDO: "haydi"}
    lexicon = jp.write_lexicon(job / "native_dub" / "pronunciation_names.json", done)
    entries = {e["display_text"]: e for e in json.loads(lexicon.read_text(encoding="utf-8"))["entries"]}
    assert entries["Court"]["tts_text_by_voice"] == {THANDO: "koort", THEMBA: "koort"} and entries["Court"]["source"] == "ear"
    assert entries["Heidi"]["tts_text_by_voice"] == {THANDO: "haydi", THEMBA: "haydi"}          # Themba has no committed spelling: the rules' stands
    assert apply_unchanged(ours)


def apply_unchanged(ours):
    spelled = [{"word": "Court", **ours.rules_spelling("Court")}]
    return jp.apply_book(spelled, {}) == spelled


def test_a_committed_word_the_rules_never_found_is_still_spoken_as_committed_wherever_the_text_writes_it(tmp_path):
    entries = {}
    path = tmp_path / "b.json"
    book.decide(path, "sotswebhu", "sotswebu", kind="custom")
    entries = book.load(path)["entries"]
    done = jp.with_committed_words([], entries, ["Lowo sotswebhu wayeshintshwe.", "uSotswebhu omkhulu"])
    assert [(d["word"], d["by_voice"]) for d in done] == [("sotswebhu", {THANDO: "sotswebu", THEMBA: "sotswebu"})]
    assert jp.with_committed_words([], entries, ["Nothing of the kind here."]) == []


def test_a_spelling_is_compared_with_the_rules_by_syllable_so_the_same_change_in_two_words_is_one_pattern(tmp_path):
    assert book.shape("aaar") == "V3r" and book.shape("aarr") == "V2rr" and book.shape("khoot") == "khV2t"
    assert book.spelling_diff("vi-eyi-aaar", "vi-eyi-aarr") == [{"from": "aaar", "to": "aarr", "shape": "V3r → V2rr", "where": "last syllable", "pattern": "V3r → V2rr (last syllable)"}]
    assert book.spelling_diff("jeykob", "jeykob") == [] and book.spelling_diff("a-b", "a-b-c")[0]["where"] == "whole word"
    path = tmp_path / "book.json"
    book.record_feedback(path, word="VAR", rule_spelling="vi-eyi-aaar", chosen="vi-eyi-aarr")
    book.record_feedback(path, word="SAR", rule_spelling="es-eyi-aaar", chosen="es-eyi-aarr")
    book.record_feedback(path, word="Cape", rule_spelling="kheyp", chosen="kheypi")
    found = book.patterns(book.load(path))
    assert found[0]["pattern"] == "V3r → V2rr (last syllable)" and found[0]["words"] == ["VAR", "SAR"] and found[0]["examples"][0] == "VAR: vi-eyi-aaar → vi-eyi-aarr"
    book.record_feedback(path, word="VAR", rule_spelling="vi-eyi-aaar", chosen="vi-eyi-aarr", note="again")          # the same lesson: updated, not repeated
    assert len(book.load(path)["feedback"]) == 3


def test_a_book_written_when_there_were_exceptions_reads_them_as_custom_spellings(tmp_path):
    path = tmp_path / "book.json"
    path.write_text(json.dumps({"entries": {"cape": {"word": "Cape", "kind": "exception", "spelling": {"*": "kaphe"}}}}), encoding="utf-8")
    assert book.load(path)["entries"]["cape"]["kind"] == "custom"
    book.decide(path, "Court", "koort", kind="exception")
    assert book.load(path)["entries"]["court"]["kind"] == "custom"


def test_a_comma_or_full_stop_in_a_spelling_is_dropped_because_the_voice_reads_it_as_a_pause(tmp_path):
    path = tmp_path / "book.json"
    entry = book.decide(path, "Congress", "khoonggres,", kind="approved", rule_spelling="khoonggres,")
    assert entry["spelling"] == {"*": "khoonggres"} and book.load(path)["log"][-1]["spelling"] == "khoonggres"
    assert book.decide(path, "Dr", "dookta.", kind="custom")["spelling"] == {"*": "dookta"} and book._valid_spelling("a , b.") == "a b"
    with pytest.raises(ValueError):
        book.decide(path, "x", " , . ", kind="approved")                                         # nothing is left of it
    book.record_feedback(path, word="Cape", rule_spelling="kheyp", chosen="kheyp,")
    assert book.load(path)["feedback"][-1]["chosen"] == "kheyp"


def test_a_book_written_with_commas_reads_clean_and_is_cleaned_once_on_disk(tmp_path):
    path = tmp_path / "book.json"
    path.write_text(json.dumps({"schema": book.SCHEMA, "entries": {"congress": {"word": "Congress", "kind": "approved", "rule_spelling": "khoonggres,", "jobs": [],
                                                                               "spelling": {THANDO: "khoonggres,", THEMBA: "khoonggres,"}},
                                                                    "court": {"word": "Court", "kind": "approved", "spelling": {"*": "khoot"}}},
                                "feedback": [], "log": []}), encoding="utf-8")
    entries = book.load(path)["entries"]
    assert book.spelling_for(entries, "Congress", THANDO) == "khoonggres" and entries["congress"]["rule_spelling"] == "khoonggres"          # clean even before it is rewritten
    assert book.clean(path) == 2 and book.clean(path) == 0                                                                                   # two spellings were changed, once
    on_disk = json.loads(path.read_text())
    assert on_disk["entries"]["congress"]["spelling"][THEMBA] == "khoonggres" and on_disk["entries"]["court"]["spelling"] == {"*": "khoot"} and on_disk["log"][-1]["action"] == "cleaned"
    assert book.clean(tmp_path / "missing.json") == 0
