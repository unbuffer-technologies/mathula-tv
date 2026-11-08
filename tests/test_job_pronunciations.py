import json

from mathula_tv import job_pronunciations as jp
from mathula_tv.english_sounds import EnglishSounds
from mathula_tv.interpreter import Interpreter

TEXT = "Sawubona, Heidi. I-MK Party ne-SABC kanye nama-MPs. Ngo-WhatsApp ngithe uJacob Zuma ~pause500 ngomuntu OwayenguNobhala-Jikelele."
ENGLISH = {"heidi", "party", "jacob", "whatsapp", "ngithe"}


COMMON = {"ngithe": 900, "party": 2}


def find(texts, researched=None):
    return jp.candidates(texts, researched=researched or {}, dictionary=lambda w: w in ENGLISH, zulu_count=lambda w: COMMON.get(w, 0))


def test_a_zulu_prefix_before_a_hyphen_is_grammar_not_the_english_word():
    assert [jp.core(t) for t in ("i-MK", "ngo-WhatsApp", "ama-MPs", "OwayenguNobhala-Jikelele", "Heidi")] == ["MK", "WhatsApp", "MPs", "OwayenguNobhala-Jikelele", "Heidi"]


def test_initialisms_researched_words_and_dictionary_words_that_are_not_zulu_are_the_english_of_a_text():
    found = find([TEXT])
    assert found == ["Heidi", "MK", "Party", "SABC", "MPs", "WhatsApp", "Jacob"]          # in order of appearance; ngithe is Zulu, the compound is Zulu
    assert "Zuma" not in found and "Sawubona" not in found
    assert find(["uLitchfield"], researched={"litchfield": {}}) == ["Litchfield"]            # a word the research has sounds for is taken (a Zulu name is not: see below)
    assert find(["Ngithe uma. Uma kuhle."]) == []                                              # Zulu words are not English because the dictionary has a namesake


def test_the_lexicon_holds_only_the_rule_spellings_for_both_voices_and_the_old_file_is_kept_once(tmp_path):
    path = tmp_path / "pronunciation_names.json"
    path.write_text(json.dumps({"entries": [{"display_text": "old"}]}), encoding="utf-8")
    (tmp_path / "cmudict.dict").write_text("heidi HH AY1 D IY0\n", encoding="utf-8")
    ours = Interpreter(EnglishSounds.load(tmp_path / "cmudict.dict"), {}, None, None)
    found = ours.rules_spelling("Heidi")
    jp.write_lexicon(path, [{"word": "Heidi", **found}])
    jp.write_lexicon(path, [{"word": "Heidi", **found}])
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    assert [e["display_text"] for e in entries] == ["Heidi"] and entries[0]["tts_text_by_voice"] == {v: "haydi" for v in jp.VOICES} and entries[0]["source"] == "rules"
    assert json.loads((tmp_path / "pronunciation_names.before_job_rules.json").read_text(encoding="utf-8"))["entries"][0]["display_text"] == "old"


def test_spellings_chosen_by_ear_for_a_job_stand_beside_the_rule_spellings(tmp_path):
    (tmp_path / "native_dub").mkdir()
    (tmp_path / "native_dub" / jp.EAR_FILE).write_text(json.dumps({"sotswebhu": "sotswebu"}), encoding="utf-8")
    assert jp.by_ear(tmp_path) == {"sotswebhu": "sotswebu"} and jp.by_ear(tmp_path / "none") == {}
    path = tmp_path / "native_dub" / "pronunciation_names.json"
    jp.write_lexicon(path, [], by_ear=jp.by_ear(tmp_path))
    entry = json.loads(path.read_text(encoding="utf-8"))["entries"][0]
    assert entry["display_text"] == "sotswebhu" and entry["tts_text_by_voice"] == {v: "sotswebu" for v in jp.VOICES} and entry["source"] == "ear"


def test_a_written_ordinal_is_said_as_the_english_number():
    assert [jp.ordinal_words(n) for n in (1, 2, 3, 5, 12, 20, 22, 31)] == [["one"], ["two"], ["three"], ["five"], ["twelve"], ["twenty"], ["twenty", "two"], ["thirty", "one"]]          # ngo-22nd is "ngo twenty two": ka- takes the cardinal
    assert jp.ordinals(["zango-22nd kule nyanga", "ngo-22nd October", "ku-3rd na-4th", "R22 nd", "2020"]) == ["22nd", "3rd", "4th"]


def test_a_zulu_name_is_left_alone_even_when_web_research_gave_it_english_sounds_but_an_english_name_is_not():
    found = jp.candidates(["uJacob Zuma noCyril Ramaphosa Nomvalo Heidi"], researched={"zuma": {}, "ramaphosa": {}, "nomvalo": {}, "heidi": {}, "jacob": {}},
                          dictionary=lambda w: w in {"jacob", "heidi"}, zulu_count=lambda w: 0)
    assert found == ["Jacob", "Heidi"]
    assert jp.candidates(["i-AI ne-MK"], researched={}, dictionary=lambda w: False, zulu_count=lambda w: 0) == ["AI", "MK"]          # initialisms are not names


def test_the_letters_after_a_digit_are_not_a_word(tmp_path):
    from mathula_tv.job_pronunciations import candidates

    found = candidates(["Ngomhla we-4th kuNovemba, ngo-22nd, ne-3rd, nango-1st. Sawubona Hello."], researched={}, dictionary=lambda w: w in {"th", "nd", "rd", "st", "hello"},
                       zulu_count=lambda w: 0)
    assert "th" not in found and "nd" not in found and "rd" not in found and "st" not in found and "Hello" in found
