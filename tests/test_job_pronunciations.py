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


def test_a_zulu_surname_the_english_dictionary_happens_to_hold_is_left_alone_because_zulu_text_writes_it_often():
    """Khumalo, Cele and Buthelezi are in the English dictionary (an English reading of a Zulu name), and a name is capitalised inside a sentence, so they were respelled as
    English ("siil", "kyumaalo").  What tells whose they are is how often Zulu text writes them; English first names it hardly writes at all."""
    usage = {"khumalo": 239, "cele": 54, "buthelezi": 136, "mchunu": 39, "andrea": 0, "fannie": 0, "johnson": 34, "kenny": 0, "nelson": 75, "africa": 287, "centre": 128}
    kept = {"khumalo": .61, "cele": .41, "buthelezi": .63, "nelson": .46, "johnson": .61, "africa": .05, "centre": .08}          # measured on the corpus pairs
    text = ["uBheki Cele noMnu Khumalo noButhelezi, no-Andrea Johnson noFannie Kenny uNelson eSouth Africa e-Centre"]
    found = jp.candidates(text, researched={"cele": {}, "khumalo": {}}, dictionary=lambda w: w in usage, zulu_count=lambda w: usage.get(w, 0), zulu_kept=lambda w: kept.get(w, 0))
    assert found == ["Andrea", "Johnson", "Fannie", "Kenny", "Nelson", "Africa", "Centre"]          # Cele and Khumalo are Zulu; an English name (Nelson, Johnson: not Zulu syllables) or a translated word (Africa, Centre: Zulu text rewrites it) is not
    assert jp.zulu_name_by_usage("Mchunu", 39) is False and jp.zulu_name_by_usage("Ngcobo", 78) and jp.zulu_name_by_usage("Mkhwanazi", 60)          # a name below the usage floor is not decided on usage; clicks and clusters are Zulu


def test_a_zulu_name_is_left_alone_even_when_web_research_gave_it_english_sounds_but_an_english_name_is_not():
    found = jp.candidates(["uJacob Zuma noCyril Ramaphosa Nomvalo Heidi Mchunu"], researched={"zuma": {}, "ramaphosa": {}, "nomvalo": {}, "heidi": {}, "jacob": {}, "mchunu": {}},
                          dictionary=lambda w: w in {"jacob", "heidi"}, zulu_count=lambda w: 0)
    assert found == ["Jacob", "Heidi"]          # Mchunu has a click letter (c): still a Zulu name
    assert jp.candidates(["i-AI ne-MK"], researched={}, dictionary=lambda w: False, zulu_count=lambda w: 0) == ["AI", "MK"]          # initialisms are not names


def test_an_initialism_with_a_zulu_prefix_glued_on_is_found_without_the_prefix():
    found = jp.candidates(["Ezinye zeziphikiso eJB Marks zalungiswa, kuSABC nakwa-IEC"], researched={}, dictionary=lambda w: False, zulu_count=lambda w: 0)
    assert found == ["JB", "SABC", "IEC"]
    assert jp.candidates(["iOS"], researched={"ios": {}}, dictionary=lambda w: False, zulu_count=lambda w: 0) == ["iOS"]          # a word the research knows stays whole


def test_a_word_the_listener_asked_for_is_listed_although_the_rules_would_leave_it_alone():
    text = ["Unobhala we-ANC, uLouis Diremelo, uthi sebefake eMahikeng"]
    assert jp.candidates(text, researched={}, dictionary=lambda w: False, zulu_count=lambda w: 0) == ["ANC"]                          # a Zulu-shaped name: left alone
    assert jp.candidates(text, researched={}, dictionary=lambda w: False, zulu_count=lambda w: 0, also_list=["diremelo"]) == ["ANC", "Diremelo"]


def test_the_letters_after_a_digit_are_not_a_word(tmp_path):
    from mathula_tv.job_pronunciations import candidates

    found = candidates(["Ngomhla we-4th kuNovemba, ngo-22nd, ne-3rd, nango-1st. Sawubona Hello."], researched={}, dictionary=lambda w: w in {"th", "nd", "rd", "st", "hello"},
                       zulu_count=lambda w: 0)
    assert "th" not in found and "nd" not in found and "rd" not in found and "st" not in found and "Hello" in found


def test_a_day_written_after_its_month_gets_the_spelling_of_its_ordinal():
    assert jp.ordinals(["Kusukela ngoNovemba 4, kuningi", "kuze kube uDisemba 1,", "ngoJanuwari 5, 2027", "kuNovemba 13 kanye"]) == ["4th", "1st", "5th", "13th"]
    assert jp.ordinals(["ngoNovemba", "ngoDisemba 2024", "R4 Novemba"]) == []             # a month alone, a year and a day before its month are not "month day"



def test_a_lowercase_word_the_listener_asked_for_is_listed_as_written():
    from types import SimpleNamespace

    prepared = SimpleNamespace(also_list=["ngevoti"], researched={}, lessons={}, unsure=[], resolved=[], said=None, zulu_count=lambda w: 0, zulu_kept=lambda w: 0.0,
                               sounds=SimpleNamespace(word=lambda w: False), interpreter=SimpleNamespace(rules_spelling=lambda w: None))
    spelled = jp.spell(prepared, ["Abavoti bakhulumile ngevoti, banquma ukunganiki muntu iningi."], progress=lambda m: None)
    assert [(s["word"], s["spelling"], s["reading"]) for s in spelled] == [("ngevoti", "ngevoti", "as written")]
