import json

from mathula_tv.pronunciation import PronunciationDictionary, with_default_organisation_initialisms
from mathula_tv.pronunciation_names import (
    NAMES_FILE, approved_names, load_names_entries, merge_into_dictionary, phones_to_spelling, plan_name, write_lexicon,
)
from mathula_tv.pronunciation_search import looks_isizulu


def test_the_rules_reproduce_every_spelling_a_listener_approved():
    # Jacob Zuma / MK Party: the respellings chosen by ear are what the rules give for how the speaker said them.
    assert phones_to_spelling(["dʒ", "eɪ", "k", "o", "b"], like="Jacob")[0] == "Jeykhob"
    assert phones_to_spelling(["ɛ", "m"], like="Em")[0] == "Em"
    assert phones_to_spelling(["k", "eɪ"], like="Khey")[0] == "Khey"
    assert phones_to_spelling(["p", "ɑː", "t", "i"], like="Party")[0] == "Phaathi"


def test_a_stop_is_aspirated_before_a_vowel_but_not_after_s_or_at_the_end():
    assert phones_to_spelling(["s", "t", "ɑː", "t"], like="Start")[0] == "Staat"
    assert phones_to_spelling(["k", "u", "k"])[0] == "khuk"                  # aspirated onset, plain final


def test_unconfirmed_rules_are_reported_so_a_listener_knows_what_was_never_checked():
    spelling, unconfirmed = phones_to_spelling(["ʃ", "ɑː", "n"], like="Shaan")
    assert spelling == "Shaan" and "ʃ->sh" in unconfirmed


class FakeScorer:
    """Scores a word by how close its spelling is to the target spelling the voice needs."""

    def __init__(self, targets):
        self.targets = targets

    def word_scores(self, spelling):
        import difflib

        return [difflib.SequenceMatcher(None, w.lower(), t).ratio() for w, t in zip(spelling.split(), self.targets)]


def test_the_english_word_is_respelt_and_the_isizulu_word_is_left_alone():
    tokens = {"dʒeɪkob": ["dʒ", "eɪ", "k", "o", "b"], "zuma": ["z", "u", "m", "a"]}
    plan = plan_name("Jacob Zuma", ["dʒeɪkob", "zuma"], FakeScorer(["jeykhob", "zuma"]), lambda ipa: tokens[ipa],
                     looks_isizulu=looks_isizulu)
    assert plan["result"] == "Jeykhob Zuma" and plan["changed"]
    jacob, zuma = plan["decisions"]
    assert jacob["changed"] and jacob["rule_score"] >= jacob["plain_score"] + 0.15
    assert zuma["changed"] is False and "isiZulu shape" in zuma["why"]


def test_a_respelling_that_does_not_beat_the_plain_reading_is_not_used():
    tokens = {"mowzəz": ["m", "oʊ", "z", "ə", "z"]}
    plan = plan_name("Moses", ["mowzəz"], FakeScorer(["moses"]), lambda ipa: tokens[ipa], looks_isizulu=looks_isizulu)
    assert plan["result"] == "Moses" and not plan["changed"]


def _dictionary():
    return with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="job-1"))


def test_a_name_a_person_approved_is_never_overridden_by_the_automatic_stage(tmp_path):
    path = write_lexicon(tmp_path / NAMES_FILE, [
        {"name": "Jacob Zuma", "result": "Jeicob Zuma", "changed": True, "decisions": []},        # approved: "Jeykhob Zuma"
        {"name": "Moses Mabhida", "result": "Mowzes Mabhida", "changed": True, "decisions": []},
        {"name": "Samkele Maseko", "result": "Samkele Maseko", "changed": False, "decisions": []},
    ], voice="zu-ZA-ThembaNeural", kinds={})
    assert "jacob zuma" in approved_names(_dictionary())
    merged = merge_into_dictionary(_dictionary(), path, "job-1")
    assert merged.apply("uJacob Zuma ukhulumile").tts_text == "uJeykhob Zuma ukhulumile"          # the person's choice stands
    assert merged.apply("kuMoses Mabhida").tts_text == "kuMowzes Mabhida"                         # the automatic one applies
    assert merged.apply("uSamkele Maseko").tts_text == "uSamkele Maseko"                          # unchanged names add nothing


def test_the_lexicon_replaces_a_researched_reading_of_the_same_name(tmp_path):
    base = _dictionary().with_job_overrides("job-1", [
        {"display_text": "Moses Mabhida", "spoken_text": "Moses Mabhida", "tts_text": "Moh-zez Mabhida", "kind": "personal_name"}])
    path = write_lexicon(tmp_path / NAMES_FILE, [{"name": "Moses Mabhida", "result": "Mowzes Mabhida", "changed": True,
                                                  "decisions": []}], voice="v", kinds={})
    merged = merge_into_dictionary(base, path, "job-1")
    assert merged.apply("Moses Mabhida").tts_text == "Mowzes Mabhida"
    assert len([o for o in merged.job_overrides if o.display_text == "Moses Mabhida"]) == 1


def test_a_missing_or_empty_lexicon_changes_nothing(tmp_path):
    assert load_names_entries(tmp_path / "absent.json") == []
    dictionary = _dictionary()
    assert merge_into_dictionary(dictionary, tmp_path / "absent.json", "job-1") is dictionary


def test_the_jobs_dictionary_picks_the_lexicon_up_for_both_measurement_and_rendering(tmp_path):
    from mathula_tv.native_dub import _apply_native_pronunciation_research_to_job_cache, _native_dub_tts_ready_text

    native = tmp_path / "native_dub"
    native.mkdir()
    write_lexicon(native / NAMES_FILE, [{"name": "Moses Mabhida", "result": "Mowzes Mabhida", "changed": True, "decisions": []}],
                  voice="v", kinds={})
    _apply_native_pronunciation_research_to_job_cache(tmp_path, {"accepted_corrections": []})
    assert _native_dub_tts_ready_text("Kuthiwe uMoses Mabhida ubekhona", job_root=tmp_path) == "Kuthiwe uMowzes Mabhida ubekhona"
    assert json.loads((native / NAMES_FILE).read_text(encoding="utf-8"))["examined"][0]["name"] == "Moses Mabhida"


def test_an_acronym_is_left_to_the_initialism_rules_and_an_existing_respelling_is_kept():
    tokens = {"ɛseɪbiːsiː": ["ɛ", "s", "eɪ", "b", "iː", "s", "iː"], "njuːz": ["n", "j", "uː", "z"], "məʊzɪz": ["m", "əʊ", "z", "ɪ", "z"]}
    plan = plan_name("SABC News", ["ɛseɪbiːsiː", "njuːz"], FakeScorer(["eseybiisii", "nyuuz"]), lambda ipa: tokens[ipa],
                     looks_isizulu=looks_isizulu, current=["SABC", "News"])
    assert plan["decisions"][0]["why"].startswith("acronym") and plan["decisions"][0]["decision"] == "SABC"
    kept = plan_name("Moses", ["məʊzɪz"], FakeScorer(["mowzes"]), lambda ipa: tokens[ipa], looks_isizulu=looks_isizulu,
                     current=["Moh-zez"])
    assert kept["result"] == "Moh-zez" and kept["changed"] is False and "already respelt" in kept["decisions"][0]["why"]


def test_a_respelling_must_beat_the_current_reading_not_just_the_raw_spelling():
    tokens = {"dʒeɪkob": ["dʒ", "eɪ", "k", "o", "b"]}
    plan = plan_name("Jacob", ["dʒeɪkob"], FakeScorer(["jeykhob"]), lambda ipa: tokens[ipa], looks_isizulu=looks_isizulu,
                     current=["Jacob"], margin=0.15)
    assert plan["result"] == "Jeykhob" and plan["changed"]


def test_a_joint_recogniser_symbol_is_spelt_as_its_separate_sounds():
    spelling, _ = phones_to_spelling(["mj", "u", "n", "ɪ"], like="Municipality")
    assert spelling == "Myuni"
