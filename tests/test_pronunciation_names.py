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


def test_a_reading_saved_in_lowercase_still_finds_the_english_word_glued_to_a_zulu_prefix():
    dictionary = _dictionary().with_job_overrides("job-1", [
        {"display_text": "science", "spoken_text": "science", "tts_text": "sayensi", "kind": "english_code_switch"}])
    assert dictionary.apply("ungqongqoshe weScience and Technology").tts_text == "ungqongqoshe wesayensi and Technology"     # capitalised in the text
    assert dictionary.apply("Phela iScience kakhulu").tts_text == "Phela isayensi kakhulu"
    assert dictionary.apply("Phela science kakhulu").tts_text == "Phela sayensi kakhulu"                                # a word on its own
    assert dictionary.apply("ukuphela isifundoscience").tts_text == "ukuphela isifundoscience"                               # lowercase glue: not this word


def test_an_initialism_with_a_lowercase_zulu_prefix_glued_on_is_respelled_only_when_written_in_capitals():
    dictionary = _dictionary().with_job_overrides("job-1", [
        {"display_text": "JB", "spoken_text": "JB", "tts_text": "jeyi-biii", "kind": "initials"}])
    assert dictionary.apply("zeziphikiso eJB Marks zalungiswa").tts_text == "zeziphikiso ejeyi-biii Marks zalungiswa"
    assert dictionary.apply("ukuthi JB uyafika").tts_text == "ukuthi jeyi-biii uyafika"
    assert dictionary.apply("Ngithe ejb nakwajbu").tts_text == "Ngithe ejb nakwajbu"                  # lowercase glue: another word


def test_a_researched_name_to_be_said_as_written_does_not_shadow_the_readings_chosen_for_its_words():
    from mathula_tv.pronunciation import PronunciationEntry, with_web_researched_organisation_pronunciations

    base = with_default_organisation_initialisms(PronunciationDictionary(dictionary_version="v1", language="zu-ZA", job_id="job"))

    def research(tts_text):
        return {"accepted_corrections": [{"canonical_text": "Tshela Thupa", "entity_type": "organisation", "confidence": 0.94, "pronunciation_mode": "word_name",
                                          "pronunciation_confidence": 0.9, "pronunciation_tts_text": tts_text, "grounded_pronunciation_evidence_urls": ["https://example.test/a"],
                                          "grounded_source_urls": ["https://example.test/a"]}]}

    words = [PronunciationEntry(display_text=w, spoken_text=w, tts_text=t, source="names_stage", reviewed_by="automatic", confidence=0.9, kind="english_code_switch")
             for w, t in (("Tshela", "Tsela"), ("Thupa", "Tupa"))]
    as_written = with_web_researched_organisation_pronunciations(base, research("Tshela Thupa"))
    assert as_written.with_job_overrides("job", [*as_written.job_overrides, *words]).apply("i-Tshela Thupa").tts_text == "i-Tsela-Tupa"
    respelt = with_web_researched_organisation_pronunciations(base, research("Tsela Tupa"))
    assert any(e.display_text == "Tshela Thupa" for e in respelt.job_overrides)                  # a real respelling of the phrase is still kept


def test_a_spelling_the_rules_or_the_ear_wrote_replaces_every_older_reading_of_its_word(tmp_path):
    # the default initialism entries (ANC is spelt out by the voice), a model's respelling from web research and a name with aliases all lose to it
    base = _dictionary().with_job_overrides("job-1", [
        {"display_text": "Western Cape High Court", "spoken_text": "Western Cape High Court", "tts_text": "Westen Keyp Hay Khout",
         "kind": "organisation_name", "source": "web_research", "aliases": ["Western", "Cape", "Court"]}])
    path = tmp_path / NAMES_FILE
    path.write_text(json.dumps({"schema": "x", "voice": "v", "entries": [
        {"display_text": "ANC", "tts_text": "ANC", "tts_text_by_voice": {"v": "eyi-en-siii"}, "kind": "english_code_switch", "source": "rules"},
        {"display_text": "Western", "tts_text": "Western", "tts_text_by_voice": {"v": "westen"}, "kind": "english_code_switch", "source": "rules"},
        {"display_text": "Court", "tts_text": "Court", "tts_text_by_voice": {"v": "khoot"}, "kind": "english_code_switch", "source": "rules"},
        {"display_text": "sotswebhu", "tts_text": "sotswebhu", "tts_text_by_voice": {"v": "sotswebu"}, "kind": "english_code_switch", "source": "ear"},
    ]}), encoding="utf-8")
    assert "anc" in approved_names(base)          # the old filter would have dropped it
    merged = merge_into_dictionary(base, path, "job-1")
    assert merged.apply("i-ANC ikhulume", voice="v").tts_text == "i-eyi-en-siii, ikhulume"          # the comma is the existing separator between a name ending in a vowel and a word starting with one
    assert merged.apply("Western Cape High Court", voice="v").tts_text == "westen Cape High khoot"          # the researched respelling is gone; words with no entry stay as written
    assert merged.apply("osotswebhu", voice="v").tts_text == "osotswebhu" or "sotswebu" in merged.apply("sotswebhu", voice="v").tts_text


def test_a_lexicon_that_is_not_the_rules_or_the_ears_still_leaves_researched_readings_alone(tmp_path):
    base = _dictionary().with_job_overrides("job-1", [
        {"display_text": "Moses Mabhida", "spoken_text": "Moses Mabhida", "tts_text": "Moh-zez Mabhida", "kind": "personal_name", "source": "web_research"}])
    path = write_lexicon(tmp_path / NAMES_FILE, [{"name": "Sibonelo", "result": "Sibonelo", "changed": True, "decisions": []}], voice="v", kinds={})
    merged = merge_into_dictionary(base, path, "job-1")
    assert merged.apply("Moses Mabhida").tts_text == "Moh-zez Mabhida"


def test_the_words_of_one_english_name_written_together_are_joined_so_they_flow():
    entries = [{"display_text": w, "spoken_text": w, "tts_text": t, "kind": "english_code_switch"} for w, t in
               (("North", "nooth"), ("West", "west"), ("University", "yuuniveesiti"), ("science", "sayensi"))]
    d = _dictionary().with_job_overrides("job-1", entries)
    assert d.apply("sinoDkt Shopola eNorth West University kuphela").tts_text == "sinoDokotela Shopola enooth-west-yuuniveesiti kuphela"          # the prefix e stays outside
    assert d.apply("sihlala North West.").tts_text == "sihlala nooth-west."
    assert d.apply("North, West").tts_text == "nooth, west"                         # punctuation between them: two phrases
    assert d.apply("North  West").tts_text == "nooth  west"                         # not exactly one space
    assert d.apply("North science West").tts_text == "nooth-sayensi-west"           # two English words side by side are one compound, capital or not ("panic attack")
    assert d.apply("ngasebenzisa science ne-North").tts_text == "ngasebenzisa sayensi ne-nooth"          # a Zulu word or a prefix between them: two terms
    assert d.apply("science ngiyabonga North").tts_text == "sayensi ngiyabonga nooth"
