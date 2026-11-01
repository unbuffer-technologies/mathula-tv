import json

from mathula_tv import lexicon_builder as lb


def test_the_word_list_keeps_english_words_and_letters_not_zulu_words_names_or_initialisms_said_as_words(tmp_path):
    work = tmp_path
    (work / "resources" / "cmudict").mkdir(parents=True)
    (work / "resources" / "attestation" / "lexicon").mkdir(parents=True)
    (work / "resources" / "cmudict" / "cmudict.dict").write_text(
        "million M IH1 L Y AH0 N\nprogramme P R OW1 G R AE2 M\ndurban D ER1 B AH0 N\numa Y UW1 M AH0\nschool S K UW1 L\nor AO1 R\n", encoding="utf-8")
    (work / "resources" / "attestation" / "lexicon" / "nouns.txt").write_text("uma\numuntu\nkanye\n", encoding="utf-8")
    lines = (["Sinezigidi ezingu million ezintathu ."] * 5 + ["Siya e-Durban ."] * 6 + ["Durban yinhle ."] * 3 + ["I-programme entsha ."] * 4
             + ["UMA uyaya ."] * 9 + ["I-ANC ne-EFF ne-COVID ne-PAIA ne-SABC ."] * 7 + ["i-school ."])
    found = lb.collect(work, lines=lines, words_limit=10, initialisms_limit=10)
    assert "million" in found["words"] and "programme" in found["words"] and "school" in found["words"]
    assert "durban" not in found["words"]                                  # written mostly with a capital: a name
    assert "uma" not in found["words"]                                     # a Zulu word
    assert found["words"][:3] == lb.SEED_WORDS[:3]                         # the seed of news words comes first
    assert found["initialisms"][:3] == ["ANC", "EFF", "DA"]
    assert "COVID" not in found["initialisms"] and "PAIA" not in found["initialisms"] and "UMA" not in found["initialisms"]      # said as words, or Zulu
    assert "SABC" in found["initialisms"]
    more = lb.collect(work, lines=["Isiphi ne-OR ne-SCHOOL ."] * 9 + ["Yenza i-Durban ."] * 4, words_limit=10, initialisms_limit=10)
    assert "OR" not in more["initialisms"] and "durban" not in more["words"]              # OR is an English word in capitals; Durban is a name on the exclusion list


def test_a_word_already_right_is_left_alone_otherwise_the_closest_accepted_spelling_is_kept():
    right = [{"spelling": "party", "closeness": 0.8, "accepted": True}, {"spelling": "padi", "closeness": 0.9, "accepted": True}]
    assert lb.decide("party", right) == {"status": "fine as written"}
    tried = [{"spelling": "ai", "closeness": 0.95, "accepted": False}, {"spelling": "eyi-aayi", "closeness": 0.85, "accepted": True},
             {"spelling": "eiai", "closeness": 0.75, "accepted": True}, {"spelling": "ae", "closeness": 0.6, "accepted": True}]
    assert lb.decide("AI", tried) == {"status": "respell", "spelling": "eyi-aayi", "closeness": 0.85}      # the closest of those a listener takes, not the closest overall
    assert lb.decide("AI", [{"spelling": "ai", "closeness": 0.9, "accepted": False}])["status"] == "unresolved"
    assert lb.decide("AI", [{"spelling": "ae", "closeness": 0.5, "accepted": True}])["status"] == "unresolved"       # accepted but too far from the reading


class FakeInterpreter:
    def readings(self, word):
        return [{"how": "letters", "accent": "rhotic", "phones": ["eɪ", "aɪ"]}] if word == "AI" else []


class FakeLab:
    voice = "zu-ZA-ThembaNeural"
    sounds = {"AI": ["aɪ"], "eyi-aayi": ["eɪ", "aɪ"], "eiai": ["eɪ", "j", "aɪ"]}

    def _interpreter(self):
        return FakeInterpreter()

    def _interpreter_spellings(self, word):
        return ["eyi-aayi", "eiai"]

    def voice_reading(self, spelling):
        return list(self.sounds.get(spelling, ["a"]))

    def english_hears(self, spelling):
        return "A." if spelling in ("AI", "eyi-aayi") else "I."                 # the strict listener misses the right one

    def _letters_heard(self, word, heard):
        return list(heard) == ["eɪ", "aɪ"]


def test_a_spelling_the_strict_listener_misses_is_kept_when_the_phone_recogniser_decodes_the_letters():
    record = lb.evaluate(FakeLab(), "AI")
    assert record["status"] == "respell" and record["spelling"] == "eyi-aayi" and record["reads_as"] == "letters"
    assert {t["spelling"]: t["accepted"] for t in record["tried"]} == {"AI": False, "eyi-aayi": True, "eiai": False}
    assert lb.evaluate(FakeLab(), "Nkandla") == {"word": "Nkandla", "status": "no reading"}          # a Zulu name has no reading: left alone


def test_the_forms_of_the_letters_make_an_initialism_with_the_earlier_letters_brief_and_the_last_held():
    records = [{"letter": "A", "tried": [{"spelling": "ey", "accepted": True, "tail_ms": 30}, {"spelling": "eyayi", "accepted": True, "tail_ms": 180}, {"spelling": "eh", "accepted": False, "tail_ms": 20}]},
               {"letter": "N", "tried": [{"spelling": "en", "accepted": True, "tail_ms": 20}]},
               {"letter": "C", "tried": [{"spelling": "sii", "accepted": True, "tail_ms": 60}, {"spelling": "siyi", "accepted": True, "tail_ms": 150}, {"spelling": "si", "accepted": True, "tail_ms": 100}]},
               {"letter": "Q", "tried": [{"spelling": "kyu", "accepted": False, "tail_ms": 40}]}]
    forms = lb.letter_forms(records)
    assert forms == {"A": {"short": "ey", "held": "eyayi"}, "N": {"short": "en", "held": "en"}, "C": {"short": "sii", "held": "siyi"}}      # Q has no spelling a listener takes
    assert lb.compose("ANC", forms) == ["Ey En Siyi", "ey-en-siyi", "Ey En Sii", "ey-en-sii"]
    assert lb.compose("AQ", forms) == [] and lb.compose("", forms) == []                      # a letter without a known form: nothing to compose


def test_a_build_resumes_and_a_merge_stores_the_spellings_for_their_own_voice(tmp_path):
    out = tmp_path / "built_zu-ZA-ThembaNeural_0.jsonl"
    messages = []
    assert lb.build(FakeLab(), ["AI", "Nkandla"], out, progress=messages.append) == 2
    assert lb.build(FakeLab(), ["AI", "Nkandla"], out, progress=messages.append) == 0                 # both are already in the file
    assert "AI: respell -> eyi-aayi" in messages[0] and json.loads(out.read_text().splitlines()[0])["voice"] == "zu-ZA-ThembaNeural"
    from mathula_tv.pronunciation_memory import entries_for

    memory = tmp_path / "memory.json"
    assert lb.merge([out], memory) == {"respell": 1, "no reading": 1}
    entry = entries_for(memory, "zu-ZA-ThembaNeural")[0]
    assert (entry["display_text"], entry["tts_text"], entry["source"]) == ("AI", "eyi-aayi", "lexicon") and entries_for(memory, "zu-ZA-ThandoNeural") == []
