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
    assert lb.decide("AI", tried, letters=True) == {"status": "respell", "spelling": "eyi-aayi", "closeness": 0.85, "held": False}      # letters: the closest of those a listener takes, not the closest overall
    assert lb.decide("AI", [{"spelling": "ai", "closeness": 0.9, "accepted": False}], letters=True)["status"] == "unresolved"
    assert lb.decide("AI", [{"spelling": "ae", "closeness": 0.5, "accepted": True}], letters=True)["status"] == "unresolved"       # accepted but too far from the reading


def test_a_word_is_judged_by_how_close_the_voice_is_to_the_dictionary_reading_and_a_listener_cannot_veto_close_sounds():
    cape = [{"spelling": "Cape", "closeness": 0.5, "accepted": False}, {"spelling": "kayp", "closeness": 1.0, "accepted": False},      # the listener wrote "Keep"
            {"spelling": "keip", "closeness": 0.75, "accepted": False}]
    assert lb.decide("Cape", cape) == {"status": "respell", "spelling": "kayp", "closeness": 1.0, "held": False}
    news = [{"spelling": "News", "closeness": 0.38, "accepted": True}, {"spelling": "nuzu", "closeness": 1.0, "accepted": True}]          # the listener knows the word
    assert lb.decide("News", news)["spelling"] == "nuzu"                                          # but the voice does not say it: not fine as written
    assert lb.decide("Court", [{"spelling": "Court", "closeness": 0.88, "accepted": False}]) == {"status": "fine as written"}      # close enough without a listener
    middling = [{"spelling": "Party", "closeness": 0.8, "accepted": False}, {"spelling": "pati", "closeness": 0.8, "accepted": False}]
    assert lb.decide("Party", middling)["status"] == "unresolved"                                  # fairly close sounds need a listener's approval as well


def test_equally_close_spellings_go_to_the_one_tried_first_which_is_the_rules_choice():
    tried = [{"spelling": "AI", "closeness": 1.0, "accepted": False}, {"spelling": "eyi-aayi", "closeness": 1.0, "accepted": True, "tail_ms": 190},
             {"spelling": "eyi-ayi", "closeness": 1.0, "accepted": True, "tail_ms": 150}]
    assert lb.decide("AI", tried, letters=True)["spelling"] == "eyi-aayi"                 # not the shorter spelling
    assert lb.decide("AI", list(reversed(tried[1:])), letters=True)["spelling"] == "eyi-ayi"       # the earlier one wins whichever it is


def test_a_word_said_by_letters_keeps_a_spelling_whose_last_sound_is_held_over_a_closer_brief_one():
    tried = [{"spelling": "Ei Ae", "closeness": 0.9, "accepted": True, "tail_ms": 75}, {"spelling": "ei-aayi", "closeness": 0.88, "accepted": True, "tail_ms": 168},
             {"spelling": "eiai", "closeness": 0.95, "accepted": False, "tail_ms": 200}]
    assert lb.decide("AI", tried)["spelling"] == "eiai"                                                   # a word that is not said by letters: the closest, and 0.95 needs no listener
    assert lb.decide("AI", tried, letters=True) == {"status": "respell", "spelling": "ei-aayi", "closeness": 0.88, "held": True}
    brief = [{"spelling": "Ei Ae", "closeness": 0.9, "accepted": True, "tail_ms": 75}]
    assert lb.decide("AI", brief, letters=True) == {"status": "respell", "spelling": "Ei Ae", "closeness": 0.9, "held": False}      # none held: the brief one, marked


class FakeInterpreter:
    def readings(self, word):
        return [{"how": "letters", "accent": "rhotic", "phones": ["eɪ", "aɪ"]}] if word == "AI" else []

    def rule_spellings(self, word, readings):
        return [{"spelling": "eiai"}]


class _Ends:
    tails = {"eyi-aayi": 170, "eiai": 60}

    def get(self, audio):
        return self.tails.get(audio, 50) / 1000


class FakeLab:
    voice = "zu-ZA-ThembaNeural"
    sounds = {"AI": ["aɪ"], "eyi-aayi": ["eɪ", "aɪ"], "eiai": ["eɪ", "j", "aɪ"]}
    _word_end = _Ends()

    def _audio_for(self, spelling):
        return spelling

    def _word_timed(self, audio):
        return [("x", 0.0, 0.0)]

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
    assert {t["spelling"]: t["tail_ms"] for t in record["tried"] if t["accepted"]} == {"eyi-aayi": 170} and record["held"] is True    # how long the last sound is held is measured


def test_an_earlier_build_is_decided_again_under_the_length_rule_without_asking_the_listeners():
    earlier = {"word": "AI", "voice": "zu-ZA-ThembaNeural", "reads_as": "letters", "status": "respell", "spelling": "eiai", "closeness": 0.9,
               "tried": [{"spelling": "AI", "closeness": 1.0, "accepted": False}, {"spelling": "eiai", "closeness": 0.9, "accepted": True},
                         {"spelling": "eyi-aayi", "closeness": 0.85, "accepted": True}]}
    fine = {"word": "TV", "voice": "zu-ZA-ThembaNeural", "reads_as": "letters", "status": "fine as written", "tried": [{"spelling": "TV", "closeness": 1.0, "accepted": True}]}
    words = {"word": "party", "reads_as": "word", "status": "respell", "spelling": "paati", "closeness": 0.9, "tried": []}
    changed = lb.redecide(FakeLab(), [earlier, fine, words])
    assert [(r["word"], r["spelling"], r["held"]) for r in changed] == [("AI", "eyi-aayi", True)]      # TV is already right, a word is not said by letters
    assert changed[0]["voice"] == "zu-ZA-ThembaNeural"


def test_the_forms_of_the_letters_make_an_initialism_with_the_earlier_letters_brief_and_the_last_held():
    records = [{"letter": "A", "tried": [{"spelling": "ey", "accepted": True, "tail_ms": 30}, {"spelling": "eyayi", "accepted": True, "tail_ms": 180}, {"spelling": "eh", "accepted": False, "tail_ms": 20}]},
               {"letter": "N", "tried": [{"spelling": "en", "accepted": True, "tail_ms": 20}]},
               {"letter": "C", "tried": [{"spelling": "sii", "accepted": True, "tail_ms": 60}, {"spelling": "siyi", "accepted": True, "tail_ms": 150}, {"spelling": "si", "accepted": True, "tail_ms": 100}]},
               {"letter": "Q", "tried": [{"spelling": "kyu", "accepted": False, "tail_ms": 40}]}]
    forms = lb.letter_forms(records)
    assert forms == {"A": {"short": "ey", "held": "eyayi"}, "N": {"short": "en", "held": "en"}, "C": {"short": "sii", "held": "siyi"}}      # Q has no spelling a listener takes
    assert lb.compose("ANC", forms) == ["Ey En Siyi", "ey-en-siyi", "Ey En Sii", "ey-en-sii"]
    assert lb.compose("AQ", forms) == [] and lb.compose("", forms) == []                      # a letter without a known form: nothing to compose


def test_a_word_that_ended_in_an_error_is_tried_again_and_the_retry_replaces_it(tmp_path):
    out = tmp_path / "built_zu-ZA-ThembaNeural_0.jsonl"

    class Down(FakeLab):
        def voice_reading(self, spelling):
            raise RuntimeError("synthesis failed: Authentication error (401)")

    assert lb.build(Down(), ["AI"], out, progress=lambda m: None) == 1
    assert json.loads(out.read_text().splitlines()[0])["status"] == "error"
    assert lb.build(FakeLab(), ["AI"], out, progress=lambda m: None) == 1                               # the service is back: the word is built, not skipped
    assert lb.build(FakeLab(), ["AI"], out, progress=lambda m: None) == 0                               # and now it is done
    from mathula_tv.pronunciation_memory import entries_for

    memory = tmp_path / "memory.json"
    assert lb.merge([out], memory) == {"respell": 1}                                                    # the error line is superseded by the later result
    assert [e["tts_text"] for e in entries_for(memory, "zu-ZA-ThembaNeural")] == ["eyi-aayi"]


def test_a_later_merge_replaces_the_lexicons_own_spelling_but_never_an_agents(tmp_path):
    from mathula_tv.pronunciation_memory import entries_for, remember

    memory = tmp_path / "memory.json"
    voice = "zu-ZA-ThembaNeural"
    remember(memory, "AI", "Ei Ae", source="lexicon", voice=voice, job_id="lexicon", closeness=0.9)
    remember(memory, "HIV", "Eitsh Ayi Vii", source="agent", voice=voice, job_id="job", closeness=0.9)
    out = tmp_path / "built.jsonl"
    rows = [{"word": "AI", "voice": voice, "status": "respell", "spelling": "ei-aayi", "closeness": 0.85},
            {"word": "HIV", "voice": voice, "status": "respell", "spelling": "hiv-lexicon", "closeness": 0.99}]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    lb.merge([out], memory)
    assert {e["display_text"]: e["tts_text"] for e in entries_for(memory, voice)} == {"AI": "ei-aayi", "HIV": "Eitsh Ayi Vii"}      # less close, but the lexicon's latest choice; the agent's stays


def test_letter_words_built_under_another_method_are_built_again_and_those_built_under_this_one_are_not(tmp_path):
    out = tmp_path / "built_zu-ZA-ThembaNeural_0.jsonl"
    old = {"word": "AI", "voice": "zu-ZA-ThembaNeural", "reads_as": "letters", "status": "respell", "spelling": "ei-ae", "closeness": 0.7, "tried": []}
    out.write_text(json.dumps(old) + "\n", encoding="utf-8")
    assert lb.build(FakeLab(), ["AI"], out, progress=lambda m: None) == 0                                    # an ordinary build leaves it
    assert lb.build(FakeLab(), ["AI"], out, progress=lambda m: None, again_letters=True) == 1                # the method changed: built again
    assert lb.build(FakeLab(), ["AI"], out, progress=lambda m: None, again_letters=True) == 0                # and not a third time
    assert json.loads(out.read_text().splitlines()[-1])["build"] == lb.BUILD


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
