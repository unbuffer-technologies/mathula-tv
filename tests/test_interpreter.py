from mathula_tv.english_sounds import EnglishSounds
from mathula_tv.interpreter import Interpreter, makeable
from mathula_tv.phonology import Phonology

DICTIONARY = "heidi HH AY1 D IY0\nparty P AA1 R T IY0\n"
TABLE = {"graphemes": {letters: [{"sounds": sounds, "seen": 3, "of": 3}] for letters, sounds in {
    "h": ["h"], "ai": ["aɪ"], "ay": ["aɪ"], "d": ["d"], "di": ["d", "i"], "i": ["i"], "ii": ["iː"], "p": ["p"], "ph": ["p"], "a": ["a"], "aa": ["aː"],
    "t": ["t"], "r": ["r"], "e": ["e"]}.items()}}
SPEAKERS = Phonology({"map": {
    "ɑː": [{"zulu": "a", "n": 49, "share": 0.55}, {"zulu": "ɑː", "n": 39, "share": 0.45}], "ɹ": [{"zulu": "r", "n": 72, "share": 0.8}, {"zulu": "ɹ", "n": 20, "share": 0.2}],
    "ø": [{"zulu": "ø", "n": 5, "share": 0.1}]}})


def interpreter(tmp_path, speakers=SPEAKERS):
    path = tmp_path / "cmudict.dict"
    path.write_text(DICTIONARY, encoding="utf-8")
    return Interpreter(EnglishSounds.load(path), TABLE, speakers)


def test_the_voice_inventory_is_every_sound_some_letter_group_makes():
    assert {"h", "aɪ", "d", "i", "iː", "p", "a", "aː", "t", "r", "e"} <= makeable(TABLE)


def test_the_aim_is_the_english_sound_when_the_voice_can_make_it(tmp_path):
    speakers = Phonology({"map": {"aɪ": [{"zulu": "i", "n": 37, "share": 0.37}, {"zulu": "aɪ", "n": 30, "share": 0.30}]}})        # Zulu readers of names say i first
    aim = interpreter(tmp_path, speakers).aim_for("aɪ")
    assert aim["aim"] == "aɪ" and aim["choices"][0]["distance"] == 0.0 and aim["why"] == "the voice makes this sound"


def test_a_sound_the_voice_cannot_make_is_weighed_between_the_chart_and_what_zulu_speakers_choose(tmp_path):
    aim = interpreter(tmp_path).aim_for("ɹ")                                                    # the voice makes the tap, not the English r
    assert aim["aim"] == "r" and aim["why"].startswith("the voice cannot")
    far = Phonology({"map": {"ø": [{"zulu": "e", "n": 9, "share": 0.9}, {"zulu": "i", "n": 1, "share": 0.1}]}})
    assert interpreter(tmp_path, far).aim_for("ø")["aim"] == "e"                                 # both near; the one Zulu speakers choose wins
    unknown = interpreter(tmp_path, speakers=None).aim_for("ɛ")
    assert unknown["aim"] in {"e", "a", "i"} and unknown["choices"][0]["zulu_speakers_say"] is None     # the nearest sound the voice has


def test_the_recognisers_symbols_are_looked_up_in_the_corpus_s_symbols(tmp_path):
    speakers = Phonology({"map": {"i": [{"zulu": "i", "n": 90, "share": 0.9}, {"zulu": "e", "n": 10, "share": 0.1}]}})      # the corpus writes the KIT vowel ɪ as i
    aim = interpreter(tmp_path, speakers).aim_for("ɪ")          # the table below has no ɪ, so the voice cannot make it: the corpus's i (90%) is what is weighed
    assert aim["aim"] == "i" and aim["choices"][0]["zulu_speakers_say"] == "i" and aim["choices"][0]["share"] == 0.9


def test_a_sound_the_voice_cannot_make_is_aimed_at_the_nearest_one_it_can(tmp_path):
    made, away = interpreter(tmp_path, speakers=None).nearest_makeable("ɛ")
    assert made in {"e", "a", "i"} and 0 < away < 0.6


def test_a_word_is_spelled_from_its_dictionary_sounds_and_keeps_its_derivation(tmp_path):
    out = interpreter(tmp_path).spellings("Heidi")
    assert out["readings"][0]["phones"] == ["h", "aɪ", "d", "iː"]
    best = out["spellings"][0]
    assert best["aim"][0] == "h" and best["spelling"].startswith("h")
    assert best["reading"] == "word" and best["english"] == ["h", "aɪ", "d", "iː"]


def test_no_sound_of_a_dictionary_reading_is_left_out_as_a_noisy_edge(tmp_path):
    from mathula_tv import letter_sounds

    table = {"graphemes": {"a": [{"sounds": ["a"], "seen": 3, "of": 3}], "i": [{"sounds": ["i"], "seen": 3, "of": 3}]}}
    assert "i" in dict(letter_sounds.candidates(["a", "i"], table, limit=200))                                   # audio: the first sound may go
    assert "i" not in dict(letter_sounds.candidates(["a", "i"], table, limit=200, noisy_edges=False))          # a dictionary's: it may not
    assert "ai" in dict(letter_sounds.candidates(["a", "i"], table, limit=200, noisy_edges=False))


def test_a_letter_by_letter_spelling_gets_variants_that_hold_its_last_letter():
    assert Interpreter.held_variants("eiensi") == ["eiensiyi"]
    assert Interpreter.held_variants("Ey Ai".replace(" ", "")) == ["eyaayi", "eyaiyi"]                    # the letter I held: aayi
    assert Interpreter.held_variants("eiensii") == ["eiensiiyi"] and Interpreter.held_variants("ef") == []


def test_the_dictionary_note_shows_the_sounds_only_the_decode_has_and_how_english_stresses_letters(tmp_path):
    (tmp_path / "cmudict.dict").write_text(DICTIONARY + "anc AE1 N K\n", encoding="utf-8")
    interpreter_ = Interpreter(EnglishSounds.load(tmp_path / "cmudict.dict"), TABLE, SPEAKERS)
    note = interpreter_.dictionary_note("ANC", ["eː", "ɛ", "n", "s", "iː", "k", "o"])                       # the decode of ANC picked up "k o" from the next word
    assert note["reads_as"] == "letters" and note["sounds"] == "eɪ ɛ n s iː" and note["stress"] == "eɪ(2) ɛ(2) n s iː(1)"
    assert any("k o" in extra for extra in note["only_in_the_speakers_decode"]) and "next word" in note["so"]
    assert "LAST letter" in note["how_english_says_it"]
    assert interpreter_.dictionary_note("eNkandla", ["ɛ", "n", "k", "a"]) is None                         # a Zulu name has no dictionary reading


def test_a_word_the_dictionary_does_not_have_is_left_alone(tmp_path):
    out = interpreter(tmp_path).spellings("eNkandla")
    assert out["readings"] == [] and out["spellings"] == []


def test_the_table_row_for_a_sound_lists_the_letters_that_make_its_aim(tmp_path):
    rows = interpreter(tmp_path).table_rows(["ɑː", "ɹ"])
    assert rows[0]["aim"] == "a" and "a" in rows[0]["letters_the_voice_reads_so"]
    assert rows[1]["aim"] == "r" and rows[1]["letters_the_voice_reads_so"] == ["r"]
