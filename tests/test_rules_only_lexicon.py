import json

from mathula_tv import lexicon_builder as lb
from mathula_tv.english_sounds import EnglishSounds
from mathula_tv.interpreter import Interpreter

DICTIONARY = "court K AO1 R T\nparty P AA1 R T IY0\nheidi HH AY1 D IY0\n"


def interpreter(tmp_path, researched=None):
    (tmp_path / "cmudict.dict").write_text(DICTIONARY, encoding="utf-8")
    return Interpreter(EnglishSounds.load(tmp_path / "cmudict.dict"), {}, None, researched)


def test_a_word_is_spelled_by_rule_alone_from_its_main_reading(tmp_path):
    ours = interpreter(tmp_path, {"heidi": {"how": "researched", "accent": "researched", "phones": ["h", "aɪ", "d", "i"], "stress": [None, 1, None, 0]}})
    heidi = ours.rules_spelling("Heidi")
    assert heidi["spelling"] == "haydi" and heidi["reading"] == "researched" and heidi["derivation"]
    court = ours.rules_spelling("Court")
    assert court["accent"] == "non_rhotic" and court["spelling"] == "khoot"          # South African English has no r; the long vowel is written long
    assert ours.rules_spelling("ANC")["reading"] == "letters" and ours.rules_spelling("AI")["spelling"] == "eyi-aayi"
    assert ours.rules_spelling("Nkandla") is None                                   # a Zulu name has no reading: left as written


def test_the_rules_only_build_writes_a_file_per_voice_that_a_merge_reads_with_nothing_spoken_or_scored(tmp_path):
    ours = interpreter(tmp_path)
    voices = ["zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"]
    directory = tmp_path / "lexicon"
    assert lb.build_rules(ours, ["AI", "Court", "Nkandla"], voices, directory) == {"spelled": 2, "no reading": 1}
    lines = [json.loads(line) for line in (directory / "built_zu-ZA-ThembaNeural_rules.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["status"] for line in lines] == ["respell", "respell", "no reading"]
    assert lines[0]["spelling"] == "eyi-aayi" and lines[0]["closeness"] is None and lines[0]["voice"] == "zu-ZA-ThembaNeural" and lines[0]["derivation"]
    from mathula_tv.pronunciation_memory import entries_for

    memory = tmp_path / "memory.json"
    assert lb.merge([directory / f"built_{v}_rules.jsonl" for v in voices], memory) == {"respell": 4, "no reading": 2}
    assert {e["display_text"]: e["tts_text"] for e in entries_for(memory, "zu-ZA-ThandoNeural")} == {"AI": "eyi-aayi", "Court": "khoot"}
