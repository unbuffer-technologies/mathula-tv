"""A commit in the console reaches every job's lexicon at once, a revoke takes it back, and nothing else is touched."""
from __future__ import annotations

import json
from pathlib import Path

from mathula_tv import job_pronunciations as jp
from mathula_tv import pronunciation_book as book

THANDO, THEMBA = "zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"


def _lexicon(work: Path, job: str, entries: list[dict]) -> Path:
    path = work / "jobs" / job / "native_dub" / "pronunciation_names.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": "mathula-names-lexicon-v1", "voice": THANDO, "entries": entries}), encoding="utf-8")
    return path


def _rule(word: str, spelling: str) -> dict:
    return {"display_text": word, "tts_text": word, "tts_text_by_voice": {THANDO: spelling, THEMBA: spelling}, "kind": "english_code_switch", "confidence": 0.9, "source": "rules",
            "reads_as": "word", "english": "x", "derivation": []}


def _spelling(path: Path, word: str) -> dict:
    return next(e for e in json.loads(path.read_text())["entries"] if e["display_text"] == word)


def test_a_committed_spelling_reaches_every_jobs_lexicon_and_only_the_words_committed_change(tmp_path):
    work = tmp_path / "work"
    a = _lexicon(work, "job-a", [_rule("Buffalo", "baafalo"), _rule("Court", "khoot")])
    b = _lexicon(work, "job-b", [_rule("Buffalo", "baafalo")])
    c = _lexicon(work, "job-c", [_rule("Court", "khoot")])
    untouched = c.read_text()
    book.decide(work / "pronunciation_book.json", "Buffalo", "bhafalo", kind="custom", rule_spelling="baafalo")
    assert jp.refresh_lexicons(work) == {"job-a": ["Buffalo"], "job-b": ["Buffalo"]}
    for path in (a, b):
        item = _spelling(path, "Buffalo")
        assert item["tts_text_by_voice"] == {THANDO: "bhafalo", THEMBA: "bhafalo"} and item["source"] == "ear" and item["decision"] == "custom" and item["rule_spelling"] == "baafalo"
    assert _spelling(a, "Court")["tts_text_by_voice"][THANDO] == "khoot" and c.read_text() == untouched          # a job without the word is not even rewritten
    assert jp.refresh_lexicons(work) == {}                                                                      # and a lexicon that is already right is left alone


def test_a_spelling_for_one_voice_leaves_the_other_voice_on_the_rules(tmp_path):
    work = tmp_path / "work"
    path = _lexicon(work, "job-a", [_rule("Joburg", "joobeeg")])
    book.decide(work / "pronunciation_book.json", "Joburg", "joobeg", kind="custom", voice=THEMBA, rule_spelling="joobeeg")
    jp.refresh_lexicons(work)
    assert _spelling(path, "Joburg")["tts_text_by_voice"] == {THANDO: "joobeeg", THEMBA: "joobeg"}


def test_a_new_commit_over_an_old_one_and_a_revoke_go_back_to_the_rules_own_spelling(tmp_path):
    work = tmp_path / "work"
    path = _lexicon(work, "job-a", [_rule("Senate", "senit")])
    book_path = work / "pronunciation_book.json"
    book.decide(book_path, "Senate", "seeneit", kind="custom", rule_spelling="senit")
    jp.refresh_lexicons(work)
    book.decide(book_path, "Senate", "seenet", kind="custom", rule_spelling="senit")
    assert jp.refresh_lexicons(work) == {"job-a": ["Senate"]} and _spelling(path, "Senate")["tts_text_by_voice"][THANDO] == "seenet"
    assert book.revoke(book_path, "Senate") is True
    assert jp.refresh_lexicons(work) == {"job-a": ["Senate"]}
    item = _spelling(path, "Senate")
    assert item["tts_text_by_voice"] == {THANDO: "senit", THEMBA: "senit"} and item["source"] == "rules" and "decision" not in item and "rule_spelling" not in item


def test_a_lexicon_that_does_not_belong_to_the_rules_is_left_alone(tmp_path):
    work = tmp_path / "work"
    by_ear = {"display_text": "sotswebhu", "tts_text": "sotswebhu", "tts_text_by_voice": {THANDO: "sotsweb", THEMBA: "sotsweb"}, "kind": "english_code_switch", "confidence": 0.9, "source": "ear"}
    path = _lexicon(work, "job-a", [by_ear])
    book.decide(work / "pronunciation_book.json", "sotswebhu", "sotsewbu", kind="custom")
    before = path.read_text()
    assert jp.refresh_lexicons(work) == {} and path.read_text() == before
    (work / "jobs" / "job-bad" / "native_dub").mkdir(parents=True)
    (work / "jobs" / "job-bad" / "native_dub" / "pronunciation_names.json").write_text("{not json", encoding="utf-8")
    assert jp.refresh_lexicons(work) == {}                                                                      # an unreadable lexicon is skipped, not a failure


def test_the_lexicon_written_by_a_build_keeps_the_rules_spelling_beside_a_commit(tmp_path):
    spelled = jp.apply_book([{"word": "Buffalo", "spelling": "baafalo", "reading": "word", "english": ["b"], "derivation": []}],
                            {"buffalo": {"word": "Buffalo", "kind": "custom", "spelling": {"*": "bhafalo"}}})
    path = jp.write_lexicon(tmp_path / "lex.json", spelled)
    item = json.loads(path.read_text())["entries"][0]
    assert item["rule_spelling"] == "baafalo" and item["decision"] == "custom" and item["tts_text_by_voice"][THANDO] == "bhafalo"
