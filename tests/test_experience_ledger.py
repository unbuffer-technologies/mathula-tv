import json
from pathlib import Path

from mathula_tv import experience_ledger as ledger
from mathula_tv import language_questions as lq

JOB = "33294c3d7de9455b880c5dea4b15c3fa"
CLIP = JOB[:8]


def _run(work: Path, run_id: str, rows: list[dict], items: list[dict]) -> Path:
    directory = work / "experience" / "runs" / run_id / CLIP / "r1"
    directory.mkdir(parents=True)
    (directory / "thinking.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (directory / "submission.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    (directory / "agent_summary.json").write_text(json.dumps({"turns": 11, "model": "gpt-test"}), encoding="utf-8")
    (work / "jobs" / JOB).mkdir(parents=True, exist_ok=True)
    return directory


ROWS = [
    {"submission": 1, "kind": "choice", "unit": "unit-001", "english": "impalas", "zulu": "izimpala", "alternatives": ["ama-impala"], "why": "attested", "evidence": "attested_zulu found it"},
    {"submission": 1, "kind": "choice", "unit": "unit-001", "english": "the Commission", "zulu": "iKhomishani", "alternatives": [], "why": "kept the repetition",
     "evidence": "Set aside unsupported_doubling because the source repeats it"},
    {"submission": 1, "kind": "choice", "unit": "unit-001", "english": "a payment", "zulu": "inkokhelo yemali", "alternatives": [], "why": "w", "evidence": "e"},
    {"submission": 1, "kind": "strategy", "unit": "unit-001", "strategy": "retold compactly"},
    {"submission": 1, "kind": "missing_evidence", "unit": "unit-001", "wanted": "the garbled words after 'third party'", "why": "both readings disagree"},
]
ITEMS = [{"item_id": "unit-001", "start_ms": 0, "end_ms": 20000, "zulu_text": "Kwakukhona izimpala. iKhomishani ithi kuyaqhubeka."}]          # "inkokhelo yemali" is not in the final text


def test_a_runs_records_are_ingested_with_the_outcome_its_final_text_shows(tmp_path):
    _run(tmp_path, "prod-1", ROWS, ITEMS)
    assert ledger.ingest_run(tmp_path, job_id=JOB, run_id="prod-1", clip_id=CLIP) == {"entries_in_run": 5, "added": 5, "ledger": 5}
    rows = {(r["kind"], r.get("english") or r.get("unit")): r for r in ledger.load(tmp_path)}
    assert rows[("choice", "impalas")]["outcome"] == "kept" and rows[("choice", "a payment")]["outcome"] == "not_in_final"
    assert rows[("choice", "the Commission")]["set_aside_evidence"] is True and rows[("choice", "impalas")]["set_aside_evidence"] is False
    assert rows[("strategy", "unit-001")]["outcome"] == "recorded"
    missing = next(r for r in ledger.load(tmp_path) if r["kind"] == "missing_evidence")
    assert missing["outcome"] == "open" and missing["source"] == "agent_inferred" and missing["model"] == "gpt-test" and missing["job"] == JOB


def test_ingesting_a_run_again_adds_nothing_and_a_new_run_adds_its_own(tmp_path):
    _run(tmp_path, "prod-1", ROWS, ITEMS)
    ledger.ingest_run(tmp_path, job_id=JOB, run_id="prod-1", clip_id=CLIP)
    assert ledger.ingest_run(tmp_path, job_id=JOB, run_id="prod-1", clip_id=CLIP)["added"] == 0
    _run(tmp_path, "prod-2", ROWS, ITEMS)
    assert ledger.ingest_run(tmp_path, job_id=JOB, run_id="prod-2", clip_id=CLIP)["added"] == 5 and len(ledger.load(tmp_path)) == 10


def test_an_answer_to_the_agents_question_marks_its_notes_answered_in_every_run_and_in_runs_ingested_later(tmp_path):
    _run(tmp_path, "prod-1", ROWS, ITEMS)
    ledger.ingest_run(tmp_path, job_id=JOB, run_id="prod-1", clip_id=CLIP)
    key = next(r["key"] for r in ledger.load(tmp_path) if r["kind"] == "missing_evidence")
    lq.answer_evidence(tmp_path / "jobs" / JOB, tmp_path, key, wanted="the garbled words after 'third party'", unit="unit-001", text="that is also a meeting place")
    first = next(r for r in ledger.load(tmp_path) if r["kind"] == "missing_evidence")
    assert first["outcome"] == "answered" and first["outcome_note"] == "that is also a meeting place"
    _run(tmp_path, "prod-2", ROWS, ITEMS)
    ledger.ingest_run(tmp_path, job_id=JOB, run_id="prod-2", clip_id=CLIP)
    second = [r for r in ledger.load(tmp_path) if r["kind"] == "missing_evidence" and r["run"] == "prod-2"]
    assert second and second[0]["outcome"] == "answered"                                  # the same question, answered once, is answered for the next run too


def test_an_entry_can_be_marked_contradicted_and_a_wrong_outcome_is_refused(tmp_path):
    _run(tmp_path, "prod-1", ROWS, ITEMS)
    ledger.ingest_run(tmp_path, job_id=JOB, run_id="prod-1", clip_id=CLIP)
    entry = next(r for r in ledger.load(tmp_path) if r.get("english") == "impalas")
    assert ledger.mark(tmp_path, entry["id"], "contradicted", "the listener wrote ama-impala") and ledger.mark(tmp_path, "nope", "kept") is False
    assert next(r for r in ledger.load(tmp_path) if r["id"] == entry["id"])["outcome"] == "contradicted"
    try:
        ledger.mark(tmp_path, entry["id"], "great")
    except ValueError as error:
        assert "outcome must be one of" in str(error)
    else:
        raise AssertionError("an unknown outcome was accepted")


WORDS = [{"text": t, "start": s, "end": s + 0.4} for t, s in [("although", 100.0), ("he", 100.5), ("made", 101.0), ("a", 101.4), ("payment", 101.8), ("to", 102.5), ("a", 102.8), ("third", 103.0), ("party", 103.5),
                                                              ("that", 104.2), ("is", 104.5), ("all", 104.8), ("says", 105.2), ("in", 105.6), ("the", 105.9), ("meat", 106.2), ("place", 106.7), ("bank", 130.0)]]


def test_the_stretch_an_evidence_note_asks_about_is_found_from_the_words_it_quotes():
    begin, end, said = lq.locate_in_source("the garbled words immediately after 'third party'", WORDS, 90.0, 140.0)
    assert 101.9 < begin < 102.1 and end == 103.0 + lq.EVIDENCE_WINDOW_S and "meat place" in said and "bank" not in said          # one second before "third", nine seconds on
    assert lq.locate_in_source("the words after 'fourth party'", WORDS, 90.0, 140.0) is None
    assert lq.locate_in_source("the words after 'third party'", WORDS, 0.0, 50.0) is None                                          # outside the unit


def test_the_console_questions_come_from_the_runs_notes_with_a_window_and_an_answer_once_given(tmp_path):
    directory = _run(tmp_path, "prod-1", ROWS, [{"item_id": "unit-001", "start_ms": 90000, "end_ms": 140000, "zulu_text": "x"}])
    questions = lq.evidence_questions(tmp_path / "jobs" / JOB, directory, [{"item_id": "unit-001", "start_ms": 90000, "end_ms": 140000, "zulu_text": "x"}], WORDS)
    assert len(questions) == 1 and questions[0]["kind"] == "evidence" and questions[0]["located"] and questions[0]["answer"] is None and questions[0]["start_ms"] > 101000
    key = questions[0]["id"]
    lq.answer_evidence(tmp_path / "jobs" / JOB, tmp_path, key, wanted=questions[0]["wanted"], unit="unit-001", text="a meeting place or business place")
    again = lq.evidence_questions(tmp_path / "jobs" / JOB, directory, [{"item_id": "unit-001", "start_ms": 90000, "end_ms": 140000, "zulu_text": "x"}], WORDS)
    assert again[0]["answer"]["text"] == "a meeting place or business place"
    notes = lq.notes_for_agent(tmp_path / "jobs" / JOB)
    assert notes == [{"you_asked_for": questions[0]["wanted"], "listener_says": "a meeting place or business place"}]
    assert lq.load_answers(tmp_path / "jobs" / JOB)["stretches"] == {}                    # evidence answers and stretch answers share a file and do not disturb each other


def test_an_answer_needs_text_or_a_comment_and_a_real_key(tmp_path):
    (tmp_path / "jobs" / JOB).mkdir(parents=True)
    for bad in ({"key": "nonsense"}, {"key": "0123456789abcdef"}):
        try:
            lq.answer_evidence(tmp_path / "jobs" / JOB, tmp_path, bad["key"], text="")
        except ValueError:
            continue
        raise AssertionError("an empty or unknown answer was accepted")


def test_the_same_stretch_asked_about_in_two_runs_is_one_question_however_the_unit_and_the_wording_changed():
    a = ledger.evidence_key(JOB, "unit-003", "A human-verified transcript or audio replay of the garbled words immediately after 'third party'.")
    b = ledger.evidence_key(JOB, "unit-2", "A reliable transcript or named business for the low-confidence phrase between ‘third party’ and ‘business bank account’.")
    assert a == b                                                                            # both quote "third party" first
    assert a != ledger.evidence_key(JOB, "unit-003", "the words after 'fourth party'") and a != ledger.evidence_key("other-job", "unit-003", "the garbled words after 'third party'")
    assert ledger.evidence_key(JOB, "unit-1", "a count of the speaker's pauses") != ledger.evidence_key(JOB, "unit-2", "a count of the speaker's pauses")          # nothing quoted: the unit counts


def test_entries_written_under_an_older_key_are_re_keyed_when_the_ledger_is_read(tmp_path):
    _run(tmp_path, "prod-1", ROWS, ITEMS)
    ledger.ingest_run(tmp_path, job_id=JOB, run_id="prod-1", clip_id=CLIP)
    path = ledger.ledger_path(tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["kind"] == "missing_evidence":
            row["key"] = "0123456789abcdef"                                            # a key from before the quoted stretch was used
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert next(r for r in ledger.load(tmp_path) if r["kind"] == "missing_evidence")["key"] == ledger.evidence_key(JOB, "unit-001", "the garbled words after 'third party'")
