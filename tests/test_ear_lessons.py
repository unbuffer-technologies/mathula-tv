import json

from mathula_tv import ear_lessons
from mathula_tv.dub_agent import SYSTEM_PROMPT, TOOL_SPECS


def _book(tmp_path, entries):
    path = tmp_path / "pronunciation_book.json"
    path.write_text(json.dumps({"schema": "x", "entries": entries, "feedback": [], "log": []}), encoding="utf-8")
    return path


def test_the_listeners_session_lessons_are_seeded_by_item_class(tmp_path):
    out = ear_lessons.for_agent(tmp_path, _book(tmp_path, {}))
    classes = {c["item_class"] for c in out["concepts"]}
    assert classes >= {"title", "figure", "year", "name", "pacing", "spelling"}
    text = json.dumps(out["concepts"])
    assert "sinoDkt" in text and "titty one" in text and "eNorth West University" in text and "eyt -> eyit" in text
    assert all(c["written"] and c["heard"] and c["fix"] for c in out["concepts"])


def test_a_lesson_added_after_a_job_is_kept_once_and_shown_to_the_agent(tmp_path):
    first = ear_lessons.add(tmp_path, item_class="figure", written="R127 wezigidi", heard="digits read in a Zulu accent", fix="write the amount with its words", job="6d95d85d")
    again = ear_lessons.add(tmp_path, item_class="figure", written="R127 wezigidi", heard="digits read in a Zulu accent", fix="a different fix text")
    assert first["id"] == again["id"] and len(ear_lessons._read(tmp_path)) == 1
    shown = ear_lessons.for_agent(tmp_path, _book(tmp_path, {}))["concepts"]
    assert any(c["written"] == "R127 wezigidi" for c in shown) and len(shown) == len(ear_lessons.SEED) + 1


def test_the_listeners_respellings_come_from_the_book_newest_first_and_only_where_they_differ(tmp_path):
    book = _book(tmp_path, {
        "eyt": {"word": "eight", "kind": "custom", "spelling": {"*": "eyit"}, "rule_spelling": "eyt", "decided_at": "2026-11-10T10:00:00"},
        "pr": {"word": "PR", "kind": "custom", "spelling": {"*": "phii-aarrri"}, "rule_spelling": "pi-aaar", "decided_at": "2026-11-11T10:00:00"},
        "north": {"word": "North", "kind": "approved", "spelling": {"*": "nooth"}, "rule_spelling": "nooth", "decided_at": "2026-11-11T11:00:00"},
        "same": {"word": "Same", "kind": "custom", "spelling": {"*": "sem"}, "rule_spelling": "sem", "decided_at": "2026-11-11T12:00:00"},
    })
    rows = ear_lessons.for_agent(tmp_path, book)["listener_respellings"]
    assert rows == [{"word": "PR", "rules_wrote": "pi-aaar", "listener_chose": "phii-aarrri"}, {"word": "eight", "rules_wrote": "eyt", "listener_chose": "eyit"}]


def test_a_missing_book_never_breaks_the_first_call_of_a_run(tmp_path):
    out = ear_lessons.for_agent(tmp_path, tmp_path / "nope.json")
    assert out["listener_respellings"] == [] and out["concepts"]


def test_recall_lessons_carries_the_ear_lessons_and_the_agent_is_told_to_read_them(tmp_path):
    from pathlib import Path
    from mathula_tv.dub_workbench import Workbench

    class Store:
        root = tmp_path / "experience"

        def clip_ids(self):
            return []

    bench = Workbench.__new__(Workbench)
    bench.store = Store()
    out = bench.recall_lessons()
    assert out["count"] == 0 and out["ear_lessons"]["concepts"]
    spec = next(s for s in TOOL_SPECS if s["function"]["name"] == "recall_lessons")
    assert "ear_lessons" in spec["function"]["description"]
    assert "Learn from the listener" in SYSTEM_PROMPT and "ear_lessons" in SYSTEM_PROMPT


def test_the_listeners_notes_on_a_spelling_are_shown_newest_first(tmp_path):
    path = tmp_path / "pronunciation_book.json"
    path.write_text(json.dumps({"schema": "x", "entries": {}, "log": [], "feedback": [
        {"word": "Act", "note": ""}, {"word": "University", "note": "the end drops off"}, {"word": "Arthur", "note": "stress the first part"}]}), encoding="utf-8")
    assert ear_lessons.for_agent(tmp_path, path)["listener_notes"] == [{"word": "Arthur", "note": "stress the first part"}, {"word": "University", "note": "the end drops off"}]


def test_every_lesson_says_where_it_came_from_and_an_inference_is_marked_as_one(tmp_path):
    out = ear_lessons.for_agent(tmp_path, _book(tmp_path, {}))
    assert {c["source"] for c in out["concepts"]} <= set(ear_lessons.SOURCES)
    by_class = {c["item_class"]: c["source"] for c in out["concepts"] if c["item_class"] in ("spelling", "title")}
    assert by_class == {"spelling": "inferred", "title": "listener"}                                  # the final-consonant pattern is the assistant's reading, not the listener's words
    assert "hypothesis" in out["note"] and "counter-example" in out["note"]
    try:
        ear_lessons.add(tmp_path, item_class="x", written="a", heard="b", fix="c", source="guess")
        raise AssertionError("an unknown source must be refused")
    except ValueError:
        pass


def test_a_retired_lesson_leaves_what_the_agent_sees_and_its_reason_is_kept(tmp_path):
    before = ear_lessons.for_agent(tmp_path, _book(tmp_path, {}))["concepts"]
    ear_lessons.retire(tmp_path, "seed-final-consonant", "the next three jobs showed no such pattern")
    after = ear_lessons.for_agent(tmp_path, _book(tmp_path, {}))["concepts"]
    assert len(after) == len(before) - 1 and not any(c["item_class"] == "spelling" for c in after)
    assert json.loads((tmp_path / ear_lessons.FILE).read_text(encoding="utf-8"))["retired"]["seed-final-consonant"]["reason"].startswith("the next three jobs")


def test_a_complaint_adds_its_line_to_the_standing_test_once(tmp_path):
    fixture = tmp_path / "hard_text.txt"
    assert ear_lessons.add_hard_text("Ngo-7 ekuseni  abantu abangu-9.", fixture) is True
    assert ear_lessons.add_hard_text("Ngo-7 ekuseni abantu abangu-9.", fixture) is False                     # the same line with other spacing: already there
    assert fixture.read_text(encoding="utf-8") == "Ngo-7 ekuseni abantu abangu-9.\n"
