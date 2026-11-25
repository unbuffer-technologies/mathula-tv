import json

from mathula_tv import ear_lessons, experience_compact as ec


def row(i, job, kind="choice", outcome="kept", **extra):
    return {"id": f"r{i}", "job": job, "run": f"prod-{i}", "kind": kind, "unit": "u1", "english": "the figure", "zulu": "inani", "why": "w", "evidence": "e", "outcome": outcome, **extra}


ROWS = [row(1, "jobA", outcome="not_in_final"), row(2, "jobA", outcome="not_in_final"), row(3, "jobB", outcome="not_in_final"), row(4, "jobB"), row(5, "jobC", kind="strategy", strategy="read the whole turn first"), row(6, "jobA", outcome="not_in_final")]


def test_a_lesson_stands_only_when_its_rows_exist_and_span_enough_videos():
    proposed = [
        {"item_class": "figures", "situation": "a figure at a sentence end", "do": "put a breath after it", "avoid": "", "because": "dropped in 3 runs", "ledger_ids": ["r1", "r2", "r3"]},
        {"item_class": "one video", "situation": "a name", "do": "spell it so", "avoid": "", "because": "x", "ledger_ids": ["r1", "r2", "r6"]},               # three rows, one video
        {"item_class": "made up", "situation": "s", "do": "d", "avoid": "", "because": "x", "ledger_ids": ["r1", "r3", "nope"]},                            # a cited row that does not exist does not count
        {"item_class": "no action", "situation": "s", "do": "", "avoid": "", "because": "x", "ledger_ids": ["r1", "r3", "r4"]},
    ]
    kept, dropped = ec.check(proposed, ROWS)
    assert [k["item_class"] for k in kept] == ["figures"] and kept[0]["videos"] == 2 and kept[0]["source"] == "inferred" and kept[0]["id"].startswith("exp-")
    assert {d["item_class"] for d in dropped} == {"one video", "made up", "no action"}
    assert "needs 3" in next(d for d in dropped if d["item_class"] == "made up")["why_dropped"] and "video" in next(d for d in dropped if d["item_class"] == "one video")["why_dropped"]


def test_the_selection_keeps_every_dropped_choice_and_stays_inside_its_budget():
    rows = [row(i, f"job{i % 3}", outcome="kept") for i in range(200)] + [row(900 + i, "jobX", outcome="not_in_final") for i in range(5)]
    shown = ec.select(rows, kept_sample=10)
    assert sum(1 for r in shown if r.get("outcome") == "not_in_final") == 5 and sum(1 for r in shown if r.get("outcome") == "kept") <= 10
    small = ec.select(rows, budget=600)
    assert len(json.dumps(small)) <= 600 + 400


def test_one_call_proposes_and_the_stored_lessons_reach_the_agent_until_retired(tmp_path):
    calls = []

    def model(messages):
        calls.append(messages)
        return json.dumps({"lessons": [{"item_class": "figures", "situation": "a figure at a sentence end", "do": "breathe after it", "avoid": "running it on", "because": "b", "ledger_ids": ["r1", "r2", "r3"]}]})

    kept, _ = ec.check(ec.propose(ROWS, model), ROWS)
    assert len(calls) == 1 and len(kept) == 1
    ec.store(tmp_path, kept, rows=len(ROWS))
    shown = ear_lessons.for_agent(tmp_path, tmp_path / "book.json")
    assert shown["experience_lessons"][0]["do"] == "breathe after it" and "counter-example" in shown["experience_note"]
    assert ec.retire(tmp_path, kept[0]["id"], "wrong") and "experience_lessons" not in ear_lessons.for_agent(tmp_path, tmp_path / "book.json")
    ec.store(tmp_path, kept, rows=len(ROWS))                                               # a re-run does not bring a retired lesson back
    assert "experience_lessons" not in ear_lessons.for_agent(tmp_path, tmp_path / "book.json")


def test_a_retired_lesson_can_be_reinstated_and_keeps_its_history(tmp_path):
    kept, _ = ec.check([{"item_class": "figures", "situation": "s", "do": "d", "avoid": "", "because": "b", "ledger_ids": ["r1", "r2", "r3"]}], ROWS)
    ec.store(tmp_path, kept, rows=len(ROWS))
    ec.retire(tmp_path, kept[0]["id"], "wrong")
    assert ec.for_agent(tmp_path) == [] and ec.reinstate(tmp_path, kept[0]["id"]) and len(ec.for_agent(tmp_path)) == 1
    assert json.loads((tmp_path / ec.FILE).read_text())["lessons"][0]["history"][0]["retired_because"] == "wrong" and not ec.reinstate(tmp_path, kept[0]["id"])
