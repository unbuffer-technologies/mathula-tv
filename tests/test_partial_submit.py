"""submit_dub(partial=true): send only changed units; get_submission shows the current state."""
from __future__ import annotations

import json
from types import SimpleNamespace

from mathula_tv import dub_workbench
from mathula_tv.dub_agent import COMPACTION_NOTE, SYSTEM_PROMPT, TOOL_SPECS, slim_submit
from mathula_tv.dub_workbench import Workbench, merge_partial


def _u(uid, start, end, *segments, text="zulu", speaker="S1"):
    return {"item_id": uid, "start_ms": start, "end_ms": end, "speaker_id": speaker, "zulu_text": text,
            "source_segment_ids": list(segments)}


PREVIOUS = [_u("a", 0, 4000, "s1"), _u("b", 4200, 8000, "s2", "s3"), _u("c", 8200, 12000, "s4")]


def test_a_unit_is_replaced_by_its_item_id_and_nothing_else():
    merged, replaced = merge_partial(PREVIOUS, [_u("a", 0, 4000, "s1", text="new")])
    assert replaced == ["a"] and [u["item_id"] for u in merged] == ["a", "b", "c"] and merged[0]["zulu_text"] == "new"


def test_a_shared_boundary_segment_does_not_drop_the_neighbouring_units():
    # The real trap: units legitimately share a boundary segment (seg-6 and seg-10 here). Sending only the middle
    # unit must not silently delete the units on either side of it.
    previous = [_u("u3", 0, 4000, "s4", "s5", "s6"), _u("u4", 4200, 8000, "s6", "s7", "s10"), _u("u5", 8200, 12000, "s10")]
    merged, replaced = merge_partial(previous, [_u("u4", 4200, 8000, "s6", "s7", "s10", text="edited")])
    assert replaced == ["u4"] and [u["item_id"] for u in merged] == ["u3", "u4", "u5"]


def test_remove_deletes_units_the_agent_has_re_split():
    merged, replaced = merge_partial(PREVIOUS, [_u("b1", 4200, 6000, "s2"), _u("b2", 6100, 8000, "s3")], remove=["b"])
    assert replaced == ["b"] and [u["item_id"] for u in merged] == ["a", "b1", "b2", "c"]


def test_untouched_units_carry_over_and_the_result_is_in_time_order():
    merged, replaced = merge_partial(PREVIOUS, [_u("c", 100, 3900, "s4", text="moved earlier")])
    assert replaced == ["c"] and [u["start_ms"] for u in merged] == sorted(u["start_ms"] for u in merged)
    assert [u["item_id"] for u in merged] == ["a", "c", "b"]


def _workbench(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    (native / "pass1_restored_transcript.json").write_text(json.dumps({"segments": [
        {"segment_id": f"s{i}", "start_ms": (i - 1) * 4200, "end_ms": (i - 1) * 4200 + 4000, "speaker_id": "S1", "restored_text": f"e{i}"}
        for i in range(1, 5)]}), encoding="utf-8")
    wb = Workbench(SimpleNamespace(load_labels=lambda clip: []), lambda: None)
    episode_dir = tmp_path / "ep"
    episode_dir.mkdir()
    wb.episode = {"clip_id": "c", "run_id": "r", "repeat": 1, "dir": str(episode_dir), "native": str(native),
                  "tts_budget": 10, "tts_used": 0, "submissions": 0, "speed_policy": "natural"}
    return wb


def _fake_score(seen):
    def score(native, submission, **kwargs):
        items = submission["items"]
        seen.append([i["item_id"] for i in items])
        return {"groups": [{"group_id": i["item_id"], "flags": [], "collision_ms": 0, "collides_with": None, "gates": {}}
                           for i in items],
                "headline": {"gate_failures": 0, "groups_rushed": 0, "collisions": 0, "cross_speaker_collisions": 0},
                "gates": {}, "turns": []}
    return score


FULL = [_u("a", 0, 4000, "s1"), _u("b", 4200, 8200, "s2", "s3"), _u("c", 12600, 16600, "s4")]


def test_partial_needs_a_previous_accepted_submission(tmp_path, monkeypatch):
    wb = _workbench(tmp_path)
    monkeypatch.setattr(dub_workbench, "score_submission_dir", _fake_score([]))
    result = wb.submit_dub([_u("a", 0, 4000, "s1")], partial=True)
    assert not result["accepted"] and "send the full set" in result["errors"][0]


def test_a_partial_update_is_merged_scored_as_the_whole_and_remembered(tmp_path, monkeypatch):
    wb = _workbench(tmp_path)
    seen = []
    monkeypatch.setattr(dub_workbench, "score_submission_dir", _fake_score(seen))
    assert wb.submit_dub(FULL)["accepted"]
    result = wb.submit_dub([_u("c", 12600, 16600, "s4", text="fixed")], partial=True)
    assert result["accepted"] and seen[1] == ["a", "b", "c"]                      # the scorer saw the WHOLE clip
    assert result["merged"] == {"sent": 1, "replaced": ["c"], "units_now": 3}
    assert {u["item_id"]: u["zulu_text"] for u in wb.last_items}["c"] == "fixed" and wb.episode["submissions"] == 2
    assert slim_submit(result)["merged"]["replaced"] == ["c"]


def test_a_partial_update_that_drops_a_segment_is_rejected_with_a_hint(tmp_path, monkeypatch):
    wb = _workbench(tmp_path)
    monkeypatch.setattr(dub_workbench, "score_submission_dir", _fake_score([]))
    wb.submit_dub(FULL)
    result = wb.submit_dub([_u("b", 4200, 8200, "s2")], partial=True)           # b used to cover s2 AND s3
    assert not result["accepted"]
    assert any("'s3' is not covered" in e for e in result["errors"]) and any("replaced your current unit(s) ['b']" in e for e in result["errors"])
    assert wb.episode["submissions"] == 1                                       # the earlier submission still stands


def test_a_full_submission_still_means_the_whole_clip(tmp_path, monkeypatch):
    wb = _workbench(tmp_path)
    monkeypatch.setattr(dub_workbench, "score_submission_dir", _fake_score([]))
    wb.submit_dub(FULL)
    result = wb.submit_dub([_u("a", 0, 4000, "s1")])                            # partial left false
    assert not result["accepted"] and any("not covered" in e for e in result["errors"])


def test_get_submission_shows_current_units_and_their_flags(tmp_path, monkeypatch):
    wb = _workbench(tmp_path)
    assert wb.get_submission()["count"] == 0
    monkeypatch.setattr(dub_workbench, "score_submission_dir", _fake_score([]))
    wb.submit_dub(FULL)
    wb.last_submit["flagged_units"] = [{"item_id": "b", "gates": ["required_literals"], "flags": ["overrun"]}]
    everything = wb.get_submission()
    assert everything["count"] == 3 and {u["item_id"] for u in everything["units"]} == {"a", "b", "c"}
    only_b = wb.get_submission(["b"])
    assert only_b["count"] == 1 and only_b["units"][0]["flagged"] == {"gates": ["required_literals"], "flags": ["overrun"]}
    assert "flagged" not in wb.get_submission(["a"])["units"][0]


def test_the_agent_is_told_about_both_and_the_compaction_note_points_at_get_submission():
    names = {t["function"]["name"] for t in TOOL_SPECS}
    assert "get_submission" in names
    submit = next(t for t in TOOL_SPECS if t["function"]["name"] == "submit_dub")["function"]["parameters"]["properties"]
    assert submit["partial"]["type"] == "boolean"
    assert "partial=true" in SYSTEM_PROMPT and "get_submission" in SYSTEM_PROMPT and "get_submission" in COMPACTION_NOTE


def test_remove_with_an_unknown_id_is_rejected_before_scoring(tmp_path, monkeypatch):
    wb = _workbench(tmp_path)
    monkeypatch.setattr(dub_workbench, "score_submission_dir", _fake_score([]))
    wb.submit_dub(FULL)
    result = wb.submit_dub([_u("c", 12600, 16600, "s4")], partial=True, remove=["nope"])
    assert not result["accepted"] and "nope" in result["errors"][0]


def test_a_partial_update_can_re_split_units_using_remove(tmp_path, monkeypatch):
    wb = _workbench(tmp_path)
    seen = []
    monkeypatch.setattr(dub_workbench, "score_submission_dir", _fake_score(seen))
    wb.submit_dub(FULL)
    result = wb.submit_dub([_u("b1", 4200, 6000, "s2"), _u("b2", 6200, 8200, "s3")], partial=True, remove=["b"])
    assert result["accepted"] and seen[1] == ["a", "b1", "b2", "c"] and result["merged"]["replaced"] == ["b"]


# --- the source's sentences: the renderer cannot split one between units ---

LONG_SENTENCE_SOURCE = [
    {"segment_id": "s1__s00", "start_ms": 0, "end_ms": 3800, "parent_segment_ids": ["s1", "s2", "s3", "s4"]},
    {"segment_id": "s1__s01", "start_ms": 4200, "end_ms": 16600, "parent_segment_ids": ["s1", "s2", "s3", "s4"]},          # one 12 s sentence over the rest of the clip
]


def test_units_that_split_one_source_sentence_are_refused_with_what_the_agent_needs_to_know(tmp_path, monkeypatch):
    wb = _workbench(tmp_path)
    wb._source_sentences = lambda: LONG_SENTENCE_SOURCE
    scored = []
    monkeypatch.setattr(dub_workbench, "score_submission_dir", _fake_score(scored))
    result = wb.submit_dub([_u("a", 0, 4000, "s1"), _u("b", 4200, 8200, "s2", "s3"), _u("c", 12600, 16600, "s4")])
    assert not result["accepted"] and not scored                                                      # refused before a single unit is measured
    message = result["errors"][0]
    assert "own no sentence" in message and "s1__s01 4.2-16.6 s" in message                          # the sentences, with their times
    assert "unit b starts at 4.2 s" not in message and "unit c starts at 12.6 s, inside sentence s1__s01" in message          # where the units cut a sentence
    assert "single unit" in message and "sentence boundary" in message


def test_units_that_start_on_the_source_sentence_boundaries_are_taken(tmp_path, monkeypatch):
    wb = _workbench(tmp_path)
    wb._source_sentences = lambda: LONG_SENTENCE_SOURCE
    scored = []
    monkeypatch.setattr(dub_workbench, "score_submission_dir", _fake_score(scored))
    result = wb.submit_dub([_u("a", 0, 4000, "s1"), _u("b", 4200, 16600, "s2", "s3", "s4")])
    assert result.get("accepted") is not False and scored                                             # the second unit holds the long sentence whole
    (tmp_path / "one").mkdir()
    one = _workbench(tmp_path / "one")
    one._source_sentences = lambda: LONG_SENTENCE_SOURCE
    assert one.submit_dub([_u("a", 0, 16600, "s1", "s2", "s3", "s4")]).get("accepted") is not False  # and so does one unit for the whole clip


def test_no_sentences_to_check_against_means_no_check(tmp_path, monkeypatch):
    wb = _workbench(tmp_path)
    wb._source_sentences = lambda: None                                                               # a job not on disk: the render finds any problem as before
    scored = []
    monkeypatch.setattr(dub_workbench, "score_submission_dir", _fake_score(scored))
    assert wb.submit_dub([_u("a", 0, 4000, "s1"), _u("b", 4200, 8200, "s2", "s3"), _u("c", 12600, 16600, "s4")]).get("accepted") is not False
