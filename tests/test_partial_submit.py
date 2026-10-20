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
