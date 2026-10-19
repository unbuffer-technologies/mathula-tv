"""Timing first: collisions, the timing ratchet, and the fit-first protocol."""
from __future__ import annotations

import json
from types import SimpleNamespace

from mathula_tv import dub_workbench
from mathula_tv.dub_agent import SYSTEM_PROMPT, pushback_message, slim_submit, unmet_targets
from mathula_tv.dub_workbench import Workbench, ratchet_regressions, segment_timing_status
from mathula_tv.episode_scoring import annotate_collisions, score_items


def _g(gid, speaker, start, end, measured, flags=(), free_gap=0):
    return {"group_id": gid, "speaker_id": speaker, "start_ms": start, "end_ms": end, "measured_ms": measured,
            "flags": list(flags), "free_gap_ms": free_gap}


# ---- collisions ---------------------------------------------------------------------------------

def test_a_unit_running_into_another_speakers_speech_is_a_cross_speaker_collision():
    groups = [_g("a", "S1", 0, 4000, 4600), _g("b", "S2", 4200, 8000, 3000)]
    annotate_collisions(groups, "natural")
    assert groups[0]["collision_ms"] == 400 and groups[0]["collides_with"] == "b" and groups[0]["cross_speaker_collision"]
    assert groups[1]["collision_ms"] == 0


def test_same_speaker_collision_is_counted_but_not_cross_speaker():
    groups = [_g("a", "S1", 0, 4000, 4600), _g("b", "S1", 4200, 8000, 3000)]
    annotate_collisions(groups, "natural")
    assert groups[0]["collision_ms"] == 400 and not groups[0]["cross_speaker_collision"]


def test_speech_that_ends_inside_the_gap_or_within_tolerance_is_not_a_collision():
    groups = [_g("a", "S1", 0, 4000, 5000), _g("b", "S2", 6000, 8000, 1500)]       # 1 s of free gap
    annotate_collisions(groups, "natural")
    assert groups[0]["collision_ms"] == 0
    near = [_g("a", "S1", 0, 4000, 4230), _g("b", "S2", 4200, 8000, 1000)]         # 30 ms over: inside tolerance
    annotate_collisions(near, "natural")
    assert near[0]["collision_ms"] == 0


def test_under_the_pipeline_policy_only_an_unrescuable_unit_can_collide():
    groups = [_g("a", "S1", 0, 4000, 4600), _g("b", "S2", 4200, 8000, 3000)]
    annotate_collisions(groups, "pipeline")                    # the renderer would speed it up: no collision
    assert groups[0]["collision_ms"] == 0
    rushed = [_g("a", "S1", 0, 4000, 4600, flags=["rushed"]), _g("b", "S2", 4200, 8000, 3000)]
    annotate_collisions(rushed, "pipeline")
    assert rushed[0]["collision_ms"] == 400


def test_headline_reports_collision_counts():
    def item(gid, speaker, start, end, measured):
        return {"group_id": gid, "speaker_id": speaker, "start_ms": start, "end_ms": end, "measured_ms": measured,
                "source_text": "x", "spoken_text": "y", "direction": "compress", "required_speed_percent": 8.0,
                "variant_id": "agent"}
    card = score_items([item("a", "S1", 0, 4000, 4600), item("b", "S2", 4200, 8000, 3000)], glossary={}, ledger={},
                       speed_policy="natural")
    h = card["headline"]
    assert (h["collisions"], h["cross_speaker_collisions"], h["worst_collision_ms"]) == (1, 1, 400)


# ---- the bar and the fit-first protocol ---------------------------------------------------------------

def test_collisions_are_part_of_the_bar_and_cross_speaker_is_named_first():
    unmet = unmet_targets({"headline": {"cross_speaker_collisions": 2, "collisions": 3, "groups_rushed": 0,
                                        "gate_failures": 0, "residual_dead_air_pct": 0.5}}, max_dead_air_pct=2.0)
    assert len(unmet) == 1 and "ANOTHER speaker" in unmet[0]
    assert "next unit" in unmet_targets({"headline": {"collisions": 1}}, max_dead_air_pct=2.0)[0]


def test_prompt_puts_fitting_first_and_explains_the_lock():
    assert "FIT FIRST" in SYSTEM_PROMPT and "rejected" in SYSTEM_PROMPT and "talking over each other" in SYSTEM_PROMPT


def test_pushback_and_slim_view_carry_the_timing_lock():
    result = {"accepted": True, "headline": {"residual_dead_air_pct": 5.0},
              "timing_lock": {"fitting_segments": 20, "total_segments": 28}, "turn_padding": [], "flagged_units": []}
    assert slim_submit(result)["timing_lock"] == {"fitting_segments": 20, "total_segments": 28}
    assert "locked for 20 of 28" in pushback_message(result, ["residual dead air is 5.0%"], tts_left=40)


# ---- the ratchet --------------------------------------------------------------------------------------

def _card(*units):
    return {"groups": [{"group_id": u, "flags": f, "collision_ms": c, "collides_with": "next" if c else None}
                       for u, f, c in units]}


def _item(uid, *segments):
    return {"item_id": uid, "source_segment_ids": list(segments)}


def test_segment_status_is_by_segment_so_regrouped_units_are_tracked():
    card = _card(("u1", [], 0), ("u2", ["overrun"], 0), ("u3", [], 250))
    status = segment_timing_status(card, [_item("u1", "s1"), _item("u2", "s2", "s3"), _item("u3", "s4")])
    assert status["s1"]["ok"] and not status["s2"]["ok"] and not status["s3"]["ok"] and not status["s4"]["ok"]
    assert status["s2"]["why"] == ["overrun"] and "collides 250ms" in status["s4"]["why"][0]


def test_ratchet_flags_only_segments_that_fitted_before_and_do_not_now():
    status = {"s1": {"ok": False, "unit": "u1", "why": ["overrun"]}, "s2": {"ok": False, "unit": "u2", "why": ["dead_air"]},
              "s3": {"ok": True, "unit": "u3", "why": []}}
    assert ratchet_regressions({"s1", "s3"}, status) == [{"segment": "s1", "unit": "u1", "now": ["overrun"]}]
    assert ratchet_regressions(set(), status) == []


def _workbench(tmp_path, cards):
    native = tmp_path / "native"
    native.mkdir()
    (native / "pass1_restored_transcript.json").write_text(json.dumps({"segments": [
        {"segment_id": "s1", "start_ms": 0, "end_ms": 4000, "speaker_id": "S1", "restored_text": "one"},
        {"segment_id": "s2", "start_ms": 4200, "end_ms": 8000, "speaker_id": "S2", "restored_text": "two"}]}),
        encoding="utf-8")
    wb = Workbench(SimpleNamespace(load_labels=lambda clip: []), lambda: None)
    episode_dir = tmp_path / "ep"
    episode_dir.mkdir()
    wb.episode = {"clip_id": "c", "run_id": "r", "repeat": 1, "dir": str(episode_dir), "native": str(native),
                  "tts_budget": 10, "tts_used": 0, "submissions": 0, "speed_policy": "natural"}
    queue = list(cards)

    def fake_score(*a, **k):
        return queue.pop(0)

    return wb, fake_score


def _full_card(*units):
    card = _card(*units)
    for group in card["groups"]:
        group["gates"] = {}
    card.update(headline={"gate_failures": 0}, gates={}, turns=[])
    return card


ITEMS = [
    {"item_id": "u1", "start_ms": 0, "end_ms": 4000, "speaker_id": "S1", "zulu_text": "a", "source_segment_ids": ["s1"]},
    {"item_id": "u2", "start_ms": 4200, "end_ms": 8000, "speaker_id": "S2", "zulu_text": "b", "source_segment_ids": ["s2"]},
]


def test_submit_rejects_a_submission_that_breaks_a_segment_that_fitted(tmp_path, monkeypatch):
    wb, fake = _workbench(tmp_path, [
        _full_card(("u1", [], 0), ("u2", ["overrun"], 0)),          # s1 fits, s2 does not
        _full_card(("u1", ["overrun"], 0), ("u2", [], 0)),          # fixes s2 but breaks s1 -> rejected
        _full_card(("u1", [], 0), ("u2", [], 0)),                   # fixes both -> accepted
    ])
    monkeypatch.setattr(dub_workbench, "score_submission_dir", fake)
    first = wb.submit_dub(ITEMS)
    assert first["accepted"] and first["timing_lock"]["fitting_segments"] == 1
    rejected = wb.submit_dub(ITEMS)
    assert not rejected["accepted"] and rejected["regressions"][0]["segment"] == "s1"
    assert wb.episode["submissions"] == 1 and wb.last_submit is first          # the earlier submission still stands
    third = wb.submit_dub(ITEMS)
    assert third["accepted"] and third["timing_lock"]["fitting_segments"] == 2 and wb.episode["submissions"] == 2


def test_the_ratchet_can_be_turned_off(tmp_path, monkeypatch):
    wb, fake = _workbench(tmp_path, [_full_card(("u1", [], 0), ("u2", [], 0)),
                                     _full_card(("u1", ["overrun"], 0), ("u2", [], 0))])
    wb.timing_ratchet = False
    monkeypatch.setattr(dub_workbench, "score_submission_dir", fake)
    assert wb.submit_dub(ITEMS)["accepted"] and wb.submit_dub(ITEMS)["accepted"]
