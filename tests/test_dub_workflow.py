"""Workflow orchestration: planning, assembly, re-dispatch and best-round selection (canned workers)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mathula_tv import dub_workflow
from mathula_tv.dub_workflow import _failing_turns, _task_message, assemble, plan_turns, run_workflow, score_key


def _seg(sid, speaker, start, end):
    return {"segment_id": sid, "speaker_id": speaker, "start_ms": start, "end_ms": end, "english": "x"}


SEGMENTS = [
    _seg("s1", "A", 0, 4000), _seg("s2", "A", 4200, 8000),      # same speaker, small gap -> one turn
    _seg("s3", "B", 8500, 12000),
    _seg("s4", "A", 12500, 16000),
]


def test_plan_turns_groups_same_speaker_runs_and_records_the_next_start():
    plan = plan_turns(SEGMENTS)
    assert [t["segment_ids"] for t in plan] == [["s1", "s2"], ["s3"], ["s4"]]
    assert [t["turn_id"] for t in plan] == ["t01", "t02", "t03"]
    assert plan[0]["next_start_ms"] == 8500 and plan[2]["next_start_ms"] is None
    assert (plan[0]["start_ms"], plan[0]["end_ms"]) == (0, 8000)


def test_score_key_orders_gates_then_overruns_then_dead_air():
    def r(g, o, d):
        return {"headline": {"gate_failures": g, "groups_rushed": o, "residual_dead_air_pct": d}}
    assert score_key(r(0, 0, 9)) < score_key(r(0, 1, 0))
    assert score_key(r(0, 5, 9)) < score_key(r(1, 0, 0))
    assert score_key(r(0, 0, 1)) < score_key(r(0, 0, 2))
    assert score_key(None) > score_key(r(9, 9, 99))


def test_assemble_renames_items_per_turn_and_sorts_by_time():
    plan = plan_turns(SEGMENTS)
    workers = {
        "t01": {"items": [{"item_id": "u1", "start_ms": 0}, {"item_id": "u2", "start_ms": 4200}]},
        "t02": {"items": [{"item_id": "u1", "start_ms": 8500}]},
        "t03": {"items": [{"item_id": "u1", "start_ms": 12500}]},
    }
    assert [i["item_id"] for i in assemble(workers, plan)] == ["t01-u1", "t01-u2", "t02-u1", "t03-u1"]


def test_failing_turns_combines_worker_scores_and_assembled_clip_flags():
    plan = plan_turns(SEGMENTS)
    workers = {"t01": {"unmet": ["residual dead air is 9%"]}, "t02": {"unmet": []}, "t03": {"unmet": []}}
    result = {"flagged_units": [{"item_id": "t03-u1", "gates": ["required_literals"], "flags": ["overrun", "possible_restatement"]}]}
    failing = _failing_turns(workers, result, plan)
    assert set(failing) == {"t01", "t03"}
    assert "required_literals" in failing["t03"][0] and "overrun" in failing["t03"][0]
    assert "possible_restatement" not in failing["t03"][0]


def test_task_message_carries_the_previous_draft_and_its_problems():
    turn = plan_turns(SEGMENTS)[0]
    first = _task_message(turn, None)
    assert "t01" in first and "yours:true" in first and "Previous units" not in first
    again = _task_message(turn, {"items": [{"item_id": "u1", "zulu_text": "ikhaya"}], "issues": ["1 unit(s) overrun"]})
    assert "overrun" in again and "ikhaya" in again


# ---- the orchestration loop, with canned workers ---------------------------------------------

def _headline(gates=0, overruns=0, dead=0.5):
    return {"headline": {"gate_failures": gates, "groups_rushed": overruns, "residual_dead_air_pct": dead},
            "accepted": True, "flagged_units": []}


class FakeParent:
    def __init__(self, tmp_path, scripted):
        self.store = SimpleNamespace(repeat_dir=lambda run, clip, rep: tmp_path / run / clip / f"r{rep}")
        self.tts = object()
        self.scripted = list(scripted)
        self.submitted: list[list[str]] = []
        self.last_submit = None

    def open_clip(self, clip_id, run_id, repeat=1, tts_budget=None, speed_policy="natural"):
        return {"segments": SEGMENTS}

    def submit_dub(self, items, approach_notes=""):
        self.submitted.append([i["item_id"] for i in items])
        self.last_submit = self.scripted.pop(0)
        return self.last_submit


def _install_workers(monkeypatch, behaviour):
    calls = []

    def fake_worker(client, parent, turn, **kw):
        calls.append((kw["round_number"], turn["turn_id"], bool(kw["draft"])))
        unmet, text = behaviour(kw["round_number"], turn["turn_id"])
        return {"turn": turn, "unmet": unmet,
                "summary": {"stop_reason": "agent_finished", "turns": 5, "tts_used": 10, "pushbacks": 0,
                            "tokens": {"in": 1000, "out": 100}},
                "items": [{"item_id": "u1", "start_ms": turn["start_ms"], "zulu_text": text}],
                "result": _headline(overruns=len(unmet))}

    monkeypatch.setattr(dub_workflow, "run_turn_worker", fake_worker)
    return calls


def test_all_turns_meeting_the_bar_needs_one_round(tmp_path, monkeypatch):
    calls = _install_workers(monkeypatch, lambda rnd, tid: ([], "ok"))
    parent = FakeParent(tmp_path, [_headline()])
    summary = run_workflow(object(), parent, clip_id="c", run_id="r", max_rounds=2)
    assert summary["rounds_run"] == 1 and summary["met_bar"] and len(calls) == 3
    assert summary["tokens"] == {"in": 3000, "out": 300} and summary["tts_used"] == 30


def test_only_failing_turns_are_redispatched_with_their_draft(tmp_path, monkeypatch):
    calls = _install_workers(monkeypatch, lambda rnd, tid: (["1 unit(s) overrun their window"], "bad") if (rnd == 1 and tid == "t02") else ([], "ok"))
    parent = FakeParent(tmp_path, [_headline(overruns=1), _headline()])
    summary = run_workflow(object(), parent, clip_id="c", run_id="r", max_rounds=2)
    round_two = [c for c in calls if c[0] == 2]
    assert round_two == [(2, "t02", True)]                     # only the failing turn, with its draft
    assert summary["rounds_run"] == 2 and summary["met_bar"]


def test_redispatch_is_bounded_by_max_rounds(tmp_path, monkeypatch):
    calls = _install_workers(monkeypatch, lambda rnd, tid: (["x"], "bad") if tid == "t01" else ([], "ok"))
    parent = FakeParent(tmp_path, [_headline(overruns=1), _headline(overruns=1), _headline(overruns=1)])
    summary = run_workflow(object(), parent, clip_id="c", run_id="r", max_rounds=2)
    assert summary["rounds_run"] == 2 and not summary["met_bar"]
    assert len([c for c in calls if c[0] == 3]) == 0


def test_the_best_round_wins_if_the_second_is_worse(tmp_path, monkeypatch):
    _install_workers(monkeypatch, lambda rnd, tid: (["x"], "bad") if tid == "t01" else ([], "ok"))
    parent = FakeParent(tmp_path, [_headline(overruns=1), _headline(overruns=4), _headline(overruns=1)])
    summary = run_workflow(object(), parent, clip_id="c", run_id="r", max_rounds=2)
    assert len(parent.submitted) == 3                          # round 1, round 2, then the best restored
    assert summary["final_headline"]["groups_rushed"] == 1
