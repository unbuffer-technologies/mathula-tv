"""Harness-problem detectors, multi-clip trials and the unattended driver."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mathula_tv import dub_improve
from mathula_tv.dub_improve import autopilot, run_batch
from mathula_tv.dub_smells import find_smells, render
from mathula_tv.playbook import current_playbook, seed_playbook


def _run(tmp_path, name, *, repeat=1, stop="agent_finished", tokens=300_000, tts=40, records=(), summary=None, groups=None,
         budget=None, clip="c"):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    (directory / "agent_summary.json").write_text(json.dumps({"turns": 10, "compactions": 2, **(summary or {})}), encoding="utf-8")
    if groups is not None:
        (directory / "scorecard.json").write_text(json.dumps({"headline": {}, "groups": groups}), encoding="utf-8")
    if budget:
        (directory / "episode.json").write_text(json.dumps({"tts_budget": budget}), encoding="utf-8")
    return {"clip": clip, "repeat": repeat, "dir": directory, "stop_reason": stop, "processed_tokens": tokens, "tts_used": tts}


def _tool_error(tool="measure_tts_batch", detail="send between 1 and 12 requests (you sent 14)"):
    return {"t": 1, "kind": "tool", "name": tool, "ok": False, "result": json.dumps({"error": "ValueError", "detail": detail})}


def _smells(runs):
    return {f["smell"]: f for f in find_smells(runs)}


def test_the_same_tool_error_in_several_runs_is_a_harness_problem(tmp_path):
    runs = [_run(tmp_path, "a", repeat=1, records=[_tool_error()]), _run(tmp_path, "b", repeat=2, records=[_tool_error(detail="send between 1 and 12 requests (you sent 20)")])]
    found = _smells(runs)["repeated_tool_error"]
    assert found["severity"] == "high" and found["owner"] == "harness" and found["runs"] == ["c:r1", "c:r2"]
    assert "measure_tts_batch failed 2x" in found["evidence"]
    assert "repeated_tool_error" not in _smells([_run(tmp_path, "solo", records=[_tool_error()])])     # one-off: not a pattern


def test_repeated_rejections_that_end_badly_are_a_trap_but_ones_that_end_well_are_not(tmp_path):
    rejected = {"t": 5, "kind": "tool", "name": "submit_dub", "ok": True, "result": json.dumps({"accepted": False, "errors": ["timing regression: 3"]})}
    trapped = _run(tmp_path, "t", stop="finished_below_target", records=[rejected, rejected])
    assert "rejection_trap" in _smells([trapped])
    assert "rejection_trap" not in _smells([_run(tmp_path, "ok", stop="agent_finished", records=[rejected, rejected])])


def test_declined_to_submit_budget_and_ceiling_problems_are_found(tmp_path):
    nudged = _run(tmp_path, "n", records=[{"t": 3, "kind": "nudge", "turn": 4, "text": "You have not submitted"}])
    assert "declined_to_submit" in _smells([nudged])
    spent = _run(tmp_path, "s", tts=598, budget=600)
    assert "budget_exhausted" in _smells([spent])
    ceiling = _run(tmp_path, "m", stop="max_turns")
    assert "ceiling_stop" in _smells([ceiling])


def test_token_blowup_and_missing_compaction(tmp_path):
    runs = [_run(tmp_path, "a", repeat=1, tokens=300_000), _run(tmp_path, "b", repeat=2, tokens=320_000),
            _run(tmp_path, "c", repeat=3, tokens=1_500_000, summary={"turns": 25, "compactions": 0})]
    found = _smells(runs)
    assert found["token_blowup"]["runs"] == ["c:r3"] and found["compaction_missing"]["runs"] == ["c:r3"]


def test_a_window_flagged_in_every_run_is_reported_as_possibly_unreachable(tmp_path):
    def g(start, flags=()):
        return {"start_ms": start, "flags": list(flags), "collision_ms": 0}
    runs = [_run(tmp_path, "a", repeat=1, groups=[g(1000, ["dead_air"]), g(9000)]),
            _run(tmp_path, "b", repeat=2, groups=[g(1010, ["dead_air", "overrun"]), g(9000, ["overrun"])])]
    found = [f for f in find_smells(runs) if f["smell"] == "unreachable_window"]
    assert len(found) == 1 and "1000 ms" in found[0]["evidence"] and "dead_air" in found[0]["evidence"]


def test_findings_are_sorted_by_severity_and_an_empty_report_says_so(tmp_path):
    runs = [_run(tmp_path, "a", repeat=1, stop="max_turns", records=[_tool_error()]), _run(tmp_path, "b", repeat=2, records=[_tool_error()])]
    severities = [f["severity"] for f in find_smells(runs)]
    assert severities == sorted(severities, key=["high", "medium", "low"].index)
    assert render([]) == "no harness problems detected" and "[high]" in render(find_smells(runs))


def test_known_harness_problems_reach_the_reflector_with_a_do_not_write_lessons_instruction(tmp_path, monkeypatch):
    captured = {}

    def fake_reflect(client, playbook, digests, clip_note=""):
        captured["note"] = clip_note
        return {"edits": [], "diagnosis": [], "contrast": "", "usage": {}}

    monkeypatch.setattr(dub_improve, "reflect", fake_reflect)
    runs = [_run(tmp_path, "a", repeat=1, records=[_tool_error()]), _run(tmp_path, "b", repeat=2, records=[_tool_error()])]
    monkeypatch.setattr(dub_improve, "collect_batches", lambda *a, **k: runs)
    monkeypatch.setattr(dub_improve, "digest_run", lambda d: "digest")
    outcome = dub_improve.reflect_on_batch(None, None, "b", ["c"], seed_playbook())
    assert "do NOT write lessons" in captured["note"] and "measure_tts_batch failed 2x" in captured["note"]
    assert outcome["smells"]


# ---- several clips ---------------------------------------------------------------------------------------------

def test_a_trial_runs_every_repeat_on_every_clip_with_the_candidate_playbook(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dub_improve, "run_agent", lambda client, wb, **kw: calls.append((kw["clip_id"], kw["repeat"], kw["playbook_version"], kw["system_prompt"])) or {})
    monkeypatch.setattr(dub_improve, "collect_batches", lambda store, run_id, clips: [])
    store = SimpleNamespace(root=tmp_path)
    playbook = seed_playbook()
    run_batch(None, store, lambda: object(), run_id="t", playbook=playbook, clip_ids=["a", "b"], repeats=3)
    assert sorted((c, r) for c, r, _, _ in calls) == [(c, r) for c in ("a", "b") for r in (1, 2, 3)]
    assert {v for _, _, v, _ in calls} == {playbook["version"]} and all("PLAYBOOK" in p for *_, p in calls)


# ---- autopilot -------------------------------------------------------------------------------------------------

def _row(tokens=300_000):
    return {"met_bar": True, "headline": {}, "processed_tokens": tokens}


def _auto_env(tmp_path, monkeypatch, decisions, baseline_tokens=300_000):
    store = SimpleNamespace(root=tmp_path / "experience")
    current_playbook(store)
    monkeypatch.setattr(dub_improve, "collect_batches", lambda *a, **k: [_row(baseline_tokens)])
    seen = []
    queue = list(decisions)

    def fake_step(client, store_, tts, *, baseline_run, trial_run, clip_ids, repeats, **kw):
        decision = queue.pop(0)
        seen.append((baseline_run, trial_run))
        return {"decision": decision, "reasons": [], "trial_run": trial_run, "from_playbook": "x",
                "candidate_stats": {"tokens_mean": 300_000.0, "n": 3 * len(clip_ids)},
                "reflection_tokens": {"prompt_tokens": 3000, "completion_tokens": 700}}

    monkeypatch.setattr(dub_improve, "improvement_step", fake_step)
    return store, seen


def test_autopilot_stops_after_patience_rounds_without_an_adoption(tmp_path, monkeypatch):
    store, seen = _auto_env(tmp_path, monkeypatch, ["rejected", "no_change", "adopted"])
    summary = autopilot(None, store, None, clip_ids=["a"], baseline_run="base", run_prefix="p", max_rounds=5, patience=2)
    assert summary["stop_reason"] == "patience" and len(summary["rounds"]) == 2 and summary["adopted"] == 0
    assert seen == [("base", "p-r1"), ("base", "p-r2")]                       # baseline unchanged without an adoption


def test_an_adoption_becomes_the_next_baseline_and_resets_patience(tmp_path, monkeypatch):
    store, seen = _auto_env(tmp_path, monkeypatch, ["adopted", "rejected", "adopted"])
    summary = autopilot(None, store, None, clip_ids=["a", "b"], baseline_run="base", run_prefix="p", max_rounds=3, patience=2)
    assert summary["stop_reason"] == "max_rounds" and summary["adopted"] == 2
    assert seen == [("base", "p-r1"), ("p-r1", "p-r2"), ("p-r1", "p-r3")]


def test_autopilot_will_not_start_a_trial_it_cannot_afford(tmp_path, monkeypatch):
    store, seen = _auto_env(tmp_path, monkeypatch, ["adopted"], baseline_tokens=1_000_000)
    summary = autopilot(None, store, None, clip_ids=["a", "b"], baseline_run="base", run_prefix="p", token_budget=5_000_000)
    assert summary["stop_reason"] == "token_budget" and seen == [] and summary["tokens_spent"] == 0   # 2 clips x 3 x 1M > 5M


def test_autopilot_counts_what_each_round_spends_and_records_itself(tmp_path, monkeypatch):
    store, _ = _auto_env(tmp_path, monkeypatch, ["adopted", "adopted"])
    summary = autopilot(None, store, None, clip_ids=["a"], baseline_run="base", run_prefix="p", max_rounds=2, token_budget=10_000_000)
    assert summary["tokens_spent"] == 2 * (3000 + 700 + 900_000)
    history = (store.root / "playbooks" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(history[-1])["decisions"] == ["adopted", "adopted"]
