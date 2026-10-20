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


# ---- sharper rejection detectors (patterns seen in real runs) -----------------------------------------------------

def _rej(text):
    return {"t": 5, "kind": "tool", "name": "submit_dub", "ok": True, "result": json.dumps({"accepted": False, "errors": [text]})}


def test_one_timing_rejection_that_froze_a_first_draft_is_a_trap(tmp_path):
    run = _run(tmp_path, "t", stop="finished_below_target", records=[_rej("timing regression: 3 segment(s)")], summary={"submissions": 1})
    assert "rejection_trap" in _smells([run])
    recovered = _run(tmp_path, "ok", stop="finished_below_target", records=[_rej("timing regression: 3 segment(s)")], summary={"submissions": 4})
    assert "rejection_trap" not in _smells([recovered])


def test_repeated_coverage_rejections_are_invalid_submissions_not_a_timing_trap(tmp_path):
    coverage = _rej("source segment 'seg-00001' is not covered by any item (untranslated content)")
    run = _run(tmp_path, "c", stop="finished_below_target", records=[coverage, coverage, coverage], summary={"submissions": 2})
    found = _smells([run])
    assert "invalid_submissions" in found and "coverage" in found["invalid_submissions"]["evidence"]
    assert "rejection_trap" not in found
    assert "invalid_submissions" not in _smells([_run(tmp_path, "one", records=[coverage])])


# ---- transient Azure failures are retried, boundedly ----------------------------------------------------------------------

def test_a_rate_limit_is_retried_with_backoff_and_then_succeeds():
    from mathula_tv.dub_fit import with_transient_retry

    calls, naps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Azure TTS synthesis HTTP 429")
        return "audio"

    assert with_transient_retry(flaky, sleep=naps.append, jitter=lambda: 0.5) == "audio"
    assert len(calls) == 3 and naps == [2.0, 4.0]                      # 2**attempt, jitter factor 1.0


def test_non_transient_errors_are_raised_immediately_and_retries_are_bounded():
    import pytest

    from mathula_tv.dub_fit import with_transient_retry

    calls = []

    def bad_request():
        calls.append(1)
        raise RuntimeError("Azure TTS synthesis HTTP 400")

    with pytest.raises(RuntimeError):
        with_transient_retry(bad_request, sleep=lambda s: None)
    assert len(calls) == 1
    always = []

    def always_limited():
        always.append(1)
        raise RuntimeError("Azure TTS synthesis HTTP 429")

    with pytest.raises(RuntimeError):
        with_transient_retry(always_limited, attempts=4, sleep=lambda s: None)
    assert len(always) == 4


# ---- a gate no wording can clear is a scorer problem (the "7,000" false positive) -------------------------------------

def _gate_run(tmp_path, name, repeat, evidence, clip="c"):
    run = _run(tmp_path, name, repeat=repeat, clip=clip, stop="finished_below_target")
    groups = [{"group_id": "u1", "flags": [], "gates": {"required_literals": {"failed": bool(evidence), "evidence": evidence}}}]
    (run["dir"] / "scorecard.json").write_text(json.dumps({"headline": {}, "groups": groups}), encoding="utf-8")
    return run


def test_the_same_gate_evidence_surviving_in_several_runs_points_at_the_scorer(tmp_path):
    runs = [_gate_run(tmp_path, "a", 1, ["7000 x1"]), _gate_run(tmp_path, "b", 2, ["7000 x1"]), _gate_run(tmp_path, "c", 3, [])]
    found = _smells(runs)["persistent_gate_failure"]
    assert found["severity"] == "high" and found["runs"] == ["c:r1", "c:r2"] and "7000 x1" in found["evidence"]
    assert "Suspect the scorer" in found["suggestion"]


def test_different_gate_failures_or_a_single_run_are_not_flagged(tmp_path):
    differing = [_gate_run(tmp_path, "a", 1, ["7000 x1"]), _gate_run(tmp_path, "b", 2, ["NPA x1"])]
    assert "persistent_gate_failure" not in _smells(differing)
    assert "persistent_gate_failure" not in _smells([_gate_run(tmp_path, "solo", 1, ["7000 x1"])])


def test_number_grouping_is_normalised_the_same_way_on_both_sides():
    from mathula_tv.episode_scoring import normalise_number_format as norm

    assert norm("about 7,000 rand") == "about 7000 rand"                       # English comma
    assert norm("amarandi ayizi-7,000 kuphela") == "amarandi ayizi-7000 kuphela"   # the Zulu comma that was missed
    assert norm("u-7 000 rand") == "u-7000 rand" and norm("7\u00a0000") == "7000" and norm("1,234,567") == "1234567"
    assert norm("2,5 kg") == "2,5 kg" and norm("10, 20 and 30") == "10, 20 and 30"    # not thousands groups
    assert norm("a", source=True) == norm("a", source=False)


def test_a_zulu_unit_that_mirrors_the_english_number_passes_the_literal_gate():
    from mathula_tv.episode_scoring import _score_group

    group = _score_group({"group_id": "u", "source_text": "He received about 7,000 rand.",
                          "spoken_text": "Kusolwa ukuthi wathola cishe amarandi ayizi-7,000 kuphela.", "start_ms": 0, "end_ms": 1000,
                          "measured_ms": 1000, "variant_id": "agent"}, known_tokens=set(), labels=[], rush_flag_pct=10.0)
    assert group["gates"]["required_literals"]["failed"] is False
