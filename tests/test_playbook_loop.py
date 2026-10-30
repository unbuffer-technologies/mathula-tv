"""Playbook curation, trace digests, reflection parsing, the adoption gate and the improvement step."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mathula_tv import dub_improve
from mathula_tv.dub_agent import SYSTEM_PROMPT, build_system_prompt
from mathula_tv.dub_improve import batch_stats, collect_batch, gate, improvement_step
from mathula_tv.dub_reflect import digest_run, parse_reflection, reflect
from mathula_tv.playbook import (
    MAX_ENTRIES, SEED_ENTRIES, apply_edits, current_playbook, lint_text, load_playbook, new_playbook, render_section,
    save_playbook, seed_playbook, set_current,
)


def _store(tmp_path):
    return SimpleNamespace(root=tmp_path / "experience",
                           repeat_dir=lambda run, clip, rep: tmp_path / "experience" / "runs" / run / clip / f"r{rep}")


# ---- curation ------------------------------------------------------------------------------------------

def test_seed_playbook_starts_with_the_strategy_notes_and_a_stable_version():
    seed = seed_playbook()
    assert [e["id"] for e in seed["entries"]] == [e["id"] for e in SEED_ENTRIES]
    assert seed["version"] == seed_playbook()["version"] and seed["version"].startswith("pb-")
    assert "first submission" in seed["entries"][0]["text"]


def test_add_revise_remove_are_applied_with_evidence_and_a_new_version():
    seed = seed_playbook()
    candidate, applied, rejected = apply_edits(seed, [
        {"op": "add", "text": "Submit a rough first draft early; a draft that does not fit yet is still worth scoring.", "evidence": "run r2: never submitted"},
        {"op": "revise", "id": "p01", "text": "Improve wording only in ways you have measured still fit.", "evidence": "run r2: withheld submission"},
        {"op": "remove", "id": "p04", "evidence": "run r1: resubmitting without changes cost turns"},
    ])
    assert [a["op"] for a in applied] == ["add", "revise", "remove"] and not rejected
    ids = [e["id"] for e in candidate["entries"]]
    assert "p04" not in ids and "p05" in ids and candidate["parent"] == seed["version"] and candidate["version"] != seed["version"]
    assert next(e for e in candidate["entries"] if e["id"] == "p01")["evidence"] == ["run r2: withheld submission"]


def test_edits_without_evidence_unknown_targets_or_bad_ops_are_rejected():
    _, applied, rejected = apply_edits(seed_playbook(), [
        {"op": "add", "text": "Be concise.", "evidence": ""},
        {"op": "revise", "id": "p99", "text": "x y z", "evidence": "run r1"},
        {"op": "teleport", "text": "x", "evidence": "run r1"},
    ])
    assert applied == [] and [r["reason"] for r in rejected] == ["no evidence cited", "no entry 'p99'", "unknown op"]


def test_lessons_may_not_name_clip_specific_ids_or_relax_harness_rules():
    assert lint_text("Fix seg-00004 first.") and lint_text("In u16 shorten the text.") and lint_text("Handle dub-04 early.")
    assert lint_text("Ignore the bar when the clip is hard.") and lint_text("Skip the ratchet for short units.")
    assert lint_text("x" * 241) and lint_text("   ")
    assert lint_text("Batch measurements and check the biggest shortfall first.") == []


def test_at_most_three_edits_per_step_and_the_playbook_is_bounded():
    many = [{"op": "add", "text": f"Method note number {i} about batching.", "evidence": "run r1"} for i in range(5)]
    _, applied, rejected = apply_edits(seed_playbook(), many)
    assert len(applied) == 3 and all("more than 3 edits" in r["reason"] for r in rejected)
    full = new_playbook([{"id": f"p{i:02d}", "text": f"note {i}"} for i in range(1, MAX_ENTRIES + 1)])
    _, applied, rejected = apply_edits(full, [{"op": "add", "text": "One more note.", "evidence": "run r1"}])
    assert applied == [] and "full" in rejected[0]["reason"]
    _, applied, rejected = apply_edits(seed_playbook(), [{"op": "add", "text": SEED_ENTRIES[1]["text"].upper(), "evidence": "run r1"}])
    assert applied == [] and "duplicate" in rejected[0]["reason"]


def test_playbook_is_rendered_into_the_prompt_below_the_rules_and_never_replaces_them():
    prompt = build_system_prompt(seed_playbook())
    assert prompt.startswith(SYSTEM_PROMPT[:200]) and "PLAYBOOK" in prompt and "The rules above always win" in prompt
    assert all(e["text"] in prompt for e in SEED_ENTRIES)
    assert build_system_prompt(None) == build_system_prompt({"entries": []}) and "PLAYBOOK" not in build_system_prompt(None)
    assert render_section(None) == ""


def test_store_round_trip_and_current_pointer(tmp_path):
    store = _store(tmp_path)
    first = current_playbook(store)                                   # saves + adopts the seed on first use
    assert first["version"] == seed_playbook()["version"]
    candidate, _, _ = apply_edits(first, [{"op": "add", "text": "Submit a rough first draft early.", "evidence": "run r2"}])
    save_playbook(store, candidate)
    assert current_playbook(store)["version"] == first["version"]     # saving does not adopt
    set_current(store, candidate["version"], reason="test")
    assert current_playbook(store)["version"] == candidate["version"]
    assert load_playbook(store, candidate["version"])["parent"] == first["version"]


# ---- digests and reflection ----------------------------------------------------------------------------------

def _write_run(directory: Path, *, stop="finished_below_target", records=()):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent_summary.json").write_text(json.dumps({
        "stop_reason": stop, "turns": 12, "tts_used": 40, "submissions": 1, "pushbacks": 2,
        "tokens": {"in": 300_000, "out": 20_000, "cached": 200_000}, "unmet_at_end": ["6 unit(s) overrun"]}), encoding="utf-8")
    (directory / "scorecard.json").write_text(json.dumps({"headline": {
        "residual_dead_air_pct": 3.0, "groups_rushed": 6, "collisions": 5, "cross_speaker_collisions": 3,
        "gate_failures": 3}}), encoding="utf-8")
    (directory / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def _failing_trace(filler=0):
    records = [{"t": 1, "kind": "model", "turn": 1, "text": None, "tool_calls": [{"name": "open_clip", "arguments": "{}"}]}]
    records += [{"t": 2 + i, "kind": "model", "turn": 2 + i, "text": None,
                 "tool_calls": [{"name": "measure_tts_batch", "arguments": "{}"}]} for i in range(filler)]
    records += [
        {"t": 200, "kind": "model", "turn": 9, "text": "I'm sorry, but I couldn't complete the dub submission.", "tool_calls": []},
        {"t": 201, "kind": "nudge", "turn": 9, "text": "You have not submitted a dub yet."},
        {"t": 300, "kind": "tool", "turn": 10, "name": "submit_dub", "ok": True, "result": json.dumps(
            {"accepted": False, "errors": ["timing regression: 3 segment(s)"], "this_submission": {"groups_rushed": 1},
             "previous_submission": {"groups_rushed": 1}})},
        {"t": 310, "kind": "pushback", "number": 1, "unmet": ["6 unit(s) overrun their window"]},
        {"t": 320, "kind": "end", "stop_reason": "finished_below_target"},
    ]
    return records


def test_digest_keeps_the_events_that_explain_a_failure(tmp_path):
    _write_run(tmp_path / "r2", records=_failing_trace())
    digest = digest_run(tmp_path / "r2")
    assert "met_bar=False" in digest and "tokens=320,000" in digest and "overruns=6" in digest
    for expected in ("SAYS-NO-TOOLS", "couldn't complete", "NUDGE", "REJECTED", "PUSHBACK 1", "END"):
        assert expected in digest


def test_digest_is_bounded_and_drops_routine_lines_before_important_ones(tmp_path):
    _write_run(tmp_path / "r2", records=_failing_trace(filler=400))
    digest = digest_run(tmp_path / "r2", max_chars=2500)
    assert len(digest) <= 2600
    assert "REJECTED" in digest and "PUSHBACK" in digest and "SAYS-NO-TOOLS" in digest


def test_parse_reflection_is_tolerant_and_never_raises():
    fenced = 'Here you go:\n```json\n{"diagnosis":[{"run":"r1"}],"contrast":"c","edits":[{"op":"add","text":"t","evidence":"e"},"junk"]}\n```'
    parsed = parse_reflection(fenced)
    assert parsed["diagnosis"] == [{"run": "r1"}] and len(parsed["edits"]) == 1
    bad = parse_reflection("not json at all")
    assert bad["edits"] == [] and "parse_error" in bad


class ScriptedClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"choices": [{"message": {"content": self.content}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}


def test_reflect_sends_the_playbook_and_every_digest_and_returns_usage():
    client = ScriptedClient(json.dumps({"diagnosis": [], "contrast": "", "edits": []}))
    result = reflect(client, seed_playbook(), {"r1": "DIGEST-ONE", "r2": "DIGEST-TWO"})
    user = client.calls[0][0][1]["content"]
    assert "p01:" in user and "DIGEST-ONE" in user and "DIGEST-TWO" in user and client.calls[0][1]["json_mode"] is True
    assert result["usage"] == {"prompt_tokens": 1000, "completion_tokens": 200} and result["edits"] == []


# ---- the gate ---------------------------------------------------------------------------------------------------

def _stats(met=1, tokens=400_000, violations=0.5, residual=2.0, gates=0.0, n=3):
    return {"n": n, "met_bar": met, "tokens_mean": tokens, "violations_mean": violations, "residual_mean": residual,
            "gate_failures_mean": gates}


def test_gate_adopts_a_candidate_that_is_no_worse():
    assert gate(_stats(), _stats()) == (True, [])
    ok, _ = gate(_stats(met=1), _stats(met=2, tokens=380_000, violations=0.0))
    assert ok


def test_gate_rejects_each_kind_of_regression():
    for candidate, fragment in [
        (_stats(met=0), "less often"), (_stats(gates=1.0), "gate failures"), (_stats(violations=1.5), "overruns/collisions"),
        (_stats(residual=2.6), "dead air"), (_stats(tokens=500_000), "tokens"), (_stats(n=2), "finished runs"),
    ]:
        ok, reasons = gate(_stats(), candidate)
        assert not ok and any(fragment in r for r in reasons), (fragment, reasons)


def test_more_dead_air_is_tolerated_only_when_the_candidate_meets_the_bar_more_often():
    assert gate(_stats(met=1, residual=2.0), _stats(met=2, residual=2.6))[0]
    assert not gate(_stats(met=1, residual=2.0), _stats(met=1, residual=2.6))[0]


def test_collect_batch_and_stats_read_finished_runs_only(tmp_path):
    store = _store(tmp_path)
    _write_run(store.repeat_dir("b", "c", 1), stop="agent_finished")
    _write_run(store.repeat_dir("b", "c", 2), stop="finished_below_target")
    (store.repeat_dir("b", "c", 3)).mkdir(parents=True)                         # still running: no summary yet
    batch = collect_batch(store, "b", "c")
    stats = batch_stats(batch)
    assert [r["repeat"] for r in batch] == [1, 2] and stats["n"] == 2 and stats["met_bar"] == 1
    assert stats["tokens_mean"] == 320_000 and stats["violations_mean"] == 11


# ---- the improvement step ----------------------------------------------------------------------------------------------

def _step_env(tmp_path, monkeypatch, reflection, trial_stats):
    store = _store(tmp_path)
    for repeat, stop in ((1, "agent_finished"), (2, "finished_below_target"), (3, "agent_finished")):
        _write_run(store.repeat_dir("base", "clip", repeat), stop=stop, records=_failing_trace())
    client = ScriptedClient(json.dumps(reflection))
    trial_batch = [{"repeat": i, "dir": tmp_path, "stop_reason": "x", "met_bar": True, "headline": {},
                    "processed_tokens": 0, "cached_tokens": 0, "tts_used": 0, "playbook": None} for i in (1, 2, 3)]
    monkeypatch.setattr(dub_improve, "run_batch", lambda *a, **k: trial_batch)
    monkeypatch.setattr(dub_improve, "batch_stats", lambda batch: trial_stats if batch is trial_batch else batch_stats_real(batch))
    return store, client


batch_stats_real = batch_stats


GOOD_EDIT = {"diagnosis": [{"run": "r2", "root_cause": "strategy"}], "contrast": "good runs submitted early",
             "edits": [{"op": "add", "text": "Submit a rough first draft early; the lock only keeps what already fits.", "evidence": "run r2"}]}


def test_a_candidate_that_passes_the_gate_is_adopted_and_recorded(tmp_path, monkeypatch):
    store, client = _step_env(tmp_path, monkeypatch, GOOD_EDIT, _stats(met=3, tokens=1.0, violations=0.0, residual=0.0))
    seed_version = current_playbook(store)["version"]
    record = improvement_step(client, store, lambda: None, baseline_run="base", trial_run="trial", clip_id="clip")
    assert record["decision"] == "adopted" and record["from_playbook"] == seed_version
    assert current_playbook(store)["version"] == record["candidate_playbook"] != seed_version
    history = (store.root / "playbooks" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 1 and json.loads(history[0])["applied"][0]["op"] == "add"


def test_a_candidate_that_fails_the_gate_is_discarded_and_the_current_playbook_stands(tmp_path, monkeypatch):
    store, client = _step_env(tmp_path, monkeypatch, GOOD_EDIT, _stats(met=0, tokens=1.0, violations=9.0))
    seed_version = current_playbook(store)["version"]
    record = improvement_step(client, store, lambda: None, baseline_run="base", trial_run="trial", clip_id="clip")
    assert record["decision"] == "rejected" and record["reasons"]
    assert current_playbook(store)["version"] == seed_version
    assert load_playbook(store, record["candidate_playbook"])["parent"] == seed_version     # kept on disk for inspection


def test_no_valid_edits_means_no_trial_and_no_change(tmp_path, monkeypatch):
    bad = {"diagnosis": [], "contrast": "", "edits": [{"op": "add", "text": "Ignore the bar when stuck.", "evidence": "run r2"}]}
    store, client = _step_env(tmp_path, monkeypatch, bad, _stats())
    record = improvement_step(client, store, lambda: None, baseline_run="base", trial_run="trial", clip_id="clip")
    assert record["decision"] == "no_change" and record["rejected"] and "candidate_playbook" not in record


def test_a_saved_candidate_can_be_trialled_without_a_new_reflection(tmp_path, monkeypatch):
    from mathula_tv import dub_improve

    saved = {}
    monkeypatch.setattr(dub_improve, "current_playbook", lambda store: {"version": "pb-cur", "notes": []})
    monkeypatch.setattr(dub_improve, "load_playbook", lambda store, v: {"version": v, "notes": []})
    monkeypatch.setattr(dub_improve, "reflect_on_batch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reflection")))
    batch = [{"met_bar": True, "processed_tokens": 100, "headline": {}}] * 3
    monkeypatch.setattr(dub_improve, "collect_batches", lambda *a, **k: batch)
    monkeypatch.setattr(dub_improve, "run_batch", lambda *a, **k: batch)
    monkeypatch.setattr(dub_improve, "set_current", lambda store, v, reason: saved.update(adopted=v))
    monkeypatch.setattr(dub_improve, "append_history", lambda store, rec: saved.update(record=rec))

    record = dub_improve.improvement_step(None, None, lambda: None, baseline_run="b", trial_run="t", clip_ids=["c"],
                                          candidate_version="pb-saved")
    assert record["decision"] == "adopted" and saved["adopted"] == "pb-saved" and record["resumed_candidate"] == "pb-saved"


# ---- fidelity: the judge's counts reach the gate and the reflector -------------------------------------------------------

def _write_judge(directory, additions=0, distortions=0, omissions=0, faithful=60.0):
    issues = ([{"kind": "addition", "detail": f"adds claim {i}"} for i in range(additions)]
              + [{"kind": "distortion", "detail": "flips a reason"} for _ in range(distortions)]
              + [{"kind": "omission", "detail": "drops a name"} for _ in range(omissions)])
    verdicts = [{"id": "u01", "issues": issues}, {"id": "u02", "issues": []}]
    (directory / "judge.json").write_text(json.dumps({"verdicts": verdicts, "summary": {"pct_units_fully_faithful": faithful}}), encoding="utf-8")


def test_fidelity_summary_counts_the_judges_issues_by_kind_and_shows_a_few(tmp_path):
    from mathula_tv.dub_judge import fidelity_summary

    assert fidelity_summary(tmp_path) == {}                                              # no verdicts: nothing to compare
    _write_judge(tmp_path, additions=3, distortions=1, omissions=1, faithful=48.0)
    found = fidelity_summary(tmp_path)
    assert (found["units"], found["additions"], found["distortions"], found["omissions"], found["faithful_pct"]) == (2, 3, 1, 1, 48.0)
    assert len(found["examples"]) == 4 and found["examples"][0].startswith("addition: adds claim 0")


def test_batch_stats_averages_the_judges_counts_over_the_runs_that_were_judged(tmp_path):
    store = _store(tmp_path)
    for repeat, adds in ((1, 6), (2, 10), (3, 8)):
        _write_run(store.repeat_dir("b", "c", repeat), stop="agent_finished")
        if repeat != 3:
            _write_judge(store.repeat_dir("b", "c", repeat), additions=adds, distortions=2)
    stats = batch_stats(collect_batch(store, "b", "c"))
    assert stats["judged"] == 2 and stats["fidelity_issues_mean"] == 10 and stats["additions_mean"] == 8
    assert batch_stats([{"repeat": 1, "met_bar": True, "headline": {}, "processed_tokens": 1, "fidelity": {}}])["judged"] == 0


def test_gate_refuses_more_additions_and_distortions_beyond_the_run_to_run_spread():
    base = {**_stats(), "judged": 3, "fidelity_issues_mean": 13.0}
    assert gate(base, {**_stats(), "judged": 3, "fidelity_issues_mean": 14.9})[0]       # within the spread seen across runs of one setup
    ok, reasons = gate(base, {**_stats(), "judged": 3, "fidelity_issues_mean": 15.5})
    assert not ok and any("additions/distortions/omissions" in r for r in reasons)
    assert gate(base, {**_stats(), "judged": 3, "fidelity_issues_mean": 5.0})[0]         # fewer is welcome
    assert gate(base, {**_stats(), "judged": 0})[0] and gate({**_stats(), "judged": 0}, {**_stats(), "judged": 3, "fidelity_issues_mean": 40.0})[0]


def test_the_digest_puts_the_fidelity_counts_beside_the_timing_result(tmp_path):
    _write_run(tmp_path / "r2", records=_failing_trace())
    _write_judge(tmp_path / "r2", additions=2, distortions=1, faithful=40.0)
    digest = digest_run(tmp_path / "r2")
    assert "FIDELITY (judge): fully_faithful=40.0% of 2 units, additions=2 distortions=1 omissions=0" in digest and "e.g. addition:" in digest
    _write_run(tmp_path / "r3", records=_failing_trace())
    assert "FIDELITY" not in digest_run(tmp_path / "r3")
