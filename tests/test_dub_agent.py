"""The agent loop's mechanics, against a scripted model and a stub environment (no network, no TTS)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.dub_agent import TOOL_SPECS, build_dispatch, execute_tool, run_agent
from mathula_tv.dub_workbench import BudgetExhausted


@pytest.fixture(autouse=True)
def _legacy_workflow(monkeypatch):
    """These tests script the older measure-first flow: the breadth-first phase is switched off here, and the tests of that phase turn it on explicitly (breadth_first=True)."""
    monkeypatch.setenv("MATHULA_AGENT_BREADTH_FIRST", "0")
    monkeypatch.setenv("MATHULA_AGENT_DRAFT_DEADLINE", "0")          # the draft deadline has its own tests (draft_deadline=True)


class StubStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def repeat_dir(self, run_id: str, clip_id: str, repeat: int) -> Path:
        return self.root / run_id / clip_id / f"r{repeat}"


class StubWorkbench:
    def __init__(self, root: Path, budget: int = 2) -> None:
        self.store = StubStore(root)
        self.episode: dict | None = None
        self.budget = budget
        self.measured = 0

    def check_texts(self, units):
        return {"results": [{"item_id": u.get("item_id"), "ok": True, "problems": []} for u in units]}

    def list_foreign_words(self):
        return {"words": []}

    def try_pronunciation(self, word, spellings):
        return {"word": word, "results": []}

    def use_pronunciation(self, word, spelling):
        return {"ok": True}

    def suggest_spellings(self, word):
        return {"word": word, "results": []}

    def create_loanword(self, word):
        return {"word": word, "zulu_words": []}

    def note_pronunciation(self, word, decision, why):
        return {"ok": True}

    def attested_zulu(self, word, zulu_forms=None):
        return {"english": word, "forms": []}

    def punctuation_examples(self, phrase, limit=4):
        return {"phrase": phrase, "after_a_comma": [], "not_after_a_comma": []}

    def human_translations(self, phrase, limit=5):
        return {"phrase": phrase, "pairs_in_all": 0, "pairs": []}

    def figure_style(self, phrase):
        return {"phrase": phrase, "pairs_checked": 0, "ways_the_figure_is_written": []}

    def say_it(self, text, speaker_id="SPEAKER_00"):
        return {"spoken": text, "changes": [], "still_as_written": []}

    def source_readings(self, segment_id="", phrase=""):
        return {"segments": [], "note": "stub"}

    def use_loanword(self, word, zulu_word):
        return {"ok": True}

    def open_clip(self, clip_id, run_id, repeat=1, tts_budget=None, speed_policy="natural"):
        self.opened_repeat = repeat
        self.episode = {"clip_id": clip_id, "tts_used": 0, "submissions": 0}
        return {"clip_id": clip_id, "segments": [{"segment_id": "s1"}]}

    def measure_tts(self, text, start_ms, end_ms, **_):
        if self.measured >= self.budget:
            raise BudgetExhausted("TTS measurement budget exhausted")
        self.measured += 1
        self.episode["tts_used"] = self.measured
        return {"measured_ms": 1000, "direction": None}

    def measure_tts_batch(self, requests):
        return {}

    def grammar_review(self, item_ids=None):
        return {"units": [], "reviewed": []}

    def estimate_fit(self, requests):
        return {"results": [], "note": "stub"}

    def timeline_fit(self, items):
        return {"sentences": [], "note": "stub"}

    def preview_turn_padding(self, members, last_free_gap_ms=0):
        return {"mechanism": "none"}

    def get_submission(self, unit_ids=None):
        return {"units": [], "count": 0}

    def check_text(self, zulu_text, source_text):
        return {"ok": True}

    def recall_lessons(self, limit=25):
        return {"count": 0}

    def submit_dub(self, items, approach_notes="", partial=False, remove=None):
        self.episode["submissions"] += 1
        self.submitted_items = list(items)
        return {"accepted": True}


def _call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _reply(content=None, calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


class ScriptedClient:
    def __init__(self, replies):
        self.config = SimpleNamespace(deployment="scripted")
        self.replies = list(replies)
        self.seen: list[list[dict]] = []
        self.tools_seen: list[list[dict] | None] = []

    def complete(self, messages, **kwargs):
        self.seen.append([dict(m) for m in messages])
        self.tools_seen.append(kwargs.get("tools"))
        return self.replies.pop(0)


def _run(tmp_path, replies, **kwargs):
    client = ScriptedClient(replies)
    workbench = StubWorkbench(tmp_path)
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", **kwargs)
    return summary, client, workbench


def test_every_tool_spec_has_a_dispatch_entry(tmp_path):
    dispatch = build_dispatch(StubWorkbench(tmp_path), clip_id="c1", run_id="r", tts_budget=5)
    assert {spec["function"]["name"] for spec in TOOL_SPECS} == set(dispatch)


def test_agent_calls_tools_then_submits_then_finishes(tmp_path):
    summary, client, _ = _run(tmp_path, [
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "measure_tts", {"text": "x", "start_ms": 0, "end_ms": 1000})]),
        _reply(calls=[_call("c", "submit_dub", {"items": []})]),
        _reply(content="done"),
    ])
    assert summary["stop_reason"] == "agent_finished"
    assert summary["turns"] == 4
    assert summary["tool_calls"] == {"open_clip": 1, "measure_tts": 1, "submit_dub": 1}
    # the observation from turn 2's tool call is fed back to the model on turn 3
    observed = [m for m in client.seen[2] if m["role"] == "tool"]
    assert json.loads(observed[-1]["content"])["measured_ms"] == 1000
    assert (tmp_path / "r" / "c1" / "r1" / "trace.jsonl").is_file()


def test_tool_failure_is_an_observation_the_agent_can_recover_from(tmp_path):
    summary, client, _ = _run(tmp_path, [
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "measure_tts", {"text": "x", "start_ms": 0, "end_ms": 1})]),
        _reply(calls=[_call("c", "measure_tts", {"text": "y", "start_ms": 0, "end_ms": 1})]),
        _reply(calls=[_call("d", "measure_tts", {"text": "z", "start_ms": 0, "end_ms": 1})]),
        _reply(calls=[_call("e", "submit_dub", {"items": []})]),
        _reply(content="done"),
    ])
    assert summary["stop_reason"] == "agent_finished"
    errors = [json.loads(m["content"]) for m in client.seen[4] if m["role"] == "tool" and "error" in m["content"]]
    assert errors and errors[0]["error"] == "budget_exhausted"


def test_bad_arguments_and_unknown_tools_become_observations(tmp_path):
    dispatch = build_dispatch(StubWorkbench(tmp_path), clip_id="c1", run_id="r", tts_budget=5)
    text, ok = execute_tool(dispatch, "no_such_tool", "{}")
    assert not ok and "unknown tool" in text
    text, ok = execute_tool(dispatch, "check_text", '{"wrong_field": 1}')
    assert not ok and "TypeError" in text
    text, ok = execute_tool(dispatch, "check_text", "not json")
    assert not ok


def test_agent_that_stops_without_submitting_is_nudged_once(tmp_path):
    summary, client, _ = _run(tmp_path, [
        _reply(content="I think it is fine."),
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),
        _reply(content="done"),
    ])
    assert summary["stop_reason"] == "agent_finished"
    assert any("not submitted" in str(m.get("content")) for m in client.seen[1] if m["role"] == "user")


def test_with_no_submission_the_history_stays_bounded_and_no_draft_is_lost(tmp_path):
    """The history must not depend on the agent having submitted: 14 working steps and no submit_dub, with the compaction threshold crossed every turn."""
    from mathula_tv.draft_ledger import LEDGER_NOTE_PREFIX
    from mathula_tv.dub_agent import KEEP_RECENT_STEPS

    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})])]
    script += [_reply(calls=[_call(f"m{i}", "measure_tts", {"text": f"draft number {i}", "start_ms": i * 1000, "end_ms": i * 1000 + 900})]) for i in range(14)]
    script += [_reply(content="done"), _reply(content="still done")]
    summary, client, _ = _run(tmp_path, script, compact_after_tokens=50)
    assert summary["stop_reason"] == "finished_without_submission" and summary["compactions"] > 0
    last = client.seen[-1]
    notes = [m["content"] for m in last if str(m.get("content") or "").startswith(LEDGER_NOTE_PREFIX)]
    assert len(notes) == 1                                                       # one ledger, not one per compaction
    assert "draft number 0" in notes[0]                                         # the earliest draft, dropped from the history long ago, is in the ledger
    visible = json.dumps(last, ensure_ascii=False)
    assert all(f"draft number {i}" in visible for i in range(14))               # every draft is somewhere the model can read: in the ledger, or verbatim in the last steps
    assert sum(1 for m in last if m["role"] == "assistant") <= KEEP_RECENT_STEPS + 4         # bounded by construction: the opening step, the last few steps and the closing replies
    assert len(client.seen[-1]) < len(client.seen[10]) + 4                      # and not growing turn after turn


def _names(tools):
    return {t["function"]["name"] for t in tools or []}


def test_with_the_tool_set_swapped_the_measurement_tools_are_not_offered_until_a_complete_submission_exists(tmp_path):
    from mathula_tv.dub_agent import DRAFT_PHASE_NOTE, MEASURE_TOOLS, REFINE_PHASE_NOTE

    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("e", "estimate_fit", {"requests": []})]), _reply(calls=[_call("s", "submit_dub", {"items": []})]),
              _reply(calls=[_call("m", "measure_tts", {"text": "x", "start_ms": 0, "end_ms": 1000})]), _reply(content="done")]
    summary, client, workbench = _run(tmp_path, script, breadth_first=True, stable_tools=False)          # the legacy mode: the declarations themselves change
    offered = [_names(t) for t in client.tools_seen]
    assert not (offered[0] & MEASURE_TOOLS) and not (offered[1] & MEASURE_TOOLS) and not (offered[2] & MEASURE_TOOLS)       # the draft phase: turns 1-3, before any submission
    assert MEASURE_TOOLS <= offered[3] and MEASURE_TOOLS <= offered[4]                                                      # the refine phase: offered from the turn after the first accepted submission
    assert {"open_clip", "estimate_fit", "timeline_fit", "check_text", "submit_dub", "say_it"} <= offered[0]                # everything else stays: the free tools are exactly what the draft needs
    saw = lambda turn, note: any(m.get("content") == note for m in client.seen[turn - 1] if m["role"] == "user")           # noqa: E731
    assert saw(1, DRAFT_PHASE_NOTE) and not saw(4, REFINE_PHASE_NOTE) is False and saw(4, REFINE_PHASE_NOTE)               # told on turn 1; told again that they are offered on turn 4
    assert summary["stop_reason"] == "agent_finished" and workbench.episode["submissions"] == 1
    trace = (tmp_path / "r" / "c1" / "r1" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count('"kind": "draft_phase"') == 1 and trace.count('"kind": "refine_phase"') == 1                         # each announced once


def test_the_tool_declarations_are_the_same_on_every_turn_and_the_phases_restrict_calls_instead(tmp_path):
    from mathula_tv.dub_agent import DRAFT_REFUSAL, MEASURE_TOOLS

    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("m", "measure_tts", {"text": "x", "start_ms": 0, "end_ms": 1000})]),
              _reply(calls=[_call("s", "submit_dub", {"items": []})]), _reply(calls=[_call("m2", "measure_tts", {"text": "y", "start_ms": 0, "end_ms": 1000})]), _reply(content="done")]
    summary, client, workbench = _run(tmp_path, script, breadth_first=True)
    offered = [json.dumps(t, sort_keys=True) for t in client.tools_seen]
    assert len(set(offered)) == 1 and MEASURE_TOOLS <= _names(client.tools_seen[0])         # one declaration set for the whole run, measurement tools included
    refused = [m for m in client.seen[2] if m["role"] == "tool" and "not_available_yet" in m["content"]]
    assert refused and json.loads(refused[0]["content"])["detail"] == DRAFT_REFUSAL         # a draft-phase call is refused, free
    assert not any("not_available_yet" in m["content"] for m in client.seen[4] if m["role"] == "tool" and m["tool_call_id"] == "m2")      # after the first submission the same tool answers
    cache = summary["cache"]
    assert summary["stable_tools"] is True and cache["turns"] == 5 and "tools_changed" not in cache["misses_by_cause"] and len(cache["header_hashes"]) == 1


def test_a_call_outside_the_final_phase_is_refused_while_the_declarations_stay_put(tmp_path):
    from mathula_tv.dub_agent import FINAL_REFUSAL

    script = [_heavy(_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]))]
    script += [_heavy(_reply(calls=[_call(f"m{i}", "measure_tts", {"text": f"draft {i}", "start_ms": 0, "end_ms": 1000})])) for i in range(17)]          # turn 18 is the first of the final phase
    script += [_heavy(_reply(calls=[_call("s", "submit_dub", {"items": []})])), _heavy(_reply(content="done"))]
    summary, client, workbench = _run(tmp_path, script, max_total_tokens=1_000_000)
    assert len({json.dumps(t, sort_keys=True) for t in client.tools_seen}) == 1                 # every turn, the final one included, declares the same tools
    refused = [m for m in client.seen[18] if m["role"] == "tool" and "not_available_now" in m["content"]]
    assert refused and json.loads(refused[-1]["content"])["detail"] == FINAL_REFUSAL
    assert workbench.episode["submissions"] == 1 and summary["budget"]["phase_reached"] == "final"


def test_a_measurement_called_in_the_draft_phase_anyway_is_refused_for_free_and_its_draft_is_kept(tmp_path):
    from mathula_tv.dub_agent import DRAFT_REFUSAL

    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("m", "measure_tts", {"text": "a draft I wanted measured", "start_ms": 0, "end_ms": 1000})]),
              _reply(content="done"), _reply(content="still done")]                        # no submission: a full one would (rightly) replace the drafts as the record
    summary, client, workbench = _run(tmp_path, script, breadth_first=True)
    answer = json.loads(next(m["content"] for m in client.seen[2] if m["role"] == "tool" and "not_available_yet" in m["content"]))
    assert answer["error"] == "not_available_yet" and answer["detail"] == DRAFT_REFUSAL
    assert workbench.episode["tts_used"] == 0                                              # nothing was spent on the refused call
    state = json.loads((tmp_path / "r" / "c1" / "r1" / "agent_state.json").read_text(encoding="utf-8"))
    assert "a draft I wanted measured" in json.dumps(state["ledger"], ensure_ascii=False)  # the draft it carried is still recorded


def test_a_repair_run_and_the_opt_out_have_no_draft_phase(tmp_path):
    from mathula_tv.dub_agent import MEASURE_TOOLS

    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(content="done"), _reply(content="still done")]
    _, off, _ = _run(tmp_path / "a", script, breadth_first=False)
    assert MEASURE_TOOLS <= _names(off.tools_seen[0])                                      # the opt-out: the old flow, every tool from turn 1
    seeded = ScriptedClient([_reply(content="done")])
    workbench = StubWorkbench(tmp_path / "b")
    workbench.open_clip("c1", "r")                                                         # a repair run starts accepted: its seed was scored already
    workbench.episode["submissions"] = 1
    run_agent(seeded, workbench, clip_id="c1", run_id="r", breadth_first=True, seed_items=None)
    assert MEASURE_TOOLS <= _names(seeded.tools_seen[0])


def test_the_environment_switches_the_phase_off_and_an_explicit_choice_wins(tmp_path, monkeypatch):
    from mathula_tv.dub_agent import MEASURE_TOOLS

    monkeypatch.delenv("MATHULA_AGENT_BREADTH_FIRST", raising=False)
    _, default, _ = _run(tmp_path / "a", [_reply(content="done"), _reply(content="done")], stable_tools=False)
    assert not (MEASURE_TOOLS & _names(default.tools_seen[0]))                             # the default is on
    monkeypatch.setenv("MATHULA_AGENT_BREADTH_FIRST", "0")
    _, off, _ = _run(tmp_path / "b", [_reply(content="done"), _reply(content="done")])
    assert MEASURE_TOOLS <= _names(off.tools_seen[0])
    _, forced, _ = _run(tmp_path / "c", [_reply(content="done"), _reply(content="done")], breadth_first=True, stable_tools=False)
    assert not (MEASURE_TOOLS & _names(forced.tools_seen[0]))


def _heavy(reply, prompt=50_000):
    """A reply that reports a large prompt, so a modest ceiling is crossed in a few turns."""
    return {**reply, "usage": {"prompt_tokens": prompt, "completion_tokens": 10}}


def _segments(workbench, windows):
    workbench.segments = [{"segment_id": f"s{i + 1}", "start_ms": a, "end_ms": b, "speaker_id": "SPEAKER_00"} for i, (a, b) in enumerate(windows)]


def test_empty_replies_saved_in_a_checkpoint_are_dropped_when_it_is_loaded_and_real_ones_are_kept():
    from mathula_tv.dub_agent import drop_empty_replies

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"},
                {"role": "assistant", "content": None, "_raw": [{"thoughtSignature": "x"}]},                      # a cut-off reply: nothing said, nothing called
                {"role": "assistant", "content": "  "},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "a", "content": "r"}, {"role": "assistant", "content": "a real answer"}]
    kept = drop_empty_replies(messages)
    assert [m["role"] for m in kept] == ["system", "user", "assistant", "tool", "assistant"] and kept[2].get("tool_calls") and kept[-1]["content"] == "a real answer"


def _cut_off():
    """A reply that ran into the output limit: no text, no tool call, finish 'length' (what prod-114 got twice when it tried to submit the whole clip)."""
    return {"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}], "usage": {"prompt_tokens": 100, "completion_tokens": 16000}}


def test_a_reply_cut_off_by_the_output_limit_is_not_the_agent_finishing_and_it_may_try_again(tmp_path):
    from mathula_tv.dub_agent import TRUNCATED_NOTE

    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _cut_off(), _reply(calls=[_call("s", "submit_dub", {"items": []})]), _reply(content="done")]
    summary, client, workbench = _run(tmp_path, script)
    assert summary["stop_reason"] == "agent_finished" and workbench.episode["submissions"] == 1
    retry = client.seen[2]                                                                    # what the model saw on the turn after the cut-off
    assert any(m.get("content") == TRUNCATED_NOTE for m in retry) and not any("not submitted a dub yet" in str(m.get("content")) for m in retry)     # told it was cut off, not nudged as if it had stopped
    assert not any(m["role"] == "assistant" and not m.get("tool_calls") and m.get("content") is None for m in retry)                                # the empty cut-off reply is not kept in the history
    assert '"kind": "truncated"' in (tmp_path / "r" / "c1" / "r1" / "trace.jsonl").read_text(encoding="utf-8")


def test_replies_cut_off_again_and_again_end_the_run_with_that_reason_and_the_drafts_are_still_submitted(tmp_path):
    from mathula_tv.dub_agent import MAX_CONSECUTIVE_TRUNCATIONS

    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("m", "measure_tts", {"text": "a draft", "start_ms": 0, "end_ms": 1000})])]
    script += [_cut_off() for _ in range(MAX_CONSECUTIVE_TRUNCATIONS + 1)]
    client = ScriptedClient(script)
    workbench = StubWorkbench(tmp_path)
    _segments(workbench, [(0, 1000)])
    summary = run_agent(client, workbench, clip_id="c1", run_id="r")
    assert summary["stop_reason"] == "output_limit"
    assert summary["budget"]["auto_submit"]["submitted"] is True and workbench.submitted_items[0]["zulu_text"] == "a draft"


def test_a_cut_off_that_is_followed_by_real_work_does_not_count_against_the_next_one(tmp_path):
    from mathula_tv.dub_agent import MAX_CONSECUTIVE_TRUNCATIONS

    work = lambda i: _reply(calls=[_call(f"m{i}", "measure_tts", {"text": f"draft {i}", "start_ms": 0, "end_ms": 1000})])           # noqa: E731
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})])]
    for i in range(MAX_CONSECUTIVE_TRUNCATIONS + 1):
        script += [_cut_off(), work(i)]                                                      # more cut-offs than the limit in all, but never that many in a row
    script += [_reply(calls=[_call("s", "submit_dub", {"items": []})]), _reply(content="done")]
    summary, _, _ = _run(tmp_path, script)
    assert summary["stop_reason"] == "agent_finished"


def test_the_budget_has_phases_the_agent_is_warned_and_the_last_turns_can_only_submit(tmp_path):
    """An agent that behaves: it is told at the soft line, the final turns offer only submitting, and it submits inside the protected reserve."""
    from mathula_tv.budget_governor import FINAL_TOOLS

    script = [_heavy(_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]))]
    script += [_heavy(_reply(calls=[_call(f"m{i}", "measure_tts", {"text": f"draft {i}", "start_ms": 0, "end_ms": 1000})])) for i in range(16)]
    script += [_heavy(_reply(calls=[_call("s", "submit_dub", {"items": []})])), _heavy(_reply(content="done"))]
    summary, client, workbench = _run(tmp_path, script, max_total_tokens=1_000_000, stable_tools=False)          # the legacy mode swaps the declared set for the final phase
    names = [{t["function"]["name"] for t in tools or []} for tools in client.tools_seen]
    assert names[0] > FINAL_TOOLS and names[16] > FINAL_TOOLS                           # the work: every tool
    assert names[17] == FINAL_TOOLS                                                     # the final phase: submitting only
    def announced(turn, prefix):                                                       # was the note in the messages the model saw on this turn (1-based)?
        return any(m["role"] == "user" and str(m.get("content")).startswith(prefix) for m in client.seen[turn - 1])

    assert not announced(14, "Budget:") and announced(15, "Budget:")                   # the soft line is crossed at 70% of the ceiling
    assert not announced(17, "FINAL PHASE") and announced(18, "FINAL PHASE")           # and the final phase opens before the reserve is touched
    assert workbench.episode["submissions"] == 1 and summary["budget"]["phase_reached"] == "final" and "auto_submit" not in summary["budget"]
    trace = (tmp_path / "r" / "c1" / "r1" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count('"kind": "budget_soft"') == 1 and trace.count('"kind": "budget_final"') == 1          # each phase announced once


def test_a_run_that_ends_with_nothing_submitted_has_its_own_drafts_submitted_when_they_cover_the_clip(tmp_path):
    """An agent that ignores the final phase: the harness submits the drafts it recorded, because they cover every segment."""
    script = [_heavy(_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]))]
    script += [_heavy(_reply(calls=[_call(f"m{i}", "measure_tts", {"text": "the only draft", "start_ms": 0, "end_ms": 1000})])) for i in range(40)]
    client = ScriptedClient(script)
    workbench = StubWorkbench(tmp_path)
    _segments(workbench, [(0, 1000)])
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", max_total_tokens=1_000_000)
    assert summary["stop_reason"] == "token_budget"
    assert workbench.episode["submissions"] == 1 and workbench.submitted_items[0]["zulu_text"] == "the only draft" and workbench.submitted_items[0]["source_segment_ids"] == ["s1"]
    assert summary["budget"]["auto_submit"]["submitted"] is True and summary["budget"]["final_turns"] == 3          # the final phase was bounded: 3 turns, then the harness acted
    assert not (tmp_path / "r" / "c1" / "r1" / "unsubmitted_drafts.json").exists()


def test_drafts_that_do_not_cover_the_clip_are_not_submitted_but_are_kept(tmp_path):
    script = [_heavy(_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]))]
    script += [_heavy(_reply(calls=[_call(f"m{i}", "measure_tts", {"text": "half of it", "start_ms": 0, "end_ms": 1000})])) for i in range(40)]
    client = ScriptedClient(script)
    workbench = StubWorkbench(tmp_path)
    _segments(workbench, [(0, 1000), (1500, 2500)])                                      # the agent never drafted the second segment
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", max_total_tokens=1_000_000)
    outcome = summary["budget"]["auto_submit"]
    assert outcome["submitted"] is False and outcome["uncovered"] == ["s2"] and not workbench.episode["submissions"]
    kept = json.loads((tmp_path / "r" / "c1" / "r1" / "unsubmitted_drafts.json").read_text(encoding="utf-8"))
    assert "half of it" in json.dumps(kept["drafts"], ensure_ascii=False) and kept["why"]["uncovered"] == ["s2"]


def test_agent_that_ignores_the_nudge_ends_without_submission(tmp_path):
    summary, _, _ = _run(tmp_path, [_reply(content="done"), _reply(content="still done")])
    assert summary["stop_reason"] == "finished_without_submission"


def test_turn_and_token_ceilings_end_the_run(tmp_path):
    looping = [_reply(calls=[_call(str(i), "recall_lessons", {})]) for i in range(10)]
    summary, _, _ = _run(tmp_path, looping, max_turns=3)
    assert summary["stop_reason"] == "max_turns" and summary["turns"] == 3
    looping = [_reply(calls=[_call(str(i), "recall_lessons", {})]) for i in range(10)]
    summary, _, _ = _run(tmp_path / "t", looping, max_total_tokens=250)
    assert summary["stop_reason"] == "token_budget"


def test_the_harness_not_the_model_owns_the_clip_and_budget(tmp_path):
    dispatch = build_dispatch(StubWorkbench(tmp_path), clip_id="c1", run_id="r", tts_budget=5)
    text, ok = execute_tool(dispatch, "open_clip", '{"clip_id": "another-clip"}')
    assert not ok and "this episode is for clip" in text


def test_responses_client_translates_history_and_keeps_reasoning_items():
    from mathula_tv.dub_agent import ResponsesClient

    api_output = [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "function_call", "call_id": "call_9", "name": "check_text", "arguments": '{"zulu_text": "a", "source_text": "b"}'},
    ]
    shaped = ResponsesClient.to_chat_shape({"output": api_output, "usage": {"input_tokens": 50, "output_tokens": 7}})
    message = shaped["choices"][0]["message"]
    assert message["tool_calls"][0]["id"] == "call_9" and message["tool_calls"][0]["function"]["name"] == "check_text"
    assert shaped["usage"] == {"prompt_tokens": 50, "completion_tokens": 7, "cached_tokens": 0}

    history = [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "go"},
        {k: v for k, v in message.items()},
        {"role": "tool", "tool_call_id": "call_9", "content": '{"ok": true}'},
    ]
    items = ResponsesClient.to_input(history)
    assert items[:2] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    assert items[2:4] == api_output            # the model's reasoning rides along, verbatim
    assert items[4] == {"type": "function_call_output", "call_id": "call_9", "output": '{"ok": true}'}


def test_responses_client_final_text_and_flat_tool_schema():
    from mathula_tv.dub_agent import ResponsesClient

    shaped = ResponsesClient.to_chat_shape({"status": "completed", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "all "}, {"type": "output_text", "text": "done"}]}]})
    assert shaped["choices"][0]["message"]["content"] == "all done"
    assert "tool_calls" not in shaped["choices"][0]["message"]
    flat = ResponsesClient.to_tools(TOOL_SPECS)
    assert flat[0]["type"] == "function" and flat[0]["name"] == "open_clip" and "parameters" in flat[0]


def test_repeat_index_reaches_the_environment_so_parallel_repeats_do_not_collide(tmp_path):
    client = ScriptedClient([
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),
        _reply(content="done"),
    ])
    workbench = StubWorkbench(tmp_path)
    run_agent(client, workbench, clip_id="c1", run_id="r", repeat=3)
    assert workbench.opened_repeat == 3
    assert (tmp_path / "r" / "c1" / "r3" / "trace.jsonl").is_file()


def test_system_prompt_ranks_overrun_above_dead_air():
    from mathula_tv.dub_agent import SYSTEM_PROMPT

    assert "overrun is WORSE than dead air" in SYSTEM_PROMPT


def test_a_long_clip_open_result_is_not_truncated_but_a_runaway_result_is(tmp_path):
    big = {"segments": ["x" * 40] * 450}                      # ~19k chars, like a 28-segment clip's open_clip
    text, ok = execute_tool({"t": lambda: big}, "t", "{}")
    assert ok and "truncated" not in text
    text, ok = execute_tool({"t": lambda: {"blob": "y" * 200_000}}, "t", "{}")
    assert ok and "[truncated" in text


# ---- the harness verifies "done" against the agent's own score -------------------------------

def _score(residual, overrun=0, gates=0, turns=()):
    return {"accepted": True, "headline": {"residual_dead_air_pct": residual, "groups_rushed": overrun,
                                          "gate_failures": gates},
            "turn_padding": list(turns), "flagged_units": []}


class ScoringWorkbench(StubWorkbench):
    """Stub whose successive submit_dub calls return scripted scores."""

    def __init__(self, root, scores, budget=50):
        super().__init__(root, budget)
        self.scores = list(scores)
        self.last_submit = None

    def open_clip(self, clip_id, run_id, repeat=1, tts_budget=None, speed_policy="natural"):
        out = super().open_clip(clip_id, run_id, repeat, tts_budget, speed_policy)
        self.episode.update(tts_budget=tts_budget or 100)
        return out

    def submit_dub(self, items, approach_notes=""):
        self.episode["submissions"] += 1
        self.last_submit = self.scores.pop(0)
        return self.last_submit


def _agent_script(*finishes):
    """open_clip, submit, 'done' text; then for each further round: submit again, 'done' text."""
    replies = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]),
               _reply(calls=[_call("s0", "submit_dub", {"items": []})]), _reply(content="done")]
    for i in range(finishes[0] if finishes else 0):
        replies += [_reply(calls=[_call(f"s{i + 1}", "submit_dub", {"items": []})]), _reply(content="done again")]
    return replies


def test_unmet_targets_reads_the_harness_score_not_the_agents_claim():
    from mathula_tv.dub_agent import unmet_targets

    assert unmet_targets(_score(1.0), max_dead_air_pct=2.0) == []
    assert unmet_targets(None, max_dead_air_pct=2.0) == []
    unmet = unmet_targets(_score(9.3, overrun=2, gates=1), max_dead_air_pct=2.0)
    assert len(unmet) == 3 and any("9.3%" in u for u in unmet)


def test_agent_that_stops_below_the_bar_is_sent_back_and_can_fix_it(tmp_path):
    workbench = ScoringWorkbench(tmp_path, [_score(9.0, turns=[{"turn_id": "dub-04", "units": ["dub-04"], "residual_ms": 12000}]),
                                            _score(1.0)])
    client = ScriptedClient(_agent_script(1))
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", tts_budget=50)
    assert summary["pushbacks"] == 1 and summary["stop_reason"] == "agent_finished" and summary["unmet_at_end"] == []
    pushed = [m["content"] for m in client.seen[3] if m["role"] == "user" and "Not done yet" in str(m.get("content"))]
    assert pushed and "dub-04" in pushed[0] and "12.0 s" in pushed[0]


def test_pushbacks_are_bounded_and_the_run_ends_below_target(tmp_path):
    workbench = ScoringWorkbench(tmp_path, [_score(9.0), _score(8.0), _score(7.0)])
    client = ScriptedClient(_agent_script(2))
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", tts_budget=50, max_pushbacks=2)
    assert summary["pushbacks"] == 2 and summary["stop_reason"] == "finished_below_target"
    assert summary["unmet_at_end"]


def test_no_pushback_when_the_bar_is_met_or_pushbacks_are_disabled(tmp_path):
    met = run_agent(ScriptedClient(_agent_script(0)), ScoringWorkbench(tmp_path / "a", [_score(0.5)]),
                    clip_id="c1", run_id="r", tts_budget=50)
    assert met["pushbacks"] == 0 and met["stop_reason"] == "agent_finished"
    off = run_agent(ScriptedClient(_agent_script(0)), ScoringWorkbench(tmp_path / "b", [_score(9.0)]),
                    clip_id="c1", run_id="r", tts_budget=50, max_pushbacks=0)
    assert off["pushbacks"] == 0 and off["stop_reason"] == "finished_below_target"


def test_no_pushback_when_the_measurement_budget_is_spent(tmp_path):
    workbench = ScoringWorkbench(tmp_path, [_score(9.0)])
    original_open = workbench.open_clip

    def spent(*a, **k):
        out = original_open(*a, **k)
        workbench.episode["tts_used"] = workbench.episode["tts_budget"]
        return out

    workbench.open_clip = spent
    summary = run_agent(ScriptedClient(_agent_script(0)), workbench, clip_id="c1", run_id="r", tts_budget=50)
    assert summary["pushbacks"] == 0 and summary["stop_reason"] == "finished_below_target"


def test_the_bar_is_in_the_system_prompt():
    from mathula_tv.dub_agent import SYSTEM_PROMPT

    assert "residual_dead_air_pct at or below" in SYSTEM_PROMPT and "{max_dead_air}" not in SYSTEM_PROMPT


# ---- token economy: history compaction, slim observations, cached tokens ----------------------------

def _assistant(call_id, name, raw=None):
    return {"role": "assistant", "content": None, "_raw": raw or [{"type": "function_call", "call_id": call_id, "name": name}],
            "tool_calls": [_call(call_id, name, {})]}


def _tool(call_id, content="result"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _history():
    return [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
        _assistant("o", "open_clip"), _tool("o", "THE CLIP"),
        _assistant("l", "recall_lessons"), _tool("l", "LESSONS"),
        _assistant("m1", "measure_tts_batch"), _tool("m1", "draft 1 measurements"),
        _assistant("s1", "submit_dub"), _tool("s1", "score 1"),
        {"role": "user", "content": "Not done yet. pushback 1"},
        _assistant("m2", "measure_tts_batch"), _tool("m2", "draft 2 measurements"),
        _assistant("c2", "check_text"), _tool("c2", "check"),
        _assistant("s2", "submit_dub"), _tool("s2", "score 2"),
        {"role": "user", "content": "Not done yet. pushback 2"},
        _assistant("m3", "measure_tts_batch"), _tool("m3", "draft 3 measurements"),
    ]


def test_compaction_drops_superseded_steps_but_keeps_clip_lessons_latest_submit_and_what_follows():
    from mathula_tv.dub_agent import COMPACTION_NOTE, compact_history

    compacted, dropped = compact_history(_history())
    texts = [m.get("content") for m in compacted]
    assert dropped == 4                                     # m1, s1, m2, c2 steps
    assert "THE CLIP" in texts and "LESSONS" in texts       # clip + lessons survive
    assert "score 2" in texts and "draft 3 measurements" in texts and "Not done yet. pushback 2" in texts
    assert not any(t in texts for t in ("draft 1 measurements", "score 1", "draft 2 measurements", "check", "Not done yet. pushback 1"))
    assert texts.count(COMPACTION_NOTE) == 1
    assert texts.index(COMPACTION_NOTE) < texts.index("score 2")


def test_compaction_keeps_every_remaining_tool_call_paired_with_its_result():
    from mathula_tv.dub_agent import compact_history

    compacted, _ = compact_history(_history())
    called = {c["id"] for m in compacted if m["role"] == "assistant" for c in m.get("tool_calls") or []}
    answered = {m["tool_call_id"] for m in compacted if m["role"] == "tool"}
    assert called == answered
    raw_ids = {i["call_id"] for m in compacted if m["role"] == "assistant" for i in m.get("_raw") or []}
    assert raw_ids == called                                # the Responses-API view stays consistent too


def test_compaction_is_idempotent_and_does_nothing_before_the_first_submission():
    from mathula_tv.dub_agent import compact_history

    once, _ = compact_history(_history())
    twice, dropped = compact_history(once)
    assert dropped == 0 and [m.get("content") for m in twice] == [m.get("content") for m in once]
    early = _history()[:8]                                   # open, lessons, one measurement step, no submit yet
    same, dropped = compact_history(early)
    assert dropped == 0 and same is early


def test_a_second_submission_replaces_the_first_note_instead_of_stacking_notes():
    from mathula_tv.dub_agent import COMPACTION_NOTE, compact_history

    once, _ = compact_history(_history())
    longer = once + [_assistant("m4", "measure_tts_batch"), _tool("m4", "draft 4"),
                     _assistant("s3", "submit_dub"), _tool("s3", "score 3")]
    again, dropped = compact_history(longer)
    texts = [m.get("content") for m in again]
    assert dropped >= 2 and texts.count(COMPACTION_NOTE) == 1
    assert "score 3" in texts and "score 2" not in texts


def test_run_agent_compacts_once_the_context_is_large_and_logs_it(tmp_path):
    big = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [_call("x", "recall_lessons", {})]}}],
           "usage": {"prompt_tokens": 30_000, "completion_tokens": 10, "cached_tokens": 25_000}}
    replies = [
        _reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("m", "measure_tts", {"text": "x", "start_ms": 0, "end_ms": 1000})]),
        _reply(calls=[_call("s", "submit_dub", {"items": []})]),
        _reply(calls=[_call("m2", "measure_tts", {"text": "y", "start_ms": 0, "end_ms": 1000})]),
        _reply(content="done"),
    ]
    for r in replies:
        r["usage"] = {"prompt_tokens": 30_000, "completion_tokens": 10, "cached_tokens": 25_000}
    client = ScriptedClient(replies)
    workbench = StubWorkbench(tmp_path)
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", compact_after_tokens=20_000)
    assert summary["compactions"] >= 1
    assert summary["tokens"]["cached"] == 5 * 25_000
    last_view = client.seen[-1]
    assert not any(m.get("tool_call_id") == "m" for m in last_view)          # the superseded measurement step is gone


def test_slim_submit_hides_noise_but_the_harness_keeps_the_full_result():
    from mathula_tv.dub_agent import slim_submit

    full = {"accepted": True, "warnings": [], "gates": {"required_literals": [], "restated_reference": ["u3"]},
            "headline": {"residual_dead_air_pct": 3.1, "groups_rushed": 0, "gate_failures": 1, "tokens_in": 0, "review_rate": 0.0,
                         "mean_rush_pct": 0.0, "speed_policy": "natural", "group_count": 12},
            "turn_padding": [{"turn_id": "a", "residual_ms": 0}, {"turn_id": "b", "residual_ms": 900}],
            "flagged_units": [{"item_id": "u3", "gates": ["restated_reference"], "flags": []}]}
    slim = slim_submit(full)
    assert slim["gates"] == {"restated_reference": ["u3"]} and "tokens_in" not in slim["headline"]
    assert [t["turn_id"] for t in slim["turns_ending_early"]] == ["b"] and "warnings" not in slim
    assert slim_submit({"accepted": False, "errors": ["x"]}) == {"accepted": False, "errors": ["x"]}
    assert "tokens_in" in full["headline"]                                   # the original is untouched


def test_prompt_tells_the_agent_that_history_costs_tokens():
    from mathula_tv.dub_agent import SYSTEM_PROMPT

    assert "re-sends your whole history" in SYSTEM_PROMPT and "trimmed from your history" in SYSTEM_PROMPT


# ---- transient network failures are retried, bounded; a crashed run does not end a batch --------------------------------

def test_a_read_timeout_is_retried_then_succeeds(monkeypatch):
    import requests

    from mathula_tv import dub_agent

    class Ok:
        status_code = 200
        text = ""

        def json(self):
            return {"ok": True}

    outcomes = [requests.exceptions.ReadTimeout("hung"), requests.exceptions.ConnectionError("reset"), Ok()]
    sleeps = []

    def fake_post(*a, **kw):
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(dub_agent.requests, "post", fake_post)
    assert dub_agent.post_with_retry("u", {}, {}, timeout=1, retries=3, sleep=sleeps.append) == {"ok": True}
    assert len(sleeps) == 2


def test_timeouts_stop_after_the_retry_ceiling(monkeypatch):
    import pytest
    import requests

    from mathula_tv import dub_agent

    calls = []

    def always_hangs(*a, **kw):
        calls.append(1)
        raise requests.exceptions.ReadTimeout("hung")

    monkeypatch.setattr(dub_agent.requests, "post", always_hangs)
    with pytest.raises(RuntimeError, match="ReadTimeout"):
        dub_agent.post_with_retry("u", {}, {}, timeout=1, retries=2, sleep=lambda s: None)
    assert len(calls) == 3                                          # one try plus two retries, then it gives up


def test_a_crashed_run_is_recorded_as_failed_and_the_batch_continues(tmp_path, monkeypatch):
    from pathlib import Path

    from mathula_tv import dub_improve

    class Store:
        def repeat_dir(self, run_id, clip, repeat):
            return Path(tmp_path) / run_id / clip / f"r{repeat}"

    def flaky(client, workbench, *, repeat, **kw):
        if repeat == 2:
            raise RuntimeError("model call failed after 4 attempts: ReadTimeout")
        return {"stop_reason": "agent_finished"}

    monkeypatch.setattr(dub_improve, "run_agent", flaky)
    monkeypatch.setattr(dub_improve, "collect_batches", lambda *a, **k: [])
    dub_improve.run_batch(object(), Store(), lambda: None, run_id="t", playbook={"version": "pb-x", "notes": []},
                          clip_ids=["c"], repeats=3, judge_meaning=False)
    failed = json.loads((tmp_path / "t" / "c" / "r2" / "agent_summary.json").read_text(encoding="utf-8"))
    assert failed["stop_reason"] == "crashed" and "ReadTimeout" in failed["error"]


# ---- repair mode: start from an earlier submission and change only what misses the bar ------------------------------------

def test_a_repair_run_scores_the_old_submission_and_tells_the_agent_what_to_fix(tmp_path):
    workbench = ScoringWorkbench(tmp_path, [_score(1.0, overrun=1), _score(1.0)])
    seen = []
    original = workbench.submit_dub

    def spy(items, approach_notes=""):
        seen.append((items, approach_notes))
        return original(items, approach_notes)

    workbench.submit_dub = spy
    client = ScriptedClient([_reply(calls=[_call("p", "submit_dub", {"items": [{"item_id": "u1"}]})]),
                             _reply(content="fixed")])
    old = [{"item_id": "u1", "zulu_text": "a"}, {"item_id": "u2", "zulu_text": "b"}]
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", tts_budget=50, seed_items=old)

    assert seen[0][0] == old and "seeded" in seen[0][1]                          # the old units were re-scored first
    task = client.seen[0][1]["content"]
    assert "Repair, do not redo" in task and "overrun" in task and "partial=true" in task
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 2


def test_a_repair_that_already_meets_the_bar_says_so_and_asks_for_no_changes(tmp_path):
    workbench = ScoringWorkbench(tmp_path, [_score(0.5)])
    client = ScriptedClient([_reply(content="nothing to change")])
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", tts_budget=50, seed_items=[{"item_id": "u1"}])
    assert "meets the bar" in client.seen[0][1]["content"] and summary["stop_reason"] == "agent_finished"


class PendingWorkbench(StubWorkbench):
    """A workbench whose pronunciation lab has not been used yet."""

    checked = False

    def pronunciation_check_pending(self):
        return [] if self.checked else ["science", "correctional"]

    def list_foreign_words(self):
        self.checked = True
        return {"words": []}


def test_a_repair_run_is_told_which_units_the_english_changed_under():
    from mathula_tv.dub_agent import _source_correction_note

    class Bench:
        source_corrections = [
            {"segment_id": "seg-08", "token": "Abu Kogo", "replacement": "aboGogo"},                      # already in the earlier submission
            {"segment_id": "seg-10", "token": "Rupert is\u2014", "replacement": ""},                     # new: a false start taken out
        ]
        segments = [{"segment_id": "seg-08", "restored_text": "we want aboGogo not to pay"},
                    {"segment_id": "seg-10", "restored_text": "Leopard Creek is going to pay."}]
        seed_unit_texts = {"u08": "we want aboGogo not to pay",
                           "u10": "Leopard Creek is going to pay. Rupert is going to pay for water and electricity."}

    seed_items = [{"item_id": "u08", "source_segment_ids": ["seg-08"]}, {"item_id": "u10", "source_segment_ids": ["seg-10"]}]
    note = _source_correction_note(Bench(), seed_items)
    assert "leave it out" in note and "(units: u10)" in note and "Abu Kogo" not in note
    assert "u10: the English now ends" in note and "Rupert is going to pay for water" in note       # it also says what the unit used to be written against
    assert "u08" not in note.split("English source of these units")[1]                                  # an unchanged unit is not mentioned
    Bench.seed_unit_texts = None                                                                             # no earlier texts known: all corrections, as before
    assert "Abu Kogo" in _source_correction_note(Bench(), seed_items)


def test_loanwords_come_first_in_the_prompt_and_are_named_in_the_note(tmp_path):
    from mathula_tv import dub_agent

    assert "Loanwords first" in dub_agent.SYSTEM_PROMPT and "isayensi" in dub_agent.SYSTEM_PROMPT

    class Loan(StubWorkbench):
        def loanwords(self):
            return [{"written": "weScience", "english": "science", "zulu": "isayensi"}, {"written": "Technology", "english": "technology", "zulu": "ubuchwepheshe"}]

    note = dub_agent.loanwords_note(Loan(tmp_path))
    assert "Loanwords first" in note and "weScience -> isayensi" in note and "Technology -> ubuchwepheshe" in note
    assert dub_agent.loanwords_note(object()) == ""


def test_the_agent_can_create_and_remember_a_loanword_by_rule_and_the_note_names_words_with_no_zulu_word(tmp_path):
    from mathula_tv import dub_agent

    names = [spec["function"]["name"] for spec in dub_agent.TOOL_SPECS]
    assert "create_loanword" in names and "use_loanword" in names and "create_loanword" in dub_agent.SYSTEM_PROMPT
    assert "listened to or spoken" in dub_agent.SYSTEM_PROMPT          # the loan rules work on the dictionary's sounds: no recogniser, no voice

    class Loan(StubWorkbench):
        def create_loanword(self, word):
            return {"word": word, "zulu_words": []}

        def use_loanword(self, word, zulu_word):
            return {"ok": True}

        def loanwords(self):
            return [{"written": "weScience", "english": "science", "zulu": "isayensi"}, {"written": "ne-medical", "english": "medical", "zulu": ""}]

    tools = dub_agent.build_dispatch(Loan(tmp_path), clip_id="c1", run_id="r", tts_budget=5)
    assert tools["create_loanword"]("medical") == {"word": "medical", "zulu_words": []} and tools["use_loanword"]("medical", "mediki") == {"ok": True}
    note = dub_agent.loanwords_note(Loan(tmp_path))
    assert "weScience -> isayensi" in note and "No everyday Zulu word is known for: ne-medical" in note and "create_loanword" in note


def test_the_loanwords_are_named_once_right_after_the_first_accepted_submission_not_only_at_the_end(tmp_path):
    from mathula_tv import dub_agent

    class Loans(StubWorkbench):
        def loanwords(self):
            return [{"written": "ne-medical", "english": "medical", "zulu": "ezokwelapha"}]

    client = ScriptedClient([
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),
        _reply(content="done"),
        _reply(content="done"),                                                        # the end-of-run reminder still names them once more
    ])
    run_agent(client, Loans(tmp_path), clip_id="c1", run_id="r")
    named = [str(m.get("content")) for m in client.seen[2] if m["role"] == "user" and "first accepted submission" in str(m.get("content"))]
    assert len(named) == 1 and "ne-medical -> ezokwelapha" in named[0]
    assert not [m for m in client.seen[1] if "first accepted submission" in str(m.get("content"))]            # nothing before a submission exists
    assert "common nouns" in dub_agent.SYSTEM_PROMPT and '("Fighters", "medical aid", names)' not in dub_agent.SYSTEM_PROMPT


def test_a_noun_prefix_glued_on_a_word_the_speaker_says_as_it_is_is_named_in_the_early_note(tmp_path):
    from mathula_tv import dub_agent

    class Glued(StubWorkbench):
        def glued_class_prefix_words(self):
            return [{"written": "MaFighter", "speaker_says": "Fighters"}]

    note = dub_agent.glued_prefix_note(Glued(tmp_path))
    assert "MaFighter (the speaker says 'Fighters')" in note and "without the prefix" in note and "ne- or ye-" in note
    assert dub_agent.glued_prefix_note(StubWorkbench(tmp_path)) == ""
    client = ScriptedClient([
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),
        _reply(content="done"),
        _reply(content="done"),
    ])
    run_agent(client, Glued(tmp_path), clip_id="c1", run_id="r")
    named = [str(m.get("content")) for m in client.seen[2] if m["role"] == "user" and "first accepted submission" in str(m.get("content"))]
    assert len(named) == 1 and "MaFighter" in named[0]


def test_the_early_note_waits_for_the_text_when_a_repair_run_is_accepted_from_its_first_turn(tmp_path):
    from mathula_tv import dub_agent

    class Glued(StubWorkbench):
        written = False

        def open_clip(self, clip_id, run_id, repeat=1, tts_budget=None, speed_policy="natural"):
            out = super().open_clip(clip_id, run_id, repeat, tts_budget, speed_policy)
            self.episode = {**self.episode, "dir": str(tmp_path), "submissions": 1}               # accepted from the start, no submission.json yet
            return out

        def submit_dub(self, items, **kw):
            (tmp_path / "submission.json").write_text("{}", encoding="utf-8")
            return super().submit_dub(items, **kw)

        def glued_class_prefix_words(self):
            return [{"written": "MaFighter", "speaker_says": "Fighters"}]

    client = ScriptedClient([
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),
        _reply(content="done"),
        _reply(content="done"),
    ])
    run_agent(client, Glued(tmp_path), clip_id="c1", run_id="r")
    assert not [m for m in client.seen[1] if "first accepted submission" in str(m.get("content"))]            # not before the text exists
    assert len([m for m in client.seen[2] if "first accepted submission" in str(m.get("content"))]) == 1


def test_the_agent_can_ask_what_real_zulu_does_and_is_told_when_the_resources_are_missing(tmp_path):
    from pathlib import Path

    from mathula_tv import attestation, dub_agent
    from mathula_tv.dub_workbench import Workbench

    assert "attested_zulu" in [spec["function"]["name"] for spec in dub_agent.TOOL_SPECS] and "code-switch" in dub_agent.SYSTEM_PROMPT
    note = dub_agent.loanwords_note(type("W", (), {"loanwords": lambda self: [{"written": "ne-provident", "english": "provident", "zulu": ""}]})())
    assert "attested_zulu" in note and "code-switch" in note and "create_loanword" in note
    work = Path(tmp_path)
    directory = attestation.resources_dir(work)
    (directory / "lexicon").mkdir(parents=True)
    (directory / "autshumato.zu.txt").write_text("Ibhasi sekwephuzile .\nIbhasi iyama lapha .\nWenyuka ibhasi waya .\n", encoding="utf-8")
    (directory / "autshumato.en.txt").write_text("The bus is late .\nThe bus stops here .\nHe took the bus .\n", encoding="utf-8")
    (directory / "lexicon" / "nouns.txt").write_text("imali\n", encoding="utf-8")

    class Bench:
        class store:
            root = work / "experience"

        attested_zulu = Workbench.attested_zulu

    answer = Bench().attested_zulu("bus", ["ibhasi", "maBus"])
    assert answer["forms"][0]["form"] == "ibhasi" and [c["corpus_count"] for c in answer["checked_forms"]] == [3, 0]
    attestation._cache.clear()
    empty = Path(tmp_path) / "elsewhere"
    Bench.store.root = empty / "experience"
    assert "not installed" in Bench().attested_zulu("bus")["error"]


def test_the_agent_has_no_pronunciation_tools_and_its_prompt_leaves_pronunciation_to_the_harness():
    from mathula_tv import dub_agent

    names = {spec["function"]["name"] for spec in dub_agent.TOOL_SPECS}
    assert not names & {"list_foreign_words", "try_pronunciation", "suggest_spellings", "use_pronunciation", "note_pronunciation"}
    assert {"open_clip", "submit_dub", "measure_tts", "create_loanword", "attested_zulu"} <= names
    for word in ("try_pronunciation", "list_foreign_words", "use_pronunciation", "closeness", "english_hears"):
        assert word not in dub_agent.SYSTEM_PROMPT
    assert "by rules" in dub_agent.SYSTEM_PROMPT and "Never glue a Zulu prefix" in dub_agent.SYSTEM_PROMPT


def test_an_agent_that_finishes_is_not_reminded_about_pronunciation(tmp_path):
    client = ScriptedClient([
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),
        _reply(content="done"),
    ])
    summary = run_agent(client, StubWorkbench(tmp_path), clip_id="c1", run_id="r")
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 3
    assert not [m for m in client.seen[-1] if m["role"] == "user" and "reminder" in str(m.get("content")).lower()]


def test_a_run_that_has_met_the_bar_is_told_to_finish_and_ends_after_a_few_more_turns_not_eighty(tmp_path):
    from mathula_tv import dub_agent

    scores = [_score(1.9)] + [_score(1.8)] * 30                       # the bar (2.0%) is met by the first submission
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})])]
    for i in range(30):                                               # an agent that would go on submitting for ever
        script.append(_reply(calls=[_call(f"s{i}", "submit_dub", {"items": []})]))
    client = ScriptedClient(script)
    workbench = ScoringWorkbench(tmp_path, scores)
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", tts_budget=50, max_turns=80)
    assert summary["stop_reason"] == "bar_met" and summary["turns"] == 2 + dub_agent.WRAPUP_TURNS          # told at turn 2, six more turns, then it ends
    told = [m["content"] for m in client.seen[2] if m["role"] == "user" and "Finish now" in str(m.get("content"))]
    assert told and "meets the bar" in told[0] and "1.9%" in told[0]
    assert dub_agent.MET_STOP_REASONS >= {"bar_met", "agent_finished"}


def test_the_stated_bar_has_half_a_point_of_slack_and_the_wrapup_can_be_turned_off(tmp_path):
    from mathula_tv.dub_agent import unmet_targets

    near = _score(2.4)
    assert unmet_targets(near, max_dead_air_pct=2.0)                       # strictly 2.4% misses 2.0%
    assert not unmet_targets(near, max_dead_air_pct=2.0 + 0.5)              # but is good enough to stop on
    missed = run_agent(ScriptedClient(_agent_script(0)), ScoringWorkbench(tmp_path / "a", [_score(3.1)]), clip_id="c1", run_id="r", tts_budget=50, max_pushbacks=0)
    assert missed["stop_reason"] == "finished_below_target"                  # 3.1% is beyond the slack
    off = run_agent(ScriptedClient([_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})])] + [_reply(calls=[_call("s0", "submit_dub", {"items": []})])] * 8),
                    ScoringWorkbench(tmp_path / "b", [_score(1.0)] * 8), clip_id="c1", run_id="r", tts_budget=50, max_turns=6, wrapup_turns=0)
    assert off["stop_reason"] == "max_turns"


def test_a_run_that_has_met_the_bar_is_shown_the_listener_points_still_open_and_given_one_round_to_act_on_them(tmp_path):
    from mathula_tv import dub_agent

    chopped = "6 Zulu sentences for 2 English, 3.0 words each: said aloud it is mostly full stops"
    first = {**_score(1.9), "naturalness": [{"item_id": "unit-002", "cadence": chopped}]}
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s0", "submit_dub", {"items": []})]), _reply(content="done")]
    client = ScriptedClient(script)
    summary = run_agent(client, ScoringWorkbench(tmp_path, [first]), clip_id="c1", run_id="r", tts_budget=50, max_turns=20)
    told = [m["content"] for m in client.seen[-1] if m["role"] == "user" and "meets the bar" in str(m.get("content"))]
    assert told and "open points" in told[0] and "unit-002" in told[0] and "full stops" in told[0] and "partial=true" in told[0] and "Finish now" not in told[0]
    assert summary["stop_reason"] in ("agent_finished", "bar_met")
    assert dub_agent.open_listener_points({"naturalness": [{"item_id": f"u{i}", "cadence": "x"} for i in range(20)]}) \
        == [f"u{i}: x" for i in range(dub_agent.LISTENER_POINTS_SHOWN)]


def test_a_run_with_no_listener_points_open_is_told_to_finish_as_before(tmp_path):
    from mathula_tv import dub_agent

    assert dub_agent.open_listener_points(_score(1.9)) == [] and dub_agent.open_listener_points(None) == []
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s0", "submit_dub", {"items": []})]), _reply(content="done")]
    client = ScriptedClient(script)
    run_agent(client, ScoringWorkbench(tmp_path, [_score(1.9)]), clip_id="c1", run_id="r", tts_budget=50, max_turns=20)
    assert [m for m in client.seen[-1] if m["role"] == "user" and "Finish now" in str(m.get("content"))]


def test_a_unit_that_ends_seconds_before_its_window_is_a_wrapup_point_even_when_the_runs_dead_air_is_low():
    from mathula_tv import dub_agent

    result = {"headline": {"residual_dead_air_pct": 1.9}, "flagged_units": [
        {"item_id": "u14", "gates": [], "flags": ["dead_air"], "ends_early_ms": 5087},
        {"item_id": "u07", "gates": [], "flags": ["dead_air"], "ends_early_ms": 1500},          # under 2 s: not a point
        {"item_id": "u03", "gates": [], "flags": ["overrun"]}]}
    points = dub_agent.open_listener_points(result)
    assert len(points) == 1 and points[0].startswith("u14: the Zulu ends 5.1 s before") and "61 more characters" in points[0]


def test_the_submit_tool_takes_decisions_and_the_result_carries_the_working_state_through_compaction():
    from mathula_tv.dub_agent import SYSTEM_PROMPT, TOOL_SPECS, slim_submit

    spec = next(s for s in TOOL_SPECS if s["function"]["name"] == "submit_dub")
    assert "decisions" in spec["function"]["parameters"]["properties"] and "Decisions so far" in SYSTEM_PROMPT
    slim = slim_submit({"accepted": True, "headline": {}, "decisions_so_far": [{"english": "strategy", "zulu": "isu", "in_units": ["u16"]}],
                        "settled_units": ["u01", "u02"], "speaker_rates_chars_per_s": {"SPEAKER_00": 12.1}})
    assert slim["decisions_so_far"][0]["zulu"] == "isu" and slim["settled_units"] == ["u01", "u02"] and "cleared from the conversation" in slim["working_state_note"]


def test_a_long_clips_opened_result_is_not_cut_and_a_runaway_one_still_is():
    from mathula_tv import dub_agent

    assert dub_agent._clip("x" * 100_000) == "x" * 100_000          # an hour of dialogue and delivery sheet fits whole
    runaway = dub_agent._clip("x" * (dub_agent.MAX_TOOL_RESULT_CHARS + 500))
    assert runaway.endswith("[truncated 500 chars]")


def test_training_mode_gives_the_agent_choice_point_fields_the_training_prompt_and_asks_for_reasoning_summaries(tmp_path):
    from mathula_tv.dub_agent import TRAINING_PROMPT, agent_mode, build_system_prompt, tool_specs

    assert agent_mode(None) == "live" and agent_mode("Training") == "training" and agent_mode("nonsense") == "live"
    live_submit = next(s for s in tool_specs("live") if s["function"]["name"] == "submit_dub")["function"]["parameters"]["properties"]
    training_submit = next(s for s in tool_specs("training") if s["function"]["name"] == "submit_dub")["function"]["parameters"]["properties"]
    assert "choice_points" not in live_submit and {"choice_points", "unit_strategies"} <= set(training_submit)
    assert "choice_points" not in next(s for s in TOOL_SPECS if s["function"]["name"] == "submit_dub")["function"]["parameters"]["properties"]          # the shared specs are never edited
    assert TRAINING_PROMPT.strip() in build_system_prompt(None, mode="training") and TRAINING_PROMPT.strip() not in build_system_prompt(None)

    seen = {}

    class Client(ScriptedClient):
        reasoning_summary = None

        def complete(self, messages, **kw):
            seen["tools"] = [t["function"]["name"] for t in kw.get("tools") or []]
            seen["summary_asked"] = self.reasoning_summary
            return self.replies.pop(0)

    reply = _reply(content="done")
    reply["choices"][0]["message"]["_reasoning"] = "I compared two words for strategy and kept isu because the glossary has it."
    reply["choices"][0]["message"]["tool_calls"] = []
    workbench = StubWorkbench(tmp_path)
    client = Client([_reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("c", "submit_dub", {"items": []})]), reply])
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", mode="training")
    assert summary["agent_mode"] == "training" and seen["summary_asked"] == "detailed"
    trace = [json.loads(line) for line in (Path(workbench.store.repeat_dir("r", "c1", 1)) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row.get("reasoning", "").startswith("I compared two words") for row in trace if row.get("kind") == "model")
    assert (Path(workbench.store.repeat_dir("r", "c1", 1)) / "thinking_report.md").is_file()


def test_the_responses_client_asks_for_a_reasoning_summary_only_when_told_to_and_returns_it_for_the_trace(monkeypatch):
    from mathula_tv import dub_agent

    bodies = []
    monkeypatch.setattr(dub_agent, "post_with_retry", lambda url, headers, body, **kw: bodies.append(body) or {
        "output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "Weighed isu against icebo."}]}, {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 5, "output_tokens": 2}})
    client = dub_agent.ResponsesClient(SimpleNamespace(endpoint="https://x", deployment="d", api_key="k", reasoning_effort="low", timeout_seconds=30))
    client.complete([{"role": "user", "content": "hi"}])
    assert "summary" not in bodies[0]["reasoning"]
    client.reasoning_summary = "detailed"
    reply = client.complete([{"role": "user", "content": "hi"}])
    assert bodies[1]["reasoning"] == {"effort": "low", "summary": "detailed"}
    assert reply["choices"][0]["message"]["_reasoning"] == "Weighed isu against icebo."


def test_every_key_the_workbench_puts_in_a_submit_result_has_a_route_to_the_agent():
    """slim_submit is an allow-list: a key the workbench computes and slim_submit never copies is evidence nobody sees.  This reads the workbench source for the keys of last_submit and
    fails when one has no route (add it to SUBMIT_KEY_ROUTES and slim_submit), and checks that a result holding every key really delivers each route."""
    import ast

    from mathula_tv.dub_agent import SUBMIT_KEY_ROUTES, slim_submit

    source = (Path(__file__).resolve().parent.parent / "src" / "mathula_tv" / "dub_workbench.py").read_text(encoding="utf-8")
    keys: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "last_submit" for t in node.targets) and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant):
                    keys.add(key.value)
                elif key is None:                                                         # **({...} if cond else {}): the keys of that one dict, not of the dicts nested in it
                    target = value.body if isinstance(value, ast.IfExp) else value
                    if isinstance(target, ast.Dict):
                        keys.update(k.value for k in target.keys if isinstance(k, ast.Constant))
    assert len(keys) >= 18, keys
    assert not keys - set(SUBMIT_KEY_ROUTES), f"computed by the workbench but never shown to the agent: {sorted(keys - set(SUBMIT_KEY_ROUTES))}"
    full = {"accepted": True, "warnings": ["w"], "headline": {"group_count": 3, "gate_failures": 0}, "gates": {"a": {"failed": False, "n": 1}}, "merged": {"sent": 1}, "regressions_accepted": [1],
            "note": "n", "meaning": {"loss": [{"u": 1}], "claims": [{"u": 2}]}, "timing_lock": {"fitting_segments": 1}, "turn_padding": [{"turn_id": "t", "residual_ms": 500, "units": ["u1"]}],
            "flagged_units": [{"item_id": "u1", "gates": [], "flags": ["x"]}], "listening_load": [{"item_id": "u1"}], "text_checks": [{"item_id": "u1", "ok": True, "problems": [{"flag": "english_leakage"}]}], "spoken_form_gaps": [{"item_id": "u1"}],
            "voice_limit_words": [{"item_id": "u1"}], "figures_as_spoken": [{"item_id": "u1"}], "decisions_so_far": [{"english": "a"}], "settled_units": ["u2"], "units_overview": ["u1 1ch 0"],
            "naturalness": [{"item_id": "u1"}], "speaker_rates_chars_per_s": {"A": 12.0}, "sentence_timing": [{"unit": "u1", "behind_the_mouth": [{"sentence": 2}]}]}
    assert set(full) == set(SUBMIT_KEY_ROUTES)
    slim = slim_submit(full)
    for key, targets in SUBMIT_KEY_ROUTES.items():
        assert all(t in slim for t in targets), f"{key} did not reach the agent as {targets}"


def test_the_free_budget_policy_takes_the_cost_framing_out_and_leaves_the_lean_prompt_as_it_was():
    import re

    from mathula_tv.dub_agent import BUDGET_LIMITS, agent_budget, build_system_prompt

    flat = lambda text: re.sub(r"\s+", " ", text)                                         # noqa: E731
    lean, free = flat(build_system_prompt(None)), flat(build_system_prompt(None, budget="free"))
    for phrase in ("Measurements are budgeted.", "each extra turn costs more than the last", "much cheaper than rewriting", "you have budget left"):
        assert phrase in lean and phrase not in free, phrase
    assert "QUALITY FIRST" in free and "QUALITY FIRST" not in lean
    assert agent_budget(None) == "free" and agent_budget("LEAN") == "lean" and agent_budget("nonsense") == "free"
    assert BUDGET_LIMITS["free"]["max_turns"] > BUDGET_LIMITS["lean"]["max_turns"] and BUDGET_LIMITS["free"]["dead_air_slack_pct"] == 0.0
    assert BUDGET_LIMITS["lean"]["max_turns"] == 80 and BUDGET_LIMITS["lean"]["tts_budget"] == 600          # lean keeps today's limits


def test_a_quality_first_run_is_not_stopped_when_the_bar_is_met_and_is_told_once_with_its_open_points(tmp_path):
    from mathula_tv.dub_agent import free_bar_message

    message = free_bar_message(1.4, ["u5: 9 Zulu sentences for 4 English"], first=True)
    assert "meets the bar" in message and "u5: 9 Zulu sentences" in message and "no tool call" in message and "ONE round" in message          # the soft points get one round, then the end of the work
    assert "ONE round" in free_bar_message(1.4, ["u9: x"], first=False) and "New open points" in free_bar_message(1.4, ["u9: x"], first=False)


def test_a_quality_first_run_keeps_going_after_the_bar_is_met_hears_it_once_and_ends_only_when_the_agent_says_it_is_done(tmp_path):
    scores = [_score(1.9)] + [_score(1.8)] * 30
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})])]
    for i in range(12):                                                # twelve submissions after the bar is met: the lean rule would have stopped it after six
        script.append(_reply(calls=[_call(f"s{i}", "submit_dub", {"items": []})]))
    script.append(_reply(content="I judge no unit would be better now."))
    client = ScriptedClient(script)
    summary = run_agent(client, ScoringWorkbench(tmp_path, scores), clip_id="c1", run_id="r", tts_budget=50, max_turns=80, free_finish=True)
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 14 and summary["free_finish"] is True
    told = [m["content"] for m in client.seen[-1] if m["role"] == "user" and "meets the bar" in str(m.get("content"))]
    assert len(told) == 1 and "1.9%" in told[0]                  # told once, not at every submission
    assert not [m for m in client.seen[-1] if m["role"] == "user" and "Finish now" in str(m.get("content"))]


class _PausingClient(ScriptedClient):
    """A scripted client that asks for a pause (a PAUSE file in the run's folder) while answering its n-th call."""

    def __init__(self, replies, workbench, pause_on_call):
        super().__init__(replies)
        self.workbench, self.pause_on_call = workbench, pause_on_call

    def complete(self, messages, **kw):
        reply = super().complete(messages, **kw)
        if len(self.seen) == self.pause_on_call:
            from mathula_tv.dub_agent import request_pause

            request_pause(self.workbench.store.repeat_dir("r", "c1", 1))
        return reply


def test_a_paused_run_is_saved_whole_and_resumes_from_the_next_turn_without_redoing_anything(tmp_path):
    from mathula_tv import dub_agent

    folder = tmp_path / "r" / "c1" / "r1"
    first_bench = ScoringWorkbench(tmp_path, [_score(1.9), _score(1.8)])
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s1", "submit_dub", {"items": []})]), _reply(calls=[_call("s2", "submit_dub", {"items": []})])]
    client = _PausingClient(script, first_bench, pause_on_call=3)
    paused = run_agent(client, first_bench, clip_id="c1", run_id="r", tts_budget=50, free_finish=True)
    assert paused["stop_reason"] == "paused" and paused["turns"] == 3 and "--resume" in paused["resume"]
    state = dub_agent.load_state(folder)
    assert state["turn"] == 3 and state["stop_reason"] == "paused" and state["workbench"]["last_submit"] == first_bench.last_submit
    assert state["flags"]["told_bar"] is True                                                   # the bar was met at the first submission: that is remembered, not announced twice
    assert len(state["messages"]) == len(client.seen[-1]) + 2                                    # the system prompt, the task, every call and result, and the last reply with its tool result

    second_bench = ScoringWorkbench(tmp_path, [_score(1.7)])                                    # a new process: the workbench starts empty
    after = _PausingClient([_reply(calls=[_call("s3", "submit_dub", {"items": []})]), _reply(content="I judge no unit would be better.")], second_bench, pause_on_call=99)
    summary = run_agent(after, second_bench, clip_id="c1", run_id="r", tts_budget=50, free_finish=True, resume=True)
    assert summary["resumed_from_turn"] == 3 and summary["turns"] == 5 and summary["stop_reason"] == "agent_finished"
    assert len(after.seen[0]) == len(state["messages"])                                          # the first call after the resume sends exactly the saved conversation: no turn is redone
    assert summary["tokens"]["in"] > paused["tokens"]["in"] and summary["tool_calls"]["submit_dub"] == 3 and second_bench.last_submit == _score(1.7)
    told = [m for m in after.seen[-1] if m["role"] == "user" and "meets the bar" in str(m.get("content"))]
    assert len(told) == 1                                                                        # still told once in the whole conversation
    assert not (folder / "PAUSE").exists()


def test_resuming_with_no_checkpoint_says_so_and_a_crash_loses_at_most_the_call_in_flight(tmp_path):
    import pytest

    from mathula_tv import dub_agent

    with pytest.raises(FileNotFoundError):
        run_agent(ScriptedClient([]), ScoringWorkbench(tmp_path, []), clip_id="c1", run_id="r", resume=True)
    bench = ScoringWorkbench(tmp_path / "x", [_score(1.9)])
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s1", "submit_dub", {"items": []})])]

    class Crash(ScriptedClient):
        def complete(self, messages, **kw):
            if not self.replies:
                raise RuntimeError("the machine went away")
            return super().complete(messages, **kw)

    with pytest.raises(RuntimeError):
        run_agent(Crash(script), bench, clip_id="c1", run_id="r", tts_budget=50, free_finish=True)
    state = dub_agent.load_state(bench.store.repeat_dir("r", "c1", 1))
    assert state["turn"] == 2 and state["stop_reason"] == "running"                              # saved at the start of turn 3, before the call that died


def test_the_resume_uses_the_current_system_prompt_so_a_fix_can_be_picked_up_mid_run(tmp_path):
    bench = ScoringWorkbench(tmp_path, [_score(1.9)])
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s1", "submit_dub", {"items": []})])]
    client = _PausingClient(script, bench, pause_on_call=2)
    run_agent(client, bench, clip_id="c1", run_id="r", tts_budget=50, system_prompt="OLD RULES")
    after = ScriptedClient([_reply(content="done"), _reply(content="done"), _reply(content="done")])
    run_agent(after, ScoringWorkbench(tmp_path, []), clip_id="c1", run_id="r", tts_budget=50, system_prompt="NEW RULES", resume=True, free_finish=True, max_turns=3, max_pushbacks=0)
    assert after.seen[0][0]["content"] == "NEW RULES"


def _points_result(score, cadence_units):
    result = dict(score)
    result["naturalness"] = [{"item_id": u, "cadence": f"{u}: 6 Zulu sentences for 2 English, 3.8 words each"} for u in cadence_units]
    return result


def test_a_quality_first_agent_that_stops_with_open_points_is_sent_back_to_them_and_only_while_it_acts(tmp_path):
    from mathula_tv import dub_agent

    scores = [_points_result(_score(1.9), ["u5"]), _points_result(_score(1.8), ["u5"]), _points_result(_score(1.7), [])]
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s1", "submit_dub", {"items": []})]),
              _reply(content="done"),                                                         # stops with u5 open: sent back (round 1)
              _reply(calls=[_call("s2", "submit_dub", {"items": []})]), _reply(content="done again"),     # acted, u5 still open: sent back (round 2)
              _reply(calls=[_call("s3", "submit_dub", {"items": []})]), _reply(content="now done")]      # acted, nothing open: finishes
    client = ScriptedClient(script)
    summary = run_agent(client, ScoringWorkbench(tmp_path, scores), clip_id="c1", run_id="r", tts_budget=50, free_finish=True, max_turns=40)
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 5                  # one round (it was three: prod-y1 spent 21 turns after the bar on soft points)
    sent = [m["content"] for m in client.seen[-1] if m["role"] == "user" and "You stopped, but your last submission still has" in str(m.get("content"))]
    assert len(sent) == 1 and "Round 1 of 1" in sent[0] and "u5:" in sent[0] and chr(10) + "- u5:" in sent[0]
    trace = [json.loads(line) for line in (Path(tmp_path) / "r" / "c1" / "r1" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["round"] for r in trace if r.get("kind") == "quality_pushback"] == [1]
    assert [r["agent_said"] for r in trace if r.get("kind") == "quality_pushback"] == ["done"]          # what the agent said when it stopped is kept


def test_an_agent_that_answers_the_push_back_without_changing_anything_is_not_asked_again_and_the_rounds_are_bounded(tmp_path):
    scores = [_points_result(_score(1.9), ["u5"])]
    client = ScriptedClient([_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s1", "submit_dub", {"items": []})]),
                             _reply(content="u5 stays: measured, adding the silence overruns by 400 ms"), _reply(content="u5 stays, as said")])
    summary = run_agent(client, ScoringWorkbench(tmp_path, scores), clip_id="c1", run_id="r", tts_budget=50, free_finish=True, max_turns=40)
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 4                  # one round: it declined, and that is its answer
    from mathula_tv.dub_agent import MAX_QUALITY_ROUNDS

    assert MAX_QUALITY_ROUNDS == 1
    lean = ScriptedClient([_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s1", "submit_dub", {"items": []})]), _reply(content="done")])
    done = run_agent(lean, ScoringWorkbench(tmp_path / "lean", [_points_result(_score(1.9), ["u5"])]), clip_id="c1", run_id="r", tts_budget=50, max_turns=40)
    assert done["stop_reason"] in ("agent_finished", "bar_met") and not [m for m in lean.seen[-1] if "You stopped, but your last submission" in str(m.get("content"))]          # lean runs are unchanged


# ---- the draft deadline -----------------------------------------------------------------------------------------------------

def _drafting_script(measures):
    script = [_heavy(_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]))]
    script += [_heavy(_reply(calls=[_call(f"m{i}", "measure_tts", {"text": "the only draft", "start_ms": 0, "end_ms": 1000})])) for i in range(measures)]
    return script


def test_an_agent_that_keeps_researching_is_told_at_the_deadline_refused_every_other_tool_and_then_its_drafts_are_submitted_for_it(tmp_path):
    from mathula_tv.budget_governor import DRAFT_GRACE_TURNS
    from mathula_tv.dub_agent import DRAFT_DEADLINE_REFUSAL

    script = _drafting_script(11) + [_heavy(_reply(content="done")), _heavy(_reply(content="still done"))]
    client = ScriptedClient(script)
    workbench = StubWorkbench(tmp_path)
    _segments(workbench, [(0, 1000)])
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", max_total_tokens=1_000_000, draft_deadline=True)
    trace = (tmp_path / "r" / "c1" / "r1" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count('"kind": "draft_deadline"') == 1 and trace.count('"kind": "draft_auto_submit"') == 1                  # told once, acted once
    seen = [m for turn in client.seen for m in turn if m["role"] == "user" and str(m.get("content")).startswith("DRAFT DEADLINE")]
    assert seen and "the only draft" in seen[0]["content"]                                                                     # the note carries the agent's own drafts
    refused = [m for m in client.seen[-1] if m["role"] == "tool" and DRAFT_DEADLINE_REFUSAL in m["content"]]
    assert len(refused) == DRAFT_GRACE_TURNS                                                                                    # every call in the grace turns was refused for free
    assert workbench.episode["submissions"] == 1 and workbench.submitted_items[0]["zulu_text"] == "the only draft"
    assert summary["budget"]["draft_deadline"] == {"turns_after_note": DRAFT_GRACE_TURNS, "harness_submitted": True} and summary["draft_deadline"] is True
    assert any("did not submit in time" in str(m.get("content")) for m in client.seen[-1] if m["role"] == "user")             # and it is told what happened and that the tools are back


def test_a_submission_inside_the_grace_turns_lifts_the_restriction_and_the_harness_does_nothing(tmp_path):
    script = _drafting_script(9) + [_heavy(_reply(calls=[_call("s", "submit_dub", {"items": []})])), _heavy(_reply(calls=[_call("m", "measure_tts", {"text": "after", "start_ms": 0, "end_ms": 1000})])),
                                    _heavy(_reply(content="done")), _heavy(_reply(content="done"))]
    client = ScriptedClient(script)
    workbench = StubWorkbench(tmp_path)
    _segments(workbench, [(0, 1000)])
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", max_total_tokens=1_000_000, draft_deadline=True)
    trace = (tmp_path / "r" / "c1" / "r1" / "trace.jsonl").read_text(encoding="utf-8")
    assert '"kind": "draft_deadline"' in trace and '"kind": "draft_auto_submit"' not in trace
    assert workbench.episode["submissions"] == 1 and workbench.submitted_items == []                                         # the agent's own submit_dub, not the harness's
    after = [m for m in client.seen[-1] if m["role"] == "tool" and m["tool_call_id"] == "m"]
    assert after and "not_available_yet" not in after[0]["content"]                                                         # the measurement after the submission was allowed
    assert summary["budget"]["draft_deadline"]["harness_submitted"] is False


def test_the_deadline_is_off_when_switched_off_and_does_not_apply_once_a_submission_exists(tmp_path):
    script = _drafting_script(11) + [_heavy(_reply(content="done")), _heavy(_reply(content="done"))]
    client = ScriptedClient(script)
    workbench = StubWorkbench(tmp_path)
    run_agent(client, workbench, clip_id="c1", run_id="r", max_total_tokens=1_000_000, draft_deadline=False)
    assert not any(str(m.get("content")).startswith("DRAFT DEADLINE") for turn in client.seen for m in turn if m["role"] == "user")
    seeded = ScriptedClient(_drafting_script(11)[1:] + [_heavy(_reply(content="done")) for _ in range(5)])          # no open_clip: a repair run starts with its seed already scored
    accepted_bench = StubWorkbench(tmp_path / "b")
    accepted_bench.open_clip("c1", "r")
    accepted_bench.episode["submissions"] = 1                                                  # a repair run starts accepted
    run_agent(seeded, accepted_bench, clip_id="c1", run_id="r", max_total_tokens=1_000_000, draft_deadline=True)
    assert not any(str(m.get("content")).startswith("DRAFT DEADLINE") for turn in seeded.seen for m in turn if m["role"] == "user")


def test_drafts_that_do_not_cover_the_clip_are_not_submitted_and_the_agent_is_told_which_units_have_no_text(tmp_path):
    script = _drafting_script(11) + [_heavy(_reply(content="done")), _heavy(_reply(content="done"))]
    client = ScriptedClient(script)
    workbench = StubWorkbench(tmp_path)
    _segments(workbench, [(0, 1000), (1000, 2000)])                                            # the agent only ever drafted the first window
    summary = run_agent(client, workbench, clip_id="c1", run_id="r", max_total_tokens=1_000_000, draft_deadline=True)
    assert workbench.episode["submissions"] == 0
    told = [m["content"] for m in client.seen[-1] if m["role"] == "user" and "could not submit your working drafts" in str(m.get("content"))]
    assert told and "s2" in told[0]
    assert summary["budget"]["draft_deadline"]["harness_submitted"] is True                  # it tried once, and does not try again every turn


def _focus_history(steps_after_submit: int = 8):
    messages = [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
        _assistant("o", "open_clip"), _tool("o", "THE CLIP"),
        _assistant("l", "recall_lessons"), _tool("l", "LESSONS"),
        _assistant("s1", "submit_dub"), _tool("s1", "SCORE 1"),
    ]
    for n in range(steps_after_submit):
        messages += [_assistant(f"w{n}", "measure_tts_batch"), _tool(f"w{n}", f"measurement {n}")]
    return messages


def _ledger_with(text="the latest text of unit 1"):
    from mathula_tv.draft_ledger import DraftLedger

    ledger = DraftLedger()
    ledger.set_segments([{"segment_id": "s1", "start_ms": 0, "end_ms": 10000}])
    ledger.observe("measure_tts", json.dumps({"text": text, "start_ms": 0, "end_ms": 10000}), json.dumps({"measured_ms": 9000, "window_ms": 10000, "fits_naturally": False}))
    return ledger


def test_focus_folds_the_working_steps_since_the_last_submission_into_the_ledger_and_keeps_the_last_few():
    from mathula_tv.dub_agent import FOCUS_KEEP_STEPS, FOCUS_NOTE_PREFIX, compact_history

    messages = _focus_history(8)
    plain, dropped_plain = compact_history(messages, _ledger_with())
    assert dropped_plain == 0 and plain == messages                                           # without the focus step nothing after the last submission is touched, as before
    folded, dropped = compact_history(messages, _ledger_with(), focus=True)
    texts = [m.get("content") for m in folded]
    assert dropped == 8 - FOCUS_KEEP_STEPS
    assert folded[:8] == messages[:8]                                                         # the head, the clip and the submission with its result: byte for byte the same, so their cache stays valid
    assert not any(f"measurement {n}" in texts for n in range(8 - FOCUS_KEEP_STEPS))
    assert all(f"measurement {n}" in texts for n in range(8 - FOCUS_KEEP_STEPS, 8))          # the last steps verbatim
    note = next(t for t in texts if str(t).startswith(FOCUS_NOTE_PREFIX))
    assert "the latest text of unit 1" in note and "measured 9.0s" in note
    called = {c["id"] for m in folded if m["role"] == "assistant" for c in m.get("tool_calls") or []}
    assert called == {m["tool_call_id"] for m in folded if m["role"] == "tool"}               # every call still has its result


def test_focus_is_idempotent_and_leaves_a_short_tail_and_an_empty_ledger_alone():
    from mathula_tv.dub_agent import compact_history
    from mathula_tv.draft_ledger import DraftLedger

    once, _ = compact_history(_focus_history(8), _ledger_with(), focus=True)
    twice, dropped = compact_history(once, _ledger_with(), focus=True)
    assert dropped == 0 and twice == once
    short = _focus_history(3)
    assert compact_history(short, _ledger_with(), focus=True) == (short, 0)                  # three steps are all recent
    assert compact_history(_focus_history(8), DraftLedger(), focus=True)[1] == 0              # nothing in the ledger to stand in for them: nothing is dropped


def test_the_focus_trigger_waits_for_growth_and_a_real_size_and_is_off_for_the_lean_budget():
    from mathula_tv.dub_agent import BUDGET_LIMITS, FOCUS_FLOOR_TOKENS

    assert BUDGET_LIMITS["free"]["focus_growth_tokens"] >= 5_000 and BUDGET_LIMITS["lean"]["focus_growth_tokens"] == 0
    assert FOCUS_FLOOR_TOKENS > 30_000                                                       # the opened clip, the system prompt and the lessons alone are about 26k



def test_a_repair_run_is_told_at_the_start_which_units_write_a_prefix_real_zulu_text_does_not():
    from types import SimpleNamespace

    from mathula_tv.attestation import Attestation, WebText
    from mathula_tv.dub_agent import prefix_corpus_note

    attestation = Attestation(["x"], ["y"], [], WebText({"ama-hawks": (6, 8), "amahawks": (5, 2), "hawks": (18, 41)}))
    workbench = SimpleNamespace(_resources=lambda: attestation, segments=[{"segment_id": "s1", "source_text": "He worked under the Hawks."}, {"segment_id": "s2", "source_text": "The police said so."}])
    items = [{"item_id": "unit-01", "source_segment_ids": ["s2"], "zulu_text": "Amaphoyisa athe kunjalo."},
             {"item_id": "unit-02", "source_segment_ids": ["s1"], "zulu_text": "Wayengaphansi kwabe-Hawks, owanikwa umsebenzi."}]
    note = prefix_corpus_note(workbench, items)
    assert "unit-02: Hawks: you wrote kwabe-Hawks" in note and "ama-Hawks" in note and "unit-01" not in note
    assert "can still be right" in note                                   # said once, as a reminder
    assert prefix_corpus_note(SimpleNamespace(_resources=lambda: None, segments=[]), items) == ""            # no corpus installed: nothing to say
    assert prefix_corpus_note(SimpleNamespace(_resources=lambda: attestation, segments=[]), [{"item_id": "u", "zulu_text": "Sawubona.", "source_segment_ids": []}]) == ""


def test_the_reviewer_reports_phrases_per_unit_and_keeps_its_answer_until_the_text_changes():
    from mathula_tv.dub_judge import UnitJudge

    class Client:
        calls = 0

        def complete(self, messages, **_):
            Client.calls += 1
            import json

            ids = [x["id"] for x in json.loads(messages[1]["content"])]
            body = {"items": [{"id": i, "issues": [{"phrase": "u-Ayanda Nyathi nentatheli", "kind": "fused", "problem": "runs together", "fix": "u-Ayanda Nyathi, nentatheli"},
                                                    {"phrase": "", "kind": "grammar", "problem": "empty", "fix": ""}] if i == "unit-01" else []} for i in ids]}
            return {"choices": [{"message": {"content": json.dumps(body)}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    judge = UnitJudge(Client())
    units = [{"id": "unit-01", "english": "Ayanda Nyathi and journalist", "zulu": "u-Ayanda Nyathi nentatheli"}, {"id": "unit-02", "english": "Hello", "zulu": "Sawubona."}]
    first = judge.grammar(units)
    assert first["unit-01"] == [{"phrase": "u-Ayanda Nyathi nentatheli", "kind": "fused", "problem": "runs together", "fix": "u-Ayanda Nyathi, nentatheli"}] and first["unit-02"] == []          # the empty issue is dropped
    calls = Client.calls
    assert judge.grammar(units) == first and Client.calls == calls                                    # nothing is asked again for an unchanged text
    changed = judge.grammar([{**units[0], "zulu": "u-Ayanda Nyathi, nentatheli"}, units[1]])
    assert Client.calls == calls + 1 and "unit-01" in changed                                          # a changed text is asked again, the unchanged one is not


def test_check_texts_checks_many_units_in_one_call_and_every_submit_carries_the_checks_for_all_its_units():
    import pytest

    from mathula_tv.dub_workbench import Workbench

    wb = Workbench.__new__(Workbench)
    seen = []

    def fake_check(zulu, source):
        seen.append((zulu, source))
        problems = []
        if "gate" in zulu:
            problems.append({"gate": "fabricated_entity", "evidence": ["x"]})
        if "soft" in zulu:
            problems.append({"flag": "english_leakage"})
        if "long" in zulu:
            problems.append({"flag": "long_run_without_breath"})
        return {"ok": not any("gate" in p for p in problems), "problems": problems, "gates_checked": []}

    wb.check_text = fake_check
    many = wb.check_texts([{"item_id": "a", "zulu_text": "fine", "source_text": "A"}, {"item_id": "b", "zulu_text": "a gate", "source_text": "B"}])
    assert [r["item_id"] for r in many["results"]] == ["a", "b"] and many["results"][1]["ok"] is False and len(seen) == 2
    with pytest.raises(ValueError):
        wb.check_texts([])
    with pytest.raises(ValueError):
        wb.check_texts([{"zulu_text": "x", "source_text": "y"}] * 41)
    items = [{"item_id": "a", "zulu_text": "fine", "source_segment_ids": ["s1"]}, {"item_id": "b", "zulu_text": "a gate", "source_segment_ids": ["s1", "s2"]},
             {"item_id": "c", "zulu_text": "a long run", "source_segment_ids": ["s2"]}, {"item_id": "d", "zulu_text": "soft", "source_segment_ids": []}]
    segments = [{"segment_id": "s1", "source_text": "one", "restored_text": "One."}, {"segment_id": "s2", "source_text": "two"}]
    seen.clear()
    checks = wb._text_checks(items, segments)
    assert [c["item_id"] for c in checks] == ["b", "d"] and checks[0]["ok"] is False and checks[1]["ok"] is True          # a unit with nothing to say, and one whose only flag listening_load already reports, are left out
    assert seen[1][1] == "One. two"                                                                                      # each unit is checked against the English of its own segments, as restored


def test_the_draft_note_tells_the_agent_to_send_every_unit_in_one_call_and_check_texts_is_a_tool():
    from mathula_tv.dub_agent import DRAFT_PHASE_NOTE, TOOL_SPECS

    assert "ALL the units of the draft in ONE call" in DRAFT_PHASE_NOTE
    spec = next(s for s in TOOL_SPECS if s["function"]["name"] == "check_texts")
    assert "ONE call" in spec["function"]["description"] and spec["function"]["parameters"]["required"] == ["units"]



def test_a_repair_run_can_be_scoped_to_named_units_and_the_note_says_to_leave_every_other_unit():
    from mathula_tv.dub_agent import FOCUS_NOTE

    note = FOCUS_NOTE.format(units="u-05, u-13")
    assert "u-05, u-13" in note and "ONLY" in note and "Do NOT change any other unit" in note and "its own pace" in note
