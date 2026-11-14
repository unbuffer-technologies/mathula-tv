"""The agent loop's mechanics, against a scripted model and a stub environment (no network, no TTS)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mathula_tv.dub_agent import TOOL_SPECS, build_dispatch, execute_tool, run_agent
from mathula_tv.dub_workbench import BudgetExhausted


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

    def preview_turn_padding(self, members, last_free_gap_ms=0):
        return {"mechanism": "none"}

    def get_submission(self, unit_ids=None):
        return {"units": [], "count": 0}

    def check_text(self, zulu_text, source_text):
        return {"ok": True}

    def recall_lessons(self, limit=25):
        return {"count": 0}

    def submit_dub(self, items, approach_notes=""):
        self.episode["submissions"] += 1
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

    def complete(self, messages, **_):
        self.seen.append([dict(m) for m in messages])
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

    held = "after “as well as” the speaker held 1.2 s; near “kakhulu, kanye noFannie” your Zulu leaves 0.15 s (~pause1200 after the word that ends the thought would match)"
    first = {**_score(1.9), "comma_check": [{"item_id": "unit-002", "commas": 12, "pauses_the_speaker_held_that_the_dub_does_not": [held],
                                             "commas_where_the_speaker_did_not_pause": ["“Impela, Ayanda” (the speaker said Ayanda with no pause before it: 0.00 s)"]}]}
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s0", "submit_dub", {"items": []})]), _reply(content="done")]
    client = ScriptedClient(script)
    summary = run_agent(client, ScoringWorkbench(tmp_path, [first]), clip_id="c1", run_id="r", tts_budget=50, max_turns=20)
    told = [m["content"] for m in client.seen[-1] if m["role"] == "user" and "meets the bar" in str(m.get("content"))]
    assert told and "open points" in told[0] and "unit-002" in told[0] and "~pause1200" in told[0] and "partial=true" in told[0] and "Finish now" not in told[0]
    assert summary["stop_reason"] in ("agent_finished", "bar_met")
    assert dub_agent.open_listener_points({"comma_check": [{"item_id": f"u{i}", "pauses_the_speaker_held_that_the_dub_does_not": ["x"]} for i in range(20)]}) \
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
            "flagged_units": [{"item_id": "u1", "gates": [], "flags": ["x"]}], "listening_load": [{"item_id": "u1"}], "spoken_form_gaps": [{"item_id": "u1"}], "comma_check": [{"item_id": "u1"}],
            "voice_limit_words": [{"item_id": "u1"}], "figures_as_spoken": [{"item_id": "u1"}], "decisions_so_far": [{"english": "a"}], "settled_units": ["u2"], "units_overview": ["u1 1ch 0"],
            "naturalness": [{"item_id": "u1"}], "speaker_rates_chars_per_s": {"A": 12.0}}
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
    assert "QUALITY FIRST" in free and "QUALITY FIRST" not in lean and "not the finish line" in free
    assert agent_budget(None) == "free" and agent_budget("LEAN") == "lean" and agent_budget("nonsense") == "free"
    assert BUDGET_LIMITS["free"]["max_turns"] > BUDGET_LIMITS["lean"]["max_turns"] and BUDGET_LIMITS["free"]["dead_air_slack_pct"] == 0.0
    assert BUDGET_LIMITS["lean"]["max_turns"] == 80 and BUDGET_LIMITS["lean"]["tts_budget"] == 600          # lean keeps today's limits


def test_a_quality_first_run_is_not_stopped_when_the_bar_is_met_and_is_told_once_with_its_open_points(tmp_path):
    from mathula_tv.dub_agent import free_bar_message

    message = free_bar_message(1.4, ["u5: 9 Zulu sentences for 4 English"], first=True)
    assert "minimum, not the finish line" in message and "u5: 9 Zulu sentences" in message and "no tool call" in message
    assert "minimum, not the finish line" not in free_bar_message(1.4, ["u9: x"], first=False) and "New open points" in free_bar_message(1.4, ["u9: x"], first=False)


def test_a_quality_first_run_keeps_going_after_the_bar_is_met_hears_it_once_and_ends_only_when_the_agent_says_it_is_done(tmp_path):
    scores = [_score(1.9)] + [_score(1.8)] * 30
    script = [_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})])]
    for i in range(12):                                                # twelve submissions after the bar is met: the lean rule would have stopped it after six
        script.append(_reply(calls=[_call(f"s{i}", "submit_dub", {"items": []})]))
    script.append(_reply(content="I judge no unit would be better now."))
    client = ScriptedClient(script)
    summary = run_agent(client, ScoringWorkbench(tmp_path, scores), clip_id="c1", run_id="r", tts_budget=50, max_turns=80, free_finish=True)
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 14 and summary["free_finish"] is True
    told = [m["content"] for m in client.seen[-1] if m["role"] == "user" and "not the finish line" in str(m.get("content"))]
    assert len(told) == 1 and "meets the minimum bar" in told[0] and "1.9%" in told[0]                  # told once, not at every submission
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
    told = [m for m in after.seen[-1] if m["role"] == "user" and "not the finish line" in str(m.get("content"))]
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
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 7
    sent = [m["content"] for m in client.seen[-1] if m["role"] == "user" and "You stopped, but your last submission still has" in str(m.get("content"))]
    assert len(sent) == 2 and "Round 1 of 3" in sent[0] and "Round 2 of 3" in sent[1] and "u5:" in sent[0] and chr(10) + "- u5:" in sent[0]
    trace = [json.loads(line) for line in (Path(tmp_path) / "r" / "c1" / "r1" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["round"] for r in trace if r.get("kind") == "quality_pushback"] == [1, 2]
    assert [r["agent_said"] for r in trace if r.get("kind") == "quality_pushback"] == ["done", "done again"]          # what the agent said when it stopped is kept


def test_an_agent_that_answers_the_push_back_without_changing_anything_is_not_asked_again_and_the_rounds_are_bounded(tmp_path):
    scores = [_points_result(_score(1.9), ["u5"])]
    client = ScriptedClient([_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s1", "submit_dub", {"items": []})]),
                             _reply(content="u5 stays: measured, adding the silence overruns by 400 ms"), _reply(content="u5 stays, as said")])
    summary = run_agent(client, ScoringWorkbench(tmp_path, scores), clip_id="c1", run_id="r", tts_budget=50, free_finish=True, max_turns=40)
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 4                  # one round: it declined, and that is its answer
    from mathula_tv.dub_agent import MAX_QUALITY_ROUNDS

    assert MAX_QUALITY_ROUNDS == 3
    lean = ScriptedClient([_reply(calls=[_call("o", "open_clip", {"clip_id": "c1"})]), _reply(calls=[_call("s1", "submit_dub", {"items": []})]), _reply(content="done")])
    done = run_agent(lean, ScoringWorkbench(tmp_path / "lean", [_points_result(_score(1.9), ["u5"])]), clip_id="c1", run_id="r", tts_budget=50, max_turns=40)
    assert done["stop_reason"] in ("agent_finished", "bar_met") and not [m for m in lean.seen[-1] if "You stopped, but your last submission" in str(m.get("content"))]          # lean runs are unchanged
