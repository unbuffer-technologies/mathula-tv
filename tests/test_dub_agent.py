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


def test_an_agent_that_never_looked_at_pronunciation_is_reminded_once_before_it_may_finish(tmp_path):
    client = ScriptedClient([
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),
        _reply(content="done"),                                                        # reminded here, not finished
        _reply(calls=[_call("c", "list_foreign_words", {})]),
        _reply(content="done"),
    ])
    summary = run_agent(client, PendingWorkbench(tmp_path), clip_id="c1", run_id="r")
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 5
    assert summary["tool_calls"].get("list_foreign_words") == 1
    reminder = [str(m.get("content")) for m in client.seen[3] if m["role"] == "user" and "One reminder" in str(m.get("content"))]
    assert reminder and "science, correctional" in reminder[0] and "you do not have to use the lab" in reminder[0]


def test_the_pronunciation_reminder_is_bounded_to_one(tmp_path):
    client = ScriptedClient([
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),
        _reply(content="done"),
        _reply(content="still done"),                                                   # ignores the reminder: finishes anyway
    ])
    summary = run_agent(client, PendingWorkbench(tmp_path), clip_id="c1", run_id="r")
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 4
    assert "list_foreign_words" not in summary["tool_calls"]


def test_no_words_to_check_means_no_reminder(tmp_path):
    class NothingToCheck(PendingWorkbench):
        def pronunciation_check_pending(self):
            return []

    client = ScriptedClient([
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),
        _reply(content="done"),
    ])
    summary = run_agent(client, NothingToCheck(tmp_path), clip_id="c1", run_id="r")
    assert summary["stop_reason"] == "agent_finished" and summary["turns"] == 3


def test_a_run_that_is_running_out_of_turns_is_reminded_early_with_the_turns_left(tmp_path, monkeypatch):
    from mathula_tv import dub_agent

    monkeypatch.setattr(dub_agent, "PRONUNCIATION_RESERVE_TURNS", 3)
    client = ScriptedClient([
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),                    # turn 1
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),                       # turn 2: accepted
        _reply(calls=[_call("c", "get_submission", {})]),                              # turn 3
        _reply(calls=[_call("d", "get_submission", {})]),                              # turn 4: 6 - 3 = 3 <= 4 -> reminded before this reply? no: before turn 3
        _reply(calls=[_call("e", "list_foreign_words", {})]),
        _reply(content="done"),
    ])
    summary = run_agent(client, PendingWorkbench(tmp_path), clip_id="c1", run_id="r", max_turns=6)
    reminders = [(i, str(m.get("content"))) for i, seen in enumerate(client.seen) for m in seen if m["role"] == "user" and "One reminder" in str(m.get("content"))]
    assert reminders and "turns left" in reminders[0][1] and "science, correctional" in reminders[0][1]
    assert reminders[0][0] <= 3                                                         # sent while there was still room to act on it
    assert summary["stop_reason"] == "agent_finished"
    assert sum(1 for seen in client.seen[-1:] for m in seen if "One reminder" in str(m.get("content"))) == 1      # once, not repeated at the end


def test_the_early_reminder_is_not_sent_when_there_is_nothing_to_check(tmp_path, monkeypatch):
    from mathula_tv import dub_agent

    class Nothing(PendingWorkbench):
        def pronunciation_check_pending(self):
            return []

    monkeypatch.setattr(dub_agent, "PRONUNCIATION_RESERVE_TURNS", 3)
    client = ScriptedClient([
        _reply(calls=[_call("a", "open_clip", {"clip_id": "c1"})]),
        _reply(calls=[_call("b", "submit_dub", {"items": []})]),
        _reply(calls=[_call("c", "get_submission", {})]),
        _reply(calls=[_call("d", "get_submission", {})]),
        _reply(content="done"),
    ])
    summary = run_agent(client, Nothing(tmp_path), clip_id="c1", run_id="r", max_turns=6)
    assert summary["stop_reason"] == "agent_finished"
    assert not any("One reminder" in str(m.get("content")) for seen in client.seen for m in seen)


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


def test_the_pronunciation_reminder_carries_the_labs_measurement_of_each_word_as_written():
    from mathula_tv.dub_agent import pronunciation_evidence_note

    class Bench:
        def pronunciation_evidence(self, words):
            return [{"word": "MaFighter", "speaker_sounds": "f aɪ t a s", "voice_sounds": "f eɪ t ə", "closeness": 0.7,
                     "differences": ["speaker aɪ, voice eɪ (vowel)"], "english_hears": "Theta.", "lost_in_its_sentence": ["s"]}]

    note = pronunciation_evidence_note(Bench(), ["MaFighter"])
    assert "MaFighter: the speaker says [f aɪ t a s]; the voice says [f eɪ t ə] as written (closeness 0.7)" in note
    assert "speaker aɪ, voice eɪ (vowel)" in note and "an English listener writes 'Theta.'" in note and "the voice loses s" in note and "no tries used" in note
    assert pronunciation_evidence_note(object(), ["x"]) == ""

    class Broken:
        def pronunciation_evidence(self, words):
            raise RuntimeError("synthesis down")

    assert pronunciation_evidence_note(Broken(), ["x"]) == ""                         # a measuring failure never blocks the reminder
