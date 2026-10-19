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
    assert shaped["usage"] == {"prompt_tokens": 50, "completion_tokens": 7}

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
