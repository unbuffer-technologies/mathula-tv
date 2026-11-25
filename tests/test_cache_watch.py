from __future__ import annotations

from types import SimpleNamespace

from mathula_tv.cache_watch import CacheWatch, fingerprint, request_fingerprint
from mathula_tv.dub_agent import _frozen_head_end, compact_history
from mathula_tv.draft_ledger import DraftLedger


def fp(contents, tools=("t1",), system="sys"):
    return fingerprint([{"role": "user", "parts": [c]} for c in contents], list(tools), system)


def use(prompt, cached, thoughts=0):
    return {"prompt_tokens": prompt, "cached_tokens": cached, "thoughts_tokens": thoughts}


BIG = "x" * 4000


def test_the_first_request_is_only_recorded_and_an_append_only_hit_has_no_cause():
    watch = CacheWatch()
    assert watch.observe(1, fp([BIG]), use(10_000, 0))["cause"] == "first_request"
    row = watch.observe(2, fp([BIG, "reply", "next"]), use(12_000, 9_000))
    assert row["cause"] is None and row["kept_pct"] > 90 and row["hit_pct"] == 75.0 and row["lost_tokens_est"] == 0


def test_a_changed_tool_set_or_system_prompt_is_named_and_the_loss_is_estimated():
    watch = CacheWatch()
    watch.observe(1, fp([BIG]), use(10_000, 0))
    row = watch.observe(2, fp([BIG, "more"], tools=("t1", "t2")), use(20_000, 0))
    assert row["cause"] == "tools_changed" and row["expected_hit_pct"] == 0.0 and row["lost_tokens_est"] == 0     # nothing could have been reused: nothing was lost to a miss
    row = watch.observe(3, fp([BIG, "more", "again"], tools=("t1", "t2"), system="new"), use(22_000, 0))
    assert row["cause"] == "system_changed"


def test_a_rewritten_history_says_where_and_how_much_of_the_front_survived():
    watch = CacheWatch()
    watch.observe(1, fp([BIG, "a", "b", "c"]), use(10_000, 0))
    row = watch.observe(2, fp([BIG, "a", "SUMMARY", "d"]), use(10_000, 8_000))
    assert row["cause"] == "history_rewritten" and row["first_changed_message"] == 2 and row["kept_messages"] == 2 and row["kept_pct"] > 90
    assert row["lost_tokens_est"] > 0 or row["hit_pct"] >= 80          # the front was kept, so a hit that large is what a kept front should give


def test_an_unchanged_front_the_provider_did_not_serve_is_a_provider_miss_but_only_above_the_minimum():
    watch = CacheWatch()
    watch.observe(1, fp([BIG]), use(10_000, 0))
    row = watch.observe(2, fp([BIG, "r", "n"]), use(12_000, 0))
    assert row["cause"] == "provider_miss" and row["lost_tokens_est"] > 9_000
    small = CacheWatch()
    small.observe(1, fp(["a"]), use(1_000, 0))
    assert small.observe(2, fp(["a", "b"]), use(1_500, 0))["cause"] is None          # under any provider's minimum: not a miss


def test_the_report_adds_the_turns_up_and_state_survives_a_checkpoint():
    watch = CacheWatch()
    watch.observe(1, fp([BIG]), use(10_000, 0, 5))
    watch.observe(2, fp([BIG, "r"]), use(12_000, 9_000, 7))
    watch.observe(3, fp([BIG, "r"], tools=("t9",)), use(12_000, 0, 1))
    report = watch.report()
    assert report["turns"] == 3 and report["prompt_tokens"] == 34_000 and report["cached_tokens"] == 9_000 and report["thought_tokens"] == 13
    assert report["misses_by_cause"] == {"first_request": 1, "tools_changed": 1} and len(report["header_hashes"]) == 2
    again = CacheWatch()
    again.load(watch.to_dict())
    assert again.report() == report
    assert again.observe(4, fp([BIG, "r"], tools=("t9",)), use(12_000, 11_000))["cause"] is None      # it carries on from where the checkpoint left off


def test_the_request_the_client_really_sent_is_what_is_fingerprinted():
    client = SimpleNamespace(last_request={"tools": [{"googleSearch": {}}], "system": {"parts": [{"text": "s"}]}, "contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    other = SimpleNamespace(last_request={"tools": [], "system": {"parts": [{"text": "s"}]}, "contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    assert request_fingerprint(client, [], None)["tools"] != request_fingerprint(other, [], None)["tools"]          # a tool the client adds on its own shows
    plain = request_fingerprint(object(), [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}], [{"function": {"name": "a"}}])
    assert len(plain["messages"]) == 1                                                                              # the system prompt is the header, not a message


# ---- compaction leaves the head alone ---------------------------------------------------------------------------------------

def call(name, call_id):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}]}


def result(call_id, text="r"):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def conversation():
    messages = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "TASK"}, {"role": "user", "content": "NOTE BEFORE ANY WORK"},
                call("open_clip", "o"), result("o", "THE WHOLE CLIP"), call("recall_lessons", "l"), result("l")]
    for i in range(8):
        messages += [call("check_text", f"c{i}"), result(f"c{i}", f"check {i}")]
    return messages


def test_the_head_ends_at_the_first_working_step_and_keeps_the_notes_placed_before_it():
    messages = conversation()
    assert _frozen_head_end(messages) == 7 and messages[2]["content"] == "NOTE BEFORE ANY WORK"


def test_compaction_without_a_submission_keeps_the_head_exactly_as_it_was():
    messages = conversation()
    ledger = DraftLedger()
    ledger.set_segments([{"segment_id": "s1", "start_ms": 0, "end_ms": 1000}])
    ledger.observe("measure_tts", '{"text": "a draft", "start_ms": 0, "end_ms": 1000}', '{"ok": true}')
    compacted, dropped = compact_history(messages, ledger)
    assert dropped > 0 and compacted[:7] == messages[:7]                      # system, task, the note, the opened clip and the lessons: byte for byte the same, so their cache stays valid
    assert compacted[7]["content"].startswith("[Ledger") or "draft" in compacted[7]["content"]


def test_compaction_after_a_submission_keeps_the_head_too():
    messages = conversation() + [call("submit_dub", "s"), result("s", "SCORED")] + [call("check_text", "z"), result("z")]
    compacted, dropped = compact_history(messages, None)
    assert dropped > 0 and compacted[:7] == messages[:7]
