"""A dubbing agent, written from first principles: no agent framework.

An agent is a language model called in a loop with tools. Each turn the model
either asks to call tools (we run them and feed the results back as
observations) or answers in plain text (it believes it is done). The loop is the
whole idea; everything else here is what makes it safe and inspectable:

  * tools        hand-written JSON schemas -> Workbench methods (the environment)
  * observations a tool failure is returned to the model as data, never raised,
                 so the agent can recover (e.g. TTS budget exhausted -> submit now)
  * stop rules   the model decides it is done, OR a hard ceiling (turns, tokens)
  * completion   an agent that stops without a scored submission gets one nudge
  * trace        every model call and tool call is appended to trace.jsonl, so a
                 run can be read turn by turn -- that is where the learning is

``run_one_shot`` is the no-tools baseline (one prompt, one answer) so the value
the tools add is measurable on the same scoreboard.

Run:  python -m mathula_tv.dub_agent --clip fb3d08b6 --run-id agent-v1 [--mode oneshot]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import requests

from .dub_workbench import BudgetExhausted, Workbench

DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT = 2.0
DEFAULT_MAX_PUSHBACKS = 2
MAX_TOOL_RESULT_CHARS = 60000   # a 28-segment clip with ledger + glossary is ~19k; this is a runaway ceiling

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert English-to-isiZulu news dubbing agent working for Mathula TV.

Goal: dub ONE complete news clip. Produce isiZulu speech that a native speaker would find
natural, faithful to the English, and that fits the time the English speaker occupies on screen.

You work through tools and decide your own method. You are scored on:

1. Hard gates (a failure is a defect): every name, number and date survives; no invented names;
   the same entity is not stated twice for one English mention; nothing left untranslated.

2. Timing. Each unit's Zulu is played at its natural speed -- it is NEVER sped up or slowed down -- so the
   words you choose are the only thing that controls timing. The speech should fill its source window:
   too short = dead air while the speaker is still on screen; too long = an overrun that spills past its
   window into the next speaker.
   FIT FIRST. Fitting every unit into its time frame is the first job, because that is what keeps speakers
   from talking over each other and lets a multi-speaker clip flow from start to finish. The harness locks
   timing as you go: once a segment has fitted in an accepted submission, a later submission that breaks
   it is rejected unless it is clearly better overall.
   Priority when timing trades off: an overrun is WORSE than dead air. Speech that spills past its window
   collides with the next speaker and breaks sync, while a little silence is only a pause. So first make
   sure no unit overruns (overrun_pct must be 0), and only then shorten the dead air that is left. Never
   accept an overrun to remove a shortfall.
   Silence padding: the renderer closes a shortfall of up to 150 ms with trailing silence, and for a
   speaker turn of several consecutive units it can spread a larger shortfall over the real gaps between
   them (each gap widened by at most 1.5 s; nothing if the units are back to back). That is a fallback,
   not a goal: a viewer hears padding as a pause, so prefer wording that fills the time.

   Figures. Write numbers and amounts as digits ("u-4,500", "abantu abangu-500"). They are spoken in native
   isiZulu (izinkulungwane ezine namakhulu amahlanu zamaRandi), which takes longer than English. Dates and
   ordinals are spoken as code-switched English. When a unit does not fit its window, you may make a figure
   shorter by writing ~en straight after it ("u-4,500~en", "abantu abangu-500~en"): it is then spoken in English
   ("four thousand fayiv hundred rand"). The ~en is never spoken, and the digits stay for the number checks.
   Use it only where the time requires it; native is the default.

   English words you keep in the Zulu ("Fighters", "medical aid", names) are spoken by the Zulu voice with Zulu letter rules and can come
   out unrecognisable ("MaFighter" was heard as "mafactor"). Never glue a Zulu prefix onto an English word the speaker says as it is. Once the dub
   fits, call list_foreign_words; for the ones the speaker says in English, try_pronunciation shows the speaker's sounds and the sounds the voice
   makes for your spelling, with the differences. Adjust the spelling from the differences, then use_pronunciation for the closest. Tries are
   limited (5 per word, 15 per clip): spend them where the voice is clearly off, not on words that already sound right.
   Each result also says what an English listener wrote when the voice said your spelling ("english_hears"), and whether that is the word
   ("english_listener_understands"). That is your main check: a spelling the listener writes as the word beats one that only matches sounds,
   and a spelling they write as something else ("Scenes" for "science") is wrong however close its sounds score. Learn from what they wrote:
   it tells you which sound to change.
   Two sounds can share a label: the recogniser says "b" for Zulu's soft plain b and for an English b. So each result may also carry
   "how_the_sounds_are_made_differently", measured from both audios for the stops (p b t d k g): how sharp the release is, how long the
   closure before it lasts, whether the vocal folds vibrate through the closure, how long until voicing starts after the release, and the
   hiss after it. Zulu writes sounds that look alike in English with different letters (a plain letter and its h-form can differ in exactly
   these ways), so read the measured difference, decide which letters would change that property, and confirm it with the English
   listener. The listener is for ordinary English words only: South African names (Nkomazi, Malema, places, parties) are not listed for
   you and it cannot judge them, so never respell a name by what it writes.
   Read each difference as the letters to change, and leave the parts that already match alone. Two spellings that make the same sounds
   are one idea (the result says so). A word whose closest spelling is still under about 0.85 deserves the rest of its tries on spellings that
   change the named sounds ("speaker k s, voice tʃ": write the k and the s so the voice keeps both) before you settle; stopping after one call
   with tries left leaves a word badly said.

3. Naturalness: retell the way a person would say it informally. This is not a word-for-word transfer.
   Use the whole clip's context.
   This is a SOCIAL MEDIA product: simplicity and maximum reach are the point. A viewer gives a clip a second or
   two, so use plain, short, everyday wording that lands immediately for as many people as possible.
   Widest understanding beats formality. Choose the word that the most everyday isiZulu listeners understand the
   moment they hear it -- urban and rural, older and younger, first- and second-language speakers -- even if it is
   a loanword or an informal form, and even if a more formal or purist word exists ("igranti", not "isibonelelo";
   "i-VAT"; "imeya"). The glossary fixes names and terms for consistency; where one of its terms is a stiffer
   or less familiar word than a common loanword for the same thing, use the familiar word.
   Language switches: the speaker moves between English, isiZulu and sometimes Sepedi or another language. Where the English transcript of a
   stretch is wrong, the clip's `code_switch` section says what he really said (rebuilt from the sounds, the recognisers, the captions and
   the argument), in which language, how sure it is and why. Politicians speak like this on purpose, so you read it as a bilingual
   listener would: use it for the meaning, keep his isiZulu as isiZulu, say other languages' meaning in plain isiZulu. When it is 'uncertain'
   decide between its reading and the 'or_perhaps' one from the argument and dub your choice; do not skip the stretch or keep only the clear
   English (a lost 'unga-' turns "do not confuse" into "confuse").
   Delivery: the clip carries a `delivery` sheet measured from the audio. A speaker people know (a politician above all) should be
   recognisable in the dub, so write the isiZulu with their shape: keep their pace (a quick run stays short and quick, a slow weighted
   line stays weighted), keep the pauses they take for effect by ending the sentence there, keep the phrases they repeat repeated, and
   let the words they lean on (stressed_words) land at the end of a phrase or right after a pause. This shapes wording only: never
   lengthen a unit past its window for it, never invent a pause, and the meaning and plain wording above still come first.
   The voice leaves only about half a second after a sentence. Where the sheet shows the speaker holding a pause of 600 ms or more
   (pauses_inside), you may write ~pause900 (milliseconds, 200-2000) at the matching spot in your isiZulu: "...ngentela yethu. ~pause900 Uma
   uya esitolo...". It is never spoken, it is a silence of that length that counts toward the unit's duration like any other part, and
   it is also the cheapest way to fill a window honestly. Use it only for pauses the speaker really takes, at most two per unit.

Tools give you real measurements. Azure TTS timing is not predictable from word counts, and with no speed
control you must get the wording right. The scorer re-measures your final text with the same function, so
what you measure is what you are graded on. Measurements are budgeted.

Every tool call re-sends your whole history, so each extra turn costs more than the last. After you
submit, your earlier working steps (drafts, measurements, checks) are trimmed from your history: the clip,
your latest submit_dub call and its result are always kept.

submit_dub takes the full set of units, or, with partial=true, only the units you changed: each replaces the
current unit with the same item_id (ids in remove are deleted) and the rest carry over. Sending only what
changed is much cheaper than rewriting the whole clip. get_submission shows your current units
(all, or just the ones you ask for), so you never have to rewrite from memory.

The bar for "done" (checked by the harness against your last submit_dub result, not by your say-so): no unit
overruns or runs into another unit's speech (collisions), no hard-gate failures, no unit
whose meaning an independent reader finds materially changed from the English (submit_dub reports these as
meaning_loss, with what your Zulu reads back as; a lost qualifier, a changed time or negation, a dropped clause that
carried a fact), and residual_dead_air_pct at or below {max_dead_air}. If you stop while a submission is still below that bar and you have budget left,
you will be sent back to the specific units to fix.

When you are satisfied, call submit_dub (it returns your score and which units are flagged; you may revise
and resubmit). Then reply with a short plain-text summary of your approach and stop.
"""

SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.replace("{max_dead_air}", f"{DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT}%")


def build_system_prompt(playbook: dict | None = None, *, max_dead_air_pct: float = DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT) -> str:
    """The rules (always) plus the learned strategy notes (if a playbook is given)."""
    from .playbook import render_section

    base = _SYSTEM_PROMPT_TEMPLATE.replace("{max_dead_air}", f"{max_dead_air_pct}%")
    return base + render_section(playbook)

_UNIT_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id": {"type": "string"},
        "start_ms": {"type": "integer"},
        "end_ms": {"type": "integer"},
        "speaker_id": {"type": "string"},
        "zulu_text": {"type": "string"},
        "source_segment_ids": {"type": "array", "items": {"type": "string"}},
        "source_text": {"type": "string", "description": "Only for units that share a source segment: the exact English excerpt this unit covers, copied from the segment (checked; a paraphrase is rejected). Ignored for other units."},
    },
    "required": ["item_id", "start_ms", "end_ms", "speaker_id", "zulu_text", "source_segment_ids"],
}
_MEASURE_PROPS = {
    "text": {"type": "string", "description": "The isiZulu text to synthesize."},
    "start_ms": {"type": "integer"},
    "end_ms": {"type": "integer"},
    "next_start_ms": {"type": ["integer", "null"], "description": "When the next speech starts."},
    "speaker_id": {"type": "string"},
    "source_text": {"type": "string", "description": "The English this text renders."},
}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOL_SPECS = [
    _tool("open_clip", "Open the clip: returns every English segment (id, speaker, start/end ms), the whole-clip "
          "context ledger, glossary and speaker modes. Call this first. If source_corrections is present, the English had "
          "words the speaker's audio showed were misheard; the english already carries the fix, so translate that meaning.",
          {"clip_id": {"type": "string"}}, ["clip_id"]),
    _tool("recall_lessons", "Mistakes native-speaker reviewers marked 'bad' in earlier clips, with corrections "
          "where given. Cross-clip memory.", {"limit": {"type": "integer"}}, []),
    _tool("measure_tts", "Synthesize text with Azure TTS at natural speed and report how it sits in "
          "[start_ms, end_ms]: measured_ms, shortfall_pct (speech missing), tail_pad_ms (trailing silence the "
          "renderer adds itself when the gap is tiny), residual_shortfall_pct (what is left as dead air), "
          "overrun_pct (how far past its allowance it runs), direction ('expand' too short, 'compress' too "
          "long, null fits), fits_naturally. Costs 1 measurement.",
          _MEASURE_PROPS, ["text", "start_ms", "end_ms"]),
    _tool("measure_tts_batch", "Measure up to 40 texts at once; same fields as measure_tts. Costs 1 per text.",
          {"requests": {"type": "array", "items": {"type": "object", "properties": _MEASURE_PROPS,
                                                    "required": ["text", "start_ms", "end_ms"]}}}, ["requests"]),
    _tool("preview_turn_padding", "Free arithmetic (no TTS). If a speaker turn made of these units finishes early, "
          "how much silence does the renderer add, and how much remains as dead air? Mechanisms in order: a "
          "tail pad of up to 150 ms when it fits the free gap, else the shortfall is spread over the turn's real "
          "inter-sentence gaps (each gap widened at most 1500 ms). members: {start_ms, end_ms, measured_ms}.",
          {"members": {"type": "array", "items": {"type": "object", "properties": {
              "start_ms": {"type": "integer"}, "end_ms": {"type": "integer"}, "measured_ms": {"type": "integer"}},
              "required": ["start_ms", "end_ms", "measured_ms"]}},
           "last_free_gap_ms": {"type": "integer", "description": "free_gap_after_ms of the turn's last unit."}},
          ["members"]),
    _tool("check_text", "Free deterministic checks on one unit (no TTS): required literals, restated references, "
          "invented names, doubling, English leakage.",
          {"zulu_text": {"type": "string"}, "source_text": {"type": "string"}}, ["zulu_text", "source_text"]),
    _tool("submit_dub", "Submit the dub as ordered, non-overlapping units covering every English segment. "
          "Returns the official score and flagged units. You may resubmit. With partial=true, send ONLY the units you "
          "changed: each replaces the current unit with the same item_id, ids listed in remove are deleted, and every "
          "other current unit carries over (needs a previous accepted submission). To re-split units, send the new "
          "units and list the old ids in remove.",
          {"items": {"type": "array", "items": _UNIT_SCHEMA}, "approach_notes": {"type": "string"},
           "partial": {"type": "boolean", "description": "true = update only the units you send."},
           "remove": {"type": "array", "items": {"type": "string"},
                      "description": "partial only: ids of current units to delete (e.g. units you re-split)."}}, ["items"]),
    _tool("list_foreign_words", "After your dub fits: the English words your Zulu keeps that the speaker says in the source audio "
          "(including ones you glued a Zulu prefix onto), so you know which to check with try_pronunciation.", {}, []),
    _tool("try_pronunciation", "Hear what the Zulu voice does with up to 3 spellings of an English word. For each spelling it returns the sounds "
          "the voice made, the speaker's own sounds for the word (decoded from the source audio), how close they are (0-1) and what differs "
          "(\"speaker ɔɪ, voice o\", \"the voice leaves out d\"), what an English listener writes for the voice's audio (english_hears), "
          "how its stops are made differently from the speaker's (release, closure, voicing), plus advice on what to change next. Use the differences to choose "
          "your next spelling. A try is budgeted "
          "(5 per word, 15 per clip).",
          {"word": {"type": "string", "description": "the word as written in your Zulu, e.g. MaFighter or medical"},
           "spellings": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                         "description": "spellings to try, e.g. [\"Fighters\", \"Fayitas\"]; respell with Zulu letters to steer the sounds"}},
          ["word", "spellings"]),
    _tool("use_pronunciation", "Make a spelling you tried the way this word is spoken: for your own measurements from now on, in the render, "
          "and remembered for later clips. Only a spelling you have tried can be used.",
          {"word": {"type": "string"}, "spelling": {"type": "string"}}, ["word", "spelling"]),
    _tool("get_submission", "The units currently accepted (all, or only unit_ids), with their text and any flags from "
          "the last score. Use it to see your own current state, e.g. after your history was trimmed.",
          {"unit_ids": {"type": "array", "items": {"type": "string"}}}, []),
]


REQUEST_TIMEOUT_SECONDS = 420.0      # one model turn; the provider default (900 s) lets a dropped connection stall a run for 15 minutes


def post_with_retry(url: str, headers: dict[str, str], body: dict[str, Any], *, timeout: float, retries: int,
                    sleep: Callable[[float], None]) -> dict[str, Any]:
    """POST once, retrying transient failures a bounded number of times.

    Transient = HTTP 429/5xx, a read timeout or a dropped connection. Each is retried with backoff up to
    ``retries`` times; anything else (a 4xx, bad JSON) is raised at once. A model call that hangs must cost a
    bounded delay, never a whole batch.
    """
    for attempt in range(1, retries + 2):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt <= retries:
                sleep(min(2 ** attempt, 20))
                continue
            raise RuntimeError(f"model call failed after {attempt} attempts: {type(exc).__name__}") from exc
        if response.status_code in (429, 500, 502, 503, 504) and attempt <= retries:
            sleep(min(2 ** attempt, 20))
            continue
        if response.status_code >= 400:
            raise RuntimeError(f"model HTTP {response.status_code}: {response.text[:600]}")
        return response.json()
    raise RuntimeError("model call failed")


class ChatClient:
    """One chat-completions call with tools, over plain HTTP. No SDK, no hidden retries beyond transport."""

    def __init__(self, config: Any, *, retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self.retries = retries
        self.sleep = sleep
        self.url = f"{config.endpoint}/openai/v1/chat/completions"

    def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
                 json_mode: bool = False, max_output_tokens: int = 16000) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.deployment,
            "messages": [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages],
            "max_completion_tokens": max_output_tokens,
        }
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        payload = post_with_retry(self.url, headers, body, timeout=min(self.config.timeout_seconds, REQUEST_TIMEOUT_SECONDS),
                                  retries=self.retries, sleep=self.sleep)
        usage = payload.get("usage") or {}
        usage["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        return payload


class ResponsesClient:
    """The /v1/responses API, which reasoning models need in order to use function tools.

    The loop keeps a neutral chat-style history; this client translates it on every call. The
    model's own output items (including its reasoning) ride along on the assistant message under
    ``_raw`` and are replayed verbatim, because a reasoning model plans across tool calls and
    loses that plan if only the visible text is sent back.
    """

    def __init__(self, config: Any, *, retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self.retries = retries
        self.sleep = sleep
        self.url = f"{config.endpoint}/openai/v1/responses"

    @staticmethod
    def to_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "tool":
                items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                              "output": message["content"]})
            elif role == "assistant":
                if message.get("_raw"):
                    items.extend(message["_raw"])
                elif message.get("content"):
                    items.append({"role": "assistant", "content": message["content"]})
            else:
                items.append({"role": role, "content": message["content"]})
        return items

    @staticmethod
    def to_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        return [{"type": "function", **t["function"]} for t in tools or []]

    @staticmethod
    def to_chat_shape(response: dict[str, Any]) -> dict[str, Any]:
        output = response.get("output") or []
        text = "".join(
            part.get("text", "") for item in output if item.get("type") == "message"
            for part in item.get("content") or [] if part.get("type") == "output_text"
        )
        calls = [
            {"id": item["call_id"], "type": "function",
             "function": {"name": item["name"], "arguments": item.get("arguments") or "{}"}}
            for item in output if item.get("type") == "function_call"
        ]
        message: dict[str, Any] = {"role": "assistant", "content": text or None, "_raw": output}
        if calls:
            message["tool_calls"] = calls
        usage = response.get("usage") or {}
        return {
            "choices": [{"message": message, "finish_reason": "tool_calls" if calls else response.get("status")}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0),
                "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
            },
        }

    def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
                 json_mode: bool = False, max_output_tokens: int = 16000) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.deployment, "input": self.to_input(messages),
            "max_output_tokens": max_output_tokens,
        }
        if self.config.reasoning_effort:
            body["reasoning"] = {"effort": self.config.reasoning_effort}
        if tools:
            body["tools"] = self.to_tools(tools)
        if json_mode:
            body["text"] = {"format": {"type": "json_object"}}
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        return self.to_chat_shape(post_with_retry(
            self.url, headers, body, timeout=min(self.config.timeout_seconds, REQUEST_TIMEOUT_SECONDS),
            retries=self.retries, sleep=self.sleep))


class Trace:
    """Append-only JSONL log of everything the agent saw and did."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._started = time.monotonic()

    def log(self, kind: str, **fields: Any) -> None:
        record = {"t": round(time.monotonic() - self._started, 2), "kind": kind, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return text[:MAX_TOOL_RESULT_CHARS] + f"\n...[truncated {len(text) - MAX_TOOL_RESULT_CHARS} chars]"


COMPACT_AFTER_TOKENS = 15_000
_KEEP_STEP_TOOLS = frozenset({"open_clip", "recall_lessons"})
COMPACTION_NOTE = (
    "[Earlier working steps (drafts, measurements, checks, past feedback) were removed to save tokens. The harness "
    "holds your accepted submission: call get_submission to see the current units. Your latest submit_dub call and "
    "its result are below; the opened clip above is unchanged.]"
)


def _step_tools(messages: list[dict[str, Any]], index: int) -> set[str]:
    return {c["function"]["name"] for c in messages[index].get("tool_calls") or []}


def compact_history(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop the working steps a newer submission has superseded. Returns (messages, dropped_count).

    A step is an assistant message plus the tool results that answer it; dropping whole steps keeps
    every remaining tool call paired with its result. What stays: the system prompt, the task, the
    steps that opened the clip and recalled lessons, the LATEST submit_dub step, and everything after
    it. What goes: every step and every user note between them (earlier drafts, measurements, checks,
    old pushbacks). Idempotent: a second pass over a compacted history changes nothing.
    """
    last_submit = None
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and "submit_dub" in _step_tools(messages, index):
            last_submit = index
    if last_submit is None:
        return messages, 0
    keep = {0, 1}
    index = 2
    while index < last_submit:
        message = messages[index]
        if message.get("role") == "assistant":
            end = index + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                end += 1
            if _step_tools(messages, index) & _KEEP_STEP_TOOLS:
                keep.update(range(index, end))
            index = end
        else:
            index += 1
    dropped = [i for i in range(2, last_submit) if i not in keep and messages[i].get("content") != COMPACTION_NOTE]
    if not dropped:
        return messages, 0
    kept = [m for i, m in enumerate(messages) if i not in dropped and m.get("content") != COMPACTION_NOTE]
    position = next(i for i, m in enumerate(kept) if m is messages[last_submit])
    kept.insert(position, {"role": "user", "content": COMPACTION_NOTE})
    return kept, sum(1 for i in dropped if messages[i].get("role") == "assistant")


_SLIM_HEADLINE = (
    "group_count", "dead_air_pct", "turn_shortfall_pct", "silence_padded_pct", "residual_dead_air_pct",
    "groups_dead_air", "groups_rushed", "gate_failures", "worst_turn_shortfall_pct",
    "collisions", "cross_speaker_collisions", "worst_collision_ms", "meaning_loss_units",
)


def slim_submit(result: dict[str, Any]) -> dict[str, Any]:
    """The model-facing view of a submit_dub result (the harness keeps the full one)."""
    if not result.get("accepted"):
        return result
    slim: dict[str, Any] = {
        "accepted": True,
        "headline": {k: result["headline"][k] for k in _SLIM_HEADLINE if k in result["headline"]},
    }
    if result.get("merged"):
        slim["merged"] = result["merged"]
    if result.get("timing_lock"):
        slim["timing_lock"] = result["timing_lock"]
    if result.get("regressions_accepted"):
        slim["regressions_accepted"] = result["regressions_accepted"]
        slim["note"] = result.get("note")
    gates = {k: v for k, v in (result.get("gates") or {}).items() if v}
    if gates:
        slim["gates"] = gates
    padding = [t for t in result.get("turn_padding", []) if t.get("residual_ms", 0) > 0]
    if padding:
        slim["turns_ending_early"] = padding
    if result.get("flagged_units"):
        slim["flagged_units"] = result["flagged_units"]
    if (result.get("meaning") or {}).get("loss"):
        slim["meaning_loss"] = result["meaning"]["loss"]
    if result.get("warnings"):
        slim["warnings"] = result["warnings"]
    return slim


def build_dispatch(
    workbench: Workbench, *, clip_id: str, run_id: str, tts_budget: int, speed_policy: str = "natural",
    repeat: int = 1,
) -> dict[str, Callable[..., Any]]:
    """Tool name -> callable. The harness, not the model, fixes the run id and the measurement budget."""

    episode_clip = clip_id

    def open_clip(clip_id: str = episode_clip) -> Any:
        if clip_id != episode_clip:
            raise ValueError(f"this episode is for clip {episode_clip!r}")
        return workbench.open_clip(
            episode_clip, run_id, repeat=repeat, tts_budget=tts_budget, speed_policy=speed_policy,
        )

    tools: dict[str, Callable[..., Any]] = {
        "open_clip": open_clip,
        "recall_lessons": workbench.recall_lessons,
        "measure_tts": workbench.measure_tts,
        "measure_tts_batch": workbench.measure_tts_batch,
        "check_text": workbench.check_text,
        "preview_turn_padding": workbench.preview_turn_padding,
        "submit_dub": lambda **kw: slim_submit(workbench.submit_dub(**kw)),
        "get_submission": workbench.get_submission,
    }
    for name in ("list_foreign_words", "try_pronunciation", "use_pronunciation"):       # the pronunciation lab, where the workbench has one
        if hasattr(workbench, name):
            tools[name] = getattr(workbench, name)
    return tools


def execute_tool(dispatch: dict[str, Callable[..., Any]], name: str, raw_arguments: str) -> tuple[str, bool]:
    """Run one tool call. Returns (observation text, ok). Failures become observations, not exceptions."""
    function = dispatch.get(name)
    if function is None:
        return json.dumps({"error": f"unknown tool {name!r}", "available": sorted(dispatch)}), False
    try:
        arguments = json.loads(raw_arguments or "{}")
        result = function(**arguments)
        return _clip(json.dumps(result, ensure_ascii=False)), True
    except BudgetExhausted as exc:
        return json.dumps({"error": "budget_exhausted", "detail": str(exc)}), False
    except Exception as exc:  # the model must be able to see and recover from any tool failure
        return json.dumps({"error": type(exc).__name__, "detail": str(exc)[:600]}), False


def unmet_targets(result: dict[str, Any] | None, *, max_dead_air_pct: float) -> list[str]:
    """What is still wrong with the last accepted submission, judged by the harness from its own score."""
    headline = (result or {}).get("headline")
    if not headline:
        return []
    unmet = []
    if headline.get("cross_speaker_collisions", 0) > 0:
        unmet.append(f"{headline['cross_speaker_collisions']} unit(s) run into ANOTHER speaker's speech")
    elif headline.get("collisions", 0) > 0:
        unmet.append(f"{headline['collisions']} unit(s) run into the next unit's speech")
    if headline.get("groups_rushed", 0) > 0:
        unmet.append(f"{headline['groups_rushed']} unit(s) overrun their window")
    if headline.get("gate_failures", 0) > 0:
        unmet.append(f"{headline['gate_failures']} hard-gate failure(s)")
    if headline.get("meaning_loss_units", 0) > 0:
        unmet.append(f"{headline['meaning_loss_units']} unit(s) lose material meaning (see meaning_loss)")
    residual = headline.get("residual_dead_air_pct")
    if residual is not None and residual > max_dead_air_pct:
        unmet.append(f"residual dead air is {residual}% (bar: {max_dead_air_pct}% or less)")
    return unmet


def pushback_message(result: dict[str, Any], unmet: list[str], *, tts_left: int) -> str:
    lines = ["Not done yet. Your last submission misses the bar: " + "; ".join(unmet) + "."]
    short = sorted((t for t in result.get("turn_padding", []) if t.get("residual_ms", 0) > 0),
                   key=lambda t: -t["residual_ms"])[:6]
    if short:
        lines.append("Turns still ending early after silence padding: " + "; ".join(
            f"{t['turn_id']} ({t['residual_ms'] / 1000:.1f} s of dead air, units {', '.join(t['units'])})" for t in short) + ".")
    flagged = result.get("flagged_units", [])[:10]
    if flagged:
        lines.append("Flagged units: " + "; ".join(
            f"{f['item_id']} [{', '.join(f['gates'] + f['flags'])}]" for f in flagged) + ".")
    lost = (result.get("meaning") or {}).get("loss", [])[:6]
    if lost:
        lines.append("Meaning lost (an independent reader of your Zulu disagrees with the English): " + "; ".join(
            f"{m['item_id']}: {'; '.join(m['issues']) or 'unclear'} (your Zulu reads back as: {m['reads_back_as']!r})"
            for m in lost) + ".")
    lock = result.get("timing_lock")
    if lock and lock.get("fitting_segments"):
        lines.append(
            f"Timing is locked for {lock['fitting_segments']} of {lock['total_segments']} segments: a submission that "
            "breaks one of them is rejected, so fix the failing units without touching the ones that fit."
        )
    lines.append(
        f"You still have {tts_left} measurements. Fix these units and resubmit: where speech is short, restore faithful "
        "content from the English you compressed or dropped (never filler); where it overruns, tighten. "
        "Overruns come first. If you truly cannot meet the bar, resubmit your best and say exactly which units and why."
    )
    return " ".join(lines)


def _source_correction_note(workbench: Any, seed_items: list[dict[str, Any]]) -> str:
    """In a repair run: what changed since the earlier submission (source corrections, glossary terms) and the units it touches."""
    note = ""
    corrections = getattr(workbench, "source_corrections", None) or []
    if corrections:
        parts = []
        for correction in corrections:
            units = [str(item.get("item_id") or "?") for item in seed_items
                     if correction["segment_id"] in [str(s) for s in item.get("source_segment_ids") or []]]
            parts.append(f"segment {correction['segment_id']}: the transcript had {correction['token']!r} but the speaker said "
                         f"{correction['replacement']!r}" + (f" (units: {', '.join(units)})" if units else ""))
        note += ("The English source was corrected after that submission, against the speaker's audio: " + "; ".join(parts) +
                 ". Re-translate the units that carry these segments so the Zulu says what the speaker meant, and treat them "
                 "as units that need change. ")
    changes = getattr(workbench, "glossary_changes", None) or []
    if changes:
        parts = []
        for change in changes:
            old_forms = [f.strip() for f in change["was"].split("/") if f.strip()]
            units = [str(item.get("item_id") or "?") for item in seed_items
                     if any(form.casefold() in str(item.get("zulu_text") or "").casefold() for form in old_forms)]
            if units:
                parts.append(f"{change['english']!r}: now {change['now']!r} (was {change['was']!r}) in units {', '.join(units)}")
        if parts:
            note += ("The glossary was revised for simpler wording with the widest reach. These units use a term it replaced; "
                     "change that term in them and treat them as units that need change: " + "; ".join(parts) + ". ")
    return note


PRONUNCIATION_REMINDER = (
    "One reminder, once: your Zulu keeps English words that the speaker says and that have no checked reading yet, so the voice may read them "
    "with Zulu letters: {words}. If any of them may sound wrong, call list_foreign_words, check those with try_pronunciation, use "
    "use_pronunciation for a spelling that an English listener writes as the word, re-measure any unit whose words change, and finish. "
    "If you judge they are fine as they are, you do not have to use the lab: finish.")


# The agent fits timing first and looks at pronunciation last, so a run that spends all its turns on timing never reaches the lab (prod-21
# stopped at max_turns with the lab untouched).  Once an accepted draft exists and this few turns remain, the same one reminder is sent
# early, with the turns left, so there is room to check the words and re-measure.
PRONUNCIATION_RESERVE_TURNS = 20
PRONUNCIATION_LATE_NOTE = "You have {left} turns left; spend at most 8 of them on this and keep the rest to re-measure and re-submit. "


def run_agent(client: ChatClient, workbench: Workbench, **kwargs: Any) -> dict[str, Any]:
    """Run the agent (see ``_run_agent``), then remove this run's scratch TTS audio. Records (submission, scorecard, trace) stay.

    Only this run's folder is touched: autopilot batches run several repeats at once, so nothing here may clean another run's."""
    from .housekeeping import clean_measure, keep_measure

    try:
        return _run_agent(client, workbench, **kwargs)
    finally:
        episode = getattr(workbench, "episode", None)
        if not keep_measure() and episode and episode.get("dir"):
            try:
                clean_measure(workbench.store.root, Path(episode["dir"]))
            except Exception:        # cleaning is housekeeping: it must never turn a finished run into a failure
                pass


def _run_agent(
    client: ChatClient, workbench: Workbench, *, clip_id: str, run_id: str, repeat: int = 1,
    max_turns: int = 40, max_total_tokens: int = 500_000, tts_budget: int = 300,
    system_prompt: str = SYSTEM_PROMPT, speed_policy: str = "natural",
    max_dead_air_pct: float = DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT, max_pushbacks: int = DEFAULT_MAX_PUSHBACKS,
    task_message: str | None = None, trace_dir: Path | None = None,
    compact_after_tokens: int = COMPACT_AFTER_TOKENS, playbook_version: str | None = None,
    seed_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """``seed_items`` starts a REPAIR run: the units of an earlier accepted submission are re-scored under today's rules
    and the agent is told to change only what misses the bar, leaving every other unit exactly as it was."""
    directory = Path(trace_dir) if trace_dir else workbench.store.repeat_dir(run_id, clip_id, repeat)
    trace = Trace(directory / "trace.jsonl")
    dispatch = build_dispatch(
        workbench, clip_id=clip_id, run_id=run_id, tts_budget=tts_budget, speed_policy=speed_policy,
        repeat=repeat,
    )
    if seed_items:
        workbench.open_clip(clip_id, run_id, repeat=repeat, tts_budget=tts_budget, speed_policy=speed_policy)
        seeded = workbench.submit_dub(list(seed_items), approach_notes="seeded from an earlier accepted submission")
        trace.log("seed", accepted=seeded.get("accepted"), errors=seeded.get("errors"), headline=seeded.get("headline"))
        if seeded.get("accepted"):
            seed_unmet = unmet_targets(seeded, max_dead_air_pct=max_dead_air_pct)
            tts_now = int((workbench.episode or {}).get("tts_budget", tts_budget)) - int((workbench.episode or {}).get("tts_used", 0))
            task_message = task_message or (
                f"Repair, do not redo. Clip {clip_id!r} already has an accepted submission (get_submission shows its units). "
                f"Re-scored under the current pronunciation and timing rules it "
                + ("misses the bar. " + pushback_message(seeded, seed_unmet, tts_left=tts_now) if seed_unmet else "meets the bar. ")
                + _source_correction_note(workbench, seed_items)
                + "Change only the units that need it, send just those with submit_dub partial=true, leave every other unit "
                  "exactly as it is, and stop once the bar is met.")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_message or f"Dub clip {clip_id!r}. Start by opening it."},
    ]
    tokens = {"in": 0, "out": 0, "cached": 0}
    last_prompt_tokens = 0
    compactions = 0
    tool_counts: dict[str, int] = {}
    stop_reason = "max_turns"
    nudged = False
    pronunciation_reminded = False
    pushbacks = 0
    unmet: list[str] = []
    turn = 0

    def accepted() -> bool:
        return bool(workbench.episode and workbench.episode.get("submissions"))

    for turn in range(1, max_turns + 1):
        if compact_after_tokens and last_prompt_tokens >= compact_after_tokens:
            compacted, dropped_steps = compact_history(messages)
            if dropped_steps:
                trace.log("compact", turn=turn, dropped_steps=dropped_steps, messages_before=len(messages),
                          messages_after=len(compacted), prompt_tokens_before=last_prompt_tokens)
                messages[:] = compacted
                compactions += 1
        if not pronunciation_reminded and accepted() and turn >= max_turns - PRONUNCIATION_RESERVE_TURNS:
            pending = getattr(workbench, "pronunciation_check_pending", None)
            words = pending() if callable(pending) else []
            pronunciation_reminded = True                      # asked once whether or not there is anything to check
            if words:
                text = PRONUNCIATION_LATE_NOTE.format(left=max_turns - turn + 1) + PRONUNCIATION_REMINDER.format(words=", ".join(words[:12]))
                messages.append({"role": "user", "content": text})
                trace.log("pronunciation_reminder", turn=turn, words=words[:12], text=text, early=True)
        response = client.complete(messages, tools=TOOL_SPECS)
        usage = response.get("usage") or {}
        last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        tokens["in"] += last_prompt_tokens
        tokens["out"] += int(usage.get("completion_tokens") or 0)
        tokens["cached"] += int(usage.get("cached_tokens") or 0)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []
        trace.log("model", turn=turn, finish=choice.get("finish_reason"), text=message.get("content"),
                  tool_calls=[{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in calls],
                  usage=usage)
        messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls", "_raw") and v is not None})

        if not calls:
            if not accepted() and not nudged:
                nudged = True
                nudge = "You have not submitted a dub yet. Call submit_dub with your best units now."
                messages.append({"role": "user", "content": nudge})
                trace.log("nudge", turn=turn, text=nudge)
                continue
            if not accepted():
                stop_reason = "finished_without_submission"
                break
            # The pronunciation lab is optional in the prompt, so an agent can finish without ever looking at the English words it kept
            # (prod-19 did).  If the sweep finds words with no checked reading, one reminder, once, naming them; with none, no reminder.
            pending = getattr(workbench, "pronunciation_check_pending", None)
            words = pending() if callable(pending) and not pronunciation_reminded and tokens["in"] + tokens["out"] < 0.8 * max_total_tokens else []
            if words:                                # only when there is something to check: no need, no reminder
                pronunciation_reminded = True
                text = PRONUNCIATION_REMINDER.format(words=", ".join(words[:12]))
                messages.append({"role": "user", "content": text})
                trace.log("pronunciation_reminder", turn=turn, words=words[:12], text=text)
                continue
            # The agent says it is done. The harness checks that against its own score instead of
            # taking its word for it -- bounded, and only while budget remains to act on it.
            unmet = unmet_targets(getattr(workbench, "last_submit", None), max_dead_air_pct=max_dead_air_pct)
            tts_left = int((workbench.episode or {}).get("tts_budget", tts_budget)) - int((workbench.episode or {}).get("tts_used", 0))
            if (unmet and pushbacks < max_pushbacks and tts_left > 0
                    and tokens["in"] + tokens["out"] < 0.8 * max_total_tokens):
                pushbacks += 1
                text = pushback_message(workbench.last_submit, unmet, tts_left=tts_left)
                messages.append({"role": "user", "content": text})
                trace.log("pushback", turn=turn, number=pushbacks, unmet=unmet, text=text)
                continue
            stop_reason = "finished_below_target" if unmet else "agent_finished"
            break

        for call in calls:
            name = call["function"]["name"]
            started = time.monotonic()
            observation, ok = execute_tool(dispatch, name, call["function"]["arguments"])
            tool_counts[name] = tool_counts.get(name, 0) + 1
            trace.log("tool", turn=turn, name=name, ok=ok, seconds=round(time.monotonic() - started, 2),
                      result=observation[:3000])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})

        if tokens["in"] + tokens["out"] > max_total_tokens:
            stop_reason = "token_budget"
            break

    summary = {
        "mode": "agent", "clip_id": clip_id, "run_id": run_id, "stop_reason": stop_reason, "turns": turn,
        "tokens": tokens, "tool_calls": tool_counts, "speed_policy": speed_policy,
        "pushbacks": pushbacks, "unmet_at_end": unmet, "compactions": compactions,
        "playbook": playbook_version,
        **({"judge_tokens": dict(workbench.judge.usage)} if getattr(workbench, "judge", None) else {}),
        "tts_used": (workbench.episode or {}).get("tts_used"), "submissions": (workbench.episode or {}).get("submissions"),
        "model": client.config.deployment,
    }
    trace.log("end", **summary)
    (directory / "agent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


ONE_SHOT_PROMPT = """\
Dub this complete English news clip into isiZulu. Reply with JSON only:
{"items":[{"item_id":"u1","start_ms":0,"end_ms":0,"speaker_id":"","zulu_text":"","source_segment_ids":["..."]}]}
Units must be ordered, non-overlapping, inside the clip, and cover every English segment.
Retell naturally and informally (not word for word), keep every name, number and date, invent no names,
and aim for speech that fills each unit's time window at natural speaking speed (audio is never sped up).

CLIP:
"""


def run_one_shot(
    client: ChatClient, workbench: Workbench, *, clip_id: str, run_id: str, repeat: int = 1, tts_budget: int = 300,
    speed_policy: str = "natural",
) -> dict[str, Any]:
    """Baseline: no tools, no loop. One model call, then the same official scoring."""
    directory = workbench.store.repeat_dir(run_id, clip_id, repeat)
    trace = Trace(directory / "trace.jsonl")
    clip = workbench.open_clip(clip_id, run_id, repeat=repeat, tts_budget=tts_budget, speed_policy=speed_policy)
    response = client.complete(
        [{"role": "system", "content": "You are an expert English-to-isiZulu news dubbing translator."},
         {"role": "user", "content": ONE_SHOT_PROMPT + json.dumps(clip, ensure_ascii=False)}],
        json_mode=True,
    )
    usage = response.get("usage") or {}
    text = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    trace.log("model", turn=1, text=text[:6000], usage=usage)
    items = json.loads(text).get("items", [])
    result = workbench.submit_dub(items, approach_notes="one-shot baseline: single call, no tools")
    trace.log("tool", turn=1, name="submit_dub", ok=bool(result.get("accepted")), result=json.dumps(result, ensure_ascii=False)[:3000])
    summary = {
        "mode": "oneshot", "clip_id": clip_id, "run_id": run_id, "speed_policy": speed_policy,
        "stop_reason": "agent_finished" if result.get("accepted") else "submission_rejected", "turns": 1,
        "tokens": {"in": int(usage.get("prompt_tokens") or 0), "out": int(usage.get("completion_tokens") or 0)},
        "tool_calls": {"submit_dub": 1}, "tts_used": (workbench.episode or {}).get("tts_used"),
        "submissions": (workbench.episode or {}).get("submissions"), "model": client.config.deployment,
        "errors": result.get("errors"),
    }
    trace.log("end", **summary)
    (directory / "agent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_agent")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("agent", "oneshot"), default="agent")
    parser.add_argument("--api", choices=("responses", "chat"), default="responses",
                        help="model API for agent mode (reasoning models need 'responses' to use tools)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--speed-policy", choices=("natural", "pipeline"), default="natural",
                        help="natural: audio is never sped up, any overrun is a failure; "
                             "pipeline: tolerate the renderer's bounded speed-up")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--tts-budget", type=int, default=300)
    parser.add_argument("--max-pushbacks", type=int, default=DEFAULT_MAX_PUSHBACKS,
                        help="times the harness sends the agent back when its last score misses the bar")
    parser.add_argument("--max-dead-air", type=float, default=DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT)
    parser.add_argument("--playbook", default=None,
                        help="playbook version to load into the system prompt ('current' = the adopted one, "
                             "'none' = rules only; default: current)")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the independent meaning check (it is part of the bar by default)")
    parser.add_argument("--compact-after", type=int, default=COMPACT_AFTER_TOKENS,
                        help="prompt-token size above which superseded working steps are dropped (0 = never)")
    args = parser.parse_args(argv)

    from .cli import create_tts_backend
    from .config import load_settings
    from .experience_store import ExperienceStore
    from .foundry_grok import native_gpt_config_from_environment

    settings = load_settings()
    config = native_gpt_config_from_environment(dict(os.environ))
    judge = None
    if not args.no_judge and args.mode == "agent":
        from .dub_judge import UnitJudge

        judge = UnitJudge(ChatClient(config))
    workbench = Workbench(ExperienceStore(settings.work_dir), lambda: create_tts_backend(settings), judge=judge)
    if args.mode == "oneshot":
        client: Any = ChatClient(config)
        summary = run_one_shot(client, workbench, clip_id=args.clip, run_id=args.run_id, repeat=args.repeat,
                               tts_budget=args.tts_budget, speed_policy=args.speed_policy)
    else:
        client = ResponsesClient(config) if args.api == "responses" else ChatClient(config)
        from .playbook import current_playbook, load_playbook

        choice = args.playbook or "current"
        playbook = None if choice == "none" else (
            current_playbook(workbench.store) if choice == "current" else load_playbook(workbench.store, choice))
        system_prompt = build_system_prompt(playbook, max_dead_air_pct=args.max_dead_air)
        print(f"playbook: {playbook['version'] if playbook else 'none'}")
        summary = run_agent(client, workbench, clip_id=args.clip, run_id=args.run_id, repeat=args.repeat,
                            max_turns=args.max_turns, max_total_tokens=args.max_tokens, tts_budget=args.tts_budget,
                            speed_policy=args.speed_policy, max_dead_air_pct=args.max_dead_air,
                            max_pushbacks=args.max_pushbacks, compact_after_tokens=args.compact_after,
                            system_prompt=system_prompt, playbook_version=playbook["version"] if playbook else None)
    print(json.dumps(summary, indent=2))
    scorecard = workbench.store.repeat_dir(args.run_id, args.clip, args.repeat) / "scorecard.json"
    if scorecard.is_file():
        card = json.loads(scorecard.read_text(encoding="utf-8"))
        print(json.dumps({"headline": card["headline"], "gates": {k: v for k, v in card["gates"].items() if v}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
