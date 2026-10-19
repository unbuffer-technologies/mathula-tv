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
2. Timing: each unit's Zulu is played at its natural speed -- it is NEVER sped up or slowed
   down -- so the words you choose are the only thing that controls timing. The speech should
   fill its source window: too short = dead air while the speaker is still on screen; too long =
   an overrun that spills past its window into the next speaker.

   Silence padding: the renderer closes a shortfall of up to 150 ms with trailing silence, and for
   a speaker turn of several consecutive units it can spread a larger shortfall over the real gaps
   between them (each gap widened by at most 1.5 s; nothing if the units are back to back). That is
   a fallback, not a goal: a viewer hears padding as a pause, so prefer wording that fills the time,
   and use preview_turn_padding to see how much silence a turn would be left with.

   Priority when timing trades off: an overrun is WORSE than dead air. Speech that spills past its
   window collides with the next speaker and breaks sync, while a little silence is only a pause.
   So first make sure no unit overruns (overrun_pct must be 0), and only then shorten the dead air
   that is left. Never accept an overrun to remove a shortfall.

The bar for "done" (checked by the harness against your last submit_dub result, not by your say-so):
no unit overruns, no hard-gate failures, and residual_dead_air_pct at or below
{max_dead_air}. If you stop while a submission is still
below that bar and you have budget left, you will be sent back to the specific units to fix.
3. Naturalness: retell the way a person would say it informally. This is not a word-for-word
   transfer. Use the whole clip's context.

Tools give you real measurements. Use them to check your work rather than guess: Azure TTS
timing is not predictable from word counts, and with no speed control you must get the wording
right. The scorer re-measures your final text with the same
function, so what you measure is what you are graded on. Measurements are budgeted.

When you are satisfied, call submit_dub (it returns your score and which units are flagged; you
may revise and resubmit). Then reply with a short plain-text summary of your approach and stop.
"""

SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.replace("{max_dead_air}", f"{DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT}%")

_UNIT_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id": {"type": "string"},
        "start_ms": {"type": "integer"},
        "end_ms": {"type": "integer"},
        "speaker_id": {"type": "string"},
        "zulu_text": {"type": "string"},
        "source_segment_ids": {"type": "array", "items": {"type": "string"}},
        "source_text": {"type": "string", "description": "Only needed when two units share a source segment."},
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
          "context ledger, glossary and speaker modes. Call this first.",
          {"clip_id": {"type": "string"}}, ["clip_id"]),
    _tool("recall_lessons", "Mistakes native-speaker reviewers marked 'bad' in earlier clips, with corrections "
          "where given. Cross-clip memory.", {"limit": {"type": "integer"}}, []),
    _tool("measure_tts", "Synthesize text with Azure TTS at natural speed and report how it sits in "
          "[start_ms, end_ms]: measured_ms, shortfall_pct (speech missing), tail_pad_ms (trailing silence the "
          "renderer adds itself when the gap is tiny), residual_shortfall_pct (what is left as dead air), "
          "overrun_pct (how far past its allowance it runs), direction ('expand' too short, 'compress' too "
          "long, null fits), fits_naturally. Costs 1 measurement.",
          _MEASURE_PROPS, ["text", "start_ms", "end_ms"]),
    _tool("measure_tts_batch", "Measure up to 12 texts at once; same fields as measure_tts. Costs 1 per text.",
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
          "Returns the official score and flagged units. You may resubmit.",
          {"items": {"type": "array", "items": _UNIT_SCHEMA}, "approach_notes": {"type": "string"}}, ["items"]),
]


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
        for attempt in range(1, self.retries + 2):
            response = requests.post(self.url, headers=headers, json=body, timeout=self.config.timeout_seconds)
            if response.status_code in (429, 500, 502, 503, 504) and attempt <= self.retries:
                self.sleep(min(2 ** attempt, 20))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"model HTTP {response.status_code}: {response.text[:600]}")
            return response.json()
        raise RuntimeError("model call failed")


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
            "usage": {"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0)},
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
        for attempt in range(1, self.retries + 2):
            response = requests.post(self.url, headers=headers, json=body, timeout=self.config.timeout_seconds)
            if response.status_code in (429, 500, 502, 503, 504) and attempt <= self.retries:
                self.sleep(min(2 ** attempt, 20))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"model HTTP {response.status_code}: {response.text[:600]}")
            return self.to_chat_shape(response.json())
        raise RuntimeError("model call failed")


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

    return {
        "open_clip": open_clip,
        "recall_lessons": workbench.recall_lessons,
        "measure_tts": workbench.measure_tts,
        "measure_tts_batch": workbench.measure_tts_batch,
        "check_text": workbench.check_text,
        "preview_turn_padding": workbench.preview_turn_padding,
        "submit_dub": workbench.submit_dub,
    }


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
    if headline.get("groups_rushed", 0) > 0:
        unmet.append(f"{headline['groups_rushed']} unit(s) overrun their window")
    if headline.get("gate_failures", 0) > 0:
        unmet.append(f"{headline['gate_failures']} hard-gate failure(s)")
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
    lines.append(
        f"You still have {tts_left} measurements. Fix these units and resubmit: where speech is short, restore faithful "
        "content from the English you compressed or dropped (never filler); where it overruns, tighten. "
        "Overruns come first. If you truly cannot meet the bar, resubmit your best and say exactly which units and why."
    )
    return " ".join(lines)


def run_agent(
    client: ChatClient, workbench: Workbench, *, clip_id: str, run_id: str, repeat: int = 1,
    max_turns: int = 40, max_total_tokens: int = 500_000, tts_budget: int = 300,
    system_prompt: str = SYSTEM_PROMPT, speed_policy: str = "natural",
    max_dead_air_pct: float = DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT, max_pushbacks: int = DEFAULT_MAX_PUSHBACKS,
    task_message: str | None = None, trace_dir: Path | None = None,
) -> dict[str, Any]:
    directory = Path(trace_dir) if trace_dir else workbench.store.repeat_dir(run_id, clip_id, repeat)
    trace = Trace(directory / "trace.jsonl")
    dispatch = build_dispatch(
        workbench, clip_id=clip_id, run_id=run_id, tts_budget=tts_budget, speed_policy=speed_policy,
        repeat=repeat,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_message or f"Dub clip {clip_id!r}. Start by opening it."},
    ]
    tokens = {"in": 0, "out": 0}
    tool_counts: dict[str, int] = {}
    stop_reason = "max_turns"
    nudged = False
    pushbacks = 0
    unmet: list[str] = []
    turn = 0

    def accepted() -> bool:
        return bool(workbench.episode and workbench.episode.get("submissions"))

    for turn in range(1, max_turns + 1):
        response = client.complete(messages, tools=TOOL_SPECS)
        usage = response.get("usage") or {}
        tokens["in"] += int(usage.get("prompt_tokens") or 0)
        tokens["out"] += int(usage.get("completion_tokens") or 0)
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
        "pushbacks": pushbacks, "unmet_at_end": unmet,
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
    args = parser.parse_args(argv)

    from .cli import create_tts_backend
    from .config import load_settings
    from .experience_store import ExperienceStore
    from .foundry_grok import native_gpt_config_from_environment

    settings = load_settings()
    workbench = Workbench(ExperienceStore(settings.work_dir), lambda: create_tts_backend(settings))
    config = native_gpt_config_from_environment(dict(os.environ))
    if args.mode == "oneshot":
        client: Any = ChatClient(config)
        summary = run_one_shot(client, workbench, clip_id=args.clip, run_id=args.run_id, repeat=args.repeat,
                               tts_budget=args.tts_budget, speed_policy=args.speed_policy)
    else:
        client = ResponsesClient(config) if args.api == "responses" else ChatClient(config)
        summary = run_agent(client, workbench, clip_id=args.clip, run_id=args.run_id, repeat=args.repeat,
                            max_turns=args.max_turns, max_total_tokens=args.max_tokens, tts_budget=args.tts_budget,
                            speed_policy=args.speed_policy, max_dead_air_pct=args.max_dead_air,
                            max_pushbacks=args.max_pushbacks)
    print(json.dumps(summary, indent=2))
    scorecard = workbench.store.repeat_dir(args.run_id, args.clip, args.repeat) / "scorecard.json"
    if scorecard.is_file():
        card = json.loads(scorecard.read_text(encoding="utf-8"))
        print(json.dumps({"headline": card["headline"], "gates": {k: v for k, v in card["gates"].items() if v}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
