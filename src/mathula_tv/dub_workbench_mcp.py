"""MCP server: the dubbing agent's workbench, for any MCP-capable agent.

A transport wrapper over ``dub_workbench.Workbench`` (the same methods the
hand-written agent in ``dub_agent`` calls directly). It exposes real Azure TTS
measurement and deterministic isiZulu text checks as tools, and nothing about
HOW to dub: the agent picks its own method, then submits units, which are scored
by re-measuring through the same function.

Run:  python -m mathula_tv.dub_workbench_mcp      (stdio)
"""
from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from .dub_workbench import Workbench
from .experience_store import ExperienceStore

INSTRUCTIONS = """\
You are dubbing a complete English news clip into isiZulu for broadcast on Mathula TV.

Workflow: list_clips -> open_clip -> (your own method) -> submit_dub. Call recall_lessons
first: it returns mistakes native-speaker reviewers have already marked in earlier clips.

What matters, in the order you will be judged:
1. HARD GATES (a failure is a defect, not a trade-off): every name, number and date in the
   English survives; no name that is not in the English/glossary/ledger; the same place or
   entity is never stated twice for one English mention; no untranslated English.
2. TIMING: speech is played at natural speed and is never sped up, so wording alone controls
   timing. The renderer pads small shortfalls with silence (<=150 ms tail pad; larger ones are spread
   over a turn's real inter-sentence gaps, <=1.5 s each) -- a fallback, so prefer wording; see
   preview_turn_padding. Each unit's Zulu must occupy its source window: dead air (finishing well short while
   the speaker is still talking on screen) and overrun (spilling past the window) both count.
   Measure with measure_tts -- the scorer re-measures your final text with the same function,
   so what you measure is what you are graded on.
3. NATURALNESS: retell the way a person would say it, not a clause-by-clause transfer.

You choose the units, wording and strategy. Cost matters: measure_tts is a real, budgeted
Azure call. A reviewing native speaker will read your units, so keep approach_notes honest.
"""

server = MCPServer("mathula-dub-workbench", instructions=INSTRUCTIONS)

_state: dict[str, Workbench] = {}


def _workbench() -> Workbench:
    if "wb" not in _state:
        from .cli import create_tts_backend
        from .config import load_settings

        settings = load_settings()
        _state["wb"] = Workbench(ExperienceStore(settings.work_dir), lambda: create_tts_backend(settings))
    return _state["wb"]


@server.tool()
def list_clips() -> dict[str, Any]:
    """List the frozen clips available to dub (id, duration, notes)."""
    return _workbench().list_clips()


@server.tool()
def open_clip(clip_id: str, run_id: str, repeat: int = 1) -> dict[str, Any]:
    """Start an episode on a clip. Returns every Pass-1 segment (id, speaker, window, English) plus
    the whole-clip context ledger, glossary and speaker modes. Re-opening the same
    (run_id, clip_id, repeat) continues its measurement budget."""
    return _workbench().open_clip(clip_id, run_id, repeat)


@server.tool()
def measure_tts(
    text: str, start_ms: int, end_ms: int, next_start_ms: int | None = None,
    speaker_id: str = "SPEAKER_00", source_text: str = "",
) -> dict[str, Any]:
    """Synthesize `text` with Azure and report how it sits in the window [start_ms, end_ms].
    `next_start_ms` is when the next speech starts (bounds how much trailing gap may be borrowed).
    Returns measured_ms, shortfall_pct (speech missing), tail_pad_ms (trailing silence the renderer adds
    when the gap is tiny), residual_shortfall_pct (left as dead air), overrun_pct (how far past its allowance it runs),
    direction ('expand' = too short, 'compress' = too long, null = fits), fits_naturally."""
    return _workbench().measure_tts(text, start_ms, end_ms, next_start_ms, speaker_id, source_text)


@server.tool()
def measure_tts_batch(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure up to 40 texts at once (each: text, start_ms, end_ms, optional next_start_ms,
    speaker_id, source_text). Measurements run in parallel."""
    return _workbench().measure_tts_batch(requests)


@server.tool()
def preview_turn_padding(members: list[dict[str, Any]], last_free_gap_ms: int = 0) -> dict[str, Any]:
    """Free arithmetic (no TTS). If a speaker turn of these units finishes early, how much silence does
    the renderer add, and how much is left as dead air? Mechanisms in order: a tail pad of up to 150 ms
    when it fits the free gap, else the shortfall is spread over the turn's real inter-sentence gaps
    (each widened at most 1500 ms). members: {start_ms, end_ms, measured_ms}."""
    return _workbench().preview_turn_padding(members, last_free_gap_ms)


@server.tool()
def check_text(zulu_text: str, source_text: str) -> dict[str, Any]:
    """Free, deterministic isiZulu checks on one unit (no Azure call): required literals, the same
    reference restated twice, names not in the source/glossary/ledger, unsupported doubled words,
    English leakage. A failure means a native reviewer is likely to mark this unit bad."""
    return _workbench().check_text(zulu_text, source_text)


@server.tool()
def recall_lessons(limit: int = 25) -> dict[str, Any]:
    """Mistakes native-speaker reviewers have marked 'bad' in earlier clips (English, the bad Zulu,
    tags, note). Read this before dubbing; it is the only cross-clip memory."""
    return _workbench().recall_lessons(limit)


@server.tool()
def get_submission(unit_ids: list[str] | None = None) -> dict[str, Any]:
    """The units currently accepted (all, or only unit_ids), with text and any flags from the last score."""
    return _workbench().get_submission(unit_ids)


@server.tool()
def submit_dub(items: list[dict[str, Any]], approach_notes: str = "", partial: bool = False,
               remove: list[str] | None = None) -> dict[str, Any]:
    """Submit the dub. items: ordered, non-overlapping units, each
    {item_id, start_ms, end_ms, speaker_id, zulu_text, source_segment_ids:[...],
     optional source_text (required only if two items share a segment), optional turn_id}.
    Every Pass-1 segment must be covered. Returns the score; you may resubmit to improve it.
    With partial=true send only the units you changed: each replaces the current unit with the same item_id, ids in
    `remove` are deleted, and the rest carry over (needs a previous accepted submission)."""
    return _workbench().submit_dub(items, approach_notes, partial, remove)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
