"""MCP server: the dubbing agent's workbench.

Exposes real Azure TTS measurement and deterministic isiZulu text checks as
tools, and nothing about HOW to dub. The agent chooses its own method -- one
shot, candidate search, anything -- and hands back a submission, which is
scored by re-measuring through the very same function (``dub_fit``).

Run:  python -m mathula_tv.dub_workbench_mcp      (stdio)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from .atomic_io import atomic_write_json, read_json
from .dub_fit import measure_text_fit
from .dub_submission import validate_submission, write_submission, score_submission_dir
from .episode_scoring import (
    GATE_NAMES, _known_entity_tokens, english_leakage, find_fabricated_entities,
    find_restated_references, find_unsupported_doubling,
)
from .experience_store import ExperienceStore

INSTRUCTIONS = """\
You are dubbing a complete English news clip into isiZulu for broadcast on Mathula TV.

Workflow: list_clips -> open_clip -> (your own method) -> submit_dub. Call recall_lessons
first: it returns mistakes native-speaker reviewers have already marked in earlier clips.

What matters, in the order you will be judged:
1. HARD GATES (a failure is a defect, not a trade-off): every name, number and date in the
   English survives; no name that is not in the English/glossary/ledger; the same place or
   entity is never stated twice for one English mention; no untranslated English.
2. TIMING: each unit's Zulu must occupy its source window. Dead air (finishing well short while
   the speaker is still talking on screen) and audible rush (>10% speed-up needed) both count.
   Measure with measure_tts -- the scorer re-measures your final text with the same function,
   so what you measure is what you are graded on.
3. NATURALNESS: retell the way a person would say it, not a clause-by-clause transfer.

You choose the units, wording and strategy. Cost matters: measure_tts is a real, budgeted
Azure call. A reviewing native speaker will read your units, so keep approach_notes honest.
"""

server = MCPServer("mathula-dub-workbench", instructions=INSTRUCTIONS)

_state: dict[str, Any] = {}


def _store() -> ExperienceStore:
    if "store" not in _state:
        from .config import load_settings

        settings = load_settings()
        _state["settings"] = settings
        _state["store"] = ExperienceStore(settings.work_dir)
    return _state["store"]


def _tts():
    if "tts" not in _state:
        from .cli import create_tts_backend

        _store()
        _state["tts"] = create_tts_backend(_state["settings"])
    return _state["tts"]


def _episode() -> dict[str, Any]:
    episode = _state.get("episode")
    if not episode:
        raise RuntimeError("No clip is open. Call open_clip first.")
    return episode


def _spend(count: int) -> int:
    episode = _episode()
    budget = int(episode["tts_budget"])
    if episode["tts_used"] + count > budget:
        raise RuntimeError(
            f"TTS measurement budget exhausted ({episode['tts_used']}/{budget}). "
            "Submit with what you have, or call submit_dub to see the current score."
        )
    episode["tts_used"] += count
    atomic_write_json(Path(episode["dir"]) / "episode.json", {
        k: v for k, v in episode.items() if k != "native"
    })
    return budget - episode["tts_used"]


@server.tool()
def list_clips() -> dict[str, Any]:
    """List the frozen clips available to dub (id, duration, notes)."""
    corpus = _store().load_corpus()["clips"]
    return {"clips": [{"clip_id": cid, **{k: v[k] for k in ("duration_s", "segment_count", "notes")}} for cid, v in corpus.items()]}


@server.tool()
def open_clip(clip_id: str, run_id: str, repeat: int = 1) -> dict[str, Any]:
    """Start an episode on a clip. Returns every Pass-1 segment (id, speaker, window, English) plus
    the whole-clip context ledger, glossary and speaker modes. Re-opening the same
    (run_id, clip_id, repeat) continues its measurement budget."""
    store = _store()
    if clip_id not in store.load_corpus()["clips"]:
        raise ValueError(f"unknown clip {clip_id!r}; call list_clips")
    native = store.clip_dir(clip_id) / "native_dub"
    directory = store.repeat_dir(run_id, clip_id, repeat)
    directory.mkdir(parents=True, exist_ok=True)
    previous = read_json(directory / "episode.json") if (directory / "episode.json").is_file() else {}
    budget = int(os.getenv("MATHULA_DUB_TTS_BUDGET", "800"))
    _state["episode"] = {
        "clip_id": clip_id, "run_id": run_id, "repeat": int(repeat), "dir": str(directory),
        "native": str(native), "tts_budget": budget, "tts_used": int(previous.get("tts_used", 0)),
        "submissions": int(previous.get("submissions", 0)),
    }
    if not store.read_run(run_id):
        store.write_run(run_id, parameters={"approach": "agent", "tts_budget": budget})
    pass1 = read_json(native / "pass1_restored_transcript.json")
    glossary_doc = read_json(native / "zulu_terminology_glossary.json")
    ledger_doc = read_json(native / "context_ledger.json")
    modes = read_json(native / "speaker_seriousness_modes.json").get("modes", {}) if (native / "speaker_seriousness_modes.json").is_file() else {}
    return {
        "clip_id": clip_id,
        "segments": [
            {"segment_id": s["segment_id"], "speaker_id": s.get("speaker_id"), "start_ms": s["start_ms"],
             "end_ms": s["end_ms"], "english": s.get("restored_text") or s.get("source_text")}
            for s in pass1["segments"]
        ],
        "context_ledger": ledger_doc.get("context_ledger", ledger_doc),
        "glossary": glossary_doc.get("zulu_terminology_glossary", glossary_doc),
        "speaker_modes": modes,
        "tts_budget_remaining": budget - _state["episode"]["tts_used"],
        "voice": getattr(_tts(), "default_voice", None),
    }


@server.tool()
def measure_tts(
    text: str, start_ms: int, end_ms: int, next_start_ms: int | None = None,
    speaker_id: str = "SPEAKER_00", source_text: str = "",
) -> dict[str, Any]:
    """Synthesize `text` with Azure and report how it sits in the window [start_ms, end_ms].
    `next_start_ms` is when the next speech starts (bounds how much trailing gap may be borrowed).
    Returns measured_ms, shortfall_pct (dead air), required_speed_percent (rush), direction
    ('expand' = too short, 'compress' = too long, null = fits), fits_naturally."""
    remaining = _spend(1)
    result = measure_text_fit(
        _tts(), text=text, start_ms=start_ms, end_ms=end_ms, next_start_ms=next_start_ms,
        speaker_id=speaker_id, source_text=source_text, output_root=Path(_episode()["dir"]) / "measure",
    )
    result.pop("audio_path", None)
    result["tts_budget_remaining"] = remaining
    return result


@server.tool()
def measure_tts_batch(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure up to 12 texts at once (each: text, start_ms, end_ms, optional next_start_ms,
    speaker_id, source_text). Measurements run in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    if not 1 <= len(requests) <= 12:
        raise ValueError("send between 1 and 12 requests")
    remaining = _spend(len(requests))
    root = Path(_episode()["dir"]) / "measure"
    tts = _tts()

    def _one(request: dict[str, Any]) -> dict[str, Any]:
        out = measure_text_fit(
            tts, text=request["text"], start_ms=request["start_ms"], end_ms=request["end_ms"],
            next_start_ms=request.get("next_start_ms"), speaker_id=request.get("speaker_id", "SPEAKER_00"),
            source_text=request.get("source_text", ""), output_root=root,
        )
        out.pop("audio_path", None)
        return out

    with ThreadPoolExecutor(max_workers=min(6, len(requests))) as pool:
        results = list(pool.map(_one, requests))
    return {"results": results, "tts_budget_remaining": remaining}


@server.tool()
def check_text(zulu_text: str, source_text: str) -> dict[str, Any]:
    """Free, deterministic isiZulu checks on one unit (no Azure call): required literals, the same
    reference restated twice, names not in the source/glossary/ledger, unsupported doubled words,
    English leakage. A failure means a native reviewer is likely to mark this unit bad."""
    from .native_dub import _missing_required_literals

    episode = _episode()
    native = Path(episode["native"])
    glossary_doc = read_json(native / "zulu_terminology_glossary.json")
    ledger_doc = read_json(native / "context_ledger.json")
    known = _known_entity_tokens(
        source_text, glossary_doc.get("zulu_terminology_glossary", glossary_doc),
        ledger_doc.get("context_ledger", ledger_doc),
    )
    missing = _missing_required_literals(source_text=source_text, candidate_text=zulu_text)
    restated = find_restated_references(zulu_text)
    fabricated = find_fabricated_entities(zulu_text, known)
    problems = []
    if missing:
        problems.append({"gate": "required_literals", "evidence": [f"{m['literal']} x{m['required_count']}" for m in missing]})
    if restated:
        problems.append({"gate": "restated_reference", "evidence": [f"{h['first']} / {h['second']}" for h in restated]})
    if fabricated:
        problems.append({"gate": "fabricated_entity", "evidence": fabricated})
    doubled = find_unsupported_doubling(zulu_text, source_text)
    if doubled:
        problems.append({"flag": "unsupported_doubling", "evidence": doubled})
    if english_leakage(zulu_text) >= 0.2:
        problems.append({"flag": "english_leakage"})
    return {"ok": not any("gate" in p for p in problems), "problems": problems, "gates_checked": list(GATE_NAMES)}


@server.tool()
def recall_lessons(limit: int = 25) -> dict[str, Any]:
    """Mistakes native-speaker reviewers have marked 'bad' in earlier clips (English, the bad Zulu,
    tags, note). Read this before dubbing; it is the only cross-clip memory."""
    store = _store()
    lessons = []
    for clip_id in store.clip_ids():
        for label in store.load_labels(clip_id):
            if label.get("verdict") == "bad":
                lessons.append({k: label.get(k) for k in ("clip_id", "tags", "english", "zulu", "bad_substring", "corrected_zulu", "note")})
    lessons.sort(key=lambda item: item.get("clip_id") or "")
    return {"count": len(lessons), "lessons": lessons[-int(limit):]}


@server.tool()
def submit_dub(items: list[dict[str, Any]], approach_notes: str = "") -> dict[str, Any]:
    """Submit the dub. items: ordered, non-overlapping units, each
    {item_id, start_ms, end_ms, speaker_id, zulu_text, source_segment_ids:[...],
     optional source_text (required only if two items share a segment), optional turn_id}.
    Every Pass-1 segment must be covered. Returns the score; you may resubmit to improve it."""
    episode = _episode()
    native = Path(episode["native"])
    segments = read_json(native / "pass1_restored_transcript.json")["segments"]
    submission = {"clip_id": episode["clip_id"], "run_id": episode["run_id"], "repeat": episode["repeat"],
                  "approach_notes": approach_notes, "items": items}
    errors, warnings = validate_submission(submission, segments)
    if errors:
        return {"accepted": False, "errors": errors, "warnings": warnings}
    directory = Path(episode["dir"])
    write_submission(directory / "submission.json", submission)
    episode["submissions"] += 1
    card = score_submission_dir(
        native, submission, tts=_tts(), measure_root=directory / "measure",
        labels=_store().load_labels(episode["clip_id"]),
        usage={"tts_measurements": episode["tts_used"]},
    )
    atomic_write_json(directory / "scorecard.json", card)
    return {
        "accepted": True, "warnings": warnings, "headline": card["headline"], "gates": card["gates"],
        "flagged_units": [
            {"item_id": g["group_id"], "gates": [n for n, v in g["gates"].items() if v["failed"]], "flags": g["flags"]}
            for g in card["groups"] if g["flags"] or any(v["failed"] for v in g["gates"].values())
        ],
    }


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
