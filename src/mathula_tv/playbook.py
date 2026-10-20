"""The dubbing agent's playbook: strategy it learns from experience, kept apart from the harness's rules.

What belongs here is *how to work inside the rules*: "submit a rough first draft early", "batch
measurements". What does not belong here is the rules themselves -- the bar, the budgets, the
scorer, the timing lock -- which live in code and the system prompt and which no lesson may
relax. The reflector (``dub_reflect``) proposes small itemised edits from run evidence; the
curator below applies them mechanically, keeps the playbook small, and refuses anything that is
clip-specific or tries to touch the harness. A candidate playbook is only adopted after a trial
(``dub_improve``) shows it is no worse.

Playbooks are immutable, content-addressed versions stored under ``<experience>/playbooks``;
``current.json`` points at the adopted one, so any run can be reproduced with the playbook it used.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic_io import atomic_write_json, read_json

PLAYBOOK_SCHEMA = "mathula-dub-playbook-v1"
MAX_ENTRIES = 12
MAX_ENTRY_CHARS = 240
MAX_EDITS_PER_STEP = 3
MAX_EVIDENCE_CHARS = 220

# Strategy the agent started with. Entry p01 is deliberately the wording that produced a real failure
# (an agent that withheld its first submission until it fully fit and then gave up) -- it is the
# test of whether the reflector can find and correct a bad lesson from evidence.
SEED_ENTRIES: tuple[dict[str, str], ...] = (
    {"id": "p01", "text": "Work toward a first submission that fits every unit's time frame, then improve wording only in ways you have measured still fit."},
    {"id": "p02", "text": "Measure candidates in batches of up to 12 per call rather than one at a time."},
    {"id": "p03", "text": "Run check_text only on near-final text; it is for catching gate problems, not for drafting."},
    {"id": "p04", "text": "Submit only after changing something that should improve the score."},
)

# Things a lesson must not contain: clip-specific identifiers (overfitting to one clip) and attempts
# to talk the agent out of the harness's own rules.
_CLIP_SPECIFIC = re.compile(
    r"\b(seg-\d+|dub-\d+|item-\d+|native_phrase_\d+|t\d{2}|u\d{1,3})\b", re.IGNORECASE,
)
_RULE_OVERRIDE = re.compile(
    r"\b(ignore|skip|bypass|disable|override|circumvent)\b[^.]{0,40}\b(bar|harness|ratchet|budget|lock|scorer|gate|rule)s?\b",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def version_id(entries: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps([[e["id"], e["text"]] for e in entries], ensure_ascii=False)
    return "pb-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def new_playbook(
    entries: Sequence[Mapping[str, Any]], *, parent: str | None = None, note: str = "",
) -> dict[str, Any]:
    cleaned = [
        {"id": e["id"], "text": e["text"], "evidence": list(e.get("evidence") or [])[:3]} for e in entries
    ]
    return {
        "schema_version": PLAYBOOK_SCHEMA, "version": version_id(cleaned), "parent": parent,
        "created_at": _now(), "note": note, "entries": cleaned,
    }


def seed_playbook() -> dict[str, Any]:
    return new_playbook(SEED_ENTRIES, note="seed: initial strategy notes")


def render_section(playbook: Mapping[str, Any] | None) -> str:
    """The block appended to the system prompt."""
    entries = (playbook or {}).get("entries") or []
    if not entries:
        return ""
    lines = "\n".join(f"- {e['text']}" for e in entries)
    return (
        "\n\nPLAYBOOK -- strategy notes learned from earlier runs. They are advice about how to work, "
        "not rules: follow them unless what you are measuring says otherwise. The rules above always win.\n"
        f"{lines}\n"
    )


def lint_text(text: str) -> list[str]:
    problems = []
    stripped = str(text or "").strip()
    if not stripped:
        problems.append("empty")
    if len(stripped) > MAX_ENTRY_CHARS:
        problems.append(f"longer than {MAX_ENTRY_CHARS} characters")
    if _CLIP_SPECIFIC.search(stripped):
        problems.append("mentions a clip-specific id (a lesson must hold for any clip)")
    if _RULE_OVERRIDE.search(stripped):
        problems.append("tries to relax a harness rule (rules are not up for revision)")
    return problems


def apply_edits(
    playbook: Mapping[str, Any], edits: Sequence[Mapping[str, Any]], *, note: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply reflector edits mechanically. Returns (new playbook, applied, rejected-with-reason).

    Each edit is {op: add|revise|remove, id?, text?, evidence}. At most ``MAX_EDITS_PER_STEP`` are
    considered per step; every edit must cite evidence; the playbook never exceeds ``MAX_ENTRIES``.
    """
    entries = [dict(e) for e in playbook["entries"]]
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    next_number = 1 + max((int(re.sub(r"\D", "", e["id"]) or 0) for e in entries), default=0)

    for position, edit in enumerate(edits):
        op = str(edit.get("op") or "").lower()
        evidence = str(edit.get("evidence") or "").strip()[:MAX_EVIDENCE_CHARS]
        text = str(edit.get("text") or "").strip()

        def reject(reason: str) -> None:
            rejected.append({"edit": dict(edit), "reason": reason})

        if position >= MAX_EDITS_PER_STEP:
            reject(f"more than {MAX_EDITS_PER_STEP} edits in one step")
            continue
        if op not in ("add", "revise", "remove"):
            reject("unknown op")
            continue
        if not evidence:
            reject("no evidence cited")
            continue
        target = next((e for e in entries if e["id"] == edit.get("id")), None)
        if op in ("revise", "remove") and target is None:
            reject(f"no entry {edit.get('id')!r}")
            continue
        if op in ("add", "revise"):
            problems = lint_text(text)
            if problems:
                reject("; ".join(problems))
                continue
            if any(e["text"].strip().casefold() == text.casefold() for e in entries):
                reject("duplicate of an existing entry")
                continue
        if op == "add":
            if len(entries) >= MAX_ENTRIES:
                reject(f"playbook is full ({MAX_ENTRIES}); revise or remove an entry first")
                continue
            entry = {"id": f"p{next_number:02d}", "text": text, "evidence": [evidence]}
            next_number += 1
            entries.append(entry)
            applied.append({"op": "add", "id": entry["id"], "text": text, "evidence": evidence})
        elif op == "revise":
            target["text"] = text
            target["evidence"] = [*(target.get("evidence") or []), evidence][-3:]
            applied.append({"op": "revise", "id": target["id"], "text": text, "evidence": evidence})
        else:
            entries.remove(target)
            applied.append({"op": "remove", "id": target["id"], "evidence": evidence})

    return new_playbook(entries, parent=playbook.get("version"), note=note), applied, rejected


# ---- storage ---------------------------------------------------------------------------------------------

def playbook_dir(store: Any) -> Path:
    return Path(store.root) / "playbooks"


def save_playbook(store: Any, playbook: Mapping[str, Any]) -> Path:
    path = playbook_dir(store) / f"{playbook['version']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, dict(playbook))
    return path


def load_playbook(store: Any, version: str) -> dict[str, Any]:
    path = playbook_dir(store) / f"{version}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no playbook {version!r} in {playbook_dir(store)}")
    return read_json(path)


def set_current(store: Any, version: str, *, reason: str = "") -> None:
    atomic_write_json(playbook_dir(store) / "current.json", {"version": version, "set_at": _now(), "reason": reason})


def current_playbook(store: Any) -> dict[str, Any]:
    """The adopted playbook; the seed (saved on first use) if none has been adopted yet."""
    pointer = playbook_dir(store) / "current.json"
    if pointer.is_file():
        return load_playbook(store, read_json(pointer)["version"])
    seed = seed_playbook()
    save_playbook(store, seed)
    set_current(store, seed["version"], reason="seed")
    return seed


def append_history(store: Any, record: Mapping[str, Any]) -> None:
    path = playbook_dir(store) / "history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), **dict(record)}, ensure_ascii=False) + "\n")
