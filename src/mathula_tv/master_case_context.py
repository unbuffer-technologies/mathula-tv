"""Case Memory helpers for working/case_state/master_case_state.json -- the
Madlanga Commission "Story Memory" built by scripts/sync_madlanga_case.py
(migrated from the sibling commission-ai project 2026-08-29, unchanged logic).

Keeps entity names, aliases, standing threads, and open questions consistent
across the translation/QA pipeline in native_dub.py, in the same spirit as
this file's own context_ledger/zulu_terminology_glossary. Real motivation:
native_phrase_0113's real "Witness G told him to meet him at Orlando SAPS"
attribution ambiguity in this session could not be resolved from the
transcript alone -- but master_case_state.json's own standing_threads (a
"Confidential Intelligence Witness G" thread, with named associated people)
is exactly the kind of case-external context that could close gaps like this.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CATEGORIES = ("people", "organizations", "companies", "locations")
NAME_FIELDS = ("canonical", "canonical_name", "name", "display_name", "full_name", "title")
ALIAS_FIELDS = ("aliases", "known_as", "title_variants", "variants", "alternate_names", "aka")


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def default_case_state_path(workdir: Path | str) -> Path | None:
    workdir = Path(workdir)
    candidates = [
        workdir / "master_case_state.json",
        workdir / "case_state" / "master_case_state.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_master_case_state(workdir: Path | str, case_state: Path | str | None = None) -> dict[str, Any]:
    path = Path(case_state) if case_state else default_case_state_path(workdir)
    data = load_json(path, {}) if path else {}
    return data if isinstance(data, dict) else {}


def _alias_values(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            out.extend(_alias_values(item))
    elif isinstance(value, dict):
        for field in NAME_FIELDS:
            if value.get(field):
                out.append(str(value[field]))
    return [clean_text(item) for item in out if clean_text(item)]


def _parse_entity_item(item: Any) -> tuple[str, list[str]] | None:
    if isinstance(item, str):
        return clean_text(item), []
    if not isinstance(item, dict):
        return None
    canonical = ""
    for field in NAME_FIELDS:
        if item.get(field):
            canonical = clean_text(item[field])
            break
    if not canonical:
        return None
    aliases: list[str] = []
    for field in ALIAS_FIELDS:
        aliases.extend(_alias_values(item.get(field)))
    return canonical, aliases


def _add_entity(registry: dict[str, dict[str, set[str]]], category: str, canonical: Any, aliases: list[Any] | None = None) -> None:
    canonical_text = clean_text(canonical)
    if not canonical_text:
        return
    registry.setdefault(category, {})
    registry[category].setdefault(canonical_text, set()).add(canonical_text)
    for alias in aliases or []:
        alias_text = clean_text(alias)
        if alias_text:
            registry[category][canonical_text].add(alias_text)


def extract_case_registry(case_state: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    registry_sets: dict[str, dict[str, set[str]]] = {category: {} for category in CATEGORIES}
    if not isinstance(case_state, dict):
        return {category: {} for category in CATEGORIES}

    def add_category(category: str, value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                parsed = _parse_entity_item(item)
                if parsed:
                    _add_entity(registry_sets, category, parsed[0], parsed[1])
        elif isinstance(value, dict):
            for key, item in value.items():
                parsed = _parse_entity_item(item)
                if parsed:
                    _add_entity(registry_sets, category, parsed[0], [key, *parsed[1]])
                elif isinstance(item, str):
                    _add_entity(registry_sets, category, item, [key])
                else:
                    _add_entity(registry_sets, category, key)

    entities = case_state.get("entities")
    if isinstance(entities, dict):
        for category in CATEGORIES:
            add_category(category, entities.get(category))

    for category in CATEGORIES:
        add_category(category, case_state.get(category))

    aliases = case_state.get("aliases")
    if isinstance(aliases, dict):
        for category in CATEGORIES:
            value = aliases.get(category)
            if isinstance(value, dict):
                for alias, canonical in value.items():
                    _add_entity(registry_sets, category, canonical, [alias])

    for thread in case_state.get("standing_threads", []) or []:
        if not isinstance(thread, dict):
            continue
        for category in CATEGORIES:
            add_category(category, thread.get(category))

    return {
        category: {
            canonical: sorted(aliases)
            for canonical, aliases in entries.items()
        }
        for category, entries in registry_sets.items()
    }


def alias_lookup_from_registry(registry: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, str]]:
    lookup = {category: {} for category in CATEGORIES}
    for category, entries in registry.items():
        for canonical, aliases in entries.items():
            for alias in aliases:
                key = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
                if key:
                    lookup[category][key] = canonical
    return lookup


def extract_case_threads(case_state: dict[str, Any]) -> list[dict[str, Any]]:
    threads = case_state.get("standing_threads", [])
    return threads if isinstance(threads, list) else []


def extract_case_relationships(case_state: dict[str, Any]) -> list[dict[str, Any]]:
    relationships = case_state.get("relationships", [])
    return relationships if isinstance(relationships, list) else []


def extract_case_timeline(case_state: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = case_state.get("timeline_events", [])
    return timeline if isinstance(timeline, list) else []


def extract_case_open_questions(case_state: dict[str, Any]) -> list[Any]:
    questions = case_state.get("open_questions", [])
    return questions if isinstance(questions, list) else []


def compact_case_context(case_state: dict[str, Any], max_chars: int = 12000) -> dict[str, Any]:
    registry = extract_case_registry(case_state)
    compact = {
        "case_key": case_state.get("case_key"),
        "trusted_entities": {
            category: [
                {"canonical": canonical, "aliases": aliases[:8]}
                for canonical, aliases in list(entries.items())[:80]
            ]
            for category, entries in registry.items()
        },
        "standing_threads": [
            {
                "thread_id": thread.get("thread_id"),
                "title": thread.get("title"),
                "status": thread.get("status"),
                "why_it_matters": thread.get("why_it_matters"),
                "people": thread.get("people", [])[:8] if isinstance(thread.get("people"), list) else [],
            }
            for thread in extract_case_threads(case_state)[:30]
            if isinstance(thread, dict)
        ],
        "important_relationships": extract_case_relationships(case_state)[:40],
        "timeline_events": extract_case_timeline(case_state)[:40],
        "open_questions": extract_case_open_questions(case_state)[:30],
        "contradictions": case_state.get("contradictions", [])[:30] if isinstance(case_state.get("contradictions"), list) else [],
        "source_refs": case_state.get("source_refs", [])[:30] if isinstance(case_state.get("source_refs"), list) else [],
    }
    raw = json.dumps(compact, ensure_ascii=False)
    if len(raw) <= max_chars:
        return compact
    compact["standing_threads"] = compact["standing_threads"][:12]
    compact["important_relationships"] = compact["important_relationships"][:12]
    compact["timeline_events"] = compact["timeline_events"][:12]
    compact["open_questions"] = compact["open_questions"][:12]
    compact["contradictions"] = compact["contradictions"][:8]
    compact["source_refs"] = compact["source_refs"][:8]
    raw = json.dumps(compact, ensure_ascii=False)
    if len(raw) <= max_chars:
        return compact
    compact["trusted_entities"] = {
        category: values[:35]
        for category, values in compact["trusted_entities"].items()
    }
    return compact
