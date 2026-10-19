"""Deterministic episode scorer for a finished native-dub clip.

Runs OFFLINE over a clip's artifacts (never inside the pipeline) and produces a
scorecard of hard gates plus a soft timing/cost vector -- deliberately not one
scalar, so a mechanism cannot win by gaming a single number (e.g. padding to
beat a timing score).

Everything here is a heuristic with a stated blind spot; ``scorer_calibration``
in review_sheet.py measures each against human labels so the scorer itself
improves. Not covered yet (slots left in the scorecard): Azure STT
intelligibility on the rendered audio, an independent-model fidelity judge.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic_io import read_json
from .experience_store import text_hash

SCORECARD_SCHEMA = "mathula-episode-scorecard-v1"
SHORTFALL_FLAG_PCT = 15.0
GATE_NAMES = (
    "required_literals", "restated_reference", "fabricated_entity", "known_bad_text", "untranslated",
)
# A last-resort compression pass omits protected literals on purpose and flags
# itself for review; reporting those as unwaived gate failures would punish the
# mechanism for doing exactly its documented job.
LITERAL_WAIVER_VARIANTS = frozenset({"protected_content_last_resort"})

_ENGLISH_STOPWORDS = frozenset(
    "the and of to in is that for with on are was this it by as at from be have has not but "
    "would could will which their they there been were".split()
)
_ZULU_PREFIXES = tuple(sorted(
    ("kwa", "nga", "nge", "ngo", "ku", "ka", "wa", "na", "ne", "no", "se", "si", "ba", "be",
     "bo", "za", "ze", "zo", "la", "le", "lo", "a", "e", "i", "o", "u"),
    key=len, reverse=True,
))
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")

# isiZulu writes some COMMON nouns with a capital once a prefix is glued on
# ("uMbuso" = the State, "eNkantolo" = in court, "ngoLwesihlanu" = on Friday),
# so capitalisation alone cannot separate them from a name the model invented.
# These are the cores (prefix stripped, lowercased) that are known not to be
# names: institutions, titles, calendar words and common SA place aliases. This
# is a language-profile list -- extend it when a reviewer marks a false alarm.
_CAPITALISED_COMMON_CORES = frozenset({
    "mbuso", "nkantolo", "masipala", "mongameli", "phalamende", "khomishini", "nobhala",
    "jenene", "ndunankulu", "jikelele", "hlelo", "melika", "afrika", "randi",
    "mvulo", "lwesibili", "lwesithathu", "lwesine", "lwesihlanu", "mgqibelo", "sonto",
    "januwari", "februwari", "mashi", "ephreli", "apreli", "meyi", "juni", "julayi",
    "agasti", "agosti", "septhemba", "okthoba", "novemba", "disemba",
    "goli", "thekwini", "mbombela", "tshwane", "gqeberha", "malahleni", "kapa",
})
# Characters after which a capital is a sentence/quotation start, not a name.
_CAPITAL_STARTS_AFTER = ".?!:“‘\"'(—–-"


def _tokens(text: str) -> list[str]:
    return _WORD.findall(str(text or ""))


def _stem(word: str) -> str:
    stem = word.lower().strip("-'")
    stripped = True
    while stripped and len(stem) > 5:
        stripped = False
        for prefix in _ZULU_PREFIXES:
            if stem.startswith(prefix) and len(stem) - len(prefix) >= 5:
                stem = stem[len(prefix):].lstrip("-'")
                stripped = True
                break
    return stem


def find_restated_references(spoken_text: str, *, window: int = 3) -> list[dict[str, str]]:
    """Two DIFFERENT surface forms of the same stem within ``window`` tokens.

    Catches "kamasipala kuMasipala" (one English mention rendered twice via two
    case-marked forms). Identical adjacent words are not flagged here -- that is
    legitimate emphasis ("kakhulu, kakhulu") and is handled separately. Blind
    spot: a legitimately repeated noun a few words apart ("umsebenzi ...
    nomsebenzi") also matches; it is a review flag, not a verdict.
    """
    tokens = [t for t in _tokens(spoken_text) if len(t) > 5]
    hits: list[dict[str, str]] = []
    for index, word in enumerate(tokens):
        stem = _stem(word)
        if len(stem) < 6:
            continue
        for other in tokens[index + 1:index + 1 + window]:
            if other.lower() != word.lower() and _stem(other) == stem:
                hits.append({
                    "first": word, "second": other, "stem": stem,
                    "name_like": _name_like(word) or _name_like(other),
                })
    return hits


def _name_like(word: str) -> bool:
    return word[:1].isupper() or any(ch.isupper() for ch in word[1:])


def split_restatements(spoken_text: str, source_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(gate hits, soft hits) for restated references.

    A repeated NAME is a defect only when the isiZulu mentions it more often than
    the English did -- "uPakula ... uPakula" is fine when the English also said
    Pakula twice (real: two baseline clips). A repeated COMMON noun cannot be
    aligned to the English without translating, so it is only a review flag.
    """
    source_tokens = [t.lower() for t in _tokens(source_text)]
    spoken_tokens = _tokens(spoken_text)
    gate: list[dict[str, Any]] = []
    soft: list[dict[str, Any]] = []
    for hit in find_restated_references(spoken_text):
        if not hit["name_like"]:
            soft.append(hit)
            continue
        spoken_count = sum(1 for t in spoken_tokens if _stem(t) == hit["stem"])
        source_count = sum(1 for t in source_tokens if t.startswith(hit["stem"][:5]))
        (gate if spoken_count > source_count else soft).append(hit)
    return gate, soft


def normalise_number_format(text: str, *, source: bool) -> str:
    """Zulu groups thousands with a space ("7 000"), English with a comma ("7,000")."""
    if source:
        return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(text or ""))
    return re.sub(r"(?<=\d)[   ](?=\d{3}(?!\d))", "", str(text or ""))


def find_unsupported_doubling(spoken_text: str, source_text: str) -> list[str]:
    """Adjacent identical Zulu words when the English never doubled a word."""
    if re.search(r"\b(\w+)\W+\1\b", str(source_text or ""), flags=re.IGNORECASE):
        return []
    words = [w.lower() for w in _tokens(spoken_text)]
    return sorted({a for a, b in zip(words, words[1:]) if a == b and len(a) > 3})


def _known_entity_tokens(
    source_text: str, glossary: Mapping[str, Any], ledger: Mapping[str, Any],
) -> set[str]:
    known = {t.lower() for t in _tokens(source_text)}
    for term in glossary.get("terms") or []:
        for key in ("english", "zulu"):
            known.update(t.lower() for t in _tokens(str(term.get(key) or "")))
    for item in glossary.get("do_not_translate") or []:
        known.update(t.lower() for t in _tokens(str(item)))
    for entity in ledger.get("entities") or []:
        known.update(t.lower() for t in _tokens(str(entity.get("name") or "")))
    return known


def find_fabricated_entities(spoken_text: str, known_tokens: set[str]) -> list[str]:
    """Capitalised mid-sentence names that appear nowhere in source/glossary/ledger.

    Strips Zulu prefixes glued to a name ("uAndre" -> "Andre", "eMbombela" ->
    "Mbombela") and tolerates localised spellings by matching on a 5-letter
    prefix ("Novemba" ~ "November"). Blind spot: a correct name the ledger never
    captured is reported (that is useful -- it means the ledger missed it).
    """
    found: list[str] = []
    for match in _WORD.finditer(str(spoken_text or "")):
        token = match.group(0)
        upper_at = next((i for i, ch in enumerate(token) if ch.isupper()), None)
        if upper_at is None:
            continue
        if upper_at == 0:
            before = spoken_text[:match.start()].rstrip()
            if not before or before[-1] in _CAPITAL_STARTS_AFTER:
                continue
            if before[-1] in "”’\"'" and len(before) > 1 and before[-2] in ".?!,":
                continue
        core = token[upper_at:].lower()
        if len(core) < 4 or core in known_tokens:
            continue
        if all(part in _CAPITALISED_COMMON_CORES or part in known_tokens for part in core.split("-")):
            continue
        if len(core) >= 5 and any(
            known.startswith(core[:5]) or core.startswith(known[:5])
            for known in known_tokens if len(known) >= 5
        ):
            continue
        found.append(token)
    return found


def english_leakage(spoken_text: str) -> float:
    words = [w.lower() for w in _tokens(spoken_text)]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
    return hits / len(words) if hits >= 2 else 0.0


def _parse_time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _selected(record: Mapping[str, Any]) -> Mapping[str, Any]:
    wanted = record.get("selected_candidate_id")
    for candidate in record.get("candidates") or []:
        if candidate.get("candidate_id") == wanted:
            return candidate
    candidates = record.get("candidates") or [{}]
    return candidates[0]


def _render_by_group(timing_plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in timing_plan.get("groups") or []:
        grouped.setdefault(str(item.get("parent_group_id") or item.get("group_id")), []).append(item)
    result: dict[str, dict[str, Any]] = {}
    for group_id, items in grouped.items():
        result[group_id] = {
            "worst_end_error_ms": max((abs(int(i.get("mouth_close_end_error_ms") or 0)) for i in items), default=0),
            "sync_ok": all(bool(i.get("mouth_close_sync_ok", True)) for i in items),
            "unresolved_early_ms": max((int(i.get("unresolved_early_mouth_close_ms") or 0) for i in items), default=0),
            "raw_speed_percent": max((float(i.get("raw_speed_percent") or 0.0) for i in items), default=0.0),
            "natural_overrun_ms": max((int(i.get("natural_overrun_ms") or 0) for i in items), default=0),
        }
    return result


def normalize_pool_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a candidate-pool record's selected candidate into a scoring item."""
    selected = _selected(record)
    start = record.get("effective_start_ms", record.get("start_ms"))
    end = record.get("effective_source_end_ms", record.get("source_end_ms"))
    return {
        "group_id": str(record.get("group_id")),
        "turn_id": record.get("turn_id"),
        "speaker_id": record.get("speaker_id"),
        "start_ms": int(start or 0),
        "end_ms": int(end or 0),
        "source_text": str(record.get("source_text") or ""),
        "spoken_text": str(selected.get("spoken_text") or ""),
        "measured_ms": int(selected.get("measured_ms") or 0),
        "direction": selected.get("direction"),
        "required_speed_percent": float(selected.get("required_speed_percent") or 0.0),
        "variant_id": str(selected.get("variant_id") or ""),
        "candidate_id": selected.get("candidate_id"),
        "requires_review": bool(record.get("requires_review") or selected.get("requires_review")),
        "morphology_warning": list(selected.get("morphology_warning") or []),
        "elaborated_points": list(selected.get("elaborated_points") or []),
    }


def _score_group(
    item: Mapping[str, Any],
    *,
    known_tokens: set[str],
    labels: Sequence[Mapping[str, Any]],
    rush_flag_pct: float,
    speed_policy: str = "pipeline",
) -> dict[str, Any]:
    from .native_dub import _missing_required_literals

    source_text = str(item.get("source_text") or "")
    spoken = str(item.get("spoken_text") or "")
    variant = str(item.get("variant_id") or "")
    start = int(item.get("start_ms") or 0)
    end = int(item.get("end_ms") or 0)
    window_ms = max(0, end - start)
    measured_ms = int(item.get("measured_ms") or 0)
    shortfall_ms = max(0, window_ms - measured_ms) if item.get("direction") == "expand" else 0
    shortfall_pct = (shortfall_ms / window_ms * 100.0) if window_ms else 0.0
    rush_pct = float(item.get("required_speed_percent") or 0.0)

    gates: dict[str, Any] = {}
    missing = _missing_required_literals(
        source_text=normalise_number_format(source_text, source=True),
        candidate_text=normalise_number_format(spoken, source=False),
    )
    if missing and variant in LITERAL_WAIVER_VARIANTS:
        gates["required_literals"] = {"failed": False, "waived": variant}
    else:
        gates["required_literals"] = {
            "failed": bool(missing),
            "evidence": [f"{m['literal']} x{m['required_count']}" for m in missing],
        }
    restated, soft_restated = split_restatements(spoken, source_text)
    gates["restated_reference"] = {
        "failed": bool(restated), "evidence": [f"{h['first']} / {h['second']}" for h in restated],
    }
    fabricated = find_fabricated_entities(spoken, known_tokens)
    gates["fabricated_entity"] = {"failed": bool(fabricated), "evidence": fabricated}
    spoken_cf = spoken.casefold()
    bad_hits = [
        str(label["bad_substring"]) for label in labels
        if label.get("verdict") == "bad" and label.get("bad_substring")
        and str(label["bad_substring"]).casefold() in spoken_cf
    ]
    gates["known_bad_text"] = {"failed": bool(bad_hits), "evidence": bad_hits}
    normalized = re.sub(r"\W+", " ", spoken).strip().casefold()
    gates["untranslated"] = {
        "failed": not normalized or normalized == re.sub(r"\W+", " ", source_text).strip().casefold(),
    }

    flags: list[str] = []
    if shortfall_pct >= SHORTFALL_FLAG_PCT:
        flags.append("dead_air")
    if speed_policy == "natural":
        # No speed-up exists in this world: a unit that needs ANY speed-up to finish inside its
        # allowance (window + the protected late tolerance) overruns. The pipeline's own `fit`
        # flag is not used here: it forgives up to the renderer's speed-up ceiling.
        if rush_pct > 0:
            flags.append("overrun")
    elif rush_pct >= rush_flag_pct:
        flags.append("rushed")
    if item.get("requires_review"):
        flags.append("needs_review")
    if english_leakage(spoken) >= 0.2:
        flags.append("english_leakage")
    if find_unsupported_doubling(spoken, source_text):
        flags.append("unsupported_doubling")
    if soft_restated:
        flags.append("possible_restatement")

    own_labels = [label for label in labels if label.get("zulu_hash") == text_hash(spoken)]
    return {
        "group_id": str(item.get("group_id")),
        "turn_id": item.get("turn_id"),
        "speaker_id": item.get("speaker_id"),
        "start_ms": start,
        "end_ms": end,
        "window_ms": window_ms,
        "measured_ms": measured_ms,
        "shortfall_ms": shortfall_ms,
        "shortfall_pct": round(shortfall_pct, 2),
        "rush_pct": round(rush_pct, 2),
        "variant": variant,
        "candidate_id": item.get("candidate_id"),
        "requires_review": bool(item.get("requires_review")),
        "morphology_warning": list(item.get("morphology_warning") or []),
        "elaborated_points": list(item.get("elaborated_points") or []),
        "source_text": source_text,
        "spoken_text": spoken,
        "zulu_hash": text_hash(spoken),
        "gates": gates,
        "flags": flags,
        "labels": [dict(label) for label in own_labels],
    }


def _turn_rollup(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_turn: dict[str, list[Mapping[str, Any]]] = {}
    for group in groups:
        by_turn.setdefault(str(group.get("turn_id") or group["group_id"]), []).append(group)
    rollup = []
    for turn_id, members in by_turn.items():
        window = max(m["end_ms"] for m in members) - min(m["start_ms"] for m in members)
        measured = sum(m["measured_ms"] for m in members)
        shortfall = max(0, window - measured) if any(m["shortfall_ms"] for m in members) else 0
        rollup.append({
            "turn_id": turn_id,
            "group_ids": [m["group_id"] for m in members],
            "window_ms": window,
            "measured_ms": measured,
            "shortfall_ms": shortfall,
            "shortfall_pct": round(shortfall / window * 100.0, 2) if window else 0.0,
        })
    return rollup


def _load_context(native: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    glossary_doc = read_json(native / "zulu_terminology_glossary.json") if (native / "zulu_terminology_glossary.json").is_file() else {}
    ledger_doc = read_json(native / "context_ledger.json") if (native / "context_ledger.json").is_file() else {}
    return (
        glossary_doc.get("zulu_terminology_glossary", glossary_doc),
        ledger_doc.get("context_ledger", ledger_doc),
    )


def _render_summary(groups: list[dict[str, Any]], timing_plan: Mapping[str, Any]) -> dict[str, Any]:
    by_group = _render_by_group(timing_plan)
    for group in groups:
        group["render"] = by_group.get(group["group_id"])
    measured = [g["render"] for g in groups if g.get("render")]
    return {
        "available": True,
        "groups_with_render": len(measured),
        "sync_failures": sum(1 for r in measured if not r["sync_ok"]),
        "worst_end_error_ms": max((r["worst_end_error_ms"] for r in measured), default=0),
        "unresolved_early_ms_total": sum(r["unresolved_early_ms"] for r in measured),
        "max_raw_speed_percent": max((r["raw_speed_percent"] for r in measured), default=0.0),
    }


def score_items(
    items: Sequence[Mapping[str, Any]],
    *,
    glossary: Mapping[str, Any],
    ledger: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]] = (),
    usage: Mapping[str, Any] | None = None,
    approach: str = "unknown",
    render: Mapping[str, Any] | None = None,
    timing_plan: Mapping[str, Any] | None = None,
    speed_policy: str = "pipeline",
) -> dict[str, Any]:
    """Score normalized items -- the approach-agnostic core.

    ``speed_policy``: "pipeline" tolerates the renderer's bounded speed-up (flag at the
    natural-speed ceiling); "natural" allows none -- a unit that cannot finish inside its
    allowance at natural speed is an ``overrun``.

    An item is one dubbed unit: window, English source, final Zulu text, and the
    real Azure measurement of that text. Where it came from (the candidate-pool
    pipeline, an agent's own plan) is irrelevant to the score.
    """
    from .native_dub import DEFAULT_MAX_NATURAL_SPEED_PERCENT

    labels_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for label in labels:
        labels_by_group.setdefault(str(label.get("group_id")), []).append(label)
    all_bad_substrings = [label for label in labels if label.get("verdict") == "bad" and label.get("bad_substring")]

    groups = []
    for item in items:
        known = _known_entity_tokens(str(item.get("source_text") or ""), glossary, ledger)
        group_labels = [*labels_by_group.get(str(item.get("group_id")), []), *all_bad_substrings]
        groups.append(_score_group(
            item, known_tokens=known, labels=group_labels, rush_flag_pct=float(DEFAULT_MAX_NATURAL_SPEED_PERCENT),
            speed_policy=speed_policy,
        ))

    render_info: dict[str, Any] = dict(render or {"available": False})
    if timing_plan is not None and not render_info.get("stale"):
        render_info = _render_summary(groups, timing_plan)

    gate_summary = {
        name: sorted(g["group_id"] for g in groups if g["gates"][name]["failed"]) for name in GATE_NAMES
    }
    window_total = sum(g["window_ms"] for g in groups)
    turns = _turn_rollup(groups)
    usage = usage or {}
    headline = {
        "group_count": len(groups),
        "dead_air_pct": round(sum(g["shortfall_ms"] for g in groups) / window_total * 100.0, 2) if window_total else 0.0,
        "worst_shortfall_pct": max((g["shortfall_pct"] for g in groups), default=0.0),
        "groups_dead_air": sum(1 for g in groups if "dead_air" in g["flags"]),
        "worst_turn_shortfall_pct": max((t["shortfall_pct"] for t in turns), default=0.0),
        "mean_rush_pct": round(sum(g["rush_pct"] for g in groups) / len(groups), 2) if groups else 0.0,
        "worst_rush_pct": max((g["rush_pct"] for g in groups), default=0.0),
        "groups_rushed": sum(1 for g in groups if "rushed" in g["flags"] or "overrun" in g["flags"]),
        "speed_policy": speed_policy,
        "review_rate": round(sum(1 for g in groups if g["requires_review"]) / len(groups), 3) if groups else 0.0,
        "gate_failures": sum(len(ids) for ids in gate_summary.values()),
        "tokens_in": int(usage.get("input_tokens") or 0),
        "tokens_out": int(usage.get("output_tokens") or 0),
        "tts_measurements": int(usage.get("tts_measurements") or 0),
    }
    return {
        "schema_version": SCORECARD_SCHEMA,
        "approach": approach,
        "level": "render" if render_info.get("available") else "pool",
        "headline": headline,
        "gates": gate_summary,
        "variant_mix": dict(Counter(g["variant"] for g in groups)),
        "render": render_info,
        "turns": turns,
        "groups": groups,
        "unscored_dimensions": ["stt_intelligibility", "independent_fidelity_judge", "fluency"],
    }


def score_native_dub_dir(
    native_dub_dir: Path, *, labels: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Score a candidate-pool pipeline run (live job or replay) -- the reference policy."""
    native = Path(native_dub_dir)
    pool = read_json(native / "pass2_candidate_pool.json")
    glossary, ledger = _load_context(native)
    render: dict[str, Any] = {"available": False}
    timing_plan = None
    timing_path = native / "timing_plan.json"
    if timing_path.is_file():
        plan = read_json(timing_path)
        pool_time = _parse_time(pool.get("completed_at"))
        render_time = _parse_time(plan.get("completed_at"))
        if pool_time and render_time and pool_time > render_time:
            render = {"available": False, "stale": True, "reason": "candidate pool is newer than the last render"}
        else:
            timing_plan = plan
    return score_items(
        [normalize_pool_record(r) for r in pool.get("candidate_pool") or []],
        glossary=glossary, ledger=ledger, labels=labels, usage=pool.get("usage"),
        approach="candidate_pool_pipeline", render=render, timing_plan=timing_plan,
    )
