"""Post-Pass-2 referent/entity fidelity audit and bounded repair.

Pass 2's inline "self-QA" instructions run in the same call, on the same
reasoning trace, as the translation itself. That makes them structurally
unable to catch a referent substitution the model is confident about --
for example rendering "close to the minister" as "close to God"
(``ngqongqoshe`` -> ``uNkulunkulu``). A model that made that mistake will
tend to confirm it when asked to grade itself in the same breath.

This module runs a second, independent, narrow-scope call per translated
group: given only the English source and the current Zulu text (no
translation reasoning, no context ledger), verify that every name, role or
title, number, date, and negation in the English is represented in the
Zulu. Only groups it flags go on to a bounded repair call that receives the
specific finding and returns a corrected translation. Everything else is
left untouched.

A free, deterministic tier runs first: named people/organisations/places
already tracked in ``config/entity_registry.json`` are checked for bare
presence in the Zulu text with no AI call at all. That tier cannot catch a
role-noun substitution like the minister/God example above -- "minister" is
not a registered named entity -- which is exactly why the independent audit
call still runs on every group regardless of the deterministic result.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from .entity_registry import EntityRegistry
from .foundry_grok import FoundryGrokProvider, GrokOutputTruncated

REFERENT_AUDIT_SCHEMA_VERSION = "mathula-native-referent-audit-v3-checked-terms-authoritative"
DEFAULT_REFERENT_AUDIT_BATCH_SIZE = 12
DEFAULT_MAX_REFERENT_REPAIR_ROUNDS = 1
# Unlike Pass 2's temporal-mask windows (which need the previous window's own real
# output for continuity), every audit batch here is fully independent -- it only reads
# the English/Zulu pair already committed for its own groups, never another batch's
# result. Real production data: 152 sentences at the default batch size of 12 made 13
# sequential audit calls averaging ~27s each (~6 minutes) with zero cross-batch
# dependency to justify that serialization. Concurrency is bounded, not unlimited, to
# stay a good citizen of the same rate-limited Foundry endpoint Pass 2 also hits.
DEFAULT_REFERENT_AUDIT_WORKERS = 4

# Only these entity types are known, empirically, to survive translation as the
# same literal English/transliterated string in this corpus (e.g. "Fadiel Adams",
# "Ayanda Nyati"). Organisations, places, and cases are properly translated into a
# natural Zulu equivalent (e.g. "the Commission" -> "iKhomishini"), so a
# literal-substring check against their English canonical text/aliases produces
# false positives, not real findings -- confirmed in production: every one of the
# first 5 real audits this tier ever flagged was "the Commission" missing from
# correctly-translated Zulu text that actually said "iKhomishini". Referent
# fidelity for translated categories is left entirely to the independent AI audit
# call, which can judge equivalence instead of literal string matching.
_LITERALLY_PRESERVED_ENTITY_TYPES = {"person", "anonymous_witness"}

_AUDIT_SYSTEM_PROMPT = """You are a bilingual English/isiZulu fidelity auditor for a courtroom \
commission-of-inquiry dub. You do not translate, improve style, or shorten anything.

A single holistic "does this look right" read is not reliable enough for this job: a swapped \
role or name reads just as fluently as a correct one, so skimming the Zulu for overall sense \
will not catch it. Work each group in two separate, strict steps and do not let the second step \
influence how thoroughly you do the first.

STEP 1 -- Extract (read ONLY the English text for this step; do not look at the Zulu yet). List \
every one of the following the English text contains, in the order they appear, into \
`checked_terms`. Do not skip any to save time, even if a group has many:
- every named person, organisation, place, or case;
- every role or title word (minister, commissioner, general, advocate, judge, president, \
governor, colonel, sergeant, and similar -- including when used alone, e.g. "the minister");
- every number, date, and docket/reference figure;
- every negation ("not", "didn't", "cannot", "never", and similar).
If a group genuinely contains none of these, `checked_terms` is an empty list -- do not invent one.

STEP 2 -- Verify (now read the Zulu). For every single entry you extracted in step 1, examine \
the Zulu text specifically for that term and record `verification` as exactly one of:
- "present": the term is correctly represented (a name kept as-is, a role translated to its \
correct Zulu equivalent, a number/date unchanged, a negation preserved) -- word order, phrasing, \
and paraphrase do not matter, only whether the referent itself is right;
- "missing": the term is not represented anywhere in the Zulu at all;
- "substituted": the Zulu says something in that position, but it is a DIFFERENT referent than \
the English -- a different name, an unrelated role/word, a changed number, or a flipped negation. \
Record what appears instead in `found_instead`.
Verify every term independently. An earlier term being "present" tells you nothing about the next \
one -- check each on its own merits.

Only after finishing both steps for every extracted term, set `status` to "issue" if and only if \
at least one entry in `checked_terms` has `verification` other than "present" -- `checked_terms` \
is what determines whether a group is repaired, so make sure every entry's `verification` is \
actually correct rather than defaulting to "present". Do not flag paraphrase, word order, dropped \
filler words, or stylistic differences -- those are expected and correct, and must never appear \
in `checked_terms`."""

_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["findings"],
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["group_id", "status", "checked_terms"],
                "additionalProperties": False,
                "properties": {
                    "group_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["ok", "issue"]},
                    "checked_terms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["term", "category", "verification"],
                            "additionalProperties": False,
                            "properties": {
                                "term": {"type": "string"},
                                "category": {
                                    "type": "string",
                                    "enum": ["name", "role_or_title", "number_or_date", "negation"],
                                },
                                "verification": {
                                    "type": "string",
                                    "enum": ["present", "missing", "substituted"],
                                },
                                "found_instead": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
    },
}

_REPAIR_SYSTEM_PROMPT = """You are repairing a specific, confirmed isiZulu translation fidelity \
error. For each group you receive the English source, the current (flawed) isiZulu text, and the \
exact finding describing what is missing or wrong. Return a corrected isiZulu translation that \
fixes only the named problem. Keep everything else -- register, phrasing, length -- as close to \
the original isiZulu as possible. Do not introduce new errors and do not omit any fact present in \
the English."""

_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["repairs"],
    "additionalProperties": False,
    "properties": {
        "repairs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["group_id", "corrected_zulu"],
                "additionalProperties": False,
                "properties": {
                    "group_id": {"type": "string"},
                    "corrected_zulu": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    size = max(1, int(size))
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _estimate_output_tokens(texts: Sequence[str], *, per_item_overhead: int) -> int:
    """Rough output-token budget from expected text length, not a flat item count.

    A flat "N tokens per item" guess badly underestimates a batch containing a
    long testimony passage (a single group can run 300+ English words). Budget
    from the actual character length of what each item needs to reproduce,
    since ``corrected_zulu`` is roughly the same length as ``current_zulu``.
    isiZulu averages well under 1 token per character in BPE tokenizers; 2.2
    chars/token is a conservative (generous) estimate, not a precise one.
    """

    total_chars = sum(len(str(text or "")) for text in texts)
    return int(total_chars / 2.2) + per_item_overhead * max(1, len(texts))


def _call_with_truncation_retry(
    call: Callable[[Sequence[Any]], tuple[dict[str, Any], dict[str, int]]],
    items: Sequence[Any],
    *,
    operation: str,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Retry a batched Grok call with a halved batch when output is truncated.

    A batch's token budget is an estimate; one unusually long group -- or
    several moderately long ones together -- can still exceed it. Rather than
    losing the whole batch, and everything already completed before it, split
    the batch in half and retry down to single items. A single item that
    still truncates is left unresolved and logged, not raised: a token-budget
    problem on one group must not crash an audit that already found and
    successfully repaired other real issues.
    """

    try:
        return call(items)
    except GrokOutputTruncated:
        if len(items) <= 1:
            group_id = items[0].get("group_id") if items else "?"
            _emit(
                progress,
                f"[referent audit] WARNING: {operation} truncated even for a single group "
                f"({group_id}); leaving it unresolved rather than failing the whole audit",
            )
            return {}, {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
        _emit(
            progress,
            f"[referent audit] {operation} truncated for a batch of {len(items)}; "
            "retrying as two smaller batches",
        )
        midpoint = len(items) // 2
        left_result, left_usage = _call_with_truncation_retry(
            call, items[:midpoint], operation=operation, progress=progress
        )
        right_result, right_usage = _call_with_truncation_retry(
            call, items[midpoint:], operation=operation, progress=progress
        )
        merged_usage = {
            key: left_usage.get(key, 0) + right_usage.get(key, 0)
            for key in set(left_usage) | set(right_usage)
        }
        return {**left_result, **right_result}, merged_usage


def deterministic_entity_findings(
    english: str, zulu: str, registry: EntityRegistry
) -> list[dict[str, Any]]:
    """Free, no-AI-call check: registered named entities must survive translation.

    Only covers entity types in ``_LITERALLY_PRESERVED_ENTITY_TYPES`` -- see its
    docstring for why. It cannot catch a generic role-noun substitution such as
    minister -> God either, since "minister" alone is not a registered entity;
    that class of error is left to the independent audit call.
    """

    findings: list[dict[str, Any]] = []
    zulu_lower = zulu.casefold()
    seen_entity_ids: set[str] = set()
    for match in registry.match_entities(english):
        if match.entity_id in seen_entity_ids:
            # match_entities may return more than one span for the same entity
            # (canonical text plus an overlapping alias); one finding per
            # entity per group is enough for the repair prompt.
            continue
        entity = registry.get_entity(match.entity_id)
        if entity.entity_type not in _LITERALLY_PRESERVED_ENTITY_TYPES:
            seen_entity_ids.add(match.entity_id)
            continue
        candidates = [entity.canonical_text, entity.display_text, *entity.aliases]
        if any(str(candidate or "").casefold() in zulu_lower for candidate in candidates if candidate):
            seen_entity_ids.add(match.entity_id)
            continue
        seen_entity_ids.add(match.entity_id)
        findings.append(
            {
                "source_term": match.matched_source_text,
                "expected_category": "name",
                "found_instead": "",
                "problem": "missing",
                "entity_id": match.entity_id,
                "source": "deterministic_entity_registry",
            }
        )
    return findings


def _derive_issues(finding: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Derive the authoritative issue list from the forced ``checked_terms`` checklist.

    The schema also allows a separate, model-authored ``issues`` field mirroring
    ``checked_terms``, but it is not reliable: a real production run had the model
    correctly extract and verify a substituted role/title in ``checked_terms``
    (status "issue") while leaving ``issues`` empty, and code that trusted only
    ``issues`` silently dropped the finding. ``checked_terms`` is the one place the
    two-step extract-then-verify contract is actually enforced, so it is the only
    source of truth here.
    """

    return [
        {
            "source_term": term.get("term"),
            "expected_category": term.get("category"),
            "found_instead": term.get("found_instead"),
            "problem": term.get("verification"),
        }
        for term in finding.get("checked_terms") or []
        if term.get("verification") != "present"
    ]


def _group_records(pass2: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build per-group {group_id, segment_ids, english, zulu} records.

    Temporal-mask Pass 2 already groups multiple segments under a joint
    translation and records each group's full English source text in
    ``temporal_mask_groups``; other Pass 2 modes carry one segment per group
    with ``source_text`` on the translation item itself.
    """

    temporal_groups = pass2.get("temporal_mask_groups")
    if temporal_groups:
        records = []
        for group in temporal_groups:
            records.append(
                {
                    "group_id": str(group["group_id"]),
                    "segment_ids": [str(value) for value in group.get("segment_ids") or []],
                    "english": str(group.get("source_text") or ""),
                    "zulu": str(group.get("spoken_text") or ""),
                }
            )
        return records

    records = []
    for item in pass2.get("translations") or []:
        segment_id = str(item["segment_id"])
        records.append(
            {
                "group_id": segment_id,
                "segment_ids": [segment_id],
                "english": str(item.get("source_text") or ""),
                "zulu": str(item.get("spoken_text") or ""),
            }
        )
    return records


def _request_audit_batch(
    *,
    provider: FoundryGrokProvider,
    groups: Sequence[Mapping[str, Any]],
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    items = [
        {"group_id": group["group_id"], "english": group["english"], "zulu": group["zulu"]}
        for group in groups
    ]
    # The forced extract-then-verify schema means output now includes one
    # `checked_terms` entry per checkable term in the group, not just a short
    # ok/issue label -- budget from English length (terms are drawn from the
    # English side), generously, since the truncation-retry wrapper is a
    # last-resort safety net, not the primary defense against undersizing.
    estimated_tokens = _estimate_output_tokens(
        [group["english"] for group in groups], per_item_overhead=250
    )
    response = provider.complete_json(
        operation="native_referent_audit_batch",
        system_prompt=_AUDIT_SYSTEM_PROMPT,
        payload={"groups": items},
        schema=_AUDIT_SCHEMA,
        max_output_tokens=min(
            max(2000, estimated_tokens), int(getattr(provider.config, "max_output_tokens", 8192))
        ),
    )
    returned = list(response.data.get("findings") or [])
    expected_ids = [str(group["group_id"]) for group in groups]
    expected_set = set(expected_ids)

    by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: list[str] = []
    unknown_ids: list[str] = []
    for item in returned:
        group_id = str(item.get("group_id") or "")
        if group_id in by_id:
            duplicate_ids.append(group_id)
            continue
        by_id[group_id] = item
        if group_id not in expected_set:
            unknown_ids.append(group_id)

    missing_ids = [group_id for group_id in expected_ids if group_id not in by_id]
    if duplicate_ids or unknown_ids or missing_ids:
        details = []
        if missing_ids:
            details.append("missing=" + ",".join(missing_ids))
        if unknown_ids:
            details.append("unknown=" + ",".join(unknown_ids))
        if duplicate_ids:
            details.append("duplicate=" + ",".join(duplicate_ids))
        raise ValueError("Native referent audit batch identity mismatch: " + "; ".join(details))

    result = {group_id: dict(by_id[group_id]) for group_id in expected_ids}
    usage = {
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "attempts": int(response.attempts),
    }
    return result, usage


def _request_repair_batch(
    *,
    provider: FoundryGrokProvider,
    items: Sequence[Mapping[str, Any]],
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, str], dict[str, int]]:
    estimated_tokens = _estimate_output_tokens(
        [str(item.get("current_zulu") or "") for item in items], per_item_overhead=80
    )
    response = provider.complete_json(
        operation="native_referent_repair_batch",
        system_prompt=_REPAIR_SYSTEM_PROMPT,
        payload={"groups": list(items)},
        schema=_REPAIR_SCHEMA,
        max_output_tokens=min(
            max(1200, estimated_tokens), int(getattr(provider.config, "max_output_tokens", 8192))
        ),
    )
    returned = list(response.data.get("repairs") or [])
    expected_ids = [str(item["group_id"]) for item in items]
    expected_set = set(expected_ids)

    by_id: dict[str, str] = {}
    duplicate_ids: list[str] = []
    unknown_ids: list[str] = []
    for item in returned:
        group_id = str(item.get("group_id") or "")
        corrected = str(item.get("corrected_zulu") or "").strip()
        if group_id in by_id:
            duplicate_ids.append(group_id)
            continue
        if group_id not in expected_set:
            unknown_ids.append(group_id)
            continue
        if not corrected:
            continue
        by_id[group_id] = corrected

    missing_ids = [group_id for group_id in expected_ids if group_id not in by_id]
    if duplicate_ids or unknown_ids or missing_ids:
        details = []
        if missing_ids:
            details.append("missing=" + ",".join(missing_ids))
        if unknown_ids:
            details.append("unknown=" + ",".join(unknown_ids))
        if duplicate_ids:
            details.append("duplicate=" + ",".join(duplicate_ids))
        raise ValueError("Native referent repair batch identity mismatch: " + "; ".join(details))

    usage = {
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "attempts": int(response.attempts),
    }
    return by_id, usage


def _apply_group_correction(pass2: MutableMapping[str, Any], group: Mapping[str, Any], corrected_zulu: str) -> None:
    segment_ids = list(group["segment_ids"])
    translations_by_id = {str(item["segment_id"]): item for item in pass2.get("translations") or []}
    if segment_ids:
        carrier_id, *blank_ids = segment_ids
        if carrier_id in translations_by_id:
            translations_by_id[carrier_id]["spoken_text"] = corrected_zulu
        for blank_id in blank_ids:
            if blank_id in translations_by_id:
                translations_by_id[blank_id]["spoken_text"] = ""

    for temporal_group in pass2.get("temporal_mask_groups") or []:
        if str(temporal_group.get("group_id")) == group["group_id"]:
            temporal_group["spoken_text"] = corrected_zulu
            break


def audit_and_repair_pass2(
    *,
    provider: FoundryGrokProvider,
    pass2: MutableMapping[str, Any],
    registry: EntityRegistry | None = None,
    batch_size: int = DEFAULT_REFERENT_AUDIT_BATCH_SIZE,
    max_repair_rounds: int = DEFAULT_MAX_REFERENT_REPAIR_ROUNDS,
    max_workers: int = DEFAULT_REFERENT_AUDIT_WORKERS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Audit every Pass-2 group for referent fidelity and repair confirmed issues in place.

    Mutates ``pass2["translations"]`` (and ``pass2["temporal_mask_groups"]`` when
    present) for any group a repair was applied to. Returns a summary suitable
    for embedding in the canonical Pass-2 artifact and for a separate,
    fuller ``referent_audit.json`` record.
    """

    groups = _group_records(pass2)
    if not groups:
        return {
            "schema_version": REFERENT_AUDIT_SCHEMA_VERSION,
            "groups_checked": 0,
            "deterministic_findings": [],
            "ai_findings": [],
            "repairs": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "attempts": 0},
        }

    _emit(progress, f"[referent audit] Checking {len(groups)} translated group(s) for referent fidelity...")

    deterministic_findings: list[dict[str, Any]] = []
    if registry is not None:
        for group in groups:
            for finding in deterministic_entity_findings(group["english"], group["zulu"], registry):
                deterministic_findings.append({"group_id": group["group_id"], **finding})

    total_usage = {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    ai_findings_by_group: dict[str, Mapping[str, Any]] = {}
    audit_batches = _chunks(groups, batch_size)
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(audit_batches)))) as pool:
        futures = {
            pool.submit(
                _call_with_truncation_retry,
                lambda subset: _request_audit_batch(provider=provider, groups=subset, progress=progress),
                batch,
                operation="native_referent_audit_batch",
                progress=progress,
            )
            for batch in audit_batches
        }
        # Results are only ever merged into ai_findings_by_group/total_usage here, on
        # the main thread, as each future completes -- no worker thread touches this
        # shared state directly, so there is nothing to lock.
        for future in as_completed(futures):
            batch_findings, usage = future.result()
            ai_findings_by_group.update(batch_findings)
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)

    flagged_group_ids = {
        group_id for group_id, finding in ai_findings_by_group.items() if _derive_issues(finding)
    }
    flagged_group_ids.update(item["group_id"] for item in deterministic_findings)

    ai_findings = [
        {"group_id": group_id, **finding}
        for group_id, finding in ai_findings_by_group.items()
        if group_id in flagged_group_ids
    ]

    _emit(
        progress,
        f"[referent audit] {len(flagged_group_ids)} group(s) flagged out of {len(groups)}"
        if flagged_group_ids
        else f"[referent audit] No referent issues found across {len(groups)} group(s)",
    )

    repairs: list[dict[str, Any]] = []
    groups_by_id = {group["group_id"]: group for group in groups}
    remaining = sorted(flagged_group_ids)
    for repair_round in range(1, max(0, int(max_repair_rounds)) + 1):
        if not remaining:
            break
        repair_items = []
        for group_id in remaining:
            group = groups_by_id[group_id]
            ai_finding = ai_findings_by_group.get(group_id) or {}
            issues = _derive_issues(ai_finding)
            issues.extend(
                {
                    "source_term": item["source_term"],
                    "expected_category": item.get("expected_category"),
                    "found_instead": item.get("found_instead"),
                    "problem": item["problem"],
                }
                for item in deterministic_findings
                if item["group_id"] == group_id
            )
            repair_items.append(
                {
                    "group_id": group_id,
                    "english": group["english"],
                    "current_zulu": group["zulu"],
                    "issues": issues,
                }
            )
        still_unresolved: list[str] = []
        repair_batches = _chunks(repair_items, batch_size)
        with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(repair_batches)))) as pool:
            futures = {
                pool.submit(
                    _call_with_truncation_retry,
                    lambda subset: _request_repair_batch(provider=provider, items=subset, progress=progress),
                    batch,
                    operation="native_referent_repair_batch",
                    progress=progress,
                ): batch
                for batch in repair_batches
            }
            # Every batch in one round touches a disjoint set of group_ids, and
            # pass2/groups_by_id are only ever mutated here on the main thread as each
            # future completes -- concurrent repair calls, strictly sequential writes.
            for future in as_completed(futures):
                batch = futures[future]
                corrected_by_id, usage = future.result()
                for key in total_usage:
                    total_usage[key] += usage.get(key, 0)
                for item in batch:
                    group_id = item["group_id"]
                    corrected = corrected_by_id.get(group_id)
                    if corrected is None:
                        still_unresolved.append(group_id)
                        continue
                    group = groups_by_id[group_id]
                    repairs.append(
                        {
                            "group_id": group_id,
                            "repair_round": repair_round,
                            "english": group["english"],
                            "previous_zulu": group["zulu"],
                            "corrected_zulu": corrected,
                        }
                    )
                    _apply_group_correction(pass2, group, corrected)
                    group["zulu"] = corrected
        remaining = still_unresolved

    if remaining:
        _emit(
            progress,
            f"[referent audit] WARNING: {len(remaining)} flagged group(s) could not be repaired "
            f"within {max_repair_rounds} round(s): {', '.join(remaining)}",
        )

    return {
        "schema_version": REFERENT_AUDIT_SCHEMA_VERSION,
        "groups_checked": len(groups),
        "flagged_group_ids": sorted(flagged_group_ids),
        "deterministic_findings": deterministic_findings,
        "ai_findings": ai_findings,
        "all_ai_verdicts": [
            {"group_id": group_id, **finding} for group_id, finding in ai_findings_by_group.items()
        ],
        "repairs": repairs,
        "unresolved_group_ids": remaining,
        "usage": total_usage,
    }


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
