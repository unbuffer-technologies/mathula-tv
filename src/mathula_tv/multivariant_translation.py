"""Production-safe multivariant translation request, validation, and import.

The manual/cloud translation boundary is deliberately file based.  Mathula TV
exports a source-bound request and stable prompts; any external language model
may answer it.  The answer is never trusted merely because it is valid JSON:
this module binds it back to the authoritative English timeline, validates all
three delivery variants independently, derives provenance hashes, and installs
it atomically with immutable override history.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .atomic_io import atomic_write_json, read_json
from .entity_registry import EntityRegistry
from .media import checksum
from .pronunciation import PronunciationDictionary, with_default_organisation_initialisms

REQUEST_SCHEMA_VERSION = "mathula.translation.request.v1"
RESPONSE_SCHEMA_VERSION = "mathula.translation.multivariant.v1"
INSTALLED_SCHEMA_VERSION = "mathula.translation.multivariant.installed.v1"
PROMPT_VERSION = (
    "mathula-multivariant-translation-prompt-v6-natural-code-switch-grammar"
)
VALIDATION_CONTRACT_VERSION = (
    "mathula-multivariant-validation-v8-local-anchor-progression"
)
BATCH_REQUEST_SCHEMA_VERSION = "mathula.translation.batch-request.v1"
VARIANT_IDS = ("natural", "concise", "compact")
VARIANT_RATIOS = {"natural": 0.95, "concise": 0.82, "compact": 0.68}
_LOCALE_NAMES = {
    "zu-ZA": "isiZulu",
    "nso-ZA": "Sepedi",
}
_LANGUAGE_SUFFIXES = {
    "zu-ZA": "zu",
    "nso-ZA": "nso",
}

_PROPER_NAME_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*\b")
_WORD_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*")
_MAX_LOCAL_ANCHOR_DRIFT = 0.30

SYSTEM_PROMPT = r"""You are the translation engine for Mathula TV.

Mathula TV is an AI code-switching dubbing system, not a conventional
word-for-word dubbing system. Translate an English transcript into natural
spoken {{TARGET_LANGUAGE}} for professional television, news, interviews,
documentaries, and online video.

Return exactly three usable spoken variations for every input unit, in
this order:
1. natural — the most natural contextually complete translation;
2. concise — materially shorter while preserving every required fact;
3. compact — the shortest fluent version that preserves essential meaning,
   attribution, negation, numbers, names, and speaker intent.

Make the three spoken_text values distinct whenever truthful, fluent compression
is possible. Never invent wording, remove a required fact, corrupt a protected
identity, or make the sentence unnatural merely to force textual difference.
When a unit is already at its safe compression floor, adjacent variants may use
the same spoken_text. Equivalent variants must use the same estimated duration
and the unit must add warning code "compression_floor_reached". The pipeline
will collapse equivalent audio candidates before Azure synthesis.

Treat duration targets as primary constraints. Character and word counts are
not primary constraints. Write wording a fluent broadcaster could reasonably
speak within each target duration. Do not merely trim one or two words.

Translate ordinary English confidently. Preserve English only where retaining
it is more accurate, natural, or safe: names, brands, slogans, technical/legal
terms, acronyms, and culturally specific or ambiguous expressions. Preserve
only the risky span, not the surrounding ordinary English. Treat reviewed
institutions and commissions as discourse identities: use a specific rendering
when introducing or contrasting the institution, then allow natural
target-language anaphora or a generic institutional noun in later timing units
when the antecedent is clear. Do not mechanically repeat a full formal
institution name in every unit. Names of people, brands, acronyms, numbers, and
quoted code-switching remain strict unit-level content. Never infer a
protected term from a substring inside another word. For example, "Ando" must
not be inferred from "Thohoyandou" and "Levy Ndou" must never become "Levy
Ando". Integrate retained English names with grammatical target-language
morphology. In isiZulu, do not produce doubled determiners such as "I-The
Hawks". When this plural organisation is the subject, use the reviewed
code-switch construction "Ama-Hawks athi"; isiZulu has no equivalent definite
article here, ``ama-`` supplies the plural noun class, and ``athi`` supplies its
agreement.

Preserve claims, attribution, uncertainty, negation, quotations, names,
institutions, numbers, dates, currencies, percentages, locations, and the
direction of every argument. Do not add facts. Do not move content between
speakers or units. Optional details may be omitted only when the omission does
not distort the speaker's essential meaning.

Within each timing unit, preserve the source's local information progression
where target-language grammar permits. In particular, do not move a person's
name, organisation, number, or quoted term from the end of a source clause to
its beginning. Keep these anchors at approximately the same relative position
in all three variants so local mouth timing remains aligned, not merely the unit
endpoint. When the first natural construction would move an anchor by more than
roughly one third of the clause, rephrase the surrounding isiZulu clause so the
anchor stays late or early with the source; grammatical naturalness does not
justify a large timing jump.

Return spoken_text only. Do not return SSML, phonemes, pronunciation respelling,
tts_text, Markdown, commentary, or reasoning. Mathula TV derives TTS text and
SSML in application code.

When the input contains translation_batch, translate only the units in the top-level
units array. Use translation_batch.context_units only to resolve source context.
Use translation_batch.prior_global_terminology and
translation_batch.prior_translated_context to keep terminology and broadcast
style consistent with validated earlier batches. Never return any context-only
unit.

Return exactly one valid JSON object using schema_version
"mathula.translation.multivariant.v1". Every input unit must appear exactly
once with unchanged IDs, speakers, source text, and timing. Each unit must
contain exactly three variants in natural, concise, compact order. Set
selection_status to "unselected" and selected_variant_id to null. If a duration
target appears impossible without losing required meaning, preserve truth,
produce the shortest safe wording, and add warning code
"duration_target_may_be_impossible".

Before returning, silently verify JSON validity, timeline identity, protected
spans, factual preservation, and that concise is materially shorter than
natural and compact is materially shorter than concise whenever safe
compression exists. Do not fail or distort an irreducible short unit merely to
make all three spoken_text values different."""

USER_PROMPT_TEMPLATE = r"""Translate the supplied English transcript into {{TARGET_LANGUAGE}}.

Target locale: {{TARGET_LOCALE}}
Prompt version: {{PROMPT_VERSION}}

Return exactly one JSON object using schema_version
"mathula.translation.multivariant.v1" and obey the complete system prompt.
Use the supplied timing targets exactly. Preserve every unit ID, speaker ID,
source timestamp, source_text, required fact, protected span, attribution,
number, date, and negation. Do not return tts_text, SSML, Markdown, or
commentary.

INPUT TRANSCRIPT JSON:
{{TRANSCRIPT_JSON}}"""

VARIANT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "variant_id",
        "compression_level",
        "target_duration_ms",
        "spoken_text",
        "estimated_duration_ms",
        "preserved_english_spans",
        "omitted_optional_details",
        "meaning_preserved",
    ],
    "properties": {
        "variant_id": {"enum": list(VARIANT_IDS)},
        "compression_level": {"enum": list(VARIANT_IDS)},
        "target_duration_ms": {"type": "integer", "minimum": 1},
        "spoken_text": {"type": "string", "minLength": 1},
        "estimated_duration_ms": {"type": "integer", "minimum": 1},
        "preserved_english_spans": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "omitted_optional_details": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "meaning_preserved": {"type": "boolean"},
    },
}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "source_language",
        "target_language",
        "target_locale",
        "translation_strategy",
        "units",
        "global_terminology",
        "warnings",
    ],
    "properties": {
        "schema_version": {"const": RESPONSE_SCHEMA_VERSION},
        "source_language": {"type": "string"},
        "target_language": {"type": "string"},
        "target_locale": {"enum": sorted(_LOCALE_NAMES)},
        "translation_strategy": {"const": "contextual_code_switching"},
        "units": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "unit_id",
                    "speaker_id",
                    "start_ms",
                    "end_ms",
                    "available_duration_ms",
                    "overlap_group",
                    "source_text",
                    "protected_spans",
                    "required_facts",
                    "optional_details",
                    "variants",
                    "selection_status",
                    "selected_variant_id",
                    "warnings",
                ],
                "properties": {
                    "unit_id": {"type": "string", "minLength": 1},
                    "speaker_id": {"type": "string", "minLength": 1},
                    "start_ms": {"type": "integer", "minimum": 0},
                    "end_ms": {"type": "integer", "minimum": 1},
                    "available_duration_ms": {"type": "integer", "minimum": 1},
                    "overlap_group": {"type": ["string", "null"]},
                    "source_text": {"type": "string"},
                    "protected_spans": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "uniqueItems": True,
                    },
                    "required_facts": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "optional_details": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "variants": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": VARIANT_SCHEMA,
                    },
                    "selection_status": {"const": "unselected"},
                    "selected_variant_id": {"type": "null"},
                    "warnings": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "additionalProperties": True,
                                    "required": ["code"],
                                    "properties": {"code": {"type": "string"}},
                                },
                            ]
                        },
                    },
                },
            },
        },
        "global_terminology": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_term",
                    "target_rendering",
                    "preserve_english",
                    "reason",
                ],
                "properties": {
                    "source_term": {"type": "string", "minLength": 1},
                    "target_rendering": {"type": "string", "minLength": 1},
                    "preserve_english": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
        },
        "warnings": {"type": "array"},
    },
}


class MultivariantTranslationError(ValueError):
    """Raised when a manual/cloud translation cannot be safely installed."""


@dataclass(frozen=True)
class SourceUnit:
    unit_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    source_text: str
    overlap_group: str | None
    protected_spans: tuple[str, ...]

    @property
    def available_duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def request_dict(self) -> dict[str, Any]:
        targets = {
            f"{variant}_target_duration_ms": max(
                1,
                min(
                    self.available_duration_ms,
                    round(self.available_duration_ms * VARIANT_RATIOS[variant]),
                ),
            )
            for variant in VARIANT_IDS
        }
        return {
            "unit_id": self.unit_id,
            "speaker_id": self.speaker_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "available_duration_ms": self.available_duration_ms,
            "overlap_group": self.overlap_group,
            "source_text": self.source_text,
            "protected_spans": list(self.protected_spans),
            "duration_targets": targets,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def language_name(locale: str) -> str:
    try:
        return _LOCALE_NAMES[locale]
    except KeyError as exc:
        raise MultivariantTranslationError(
            f"Unsupported target locale {locale!r}; supported locales are "
            f"{', '.join(sorted(_LOCALE_NAMES))}"
        ) from exc


def language_suffix(locale: str) -> str:
    language_name(locale)
    return _LANGUAGE_SUFFIXES[locale]


def _first_text(item: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_punctuation_only_source_text(value: str) -> bool:
    """Return true only when every visible character is punctuation."""

    text = str(value or "").strip()
    return bool(text) and all(
        unicodedata.category(character).startswith(("P", "Z"))
        for character in text
    )


def _timeline_items(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("segments", "units", "turns"):
        items = value.get(key)
        if isinstance(items, list) and items:
            if not all(isinstance(item, Mapping) for item in items):
                raise MultivariantTranslationError(
                    f"Source transcript {key} must contain only objects"
                )
            return list(items)
    raise MultivariantTranslationError(
        "Source transcript must contain a non-empty segments, units, or turns array"
    )


def _finite_milliseconds(raw: Any, *, field: str, seconds: bool = False) -> int:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise MultivariantTranslationError(f"Invalid {field}: {raw!r}") from exc
    if not math.isfinite(value):
        raise MultivariantTranslationError(f"Invalid {field}: {raw!r}")
    if seconds:
        value *= 1000
    return int(round(value))


def _milliseconds(item: Mapping[str, Any], *, start: bool) -> int:
    """Return a canonical timing boundary for transcripts or dubbing units.

    Normalized transcripts expose ``start_ms``/``end_ms`` (or seconds), while
    deterministic dubbing units expose ``start_ms``, ``preferred_end_ms`` and
    ``hard_end_ms``. Translation variants are generated against the hard window
    because it is the maximum collision-safe window already calculated by the
    dubbing-unit builder.
    """

    if start:
        for key, seconds in (("start_ms", False), ("window_start_ms", False), ("start", True)):
            if key in item and item.get(key) is not None:
                return _finite_milliseconds(item.get(key), field=key, seconds=seconds)
        raise MultivariantTranslationError(
            "Timeline unit is missing a valid start_ms, window_start_ms, or start"
        )

    # Prefer the explicit hard window produced by dubbing_units.py. It includes
    # only bounded, collision-safe trailing silence and is therefore the correct
    # duration budget for translation.
    for key, seconds in (
        ("end_ms", False),
        ("hard_end_ms", False),
        ("window_end_ms", False),
        ("preferred_end_ms", False),
        ("end", True),
    ):
        if key in item and item.get(key) is not None:
            return _finite_milliseconds(item.get(key), field=key, seconds=seconds)

    start_ms = _milliseconds(item, start=True)
    for key in ("maximum_duration_ms", "preferred_duration_ms", "duration_ms"):
        if key in item and item.get(key) is not None:
            duration_ms = _finite_milliseconds(item.get(key), field=key)
            return start_ms + duration_ms
    if "duration" in item and item.get("duration") is not None:
        return start_ms + _finite_milliseconds(
            item.get("duration"), field="duration", seconds=True
        )
    raise MultivariantTranslationError(
        "Timeline unit is missing a valid end_ms, hard_end_ms, "
        "preferred_end_ms, end, or duration"
    )


def _explicit_protected_spans(item: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for key in (
        "protected_spans",
        "protected_entities",
        "protected_entities_found",
        "preserved_english_spans",
    ):
        values = item.get(key)
        if isinstance(values, (list, tuple)):
            for value in values:
                text = str(value or "").strip()
                if text and text not in result:
                    result.append(text)
    return result


def _binding_spans(
    entity_bindings: Mapping[str, Any] | None,
    unit_id: str,
) -> list[str]:
    if not isinstance(entity_bindings, Mapping):
        return []
    unit = (entity_bindings.get("per_unit_bindings") or {}).get(unit_id)
    if not isinstance(unit, Mapping):
        return []
    result: list[str] = []
    for match in unit.get("matches", []):
        if not isinstance(match, Mapping):
            continue
        text = str(
            # The authoritative, autocorrected source surface controls how
            # explicit the reference is. A surname or acronym must remain a
            # surname or acronym; expanding it to the registry's full display
            # name changes referential information and local timing.
            match.get("matched_source_text")
            or match.get("display_text")
            or match.get("canonical_text")
            or ""
        ).strip()
        if text and text not in result:
            result.append(text)
    return result


def extract_source_units(
    transcript: Mapping[str, Any],
    *,
    entity_bindings: Mapping[str, Any] | None = None,
) -> list[SourceUnit]:
    units: list[SourceUnit] = []
    for index, item in enumerate(_timeline_items(transcript), start=1):
        unit_id = str(
            item.get("unit_id")
            or item.get("segment_id")
            or item.get("id")
            or f"seg-{index:05d}"
        ).strip()
        speaker_id = str(item.get("speaker_id") or item.get("speaker") or "").strip()
        source_text = _first_text(item, ("source_text", "text", "display", "lexical"))
        start_ms = _milliseconds(item, start=True)
        end_ms = _milliseconds(item, start=False)
        if not unit_id:
            raise MultivariantTranslationError(f"Source unit {index} has no ID")
        if not speaker_id:
            raise MultivariantTranslationError(f"Source unit {unit_id} has no speaker ID")
        if end_ms <= start_ms or start_ms < 0:
            raise MultivariantTranslationError(
                f"Source unit {unit_id} has invalid timing {start_ms}-{end_ms}ms"
            )
        # Azure can occasionally emit punctuation-only pseudo-phrases such as
        # "......." for silence or an abandoned recognition hypothesis. They
        # contain no speech to translate or synthesize, so exclude them before
        # spending AI tokens. Their source-video interval remains untouched.
        if _is_punctuation_only_source_text(source_text):
            continue
        protected = _explicit_protected_spans(item)
        for value in _binding_spans(entity_bindings, unit_id):
            if value not in protected:
                protected.append(value)
        overlap_group = item.get("overlap_group")
        if overlap_group in (None, ""):
            overlap_group = item.get("overlap_group_id")
        units.append(
            SourceUnit(
                unit_id=unit_id,
                speaker_id=speaker_id,
                start_ms=start_ms,
                end_ms=end_ms,
                source_text=source_text,
                overlap_group=(
                    str(overlap_group).strip() if overlap_group not in (None, "") else None
                ),
                protected_spans=tuple(protected),
            )
        )
    if len({unit.unit_id for unit in units}) != len(units):
        raise MultivariantTranslationError("Source transcript contains duplicate unit IDs")
    ordered = sorted(units, key=lambda item: (item.start_ms, item.end_ms, item.unit_id))
    if ordered != units:
        raise MultivariantTranslationError("Source transcript units are not in timeline order")
    return units


def build_translation_request(
    *,
    transcript: Mapping[str, Any],
    target_locale: str,
    job_id: str,
    entity_bindings: Mapping[str, Any] | None = None,
    global_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target_language = language_name(target_locale)
    units = extract_source_units(transcript, entity_bindings=entity_bindings)
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "job_id": job_id,
        "source_language": str(transcript.get("language") or "en-ZA"),
        "target_language": target_language,
        "target_locale": target_locale,
        "translation_strategy": "contextual_code_switching",
        "global_context": dict(global_context or {}),
        "global_protected_spans": list(
            dict.fromkeys(
                span for unit in units for span in unit.protected_spans if span
            )
        ),
        "units": [unit.request_dict() for unit in units],
    }
    request["source_transcript_sha256"] = _sha256_json(transcript)
    request["request_sha256"] = _sha256_json(request)
    return request



def build_translation_batch_requests(
    request: Mapping[str, Any],
    *,
    batch_size: int,
    context_units: int = 2,
) -> list[dict[str, Any]]:
    """Split one source-bound request into deterministic, contextual batches.

    Only top-level ``units`` are translated. Adjacent context is copied into
    ``translation_batch.context_units`` so the model can resolve references
    without returning duplicate units. Batch request hashes are stable, which
    makes completed checkpoints safely resumable across process restarts.
    """

    if batch_size <= 0:
        raise ValueError("translation batch_size must be positive")
    if context_units < 0:
        raise ValueError("translation context_units must not be negative")
    source_units = request.get("units")
    if not isinstance(source_units, list) or not source_units:
        raise MultivariantTranslationError("Translation request contains no units")

    full_request_sha256 = str(request.get("request_sha256") or _sha256_json(request))
    total_batches = math.ceil(len(source_units) / batch_size)
    batches: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(source_units), batch_size), start=1):
        stop = min(len(source_units), start + batch_size)
        chunk = [dict(item) for item in source_units[start:stop]]
        context_start = max(0, start - context_units)
        context_stop = min(len(source_units), stop + context_units)
        context = [
            {
                "unit_id": str(item.get("unit_id") or ""),
                "speaker_id": str(item.get("speaker_id") or ""),
                "start_ms": int(item.get("start_ms") or 0),
                "end_ms": int(item.get("end_ms") or 0),
                "source_text": str(item.get("source_text") or ""),
                "protected_spans": list(item.get("protected_spans") or []),
                "translate_this_unit": start <= index < stop,
            }
            for index, item in enumerate(source_units[context_start:context_stop], start=context_start)
        ]
        batch = {
            key: value
            for key, value in request.items()
            if key not in {"units", "request_sha256", "translation_batch"}
        }
        batch["units"] = chunk
        batch["translation_batch"] = {
            "schema_version": BATCH_REQUEST_SCHEMA_VERSION,
            "batch_index": batch_index,
            "batch_count": total_batches,
            "batch_size": len(chunk),
            "requested_unit_ids": [str(item.get("unit_id") or "") for item in chunk],
            "full_request_sha256": full_request_sha256,
            "context_radius_units": context_units,
            "context_units": context,
        }
        batch["request_sha256"] = _sha256_json(batch)
        batches.append(batch)
    return batches


def merge_multivariant_batch_responses(
    *,
    request: Mapping[str, Any],
    responses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge independently validated batch responses into one full response."""

    expected_units = list(request.get("units") or [])
    expected_ids = [str(item.get("unit_id") or "") for item in expected_units]
    units_by_id: dict[str, dict[str, Any]] = {}
    terminology: list[dict[str, Any]] = []
    terminology_seen: set[str] = set()
    warnings: list[Any] = []
    warning_seen: set[str] = set()

    for response in responses:
        for item in response.get("units", []):
            unit_id = str(item.get("unit_id") or "")
            if unit_id not in expected_ids:
                raise MultivariantTranslationError(
                    f"Batch response returned unexpected unit {unit_id or '<empty>'}"
                )
            if unit_id in units_by_id:
                raise MultivariantTranslationError(
                    f"Batch responses returned duplicate unit {unit_id}"
                )
            units_by_id[unit_id] = dict(item)
        for item in response.get("global_terminology", []):
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key not in terminology_seen:
                terminology_seen.add(key)
                terminology.append(dict(item))
        for item in response.get("warnings", []):
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key not in warning_seen:
                warning_seen.add(key)
                warnings.append(item)

    missing = [unit_id for unit_id in expected_ids if unit_id not in units_by_id]
    if missing:
        raise MultivariantTranslationError(
            f"Batch translation is incomplete; missing units: {missing}"
        )

    merged = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "source_language": str(request.get("source_language") or "en-ZA"),
        "target_language": str(request.get("target_language") or language_name(str(request.get("target_locale") or ""))),
        "target_locale": str(request.get("target_locale") or ""),
        "translation_strategy": "contextual_code_switching",
        "units": [units_by_id[unit_id] for unit_id in expected_ids],
        "global_terminology": terminology,
        "warnings": warnings,
    }
    validate_multivariant_response(merged, request=request)
    return merged



_COMPACT_OPTIONAL_DETAIL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "department",
        "former",
        "in",
        "of",
        "office",
        "position",
        "post",
        "role",
        "the",
        "title",
        "to",
    }
)


def _compact_optional_detail_anchor(detail: str) -> str:
    """Return the source-faithful anchor that longer variants must preserve.

    Provider output sometimes correctly marks a detail as omitted only in the
    compact variant but forgets to copy that detail into the unit-level
    ``optional_details`` list.  We may repair that metadata locally only when a
    strong lexical anchor remains in both longer variants.  Generic English
    scaffolding such as ``Department of`` is ignored, while distinctive names,
    acronyms and domain phrases remain mandatory evidence.
    """

    words = _identity_words(detail)
    anchors = [
        word
        for word in words
        if word.casefold() not in _COMPACT_OPTIONAL_DETAIL_STOPWORDS
    ]
    return " ".join(anchors)


def _compact_optional_detail_is_source_grounded(
    detail: str,
    *,
    source_text: str,
    required_facts: Sequence[str],
    protected_spans: Sequence[str],
    variants_by_id: Mapping[str, Mapping[str, Any]],
    target_locale: str,
) -> bool:
    """Approve a missing optional-detail declaration under a strict local rule.

    The rule is intentionally narrow: the detail must be quoted from the source,
    omitted only by compact, preserved through a distinctive anchor in both
    natural and concise, absent from compact, and unrelated to required or
    protected content.  This changes metadata only; spoken text is never edited.
    """

    candidate = _normalise_space(str(detail))
    if not candidate:
        return False

    candidate_words = _identity_words(candidate)
    semantic_label = bool(candidate_words) and candidate_words[-1].casefold() in {
        "position",
        "post",
        "role",
        "title",
    }
    anchor = _compact_optional_detail_anchor(candidate)
    anchor_words = [word.casefold() for word in _identity_words(anchor)]
    source_words = {word.casefold() for word in _identity_words(source_text)}
    exact_source_phrase = _contains_protected_span(source_text, candidate)
    # GPT can describe an omitted source detail with a compact
    # semantic label rather than an exact quotation, for example
    # ``Crime Intelligence Head role`` for source wording such as ``the head of
    # Crime Intelligence``. Accept that metadata label only when at least two
    # distinctive anchor words are independently present in the source.
    source_grounded_semantic_label = (
        semantic_label
        and len(anchor_words) >= 2
        and all(word in source_words for word in anchor_words)
    )
    if not exact_source_phrase and not source_grounded_semantic_label:
        return False

    candidate_key = candidate.casefold()
    for fact in required_facts:
        fact_value = _normalise_space(str(fact))
        fact_key = fact_value.casefold()
        if candidate_key in fact_key or fact_key in candidate_key:
            return False
    for span in protected_spans:
        span_value = _normalise_space(str(span))
        span_key = span_value.casefold()
        if candidate_key in span_key or span_key in candidate_key:
            return False

    if not anchor:
        return False

    def preserves(variant_id: str) -> bool:
        variant = variants_by_id.get(variant_id)
        if not isinstance(variant, Mapping):
            return False
        spoken = _normalise_space(str(variant.get("spoken_text") or ""))
        return _matched_translation_identity_form(
            spoken,
            (anchor,),
            span=anchor,
            target_locale=target_locale,
        ) is not None

    if exact_source_phrase:
        return preserves("natural") and preserves("concise") and not preserves("compact")

    # A source-grounded semantic role label may be translated in the longer
    # target-language variants, so English lexical matching is not reliable.
    # The repair remains metadata-only and compact-only: all three variants
    # must explicitly claim preserved meaning, and the compact delivery must be
    # shorter than both longer deliveries. Human review remains mandatory.
    variant_records = [variants_by_id.get(key) for key in ("natural", "concise", "compact")]
    if not all(isinstance(item, Mapping) for item in variant_records):
        return False
    if not all(bool(item.get("meaning_preserved")) for item in variant_records):
        return False
    spoken_lengths = [
        len(_normalise_space(str(item.get("spoken_text") or "")))
        for item in variant_records
    ]
    return spoken_lengths[2] < spoken_lengths[0] and spoken_lengths[2] < spoken_lengths[1]

def _apply_source_specific_identity_expansion(
    variants: Sequence[dict[str, Any]],
    *,
    protected_spans: Sequence[str],
    target_locale: str,
    source_text: str,
) -> list[dict[str, Any]]:
    """Restore a source-specific reviewed identity before strict validation.

    A deterministic entity binding may require a fully identified institution
    even when GPT returns only a generic target noun (for example
    ``iKhomishini``) or an anaphoric sentence that omits the institution name.
    Rejecting the complete paid chunk is wasteful when the application already
    owns a reviewed target-language identity.

    The repair remains deliberately narrow:

    * generic source aliases such as ``the Commission`` are left untouched;
    * only identities with a reviewed locale profile are eligible;
    * a generic target noun is expanded using an exact configured suffix;
    * when the target omitted the identity completely, the shortest reviewed
      specific form is inserted as a spoken appositional topic; and
    * the normal strict validator still runs after the repair.

    No web result, free-form translation, or guessed person name is introduced.
    """

    repairs: list[dict[str, Any]] = []
    for raw_span in protected_spans:
        span = _normalise_space(str(raw_span))
        if not span:
            continue
        locale_profile = _identity_alias_profile(span, target_locale=target_locale)
        if locale_profile is None:
            continue

        # A unit whose source really says only "the Commission" may translate
        # it generically.  Every other source-bound protected identity must keep
        # a specific reviewed rendering, even when the exact canonical wording
        # was reconstructed by entity binding rather than repeated verbatim in
        # this small source unit.
        if _source_uses_generic_translation_identity(
            span,
            target_locale=target_locale,
            source_text=source_text,
        ):
            continue

        generic_forms = [
            _normalise_space(str(value))
            for value in locale_profile.get("generic_forms") or []
            if _normalise_space(str(value))
        ]
        specific_forms = [
            _normalise_space(str(value))
            for value in locale_profile.get("specific_forms") or []
            if _normalise_space(str(value))
        ]
        if not specific_forms:
            continue

        # Prefer the shortest reviewed identity because it adds the least timing
        # pressure.  For each generic root, prefer the shortest specific form
        # that extends that same root, allowing normal isiZulu prefixes and
        # suffixes around the root to remain untouched.
        # Configuration order is editorial preference order.  The reviewed
        # Madlanga profile intentionally lists ``Khomishini kaMadlanga`` first.
        shortest_specific = specific_forms[0]
        expansions: list[tuple[str, str, str]] = []
        for generic in generic_forms:
            generic_key = generic.casefold()
            candidates = [
                specific
                for specific in specific_forms
                if specific.casefold().startswith(generic_key + " ")
            ]
            if not candidates:
                continue
            specific = min(
                candidates,
                key=lambda value: (len(value), value.casefold()),
            )
            suffix = specific[len(generic) :]
            if suffix.strip():
                expansions.append((generic, suffix, specific))

        approved_forms = _approved_translation_identity_forms(
            span,
            target_locale=target_locale,
            source_text=source_text,
        )
        for variant in variants:
            spoken = _normalise_space(str(variant.get("spoken_text") or ""))
            if not spoken:
                continue
            if _matched_translation_identity_form(
                spoken,
                approved_forms,
                span=span,
                target_locale=target_locale,
            ) is not None:
                continue

            expanded: str | None = None
            matched_generic_form: str | None = None
            reviewed_specific_form = shortest_specific
            for generic, suffix, specific in expansions:
                # _span_pattern already handles attached isiZulu/Sepedi
                # grammatical prefixes (iKhomishini, eKhomishini, yiKhomishini)
                # and proper lexical boundaries.  The v13.11.2 custom regex did
                # not reliably cover every attached morphology seen in live
                # provider output.
                match = _span_pattern(generic).search(spoken)
                if match is None:
                    continue
                expanded = spoken[: match.end()] + suffix + spoken[match.end() :]
                matched_generic_form = match.group(0)
                reviewed_specific_form = specific
                break

            policy = "append_reviewed_identity_suffix_to_generic_target"
            if expanded is None:
                # GPT may omit the commission noun entirely while preserving
                # the rest of the sentence.  Use the exact reviewed identity as
                # an appositional spoken topic rather than paying for another
                # generation or inventing new target-language wording.
                spoken_label = shortest_specific
                if target_locale == "zu-ZA" and re.match(
                    r"(?iu)^(?:Khomishini|Khomishana|Khomishani|Khomishane)\b",
                    spoken_label,
                ):
                    spoken_label = "i" + spoken_label
                expanded = f"{spoken_label}, {spoken}"
                policy = "prepend_shortest_reviewed_source_specific_identity"

            expanded = _normalise_space(expanded)
            old_duration = int(variant.get("estimated_duration_ms") or 0)
            if old_duration > 0:
                # Add the same deterministic speech-cost estimate to each
                # delivery variant.  A proportional multiplier can make the
                # shortest compact sentence appear longer than natural and
                # violate the ordered-duration contract merely because the
                # inserted identity is a larger fraction of its text.
                added_characters = max(0, len(expanded) - len(spoken))
                added_duration_ms = max(220, added_characters * 55)
                variant["estimated_duration_ms"] = (
                    old_duration + added_duration_ms
                )
            variant["spoken_text"] = expanded
            repairs.append(
                {
                    "code": "source_specific_identity_restored_before_validation",
                    "required_span": span,
                    "variant_id": str(variant.get("variant_id") or ""),
                    "matched_generic_form": matched_generic_form,
                    "reviewed_specific_form": reviewed_specific_form,
                    "replacement_policy": policy,
                }
            )
    return repairs

def _apply_protected_identity_compression_floor_fallback(
    variants: Sequence[dict[str, Any]],
    *,
    protected_spans: Sequence[str],
    target_locale: str,
    source_text: str,
) -> list[dict[str, Any]]:
    """Restore missing protected identities from a valid sibling variant.

    GPT can preserve all required identities in one delivery variant while
    dropping or translating them beyond recognition in another. Repeating the
    paid full-transcript request is wasteful, but inventing or splicing target-
    language wording locally would be unsafe. The deterministic fallback is to
    reuse one *already returned* sibling variant verbatim.

    A donor is eligible only when it preserves every protected identity in the
    unit under the normal reviewed-alias matcher. Each failing variant is
    replaced by the nearest eligible sibling, preferring the richer variant on
    an equal-distance tie. Repaired variants may become donors because their
    payload is an exact copy of an already validated provider variant. This
    preserves the previous natural -> concise -> compact recovery chain while
    also allowing ``natural`` to recover from ``concise`` or ``compact``.

    If no single sibling preserves every protected identity, nothing is changed
    and normal validation fails closed.
    """

    by_id = {
        str(item.get("variant_id") or ""): item
        for item in variants
        if isinstance(item, dict)
    }
    spans = [
        _normalise_space(str(raw_span))
        for raw_span in protected_spans
        if _normalise_space(str(raw_span))
        and _unit_protected_span_policy(
            _normalise_space(str(raw_span)),
            target_locale=target_locale,
            source_text=source_text,
        )
        == "hard_identity"
    ]
    if not spans:
        return []

    approved_by_span = {
        span: _approved_translation_identity_forms(
            span,
            target_locale=target_locale,
            source_text=source_text,
        )
        for span in spans
    }

    def raw_missing_spans(variant: Mapping[str, Any]) -> list[str]:
        spoken = _normalise_space(str(variant.get("spoken_text") or ""))
        return [
            span
            for span in spans
            if _matched_translation_identity_form(
                spoken,
                approved_by_span[span],
                span=span,
                target_locale=target_locale,
            )
            is None
        ]

    raw_missing_by_id = {
        variant_id: raw_missing_spans(variant)
        for variant_id, variant in by_id.items()
    }

    def unresolved_missing_spans(
        variant_id: str,
        variant: Mapping[str, Any],
    ) -> list[str]:
        missing = list(raw_missing_by_id.get(variant_id, []))
        if variant_id != "compact" or not missing:
            return missing

        spoken = _normalise_space(str(variant.get("spoken_text") or ""))
        unresolved: list[str] = []
        for span in missing:
            longer_variants_preserve = all(
                span not in raw_missing_by_id.get(longer_id, spans)
                for longer_id in ("natural", "concise")
            )
            compact_substitution = None
            if longer_variants_preserve:
                compact_substitution = _matched_compact_identity_substitution(
                    spoken,
                    span=span,
                    target_locale=target_locale,
                    source_text=source_text,
                )
            if compact_substitution is not None:
                continue
            if longer_variants_preserve and _source_uses_generic_translation_identity(
                span,
                target_locale=target_locale,
                source_text=source_text,
            ):
                continue
            unresolved.append(span)
        return unresolved

    # A donor must preserve every identity under the normal reviewed matcher.
    # A compact policy exception is sufficient to keep that compact variant,
    # but it must never become a donor for natural or concise.
    valid_donor_ids = {
        variant_id
        for variant_id in VARIANT_IDS
        if (variant := by_id.get(variant_id)) is not None
        and not raw_missing_by_id.get(variant_id, [])
        and bool(variant.get("meaning_preserved"))
    }
    if not valid_donor_ids:
        return []

    order = {variant_id: index for index, variant_id in enumerate(VARIANT_IDS)}
    repairs: list[dict[str, Any]] = []
    for variant_id in VARIANT_IDS:
        variant = by_id.get(variant_id)
        if variant is None:
            continue
        missing = unresolved_missing_spans(variant_id, variant)
        if not missing:
            valid_donor_ids.add(variant_id)
            continue
        if not bool(variant.get("meaning_preserved")):
            continue

        donor_id = min(
            valid_donor_ids,
            key=lambda candidate: (
                abs(order[candidate] - order[variant_id]),
                order[candidate],
            ),
        )
        donor = by_id[donor_id]
        original_spoken = _normalise_space(str(variant.get("spoken_text") or ""))
        variant["spoken_text"] = str(donor.get("spoken_text") or "")
        variant["estimated_duration_ms"] = int(
            donor.get("estimated_duration_ms") or 0
        )
        variant["preserved_english_spans"] = list(
            donor.get("preserved_english_spans") or []
        )
        variant["omitted_optional_details"] = list(
            donor.get("omitted_optional_details") or []
        )
        valid_donor_ids.add(variant_id)
        repairs.append(
            {
                "code": "protected_identity_compression_floor_fallback",
                "missing_required_spans": missing,
                "from_variant_id": donor_id,
                "to_variant_id": variant_id,
                "discarded_spoken_text": original_spoken,
                "replacement_policy": (
                    "reuse_nearest_sibling_preserving_all_protected_identities"
                ),
            }
        )
    return repairs


def _apply_undeclared_omission_compression_floor_fallback(
    variants: Sequence[dict[str, Any]],
    *,
    optional_details: Sequence[str],
) -> list[dict[str, Any]]:
    """Replace an unsafe shorter variant with the nearest richer model output.

    GPT can correctly shorten a sentence but describe the removed material in
    ``omitted_optional_details`` without declaring the same item at unit level.
    The validator must not silently bless that metadata disagreement because the
    omitted item may be required or protected.  Repeating the entire paid
    full-transcript request is also unnecessary when a richer model-returned
    variant is already available.

    For concise or compact only, reuse the nearest richer variant verbatim when
    undeclared omissions remain after the narrow source-grounded metadata repair.
    This never invents or edits target-language wording and cannot weaken
    required-fact or protected-identity checks.  A natural variant with an
    undeclared omission still fails closed because no richer output exists.
    """

    declared = {
        _normalise_space(str(value)).casefold()
        for value in optional_details
        if _normalise_space(str(value))
    }
    by_id = {
        str(item.get("variant_id") or ""): item
        for item in variants
        if isinstance(item, dict)
    }
    repairs: list[dict[str, Any]] = []
    donor: dict[str, Any] | None = None
    donor_id: str | None = None

    for variant_id in VARIANT_IDS:
        variant = by_id.get(variant_id)
        if variant is None:
            continue
        undeclared = [
            _normalise_space(str(value))
            for value in variant.get("omitted_optional_details") or []
            if _normalise_space(str(value))
            and _normalise_space(str(value)).casefold() not in declared
        ]
        if not undeclared:
            donor = variant
            donor_id = variant_id
            continue

        if (
            donor is None
            or donor_id is None
            or variant_id == "natural"
            or not bool(variant.get("meaning_preserved"))
            or not bool(donor.get("meaning_preserved"))
        ):
            # Fail closed during normal validation.  In particular, never mask
            # an undeclared omission in the natural translation.
            continue

        # The undeclared metadata may itself be a target-language paraphrase
        # rather than an exact source quotation. Do not approve, translate or
        # reinterpret it. Reject the entire shorter variant and reuse the
        # nearest richer model-returned sibling, including that sibling's clean
        # omission metadata. This cannot launder the questionable declaration:
        # none of the rejected variant survives.
        original_spoken = _normalise_space(str(variant.get("spoken_text") or ""))
        variant["spoken_text"] = str(donor.get("spoken_text") or "")
        variant["estimated_duration_ms"] = int(
            donor.get("estimated_duration_ms") or 0
        )
        variant["preserved_english_spans"] = list(
            donor.get("preserved_english_spans") or []
        )
        variant["omitted_optional_details"] = list(
            donor.get("omitted_optional_details") or []
        )
        repairs.append(
            {
                "code": "undeclared_omission_compression_floor_fallback",
                "from_variant_id": donor_id,
                "to_variant_id": variant_id,
                "undeclared_details": undeclared,
                "discarded_spoken_text": original_spoken,
                "replacement_policy": (
                    "reuse_nearest_richer_valid_variant_verbatim"
                ),
            }
        )
        donor = variant
        donor_id = variant_id

    return repairs


def normalize_multivariant_response(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore deterministic envelope and source-bound fields locally.

    Only generated translation fields are accepted from GPT. Immutable job
    metadata, source timing, source text, and variant identities are restored
    from the exact request. Unknown model fields are discarded rather than
    leaking into the strict installed schema. No model retry or repair request
    is performed.
    """

    if not isinstance(response, Mapping):
        raise MultivariantTranslationError(
            "Translation response must be a JSON object"
        )

    value: Mapping[str, Any] = response
    for wrapper_key in ("result", "response", "translation", "output"):
        nested = value.get(wrapper_key)
        if isinstance(nested, Mapping) and isinstance(nested.get("units"), list):
            value = nested
            break

    raw_units = value.get("units")
    if not isinstance(raw_units, list):
        raw_units = []

    expected_units = list(request.get("units") or [])
    expected_by_id = {str(item.get("unit_id") or ""): item for item in expected_units}
    normalized_by_id: dict[str, dict[str, Any]] = {}
    unexpected_ids: list[str] = []

    for raw_unit in raw_units:
        if not isinstance(raw_unit, Mapping):
            continue
        unit_id = str(raw_unit.get("unit_id") or "").strip()
        # Glossary entries and other non-unit fragments have no valid unit ID.
        if not unit_id:
            continue
        source = expected_by_id.get(unit_id)
        if source is None:
            unexpected_ids.append(unit_id)
            continue
        if unit_id in normalized_by_id:
            raise MultivariantTranslationError(
                f"Translation response returned duplicate unit {unit_id}"
            )

        raw_variants = raw_unit.get("variants")
        variants_by_id: dict[str, Mapping[str, Any]] = {}
        positional_variants: list[Mapping[str, Any]] = []
        if isinstance(raw_variants, list):
            positional_variants = [item for item in raw_variants if isinstance(item, Mapping)]
            for item in positional_variants:
                variant_id = str(
                    item.get("variant_id") or item.get("compression_level") or ""
                ).strip()
                if variant_id in VARIANT_IDS and variant_id not in variants_by_id:
                    variants_by_id[variant_id] = item
            # GPT can preserve order but omit the redundant ID.
            if len(positional_variants) == len(VARIANT_IDS):
                for variant_id, item in zip(VARIANT_IDS, positional_variants):
                    variants_by_id.setdefault(variant_id, item)

        targets = source.get("duration_targets")
        target_map = targets if isinstance(targets, Mapping) else {}
        normalized_variants: list[dict[str, Any]] = []
        discarded_preservation_claims: list[dict[str, str]] = []
        for variant_id in VARIANT_IDS:
            raw_variant = variants_by_id.get(variant_id)
            if raw_variant is None:
                continue
            variant: dict[str, Any] = {
                "variant_id": variant_id,
                "compression_level": variant_id,
            }
            expected_target = target_map.get(f"{variant_id}_target_duration_ms")
            if expected_target is not None:
                variant["target_duration_ms"] = int(expected_target)
            for field in (
                "spoken_text",
                "estimated_duration_ms",
                "omitted_optional_details",
                "meaning_preserved",
            ):
                if field in raw_variant:
                    variant[field] = raw_variant[field]

            # ``preserved_english_spans`` is descriptive model metadata, not
            # translated speech. GPT can copy a source noun into
            # this list even after translating it (for example ``Witness`` ->
            # ``ufakazi``). A false metadata claim must not invalidate an
            # otherwise complete and truthful one-call translation. Required
            # protected spans are still enforced independently by the strict
            # validator below, so pruning only unsupported *claims* cannot hide a
            # missing required name, brand, number, or reviewed identity alias.
            spoken_text = _normalise_space(str(variant.get("spoken_text") or ""))
            preserved: list[str] = []
            seen_preserved: set[str] = set()
            raw_preserved = raw_variant.get("preserved_english_spans")
            if isinstance(raw_preserved, list):
                for raw_span in raw_preserved:
                    span = _normalise_space(str(raw_span))
                    key = span.casefold()
                    if not span or key in seen_preserved:
                        continue
                    seen_preserved.add(key)
                    if _matched_protected_form(spoken_text, span) is None:
                        discarded_preservation_claims.append(
                            {"variant_id": variant_id, "span": span}
                        )
                        continue
                    preserved.append(span)
            variant["preserved_english_spans"] = preserved
            variant.setdefault("omitted_optional_details", [])
            normalized_variants.append(variant)

        normalized_unit: dict[str, Any] = {}
        for field in (
            "unit_id",
            "speaker_id",
            "start_ms",
            "end_ms",
            "available_duration_ms",
            "overlap_group",
            "source_text",
            "protected_spans",
        ):
            if field in source:
                normalized_unit[field] = source[field]
        required_facts = (
            list(raw_unit.get("required_facts"))
            if isinstance(raw_unit.get("required_facts"), list)
            else list(source.get("required_facts") or [])
        )
        optional_details = (
            list(raw_unit.get("optional_details"))
            if isinstance(raw_unit.get("optional_details"), list)
            else list(source.get("optional_details") or [])
        )
        promoted_optional_details: list[str] = []
        identity_specific_expansion_repairs: list[dict[str, Any]] = []
        identity_compression_floor_repairs = (
            _apply_protected_identity_compression_floor_fallback(
                normalized_variants,
                protected_spans=list(source.get("protected_spans") or []),
                target_locale=str(request.get("target_locale") or ""),
                source_text=str(source.get("source_text") or ""),
            )
        )
        normalized_variants_by_id = {
            str(item.get("variant_id") or ""): item
            for item in normalized_variants
        }
        compact_variant = normalized_variants_by_id.get("compact")
        if isinstance(compact_variant, Mapping):
            known_optional = {
                _normalise_space(str(value)).casefold()
                for value in optional_details
                if _normalise_space(str(value))
            }
            for raw_detail in compact_variant.get("omitted_optional_details") or []:
                detail = _normalise_space(str(raw_detail))
                detail_key = detail.casefold()
                if not detail or detail_key in known_optional:
                    continue
                if not _compact_optional_detail_is_source_grounded(
                    detail,
                    source_text=str(source.get("source_text") or ""),
                    required_facts=required_facts,
                    protected_spans=list(source.get("protected_spans") or []),
                    variants_by_id=normalized_variants_by_id,
                    target_locale=str(request.get("target_locale") or ""),
                ):
                    continue
                optional_details.append(detail)
                known_optional.add(detail_key)
                promoted_optional_details.append(detail)

        undeclared_omission_floor_repairs = (
            _apply_undeclared_omission_compression_floor_fallback(
                normalized_variants,
                optional_details=optional_details,
            )
        )

        normalized_unit["required_facts"] = required_facts
        normalized_unit["optional_details"] = optional_details
        normalized_unit["variants"] = normalized_variants
        normalized_unit["selection_status"] = "unselected"
        normalized_unit["selected_variant_id"] = None
        normalized_unit["warnings"] = (
            list(raw_unit.get("warnings"))
            if isinstance(raw_unit.get("warnings"), list)
            else []
        )
        normalized_unit["warnings"].extend(
            {
                "code": "discarded_false_preserved_english_span_claim",
                "variant_id": item["variant_id"],
                "span": item["span"],
            }
            for item in discarded_preservation_claims
        )
        normalized_unit["warnings"].extend(
            {
                "code": "promoted_source_grounded_compact_optional_detail",
                "variant_id": "compact",
                "detail": detail,
            }
            for detail in promoted_optional_details
        )
        normalized_unit["warnings"].extend(identity_specific_expansion_repairs)
        normalized_unit["warnings"].extend(identity_compression_floor_repairs)
        normalized_unit["warnings"].extend(undeclared_omission_floor_repairs)
        normalized_by_id[unit_id] = normalized_unit

    if unexpected_ids:
        raise MultivariantTranslationError(
            "Translation response returned unexpected units: "
            + ", ".join(sorted(set(unexpected_ids)))
        )

    ordered_units: list[dict[str, Any]] = []
    for source in expected_units:
        unit_id = str(source.get("unit_id") or "")
        if unit_id in normalized_by_id:
            ordered_units.append(normalized_by_id[unit_id])

    terminology: list[dict[str, Any]] = []
    raw_terminology = value.get("global_terminology")
    if isinstance(raw_terminology, list):
        for item in raw_terminology:
            if not isinstance(item, Mapping):
                continue
            if not str(item.get("source_term") or "").strip():
                continue
            normalized_term = {
                field: item[field]
                for field in (
                    "source_term",
                    "target_rendering",
                    "preserve_english",
                    "reason",
                )
                if field in item
            }
            terminology.append(normalized_term)

    locale = str(request.get("target_locale") or "")
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "source_language": str(request.get("source_language") or "en-ZA"),
        "target_language": str(
            request.get("target_language") or language_name(locale)
        ),
        "target_locale": locale,
        "translation_strategy": "contextual_code_switching",
        "units": ordered_units,
        "global_terminology": terminology,
        "warnings": (
            list(value.get("warnings"))
            if isinstance(value.get("warnings"), list)
            else []
        ),
    }

def render_system_prompt(target_locale: str) -> str:
    return SYSTEM_PROMPT.replace("{{TARGET_LANGUAGE}}", language_name(target_locale))


def render_user_prompt(request: Mapping[str, Any]) -> str:
    locale = str(request.get("target_locale") or "")
    return (
        USER_PROMPT_TEMPLATE.replace("{{TARGET_LANGUAGE}}", language_name(locale))
        .replace("{{TARGET_LOCALE}}", locale)
        .replace("{{PROMPT_VERSION}}", PROMPT_VERSION)
        .replace(
            "{{TRANSCRIPT_JSON}}",
            json.dumps(request, ensure_ascii=False, indent=2),
        )
    )


def build_job_translation_request(
    *,
    job_root: Path,
    job_id: str,
    target_locale: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact source-bound request used by cloud and manual translation.

    The deterministic dubbing units are preferred because their timing windows are
    what Azure must fit. The authoritative transcript hash is also bound into the
    request so an autocorrection change invalidates an old response even when unit
    IDs happen to remain stable.
    """

    transcript_path = job_root / "analysis" / "transcript_en.json"
    if not transcript_path.is_file():
        raise FileNotFoundError(transcript_path)
    transcript = read_json(transcript_path)
    dubbing_units_path = job_root / "dubbing" / "dubbing_units.json"
    timeline_path = dubbing_units_path if dubbing_units_path.is_file() else transcript_path
    timeline = read_json(timeline_path)
    bindings_path = job_root / "analysis" / "entity_bindings.json"
    entity_bindings = read_json(bindings_path) if bindings_path.is_file() else None
    context: dict[str, Any] = {}
    for name in ("context.json", "domain_classification.json"):
        path = job_root / "analysis" / name
        if path.is_file():
            context[path.stem] = read_json(path)

    request = build_translation_request(
        transcript=timeline,
        target_locale=target_locale,
        job_id=job_id,
        entity_bindings=entity_bindings,
        global_context=context,
    )
    request["source_transcript_sha256"] = _sha256_json(transcript)
    request["source_transcript_file_sha256"] = checksum(transcript_path)
    request["source_timeline_kind"] = (
        "dubbing_units" if timeline_path == dubbing_units_path else "transcript_segments"
    )
    request["source_timeline_sha256"] = _sha256_json(timeline)
    request["source_timeline_file_sha256"] = checksum(timeline_path)
    request["request_sha256"] = _sha256_json(
        {key: value for key, value in request.items() if key != "request_sha256"}
    )
    return request, {
        "transcript_path": transcript_path,
        "timeline_path": timeline_path,
        "bindings_path": bindings_path if bindings_path.is_file() else None,
    }


def export_translation_package(
    *,
    job_root: Path,
    job_id: str,
    target_locale: str,
    output_dir: Path,
) -> dict[str, Any]:
    request, source_paths = build_job_translation_request(
        job_root=job_root,
        job_id=job_id,
        target_locale=target_locale,
    )
    transcript_path = Path(source_paths["transcript_path"])
    timeline_path = Path(source_paths["timeline_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "translation_request.json"
    system_path = output_dir / "system_prompt.txt"
    user_path = output_dir / "user_prompt.txt"
    schema_path = output_dir / "response_schema.json"
    manifest_path = output_dir / "package_manifest.json"
    atomic_write_json(request_path, request)
    system_path.write_text(render_system_prompt(target_locale) + "\n", encoding="utf-8")
    user_path.write_text(render_user_prompt(request) + "\n", encoding="utf-8")
    atomic_write_json(schema_path, RESPONSE_SCHEMA)
    manifest = {
        "schema_version": "mathula.translation.prompt-package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "target_locale": target_locale,
        "prompt_version": PROMPT_VERSION,
        "source_transcript": str(transcript_path),
        "source_transcript_sha256": checksum(transcript_path),
        "source_timeline": str(timeline_path),
        "source_timeline_sha256": checksum(timeline_path),
        "request_path": str(request_path),
        "request_sha256": checksum(request_path),
        "system_prompt_path": str(system_path),
        "system_prompt_sha256": checksum(system_path),
        "user_prompt_path": str(user_path),
        "user_prompt_sha256": checksum(user_path),
        "response_schema_path": str(schema_path),
        "response_schema_sha256": checksum(schema_path),
    }
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _normalise_space(value: str) -> str:
    return " ".join(value.split())


def _proper_name_local_anchors(text: str) -> list[tuple[str, tuple[str, ...], float]]:
    """Return stable multiword name anchors and their relative source positions.

    Three or more title-cased words are deliberately required. This catches
    full personal and institutional names while avoiding ordinary sentence
    openings. Quotation marks inside a name (for example a nickname) do not
    break the anchor.
    """

    matches = list(_PROPER_NAME_TOKEN_RE.finditer(text))
    runs: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = []
    for match in matches:
        if current:
            gap = text[current[-1].end() : match.start()]
            if re.fullmatch(r"[\s\"'“”‘’]*", gap) is None:
                if len(current) >= 3:
                    runs.append(current)
                current = []
        current.append(match)
    if len(current) >= 3:
        runs.append(current)

    anchors: list[tuple[str, tuple[str, ...], float]] = []
    text_length = max(1, len(text))
    for run in runs:
        words = tuple(match.group(0) for match in run)
        anchors.append((" ".join(words), words, run[0].start() / text_length))
    return anchors


def _target_anchor_relative_position(
    text: str,
    anchor_words: Sequence[str],
) -> float | None:
    """Locate a source name in translated speech, allowing a Zulu noun prefix."""

    target_words = list(_WORD_TOKEN_RE.finditer(text))
    if not anchor_words or len(target_words) < len(anchor_words):
        return None
    folded_anchor = [word.casefold() for word in anchor_words]
    grammatical_prefixes = {
        "a", "e", "i", "o", "u", "ka", "ku", "kwa", "kuka", "laka",
        "luka", "na", "ne", "no", "nga", "ngo", "ngu", "se", "si",
    }
    for index in range(len(target_words) - len(anchor_words) + 1):
        candidate = [
            match.group(0).casefold()
            for match in target_words[index : index + len(anchor_words)]
        ]
        first = candidate[0]
        first_anchor = folded_anchor[0]
        first_matches = first == first_anchor
        if not first_matches and first.endswith(first_anchor):
            first_matches = first[: -len(first_anchor)] in grammatical_prefixes
        if first_matches and candidate[1:] == folded_anchor[1:]:
            return target_words[index].start() / max(1, len(text))
    return None


def _local_anchor_drift(
    source_text: str,
    spoken_text: str,
) -> tuple[str, float, float] | None:
    for label, words, source_position in _proper_name_local_anchors(source_text):
        target_position = _target_anchor_relative_position(spoken_text, words)
        if target_position is None:
            continue
        if abs(target_position - source_position) > _MAX_LOCAL_ANCHOR_DRIFT:
            return label, source_position, target_position
    return None


def _span_pattern(span: str) -> re.Pattern[str]:
    escaped = re.escape(span.strip())
    # Match only at the start of a lexical token, optionally after a small
    # grammatical prefix used by isiZulu/Sepedi. This accepts eThohoyandou,
    # i-EFF, uDr Levy Ndou and noDr Levy Ndou, but cannot infer Ando from the
    # middle of Thohoyandou.
    prefix = r"(?:(?:kuka|uno|ise|kwi|kwe|kwa|nga|ngo|ngu|no|ne|na|ku|ka|go|yi|ye|we|le|la|lo|se|sa|so|si|u|e|i|o|a)-?)?"
    return re.compile(rf"(?<![\w]){prefix}{escaped}(?![\w])", re.IGNORECASE)


def _contains_protected_span(text: str, span: str) -> bool:
    return bool(_span_pattern(span).search(text))


@lru_cache(maxsize=1)
def _default_entity_registry() -> EntityRegistry | None:
    """Load reviewed entity aliases once and fail closed when unavailable."""

    try:
        return EntityRegistry()
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None


@lru_cache(maxsize=1024)
def _approved_protected_forms(span: str) -> tuple[str, ...]:
    canonical = str(span or "").strip()
    if not canonical:
        return ()
    registry = _default_entity_registry()
    if registry is None:
        return (canonical,)
    forms = registry.approved_identity_forms(canonical)
    return forms or (canonical,)


def _matched_form_from_approved_forms(
    text: str,
    approved_forms: Sequence[str],
) -> str | None:
    for form in approved_forms:
        if _contains_protected_span(text, form):
            return form
    return None


_ISIZULU_IDENTITY_PREFIXES = tuple(
    sorted(
        {
            "njengelika", "njengoka", "njengo", "njenge", "njenga",
            "ngelika", "ngoka", "buka", "lika", "luka", "sika",
            "kuka", "kwi", "kwe", "kwa", "nga", "nge", "ngo", "ngu", "ise", "uno",
            "yi", "ye", "we", "ne", "no", "na", "ku", "ka", "le", "la",
            "lo", "se", "sa", "so", "si", "u", "e", "i", "o", "a",
        },
        key=len,
        reverse=True,
    )
)


def _identity_words(text: str) -> list[str]:
    """Return Unicode lexical words used by reviewed identity matching."""

    return re.findall(r"[^\W_]+", str(text or ""), flags=re.UNICODE)


def _isizulu_identity_word_matches(actual: str, expected: str) -> bool:
    """Match an identity word through reviewed isiZulu morphology only.

    Protected identities frequently occur with attached noun-class or locative
    morphology: ``iKhomishini``, ``eKhomishini``, ``kweKhomishini`` and
    ``eKhomishinini`` all retain the reviewed lexical identity ``Khomishini``.
    Likewise ``kaMadlanga`` retains ``Madlanga``.  This matcher is deliberately
    token-local; it cannot infer an identity from a substring inside an
    unrelated word.
    """

    actual_key = str(actual or "").casefold()
    expected_key = str(expected or "").casefold()
    if not actual_key or not expected_key:
        return False

    accepted_stems = {expected_key}
    # Some speakers/writers use the productive locative ``-ini`` in addition
    # to the locative prefix.  For ``Khomishini`` that surfaces as
    # ``eKhomishinini``.  Restrict this to the reviewed common noun so names and
    # brands cannot acquire arbitrary suffixes.
    if expected_key == "khomishini":
        accepted_stems.add(f"{expected_key}ni")
    elif expected_key == "khomishana":
        # ``iKhomishana`` can take the locative form ``eKhomishaneni``.
        accepted_stems.add("khomishaneni")
    elif expected_key == "khomishani":
        # ``iKhomishani`` is common in broadcast isiZulu and takes the
        # productive locative form ``eKhomishanini``.
        accepted_stems.add("khomishanini")

    if actual_key in accepted_stems:
        return True
    # Strip at most three reviewed isiZulu morphology prefixes.  This covers
    # forms such as ``njengoWitness``, ``likaMokwele`` and
    # ``njengelikaMokwele`` without permitting arbitrary substring matches.
    frontier = {actual_key}
    seen = {actual_key}
    for _ in range(3):
        next_frontier: set[str] = set()
        for candidate in frontier:
            for prefix in _ISIZULU_IDENTITY_PREFIXES:
                if not candidate.startswith(prefix):
                    continue
                remainder = candidate[len(prefix):]
                if not remainder or remainder in seen:
                    continue
                if remainder in accepted_stems:
                    return True
                seen.add(remainder)
                next_frontier.add(remainder)
        if not next_frontier:
            break
        frontier = next_frontier
    return False


def _matched_reviewed_identity_token_group(
    text: str,
    *,
    span: str,
    target_locale: str,
) -> str | None:
    """Match a reviewed identity whose natural target words are non-contiguous.

    A truthful isiZulu rendering may say ``iKhomishini Yophenyo eholwa
    uJustice Madlanga`` instead of using the contiguous phrase
    ``iKhomishini kaMadlanga``.  The locale registry can therefore declare an
    ordered token group and a small maximum gap.  Both identity anchors remain
    mandatory, so a named commission still cannot collapse to generic
    ``iKhomishini``.
    """

    locale_profile = _identity_alias_profile(span, target_locale=target_locale)
    if locale_profile is None:
        return None
    raw_groups = locale_profile.get("specific_token_groups")
    if not isinstance(raw_groups, list):
        return None

    words = _identity_words(text)
    for raw_group in raw_groups:
        if not isinstance(raw_group, Mapping):
            continue
        tokens = [
            _normalise_space(str(value))
            for value in raw_group.get("tokens") or []
            if _normalise_space(str(value))
        ]
        if len(tokens) < 2:
            continue
        try:
            max_gap = max(0, min(20, int(raw_group.get("max_gap_tokens", 8))))
        except (TypeError, ValueError):
            max_gap = 8

        previous_index = -1
        matched = True
        for token in tokens:
            start = previous_index + 1
            stop = len(words) if previous_index < 0 else min(
                len(words), previous_index + max_gap + 2
            )
            found_index = next(
                (
                    index
                    for index in range(start, stop)
                    if _isizulu_identity_word_matches(words[index], token)
                ),
                None,
            )
            if found_index is None:
                matched = False
                break
            previous_index = found_index
        if matched:
            return str(raw_group.get("label") or " … ".join(tokens))
    return None


def _matched_reviewed_identity_sequence(
    text: str,
    approved_forms: Sequence[str],
    *,
    target_locale: str,
) -> str | None:
    """Match reviewed aliases through controlled target-language morphology.

    English-style word boundaries miss natural forms such as
    ``njengoWitness F``, ``likaMokwele`` and ``eKhomishanini``.  This matcher
    compares complete lexical tokens in order and applies only the reviewed
    isiZulu prefix/suffix rules above.
    """

    if target_locale != "zu-ZA":
        return None
    actual_words = _identity_words(text)
    for form in approved_forms:
        expected_words = _identity_words(form)
        if not expected_words or len(expected_words) > len(actual_words):
            continue
        width = len(expected_words)
        for start in range(0, len(actual_words) - width + 1):
            window = actual_words[start:start + width]
            if all(
                _isizulu_identity_word_matches(actual, expected)
                for actual, expected in zip(window, expected_words, strict=True)
            ):
                return form
    return None


def _matched_translation_identity_form(
    text: str,
    approved_forms: Sequence[str],
    *,
    span: str,
    target_locale: str,
) -> str | None:
    direct = _matched_form_from_approved_forms(text, approved_forms)
    if direct is not None:
        return direct
    grouped = _matched_reviewed_identity_token_group(
        text, span=span, target_locale=target_locale
    )
    if grouped is not None:
        return grouped
    return _matched_reviewed_identity_sequence(
        text, approved_forms, target_locale=target_locale
    )


def _matched_protected_form(text: str, span: str) -> str | None:
    approved_forms = _approved_protected_forms(span)
    direct = _matched_form_from_approved_forms(text, approved_forms)
    if direct is not None:
        return direct
    return _matched_reviewed_identity_sequence(
        text, approved_forms, target_locale="zu-ZA"
    )


TRANSLATION_IDENTITY_ALIASES_SCHEMA = "mathula-translation-identity-aliases-v1"


@lru_cache(maxsize=1)
def _translation_identity_alias_profiles() -> tuple[dict[str, Any], ...]:
    """Load reviewed target-language identity forms used by validation.

    The source entity registry intentionally contains source/publication identity
    aliases. This supplementary registry contains reviewed target-language
    renderings such as ``iKhomishini kaMadlanga``. Missing or malformed optional
    configuration fails closed by returning no additional forms.
    """

    path = Path(__file__).resolve().parents[2] / "config" / "translation_identity_aliases.json"
    try:
        payload = read_json(path)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, Mapping):
        return ()
    if payload.get("schema_version") != TRANSLATION_IDENTITY_ALIASES_SCHEMA:
        return ()
    entities = payload.get("entities")
    if not isinstance(entities, list):
        return ()
    return tuple(item for item in entities if isinstance(item, dict))


def _identity_alias_profile(
    span: str,
    *,
    target_locale: str,
) -> Mapping[str, Any] | None:
    approved_source_forms = _approved_protected_forms(span)
    source_keys = {str(form).strip().casefold() for form in approved_source_forms if str(form).strip()}
    supplied_key = str(span or "").strip().casefold()
    for profile in _translation_identity_alias_profiles():
        canonical = str(profile.get("canonical_identity") or "").strip()
        configured_source_forms = [
            str(form).strip()
            for form in profile.get("source_forms") or []
            if str(form).strip()
        ]
        configured_keys = {canonical.casefold(), *(form.casefold() for form in configured_source_forms)}
        if supplied_key not in configured_keys and not source_keys.intersection(configured_keys):
            continue
        locales = profile.get("locales")
        if not isinstance(locales, Mapping):
            return None
        locale_profile = locales.get(target_locale)
        return locale_profile if isinstance(locale_profile, Mapping) else None
    return None


def _protected_span_policy(
    span: str,
    *,
    target_locale: str,
) -> str:
    """Classify source protection as lexical or discourse-level.

    Names, acronyms, brands, quoted code-switching, and unprofiled spans remain
    strict per-variant lexical requirements. Reviewed institutions that expose
    generic target forms are discourse identities: natural target-language
    anaphora may refer to them without repeating the complete formal name in
    every small timing unit.
    """

    locale_profile = _identity_alias_profile(span, target_locale=target_locale)
    if locale_profile is None:
        return "hard_identity"
    generic_forms = [
        _normalise_space(str(value))
        for value in locale_profile.get("generic_forms") or []
        if _normalise_space(str(value))
    ]
    generic_source_forms = [
        _normalise_space(str(value))
        for value in locale_profile.get("generic_source_forms") or []
        if _normalise_space(str(value))
    ]
    if generic_forms or generic_source_forms:
        return "contextual_identity"
    return "hard_identity"


def _source_text_grounds_protected_identity(
    span: str,
    *,
    source_text: str,
) -> bool:
    """Return whether this exact source unit contains a reviewed source form."""

    return any(
        _contains_protected_span(source_text, form)
        for form in _approved_protected_forms(span)
        if str(form).strip()
    )


def _unit_protected_span_policy(
    span: str,
    *,
    target_locale: str,
    source_text: str,
) -> str:
    """Classify a protected identity within its source-unit boundary.

    A binding can carry useful context from the registry, but it must not force
    a brand or person name into translated speech when no reviewed source form
    occurs in this unit.
    """

    policy = _protected_span_policy(span, target_locale=target_locale)
    if policy != "hard_identity":
        return policy
    if _source_text_grounds_protected_identity(span, source_text=source_text):
        return policy
    return "context_bound_identity"


def _contextual_translation_identity_forms(
    span: str,
    *,
    target_locale: str,
) -> tuple[str, ...]:
    """Return reviewed generic/anaphoric renderings for a discourse identity."""

    locale_profile = _identity_alias_profile(span, target_locale=target_locale)
    if locale_profile is None:
        return ()
    values = [
        *list(locale_profile.get("generic_forms") or []),
        *list(locale_profile.get("target_anchor_forms") or []),
    ]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = _normalise_space(str(value))
        key = candidate.casefold()
        if not candidate or key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return tuple(result)


def _source_uses_generic_translation_identity(
    span: str,
    *,
    target_locale: str,
    source_text: str,
) -> bool:
    """Return whether this unit used only a reviewed generic source alias."""

    locale_profile = _identity_alias_profile(span, target_locale=target_locale)
    if locale_profile is None:
        return False
    base_forms = list(_approved_protected_forms(span))
    generic_source_forms = [
        _normalise_space(str(value))
        for value in locale_profile.get("generic_source_forms") or []
        if _normalise_space(str(value))
    ]
    generic_keys = {value.casefold() for value in generic_source_forms}
    specific_source_forms = [
        value for value in base_forms if value.casefold() not in generic_keys
    ]
    source_has_specific_identity = any(
        _contains_protected_span(source_text, value)
        for value in specific_source_forms
    )
    return (
        not source_has_specific_identity
        and any(
            _contains_protected_span(source_text, value)
            for value in generic_source_forms
        )
    )


def _approved_translation_identity_forms(
    span: str,
    *,
    target_locale: str,
    source_text: str,
) -> tuple[str, ...]:
    """Return source aliases plus reviewed target-language identity renderings.

    Specific localized forms are always allowed for the resolved identity. A
    generic localized form such as ``iKhomishini`` is allowed only when the
    source unit itself used a generic source alias such as ``the Commission``.
    This prevents a fully named institution from collapsing to an ambiguous
    generic noun while allowing natural anaphora to be translated.
    """

    registry = _default_entity_registry()
    if registry is None:
        base_forms = list(_approved_protected_forms(span))
    else:
        base_forms = list(
            registry.approved_identity_forms(span, source_text=source_text)
        )
    locale_profile = _identity_alias_profile(span, target_locale=target_locale)
    if locale_profile is None:
        return tuple(base_forms)

    specific_forms = [
        _normalise_space(str(value))
        for value in locale_profile.get("specific_forms") or []
        if _normalise_space(str(value))
    ]
    generic_source_forms = [
        _normalise_space(str(value))
        for value in locale_profile.get("generic_source_forms") or []
        if _normalise_space(str(value))
    ]
    generic_forms = [
        _normalise_space(str(value))
        for value in locale_profile.get("generic_forms") or []
        if _normalise_space(str(value))
    ]
    source_anchor_aliases = [
        _normalise_space(str(value))
        for value in locale_profile.get("source_anchor_aliases") or []
        if _normalise_space(str(value))
    ]
    target_anchor_forms = [
        _normalise_space(str(value))
        for value in locale_profile.get("target_anchor_forms") or []
        if _normalise_space(str(value))
    ]

    generic_keys = {value.casefold() for value in generic_source_forms}
    specific_source_forms = [
        value for value in base_forms if value.casefold() not in generic_keys
    ]
    source_has_specific_identity = any(
        _contains_protected_span(source_text, value)
        for value in specific_source_forms
    )
    source_uses_generic_identity = (
        not source_has_specific_identity
        and any(
            _contains_protected_span(source_text, value)
            for value in generic_source_forms
        )
    )
    # Some transcripts use a reviewed proper-name shorthand for an institution,
    # for example "before Madlanga" to mean an appearance before the Madlanga
    # Commission.  When the source itself uses that standalone shorthand, allow
    # the same proper-name anchor in the target (including normal isiZulu prefixes
    # such as ``uMadlanga`` or ``kukaMadlanga``).  Do not enable this escape hatch
    # when the source used the full institutional name: a fully named commission
    # must still remain a commission in translation.
    source_uses_anchor_identity = (
        not source_has_specific_identity
        and any(
            _contains_protected_span(source_text, value)
            for value in source_anchor_aliases
        )
    )

    values = [*base_forms, *specific_forms]
    if source_uses_generic_identity:
        values.extend(generic_forms)
    if source_uses_anchor_identity:
        values.extend(target_anchor_forms)
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        candidate = _normalise_space(str(value))
        key = candidate.casefold()
        if not candidate or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return tuple(unique)


def identity_requirements_for_unit(
    unit: Mapping[str, Any],
    *,
    target_locale: str,
) -> list[dict[str, Any]]:
    """Return the exact reviewed identity contract for one source unit.

    The semantic-chunk coordinator uses this public, read-only view when a paid
    chunk candidate fails protected-identity validation.  Supplying the same
    accepted forms used by the validator keeps a targeted repair call aligned
    with the local contract instead of asking the model to infer aliases.
    """

    source_text = _normalise_space(str(unit.get("source_text") or ""))
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_span in unit.get("protected_spans") or []:
        span = _normalise_space(str(raw_span))
        key = span.casefold()
        if not span or key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "required_span": span,
                "policy": _unit_protected_span_policy(
                    span,
                    target_locale=target_locale,
                    source_text=source_text,
                ),
                "accepted_forms": list(
                    _approved_translation_identity_forms(
                        span,
                        target_locale=target_locale,
                        source_text=source_text,
                    )
                ),
            }
        )
    return result


def restore_missing_hard_identities_in_unit(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    unit_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Restore reviewed hard-identity aliases without another model call.

    This recovery is deliberately narrower than translation. It may only add an
    alias already accepted by the normal validator, only to the named failed
    unit, and it re-runs the complete source-bound validator before returning.
    The shortest acronym-like accepted form is preferred to minimise timing
    pressure (for example ``IDAC``).
    """

    candidate = copy.deepcopy(dict(response))
    source_units = [
        item
        for item in request.get("units") or []
        if isinstance(item, Mapping)
        and str(item.get("unit_id") or "") == unit_id
    ]
    translated_units = [
        item
        for item in candidate.get("units") or []
        if isinstance(item, dict)
        and str(item.get("unit_id") or "") == unit_id
    ]
    if len(source_units) != 1 or len(translated_units) != 1:
        raise MultivariantTranslationError(
            f"Deterministic identity recovery could not resolve unique unit {unit_id}"
        )

    source_unit = source_units[0]
    translated_unit = translated_units[0]
    target_locale = str(request.get("target_locale") or "")
    requirements = identity_requirements_for_unit(
        source_unit,
        target_locale=target_locale,
    )
    repairs: list[dict[str, Any]] = []
    for requirement in requirements:
        if requirement.get("policy") != "hard_identity":
            continue
        span = _normalise_space(str(requirement.get("required_span") or ""))
        accepted_forms = [
            _normalise_space(str(value))
            for value in requirement.get("accepted_forms") or []
            if _normalise_space(str(value))
        ]
        if not span or not accepted_forms:
            continue
        acronym_forms = [
            value
            for value in accepted_forms
            if " " not in value
            and len(value) <= 12
            and re.fullmatch(r"[A-Z0-9][A-Z0-9&./'-]*", value) is not None
        ]
        selected_form = min(
            acronym_forms or accepted_forms,
            key=lambda value: (len(value), value.casefold()),
        )
        for variant in translated_unit.get("variants") or []:
            if not isinstance(variant, dict):
                continue
            spoken = _normalise_space(str(variant.get("spoken_text") or ""))
            if not spoken:
                continue
            if (
                _matched_translation_identity_form(
                    spoken,
                    accepted_forms,
                    span=span,
                    target_locale=target_locale,
                )
                is not None
            ):
                continue
            variant["spoken_text"] = _normalise_space(
                f"{selected_form}, {spoken}"
            )
            old_duration = int(variant.get("estimated_duration_ms") or 0)
            variant["estimated_duration_ms"] = old_duration + max(
                220,
                len(selected_form) * 55,
            )
            preserved = [
                _normalise_space(str(value))
                for value in variant.get("preserved_english_spans") or []
                if _normalise_space(str(value))
            ]
            if selected_form.casefold() not in {
                value.casefold() for value in preserved
            }:
                preserved.append(selected_form)
            variant["preserved_english_spans"] = preserved
            repairs.append(
                {
                    "code": "reviewed_hard_identity_alias_restored_locally",
                    "unit_id": unit_id,
                    "variant_id": str(variant.get("variant_id") or ""),
                    "required_span": span,
                    "inserted_accepted_form": selected_form,
                    "replacement_policy": (
                        "prepend_shortest_reviewed_acronym_or_alias"
                    ),
                }
            )

    if not repairs:
        raise MultivariantTranslationError(
            f"Deterministic identity recovery made no change to unit {unit_id}"
        )
    warnings = list(translated_unit.get("warnings") or [])
    warnings.extend(repairs)
    translated_unit["warnings"] = warnings
    normalized = normalize_multivariant_response(candidate, request=request)
    try:
        validation = validate_multivariant_response(normalized, request=request)
    except MultivariantTranslationError as exc:
        # Preserve the successful mutation when full-chunk validation advances
        # to another unit. The coordinator can then repair that next unit
        # locally instead of misreporting this mutation as a failure and
        # issuing a provider request for the already-fixed first unit.
        exc.details = {
            "validation_stage": "validator",
            "validation_error": f"MultivariantTranslationError: {exc}",
            "candidate_output": normalized,
            "local_identity_repairs": repairs,
            "local_identity_repair_advanced_to_next_failure": True,
        }
        raise
    return normalized, validation, repairs



def _matched_compact_identity_substitution(
    text: str,
    *,
    span: str,
    target_locale: str,
    source_text: str,
) -> dict[str, str] | None:
    """Match a narrowly reviewed compact substitute for a protected identity.

    Some source sentences name an institution and then immediately identify a
    more specific office or direct recipient.  A compact delivery may retain
    only that more specific identity without changing the fact being conveyed.
    This is never inferred generically: every allowed source/target relationship
    must be declared in ``translation_identity_aliases.json`` and all declared
    source anchors must occur in the same source unit.
    """

    locale_profile = _identity_alias_profile(span, target_locale=target_locale)
    if locale_profile is None:
        return None
    raw_rules = locale_profile.get("compact_substitutions")
    if not isinstance(raw_rules, list):
        return None

    for raw_rule in raw_rules:
        if not isinstance(raw_rule, Mapping):
            continue
        source_requires = [
            _normalise_space(str(value))
            for value in raw_rule.get("source_requires_all") or []
            if _normalise_space(str(value))
        ]
        target_forms = [
            _normalise_space(str(value))
            for value in raw_rule.get("target_forms") or []
            if _normalise_space(str(value))
        ]
        if not source_requires or not target_forms:
            continue
        if not all(
            _contains_protected_span(source_text, value)
            for value in source_requires
        ):
            continue

        matched = _matched_form_from_approved_forms(text, target_forms)
        if matched is None:
            matched = _matched_reviewed_identity_sequence(
                text, target_forms, target_locale=target_locale
            )
        if matched is None:
            continue
        return {
            "matched_form": matched,
            "label": str(raw_rule.get("label") or matched),
            "reason": str(
                raw_rule.get("reason")
                or "reviewed compact identity substitution"
            ),
        }
    return None

def validate_multivariant_response(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    require_all_meanings_preserved: bool = True,
) -> dict[str, Any]:
    try:
        jsonschema.validate(response, RESPONSE_SCHEMA)
    except jsonschema.ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path)
        raise MultivariantTranslationError(
            f"Translation response schema validation failed at {path or '<root>'}: {exc.message}"
        ) from exc

    expected_locale = str(request.get("target_locale") or "")
    if response.get("target_locale") != expected_locale:
        raise MultivariantTranslationError(
            f"Target locale mismatch: expected {expected_locale}, got {response.get('target_locale')}"
        )
    expected_language = language_name(expected_locale)
    if response.get("target_language") != expected_language:
        raise MultivariantTranslationError(
            f"Target language mismatch: expected {expected_language}, "
            f"got {response.get('target_language')}"
        )
    expected_source_language = str(request.get("source_language") or "en-ZA")
    if response.get("source_language") != expected_source_language:
        raise MultivariantTranslationError(
            f"Source language mismatch: expected {expected_source_language}, "
            f"got {response.get('source_language')}"
        )
    expected_units = list(request.get("units") or [])
    actual_units = list(response.get("units") or [])
    expected_ids = [str(item["unit_id"]) for item in expected_units]
    actual_ids = [str(item["unit_id"]) for item in actual_units]
    if actual_ids != expected_ids:
        raise MultivariantTranslationError(
            "Translated unit IDs/order do not exactly match the exported request"
        )

    validation_units: list[dict[str, Any]] = []
    for source, translated in zip(expected_units, actual_units, strict=True):
        unit_id = str(source["unit_id"])
        immutable_pairs = {
            "speaker_id": source["speaker_id"],
            "start_ms": source["start_ms"],
            "end_ms": source["end_ms"],
            "available_duration_ms": source["available_duration_ms"],
            "overlap_group": source.get("overlap_group"),
            "source_text": source.get("source_text", ""),
        }
        for field, expected in immutable_pairs.items():
            actual = translated.get(field)
            if field == "source_text":
                equal = _normalise_space(str(actual)) == _normalise_space(str(expected))
            else:
                equal = actual == expected
            if not equal:
                raise MultivariantTranslationError(
                    f"Unit {unit_id} changed immutable field {field}: "
                    f"expected {expected!r}, got {actual!r}"
                )

        variants = list(translated["variants"])
        ids = [str(item["variant_id"]) for item in variants]
        levels = [str(item["compression_level"]) for item in variants]
        if ids != list(VARIANT_IDS) or levels != list(VARIANT_IDS):
            raise MultivariantTranslationError(
                f"Unit {unit_id} variants must be natural, concise, compact in that order"
            )
        targets = source.get("duration_targets") or {}
        expected_targets = [
            int(targets[f"{variant}_target_duration_ms"])
            for variant in VARIANT_IDS
        ]
        actual_targets = [int(item["target_duration_ms"]) for item in variants]
        if actual_targets != expected_targets:
            raise MultivariantTranslationError(
                f"Unit {unit_id} changed duration targets: expected {expected_targets}, "
                f"got {actual_targets}"
            )
        if not (actual_targets[0] >= actual_targets[1] >= actual_targets[2] > 0):
            raise MultivariantTranslationError(
                f"Unit {unit_id} has invalid duration target ordering"
            )

        required = [str(value) for value in translated.get("required_facts", [])]
        optional = [str(value) for value in translated.get("optional_details", [])]
        optional_set = set(optional)
        variants_by_id_for_validation = {
            str(item.get("variant_id") or ""): item
            for item in variants
            if isinstance(item, Mapping)
        }
        validation_promoted_optional_details: list[str] = []
        protected = list(
            dict.fromkeys(
                [str(value) for value in source.get("protected_spans", [])]
                + [str(value) for value in translated.get("protected_spans", [])]
            )
        )
        variant_audit: list[dict[str, Any]] = []
        contextual_identity_reviews: list[dict[str, Any]] = []
        source_text = str(source.get("source_text") or "")
        evidence_by_variant: dict[str, list[dict[str, Any]]] = {}
        for evidence_variant in variants:
            evidence_variant_id = str(evidence_variant["variant_id"])
            evidence_spoken = _normalise_space(str(evidence_variant["spoken_text"]))
            evidence_items: list[dict[str, Any]] = []
            for span in protected:
                if not span:
                    continue
                policy = _unit_protected_span_policy(
                    span,
                    target_locale=expected_locale,
                    source_text=source_text,
                )
                approved_forms = _approved_translation_identity_forms(
                    span,
                    target_locale=expected_locale,
                    source_text=source_text,
                )
                matched_form = _matched_translation_identity_form(
                    evidence_spoken,
                    approved_forms,
                    span=span,
                    target_locale=expected_locale,
                )
                contextual_forms = (
                    _contextual_translation_identity_forms(
                        span,
                        target_locale=expected_locale,
                    )
                    if policy == "contextual_identity"
                    else ()
                )
                contextual_match = (
                    _matched_translation_identity_form(
                        evidence_spoken,
                        contextual_forms,
                        span=span,
                        target_locale=expected_locale,
                    )
                    if matched_form is None and contextual_forms
                    else None
                )
                evidence_items.append(
                    {
                        "required_span": span,
                        "policy": policy,
                        "matched_form": matched_form,
                        "contextual_match": contextual_match,
                        "approved_forms": list(approved_forms),
                        "contextual_forms": list(contextual_forms),
                        "evidence_status": (
                            "specific_identity"
                            if matched_form is not None
                            else "generic_contextual_reference"
                            if contextual_match is not None
                            else "contextual_reference_unverified"
                            if policy
                            in {"contextual_identity", "context_bound_identity"}
                            else "missing_hard_identity"
                        ),
                    }
                )
            evidence_by_variant[evidence_variant_id] = evidence_items

        for variant in variants:
            variant_id = str(variant["variant_id"])
            spoken = _normalise_space(str(variant["spoken_text"]))
            if require_all_meanings_preserved and not bool(variant["meaning_preserved"]):
                raise MultivariantTranslationError(
                    f"Unit {unit_id} variant {variant_id} does not preserve required meaning"
                )
            anchor_drift = _local_anchor_drift(source_text, spoken)
            if anchor_drift is not None:
                anchor, source_position, target_position = anchor_drift
                raise MultivariantTranslationError(
                    f"Unit {unit_id} variant {variant_id} moves local named anchor "
                    f"{anchor!r} from relative position {source_position:.2f} to "
                    f"{target_position:.2f}; preserve its local source progression"
                )
            invalid_omissions: list[str] = []
            for raw_detail in variant.get("omitted_optional_details", []):
                detail = _normalise_space(str(raw_detail))
                if not detail or detail in optional_set:
                    continue
                # Recovery and installation must not depend exclusively on the
                # earlier normalization side effect. Re-evaluate the same strict
                # source-grounded compact rule at the validation boundary. This
                # is intentionally compact-only and cannot approve required or
                # protected content.
                if (
                    variant_id == "compact"
                    and _compact_optional_detail_is_source_grounded(
                        detail,
                        source_text=source_text,
                        required_facts=required,
                        protected_spans=protected,
                        variants_by_id=variants_by_id_for_validation,
                        target_locale=expected_locale,
                    )
                ):
                    optional.append(detail)
                    optional_set.add(detail)
                    validation_promoted_optional_details.append(detail)
                    continue
                invalid_omissions.append(detail)
            if invalid_omissions:
                raise MultivariantTranslationError(
                    f"Unit {unit_id} variant {variant_id} omits undeclared details: "
                    f"{invalid_omissions}"
                )
            protected_evidence = evidence_by_variant[variant_id]
            missing_spans = [
                item for item in protected_evidence if item["matched_form"] is None
            ]
            unresolved_missing_spans: list[dict[str, Any]] = []
            for item in missing_spans:
                span = str(item["required_span"])
                if item.get("policy") in {
                    "contextual_identity",
                    "context_bound_identity",
                }:
                    if item.get("contextual_match") is not None:
                        item["omission_policy"] = (
                            "reviewed_generic_contextual_identity_reference"
                        )
                    elif item.get("policy") == "context_bound_identity":
                        item["omission_policy"] = (
                            "context_binding_not_lexically_grounded_in_source_unit"
                        )
                    else:
                        # Entity binding remains authoritative metadata. Spoken
                        # language may use anaphora or omit a repeated formal
                        # institution name inside a small timing unit. Keep the
                        # unit, surface it for human review, and never rewrite
                        # the translated speech locally.
                        item["omission_policy"] = (
                            "discourse_identity_bound_in_metadata_not_lexically_verified"
                        )
                    contextual_identity_reviews.append(
                        {
                            "variant_id": variant_id,
                            "required_span": span,
                            "matched_contextual_form": item.get("contextual_match"),
                            "review_policy": item["omission_policy"],
                        }
                    )
                    continue

                longer_variants_preserve = all(
                    any(
                        evidence["required_span"] == span
                        and evidence["matched_form"] is not None
                        for evidence in evidence_by_variant[longer_id]
                    )
                    for longer_id in ("natural", "concise")
                )
                compact_substitution = None
                if variant_id == "compact" and longer_variants_preserve:
                    compact_substitution = _matched_compact_identity_substitution(
                        spoken,
                        span=span,
                        target_locale=expected_locale,
                        source_text=source_text,
                    )
                if compact_substitution is not None:
                    item["omission_policy"] = (
                        "compact_reviewed_identity_substitution_with_longer_variants_preserved"
                    )
                    item["substitute_form"] = compact_substitution["matched_form"]
                    item["substitution_label"] = compact_substitution["label"]
                    item["substitution_reason"] = compact_substitution["reason"]
                    continue
                if (
                    variant_id == "compact"
                    and longer_variants_preserve
                    and _source_uses_generic_translation_identity(
                        span,
                        target_locale=expected_locale,
                        source_text=source_text,
                    )
                ):
                    item["omission_policy"] = (
                        "compact_generic_identity_omission_with_longer_variants_preserved"
                    )
                    continue
                unresolved_missing_spans.append(item)
            if unresolved_missing_spans:
                descriptions = [
                    (
                        f"{item['required_span']} "
                        f"(accepted forms: {', '.join(item['approved_forms'])})"
                    )
                    for item in unresolved_missing_spans
                ]
                raise MultivariantTranslationError(
                    f"Unit {unit_id} variant {variant_id} is missing protected spans "
                    f"or approved identity aliases: {descriptions}"
                )
            false_preservation_claims = [
                str(span)
                for span in variant.get("preserved_english_spans", [])
                if str(span).strip() and _matched_protected_form(spoken, str(span)) is None
            ]
            if false_preservation_claims:
                raise MultivariantTranslationError(
                    f"Unit {unit_id} variant {variant_id} claims preserved English "
                    f"spans that are absent from spoken_text: {false_preservation_claims}"
                )
            variant_audit.append(
                {
                    "variant_id": variant_id,
                    "spoken_characters": len(spoken),
                    "spoken_words": len(spoken.split()),
                    "estimated_duration_ms": int(variant["estimated_duration_ms"]),
                    "target_duration_ms": int(variant["target_duration_ms"]),
                    "protected_spans_preserved": protected,
                    "protected_span_evidence": protected_evidence,
                    "required_fact_count": len(required),
                    "optional_detail_count": len(optional),
                    "validation_promoted_optional_details": (
                        list(validation_promoted_optional_details)
                        if variant_id == "compact"
                        else []
                    ),
                }
            )
        # Textual distinctness is desirable, but not an invariant of truthful
        # translation. Very short units can already be at their safe compression
        # floor. In that case adjacent semantic slots may legitimately share the
        # same spoken text. Non-adjacent equality would imply broken compression
        # ordering (for example natural == compact while concise differs), so it
        # remains invalid.
        spoken_values = [_normalise_space(str(item["spoken_text"])) for item in variants]
        spoken_keys = [value.casefold() for value in spoken_values]
        groups_by_text: dict[str, list[int]] = {}
        for variant_index, key in enumerate(spoken_keys):
            groups_by_text.setdefault(key, []).append(variant_index)
        duplicate_index_groups = [
            indices for indices in groups_by_text.values() if len(indices) > 1
        ]
        for indices in duplicate_index_groups:
            if indices != list(range(indices[0], indices[-1] + 1)):
                duplicate_ids = [VARIANT_IDS[index] for index in indices]
                raise MultivariantTranslationError(
                    f"Unit {unit_id} has non-adjacent equivalent variants: "
                    f"{duplicate_ids}"
                )

        estimated = [int(item["estimated_duration_ms"]) for item in variants]
        for index in range(len(estimated) - 1):
            current = estimated[index]
            following = estimated[index + 1]
            same_spoken_text = spoken_keys[index] == spoken_keys[index + 1]
            if same_spoken_text:
                if current < following:
                    raise MultivariantTranslationError(
                        f"Unit {unit_id} equivalent variants must not have increasing "
                        "estimated durations"
                    )
            elif current <= following:
                raise MultivariantTranslationError(
                    f"Unit {unit_id} estimated durations must strictly decrease when "
                    "spoken_text changes from natural to concise to compact"
                )

        equivalent_variant_groups = [
            [VARIANT_IDS[index] for index in indices]
            for indices in duplicate_index_groups
        ]
        validation_units.append(
            {
                "unit_id": unit_id,
                "variants": variant_audit,
                "distinct_spoken_variant_count": len(groups_by_text),
                "compression_floor_reached": bool(duplicate_index_groups),
                "equivalent_variant_groups": equivalent_variant_groups,
                "contextual_identity_reviews": contextual_identity_reviews,
            }
        )

    contextual_identity_review_count = sum(
        len(item.get("contextual_identity_reviews") or [])
        for item in validation_units
    )
    return {
        "schema_version": "mathula.translation.validation.v1",
        "validation_contract_version": VALIDATION_CONTRACT_VERSION,
        "valid": True,
        "request_sha256": str(request.get("request_sha256") or _sha256_json(request)),
        "response_sha256": _sha256_json(response),
        "unit_count": len(actual_units),
        "contextual_identity_review_count": contextual_identity_review_count,
        "units": validation_units,
    }


def prepare_installed_translation(
    *,
    response: Mapping[str, Any],
    request: Mapping[str, Any],
    mode: str,
    reviewed_by: str | None = None,
    provider_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and enrich a source-bound response without writing files."""

    validation = validate_multivariant_response(response, request=request)
    now = datetime.now(timezone.utc).isoformat()
    metadata = dict(provider_metadata or {})
    installation = {
        "mode": mode,
        "installed_at": now,
        "reviewed_by": reviewed_by.strip() if reviewed_by else None,
        "prompt_version": str(request.get("prompt_version") or PROMPT_VERSION),
        "request_sha256": str(request.get("request_sha256") or _sha256_json(request)),
        "response_sha256": _sha256_json(response),
        "source_transcript_sha256": str(request.get("source_transcript_sha256") or ""),
        "source_transcript_file_sha256": str(
            request.get("source_transcript_file_sha256") or ""
        ),
        "source_timeline_kind": str(request.get("source_timeline_kind") or ""),
        "source_timeline_sha256": str(request.get("source_timeline_sha256") or ""),
        "source_timeline_file_sha256": str(
            request.get("source_timeline_file_sha256") or ""
        ),
        "source_bound": True,
        "human_review_required": True,
        "provider": metadata,
    }
    installed = {
        **dict(response),
        "schema_version": INSTALLED_SCHEMA_VERSION,
        "source_response_schema_version": response.get("schema_version"),
        "source_transcript_sha256": (
            request.get("source_transcript_file_sha256")
            or request.get("source_transcript_sha256")
        ),
        "source_dubbing_units_sha256": (
            request.get("source_timeline_file_sha256")
            if request.get("source_timeline_kind") == "dubbing_units"
            else None
        ),
        "translation_request_sha256": installation["request_sha256"],
        "installation": installation,
        "validation": validation,
    }
    return installed, validation


def build_three_text_compatibility_artifact(
    installed: Mapping[str, Any],
    *,
    pronunciation_dictionary: PronunciationDictionary | None = None,
) -> dict[str, Any]:
    """Build the legacy-compatible natural delivery while retaining all variants."""

    dictionary = (
        with_default_organisation_initialisms(pronunciation_dictionary)
        if pronunciation_dictionary is not None
        else None
    )
    units: list[dict[str, Any]] = []
    for item in installed.get("units", []):
        variants_out: list[dict[str, Any]] = []
        for variant in item.get("variants", []):
            spoken = str(variant.get("spoken_text") or "").strip()
            tts = dictionary.apply(spoken).tts_text if dictionary is not None else spoken
            variants_out.append({**dict(variant), "tts_text": tts})
        natural = next(
            value for value in variants_out if value.get("variant_id") == "natural"
        )
        units.append(
            {
                "unit_id": str(item["unit_id"]),
                "speaker_id": str(item.get("speaker_id") or ""),
                "start_ms": int(item.get("start_ms") or 0),
                "end_ms": int(item.get("end_ms") or 0),
                "available_duration_ms": int(item.get("available_duration_ms") or 0),
                "overlap_group": item.get("overlap_group"),
                "source_text": str(item.get("source_text") or ""),
                "faithful_translation": natural["spoken_text"],
                "spoken_text": natural["spoken_text"],
                "tts_text": natural["tts_text"],
                "translation_variants": variants_out,
                # Also expose variants under the canonical key so newer readers
                # do not need to understand the compatibility wrapper.
                "variants": variants_out,
                "selected_variant_id": None,
                "selection_status": "unselected",
                "protected_spans": list(item.get("protected_spans", [])),
                "protected_entities_found": list(item.get("protected_spans", [])),
                "protected_entities_expected": len(item.get("protected_spans", [])),
                "protected_entities_restored": len(item.get("protected_spans", [])),
                "protected_entities_missing": [],
                "protected_entities_unexpected": [],
                "human_review_flags": [
                    "multivariant_translation_requires_human_review"
                ],
                "timing_strategy": "prepared_natural_concise_compact_variants",
                "omitted_or_compressed_detail": [],
            }
        )
    installation = dict(installed.get("installation") or {})
    mode = str(installation.get("mode") or "")
    return {
        "schema_version": "three-text-translation-v2-multivariant",
        "language": installed.get("target_locale"),
        "source_schema_version": installed.get("schema_version"),
        "source_artifact_sha256": _sha256_json(installed),
        "source_transcript_sha256": installed.get("source_transcript_sha256"),
        "source_dubbing_units_sha256": installed.get("source_dubbing_units_sha256"),
        "translation_request_sha256": installed.get("translation_request_sha256"),
        "ai_called": mode not in {"manual_override", "manual_import"},
        "manual_override": mode in {"manual_override", "manual_import"},
        "translation_strategy": "contextual_code_switching_multivariant",
        "units": units,
        "seo": {},
        "metadata": installation,
    }


def _backup(path: Path, backup_dir: Path) -> str | None:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / path.name
    shutil.copy2(path, destination)
    return str(destination)


def write_translation_artifacts(
    *,
    job_root: Path,
    target_locale: str,
    installed: Mapping[str, Any],
    pronunciation_dictionary: PronunciationDictionary | None = None,
    backup_dir: Path | None = None,
) -> dict[str, Any]:
    """Atomically replace raw and compatibility artifacts with rollback.

    Each file replacement is atomic. A temporary rollback directory is still
    used when the caller does not request persistent backups, because the two
    authoritative files must advance as one logical transaction.
    """

    suffix = language_suffix(target_locale)
    translation_target = job_root / "translation" / f"transcript_{suffix}.json"
    dubbing_target = job_root / "dubbing" / f"translation_{suffix}.json"
    compatibility = build_three_text_compatibility_artifact(
        installed,
        pronunciation_dictionary=pronunciation_dictionary,
    )
    ephemeral_backup_dir: Path | None = None
    effective_backup_dir = backup_dir
    if effective_backup_dir is None:
        ephemeral_backup_dir = (
            job_root
            / "translation"
            / ".install_rollback"
            / uuid.uuid4().hex
        )
        effective_backup_dir = ephemeral_backup_dir

    backups = {
        "translation": _backup(translation_target, effective_backup_dir),
        "dubbing": _backup(dubbing_target, effective_backup_dir),
    }
    existed = {
        "translation": translation_target.exists(),
        "dubbing": dubbing_target.exists(),
    }
    try:
        atomic_write_json(translation_target, dict(installed))
        atomic_write_json(dubbing_target, compatibility)
    except Exception:
        for label, target in (("translation", translation_target), ("dubbing", dubbing_target)):
            backup = backups[label]
            if backup:
                shutil.copy2(Path(backup), target)
            elif not existed[label]:
                target.unlink(missing_ok=True)
        raise
    finally:
        if ephemeral_backup_dir is not None:
            shutil.rmtree(ephemeral_backup_dir, ignore_errors=True)
            parent = ephemeral_backup_dir.parent
            try:
                parent.rmdir()
            except OSError:
                pass

    visible_backups = backups if backup_dir is not None else {"translation": None, "dubbing": None}
    return {
        "translation_target": str(translation_target),
        "dubbing_target": str(dubbing_target),
        "translation_sha256": checksum(translation_target),
        "dubbing_sha256": checksum(dubbing_target),
        "backups": visible_backups,
    }


def install_manual_translation(
    *,
    job_root: Path,
    job_id: str,
    target_locale: str,
    response_path: Path,
    reviewed_by: str,
    request_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not reviewed_by.strip():
        raise MultivariantTranslationError("--reviewed-by is required and cannot be empty")

    current_request, source_paths = build_job_translation_request(
        job_root=job_root,
        job_id=job_id,
        target_locale=target_locale,
    )
    transcript_path = Path(source_paths["transcript_path"])
    if request_path is not None:
        request = read_json(request_path)
        if request.get("job_id") != job_id:
            raise MultivariantTranslationError(
                f"Translation request belongs to {request.get('job_id')!r}, not {job_id!r}"
            )
        if request.get("target_locale") != target_locale:
            raise MultivariantTranslationError(
                f"Translation request target locale {request.get('target_locale')!r} "
                f"does not match job locale {target_locale!r}"
            )
        for field, message in (
            (
                "source_transcript_sha256",
                "Translation request is stale because the authoritative English transcript changed",
            ),
            (
                "source_timeline_sha256",
                "Translation request is stale because the source timing units changed",
            ),
        ):
            supplied = request.get(field)
            current = current_request.get(field)
            if supplied not in {None, current}:
                raise MultivariantTranslationError(message)
    else:
        request = current_request

    response = read_json(response_path)
    installed, validation = prepare_installed_translation(
        response=response,
        request=request,
        mode="manual_override",
        reviewed_by=reviewed_by,
        provider_metadata={"name": "external_manual_override"},
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = language_suffix(target_locale)
    translation_target = job_root / "translation" / f"transcript_{suffix}.json"
    dubbing_target = job_root / "dubbing" / f"translation_{suffix}.json"
    history_dir = job_root / "translation" / "manual_overrides" / stamp
    result = {
        "schema_version": "mathula.translation.manual-install-result.v1",
        "job_id": job_id,
        "target_locale": target_locale,
        "status": "validated" if dry_run else "installed",
        "dry_run": dry_run,
        "unit_count": validation["unit_count"],
        "response_path": str(response_path),
        "request_path": str(request_path) if request_path else None,
        "translation_target": str(translation_target),
        "dubbing_target": str(dubbing_target),
        "history_directory": str(history_dir),
        "source_transcript_path": str(transcript_path),
        "validation": validation,
    }
    if dry_run:
        return result

    history_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(history_dir / "translation_request.json", request)
    atomic_write_json(history_dir / "translation_response.json", response)
    atomic_write_json(history_dir / "validation.json", validation)
    dictionary_path = job_root / "dubbing" / "pronunciation_dictionary.json"
    dictionary = (
        PronunciationDictionary.from_dict(read_json(dictionary_path))
        if dictionary_path.is_file()
        else PronunciationDictionary(
            dictionary_version="v1",
            language=target_locale,
            job_id=job_id,
        )
    )
    written = write_translation_artifacts(
        job_root=job_root,
        target_locale=target_locale,
        installed=installed,
        pronunciation_dictionary=dictionary,
        backup_dir=history_dir / "backups",
    )
    install_manifest = {
        **result,
        "installed_translation_sha256": written["translation_sha256"],
        "installed_dubbing_sha256": written["dubbing_sha256"],
        "backups": written["backups"],
    }
    atomic_write_json(history_dir / "installed_translation.json", installed)
    atomic_write_json(history_dir / "install_manifest.json", install_manifest)
    return install_manifest


def load_request_for_response(
    response_path: Path,
    *,
    explicit_request_path: Path | None = None,
) -> Path | None:
    if explicit_request_path is not None:
        return explicit_request_path
    sibling = response_path.parent / "translation_request.json"
    return sibling if sibling.is_file() else None


__all__ = [
    "INSTALLED_SCHEMA_VERSION",
    "MultivariantTranslationError",
    "PROMPT_VERSION",
    "REQUEST_SCHEMA",
    "REQUEST_SCHEMA_VERSION",
    "RESPONSE_SCHEMA",
    "RESPONSE_SCHEMA_VERSION",
    "normalize_multivariant_response",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "VARIANT_IDS",
    "VARIANT_RATIOS",
    "build_job_translation_request",
    "build_three_text_compatibility_artifact",
    "build_translation_request",
    "export_translation_package",
    "extract_source_units",
    "install_manual_translation",
    "identity_requirements_for_unit",
    "restore_missing_hard_identities_in_unit",
    "prepare_installed_translation",
    "language_name",
    "language_suffix",
    "load_request_for_response",
    "render_system_prompt",
    "render_user_prompt",
    "build_translation_batch_requests",
    "merge_multivariant_batch_responses",
    "validate_multivariant_response",
    "write_translation_artifacts",
]

# Alias retained for callers that want the request schema object beside the
# response schema. The request contains source-bound dynamic timing values, so
# validation is primarily performed by build_translation_request.
REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "job_id", "target_locale", "units"],
    "properties": {
        "schema_version": {"const": REQUEST_SCHEMA_VERSION},
        "job_id": {"type": "string"},
        "target_locale": {"enum": sorted(_LOCALE_NAMES)},
        "units": {"type": "array", "minItems": 1},
    },
}
