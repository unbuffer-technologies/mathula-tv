"""Closed-loop Azure STT mastering for completed direct-Azure dubs.

The loop audits the actual final dialogue mix, not duration estimates. It uses
Azure word timestamps and the direct-dub report to detect intelligibility loss,
false internal pauses, unstable tempo, and cross-speaker timeline collisions.
Only already-approved translation variants may be selected automatically. The
authoritative translation and speaker identities remain immutable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import unicodedata
import wave
from array import array
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .ai_consumption import consumption_record, update_ai_consumption_report
from .atomic_io import atomic_write_json, read_json
from .azure_stt import SpeechBackend, normalize as normalize_azure_stt
from .direct_azure_dub import (
    DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
    DirectAzureDubRenderer,
    DirectDubOptions,
)
from .entity_pronunciation_audit import entity_surface_recognized
from .media import checksum
from .models import JobManifest


DUB_MASTERING_SCHEMA_VERSION = "mathula-dub-mastering-loop-v9-dual-locale-entity-qa"
DUB_MASTERING_ROUND_SCHEMA_VERSION = "mathula-dub-mastering-round-v9"
DUB_MASTERING_POLICY_VERSION = "mathula-production-dub-score-v9"
POST_TTS_QUALITY_GATE_SCHEMA_VERSION = "mathula-post-tts-stt-quality-gate-v1"
SAFE_OVERRIDE_CARRY_FORWARD_POLICIES = {
    "mathula-production-dub-score-v5",
    "mathula-production-dub-score-v6",
    "mathula-production-dub-score-v7",
    "mathula-production-dub-score-v8",
}
ProgressCallback = Callable[[dict[str, Any]], None]
_WORD = re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*", re.UNICODE)
_TITLE_WORD = re.compile(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*\b")
_VARIANT_ORDER = ("natural", "concise", "compact")
_TITLED_PERSON_ENTITY = re.compile(
    r"^(?:advocate|chief|doctor|dr|general|justice|judge|major[-\s]+general|"
    r"minister|mr|mrs|ms|professor|sergeant)\b",
    re.IGNORECASE,
)
_IDENTITY_STOP_WORDS = frozenset(
    {
        "advocate",
        "chief",
        "commission",
        "commissioner",
        "doctor",
        "dr",
        "general",
        "inquiry",
        "justice",
        "judge",
        "major",
        "miss",
        "mister",
        "mr",
        "mrs",
        "ms",
        "of",
        "sc",
        "the",
    }
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_word(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def _words(value: str) -> list[str]:
    return [
        normalized
        for token in _WORD.findall(unicodedata.normalize("NFKC", str(value)))
        if (normalized := _normalise_word(token))
    ]


def _azure_recognized_text(raw: Mapping[str, Any]) -> str:
    combined = " ".join(
        str(item.get("text") or "")
        for item in raw.get("combinedPhrases") or ()
        if isinstance(item, Mapping)
    ).strip()
    if combined:
        return combined
    return " ".join(
        str(item.get("text") or "")
        for item in raw.get("phrases") or ()
        if isinstance(item, Mapping)
    ).strip()


def apply_english_entity_pronunciation_gate(
    audit: dict[str, Any],
    *,
    protected_entities: Sequence[str],
    raw_english_stt: Mapping[str, Any],
    raw_response_path: Path | None = None,
) -> dict[str, Any]:
    """Fail a mastered dub when en-ZA STT loses a protected entity surface."""

    transcription = _azure_recognized_text(raw_english_stt)
    supplied_entities = list(
        dict.fromkeys(
            " ".join(str(value).split()).strip()
            for value in protected_entities
            if " ".join(str(value).split()).strip()
        )
    )
    dimensions = audit.setdefault("quality_dimensions", {})
    old_identity_score = float(dimensions.get("protected_identity_recognition") or 0.0)
    assessed_keys: set[str] = set()
    locally_recognized_keys: set[str] = set()
    assessment_block_counts: dict[str, int] = {}
    assessed_surfaces: dict[str, list[str]] = {}
    assessment_metadata_present = False
    for block in audit.get("blocks") or ():
        if not isinstance(block, Mapping):
            continue
        if "protected_entities_assessed" in block:
            assessment_metadata_present = True
        for assessment in block.get("protected_entities_assessed") or ():
            if not isinstance(assessment, Mapping):
                continue
            key = str(assessment.get("entity") or "").casefold()
            if not key:
                continue
            assessed_keys.add(key)
            assessment_block_counts[key] = assessment_block_counts.get(key, 0) + 1
            surface = " ".join(
                str(value)
                for value in assessment.get("expected_surface_tokens") or ()
                if str(value).strip()
            ).strip()
            if surface:
                assessed_surfaces.setdefault(key, []).append(surface)
            if assessment.get("recognized") is True:
                locally_recognized_keys.add(key)
    entities = [
        value
        for value in supplied_entities
        if not assessment_metadata_present or value.casefold() in assessed_keys
    ]
    english_recognized_keys = {
        value.casefold()
        for value in entities
        if entity_surface_recognized(value, transcription)
        or any(
            entity_surface_recognized(surface, transcription)
            for surface in assessed_surfaces.get(value.casefold(), ())
        )
    }

    # Target-locale STT predictably anglicises some isolated code switches
    # (for example ``Hawks`` -> ``house``).  Let the dedicated en-ZA pass
    # arbitrate only when that entity is assessed in one timeline block.  A
    # repeated entity remains blocked because whole-program English STT cannot
    # prove which occurrence it recognized.
    for block in audit.get("blocks") or ():
        if not isinstance(block, dict):
            continue
        reconciled_keys: set[str] = set()
        for assessment in block.get("protected_entities_assessed") or ():
            if not isinstance(assessment, dict):
                continue
            key = str(assessment.get("entity") or "").casefold()
            if (
                assessment.get("recognized") is not True
                and key in english_recognized_keys
                and assessment_block_counts.get(key) == 1
            ):
                assessment["recognized"] = True
                assessment["match_policy"] = "dual-locale-en-za-arbitrated-v1"
                assessment["verification_locale"] = "en-ZA"
                reconciled_keys.add(key)
                locally_recognized_keys.add(key)
        if not reconciled_keys:
            continue
        block["protected_entities_missing"] = [
            value
            for value in block.get("protected_entities_missing") or ()
            if str(value).casefold() not in reconciled_keys
        ]
        if not block["protected_entities_missing"]:
            block["reason_codes"] = [
                value
                for value in block.get("reason_codes") or ()
                if value != "protected_entity_not_recognized"
            ]
            block["production_blockers"] = [
                value
                for value in block.get("production_blockers") or ()
                if value != "protected_entity_not_recognized"
            ]
            block["passed"] = not block["reason_codes"]
            block["production_gate_passed"] = not block["production_blockers"]

    # The en-ZA pass is an arbiter for entities the target-locale pass could
    # not recognize; it must never overturn a positive block-local result.
    # Whole-program English STT is intentionally noisy on isiZulu carrier
    # speech and cannot localize a repeated name to a particular occurrence.
    failed = [
        value
        for value in entities
        if value.casefold() not in locally_recognized_keys
        and value.casefold() not in english_recognized_keys
    ]
    failed_keys = {value.casefold() for value in failed}
    for block in audit.get("blocks") or ():
        if not isinstance(block, dict):
            continue
        block_failures = [
            str(value)
            for value in block.get("protected_entities") or ()
            if str(value).casefold() in failed_keys
        ]
        if not block_failures:
            continue
        block["english_stt_entities_missing"] = block_failures
        block["reason_codes"] = list(
            dict.fromkeys(
                [
                    *block.get("reason_codes", []),
                    "english_code_switch_pronunciation_failure",
                ]
            )
        )
        block["production_blockers"] = list(
            dict.fromkeys(
                [
                    *block.get("production_blockers", []),
                    "english_code_switch_pronunciation_failure",
                ]
            )
        )
        block["passed"] = False
        block["production_gate_passed"] = False

    passed_count = len(entities) - len(failed)
    english_score = 100.0 * passed_count / len(entities) if entities else 100.0
    local_score = (
        100.0 * len(locally_recognized_keys) / len(assessed_keys)
        if assessed_keys
        else 100.0
    )
    dimensions["protected_identity_recognition"] = round(
        min(local_score, english_score), 2
    )
    audit["production_score"] = round(
        min(
            100.0,
            max(
                0.0,
                float(audit.get("production_score") or 0.0)
                + 0.15
                * (
                    float(dimensions["protected_identity_recognition"])
                    - old_identity_score
                ),
            ),
        ),
        2,
    )
    failed_block_count = sum(
        not bool(block.get("passed"))
        for block in audit.get("blocks") or ()
        if isinstance(block, Mapping)
    )
    production_blocker_count = sum(
        not bool(block.get("production_gate_passed"))
        for block in audit.get("blocks") or ()
        if isinstance(block, Mapping)
    )
    audit["failed_block_count"] = max(failed_block_count, len(failed))
    audit["production_blocker_count"] = max(
        production_blocker_count, len(failed)
    )
    thresholds = audit.get("thresholds") or {}
    audit["production_ready"] = (
        not failed
        and audit["production_blocker_count"] == 0
        and float(audit.get("production_score") or 0.0)
        >= float(thresholds.get("minimum_production_score") or 0.0)
        and float(audit.get("aggregate_word_error_rate") or 0.0)
        <= float(thresholds.get("maximum_aggregate_wer") or 1.0)
    )
    result = {
        "verification_locale": "en-ZA",
        "phrase_hints_used": False,
        "recognized_text": transcription,
        "protected_entity_count": len(entities),
        "unassessable_entity_count": len(supplied_entities) - len(entities),
        "passed_entity_count": passed_count,
        "failed_entity_count": len(failed),
        "failed_entities": failed,
        "passed": not failed,
        "raw_response_path": (
            str(raw_response_path) if raw_response_path is not None else None
        ),
    }
    audit["english_code_switch_qa"] = result
    return result


def _named_local_anchors(text: str) -> list[dict[str, Any]]:
    """Extract conservative multiword names with source-relative positions."""

    matches = list(_TITLE_WORD.finditer(text))
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
    return [
        {
            "anchor": " ".join(match.group(0) for match in run),
            "tokens": [_normalise_word(match.group(0)) for match in run],
            "relative_position": round(run[0].start() / max(1, len(text)), 6),
        }
        for run in runs
    ]


def _anchor_position_in_target(
    text: str,
    anchor_tokens: Sequence[str],
    *,
    reference_position: float | None = None,
) -> float | None:
    matches = list(_WORD.finditer(text))
    tokens = [_normalise_word(match.group(0)) for match in matches]
    prefixes = {
        "a", "e", "i", "o", "u", "ka", "ku", "kwa", "kuka", "laka",
        "luka", "na", "ne", "no", "nga", "ngo", "ngu", "se", "si",
    }
    positions: list[float] = []
    for index in range(len(tokens) - len(anchor_tokens) + 1):
        candidate = tokens[index : index + len(anchor_tokens)]
        first = candidate[0]
        expected_first = anchor_tokens[0]
        first_matches = first == expected_first
        if not first_matches and first.endswith(expected_first):
            first_matches = first[: -len(expected_first)] in prefixes
        if first_matches and candidate[1:] == list(anchor_tokens[1:]):
            positions.append(
                round(matches[index].start() / max(1, len(text)), 6)
            )
    if not positions:
        return None
    if reference_position is None:
        return positions[0]
    # Repeated named entities are common in broadcast summaries. Compare the
    # source occurrence with the nearest target occurrence instead of falsely
    # treating an earlier, additional mention as information-order drift.
    return min(positions, key=lambda value: abs(value - reference_position))


def _local_information_order_anomalies(
    source_text: str,
    target_text: str,
    *,
    maximum_drift: float,
) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    for anchor in _named_local_anchors(source_text):
        target_position = _anchor_position_in_target(
            target_text,
            anchor["tokens"],
            reference_position=float(anchor["relative_position"]),
        )
        if target_position is None:
            continue
        drift = abs(float(target_position) - float(anchor["relative_position"]))
        if drift > maximum_drift:
            anomalies.append(
                {
                    "anchor": anchor["anchor"],
                    "source_relative_position": anchor["relative_position"],
                    "target_relative_position": target_position,
                    "absolute_relative_drift": round(drift, 6),
                    "maximum_allowed_drift": maximum_drift,
                    "detector": "multiword_named_anchor_relative_position_v1",
                }
            )
    return anomalies


def _tts_text_fingerprint(value: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_overrides_for_policy_migration(
    payload: Mapping[str, Any],
    *,
    installed_policy: str,
) -> list[dict[str, Any]]:
    if installed_policy not in SAFE_OVERRIDE_CARRY_FORWARD_POLICIES:
        return []
    return [
        dict(value)
        for value in payload.get("overrides") or []
        if isinstance(value, Mapping)
        and str(value.get("strategy") or "") == "select_variant"
        and str(value.get("variant_id") or "") in _VARIANT_ORDER
    ]


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = _words(reference)
    observed = _words(hypothesis)
    if not expected:
        return 0.0 if not observed else 1.0
    previous = list(range(len(observed) + 1))
    for row, expected_word in enumerate(expected, start=1):
        current = [row]
        for column, observed_word in enumerate(observed, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1]
                    + (0 if expected_word == observed_word else 1),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def _delivery_token_match(left: str, right: str) -> bool:
    """Accept conservative STT spelling variants without hiding omissions."""

    if left == right:
        return True
    return (
        min(len(left), len(right)) >= 5
        and SequenceMatcher(None, left, right).ratio() >= 0.82
    )


def delivery_word_error_rate(reference: str, hypothesis: str) -> float:
    """WER tolerant of isiZulu token joining and close STT orthography.

    Azure may recognize ``mhla ziyishumi`` as ``mhlaziyishumi`` even when the
    synthesized delivery is correct.  The ordinary WER remains available for
    diagnostics; this bounded alignment is used only for the production
    intelligibility score.  It permits one-to-two/two-to-one token boundaries
    and conservative long-token spelling variation.
    """

    expected = _words(reference)
    observed = _words(hypothesis)
    if not expected:
        return 0.0 if not observed else 1.0
    height = len(expected) + 1
    width = len(observed) + 1
    matrix = [[0] * width for _ in range(height)]
    for row in range(height):
        matrix[row][0] = row
    for column in range(width):
        matrix[0][column] = column
    for row in range(1, height):
        for column in range(1, width):
            value = min(
                matrix[row - 1][column] + 1,
                matrix[row][column - 1] + 1,
                matrix[row - 1][column - 1]
                + (0 if _delivery_token_match(expected[row - 1], observed[column - 1]) else 1),
            )
            if row >= 2 and _delivery_token_match(
                expected[row - 2] + expected[row - 1],
                observed[column - 1],
            ):
                value = min(value, matrix[row - 2][column - 1])
            if column >= 2 and _delivery_token_match(
                expected[row - 1],
                observed[column - 2] + observed[column - 1],
            ):
                value = min(value, matrix[row - 1][column - 2])
            matrix[row][column] = value
    return matrix[-1][-1] / len(expected)


def _word_alignment_partitions(
    references: Sequence[str],
    words: Sequence[Mapping[str, Any]],
) -> tuple[list[list[Mapping[str, Any]]], dict[str, Any]]:
    """Partition one STT stream across ordered blocks with global alignment.

    Rigid timestamp windows lose short interjections and words that Azure puts a
    few milliseconds across a fitted block boundary. A global edit alignment
    preserves block order while allowing those small timing differences.
    """

    reference_words = [_words(value) for value in references]
    expected = [token for block in reference_words for token in block]
    observed = [_normalise_word(str(value.get("text") or "")) for value in words]
    observed = [value for value in observed if value]
    if not expected or not observed:
        return (
            [[] for _ in references],
            {
                "method": "global_monotonic_word_edit_alignment_v1",
                "expected_word_count": len(expected),
                "observed_word_count": len(observed),
                "boundary_positions": [0 for _ in range(len(references) + 1)],
            },
        )

    observed_items = [
        value
        for value in words
        if _normalise_word(str(value.get("text") or ""))
    ]
    width = len(observed) + 1
    previous = list(range(width))
    directions = [bytearray(width) for _ in range(len(expected) + 1)]
    for column in range(1, width):
        directions[0][column] = 3  # observed insertion
    for row, expected_word in enumerate(expected, start=1):
        current = [row]
        directions[row][0] = 2  # expected deletion
        for column, observed_word in enumerate(observed, start=1):
            substitution = previous[column - 1] + (
                0 if expected_word == observed_word else 1
            )
            deletion = previous[column] + 1
            insertion = current[column - 1] + 1
            if substitution <= deletion and substitution <= insertion:
                current.append(substitution)
                directions[row][column] = 1
            elif deletion <= insertion:
                current.append(deletion)
                directions[row][column] = 2
            else:
                current.append(insertion)
                directions[row][column] = 3
        previous = current

    operations: list[int] = []
    row = len(expected)
    column = len(observed)
    while row or column:
        direction = directions[row][column]
        if direction == 1:
            row -= 1
            column -= 1
        elif direction == 2:
            row -= 1
        elif direction == 3:
            column -= 1
        else:  # defensive fallback at a malformed matrix edge
            if row:
                row -= 1
                direction = 2
            else:
                column -= 1
                direction = 3
        operations.append(direction)
    operations.reverse()

    expected_boundaries: list[int] = [0]
    for block in reference_words:
        expected_boundaries.append(expected_boundaries[-1] + len(block))
    boundary_positions = [0]
    next_boundary = 1
    row = 0
    column = 0
    for direction in operations:
        # Insertions at an inter-block boundary are normally Azure repeating or
        # extending the preceding utterance. Allocate them to that preceding
        # block; freeze the boundary only when alignment starts consuming the
        # next block's expected text.
        while (
            next_boundary < len(expected_boundaries)
            and row == expected_boundaries[next_boundary]
            and direction in (1, 2)
        ):
            boundary_positions.append(column)
            next_boundary += 1
        if direction == 1:
            row += 1
            column += 1
        elif direction == 2:
            row += 1
        else:
            column += 1
    while len(boundary_positions) < len(expected_boundaries):
        boundary_positions.append(len(observed_items))
    boundary_positions[-1] = len(observed_items)

    partitions = [
        observed_items[boundary_positions[index] : boundary_positions[index + 1]]
        for index in range(len(references))
    ]
    return (
        partitions,
        {
            "method": "global_monotonic_word_edit_alignment_v1",
            "expected_word_count": len(expected),
            "observed_word_count": len(observed),
            "edit_distance": previous[-1],
            "boundary_positions": boundary_positions,
        },
    )


def _identity_surface_tokens(entity: str, *references: str) -> list[str]:
    identity_tokens = [
        value
        for value in _words(entity)
        if value not in _IDENTITY_STOP_WORDS and len(value) >= 3
    ]
    reference_tokens = [value for reference in references for value in _words(reference)]
    return [
        value
        for value in identity_tokens
        if any(
            value == reference
            or (
                reference.endswith(value)
                and len(reference) - len(value) <= 4
            )
            for reference in reference_tokens
        )
    ]


def _delivery_identity_surface_tokens(
    entity: str,
    spoken_text: str,
    tts_text: str,
) -> list[str]:
    """Return the surface that was actually sent to TTS for an identity.

    Pronunciation entries may replace a protected written name with a reviewed
    spoken alias (for example the full Independent Police Investigative
    Directorate name with ``Ai-pid``).  Auditing only the written surface then
    rejects audio that correctly contains the approved alias.  Map the entity's
    token span from approved spoken text into the corresponding TTS token span;
    fall back to the historic shared-token matcher when no complete span exists.
    """

    entity_tokens = _words(entity)
    spoken_tokens = _words(spoken_text)
    tts_tokens = _words(tts_text)
    if not entity_tokens or not spoken_tokens or not tts_tokens:
        return _identity_surface_tokens(entity, spoken_text, tts_text)

    entity_start: int | None = None
    for start in range(len(spoken_tokens) - len(entity_tokens) + 1):
        candidate = spoken_tokens[start : start + len(entity_tokens)]
        if all(
            actual == expected
            or (
                actual.endswith(expected)
                and len(actual) - len(expected) <= 4
            )
            for expected, actual in zip(entity_tokens, candidate, strict=True)
        ):
            entity_start = start
            break
    if entity_start is None:
        return _identity_surface_tokens(entity, spoken_text, tts_text)

    entity_end = entity_start + len(entity_tokens)
    mapped: list[str] = []
    matcher = SequenceMatcher(a=spoken_tokens, b=tts_tokens, autojunk=False)
    for tag, spoken_start, spoken_end, tts_start, tts_end in matcher.get_opcodes():
        overlap_start = max(entity_start, spoken_start)
        overlap_end = min(entity_end, spoken_end)
        if overlap_start >= overlap_end:
            continue
        if tag == "equal":
            offset_start = overlap_start - spoken_start
            offset_end = overlap_end - spoken_start
            mapped.extend(tts_tokens[tts_start + offset_start : tts_start + offset_end])
        elif tag == "replace":
            mapped.extend(tts_tokens[tts_start:tts_end])

    # Use the delivery mapping only when pronunciation deliberately compresses
    # the identity (normally a full organisation name spoken as its acronym).
    # Same-length phonetic respellings such as ``Five`` -> ``Faiv`` are better
    # assessed against the canonical tokens so numeric STT forms still match.
    if mapped and len(mapped) < len(entity_tokens):
        return mapped
    return _identity_surface_tokens(entity, spoken_text, tts_text)


def _code_switch_phonetic_skeleton(value: str) -> str:
    """Collapse conservative English/isiZulu STT spelling equivalents."""

    normalized = _normalise_word(value)
    # en-ZA is non-rhotic before a consonant (``cartel`` -> ``katel``,
    # ``Sergeant`` -> ``sagent``).  Model that narrow environment without
    # deleting intervocalic/name-internal r sounds such as ``Carrim``.
    normalized = re.sub(r"r(?=[^aeiouy])", "", normalized)
    normalized = re.sub(r"r$", "", normalized)
    normalized = normalized.replace("ph", "f").replace("th", "t")
    normalized = normalized.replace("c", "k").replace("q", "k")
    normalized = normalized.translate(
        str.maketrans({"b": "p", "d": "t", "g": "k", "v": "f", "z": "s"})
    )
    normalized = re.sub(r"[aeiouywh]+", "", normalized)
    return re.sub(r"(.)\1+", r"\1", normalized)


def _identity_token_match(expected: str, candidate: str) -> bool:
    if expected == candidate:
        return True
    number_equivalents = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    if (
        number_equivalents.get(candidate) == expected
        or number_equivalents.get(expected) == candidate
    ):
        return True
    if candidate.endswith(expected) and len(candidate) - len(expected) <= 4:
        return True
    if len(expected) >= 3:
        expected_skeleton = _code_switch_phonetic_skeleton(expected)
        candidate_skeleton = _code_switch_phonetic_skeleton(candidate)
        if (
            expected_skeleton == candidate_skeleton
            and (
                len(expected_skeleton) >= 2
                or len(expected) >= 4
            )
        ):
            return True
        if (
            len(expected_skeleton) >= 3
            and candidate_skeleton.startswith(expected_skeleton)
            and len(candidate_skeleton) - len(expected_skeleton) <= 2
        ):
            # Azure often fuses a following isiZulu conjunction to a foreign
            # surname (``Carrim kanye`` -> ``karimganye``).
            return True
        if (
            len(expected_skeleton) >= 2
            and candidate_skeleton == expected_skeleton + "s"
        ):
            # Likewise, a following concord can surface as a single sibilant
            # on a short nickname (``Cat`` -> ``kats``).
            return True
    threshold = 0.78 if len(expected) <= 5 else 0.70
    return SequenceMatcher(None, expected, candidate).ratio() >= threshold


def _identity_joined_token_match(expected: str, candidate: str) -> bool:
    """Conservatively match only genuine one-to-two token boundary errors."""

    if expected == candidate:
        return True
    expected_skeleton = _code_switch_phonetic_skeleton(expected)
    candidate_skeleton = _code_switch_phonetic_skeleton(candidate)
    if len(expected_skeleton) >= 3 and expected_skeleton == candidate_skeleton:
        return True
    if (
        len(expected_skeleton) >= 3
        and candidate_skeleton.endswith(expected_skeleton)
        and len(candidate_skeleton) - len(expected_skeleton) <= 2
    ):
        return True
    return SequenceMatcher(None, expected, candidate).ratio() >= 0.86


def _contains_identity_surface(text: str, surface_tokens: Sequence[str]) -> bool:
    observed = _words(text)
    expected = list(surface_tokens)
    if not expected:
        return True

    # Match one continuous identity surface while tolerating the two common
    # Azure boundary errors: an isiZulu concord fused to the first name, and a
    # pair of code-switched English words returned as one token (or vice versa).
    def matches(expected_index: int, observed_index: int) -> bool:
        if expected_index == len(expected):
            return True
        if observed_index >= len(observed):
            return False
        if _identity_token_match(
            expected[expected_index],
            observed[observed_index],
        ) and matches(expected_index + 1, observed_index + 1):
            return True
        if expected_index + 1 < len(expected) and _identity_joined_token_match(
            expected[expected_index] + expected[expected_index + 1],
            observed[observed_index],
        ) and matches(expected_index + 2, observed_index + 1):
            return True
        if observed_index + 1 < len(observed) and _identity_joined_token_match(
            expected[expected_index],
            observed[observed_index] + observed[observed_index + 1],
        ) and matches(expected_index + 1, observed_index + 2):
            return True
        return False

    return any(matches(0, index) for index in range(len(observed)))


def _planned_pause_boundaries(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    phrases = (block.get("cadence_plan") or {}).get("phrases") or []
    return [
        {
            "time_ms": int(value.get("end_ms") or 0),
            "boundary": str(value.get("boundary_after") or "none"),
        }
        for value in phrases[:-1]
        if isinstance(value, Mapping) and value.get("end_ms") is not None
    ]


def _contains_entity(text: str, entity: str) -> bool:
    expected = _words(entity)
    observed = _words(text)
    if not expected:
        return True
    width = len(expected)
    return any(observed[index : index + width] == expected for index in range(len(observed) - width + 1))


def _source_block_id(value: Any) -> str:
    return str(value or "").split("_variant_", 1)[0]


def _midpoint_ms(word: Mapping[str, Any]) -> int:
    start = round(float(word.get("start") or 0.0) * 1000)
    end = round(float(word.get("end") or word.get("start") or 0.0) * 1000)
    return (start + end) // 2


def _block_words(
    words: Sequence[Mapping[str, Any]],
    *,
    start_ms: int,
    end_ms: int,
) -> list[Mapping[str, Any]]:
    return [word for word in words if start_ms <= _midpoint_ms(word) < end_ms]


def _timing_sanity_partitions(
    partitions: Sequence[Sequence[Mapping[str, Any]]],
    blocks: Sequence[Mapping[str, Any]],
    *,
    tolerance_ms: int = 600,
) -> tuple[list[list[Mapping[str, Any]]], list[dict[str, Any]]]:
    """Move globally aligned words that are grossly outside a block window.

    Global text alignment intentionally tolerates boundary drift, but repeated
    greetings can make an identical word align to the following block several
    seconds too early. Timing is used only as a broad sanity check, never as the
    original rigid partition rule.
    """

    adjusted = [list(value) for value in partitions]
    corrections: list[dict[str, Any]] = []
    for index in range(1, min(len(adjusted), len(blocks))):
        start_ms = int(blocks[index].get("start_ms") or 0)
        while (
            adjusted[index]
            and _midpoint_ms(adjusted[index][0]) < start_ms - tolerance_ms
        ):
            word = adjusted[index].pop(0)
            target_index = index - 1
            while (
                target_index > 0
                and _midpoint_ms(word)
                < int(blocks[target_index].get("start_ms") or 0) - tolerance_ms
            ):
                target_index -= 1
            adjusted[target_index].append(word)
            corrections.append(
                {
                    "word": str(word.get("text") or ""),
                    "from_block_index": index,
                    "to_block_index": target_index,
                    "word_midpoint_ms": _midpoint_ms(word),
                    "next_block_start_ms": start_ms,
                    "reason": "aligned_word_precedes_next_block_beyond_tolerance",
                }
            )
    return adjusted, corrections


def _pcm_rms_ratio(data: bytes, sample_width: int) -> float:
    """Return PCM RMS as a fraction of full scale without external codecs."""

    if not data:
        return 0.0
    if sample_width == 1:
        samples = [value - 128 for value in data]
        full_scale = 127
    elif sample_width in (2, 4):
        typecode = "h" if sample_width == 2 else "i"
        values = array(typecode)
        values.frombytes(data[: len(data) - len(data) % sample_width])
        if sys.byteorder != "little":
            values.byteswap()
        samples = values
        full_scale = (1 << (sample_width * 8 - 1)) - 1
    elif sample_width == 3:
        samples = [
            int.from_bytes(data[index : index + 3], "little", signed=True)
            for index in range(0, len(data) - 2, 3)
        ]
        full_scale = (1 << 23) - 1
    else:
        raise ValueError("unsupported PCM sample width")
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples)) / full_scale


def _audio_silence_evidence(
    audio_path: Path | None,
    *,
    start_ms: int,
    end_ms: int,
    silence_dbfs: float = -45.0,
    frame_ms: int = 20,
) -> dict[str, Any] | None:
    """Measure the longest real PCM silence inside an STT word gap.

    The isolated dialogue bus is preferred by the caller. This avoids treating
    background music as speech and prevents an unrecognized spoken phrase from
    becoming a fabricated pause blocker.
    """

    if audio_path is None or not audio_path.is_file() or end_ms <= start_ms:
        return None
    try:
        with wave.open(str(audio_path), "rb") as source:
            if source.getcomptype() != "NONE":
                return None
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            if sample_rate <= 0 or channels <= 0 or sample_width not in (1, 2, 3, 4):
                return None
            first_frame = max(0, round(start_ms * sample_rate / 1000))
            final_frame = min(
                source.getnframes(),
                round(end_ms * sample_rate / 1000),
            )
            if final_frame <= first_frame:
                return None
            source.setpos(first_frame)
            frames_per_window = max(1, round(frame_ms * sample_rate / 1000))
            silence_ratio = 10 ** (silence_dbfs / 20.0)
            cursor_frame = first_frame
            current_start: int | None = None
            longest_start = first_frame
            longest_end = first_frame
            while cursor_frame < final_frame:
                frame_count = min(frames_per_window, final_frame - cursor_frame)
                data = source.readframes(frame_count)
                if not data:
                    break
                silent = _pcm_rms_ratio(data, sample_width) <= silence_ratio
                if silent and current_start is None:
                    current_start = cursor_frame
                if not silent and current_start is not None:
                    if cursor_frame - current_start > longest_end - longest_start:
                        longest_start, longest_end = current_start, cursor_frame
                    current_start = None
                cursor_frame += frame_count
            if current_start is not None and cursor_frame - current_start > longest_end - longest_start:
                longest_start, longest_end = current_start, cursor_frame
    except (EOFError, OSError, ValueError, wave.Error):
        return None

    silence_start_ms = round(longest_start * 1000 / sample_rate)
    silence_end_ms = round(longest_end * 1000 / sample_rate)
    return {
        "path": str(audio_path),
        "method": "dialogue_pcm_silence_v1",
        "threshold_dbfs": silence_dbfs,
        "start_ms": silence_start_ms,
        "end_ms": silence_end_ms,
        "duration_ms": max(0, silence_end_ms - silence_start_ms),
    }


def _internal_pause_observations(
    words: Sequence[Mapping[str, Any]],
    *,
    expected_text: str,
    threshold_ms: int,
    planned_boundaries: Sequence[Mapping[str, Any]] = (),
    dialogue_audio_path: Path | None = None,
) -> list[dict[str, Any]]:
    pauses: list[dict[str, Any]] = []
    expected_tokens = re.findall(r"\S+", expected_text)
    expected_normalized = [_normalise_word(value) for value in expected_tokens]
    expected_cursor = 0
    for left, right in zip(words, words[1:]):
        left_end = round(float(left.get("end") or 0.0) * 1000)
        right_start = round(float(right.get("start") or 0.0) * 1000)
        gap_ms = max(0, right_start - left_end)
        left_word = _normalise_word(str(left.get("text") or ""))
        matched_index: int | None = None
        for index in range(expected_cursor, len(expected_normalized)):
            if expected_normalized[index] == left_word:
                matched_index = index
                expected_cursor = index + 1
                break
        boundary = "none"
        allowed_ms = threshold_ms
        midpoint_ms = (left_end + right_start) // 2
        planned = min(
            planned_boundaries,
            key=lambda value: abs(int(value.get("time_ms") or 0) - midpoint_ms),
            default=None,
        )
        if planned is not None and abs(
            int(planned.get("time_ms") or 0) - midpoint_ms
        ) <= 650:
            boundary = str(planned.get("boundary") or "none")
        if matched_index is not None:
            token = expected_tokens[matched_index].rstrip('"”’\')]}')
            if boundary == "none" and token.endswith((".", "!", "?", "…")):
                boundary = "terminal"
            elif boundary == "none" and token.endswith((",", ";", ":")):
                boundary = "clause"
        if boundary == "terminal":
            allowed_ms = max(1800, threshold_ms)
        elif boundary == "clause":
            allowed_ms = max(1300, threshold_ms)
        if gap_ms < allowed_ms:
            continue
        audio_evidence = _audio_silence_evidence(
            dialogue_audio_path,
            start_ms=left_end,
            end_ms=right_start,
        )
        if audio_evidence is not None:
            measured_pause_ms = int(audio_evidence["duration_ms"])
            if measured_pause_ms < allowed_ms:
                continue
            pause_start_ms = int(audio_evidence["start_ms"])
            pause_end_ms = int(audio_evidence["end_ms"])
            validation_method = str(audio_evidence["method"])
        else:
            measured_pause_ms = gap_ms
            pause_start_ms = left_end
            pause_end_ms = right_start
            validation_method = "stt_word_gap_fallback"
        pauses.append(
            {
                "after": str(left.get("text") or ""),
                "before": str(right.get("text") or ""),
                "start_ms": pause_start_ms,
                "end_ms": pause_end_ms,
                "duration_ms": measured_pause_ms,
                "observed_word_gap_ms": gap_ms,
                "expected_boundary": boundary,
                "allowed_duration_ms": allowed_ms,
                "validation_method": validation_method,
                "audio_evidence_path": (
                    str(audio_evidence["path"])
                    if audio_evidence is not None
                    else None
                ),
            }
        )
    return pauses


def _has_production_pause_blocker(
    pauses: Sequence[Mapping[str, Any]],
) -> bool:
    """Separate audible stoppages from small STT boundary disagreement.

    Azure word timestamps can place a natural clause/date breath 100-250 ms
    beyond the planned punctuation allowance. Keep that visible as a quality
    warning, but reserve the production blocker for a pause over 1.5 seconds
    and at least 250 ms beyond its context-specific allowance.
    """

    return any(
        int(value.get("duration_ms") or 0)
        > max(
            1500,
            int(value.get("allowed_duration_ms") or 0) + 250,
        )
        for value in pauses
    )


def _local_word_rates(words: Sequence[Mapping[str, Any]], chunk_size: int = 5) -> list[float]:
    rates: list[float] = []
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least two")
    for index in range(0, len(words), chunk_size):
        chunk = words[index : index + chunk_size]
        if len(chunk) < 2:
            continue
        start = float(chunk[0].get("start") or 0.0)
        end = float(chunk[-1].get("end") or chunk[-1].get("start") or 0.0)
        duration = end - start
        if duration > 0:
            rates.append(len(chunk) / duration)
    return rates


def _coefficient_of_variation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def _effective_tempo_factor(block: Mapping[str, Any]) -> float:
    rate = float(block.get("selected_azure_rate_percent") or 0.0)
    stretch = float(block.get("post_synthesis_time_stretch_ratio") or 1.0)
    return max(0.01, (1.0 + rate / 100.0) * stretch)


@dataclass(frozen=True)
class DubMasteringThresholds:
    maximum_aggregate_wer: float = 0.40
    maximum_block_wer: float = 0.55
    minimum_production_score: float = 82.0
    maximum_internal_pause_ms: int = 900
    maximum_tempo_cv: float = 0.36
    maximum_same_speaker_tempo_jump: float = 0.14
    maximum_time_stretch_ratio: float = 1.08
    minimum_time_stretch_ratio: float = 1 / 1.08
    maximum_rate_delta_percent: int = 10
    maximum_cross_speaker_overlap_ms: int = 250
    maximum_local_anchor_drift: float = 0.30
    maximum_overrides_per_round: int = 8

    def validate(self) -> None:
        if not 0 <= self.maximum_aggregate_wer <= 2:
            raise ValueError("maximum aggregate WER is invalid")
        if not 0 <= self.maximum_block_wer <= 2:
            raise ValueError("maximum block WER is invalid")
        if not 0 <= self.minimum_production_score <= 100:
            raise ValueError("minimum production score is invalid")
        if self.maximum_internal_pause_ms < 100:
            raise ValueError("maximum internal pause is too small")
        if not 0 <= self.maximum_local_anchor_drift <= 1:
            raise ValueError("maximum local anchor drift is invalid")
        if self.maximum_overrides_per_round < 1:
            raise ValueError("maximum overrides per round must be positive")


def analyze_mastered_dub(
    *,
    job_id: str,
    direct_dub_report: Mapping[str, Any],
    normalized_stt: Mapping[str, Any],
    audio_path: Path,
    round_number: int,
    thresholds: DubMasteringThresholds | None = None,
) -> dict[str, Any]:
    """Score the final rendered mix using Azure-recognized words and timestamps."""

    thresholds = thresholds or DubMasteringThresholds()
    thresholds.validate()
    dialogue_audio_path = audio_path.with_name("dialogue.wav")
    if not dialogue_audio_path.is_file():
        dialogue_audio_path = audio_path
    stt_words = [
        dict(value)
        for value in normalized_stt.get("words") or []
        if isinstance(value, Mapping)
    ]
    raw_blocks = [
        dict(value)
        for value in direct_dub_report.get("blocks") or []
        if isinstance(value, Mapping)
    ]
    tts_references = [
        str(
            value.get("tts_text")
            or (value.get("translation_variant") or {}).get("tts_text")
            or (value.get("translation_variant") or {}).get("spoken_text")
            or value.get("translated_text")
            or ""
        ).strip()
        for value in raw_blocks
    ]
    aligned_block_words, alignment_audit = _word_alignment_partitions(
        tts_references,
        stt_words,
    )
    aligned_block_words, timing_sanity_corrections = _timing_sanity_partitions(
        aligned_block_words,
        raw_blocks,
    )
    alignment_audit["timing_sanity_tolerance_ms"] = 600
    alignment_audit["timing_sanity_corrections"] = timing_sanity_corrections
    observations: list[dict[str, Any]] = []
    expected_word_total = 0
    weighted_wer_total = 0.0
    missing_entity_total = 0

    for index, block in enumerate(raw_blocks):
        start_ms = int(block.get("start_ms") or 0)
        end_ms = int(block.get("end_ms") or start_ms)
        timestamp_words = _block_words(stt_words, start_ms=start_ms, end_ms=end_ms)
        recognized_words = aligned_block_words[index]
        hypothesis = " ".join(str(word.get("text") or "") for word in recognized_words).strip()
        expected_spoken = str(
            (block.get("translation_variant") or {}).get("spoken_text")
            or block.get("translated_text")
            or block.get("tts_text")
            or ""
        ).strip()
        source_text = str(block.get("source_text") or "").strip()
        local_order_anomalies = _local_information_order_anomalies(
            source_text,
            expected_spoken,
            maximum_drift=thresholds.maximum_local_anchor_drift,
        )
        expected_tts = tts_references[index] or expected_spoken
        exact_spoken_wer = word_error_rate(expected_spoken, hypothesis)
        exact_tts_wer = word_error_rate(expected_tts, hypothesis)
        spoken_wer = delivery_word_error_rate(expected_spoken, hypothesis)
        tts_wer = delivery_word_error_rate(expected_tts, hypothesis)
        if tts_wer <= spoken_wer:
            expected = expected_tts
            reference_used = "tts_text"
            wer = tts_wer
            exact_wer = exact_tts_wer
        else:
            expected = expected_spoken
            reference_used = "spoken_text"
            wer = spoken_wer
            exact_wer = exact_spoken_wer
        expected_count = max(1, len(_words(expected)))
        expected_word_total += expected_count
        weighted_wer_total += wer * expected_count
        protected = [
            str(value)
            for value in block.get("protected_entities") or []
            if str(value).strip()
        ]
        assessed_entities: list[dict[str, Any]] = []
        missing: list[str] = []
        unassessable: list[str] = []
        for entity in protected:
            surface = _delivery_identity_surface_tokens(
                entity,
                expected_spoken,
                expected_tts,
            )
            if not surface:
                unassessable.append(entity)
                continue
            recognized = _contains_identity_surface(hypothesis, surface)
            assessed_entities.append(
                {
                    "entity": entity,
                    "expected_surface_tokens": surface,
                    "recognized": recognized,
                    "match_policy": "expected-surface-fuzzy-v1",
                }
            )
            if not recognized:
                missing.append(entity)
        missing_entity_total += len(missing)
        pauses = (
            _internal_pause_observations(
                recognized_words,
                expected_text=expected,
                threshold_ms=thresholds.maximum_internal_pause_ms,
                planned_boundaries=_planned_pause_boundaries(block),
                dialogue_audio_path=dialogue_audio_path,
            )
            if wer <= 0.35
            and len(recognized_words) >= max(2, round(expected_count * 0.70))
            else []
        )
        local_rates = _local_word_rates(recognized_words)
        tempo_cv = _coefficient_of_variation(local_rates)
        effective_tempo = _effective_tempo_factor(block)
        stretch = float(block.get("post_synthesis_time_stretch_ratio") or 1.0)
        base_rate = int((block.get("base_prosody") or {}).get("rate_percent") or 0)
        selected_rate = int(block.get("selected_azure_rate_percent") or 0)
        rate_delta = selected_rate - base_rate
        reason_codes: list[str] = []
        if wer > thresholds.maximum_block_wer:
            reason_codes.append("high_word_error_rate")
        if missing:
            reason_codes.append("protected_entity_not_recognized")
        if pauses:
            reason_codes.append("unexpected_internal_pause")
        if tempo_cv > thresholds.maximum_tempo_cv:
            reason_codes.append("within_block_tempo_instability")
        if stretch > thresholds.maximum_time_stretch_ratio:
            reason_codes.append("excessive_time_compression")
        if stretch < thresholds.minimum_time_stretch_ratio:
            reason_codes.append("excessive_time_expansion")
        if abs(rate_delta) > thresholds.maximum_rate_delta_percent:
            reason_codes.append("azure_rate_delta_excessive")
        if local_order_anomalies:
            reason_codes.append("local_information_order_drift")

        production_blockers: list[str] = []
        if wer > thresholds.maximum_block_wer and expected_count >= 4:
            production_blockers.append("high_word_error_rate")
        if missing:
            production_blockers.append("protected_entity_not_recognized")
        if _has_production_pause_blocker(pauses):
            production_blockers.append("unexpected_internal_pause")
        if tempo_cv > max(0.60, thresholds.maximum_tempo_cv):
            production_blockers.append("within_block_tempo_instability")
        if stretch > max(1.15, thresholds.maximum_time_stretch_ratio):
            production_blockers.append("excessive_time_compression")
        if stretch < min(0.84, thresholds.minimum_time_stretch_ratio):
            production_blockers.append("excessive_time_expansion")
        if abs(rate_delta) > max(14, thresholds.maximum_rate_delta_percent):
            production_blockers.append("azure_rate_delta_excessive")
        if local_order_anomalies:
            production_blockers.append("local_information_order_drift")

        previous = raw_blocks[index - 1] if index else None
        same_speaker_jump = 0.0
        if previous is not None and previous.get("speaker_id") == block.get("speaker_id"):
            previous_tempo = _effective_tempo_factor(previous)
            same_speaker_jump = abs(effective_tempo / previous_tempo - 1.0)
            if same_speaker_jump > thresholds.maximum_same_speaker_tempo_jump:
                reason_codes.append("same_speaker_tempo_jump")
            if same_speaker_jump > max(
                0.18,
                thresholds.maximum_same_speaker_tempo_jump,
            ):
                production_blockers.append("same_speaker_tempo_jump")

        cross_speaker_overlap_ms = 0
        if previous is not None and previous.get("speaker_id") != block.get("speaker_id"):
            cross_speaker_overlap_ms = max(
                0,
                int(previous.get("end_ms") or 0) - start_ms,
            )
            if cross_speaker_overlap_ms > thresholds.maximum_cross_speaker_overlap_ms:
                reason_codes.append("cross_speaker_timeline_collision")
                production_blockers.append("cross_speaker_timeline_collision")

        observations.append(
            {
                "block_id": _source_block_id(block.get("block_id")),
                "rendered_block_id": str(block.get("block_id") or ""),
                "block_index": index,
                "speaker_id": str(block.get("speaker_id") or ""),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "expected_text": expected,
                "source_text": source_text,
                "expected_spoken_text": expected_spoken,
                "expected_tts_text": expected_tts,
                "quality_reference_used": reference_used,
                "transcribed_text": hypothesis,
                "recognized_word_timeline": [
                    {
                        "text": str(word.get("text") or ""),
                        "start": word.get("start"),
                        "end": word.get("end"),
                        "confidence": word.get("confidence"),
                    }
                    for word in recognized_words
                ],
                "local_information_order_anomalies": local_order_anomalies,
                "expected_word_count": expected_count,
                "recognized_word_count": len(recognized_words),
                "timestamp_window_word_count": len(timestamp_words),
                "word_partition_method": alignment_audit["method"],
                "word_error_rate": round(wer, 6),
                "exact_word_error_rate": round(exact_wer, 6),
                "spoken_text_word_error_rate": round(spoken_wer, 6),
                "tts_text_word_error_rate": round(tts_wer, 6),
                "exact_spoken_text_word_error_rate": round(exact_spoken_wer, 6),
                "exact_tts_text_word_error_rate": round(exact_tts_wer, 6),
                "word_error_policy": "token-boundary-and-close-orthography-v1",
                "protected_entities": protected,
                "protected_entities_assessed": assessed_entities,
                "protected_entities_not_assessable": unassessable,
                "protected_entities_missing": missing,
                "internal_pauses": pauses,
                "maximum_internal_pause_ms": max(
                    (int(item["duration_ms"]) for item in pauses),
                    default=0,
                ),
                "local_word_rates": [round(value, 6) for value in local_rates],
                "tempo_coefficient_of_variation": round(tempo_cv, 6),
                "effective_tempo_factor": round(effective_tempo, 6),
                "same_speaker_tempo_jump": round(same_speaker_jump, 6),
                "cross_speaker_overlap_ms": cross_speaker_overlap_ms,
                "selected_variant_id": str(
                    (block.get("translation_variant") or {}).get("selected_variant_id")
                    or block.get("selected_variant_id")
                    or "natural"
                ),
                "available_variant_ids": [
                    str(value.get("variant_id") or "")
                    for value in block.get("variants") or []
                    if isinstance(value, Mapping) and value.get("meaning_preserved", True)
                ],
                "available_variants": [
                    {
                        "variant_id": str(value.get("variant_id") or ""),
                        "tts_text_sha256": _tts_text_fingerprint(
                            str(value.get("tts_text") or value.get("spoken_text") or "")
                        ),
                        "audio_equivalent_to_selected": (
                            _tts_text_fingerprint(
                                str(value.get("tts_text") or value.get("spoken_text") or "")
                            )
                            == _tts_text_fingerprint(expected_tts)
                        ),
                    }
                    for value in block.get("variants") or []
                    if isinstance(value, Mapping) and value.get("meaning_preserved", True)
                ],
                "selected_azure_rate_percent": selected_rate,
                "base_rate_percent": base_rate,
                "rate_delta_percent": rate_delta,
                "post_synthesis_time_stretch_ratio": round(stretch, 6),
                "reason_codes": list(dict.fromkeys(reason_codes)),
                "production_blockers": list(dict.fromkeys(production_blockers)),
                "passed": not reason_codes,
                "production_gate_passed": not production_blockers,
            }
        )

    aggregate_wer = weighted_wer_total / expected_word_total if expected_word_total else 0.0
    failed = [value for value in observations if not value["passed"]]
    production_failed = [
        value for value in observations if not value["production_gate_passed"]
    ]
    pause_count = sum(len(value["internal_pauses"]) for value in observations)
    tempo_failure_count = sum(
        any("tempo" in code or "compression" in code or "expansion" in code for code in value["reason_codes"])
        for value in observations
    )
    collision_count = sum(
        "cross_speaker_timeline_collision" in value["reason_codes"]
        for value in observations
    )
    local_information_order_failure_count = sum(
        "local_information_order_drift" in value["production_blockers"]
        for value in observations
    )
    production_pause_count = sum(
        "unexpected_internal_pause" in value["production_blockers"]
        for value in observations
    )
    production_tempo_failure_count = sum(
        any(
            "tempo" in code or "compression" in code or "expansion" in code
            for code in value["production_blockers"]
        )
        for value in observations
    )
    assessed_entity_count = sum(
        len(value["protected_entities_assessed"]) for value in observations
    )
    intelligibility_score = max(0.0, 100.0 * (1.0 - min(1.0, aggregate_wer)))
    identity_score = (
        max(
            0.0,
            100.0 * (1.0 - missing_entity_total / assessed_entity_count),
        )
        if assessed_entity_count
        else 100.0
    )
    pause_severity = sum(
        max(
            0.0,
            (
                float(item.get("duration_ms") or 0)
                - float(item.get("allowed_duration_ms") or 0)
            )
            / max(1.0, float(item.get("allowed_duration_ms") or 1)),
        )
        for value in observations
        for item in value["internal_pauses"]
    )
    pause_score = max(
        0.0,
        100.0 * (1.0 - min(1.0, pause_severity / max(1, len(observations)))),
    )
    tempo_severity = 0.0
    for value in observations:
        stretch = float(value["post_synthesis_time_stretch_ratio"])
        jump = float(value["same_speaker_tempo_jump"])
        coefficient = float(value["tempo_coefficient_of_variation"])
        tempo_severity += min(
            1.0,
            max(0.0, abs(stretch - 1.0) - 0.08) / 0.50
            + max(0.0, jump - 0.14) / 0.40
            + max(0.0, coefficient - 0.36) / 0.64,
        )
    tempo_score = max(
        0.0,
        100.0 * (1.0 - min(1.0, tempo_severity / max(1, len(observations)))),
    )
    collision_score = max(
        0.0,
        100.0 * (1.0 - collision_count / max(1, len(observations))),
    )
    local_semantic_alignment_score = (
        0.0 if local_information_order_failure_count else 100.0
    )
    dimensions = {
        "intelligibility": round(intelligibility_score, 2),
        "protected_identity_recognition": round(identity_score, 2),
        "pause_naturalness": round(pause_score, 2),
        "tempo_naturalness": round(tempo_score, 2),
        "speaker_collision_safety": round(collision_score, 2),
        "local_semantic_alignment": round(local_semantic_alignment_score, 2),
    }
    weights = {
        "intelligibility": 0.35,
        "protected_identity_recognition": 0.15,
        "pause_naturalness": 0.10,
        "tempo_naturalness": 0.15,
        "speaker_collision_safety": 0.05,
        "local_semantic_alignment": 0.20,
    }
    score = round(
        sum(dimensions[key] * weight for key, weight in weights.items()),
        2,
    )
    production_ready = (
        score >= thresholds.minimum_production_score
        and aggregate_wer <= thresholds.maximum_aggregate_wer
        and missing_entity_total == 0
        and production_pause_count == 0
        and production_tempo_failure_count == 0
        and collision_count == 0
        and local_information_order_failure_count == 0
        and not production_failed
    )
    return {
        "schema_version": DUB_MASTERING_ROUND_SCHEMA_VERSION,
        "policy_version": DUB_MASTERING_POLICY_VERSION,
        "job_id": job_id,
        "round": round_number,
        "generated_at": _utcnow(),
        "audio": {
            "path": str(audio_path),
            "sha256": checksum(audio_path),
        },
        "azure_stt": {
            "provider": normalized_stt.get("provider"),
            "recognized_word_count": len(stt_words),
            "recognized_phrase_count": len(normalized_stt.get("phrases") or []),
        },
        "word_alignment": alignment_audit,
        "thresholds": asdict(thresholds),
        "production_score": score,
        "quality_dimensions": dimensions,
        "quality_dimension_weights": weights,
        "score_interpretation": (
            "Weighted whole-program quality estimate; production_ready also "
            "requires every severe local blocker to pass"
        ),
        "production_ready": production_ready,
        "aggregate_word_error_rate": round(aggregate_wer, 6),
        "block_count": len(observations),
        "failed_block_count": len(failed),
        "production_blocker_count": len(production_failed),
        "missing_protected_entity_count": missing_entity_total,
        "assessed_protected_entity_count": assessed_entity_count,
        "unexpected_pause_count": pause_count,
        "production_pause_count": production_pause_count,
        "tempo_failure_count": tempo_failure_count,
        "production_tempo_failure_count": production_tempo_failure_count,
        "collision_count": collision_count,
        "local_information_order_failure_count": (
            local_information_order_failure_count
        ),
        "blocks": observations,
    }


def _next_variant(
    current: str,
    available: Sequence[str],
    *,
    direction: str,
) -> str | None:
    values = [value for value in _VARIANT_ORDER if value in set(available)]
    if current not in values:
        # Renderer-local AI revisions are not members of the approved prepared
        # variant ladder.  Guessing their rank used to turn a request for a
        # shorter delivery into ``natural`` (the richest variant), which could
        # destabilise several neighbouring blocks in the following STT round.
        # Keep the last known-good audio when its duration rank is unknown.
        return None
    index = values.index(current)
    target = index + (1 if direction == "shorter" else -1)
    return values[target] if 0 <= target < len(values) else None


def _next_audio_distinct_variant(
    current: str,
    available: Sequence[str],
    variant_details: Sequence[Mapping[str, Any]],
    *,
    direction: str,
) -> str | None:
    equivalence = {
        str(value.get("variant_id") or ""): bool(
            value.get("audio_equivalent_to_selected")
        )
        for value in variant_details
        if isinstance(value, Mapping)
    }
    cursor = current
    while selected := _next_variant(cursor, available, direction=direction):
        if not equivalence.get(selected, False):
            return selected
        cursor = selected
    return None


def build_mastering_override_plan(
    *,
    job_id: str,
    audit: Mapping[str, Any],
    previous_payload: Mapping[str, Any] | None = None,
    thresholds: DubMasteringThresholds | None = None,
) -> dict[str, Any]:
    """Choose bounded approved-variant changes from measured failures."""

    thresholds = thresholds or DubMasteringThresholds()
    previous_payload = dict(previous_payload or {})
    previous = {
        _source_block_id(item.get("block_id")): dict(item)
        for item in previous_payload.get("overrides") or []
        if isinstance(item, Mapping)
    }
    proposals: list[dict[str, Any]] = []
    for block in audit.get("blocks") or []:
        if not isinstance(block, Mapping):
            continue
        legacy_reasons = [str(value) for value in block.get("reason_codes") or []]
        if block.get("production_gate_passed") is True:
            continue
        reason_codes = [
            str(value) for value in block.get("production_blockers") or []
        ]
        if "production_gate_passed" not in block:
            reason_codes = legacy_reasons
        if not reason_codes:
            continue
        current = str(block.get("selected_variant_id") or "natural")
        available = [str(value) for value in block.get("available_variant_ids") or []]
        if current not in _VARIANT_ORDER:
            # Final-conductor variants have measured audio but no stable place
            # in the natural/concise/compact semantic ladder.  They can only be
            # replaced after an explicit duration comparison, not by ordinal
            # inference in the mastering feedback loop.
            continue
        direction: str | None = None
        if any(
            code in reason_codes
            for code in (
                "excessive_time_compression",
                "azure_rate_delta_excessive",
                "cross_speaker_timeline_collision",
            )
        ):
            direction = "shorter"
        elif "excessive_time_expansion" in reason_codes:
            direction = "richer"
        elif "same_speaker_tempo_jump" in reason_codes:
            factor = float(block.get("effective_tempo_factor") or 1.0)
            direction = "shorter" if factor > 1.0 else "richer"
        elif "high_word_error_rate" in reason_codes:
            direction = "richer" if current != "natural" else None
        elif "unexpected_internal_pause" in reason_codes:
            direction = "shorter"
        elif "protected_entity_not_recognized" in reason_codes:
            # All approved variants preserve protected names verbatim. Moving
            # along the semantic richness ladder cannot repair pronunciation
            # and previously spent the only repair round on unrelated wording.
            direction = None
        if direction is None:
            continue
        selected = _next_audio_distinct_variant(
            current,
            available,
            [
                value
                for value in block.get("available_variants") or []
                if isinstance(value, Mapping)
            ],
            direction=direction,
        )
        if selected is None or selected == current:
            continue
        block_id = _source_block_id(block.get("block_id"))
        installed = previous.get(block_id)
        if (
            installed
            and str(installed.get("strategy") or "") == "select_variant"
            and str(installed.get("variant_id") or "") == selected
        ):
            # The renderer did not turn the previous instruction into changed
            # audio. Repeating the same override would only spend another STT
            # round on the same checksum.
            continue
        severity = (
            float(block.get("word_error_rate") or 0.0) * 100
            + float(block.get("maximum_internal_pause_ms") or 0) / 100
            + float(block.get("same_speaker_tempo_jump") or 0.0) * 50
            + float(block.get("cross_speaker_overlap_ms") or 0) / 100
            + (
                100
                if "cross_speaker_timeline_collision" in reason_codes
                else 0
            )
            + (
                50
                if "protected_entity_not_recognized" in reason_codes
                else 0
            )
        )
        proposals.append(
            {
                "block_id": block_id,
                "strategy": "select_variant",
                "variant_id": selected,
                "previous_variant_id": current,
                "direction": direction,
                "reason_codes": reason_codes,
                "source_round": int(audit.get("round") or 0),
                "severity": round(severity, 6),
                "evidence": {
                    "word_error_rate": block.get("word_error_rate"),
                    "maximum_internal_pause_ms": block.get("maximum_internal_pause_ms"),
                    "same_speaker_tempo_jump": block.get("same_speaker_tempo_jump"),
                    "post_synthesis_time_stretch_ratio": block.get("post_synthesis_time_stretch_ratio"),
                    "cross_speaker_overlap_ms": block.get("cross_speaker_overlap_ms"),
                },
            }
        )

    proposals.sort(key=lambda item: (-float(item["severity"]), item["block_id"]))
    selected_proposals: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    index_by_id = {
        _source_block_id(value.get("block_id")): int(value.get("block_index") or 0)
        for value in audit.get("blocks") or []
        if isinstance(value, Mapping)
    }
    for proposal in proposals:
        index = index_by_id.get(proposal["block_id"], 0)
        # One change per five-block neighbourhood keeps the feedback loop from
        # creating compensating oscillations across adjacent turns.
        if any(abs(index - selected_index) <= 2 for selected_index in selected_indices):
            continue
        selected_indices.append(index)
        selected_proposals.append(proposal)
        if len(selected_proposals) >= thresholds.maximum_overrides_per_round:
            break

    merged = dict(previous)
    for proposal in selected_proposals:
        merged[proposal["block_id"]] = proposal
    revision = int(previous_payload.get("revision") or 0) + (1 if selected_proposals else 0)
    return {
        "schema_version": DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
        "policy_version": DUB_MASTERING_POLICY_VERSION,
        "job_id": job_id,
        "revision": revision,
        "created_at": _utcnow(),
        "source_round": int(audit.get("round") or 0),
        "new_override_count": len(selected_proposals),
        "overrides": [merged[key] for key in sorted(merged)],
        "new_overrides": selected_proposals,
        "authoritative_translation_changed": False,
        "speaker_assignments_changed": False,
        "timeline_boundaries_changed": False,
        "five_block_neighbourhood_control": True,
    }


class DubMasteringLoop:
    """Bounded render → Azure STT → score → approved-variant feedback loop."""

    def __init__(
        self,
        *,
        stt_backend: SpeechBackend,
        renderer: DirectAzureDubRenderer,
        qa_review_provider_factory: Callable[[], Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.stt_backend = stt_backend
        self.renderer = renderer
        self.qa_review_provider_factory = qa_review_provider_factory
        self._qa_review_provider: Any | None = None
        self.progress_callback = progress_callback

    def _emit(self, stage: str, message: str, **details: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback({"stage": stage, "message": message, **details})

    def _apply_ai_local_timeline_reviews(
        self,
        *,
        audit: dict[str, Any],
        direct_report: Mapping[str, Any],
        cache_root: Path | None = None,
    ) -> None:
        """Independently review only blocks prefiltered for local semantic drift."""

        candidates = [
            block
            for block in audit.get("blocks") or []
            if isinstance(block, dict)
            and block.get("local_information_order_anomalies")
        ]
        audit["qa_ai"] = {
            "policy": "deterministic_prefilter_then_independent_ai_review_v1",
            "candidate_count": len(candidates),
            "reviewed_count": 0,
            "rejected_count": 0,
            "error_count": 0,
            "cache_reused_count": 0,
            "provider_configured": self.qa_review_provider_factory is not None,
        }
        if not candidates:
            return
        if self.qa_review_provider_factory is None:
            for block in candidates:
                block["qa_ai_review"] = {
                    "status": "not_configured",
                    "deterministic_gate_retained": True,
                }
            return
        if self._qa_review_provider is None:
            self._qa_review_provider = self.qa_review_provider_factory()
        provider = self._qa_review_provider
        raw_by_id = {
            _source_block_id(value.get("block_id")): value
            for value in direct_report.get("blocks") or []
            if isinstance(value, Mapping)
        }
        self._emit(
            "mastering-ai-qa",
            (
                f"Reviewing {len(candidates)} timestamp-localized semantic "
                "anomaly block(s) with AI"
            ),
            candidate_count=len(candidates),
        )
        for block in candidates:
            block_id = str(block.get("block_id") or "")
            raw = raw_by_id.get(block_id, {})
            natural = next(
                (
                    str(value.get("spoken_text") or "")
                    for value in raw.get("variants") or []
                    if isinstance(value, Mapping)
                    and str(value.get("variant_id") or "") == "natural"
                ),
                str(block.get("expected_spoken_text") or ""),
            )
            required_facts = []
            for index, value in enumerate(raw.get("required_facts") or [], start=1):
                if isinstance(value, Mapping):
                    fact = dict(value)
                    fact.setdefault("fact_id", f"fact_{index:04d}")
                    required_facts.append(fact)
                elif str(value).strip():
                    required_facts.append(
                        {"fact_id": f"fact_{index:04d}", "text": str(value).strip()}
                    )
            payload = {
                "schema_version": "mathula.semantic-fit.v1",
                "operation": "semantic_fit_review",
                "unit_id": block_id,
                "source_text": str(block.get("source_text") or ""),
                "approved_faithful_translation": natural,
                "candidate": {
                    "spoken_text": str(block.get("expected_spoken_text") or ""),
                    "tts_text": str(block.get("expected_tts_text") or ""),
                    "post_tts_transcribed_text": str(
                        block.get("transcribed_text") or ""
                    ),
                    "measured_duration_ms": max(
                        0,
                        int(block.get("end_ms") or 0)
                        - int(block.get("start_ms") or 0),
                    ),
                    "omitted_or_compressed_detail": [],
                },
                "required_facts": required_facts,
                "optional_details": list(raw.get("optional_details") or []),
                "protected_entities": list(raw.get("protected_entities") or []),
                "post_tts_timeline_evidence": {
                    "review_contract_version": (
                        "post_tts_local_timeline_reference_specificity_v2"
                    ),
                    "block_start_ms": block.get("start_ms"),
                    "block_end_ms": block.get("end_ms"),
                    "deterministic_anomalies": list(
                        block.get("local_information_order_anomalies") or []
                    ),
                    "recognized_words": list(
                        block.get("recognized_word_timeline") or []
                    ),
                    "review_contract": (
                        "Check local information order and recognizable "
                        "pronunciation, not only whole-block meaning or WER."
                    ),
                },
                "content_trust": "untrusted_data",
            }
            try:
                payload_sha256 = hashlib.sha256(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                cache_path = (
                    cache_root / f"{payload_sha256}.json"
                    if cache_root is not None
                    else None
                )
                cache_reused = bool(cache_path is not None and cache_path.is_file())
                if cache_reused and cache_path is not None:
                    cached = read_json(cache_path)
                    review = dict(cached.get("review") or {})
                    metadata_dict = dict(cached.get("metadata") or {})
                    audit["qa_ai"]["cache_reused_count"] += 1
                else:
                    response = provider.review_semantic_fit(payload)
                    review = dict(getattr(response, "data", {}) or {})
                    metadata = getattr(response, "metadata", None)
                    metadata_dict = (
                        metadata.to_dict()
                        if metadata is not None
                        and callable(getattr(metadata, "to_dict", None))
                        else {}
                    )
                    if cache_path is not None:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_write_json(
                            cache_path,
                            {
                                "schema_version": "mathula-post-tts-ai-qa-cache-v1",
                                "payload_sha256": payload_sha256,
                                "review": review,
                                "metadata": metadata_dict,
                            },
                        )
                accepted = review.get("accepted") is True
                block["qa_ai_review"] = {
                    "status": "accepted" if accepted else "rejected",
                    "review": review,
                    "metadata": metadata_dict,
                    "cache_reused": cache_reused,
                    "payload_sha256": payload_sha256,
                    "deterministic_gate_retained": True,
                }
                audit["qa_ai"]["reviewed_count"] += 1
                if not accepted:
                    audit["qa_ai"]["rejected_count"] += 1
                    block["reason_codes"] = list(
                        dict.fromkeys(
                            [
                                *block.get("reason_codes", []),
                                "qa_ai_local_timeline_rejection",
                            ]
                        )
                    )
                    block["production_blockers"] = list(
                        dict.fromkeys(
                            [
                                *block.get("production_blockers", []),
                                "qa_ai_local_timeline_rejection",
                            ]
                        )
                    )
            except Exception as exc:
                audit["qa_ai"]["error_count"] += 1
                block["qa_ai_review"] = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "deterministic_gate_retained": True,
                }

    def _configure_protected_entity_phrases(
        self,
        direct_report: Mapping[str, Any],
        *,
        job_root: Path | None = None,
    ) -> tuple[str, ...]:
        """Bias mastering STT toward the identities it is required to audit."""

        phrases: list[str] = []
        seen: set[str] = set()
        for block in direct_report.get("blocks") or []:
            if not isinstance(block, Mapping):
                continue
            for value in block.get("protected_entities") or []:
                phrase = " ".join(str(value).split()).strip()
                key = phrase.casefold()
                if phrase and key not in seen:
                    seen.add(key)
                    phrases.append(phrase)
        # Older direct-dub reports may predate entity propagation. The binding
        # artifact is the authoritative job inventory and must still drive the
        # final English-STT publication gate.
        bindings_path = (
            job_root / "analysis" / "entity_bindings.json"
            if job_root is not None
            else None
        )
        if bindings_path is not None and bindings_path.is_file():
            bindings = read_json(bindings_path)
            for unit in (bindings.get("per_unit_bindings") or {}).values():
                if not isinstance(unit, Mapping):
                    continue
                for match in unit.get("matches") or ():
                    if not isinstance(match, Mapping):
                        continue
                    phrase = " ".join(
                        str(
                            match.get("canonical_text")
                            or match.get("matched_source_text")
                            or ""
                        ).split()
                    ).strip()
                    key = phrase.casefold()
                    if phrase and key not in seen:
                        seen.add(key)
                        phrases.append(phrase)
        target = getattr(self.stt_backend, "fast", self.stt_backend)
        if hasattr(target, "phrase_list"):
            existing = [
                " ".join(str(value).split()).strip()
                for value in getattr(target, "phrase_list", ()) or ()
            ]
            combined = list(dict.fromkeys(value for value in (*existing, *phrases) if value))
            object.__setattr__(target, "phrase_list", tuple(combined))
            if getattr(target, "phrase_biasing_weight", None) is None:
                object.__setattr__(target, "phrase_biasing_weight", 1.6)
            return tuple(combined)
        return ()

    def _transcribe_cached(
        self,
        *,
        audio_path: Path,
        locale: str,
        mastering_root: Path,
        round_number: int,
        phrase_list: Sequence[str] = (),
    ) -> tuple[dict[str, Any], Path, bool]:
        audio_sha = checksum(audio_path)
        phrase_fingerprint = hashlib.sha256(
            (locale + "\n" + "\n".join(phrase_list)).encode("utf-8")
        ).hexdigest()[:12]
        cache_root = mastering_root / "stt_cache"
        cache_path = cache_root / f"{audio_sha}-{phrase_fingerprint}.json"
        if cache_path.is_file():
            return read_json(cache_path), cache_path, True
        self._emit(
            "mastering_stt",
            f"Round {round_number}: transcribing the rendered final mix with Azure STT",
            round=round_number,
            audio_path=str(audio_path),
        )
        raw = self.stt_backend.transcribe(audio_path, locale)
        cache_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cache_path, raw)
        return raw, cache_path, False

    def _prepare_stt_input(
        self,
        *,
        audio_path: Path,
        mastering_root: Path,
    ) -> tuple[Path, bool]:
        """Downmix only large masters so the full timeline remains one STT call."""

        if audio_path.stat().st_size < 450 * 1024 * 1024:
            return audio_path, False
        source_sha = checksum(audio_path)
        output = mastering_root / "stt_input" / f"{source_sha}-mono-16k.wav"
        if output.is_file() and output.stat().st_size > 0:
            return output, True
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.partial.wav")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            detail = " ".join((completed.stderr or "").split())[-500:]
            raise RuntimeError(f"FFmpeg could not prepare Azure STT mastering input: {detail}")
        temporary.replace(output)
        return output, True

    def run(
        self,
        job: JobManifest,
        *,
        options: DirectDubOptions,
        max_rounds: int = 2,
        audit_only: bool = False,
        thresholds: DubMasteringThresholds | None = None,
        voice_map_path: Path | None = None,
    ) -> dict[str, Any]:
        if not 0 <= max_rounds <= 5:
            raise ValueError("max_rounds must be between zero and five")
        thresholds = thresholds or DubMasteringThresholds()
        thresholds.validate()
        job_root = self.renderer.jobs.job_dir(job.job_id)
        direct_root = job_root / "direct_dub"
        mastering_root = direct_root / "mastering"
        report_path = direct_root / "report.json"
        override_path = direct_root / "mastering_overrides.json"
        mastering_root.mkdir(parents=True, exist_ok=True)
        policy_rebased = False
        retired_override_path: Path | None = None
        if override_path.is_file() and not audit_only:
            installed_overrides = read_json(override_path)
            installed_policy = str(installed_overrides.get("policy_version") or "")
            if installed_policy != DUB_MASTERING_POLICY_VERSION:
                revision = int(installed_overrides.get("revision") or 0)
                policy_label = re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "-",
                    installed_policy or "unversioned",
                ).strip("-")
                retired_override_path = (
                    mastering_root
                    / "policy_migrations"
                    / f"revision_{revision:04d}-{policy_label}.json"
                )
                atomic_write_json(retired_override_path, installed_overrides)
                retained_overrides = _safe_overrides_for_policy_migration(
                    installed_overrides,
                    installed_policy=installed_policy,
                )
                atomic_write_json(
                    override_path,
                    {
                        "schema_version": DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
                        "policy_version": DUB_MASTERING_POLICY_VERSION,
                        "job_id": job.job_id,
                        "revision": revision + 1,
                        "created_at": _utcnow(),
                        "source_round": 0,
                        "new_override_count": 0,
                        "overrides": retained_overrides,
                        "new_overrides": [],
                        "policy_rebase": {
                            "from": installed_policy or None,
                            "to": DUB_MASTERING_POLICY_VERSION,
                            "retired_payload_path": str(retired_override_path),
                            "reason": (
                                "The installed mastering policy predates "
                                "affix-aware identity matching and dialogue-"
                                "audio-confirmed production pause gates"
                            ),
                            "retained_safe_override_count": len(
                                retained_overrides
                            ),
                        },
                        "authoritative_translation_changed": False,
                        "speaker_assignments_changed": False,
                        "timeline_boundaries_changed": False,
                        "five_block_neighbourhood_control": True,
                    },
                )
                policy_rebased = True
                self._emit(
                    "mastering_policy_rebase",
                    (
                        "Retiring older mastering choices and rerendering the "
                        "approved baseline under score policy v8; safe prepared-"
                        f"variant overrides retained: {len(retained_overrides)}"
                    ),
                    retired_override_path=str(retired_override_path),
                )
                self.renderer.render(
                    job,
                    voice_map_path=voice_map_path,
                    options=options,
                    force=False,
                    progress_callback=self.progress_callback,
                )
        if not report_path.is_file():
            if audit_only:
                raise FileNotFoundError(
                    "Direct dub report is missing; run dub-azure or omit --audit-only"
                )
            self._emit("mastering_render", "Creating the initial direct dub before STT mastering")
            self.renderer.render(
                job,
                voice_map_path=voice_map_path,
                options=options,
                force=False,
                progress_callback=self.progress_callback,
            )

        rounds: list[dict[str, Any]] = []
        best_score = -1.0
        best_override_payload: dict[str, Any] = {
            "schema_version": DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
            "policy_version": DUB_MASTERING_POLICY_VERSION,
            "job_id": job.job_id,
            "revision": 0,
            "created_at": _utcnow(),
            "source_round": 0,
            "new_override_count": 0,
            "overrides": [],
            "new_overrides": [],
            "authoritative_translation_changed": False,
            "speaker_assignments_changed": False,
            "timeline_boundaries_changed": False,
            "five_block_neighbourhood_control": True,
        }
        if override_path.is_file():
            best_override_payload = read_json(override_path)
        best_round = 0
        stopped_reason = "round_budget_exhausted"

        for round_number in range(max_rounds + 1):
            direct_report = read_json(report_path)
            protected_entity_phrases = self._configure_protected_entity_phrases(
                direct_report,
                job_root=job_root,
            )
            audio_path = Path(
                str((direct_report.get("outputs") or {}).get("final_mix") or "")
            )
            if not audio_path.is_file():
                raise FileNotFoundError(f"Rendered final mix is missing: {audio_path}")
            stt_input_path, stt_input_prepared = self._prepare_stt_input(
                audio_path=audio_path,
                mastering_root=mastering_root,
            )
            raw, raw_path, cache_reused = self._transcribe_cached(
                audio_path=stt_input_path,
                locale=job.target_language,
                mastering_root=mastering_root,
                round_number=round_number,
                phrase_list=protected_entity_phrases,
            )
            normalized = normalize_azure_stt(
                raw,
                raw_reference=str(raw_path),
                provider=str(getattr(self.stt_backend, "provider", "azure-speech")),
            )
            # The target-locale transcript measures the whole isiZulu dub.
            # A second, unbiased en-ZA pass is authoritative for protected
            # English code switches and catches First/fast, Five/fever and
            # police/policy failures that zu-ZA STT can incorrectly approve.
            stt_target = getattr(self.stt_backend, "fast", self.stt_backend)
            previous_phrase_list = tuple(
                getattr(stt_target, "phrase_list", ()) or ()
            )
            try:
                if hasattr(stt_target, "phrase_list"):
                    object.__setattr__(stt_target, "phrase_list", ())
                english_raw, english_raw_path, english_cache_reused = (
                    self._transcribe_cached(
                        audio_path=stt_input_path,
                        locale="en-ZA",
                        mastering_root=mastering_root,
                        round_number=round_number,
                        phrase_list=(),
                    )
                )
            finally:
                if hasattr(stt_target, "phrase_list"):
                    object.__setattr__(
                        stt_target, "phrase_list", previous_phrase_list
                    )
            audit = analyze_mastered_dub(
                job_id=job.job_id,
                direct_dub_report=direct_report,
                normalized_stt=normalized,
                audio_path=audio_path,
                round_number=round_number,
                thresholds=thresholds,
            )
            english_entity_qa = apply_english_entity_pronunciation_gate(
                audit,
                protected_entities=protected_entity_phrases,
                raw_english_stt=english_raw,
                raw_response_path=english_raw_path,
            )
            english_entity_qa["cache_reused"] = english_cache_reused
            self._apply_ai_local_timeline_reviews(
                audit=audit,
                direct_report=direct_report,
                cache_root=mastering_root / "ai_qa_cache",
            )
            ai_qa_records = [
                consumption_record(
                    operation="post_tts_local_timeline_qa",
                    scope=(
                        f"round_{round_number:02d}:"
                        f"{str(block.get('block_id') or '')}"
                    ),
                    metadata=(block.get("qa_ai_review") or {}).get("metadata") or {},
                )
                for block in audit.get("blocks") or []
                if isinstance(block, Mapping)
                and isinstance(block.get("qa_ai_review"), Mapping)
                and (block.get("qa_ai_review") or {}).get("metadata")
            ]
            if ai_qa_records:
                update_ai_consumption_report(
                    job_root=job_root,
                    records=ai_qa_records,
                    video_duration_seconds=float(job.source_duration or 0.0),
                )
            round_root = mastering_root / f"round_{round_number:02d}"
            round_root.mkdir(parents=True, exist_ok=True)
            audit["azure_stt"]["raw_response_path"] = str(raw_path)
            audit["azure_stt"]["cache_reused"] = cache_reused
            audit["azure_stt"]["input_path"] = str(stt_input_path)
            audit["azure_stt"]["input_bytes"] = stt_input_path.stat().st_size
            audit["azure_stt"]["source_mix_bytes"] = audio_path.stat().st_size
            audit["azure_stt"]["mono_16k_prepared"] = stt_input_prepared
            audit["azure_stt"]["protected_entity_phrases"] = list(
                protected_entity_phrases
            )
            audit["azure_stt"]["request_duration_seconds"] = getattr(
                self.stt_backend,
                "request_duration_seconds",
                None,
            )
            atomic_write_json(round_root / "audit.json", audit)
            rounds.append(
                {
                    "round": round_number,
                    "production_score": audit["production_score"],
                    "production_ready": audit["production_ready"],
                    "aggregate_word_error_rate": audit["aggregate_word_error_rate"],
                    "failed_block_count": audit["failed_block_count"],
                    "production_blocker_count": audit["production_blocker_count"],
                    "local_information_order_failure_count": audit.get(
                        "local_information_order_failure_count", 0
                    ),
                    "qa_ai": dict(audit.get("qa_ai") or {}),
                    "english_code_switch_qa": dict(
                        audit.get("english_code_switch_qa") or {}
                    ),
                    "quality_dimensions": audit["quality_dimensions"],
                    "audit_path": str(round_root / "audit.json"),
                    "audio_sha256": audit["audio"]["sha256"],
                    "stt_cache_reused": cache_reused,
                }
            )
            self._emit(
                "mastering_score",
                (
                    f"Round {round_number}: production score "
                    f"{audit['production_score']:.1f}/100; "
                    f"WER {audit['aggregate_word_error_rate']:.3f}; "
                    f"{audit['production_blocker_count']} production blocker(s); "
                    f"{audit['failed_block_count']} warning block(s); "
                    f"AI local reviews {int((audit.get('qa_ai') or {}).get('reviewed_count') or 0)}"
                ),
                **rounds[-1],
            )
            if float(audit["production_score"]) > best_score:
                best_score = float(audit["production_score"])
                best_round = round_number
                best_override_payload = (
                    read_json(override_path)
                    if override_path.is_file()
                    else best_override_payload
                )
            if audit["production_ready"]:
                stopped_reason = "production_threshold_reached"
                break
            if audit_only:
                stopped_reason = "audit_only"
                break
            if round_number >= max_rounds:
                break
            previous_payload = (
                read_json(override_path)
                if override_path.is_file()
                else best_override_payload
            )
            plan = build_mastering_override_plan(
                job_id=job.job_id,
                audit=audit,
                previous_payload=previous_payload,
                thresholds=thresholds,
            )
            atomic_write_json(round_root / "repair_plan.json", plan)
            if not plan["new_override_count"]:
                stopped_reason = "no_safe_approved_variant_repair"
                break
            atomic_write_json(override_path, plan)
            self._emit(
                "mastering_repair",
                (
                    f"Round {round_number}: applying {plan['new_override_count']} "
                    "approved-variant repair(s) across separate five-block windows"
                ),
                round=round_number,
                override_count=plan["new_override_count"],
                override_path=str(override_path),
            )
            self.renderer.render(
                job,
                voice_map_path=voice_map_path,
                options=options,
                force=False,
                progress_callback=self.progress_callback,
            )
            rerendered_report = read_json(report_path)
            rerendered_audio_path = Path(
                str((rerendered_report.get("outputs") or {}).get("final_mix") or "")
            )
            rerendered_audio_sha = (
                checksum(rerendered_audio_path)
                if rerendered_audio_path.is_file()
                else None
            )
            audio_changed = rerendered_audio_sha != audit["audio"]["sha256"]
            rounds[-1]["repair_audio_changed"] = audio_changed
            rounds[-1]["repair_audio_sha256"] = rerendered_audio_sha
            rounds[-1]["repair_plan_path"] = str(round_root / "repair_plan.json")
            if not audio_changed:
                stopped_reason = "repair_render_produced_identical_audio"
                self._emit(
                    "mastering_no_change",
                    (
                        f"Round {round_number}: repair render produced the same "
                        "audio checksum; stopping before another Azure STT call"
                    ),
                    round=round_number,
                    audio_sha256=rerendered_audio_sha,
                    override_path=str(override_path),
                )
                break

        final_round = rounds[-1]["round"] if rounds else 0
        restored_best = False
        active_override_payload = (
            read_json(override_path) if override_path.is_file() else {}
        )
        active_overrides_differ = (
            active_override_payload.get("overrides")
            != best_override_payload.get("overrides")
        )
        if not audit_only and (best_round != final_round or active_overrides_differ):
            atomic_write_json(override_path, best_override_payload)
            self._emit(
                "mastering_restore",
                f"Restoring better-scoring round {best_round} instead of keeping a regression",
                best_round=best_round,
                final_round=final_round,
            )
            self.renderer.render(
                job,
                voice_map_path=voice_map_path,
                options=options,
                force=False,
                progress_callback=self.progress_callback,
            )
            restored_best = True

        final = {
            "schema_version": DUB_MASTERING_SCHEMA_VERSION,
            "policy_version": DUB_MASTERING_POLICY_VERSION,
            "job_id": job.job_id,
            "created_at": _utcnow(),
            "state": (
                "production_ready"
                if any(value["production_ready"] for value in rounds)
                else "best_effort_completed"
            ),
            "stopped_reason": stopped_reason,
            "audit_only": audit_only,
            "maximum_repair_rounds": max_rounds,
            "round_count": len(rounds),
            "best_round": best_round,
            "best_production_score": round(best_score, 2),
            "best_round_restored": restored_best,
            "prior_policy_overrides_rebased": policy_rebased,
            "retired_override_path": (
                str(retired_override_path) if retired_override_path else None
            ),
            "thresholds": asdict(thresholds),
            "rounds": rounds,
            "overrides_path": str(override_path) if override_path.is_file() else None,
            "tools": [
                "Azure Speech word-level transcription",
                "word-error and protected-entity audit",
                "timestamp-localized semantic drift prefilter and AI review",
                "pause and local-tempo analysis",
                "same-speaker tempo continuity",
                "cross-speaker collision analysis",
                "approved natural/concise/compact variants",
                "five-block timeline conductor",
                "Azure TTS measurement and content-addressed cache",
                "validated AI finalization conductor when prepared variants overflow",
                "FFmpeg exact-boundary and final-mix rendering",
            ],
            "authoritative_translation_changed": False,
            "manual_collision_panel_required": False,
        }
        atomic_write_json(mastering_root / "report.json", final)
        self._emit(
            "mastering_complete",
            (
                f"Dub mastering completed: {final['state']}; "
                f"best score {final['best_production_score']:.1f}/100"
            ),
            report_path=str(mastering_root / "report.json"),
            **final,
        )
        return final


def build_post_tts_quality_gate(
    mastering_report: Mapping[str, Any],
    *,
    report_path: Path,
    grammar_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine acoustic/STT evidence with the independent grammar gate."""

    rounds = [
        dict(value)
        for value in mastering_report.get("rounds") or []
        if isinstance(value, Mapping)
    ]
    best_round_number = int(mastering_report.get("best_round") or 0)
    best_round = next(
        (
            value
            for value in rounds
            if int(value.get("round") or 0) == best_round_number
        ),
        rounds[-1] if rounds else {},
    )
    blocker_count = int(best_round.get("production_blocker_count") or 0)
    grammar_blocker_count = (
        int(grammar_gate.get("blocking_issue_count") or 0)
        if grammar_gate is not None
        else 0
    )
    # Explicit user direction: contextual grammar QA should only flag issues,
    # never terminate the process -- it no longer contributes to whether
    # publication is authorized. Its issue count/state are still surfaced
    # below (grammar_blocker_count, grammar_quality_gate, reasons) purely for
    # visibility/human review, same as any other flagged-but-not-blocking
    # signal.
    grammar_flagged = bool(
        grammar_gate is not None
        and not (
            grammar_gate.get("state") == "passed"
            and grammar_gate.get("publication_authorized") is True
            and grammar_blocker_count == 0
        )
    )
    production_ready = bool(best_round.get("production_ready"))
    authorized = bool(
        mastering_report.get("state") == "production_ready"
        and production_ready
        and blocker_count == 0
    )
    reasons: list[str] = []
    if not rounds:
        reasons.append("missing_stt_qa_round")
    if mastering_report.get("state") != "production_ready":
        reasons.append("production_threshold_not_reached")
    if not production_ready and rounds:
        reasons.append("best_round_not_production_ready")
    if blocker_count:
        reasons.append("production_blockers_present")
    if grammar_flagged:
        reasons.append("contextual_grammar_qa_flagged")

    return {
        "schema_version": POST_TTS_QUALITY_GATE_SCHEMA_VERSION,
        "generated_at": _utcnow(),
        "job_id": str(mastering_report.get("job_id") or ""),
        "state": "passed" if authorized else "failed",
        "publication_authorized": authorized,
        "reasons": reasons,
        "mastering_report_path": str(report_path),
        "mastering_report_sha256": (
            checksum(report_path) if report_path.is_file() else None
        ),
        "best_round": best_round_number,
        "best_production_score": float(
            mastering_report.get("best_production_score") or 0.0
        ),
        "aggregate_word_error_rate": float(
            best_round.get("aggregate_word_error_rate") or 0.0
        ),
        "production_blocker_count": blocker_count + grammar_blocker_count,
        "acoustic_production_blocker_count": blocker_count,
        "grammar_blocker_count": grammar_blocker_count,
        "grammar_quality_gate": (
            {
                "state": grammar_gate.get("state"),
                "publication_authorized": grammar_gate.get(
                    "publication_authorized"
                ),
                "blocking_issue_count": grammar_blocker_count,
                "warning_issue_count": int(
                    grammar_gate.get("warning_issue_count") or 0
                ),
                "translation_sha256": grammar_gate.get("translation_sha256"),
                "run_directory": grammar_gate.get("run_directory"),
            }
            if grammar_gate is not None
            else None
        ),
        "quality_dimensions": dict(best_round.get("quality_dimensions") or {}),
        "thresholds": dict(mastering_report.get("thresholds") or {}),
        "stopped_reason": mastering_report.get("stopped_reason"),
    }


__all__ = [
    "DUB_MASTERING_POLICY_VERSION",
    "DUB_MASTERING_ROUND_SCHEMA_VERSION",
    "DUB_MASTERING_SCHEMA_VERSION",
    "POST_TTS_QUALITY_GATE_SCHEMA_VERSION",
    "DubMasteringLoop",
    "DubMasteringThresholds",
    "analyze_mastered_dub",
    "apply_english_entity_pronunciation_gate",
    "build_post_tts_quality_gate",
    "build_mastering_override_plan",
    "word_error_rate",
]
