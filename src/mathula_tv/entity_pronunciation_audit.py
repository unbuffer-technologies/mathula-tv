"""Dual-locale pronunciation audit for protected code-switched entities.

The production voice speaks an isiZulu carrier sentence while Azure en-ZA STT
judges only the protected entity surface.  The synthesized sample is passed
through the same bounded speed-up used by the direct dubbing renderer so an
entity cannot pass in an isolated, uncompressed calibration and then fail in a
published dub.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping

from .atomic_io import atomic_write_json, read_json
from .azure_stt import AzureSpeechError
from .azure_tts import AzureTTSBackend, AzureTTSRequest
from .pronunciation import (
    PronunciationDictionary,
    build_initialism_ssml_parts,
    with_default_organisation_initialisms,
)


ENTITY_PRONUNCIATION_AUDIT_SCHEMA = "mathula-entity-pronunciation-audit-v1"
ENTITY_PRONUNCIATION_AUDIT_VERSION = (
    "mathula-zu-en-code-switch-production-audit-v4-complete-english-surface"
)
DEFAULT_PRODUCTION_RATE_PERCENT = 5
DEFAULT_PRODUCTION_ATEMPO_RATIO = 1.03
_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_IGNORED_IDENTITY_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "to",
}
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ii": "2",
    "second": "2",
    "sekond": "2",
    "tesekond": "2",
    "tsekond": "2",
}
_FORBIDDEN_IDENTITY_SUBSTITUTIONS = {
    frozenset({"correction", "corruption"}),
    frozenset({"fast", "first"}),
    frozenset({"fever", "five"}),
    frozenset({"chassis", "jersey"}),
    frozenset({"district", "street"}),
    frozenset({"lettuce", "lights"}),
    frozenset({"lights", "liquids"}),
    frozenset({"police", "policy"}),
    frozenset({"service", "servite"}),
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _words(value: str) -> list[str]:
    return [match.group(0).casefold().replace("’", "'") for match in _WORD.finditer(value)]


def _skeleton(value: str) -> str:
    normalized = value.casefold()
    normalized = normalized.replace("ph", "f").replace("th", "t")
    normalized = normalized.replace("c", "k").replace("q", "k")
    normalized = normalized.translate(
        str.maketrans({"b": "p", "d": "t", "g": "k", "v": "f", "z": "s"})
    )
    normalized = re.sub(r"[aeiouywh']+", "", normalized)
    return re.sub(r"(.)\1+", r"\1", normalized)


def _accent_spelling(value: str) -> str:
    """Normalize narrow en-ZA STT vowel spellings without merging identities."""

    # Azure can spell the /ɔː/ vowel in ``Hawks`` as ``Hauks`` when the word
    # is code-switched from an isiZulu carrier. Keep this deliberately narrow:
    # broad vowel removal would make distinct English entity names identical.
    return value.replace("au", "aw")


def _token_matches(expected: str, observed: str) -> bool:
    if expected == observed:
        return True
    if frozenset({expected, observed}) in _FORBIDDEN_IDENTITY_SUBSTITUTIONS:
        return False
    if expected == "the":
        # The full English organisation name must code-switch its article too.
        # Literal text under a zu-ZA voice is heard as unvoiced ``te``; the
        # calibrated voiced forms ``Dha``/``de`` are the accepted approximation.
        # In the complete production sentence en-ZA STT writes separated
        # ``Dha`` as ``dia``; this is a measured article surface, not the old
        # fused/missing ``jahawks`` failure.
        return observed in {"the", "dha", "de", "dia"}
    if _NUMBER_WORDS.get(expected) == observed:
        return True
    if len(expected) == 1:
        return observed == expected
    if (
        expected == "hawks"
        and observed.endswith(("hoks", "hauks"))
        and len(observed) - len(expected) <= 4
    ):
        # In a fused isiZulu prefix, en-ZA STT writes the non-rhotic calibrated
        # /hɔːks/ as ``hoks`` or ``hauks``. Keep this entity-specific so
        # unrelated English vowels cannot pass via broad normalization.
        return True
    accent_expected = _accent_spelling(expected)
    accent_observed = _accent_spelling(observed)
    if (
        accent_observed.endswith(accent_expected)
        and len(accent_observed) - len(accent_expected) <= 4
    ):
        return True
    expected_skeleton = _skeleton(expected)
    observed_skeleton = _skeleton(observed)
    if (
        len(expected_skeleton) >= 3
        and expected_skeleton == observed_skeleton
        and SequenceMatcher(None, expected, observed).ratio() >= 0.30
    ):
        return True
    threshold = 0.90 if len(expected) <= 5 else 0.86
    return SequenceMatcher(None, expected, observed).ratio() >= threshold


def entity_surface_recognized(expected: str, transcription: str) -> bool:
    """Return whether en-ZA STT retained the complete meaningful entity surface."""

    expected_words = [
        value
        for value in _words(expected)
        if value not in _IGNORED_IDENTITY_WORDS or len(value) == 1
    ]
    observed_words = _words(transcription)
    if not expected_words:
        return False

    def matches(expected_index: int, observed_index: int) -> bool:
        if expected_index == len(expected_words):
            return True
        if observed_index >= len(observed_words):
            return False
        if _token_matches(
            expected_words[expected_index], observed_words[observed_index]
        ) and matches(expected_index + 1, observed_index + 1):
            return True
        if (
            expected_index + 1 < len(expected_words)
            and len(expected_words[expected_index]) > 1
            and len(expected_words[expected_index + 1]) > 1
            and expected_words[expected_index] != "the"
            and _token_matches(
            expected_words[expected_index] + expected_words[expected_index + 1],
            observed_words[observed_index],
            )
            and matches(expected_index + 2, observed_index + 1)
        ):
            return True
        if (
            observed_index + 1 < len(observed_words)
            and len(expected_words[expected_index]) > 1
            and _token_matches(
            expected_words[expected_index],
            observed_words[observed_index] + observed_words[observed_index + 1],
            )
            and matches(expected_index + 1, observed_index + 2)
        ):
            return True
        return False

    if any(matches(0, index) for index in range(len(observed_words))):
        return True

    # A leading English definite article is part of the audible code switch.
    # Do not let the broad boundary-recovery fallback merge Zulu ``te`` with
    # the following noun and silently approve only the noun pronunciation.
    if expected_words[0] == "the" and not any(
        _token_matches("the", word) for word in observed_words
    ):
        return False

    # A witness designation is an identity, not a pronunciation similarity.
    # Keep alphabetic single-letter suffixes on the literal-token path above.
    if any(len(word) == 1 and word.isalpha() for word in expected_words):
        return False

    # en-ZA STT often changes word boundaries around an isiZulu-to-English
    # switch (``Shepstone`` -> ``shaped stone``, ``Pier one`` -> ``piaon``).
    # Judge a bounded contiguous surface as a second pass so those boundary
    # choices do not become false pronunciation failures.  This remains
    # deliberately stricter than generic fuzzy matching: meaning-changing
    # substitutions above are rejected before the phonetic comparison, and a
    # candidate must retain nearly all of the expected consonant skeleton.
    expected_set = set(expected_words)
    for pair in _FORBIDDEN_IDENTITY_SUBSTITUTIONS:
        expected_member = next((word for word in pair if word in expected_set), None)
        if expected_member is None:
            continue
        forbidden = pair - {expected_member}
        if any(word in observed_words for word in forbidden):
            return False

    normalized_expected = [
        _NUMBER_WORDS.get(word, word)
        for word in expected_words
    ]
    expects_numeric_designation = any(word.isdigit() for word in normalized_expected)
    expected_joined = "".join(normalized_expected)
    expected_skeleton = _skeleton(expected_joined)
    if not expected_skeleton:
        return False
    minimum_window = max(1, len(normalized_expected) - 2)
    maximum_window = len(normalized_expected) + 2
    for start in range(len(observed_words)):
        for width in range(minimum_window, maximum_window + 1):
            window_words = observed_words[start : start + width]
            if not window_words:
                continue
            normalized_window = [
                _NUMBER_WORDS.get(word, word)
                for word in window_words
                if not (expects_numeric_designation and word == "number")
            ]
            observed_joined = "".join(normalized_window)
            observed_skeleton = _skeleton(observed_joined)
            if not observed_skeleton:
                continue
            skeleton_coverage = len(observed_skeleton) / len(expected_skeleton)
            skeleton_similarity = SequenceMatcher(
                None,
                expected_skeleton,
                observed_skeleton,
            ).ratio()
            spelling_similarity = SequenceMatcher(
                None,
                expected_joined,
                observed_joined,
            ).ratio()
            if (
                skeleton_coverage >= 0.80
                and skeleton_similarity >= 0.84
                and spelling_similarity >= 0.62
                and (0.60 * skeleton_similarity + 0.40 * spelling_similarity)
                >= 0.80
            ):
                return True
    return False


def _recognized_text(raw: Mapping[str, Any]) -> str:
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


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:180]


def _code_switch_sample(tts_text: str) -> str:
    # Repetition makes the audit resistant to one STT boundary error while the
    # isiZulu carrier keeps the entity in the same language-switching context
    # used by production dialogue.
    return (
        f"Lokho kubizwa ngokuthi {tts_text}. "
        f"Ngiyaphinda, {tts_text}."
    )


def _timing_adjust(
    source: Path,
    output: Path,
    *,
    ratio: float,
    sample_rate: int,
    runner: Callable[..., Any] = subprocess.run,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    runner(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter:a",
            f"atempo={ratio:.8f}",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
    )


def _selected_tts_text(
    entity: Mapping[str, Any],
    voice: str,
    pronunciation_dictionary: PronunciationDictionary,
) -> str:
    locale = ((entity.get("spoken_forms") or {}).get("zu-ZA") or {})
    voices = locale.get("voices") or {}
    selected = str(
        voices.get(voice)
        or locale.get("default")
        or entity.get("canonical_text")
        or ""
    ).strip()
    return pronunciation_dictionary.apply(selected).tts_text


@dataclass(frozen=True)
class EntityPronunciationAuditOptions:
    voices: tuple[str, ...]
    rate_percent: int = DEFAULT_PRODUCTION_RATE_PERCENT
    atempo_ratio: float = DEFAULT_PRODUCTION_ATEMPO_RATIO
    force: bool = False

    def __post_init__(self) -> None:
        if not self.voices:
            raise ValueError("At least one isiZulu Azure voice is required")
        if not -10 <= self.rate_percent <= 10:
            raise ValueError("Audit Azure rate must be between -10 and +10 percent")
        if not 1.0 <= self.atempo_ratio <= 1.10:
            raise ValueError("Audit atempo ratio must be between 1.0 and 1.10")


def audit_entity_pronunciations(
    *,
    registry_path: Path,
    output_dir: Path,
    tts_backend: AzureTTSBackend,
    stt_backend: Any,
    options: EntityPronunciationAuditOptions,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Audit every active protected registry entity with both production voices."""

    registry = read_json(registry_path)
    entities = [
        item
        for item in registry.get("entities") or ()
        if isinstance(item, Mapping)
        and bool(item.get("active", True))
        and item.get("translation_policy") == "protected_token"
        and str(item.get("canonical_text") or "").strip()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    pronunciation_dictionary = with_default_organisation_initialisms(
        PronunciationDictionary(
            dictionary_version="entity-pronunciation-audit-v1",
            language="zu-ZA",
        )
    )
    observations: list[dict[str, Any]] = []
    total = len(entities) * len(options.voices)
    current = 0
    for entity in entities:
        entity_id = str(entity["entity_id"])
        canonical = str(entity["canonical_text"]).strip()
        for voice in options.voices:
            current += 1
            tts_text = _selected_tts_text(
                entity,
                voice,
                pronunciation_dictionary,
            )
            sample_key = hashlib.sha256(
                json.dumps(
                    {
                        "audit_version": ENTITY_PRONUNCIATION_AUDIT_VERSION,
                        "entity_id": entity_id,
                        "voice": voice,
                        "tts_text": tts_text,
                        "rate_percent": options.rate_percent,
                        "atempo_ratio": options.atempo_ratio,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:12]
            stem = _safe_filename(f"{entity_id}-{voice}-{sample_key}")
            raw_audio = audio_dir / f"{stem}-raw.wav"
            production_audio = audio_dir / f"{stem}-production.wav"
            stt_path = audio_dir / f"{stem}-en-ZA.json"
            request = AzureTTSRequest(
                turn_id=f"entity-pronunciation-{entity_id}-{voice}",
                speaker_id="ENTITY_AUDIT",
                text=_code_switch_sample(tts_text),
                preferred_duration_ms=20_000,
                maximum_duration_ms=60_000,
                voice=voice,
                language="zu-ZA",
                rate_percent=options.rate_percent,
                parts=build_initialism_ssml_parts(
                    _code_switch_sample(tts_text),
                    language="zu-ZA",
                ),
            )
            synthesized = tts_backend.synthesize(
                request,
                raw_audio,
                force=options.force,
            )
            if options.force or not production_audio.is_file():
                _timing_adjust(
                    raw_audio,
                    production_audio,
                    ratio=options.atempo_ratio,
                    sample_rate=tts_backend.sample_rate,
                )
            if options.force or not stt_path.is_file():
                try:
                    raw_stt = stt_backend.transcribe(production_audio, "en-ZA")
                except AzureSpeechError as exc:
                    # No-speech is a valid negative pronunciation observation,
                    # not a reason to discard the rest of an inventory audit.
                    # Transport/authentication/provider failures must still stop
                    # the run so they cannot be misreported as entity failures.
                    if str(exc) != "Azure Speech returned an empty transcription":
                        raise
                    raw_stt = {
                        "phrases": [],
                        "combinedPhrases": [],
                        "audit_error": "empty_transcription",
                    }
                atomic_write_json(stt_path, raw_stt)
            else:
                raw_stt = read_json(stt_path)
            transcription = _recognized_text(raw_stt)
            passed = entity_surface_recognized(canonical, transcription)
            observation = {
                "entity_id": entity_id,
                "entity_type": str(entity.get("entity_type") or ""),
                "canonical_text": canonical,
                "voice": voice,
                "tts_text": tts_text,
                "rate_percent": options.rate_percent,
                "atempo_ratio": options.atempo_ratio,
                "recognized_text": transcription,
                "passed": passed,
                "audio_path": str(production_audio),
                "audio_sha256": hashlib.sha256(
                    production_audio.read_bytes()
                ).hexdigest(),
                "raw_duration_ms": synthesized.duration_ms,
            }
            observations.append(observation)
            if progress is not None:
                progress(
                    {
                        "current": current,
                        "total": total,
                        "entity_id": entity_id,
                        "voice": voice,
                        "passed": passed,
                        "recognized_text": transcription,
                    }
                )

    failed_entities = sorted(
        {
            str(item["entity_id"])
            for item in observations
            if not bool(item["passed"])
        }
    )
    report = {
        "schema_version": ENTITY_PRONUNCIATION_AUDIT_SCHEMA,
        "audit_version": ENTITY_PRONUNCIATION_AUDIT_VERSION,
        "generated_at": _utcnow(),
        "registry_path": str(registry_path),
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "target_locale": "zu-ZA",
        "verification_locale": "en-ZA",
        "voices": list(options.voices),
        "rate_percent": options.rate_percent,
        "atempo_ratio": options.atempo_ratio,
        "entity_count": len(entities),
        "observation_count": len(observations),
        "passed_observation_count": sum(
            1 for item in observations if bool(item["passed"])
        ),
        "failed_observation_count": sum(
            1 for item in observations if not bool(item["passed"])
        ),
        "failed_entity_count": len(failed_entities),
        "failed_entity_ids": failed_entities,
        "observations": observations,
    }
    atomic_write_json(output_dir / "report.json", report)
    return report


__all__ = [
    "DEFAULT_PRODUCTION_ATEMPO_RATIO",
    "DEFAULT_PRODUCTION_RATE_PERCENT",
    "ENTITY_PRONUNCIATION_AUDIT_SCHEMA",
    "ENTITY_PRONUNCIATION_AUDIT_VERSION",
    "EntityPronunciationAuditOptions",
    "audit_entity_pronunciations",
    "entity_surface_recognized",
]
