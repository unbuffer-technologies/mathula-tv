"""Deterministic construction of semantic, provenance-preserving dubbing units."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


DUBBING_UNITS_SCHEMA_VERSION = "dubbing-units-v1"
DUBBING_UNIT_SCHEMA_VERSION = "1.0"
_SPLIT_PUNCTUATION = re.compile(r"[.!?;,:][\"'”’)]*$")
_SEMANTIC_END = re.compile(r"[.!?][\"'”’)]*$")
_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|[^\w\s]", re.UNICODE)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _milliseconds(item: Mapping[str, Any], stem: str) -> int:
    milliseconds_key = f"{stem}_ms"
    if milliseconds_key in item and item[milliseconds_key] is not None:
        return int(round(_finite_number(item[milliseconds_key], milliseconds_key)))
    if stem not in item or item[stem] is None:
        raise ValueError(f"{stem} or {milliseconds_key} is required")
    return int(round(_finite_number(item[stem], stem) * 1000.0))


def _optional_milliseconds(item: Mapping[str, Any], stem: str) -> int | None:
    if f"{stem}_ms" not in item and stem not in item:
        return None
    return _milliseconds(item, stem)


def _dedupe(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    fingerprints: set[str] = set()
    for value in values:
        fingerprint = _canonical_json(value)
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            result.append(value)
    return result


def _join_tokens(tokens: Sequence[str]) -> str:
    text = ""
    closing = set(".,!?;:%)]}”’")
    opening = set("([{“‘")
    for token in (str(value).strip() for value in tokens):
        if not token:
            continue
        if not text or token[0] in closing or text[-1] in opening:
            text += token
        else:
            text += " " + token
    return text.strip()


@dataclass(frozen=True)
class DubbingUnitConfig:
    minimum_duration_ms: int = 1_200
    maximum_duration_ms: int = 8_000
    long_pause_ms: int = 1_200
    hard_end_extension_ms: int = 500
    final_tail_ms: int = 250

    def __post_init__(self) -> None:
        if self.minimum_duration_ms <= 0:
            raise ValueError("minimum_duration_ms must be positive")
        if self.maximum_duration_ms < self.minimum_duration_ms:
            raise ValueError("maximum_duration_ms must be at least minimum_duration_ms")
        if self.long_pause_ms < 0 or self.hard_end_extension_ms < 0 or self.final_tail_ms < 0:
            raise ValueError("pause and timing extensions must not be negative")


@dataclass(frozen=True)
class _Word:
    word_id: Any | None
    text: str
    start_ms: int
    end_ms: int
    source_index: int


@dataclass
class _Piece:
    speaker_id: str
    phrase_ids: list[Any]
    word_ids: list[Any]
    source_text: str
    start_ms: int
    end_ms: int
    has_overlap: bool
    overlap_group_id: Any | None
    overlap_metadata: list[Any] = field(default_factory=list)
    manual_overrides: list[dict[str, Any]] = field(default_factory=list)
    hard_boundary_before: bool = False
    hard_boundary_after: bool = False

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def merge(self, other: "_Piece") -> None:
        self.phrase_ids = _dedupe([*self.phrase_ids, *other.phrase_ids])
        self.word_ids = _dedupe([*self.word_ids, *other.word_ids])
        self.source_text = _join_tokens([self.source_text, other.source_text])
        self.end_ms = max(self.end_ms, other.end_ms)
        self.has_overlap = self.has_overlap or other.has_overlap
        self.overlap_metadata.extend(other.overlap_metadata)
        self.manual_overrides.extend(other.manual_overrides)
        self.hard_boundary_after = other.hard_boundary_after


def _phrase_id(phrase: Mapping[str, Any], index: int) -> Any:
    for key in ("phrase_id", "segment_id", "id"):
        if key in phrase and phrase[key] is not None:
            return phrase[key]
    return f"phrase_{index + 1:05d}"


def _word_id(word: Mapping[str, Any], index: int) -> Any:
    for key in ("word_id", "id"):
        if key in word and word[key] is not None:
            return word[key]
    return f"word_{index + 1:06d}"


def _speaker(phrase: Mapping[str, Any]) -> str:
    value = phrase.get("speaker_id", phrase.get("speaker"))
    if value is None or not str(value).strip():
        raise ValueError("Every source phrase must have a speaker ID")
    return str(value)


def _phrase_text(phrase: Mapping[str, Any]) -> str:
    value = phrase.get("source_text", phrase.get("text", ""))
    if not str(value).strip():
        raise ValueError("Every source phrase must contain text")
    return str(value).strip()


def _overlap_amount(start_ms: int, end_ms: int, phrase: Mapping[str, Any]) -> int:
    return max(0, min(end_ms, _milliseconds(phrase, "end")) - max(start_ms, _milliseconds(phrase, "start")))


def _normalise_words(
    transcript: Mapping[str, Any],
    phrases: Sequence[Mapping[str, Any]],
) -> dict[int, list[_Word]]:
    """Assign each source word to exactly one source phrase."""

    assigned: dict[int, list[_Word]] = {index: [] for index in range(len(phrases))}
    explicit_phrase_indexes: set[int] = set()
    source_index = 0
    for phrase_index, phrase in enumerate(phrases):
        if "words" not in phrase:
            continue
        if not isinstance(phrase.get("words"), list):
            raise ValueError("Phrase words must be a list")
        explicit_phrase_indexes.add(phrase_index)
        for word in phrase.get("words") or []:
            if not isinstance(word, Mapping):
                raise ValueError("Phrase words must be objects")
            start_ms = _milliseconds(word, "start")
            end_ms = _milliseconds(word, "end")
            if end_ms <= start_ms:
                raise ValueError("Source word end must be after start")
            assigned[phrase_index].append(
                _Word(_word_id(word, source_index), str(word.get("text", "")).strip(), start_ms, end_ms, source_index)
            )
            source_index += 1

    words = transcript.get("words") or []
    if not isinstance(words, list):
        raise ValueError("Transcript words must be a list")
    phrase_ids = [_phrase_id(phrase, index) for index, phrase in enumerate(phrases)]
    for word_index, word in enumerate(words):
        if not isinstance(word, Mapping):
            raise ValueError("Transcript words must be objects")
        start_ms = _milliseconds(word, "start")
        end_ms = _milliseconds(word, "end")
        if end_ms <= start_ms:
            raise ValueError("Source word end must be after start")
        explicit_phrase = word.get("phrase_id", word.get("segment_id"))
        candidates: list[int] = []
        if explicit_phrase is not None:
            candidates = [index for index, phrase_id in enumerate(phrase_ids) if phrase_id == explicit_phrase]
        if not candidates:
            word_speaker = word.get("speaker_id", word.get("speaker"))
            candidates = [
                index
                for index, phrase in enumerate(phrases)
                if index not in explicit_phrase_indexes
                if (word_speaker is None or str(word_speaker) == _speaker(phrase))
                and _overlap_amount(start_ms, end_ms, phrase) > 0
            ]
        if not candidates:
            continue
        phrase_index = max(candidates, key=lambda index: (_overlap_amount(start_ms, end_ms, phrases[index]), -index))
        assigned[phrase_index].append(
            _Word(_word_id(word, word_index), str(word.get("text", "")).strip(), start_ms, end_ms, word_index)
        )
    for phrase_words in assigned.values():
        phrase_words.sort(key=lambda item: (item.start_ms, item.end_ms, item.source_index))
    return assigned


def _derived_words(phrase: Mapping[str, Any]) -> list[_Word]:
    """Tokenize for deterministic splitting when Azure word timing is absent.

    Derived tokens intentionally have no source word ID; the builder never
    presents generated IDs as Azure provenance.
    """

    tokens = _TOKEN.findall(_phrase_text(phrase))
    if not tokens:
        return []
    start_ms, end_ms = _milliseconds(phrase, "start"), _milliseconds(phrase, "end")
    span = end_ms - start_ms
    return [
        _Word(
            None,
            token,
            start_ms + round(span * index / len(tokens)),
            start_ms + round(span * (index + 1) / len(tokens)),
            index,
        )
        for index, token in enumerate(tokens)
    ]


def _boundary_index(words: Sequence[_Word], start_index: int, limit_ms: int, minimum_ms: int) -> int:
    viable = [index for index in range(start_index, len(words)) if words[index].end_ms <= limit_ms]
    if not viable:
        return start_index
    preferred = [
        index
        for index in viable
        if words[index].end_ms - words[start_index].start_ms >= minimum_ms
        and _SPLIT_PUNCTUATION.search(words[index].text)
    ]
    if preferred:
        return preferred[-1]
    sufficiently_long = [
        index for index in viable if words[index].end_ms - words[start_index].start_ms >= minimum_ms
    ]
    return (sufficiently_long or viable)[-1]


def _piece_metadata(
    phrase: Mapping[str, Any],
    phrase_id: Any,
    external_override: Mapping[str, Any] | None,
) -> tuple[bool, Any | None, list[Any], list[dict[str, Any]], bool, bool]:
    group = phrase.get("overlap_group_id")
    has_overlap = bool(phrase.get("has_overlap", phrase.get("overlap", False)) or group is not None)
    overlap_metadata: list[Any] = []
    if "overlap_metadata" in phrase:
        overlap_metadata.append(phrase["overlap_metadata"])
    manual: list[dict[str, Any]] = []
    embedded = phrase.get("manual_override", phrase.get("manual_override_metadata"))
    if isinstance(embedded, Mapping):
        manual.append({"source_phrase_id": phrase_id, **dict(embedded)})
    if external_override:
        manual.append({"source_phrase_id": phrase_id, **dict(external_override)})
    hard_before = bool(phrase.get("hard_overlap_boundary_before") or phrase.get("hard_boundary_before"))
    hard_after = bool(phrase.get("hard_overlap_boundary_after") or phrase.get("hard_boundary_after"))
    for override in manual:
        hard_before = hard_before or bool(override.get("force_boundary_before"))
        hard_after = hard_after or bool(override.get("force_boundary_after"))
    return has_overlap, group, overlap_metadata, manual, hard_before, hard_after


def _split_phrase(
    phrase: Mapping[str, Any],
    phrase_id: Any,
    words: list[_Word],
    config: DubbingUnitConfig,
    external_override: Mapping[str, Any] | None,
) -> list[_Piece]:
    phrase_start, phrase_end = _milliseconds(phrase, "start"), _milliseconds(phrase, "end")
    if phrase_end <= phrase_start:
        raise ValueError(f"Source phrase {phrase_id!r} ends before it starts")
    speaker = _speaker(phrase)
    has_overlap, group, overlap_metadata, manual, hard_before, hard_after = _piece_metadata(
        phrase, phrase_id, external_override
    )
    real_words = words
    algorithm_words = real_words or _derived_words(phrase)
    if phrase_end - phrase_start <= config.maximum_duration_ms or not algorithm_words:
        return [
            _Piece(
                speaker,
                [phrase_id],
                [word.word_id for word in real_words if word.word_id is not None],
                _phrase_text(phrase),
                phrase_start,
                phrase_end,
                has_overlap,
                group,
                list(overlap_metadata),
                list(manual),
                hard_before,
                hard_after,
            )
        ]

    chunks: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(algorithm_words):
        limit_ms = algorithm_words[cursor].start_ms + config.maximum_duration_ms
        boundary = _boundary_index(algorithm_words, cursor, limit_ms, config.minimum_duration_ms)
        chunks.append((cursor, boundary + 1))
        cursor = boundary + 1

    pieces: list[_Piece] = []
    for chunk_index, (begin, finish) in enumerate(chunks):
        chunk_words = algorithm_words[begin:finish]
        start_ms = phrase_start if chunk_index == 0 else (algorithm_words[begin - 1].end_ms + chunk_words[0].start_ms) // 2
        end_ms = phrase_end if chunk_index == len(chunks) - 1 else (
            chunk_words[-1].end_ms + algorithm_words[finish].start_ms
        ) // 2
        real_chunk_ids = [word.word_id for word in chunk_words if word.word_id is not None]
        pieces.append(
            _Piece(
                speaker,
                [phrase_id],
                real_chunk_ids,
                _join_tokens([word.text for word in chunk_words]),
                start_ms,
                end_ms,
                has_overlap,
                group,
                list(overlap_metadata),
                list(manual),
                hard_before if chunk_index == 0 else False,
                hard_after if chunk_index == len(chunks) - 1 else False,
            )
        )
    return pieces


def _hard_boundary(current: _Piece, following: _Piece, config: DubbingUnitConfig) -> bool:
    if current.speaker_id != following.speaker_id:
        return True
    if current.hard_boundary_after or following.hard_boundary_before:
        return True
    if current.has_overlap != following.has_overlap:
        return True
    if current.overlap_group_id != following.overlap_group_id and (
        current.overlap_group_id is not None or following.overlap_group_id is not None
    ):
        return True
    return following.start_ms - current.end_ms >= config.long_pause_ms


def _should_merge(current: _Piece, following: _Piece, config: DubbingUnitConfig) -> bool:
    if _hard_boundary(current, following, config):
        return False
    if max(current.end_ms, following.end_ms) - current.start_ms > config.maximum_duration_ms:
        return False
    force_merge = any(bool(value.get("merge_with_next")) for value in current.manual_overrides)
    if force_merge:
        return True
    short = current.duration_ms < config.minimum_duration_ms or following.duration_ms < config.minimum_duration_ms
    semantically_incomplete = _SEMANTIC_END.search(current.source_text.strip()) is None
    return short or semantically_incomplete


def _merge_pieces(pieces: Sequence[_Piece], config: DubbingUnitConfig) -> list[_Piece]:
    merged: list[_Piece] = []
    for source in pieces:
        piece = _Piece(**asdict(source))
        if merged and _should_merge(merged[-1], piece, config):
            merged[-1].merge(piece)
        else:
            merged.append(piece)
    return merged


def _unit_override(
    unit_id: str,
    piece: _Piece,
    unit_overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for item in piece.manual_overrides:
        for key in ("preferred_end_ms", "hard_end_ms"):
            if key in item:
                combined[key] = item[key]
    if unit_id in unit_overrides:
        combined.update(unit_overrides[unit_id])
    return combined


def _timed_units(
    pieces: Sequence[_Piece],
    config: DubbingUnitConfig,
    transcript_end_ms: int | None,
    unit_overrides: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces):
        unit_id = f"unit_{index + 1:04d}"
        previous_end = max((other.end_ms for other in pieces[:index]), default=0)
        next_start = min((other.start_ms for other in pieces[index + 1 :]), default=None)
        preceding_silence = max(0, piece.start_ms - previous_end)
        if next_start is None:
            available_tail = (
                max(0, transcript_end_ms - piece.end_ms)
                if transcript_end_ms is not None
                else config.final_tail_ms
            )
        else:
            available_tail = max(0, next_start - piece.end_ms)
        following_silence = available_tail
        preferred_end = piece.end_ms
        hard_end = preferred_end + min(available_tail, config.hard_end_extension_ms)
        override = _unit_override(unit_id, piece, unit_overrides)
        if "preferred_end_ms" in override:
            preferred_end = int(round(_finite_number(override["preferred_end_ms"], "preferred_end_ms")))
        if "hard_end_ms" in override:
            hard_end = int(round(_finite_number(override["hard_end_ms"], "hard_end_ms")))
        else:
            hard_end = max(hard_end, preferred_end)
        if preferred_end < piece.start_ms or hard_end < preferred_end:
            raise ValueError(f"Invalid manual timing override for {unit_id}")

        overlap_metadata = _dedupe(piece.overlap_metadata)
        manual_metadata = _dedupe([*piece.manual_overrides, *([dict(override)] if override else [])])
        units.append(
            {
                "schema_version": DUBBING_UNIT_SCHEMA_VERSION,
                "unit_id": unit_id,
                "speaker_id": piece.speaker_id,
                "source_phrase_ids": piece.phrase_ids,
                "source_word_ids": piece.word_ids,
                "source_text": piece.source_text,
                "start_ms": piece.start_ms,
                "preferred_end_ms": preferred_end,
                "hard_end_ms": hard_end,
                "preferred_duration_ms": preferred_end - piece.start_ms,
                "maximum_duration_ms": hard_end - piece.start_ms,
                "preceding_silence_ms": preceding_silence,
                "following_silence_ms": following_silence,
                "has_overlap": piece.has_overlap,
                "overlap_group_id": piece.overlap_group_id,
                "overlap_metadata": overlap_metadata,
                "manual_override_metadata": manual_metadata,
            }
        )
    return units


def _transcript_end_ms(transcript: Mapping[str, Any], phrases: Sequence[Mapping[str, Any]]) -> int | None:
    for key in ("duration_ms", "source_duration_ms"):
        if key in transcript and transcript[key] is not None:
            return int(round(_finite_number(transcript[key], key)))
    for key in ("duration", "source_duration"):
        if key in transcript and transcript[key] is not None:
            return int(round(_finite_number(transcript[key], key) * 1000))
    return None


@dataclass(frozen=True)
class DubbingUnitBuilder:
    config: DubbingUnitConfig = field(default_factory=DubbingUnitConfig)

    def build(
        self,
        transcript: Mapping[str, Any],
        *,
        phrase_overrides: Mapping[Any, Mapping[str, Any]] | None = None,
        unit_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        phrases_value = transcript.get("phrases")
        if phrases_value is None:
            phrases_value = transcript.get("segments")
        if not isinstance(phrases_value, list) or not phrases_value:
            raise ValueError("Transcript must contain at least one phrase or segment")
        if any(not isinstance(phrase, Mapping) for phrase in phrases_value):
            raise ValueError("Transcript phrases must be objects")
        indexed_phrases = list(enumerate(phrases_value))
        indexed_phrases.sort(
            key=lambda item: (
                _milliseconds(item[1], "start"),
                _milliseconds(item[1], "end"),
                _canonical_json(_phrase_id(item[1], item[0])),
                item[0],
            )
        )
        phrases: list[Mapping[str, Any]] = [phrase for _index, phrase in indexed_phrases]
        words = _normalise_words(transcript, phrases)
        phrase_overrides = phrase_overrides or {}
        pieces: list[_Piece] = []
        for index, phrase in enumerate(phrases):
            phrase_id = _phrase_id(phrase, index)
            pieces.extend(
                _split_phrase(
                    phrase,
                    phrase_id,
                    words[index],
                    self.config,
                    phrase_overrides.get(phrase_id),
                )
            )
        merged = _merge_pieces(pieces, self.config)
        units = _timed_units(
            merged,
            self.config,
            _transcript_end_ms(transcript, phrases),
            unit_overrides or {},
        )
        source_identity = {
            "schema_version": transcript.get("schema_version"),
            "phrases": phrases_value,
            "words": transcript.get("words", []),
        }
        artifact = {
            "schema_version": DUBBING_UNITS_SCHEMA_VERSION,
            "source_schema_version": transcript.get("schema_version"),
            "source_sha256": _sha256(source_identity),
            "config": asdict(self.config),
            "units": units,
        }
        artifact["artifact_sha256"] = _sha256(artifact)
        validate_dubbing_unit_artifact(artifact)
        return artifact

    def build_units(self, transcript: Mapping[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        return self.build(transcript, **kwargs)["units"]


def validate_dubbing_unit_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != DUBBING_UNITS_SCHEMA_VERSION:
        raise ValueError("Unsupported dubbing-unit artifact schema")
    units = artifact.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("Dubbing-unit artifact must contain units")
    expected_ids = [f"unit_{index + 1:04d}" for index in range(len(units))]
    if [unit.get("unit_id") for unit in units] != expected_ids:
        raise ValueError("Dubbing unit IDs must be deterministic and sequential")
    for unit in units:
        required = {
            "schema_version",
            "speaker_id",
            "source_phrase_ids",
            "source_word_ids",
            "source_text",
            "start_ms",
            "preferred_end_ms",
            "hard_end_ms",
            "preferred_duration_ms",
            "maximum_duration_ms",
            "has_overlap",
            "overlap_group_id",
        }
        missing = required - unit.keys()
        if missing:
            raise ValueError(f"Dubbing unit is missing {sorted(missing)}")
        if not unit["source_phrase_ids"] or not str(unit["source_text"]).strip():
            raise ValueError("Dubbing unit provenance and text are required")
        if unit["start_ms"] > unit["preferred_end_ms"] or unit["preferred_end_ms"] > unit["hard_end_ms"]:
            raise ValueError("Dubbing unit timing windows are invalid")
        if unit["preferred_duration_ms"] != unit["preferred_end_ms"] - unit["start_ms"]:
            raise ValueError("Preferred duration does not match preferred timing")
        if unit["maximum_duration_ms"] != unit["hard_end_ms"] - unit["start_ms"]:
            raise ValueError("Maximum duration does not match hard timing")


def build_dubbing_units(
    transcript: Mapping[str, Any],
    config: DubbingUnitConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return DubbingUnitBuilder(config or DubbingUnitConfig()).build(transcript, **kwargs)


def build_units(
    transcript: Mapping[str, Any],
    config: DubbingUnitConfig | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return build_dubbing_units(transcript, config, **kwargs)["units"]


__all__ = [
    "DUBBING_UNIT_SCHEMA_VERSION",
    "DUBBING_UNITS_SCHEMA_VERSION",
    "DubbingUnitBuilder",
    "DubbingUnitConfig",
    "build_dubbing_units",
    "build_units",
    "validate_dubbing_unit_artifact",
]
