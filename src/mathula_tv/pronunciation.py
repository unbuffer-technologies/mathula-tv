"""Versioned, application-controlled pronunciation dictionaries.

Entries are literal substitutions.  They are never interpreted as XML and
cannot inject SSML; an SSML builder must escape the resulting text later.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any, Iterable, Mapping


PRONUNCIATION_DICTIONARY_SCHEMA_VERSION = "pronunciation-dictionary-v1"
PRONUNCIATION_ENTRY_SCHEMA_VERSION = "pronunciation-entry-v1"
PRONUNCIATION_KINDS = {
    "personal_name",
    "place_name",
    "organisation_name",
    "initials",
    "acronym",
    "number",
    "date",
    "percentage",
    "currency",
    "english_code_switch",
    "commission_term",
    "politics_term",
    "other",
}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_literal(value: str, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required")
    if "<" in text or ">" in text or _CONTROL.search(text):
        raise ValueError(f"{label} must be plain text, not XML or control data")
    return text


def _validate_reviewed_at(value: str | None) -> None:
    if value is None:
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp") from exc


@dataclass(frozen=True)
class PronunciationEntry:
    display_text: str
    spoken_text: str
    tts_text: str
    language: str = "zu-ZA"
    source: str = "human_review"
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    confidence: float = 1.0
    notes: tuple[str, ...] = ()
    kind: str = "other"
    aliases: tuple[str, ...] = ()
    entry_id: str | None = None
    schema_version: str = PRONUNCIATION_ENTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRONUNCIATION_ENTRY_SCHEMA_VERSION:
            raise ValueError("Unsupported pronunciation entry schema")
        object.__setattr__(self, "display_text", _safe_literal(self.display_text, "display_text"))
        object.__setattr__(self, "spoken_text", _safe_literal(self.spoken_text, "spoken_text"))
        object.__setattr__(self, "tts_text", _safe_literal(self.tts_text, "tts_text"))
        if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?", self.language):
            raise ValueError("language must be a locale such as zu-ZA")
        if not self.source.strip():
            raise ValueError("source is required")
        if self.kind not in PRONUNCIATION_KINDS:
            raise ValueError(f"Unsupported pronunciation kind: {self.kind}")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        _validate_reviewed_at(self.reviewed_at)
        notes = tuple(str(note) for note in self.notes)
        aliases = tuple(_safe_literal(alias, "alias") for alias in self.aliases)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "aliases", aliases)
        if self.entry_id is None:
            identity = {
                "display_text": self.display_text,
                "spoken_text": self.spoken_text,
                "tts_text": self.tts_text,
                "language": self.language,
                "kind": self.kind,
                "source": self.source,
            }
            object.__setattr__(self, "entry_id", "pron_" + hashlib.sha256(_canonical_json(identity).encode()).hexdigest()[:16])
        elif not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.entry_id):
            raise ValueError("entry_id contains unsupported characters")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PronunciationEntry":
        allowed = {item.name for item in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown pronunciation entry fields: {sorted(unknown)}")
        return cls(
            **{
                **dict(value),
                "notes": tuple(value.get("notes") or ()),
                "aliases": tuple(value.get("aliases") or ()),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["notes"] = list(self.notes)
        value["aliases"] = list(self.aliases)
        return value

    def match_texts(self) -> tuple[str, ...]:
        seen: set[str] = set()
        values: list[str] = []
        for value in (self.spoken_text, self.display_text, *self.aliases):
            folded = value.casefold()
            if folded not in seen:
                seen.add(folded)
                values.append(value)
        return tuple(values)


@dataclass(frozen=True)
class PronunciationSubstitution:
    entry_id: str
    kind: str
    display_text: str
    spoken_text: str
    tts_text: str
    before: str
    after: str
    source_start: int
    source_end: int
    source: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PronunciationResult:
    spoken_text: str
    tts_text: str
    substitutions: tuple[PronunciationSubstitution, ...]
    dictionary_version: str
    dictionary_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "spoken_text": self.spoken_text,
            "tts_text": self.tts_text,
            "substitutions": [item.to_dict() for item in self.substitutions],
            "dictionary_version": self.dictionary_version,
            "dictionary_sha256": self.dictionary_sha256,
        }


@dataclass(frozen=True)
class PronunciationDictionary:
    dictionary_version: str
    entries: tuple[PronunciationEntry, ...] = ()
    job_overrides: tuple[PronunciationEntry, ...] = ()
    language: str = "zu-ZA"
    job_id: str | None = None
    schema_version: str = PRONUNCIATION_DICTIONARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRONUNCIATION_DICTIONARY_SCHEMA_VERSION:
            raise ValueError("Unsupported pronunciation dictionary schema")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", self.dictionary_version):
            raise ValueError("dictionary_version must be a safe version label, for example 1.0.0 or v1")
        entries = tuple(
            entry if isinstance(entry, PronunciationEntry) else PronunciationEntry.from_dict(entry)
            for entry in self.entries
        )
        overrides = tuple(
            entry if isinstance(entry, PronunciationEntry) else PronunciationEntry.from_dict(entry)
            for entry in self.job_overrides
        )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "job_overrides", overrides)
        for entry in (*entries, *overrides):
            if entry.language.casefold() != self.language.casefold():
                raise ValueError("Pronunciation entry language does not match dictionary language")
        for label, scoped in (("global", entries), ("job override", overrides)):
            entry_ids = [entry.entry_id for entry in scoped]
            if len(entry_ids) != len(set(entry_ids)):
                raise ValueError(f"Pronunciation entry IDs must be unique within {label} scope")
        if overrides and not self.job_id:
            raise ValueError("job_overrides require a job-local job_id")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PronunciationDictionary":
        if value.get("schema_version") != PRONUNCIATION_DICTIONARY_SCHEMA_VERSION:
            raise ValueError("Unsupported pronunciation dictionary schema")
        allowed = {
            "schema_version",
            "dictionary_version",
            "language",
            "job_id",
            "entries",
            "job_overrides",
            "dictionary_sha256",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown pronunciation dictionary fields: {sorted(unknown)}")
        dictionary = cls(
            dictionary_version=str(value["dictionary_version"]),
            language=str(value.get("language", "zu-ZA")),
            job_id=value.get("job_id"),
            entries=tuple(PronunciationEntry.from_dict(item) for item in value.get("entries") or []),
            job_overrides=tuple(PronunciationEntry.from_dict(item) for item in value.get("job_overrides") or []),
        )
        expected = value.get("dictionary_sha256")
        if expected is not None and expected != dictionary.sha256:
            raise ValueError("Pronunciation dictionary hash mismatch")
        return dictionary

    def _without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dictionary_version": self.dictionary_version,
            "language": self.language,
            "job_id": self.job_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "job_overrides": [entry.to_dict() for entry in self.job_overrides],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._without_hash()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._without_hash(), "dictionary_sha256": self.sha256}

    def with_job_overrides(
        self,
        job_id: str,
        overrides: Iterable[PronunciationEntry | Mapping[str, Any]],
    ) -> "PronunciationDictionary":
        if not str(job_id).strip():
            raise ValueError("job_id is required for job pronunciation overrides")
        return PronunciationDictionary(
            dictionary_version=self.dictionary_version,
            entries=self.entries,
            job_overrides=tuple(
                item if isinstance(item, PronunciationEntry) else PronunciationEntry.from_dict(item)
                for item in overrides
            ),
            language=self.language,
            job_id=str(job_id),
        )

    def apply(
        self,
        spoken_text: str,
        *,
        kinds: set[str] | None = None,
    ) -> PronunciationResult:
        text = _safe_literal(spoken_text, "spoken_text")
        candidates: list[tuple[int, int, int, int, str, PronunciationEntry]] = []
        scoped_entries = [(0, entry) for entry in self.job_overrides] + [(1, entry) for entry in self.entries]
        for scope_priority, entry in scoped_entries:
            if kinds is not None and entry.kind not in kinds:
                continue
            for match_text in entry.match_texts():
                pattern = re.compile(rf"(?<!\w){re.escape(match_text)}(?!\w)", re.IGNORECASE | re.UNICODE)
                for match in pattern.finditer(text):
                    candidates.append(
                        (
                            match.start(),
                            match.end(),
                            scope_priority,
                            -(match.end() - match.start()),
                            str(entry.entry_id),
                            entry,
                        )
                    )
        candidates.sort(key=lambda item: (item[0], item[2], item[3], item[4]))
        selected: list[tuple[int, int, PronunciationEntry]] = []
        occupied_until = -1
        for start, end, _scope, _length, _entry_id, entry in candidates:
            if start < occupied_until:
                continue
            selected.append((start, end, entry))
            occupied_until = end

        output: list[str] = []
        substitutions: list[PronunciationSubstitution] = []
        cursor = 0
        for start, end, entry in selected:
            output.append(text[cursor:start])
            output.append(entry.tts_text)
            before = text[start:end]
            substitutions.append(
                PronunciationSubstitution(
                    entry_id=str(entry.entry_id),
                    kind=entry.kind,
                    display_text=entry.display_text,
                    spoken_text=entry.spoken_text,
                    tts_text=entry.tts_text,
                    before=before,
                    after=entry.tts_text,
                    source_start=start,
                    source_end=end,
                    source=entry.source,
                    confidence=float(entry.confidence),
                )
            )
            cursor = end
        output.append(text[cursor:])
        return PronunciationResult(
            spoken_text=text,
            tts_text="".join(output),
            substitutions=tuple(substitutions),
            dictionary_version=self.dictionary_version,
            dictionary_sha256=self.sha256,
        )


def apply_pronunciation_dictionary(
    spoken_text: str,
    dictionary: PronunciationDictionary,
) -> PronunciationResult:
    return dictionary.apply(spoken_text)


def normalise_numbers_for_tts(spoken_text: str, dictionary: PronunciationDictionary) -> PronunciationResult:
    return dictionary.apply(spoken_text, kinds={"number", "percentage", "currency"})


def normalize_numbers_for_tts(spoken_text: str, dictionary: PronunciationDictionary) -> PronunciationResult:
    return normalise_numbers_for_tts(spoken_text, dictionary)


def normalise_dates_for_tts(spoken_text: str, dictionary: PronunciationDictionary) -> PronunciationResult:
    return dictionary.apply(spoken_text, kinds={"date"})


def normalize_dates_for_tts(spoken_text: str, dictionary: PronunciationDictionary) -> PronunciationResult:
    return normalise_dates_for_tts(spoken_text, dictionary)


def expand_initials_for_tts(spoken_text: str, dictionary: PronunciationDictionary) -> PronunciationResult:
    return dictionary.apply(spoken_text, kinds={"initials", "acronym"})


__all__ = [
    "PRONUNCIATION_DICTIONARY_SCHEMA_VERSION",
    "PRONUNCIATION_ENTRY_SCHEMA_VERSION",
    "PRONUNCIATION_KINDS",
    "PronunciationDictionary",
    "PronunciationEntry",
    "PronunciationResult",
    "PronunciationSubstitution",
    "apply_pronunciation_dictionary",
    "expand_initials_for_tts",
    "normalise_dates_for_tts",
    "normalise_numbers_for_tts",
    "normalize_dates_for_tts",
    "normalize_numbers_for_tts",
]
