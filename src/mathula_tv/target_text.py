"""Auditable three-text representation for every translated dubbing unit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .pronunciation import (
    PronunciationDictionary,
    PronunciationResult,
    PronunciationSubstitution,
)


TARGET_TEXT_SCHEMA_VERSION = "target-text-v1"
TEXT_CHANGE_SCHEMA_VERSION = "text-change-v1"
TEXT_CHANGE_TYPES = {
    "omission",
    "contraction",
    "paraphrase",
    "expansion",
    "pronunciation_substitution",
    "number_normalisation",
    "date_normalisation",
    "initial_expansion",
    "code_switch",
    "other",
}
TEXT_CHANGE_LAYERS = {"faithful_to_spoken", "spoken_to_tts"}


@dataclass(frozen=True)
class TextChange:
    layer: str
    change_type: str
    source_start: int
    source_end: int
    before: str
    after: str
    reason: str
    protected_terms: tuple[str, ...] = ()
    human_review_required: bool = False
    schema_version: str = TEXT_CHANGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TEXT_CHANGE_SCHEMA_VERSION:
            raise ValueError("Unsupported text-change schema")
        if self.layer not in TEXT_CHANGE_LAYERS:
            raise ValueError(f"Unsupported text-change layer: {self.layer}")
        if self.change_type not in TEXT_CHANGE_TYPES:
            raise ValueError(f"Unsupported text-change type: {self.change_type}")
        if self.source_start < 0 or self.source_end < self.source_start:
            raise ValueError("Text-change source span is invalid")
        if not self.reason.strip():
            raise ValueError("Every text change must have a reason")
        object.__setattr__(self, "protected_terms", tuple(str(term) for term in self.protected_terms))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TextChange":
        return cls(
            layer=str(value["layer"]),
            change_type=str(value["change_type"]),
            source_start=int(value["source_start"]),
            source_end=int(value["source_end"]),
            before=str(value["before"]),
            after=str(value["after"]),
            reason=str(value["reason"]),
            protected_terms=tuple(value.get("protected_terms") or ()),
            human_review_required=bool(value.get("human_review_required", False)),
            schema_version=str(value.get("schema_version", TEXT_CHANGE_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["protected_terms"] = list(self.protected_terms)
        return value


def _apply_recorded_changes(source: str, changes: Sequence[TextChange], layer: str) -> str:
    selected = sorted(
        (change for change in changes if change.layer == layer),
        key=lambda change: (change.source_start, change.source_end),
    )
    if not selected:
        return source
    output: list[str] = []
    cursor = 0
    for change in selected:
        if change.source_start < cursor:
            raise ValueError(f"Overlapping {layer} text-change spans are not allowed")
        if change.source_end > len(source):
            raise ValueError(f"{layer} text-change span exceeds source text")
        actual = source[change.source_start : change.source_end]
        if actual != change.before:
            raise ValueError(f"{layer} text-change 'before' value does not match its source span")
        output.append(source[cursor : change.source_start])
        output.append(change.after)
        cursor = change.source_end
    output.append(source[cursor:])
    return "".join(output)


def _pronunciation_change(substitution: PronunciationSubstitution) -> TextChange:
    kinds = {
        "number": "number_normalisation",
        "percentage": "number_normalisation",
        "currency": "number_normalisation",
        "date": "date_normalisation",
        "initials": "initial_expansion",
        "acronym": "initial_expansion",
        "english_code_switch": "code_switch",
    }
    return TextChange(
        layer="spoken_to_tts",
        change_type=kinds.get(substitution.kind, "pronunciation_substitution"),
        source_start=substitution.source_start,
        source_end=substitution.source_end,
        before=substitution.before,
        after=substitution.after,
        reason=f"Application-controlled pronunciation dictionary entry {substitution.entry_id}",
    )


@dataclass(frozen=True)
class TargetText:
    faithful_translation: str
    spoken_text: str
    tts_text: str
    changes: tuple[TextChange, ...] = ()
    pronunciation_substitutions: tuple[PronunciationSubstitution, ...] = ()
    protected_terms: tuple[str, ...] = ()
    pronunciation_dictionary_version: str | None = None
    pronunciation_dictionary_sha256: str | None = None
    schema_version: str = TARGET_TEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_TEXT_SCHEMA_VERSION:
            raise ValueError("Unsupported target-text schema")
        for name in ("faithful_translation", "spoken_text", "tts_text"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        changes = tuple(
            change if isinstance(change, TextChange) else TextChange.from_dict(change)
            for change in self.changes
        )
        substitutions = tuple(
            item
            if isinstance(item, PronunciationSubstitution)
            else PronunciationSubstitution(**dict(item))
            for item in self.pronunciation_substitutions
        )
        object.__setattr__(self, "changes", changes)
        object.__setattr__(self, "pronunciation_substitutions", substitutions)
        object.__setattr__(self, "protected_terms", tuple(str(term) for term in self.protected_terms))
        self.validate()

    def validate(self) -> None:
        reconstructed_spoken = _apply_recorded_changes(
            self.faithful_translation,
            self.changes,
            "faithful_to_spoken",
        )
        if reconstructed_spoken != self.spoken_text:
            if self.faithful_translation == self.spoken_text:
                raise ValueError("Recorded faithful-to-spoken changes alter otherwise identical text")
            raise ValueError(
                "Every omission, contraction, paraphrase, or expansion from faithful_translation to spoken_text must be recorded"
            )
        reconstructed_tts = _apply_recorded_changes(self.spoken_text, self.changes, "spoken_to_tts")
        if reconstructed_tts != self.tts_text:
            raise ValueError("Every difference between spoken_text and tts_text must be recorded")

        recorded_substitutions = {
            (
                substitution.source_start,
                substitution.source_end,
                substitution.before,
                substitution.after,
            )
            for substitution in self.pronunciation_substitutions
        }
        recorded_changes = {
            (change.source_start, change.source_end, change.before, change.after)
            for change in self.changes
            if change.layer == "spoken_to_tts"
        }
        if not recorded_substitutions.issubset(recorded_changes):
            raise ValueError("Pronunciation substitutions must also be recorded as spoken-to-TTS changes")

        faithful_folded = self.faithful_translation.casefold()
        spoken_folded = self.spoken_text.casefold()
        for term in self.protected_terms:
            folded = term.casefold()
            if folded not in faithful_folded or folded in spoken_folded:
                continue
            reviewed = any(
                change.layer == "faithful_to_spoken"
                and change.human_review_required
                and folded in {item.casefold() for item in change.protected_terms}
                for change in self.changes
            )
            if not reviewed:
                raise ValueError(f"Protected term removed from spoken_text without explicit human-review flag: {term}")

    @property
    def subtitle_text(self) -> str:
        return self.spoken_text

    @property
    def qc_reference_text(self) -> str:
        return self.spoken_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "faithful_translation": self.faithful_translation,
            "spoken_text": self.spoken_text,
            "tts_text": self.tts_text,
            "text_changes": [change.to_dict() for change in self.changes],
            "pronunciation_substitutions": [item.to_dict() for item in self.pronunciation_substitutions],
            "protected_terms": list(self.protected_terms),
            "pronunciation_dictionary_version": self.pronunciation_dictionary_version,
            "pronunciation_dictionary_sha256": self.pronunciation_dictionary_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetText":
        if value.get("schema_version") != TARGET_TEXT_SCHEMA_VERSION:
            raise ValueError("Unsupported target-text schema")
        return cls(
            faithful_translation=str(value["faithful_translation"]),
            spoken_text=str(value["spoken_text"]),
            tts_text=str(value["tts_text"]),
            changes=tuple(TextChange.from_dict(item) for item in value.get("text_changes") or []),
            pronunciation_substitutions=tuple(
                PronunciationSubstitution(**dict(item))
                for item in value.get("pronunciation_substitutions") or []
            ),
            protected_terms=tuple(value.get("protected_terms") or ()),
            pronunciation_dictionary_version=value.get("pronunciation_dictionary_version"),
            pronunciation_dictionary_sha256=value.get("pronunciation_dictionary_sha256"),
        )

    @classmethod
    def from_legacy_approved_text(
        cls,
        approved_text: str,
        *,
        protected_terms: Iterable[str] = (),
    ) -> "TargetText":
        """Upgrade an approved legacy turn without changing its wording."""

        return cls(
            faithful_translation=approved_text,
            spoken_text=approved_text,
            tts_text=approved_text,
            protected_terms=tuple(protected_terms),
        )


def build_target_text(
    faithful_translation: str,
    *,
    spoken_text: str | None = None,
    dictionary: PronunciationDictionary | None = None,
    changes: Iterable[TextChange | Mapping[str, Any]] = (),
    protected_terms: Iterable[str] = (),
) -> TargetText:
    spoken = faithful_translation if spoken_text is None else spoken_text
    normalised_changes = [
        change if isinstance(change, TextChange) else TextChange.from_dict(change)
        for change in changes
    ]
    pronunciation: PronunciationResult | None = dictionary.apply(spoken) if dictionary is not None else None
    substitutions = pronunciation.substitutions if pronunciation is not None else ()
    normalised_changes.extend(_pronunciation_change(item) for item in substitutions)
    return TargetText(
        faithful_translation=faithful_translation,
        spoken_text=spoken,
        tts_text=pronunciation.tts_text if pronunciation is not None else spoken,
        changes=tuple(normalised_changes),
        pronunciation_substitutions=tuple(substitutions),
        protected_terms=tuple(protected_terms),
        pronunciation_dictionary_version=pronunciation.dictionary_version if pronunciation else None,
        pronunciation_dictionary_sha256=pronunciation.dictionary_sha256 if pronunciation else None,
    )


def record_full_text_change(
    source: str,
    target: str,
    *,
    layer: str,
    change_type: str,
    reason: str,
    protected_terms: Iterable[str] = (),
    human_review_required: bool = False,
) -> TextChange:
    """Record a reviewed whole-text rewrite when token spans are not useful."""

    return TextChange(
        layer=layer,
        change_type=change_type,
        source_start=0,
        source_end=len(source),
        before=source,
        after=target,
        reason=reason,
        protected_terms=tuple(protected_terms),
        human_review_required=human_review_required,
    )


def subtitle_text(value: TargetText | Mapping[str, Any]) -> str:
    target = value if isinstance(value, TargetText) else TargetText.from_dict(value)
    return target.spoken_text


def qc_reference_text(value: TargetText | Mapping[str, Any]) -> str:
    target = value if isinstance(value, TargetText) else TargetText.from_dict(value)
    return target.spoken_text


def upgrade_legacy_translation(approved_text: str, *, protected_terms: Iterable[str] = ()) -> dict[str, Any]:
    return TargetText.from_legacy_approved_text(approved_text, protected_terms=protected_terms).to_dict()


ThreeTextRepresentation = TargetText


__all__ = [
    "TARGET_TEXT_SCHEMA_VERSION",
    "TEXT_CHANGE_SCHEMA_VERSION",
    "TEXT_CHANGE_TYPES",
    "TargetText",
    "TextChange",
    "ThreeTextRepresentation",
    "build_target_text",
    "qc_reference_text",
    "record_full_text_change",
    "subtitle_text",
    "upgrade_legacy_translation",
]
