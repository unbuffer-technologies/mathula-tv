"""Entity registry library for protected entity matching and placeholder protection."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .atomic_io import read_json


ENTITY_REGISTRY_SCHEMA = "mathula-entity-registry-v1"
ENTITY_BINDINGS_SCHEMA = "entity-bindings-v1"
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"[0-9a-f]{64}")

# Generic one-word aliases that are too short/ambiguous to match safely
_AMBIGUOUS_SHORT_ALIASES = {
    "johnson", "khan", "jacob", "mk", "ci", "cat", "jamie", "best", "thabo",
    "madlanga", "bester", "woolworths", "hoshkhari", "mighti",
}

# Unicode normalization patterns
_CURLY_QUOTES = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u201a": "'", "\u201b": "'", "\u201e": '"', "\u201f": '"',
}
_TITLE_VARIANTS = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Hon|Adv|Chief|Justice|Judge|President|Minister|Premier|Mayor|Cllr|Sen|Rep|Gov|Gen|Col|Capt|Lt|Sgt|Rev|Fr|Sr)\.?\s*", re.IGNORECASE)


@dataclass(frozen=True)
class SpokenForm:
    """Voice-specific spoken form configuration."""
    default: str | None
    voices: dict[str, str | None]
    status: str
    last_tested_at: str | None
    approved_by: str | None

    def __post_init__(self) -> None:
        # Allow any status for extensibility - registry is append-friendly
        pass


@dataclass(frozen=True)
class Entity:
    """Protected entity from the registry."""
    entity_id: str
    entity_type: str
    canonical_text: str
    display_text: str
    aliases: tuple[str, ...]
    domains: tuple[str, ...]
    roles: tuple[Mapping[str, Any], ...]
    source_status: str
    active: bool
    must_preserve: bool
    translation_policy: str
    stt_phrases: tuple[str, ...]
    spoken_forms: dict[str, SpokenForm]
    do_not_confuse_with: tuple[str, ...]
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _SAFE_IDENTIFIER.fullmatch(self.entity_id):
            raise ValueError(f"Invalid entity_id: {self.entity_id}")
        if self.entity_type not in {
            "person", "anonymous_witness", "organisation", "place",
            "case", "case_or_inquiry", "alleged_network", "phrase"
        }:
            raise ValueError(f"Unsupported entity_type: {self.entity_type}")
        if not self.canonical_text.strip():
            raise ValueError("canonical_text must be non-empty")
        if not self.display_text.strip():
            raise ValueError("display_text must be non-empty")
        if not isinstance(self.aliases, (list, tuple)):
            raise ValueError("aliases must be a list")
        if not isinstance(self.stt_phrases, (list, tuple)):
            raise ValueError("stt_phrases must be a list")
        if self.translation_policy not in {"protected_token", "advisory"}:
            raise ValueError(f"Invalid translation_policy: {self.translation_policy}")
        for ref in self.do_not_confuse_with:
            if not _SAFE_IDENTIFIER.fullmatch(ref):
                raise ValueError(f"Invalid do_not_confuse_with reference: {ref}")

    def get_spoken_form(self, locale: str, voice: str | None = None) -> tuple[str, bool]:
        """Get approved spoken form for locale/voice, returning (text, needs_calibration)."""
        locale_config = self.spoken_forms.get(locale)
        if locale_config is None:
            return self.canonical_text, True
        if voice:
            voice_form = locale_config.voices.get(voice)
            if voice_form:
                return voice_form, False
        if locale_config.default:
            return locale_config.default, False
        return self.canonical_text, True


@dataclass(frozen=True)
class EntityMatch:
    """Result of entity matching in source text."""
    entity_id: str
    matched_source_text: str
    canonical_text: str
    display_text: str
    source_char_start: int
    source_char_end: int
    matched_alias: str
    match_confidence: str
    match_method: str
    domain: str
    role: str
    requires_pronunciation_calibration: bool


@dataclass(frozen=True)
class EntityBinding:
    """Job-local entity binding for a single unit."""
    unit_id: str
    protected_source_text: str
    bindings: dict[str, EntityMatch]
    placeholder_map: dict[str, str]  # placeholder -> entity_id


@dataclass(frozen=True)
class EntityBindingsArtifact:
    """Job-local entity binding artifact."""
    schema_version: str
    job_id: str
    registry_path: str
    registry_sha256: str
    registry_schema_version: str
    source_transcript_sha256: str
    generated_at: str
    entities_found_global: tuple[str, ...]
    per_unit_bindings: dict[str, dict[str, Any]]
    unresolved_ambiguous_matches: tuple[str, ...]
    protected_source_text_per_unit: dict[str, str]
    placeholder_mapping: dict[str, str]
    summary_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntityRegistry:
    """Typed, testable entity registry loader and matcher."""

    def __init__(self, registry_path: Path | None = None):
        self._registry_path = registry_path or Path(__file__).resolve().parents[2] / "config" / "entity_registry.json"
        self._entities: dict[str, Entity] = {}
        self._alias_index: dict[str, list[str]] = {}  # normalized alias -> entity_ids
        self._canonical_index: dict[str, str] = {}  # normalized canonical -> entity_id
        self._stt_phrase_index: dict[str, list[str]] = {}  # normalized phrase -> entity_ids
        self._alias_collisions: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        """Load and validate the entity registry."""
        if not self._registry_path.is_file():
            raise FileNotFoundError(f"Entity registry not found: {self._registry_path}")
        data = read_json(self._registry_path)
        
        schema_version = data.get("schema_version")
        if schema_version != ENTITY_REGISTRY_SCHEMA:
            raise ValueError(f"Unsupported registry schema version: {schema_version}")
        
        entity_ids = set()
        for item in data.get("entities", []):
            entity_id = item.get("entity_id")
            if not entity_id:
                raise ValueError("Entity missing entity_id")
            if entity_id in entity_ids:
                raise ValueError(f"Duplicate entity_id: {entity_id}")
            entity_ids.add(entity_id)
            
            entity = Entity(
                entity_id=entity_id,
                entity_type=item.get("entity_type"),
                canonical_text=item.get("canonical_text"),
                display_text=item.get("display_text"),
                aliases=tuple(item.get("aliases", [])),
                domains=tuple(item.get("domains", [])),
                roles=tuple(item.get("roles", [])),
                source_status=item.get("source_status", "unknown"),
                active=item.get("active", True),
                must_preserve=item.get("must_preserve", True),
                translation_policy=item.get("translation_policy", "protected_token"),
                stt_phrases=tuple(item.get("stt_phrases", [])),
                spoken_forms=self._parse_spoken_forms(item.get("spoken_forms", {})),
                do_not_confuse_with=tuple(item.get("do_not_confuse_with", [])),
                sources=tuple(item.get("sources", [])),
            )
            self._entities[entity_id] = entity
            self._index_entity(entity)
        
        # Validate do_not_confuse_with references
        for entity in self._entities.values():
            for ref in entity.do_not_confuse_with:
                if ref not in self._entities:
                    raise ValueError(f"Unknown do_not_confuse_with reference: {ref}")
        
        # Record alias collisions from validation section
        validation = data.get("validation", {})
        for collision in validation.get("alias_collisions", []):
            text = collision.get("text")
            entity_ids = collision.get("entity_ids", [])
            if text and entity_ids:
                self._alias_collisions[self._normalize_text(text)] = entity_ids

    def _parse_spoken_forms(self, data: dict[str, Any]) -> dict[str, SpokenForm]:
        """Parse spoken forms configuration."""
        result = {}
        for locale, config in data.items():
            if not isinstance(config, dict):
                continue
            result[locale] = SpokenForm(
                default=config.get("default"),
                voices=dict(config.get("voices", {})),
                status=config.get("status", "not_tested"),
                last_tested_at=config.get("last_tested_at"),
                approved_by=config.get("approved_by"),
            )
        return result

    def _index_entity(self, entity: Entity) -> None:
        """Index entity for fast matching."""
        # Index canonical
        norm_canonical = self._normalize_text(entity.canonical_text)
        if norm_canonical in self._canonical_index:
            self._alias_collisions.setdefault(norm_canonical, []).append(entity.entity_id)
        self._canonical_index[norm_canonical] = entity.entity_id
        
        # Index aliases
        for alias in entity.aliases:
            norm_alias = self._normalize_text(alias)
            self._alias_index.setdefault(norm_alias, []).append(entity.entity_id)
            if len(self._alias_index[norm_alias]) > 1:
                self._alias_collisions.setdefault(norm_alias, []).extend(self._alias_index[norm_alias])
        
        # Index STT phrases
        for phrase in entity.stt_phrases:
            norm_phrase = self._normalize_text(phrase)
            self._stt_phrase_index.setdefault(norm_phrase, []).append(entity.entity_id)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching (Unicode, punctuation, whitespace)."""
        # Replace curly quotes
        for curly, straight in _CURLY_QUOTES.items():
            text = text.replace(curly, straight)
        # Remove title variants
        text = _TITLE_VARIANTS.sub("", text)
        # Normalize Unicode
        text = unicodedata.normalize("NFKC", text)
        # Collapse whitespace and hyphens
        text = re.sub(r"\s+", " ", text)
        text = text.replace("-", " ")
        text = text.strip().lower()
        return text

    def match_entities(self, text: str, unit_id: str = "") -> list[EntityMatch]:
        """Match entities in source text using longest-match-first."""
        matches = []
        text_lower = self._normalize_text(text)
        
        # Build candidate matches (canonical, aliases, STT phrases)
        candidates = []
        for entity in self._entities.values():
            if not entity.active:
                continue
            
            # Check canonical
            norm_canonical = self._normalize_text(entity.canonical_text)
            if norm_canonical in text_lower:
                candidates.append((entity, norm_canonical, "canonical"))
            
            # Check aliases (skip ambiguous short ones)
            for alias in entity.aliases:
                norm_alias = self._normalize_text(alias)
                if norm_alias in text_lower:
                    # Skip ambiguous short aliases
                    if norm_alias.split() == 1 and norm_alias in _AMBIGUOUS_SHORT_ALIASES:
                        continue
                    # Skip if alias collision exists
                    if norm_alias in self._alias_collisions:
                        continue
                    candidates.append((entity, norm_alias, "alias"))
            
            # Check STT phrases
            for phrase in entity.stt_phrases:
                norm_phrase = self._normalize_text(phrase)
                if norm_phrase in text_lower:
                    candidates.append((entity, norm_phrase, "stt_phrase"))
        
        # Sort by length (longest first) for proper matching
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Apply matches with word boundaries
        matched_positions = set()
        for entity, norm_match, method in candidates:
            # Find all positions in original text
            pattern = re.compile(re.escape(norm_match), re.IGNORECASE)
            for match in pattern.finditer(text):
                start, end = match.span()
                # Check for word boundaries
                if (start > 0 and text[start - 1].isalnum()) or (end < len(text) and text[end].isalnum()):
                    continue
                # Check if position already used
                if any(pos < end and pos + len(norm_match) > start for pos in matched_positions):
                    continue
                
                # Get spoken form calibration status
                locale = "zu-ZA"  # Default target locale
                _, needs_cal = entity.get_spoken_form(locale)
                
                matches.append(EntityMatch(
                    entity_id=entity.entity_id,
                    matched_source_text=text[start:end],
                    canonical_text=entity.canonical_text,
                    display_text=entity.display_text,
                    source_char_start=start,
                    source_char_end=end,
                    matched_alias=norm_match,
                    match_confidence="high" if method == "canonical" else "medium",
                    match_method=method,
                    domain=entity.domains[0] if entity.domains else "unknown",
                    role=entity.roles[0].get("role", "unknown") if entity.roles else "unknown",
                    requires_pronunciation_calibration=needs_cal,
                ))
                matched_positions.add(start)
        
        return matches

    def get_entity(self, entity_id: str) -> Entity:
        """Get entity by ID."""
        if entity_id not in self._entities:
            raise ValueError(f"Unknown entity_id: {entity_id}")
        return self._entities[entity_id]

    @property
    def sha256(self) -> str:
        """SHA-256 of the registry file."""
        digest = hashlib.sha256()
        with self._registry_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @property
    def alias_collisions(self) -> dict[str, list[str]]:
        """Get recorded alias collisions."""
        return dict(self._alias_collisions)


def protect_text_with_placeholders(text: str, matches: list[EntityMatch]) -> tuple[str, dict[str, EntityMatch]]:
    """Replace matched entities with stable placeholders."""
    if not matches:
        return text, {}
    
    # Sort matches by position (reverse order to replace without offset issues)
    sorted_matches = sorted(matches, key=lambda m: m.source_char_start, reverse=True)
    
    protected_text = text
    bindings: dict[str, EntityMatch] = {}
    
    for match in sorted_matches:
        placeholder = f"[[MATHULA_ENTITY:{match.entity_id}]]"
        protected_text = (
            protected_text[:match.source_char_start] +
            placeholder +
            protected_text[match.source_char_end:]
        )
        bindings[placeholder] = match.entity_id
    
    return protected_text, bindings


def restore_display_text(text: str, bindings: dict[str, EntityMatch], registry: EntityRegistry) -> str:
    """Restore display text from placeholders."""
    for placeholder, match in bindings.items():
        entity = registry.get_entity(match.entity_id)
        text = text.replace(placeholder, entity.display_text)
    return text


def restore_tts_text(text: str, bindings: dict[str, EntityMatch], registry: EntityRegistry, locale: str = "zu-ZA", voice: str | None = None) -> str:
    """Restore approved spoken form from placeholders."""
    for placeholder, match in bindings.items():
        entity = registry.get_entity(match.entity_id)
        spoken_form, _ = entity.get_spoken_form(locale, voice)
        text = text.replace(placeholder, spoken_form)
    return text


def validate_placeholder_integrity(
    original_bindings: dict[str, EntityMatch],
    restored_bindings: dict[str, EntityMatch],
    unit_id: str = "",
) -> dict[str, Any]:
    """Validate that all placeholders survive exactly once."""
    original_placeholders = set(original_bindings.keys())
    restored_placeholders = set(restored_bindings.keys())
    
    missing = original_placeholders - restored_placeholders
    unexpected = restored_placeholders - original_placeholders
    
    # Check for duplicates
    placeholder_counts: dict[str, int] = {}
    for placeholder in restored_placeholders:
        placeholder_counts[placeholder] = placeholder_counts.get(placeholder, 0) + 1
    duplicates = [p for p, count in placeholder_counts.items() if count > 1]
    
    return {
        "unit_id": unit_id,
        "valid": len(missing) == 0 and len(unexpected) == 0 and len(duplicates) == 0,
        "missing_placeholders": sorted(missing),
        "unexpected_placeholders": sorted(unexpected),
        "duplicate_placeholders": duplicates,
        "protected_entities_expected": len(original_placeholders),
        "protected_entities_restored": len(restored_placeholders),
    }


__all__ = [
    "ENTITY_REGISTRY_SCHEMA",
    "ENTITY_BINDINGS_SCHEMA",
    "Entity",
    "EntityMatch",
    "EntityBinding",
    "EntityBindingsArtifact",
    "EntityRegistry",
    "protect_text_with_placeholders",
    "restore_display_text",
    "restore_tts_text",
    "validate_placeholder_integrity",
]
