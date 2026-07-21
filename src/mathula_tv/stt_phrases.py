from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STT_PHRASE_BUNDLE_SCHEMA = "mathula-stt-phrase-bundle-v1"
DEFAULT_MAX_PHRASES = 450
DEFAULT_BIASING_WEIGHT = 1.6

UNSAFE_GENERIC_SINGLE_TOKENS = {
    "cat",
    "ci",
    "mk",
    "khan",
    "johnson",
    "jacob",
}

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class STTPhraseBundle:
    registry_path: str
    registry_sha256: str
    registry_schema_version: str
    overrides_path: str | None
    overrides_sha256: str | None
    phrases: tuple[str, ...]
    max_phrases: int
    biasing_weight: float
    excluded_unsafe_aliases: tuple[str, ...]
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STT_PHRASE_BUNDLE_SCHEMA,
            "registry_path": self.registry_path,
            "registry_sha256": self.registry_sha256,
            "registry_schema_version": self.registry_schema_version,
            "overrides_path": self.overrides_path,
            "overrides_sha256": self.overrides_sha256,
            "phrase_count": len(self.phrases),
            "max_phrases": self.max_phrases,
            "biasing_weight": self.biasing_weight,
            "excluded_unsafe_aliases": list(self.excluded_unsafe_aliases),
            "truncated": self.truncated,
            "phrases": list(self.phrases),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_phrase(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _WHITESPACE.sub(" ", value).strip()
    if not cleaned or "[[MATHULA_ENTITY:" in cleaned:
        return None
    return cleaned


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def load_stt_phrase_bundle(
    *,
    registry_path: Path | None = None,
    overrides_path: Path | None = None,
    max_phrases: int | None = None,
    biasing_weight: float | None = None,
) -> STTPhraseBundle:
    root = _repo_root()
    registry_path = Path(
        registry_path
        or os.environ.get("MATHULA_TV_STT_ENTITY_REGISTRY")
        or root / "config/entity_registry.json"
    )
    overrides_path = Path(
        overrides_path
        or os.environ.get("MATHULA_TV_STT_PHRASE_OVERRIDES")
        or root / "config/stt_phrase_overrides.json"
    )
    max_phrases = int(
        max_phrases
        if max_phrases is not None
        else os.environ.get("MATHULA_TV_STT_MAX_PHRASES", DEFAULT_MAX_PHRASES)
    )
    biasing_weight = float(
        biasing_weight
        if biasing_weight is not None
        else os.environ.get(
            "MATHULA_TV_STT_PHRASE_BIASING_WEIGHT",
            DEFAULT_BIASING_WEIGHT,
        )
    )

    if max_phrases < 1:
        raise ValueError("MATHULA_TV_STT_MAX_PHRASES must be positive")
    if not 0.0 <= biasing_weight <= 2.0:
        raise ValueError("Azure phrase-list biasing weight must be between 0.0 and 2.0")
    if not registry_path.is_file():
        raise FileNotFoundError(f"Entity registry not found: {registry_path}")

    registry = _read_json(registry_path)
    entities = registry.get("entities")
    if not isinstance(entities, list):
        raise ValueError(f"Registry has no entities list: {registry_path}")

    candidates: list[tuple[int, str, str]] = []
    excluded: set[str] = set()

    overrides_sha256: str | None = None
    stored_overrides_path: str | None = None
    if overrides_path.is_file():
        overrides = _read_json(overrides_path)
        phrases = overrides.get("phrases", [])
        if not isinstance(phrases, list):
            raise ValueError(f"Override phrases must be a list: {overrides_path}")
        for phrase in phrases:
            cleaned = _clean_phrase(phrase)
            if cleaned:
                candidates.append((0, cleaned, "override"))
        overrides_sha256 = _sha256(overrides_path)
        stored_overrides_path = str(overrides_path)

    for entity in entities:
        if not isinstance(entity, dict) or not entity.get("active", True):
            continue

        for phrase in entity.get("stt_phrases", []) or []:
            cleaned = _clean_phrase(phrase)
            if cleaned:
                candidates.append((1, cleaned, "stt_phrase"))

        for key in ("canonical_text", "display_text"):
            cleaned = _clean_phrase(entity.get(key))
            if cleaned:
                candidates.append((2, cleaned, key))

        for alias in entity.get("aliases", []) or []:
            cleaned = _clean_phrase(alias)
            if not cleaned:
                continue
            if " " not in cleaned and cleaned.casefold() in UNSAFE_GENERIC_SINGLE_TOKENS:
                excluded.add(cleaned)
                continue
            candidates.append((3, cleaned, "alias"))

    candidates.sort(key=lambda item: (item[0], -len(item[1]), item[1].casefold()))

    seen: set[str] = set()
    selected: list[str] = []
    total_unique = 0
    for _, phrase, _source in candidates:
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        total_unique += 1
        if len(selected) < max_phrases:
            selected.append(phrase)

    return STTPhraseBundle(
        registry_path=str(registry_path),
        registry_sha256=_sha256(registry_path),
        registry_schema_version=str(registry.get("schema_version", "")),
        overrides_path=stored_overrides_path,
        overrides_sha256=overrides_sha256,
        phrases=tuple(selected),
        max_phrases=max_phrases,
        biasing_weight=biasing_weight,
        excluded_unsafe_aliases=tuple(sorted(excluded, key=str.casefold)),
        truncated=total_unique > max_phrases,
    )


def configure_source_stt_backend(backend: Any) -> STTPhraseBundle | None:
    """Attach registry phrases only to source Fast STT, never blind QC STT."""

    fast = getattr(backend, "fast", backend)
    if not hasattr(fast, "phrase_list"):
        return None

    bundle = load_stt_phrase_bundle()
    object.__setattr__(fast, "phrase_list", bundle.phrases)
    object.__setattr__(fast, "phrase_biasing_weight", bundle.biasing_weight)
    return bundle
