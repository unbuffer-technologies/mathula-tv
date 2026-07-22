"""Target-locale Azure voice catalogue for deterministic voice resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AzureVoice:
    """Azure TTS voice with voice family classification."""
    
    name: str
    locale: str
    voice_family: str  # "masculine" | "feminine" | "neutral"
    gender: str  # "Male" | "Female" | "Neutral" (Azure's classification)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "locale": self.locale,
            "voice_family": self.voice_family,
            "gender": self.gender,
        }


@dataclass
class LocaleVoiceCatalogue:
    """Azure voice catalogue for a specific locale."""
    
    locale: str
    voices: list[AzureVoice]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "locale": self.locale,
            "voices": [v.to_dict() for v in self.voices],
        }
    
    def get_voice_by_family(self, voice_family: str) -> AzureVoice | None:
        """Get a voice by voice family, or None if not available."""
        for voice in self.voices:
            if voice.voice_family == voice_family:
                return voice
        return None
    
    def get_voice_by_name(self, name: str) -> AzureVoice | None:
        """Get a voice by name, or None if not found."""
        for voice in self.voices:
            if voice.name == name:
                return voice
        return None
    
    def available_families(self) -> set[str]:
        """Return set of available voice families in this locale."""
        return {v.voice_family for v in self.voices}


# Default isiZulu (zu-ZA) voice catalogue
ZULU_CATALOGUE = LocaleVoiceCatalogue(
    locale="zu-ZA",
    voices=[
        AzureVoice(
            name="zu-ZA-ThembaNeural",
            locale="zu-ZA",
            voice_family="masculine",
            gender="Male",
        ),
        AzureVoice(
            name="zu-ZA-ThandoNeural",
            locale="zu-ZA",
            voice_family="feminine",
            gender="Female",
        ),
    ],
)


# Catalogue for all supported locales
DEFAULT_CATALOGUES: dict[str, LocaleVoiceCatalogue] = {
    "zu-ZA": ZULU_CATALOGUE,
}


def get_catalogue(locale: str) -> LocaleVoiceCatalogue:
    """Get voice catalogue for a locale.
    
    Args:
        locale: Target locale (e.g., "zu-ZA")
    
    Returns:
        LocaleVoiceCatalogue for the locale
    
    Raises:
        ValueError: If locale is not supported
    """
    if locale not in DEFAULT_CATALOGUES:
        raise ValueError(
            f"Unsupported locale '{locale}'. "
            f"Supported locales: {sorted(DEFAULT_CATALOGUES.keys())}"
        )
    return DEFAULT_CATALOGUES[locale]


def resolve_voice_for_family(
    locale: str,
    voice_family: str,
    fallback_allowed: bool = False,
) -> tuple[str, str]:
    """Resolve an Azure voice name for a given voice family.
    
    Args:
        locale: Target locale (e.g., "zu-ZA")
        voice_family: Desired voice family ("masculine" | "feminine" | "neutral")
        fallback_allowed: If True, allow fallback to any available voice
    
    Returns:
        tuple of (voice_name, resolution_status)
    
    Raises:
        ValueError: If voice family is not available and fallback is not allowed
    """
    catalogue = get_catalogue(locale)
    voice = catalogue.get_voice_by_family(voice_family)
    
    if voice is not None:
        return voice.name, "resolved_by_voice_family"
    
    if fallback_allowed and catalogue.voices:
        # Fallback to first available voice
        fallback_voice = catalogue.voices[0]
        return fallback_voice.name, f"fallback_to_{fallback_voice.voice_family}"
    
    raise ValueError(
        f"No voice available for voice family '{voice_family}' in locale '{locale}'. "
        f"Available families: {sorted(catalogue.available_families())}"
    )


def load_catalogue_from_config(config_path: Path) -> dict[str, LocaleVoiceCatalogue]:
    """Load voice catalogue from a configuration file.
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        Dictionary mapping locale to LocaleVoiceCatalogue
    """
    if not config_path.is_file():
        raise ValueError(f"Voice catalogue config not found: {config_path}")
    
    data = json.loads(config_path.read_text(encoding="utf-8"))
    
    catalogues = {}
    for locale_data in data.get("locales", []):
        locale = locale_data.get("locale")
        if not locale:
            continue
        
        voices = []
        for voice_data in locale_data.get("voices", []):
            voice = AzureVoice(
                name=voice_data.get("name", ""),
                locale=locale,
                voice_family=voice_data.get("voice_family", "neutral"),
                gender=voice_data.get("gender", "Neutral"),
            )
            voices.append(voice)
        
        catalogues[locale] = LocaleVoiceCatalogue(locale=locale, voices=voices)
    
    return catalogues


def validate_voice_in_catalogue(locale: str, voice_name: str) -> bool:
    """Validate that a voice name exists in the catalogue for a locale.
    
    Args:
        locale: Target locale
        voice_name: Azure voice name to validate
    
    Returns:
        True if voice exists in catalogue, False otherwise
    """
    try:
        catalogue = get_catalogue(locale)
        return catalogue.get_voice_by_name(voice_name) is not None
    except ValueError:
        return False


__all__ = [
    "AzureVoice",
    "LocaleVoiceCatalogue",
    "ZULU_CATALOGUE",
    "DEFAULT_CATALOGUES",
    "get_catalogue",
    "resolve_voice_for_family",
    "load_catalogue_from_config",
    "validate_voice_in_catalogue",
]
