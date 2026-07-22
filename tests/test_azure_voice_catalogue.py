"""Tests for Azure voice catalogue."""

import pytest

from mathula_tv.azure_voice_catalogue import (
    AzureVoice,
    LocaleVoiceCatalogue,
    ZULU_CATALOGUE,
    DEFAULT_CATALOGUES,
    get_catalogue,
    resolve_voice_for_family,
    validate_voice_in_catalogue,
)


def test_azure_voice_creation():
    """Test AzureVoice dataclass creation."""
    voice = AzureVoice(
        name="zu-ZA-ThembaNeural",
        locale="zu-ZA",
        voice_family="masculine",
        gender="Male",
    )
    
    data = voice.to_dict()
    assert data["name"] == "zu-ZA-ThembaNeural"
    assert data["locale"] == "zu-ZA"
    assert data["voice_family"] == "masculine"
    assert data["gender"] == "Male"


def test_zulu_catalogue():
    """Test Zulu voice catalogue."""
    assert ZULU_CATALOGUE.locale == "zu-ZA"
    assert len(ZULU_CATALOGUE.voices) == 2
    
    voice_names = {v.name for v in ZULU_CATALOGUE.voices}
    assert "zu-ZA-ThembaNeural" in voice_names
    assert "zu-ZA-ThandoNeural" in voice_names


def test_get_voice_by_family():
    """Test getting voice by family."""
    masculine_voice = ZULU_CATALOGUE.get_voice_by_family("masculine")
    assert masculine_voice is not None
    assert masculine_voice.voice_family == "masculine"
    
    feminine_voice = ZULU_CATALOGUE.get_voice_by_family("feminine")
    assert feminine_voice is not None
    assert feminine_voice.voice_family == "feminine"
    
    unknown_voice = ZULU_CATALOGUE.get_voice_by_family("neutral")
    assert unknown_voice is None


def test_get_voice_by_name():
    """Test getting voice by name."""
    voice = ZULU_CATALOGUE.get_voice_by_name("zu-ZA-ThembaNeural")
    assert voice is not None
    assert voice.name == "zu-ZA-ThembaNeural"
    
    voice = ZULU_CATALOGUE.get_voice_by_name("invalid-voice")
    assert voice is None


def test_available_families():
    """Test getting available voice families."""
    families = ZULU_CATALOGUE.available_families()
    assert "masculine" in families
    assert "feminine" in families


def test_get_catalogue():
    """Test getting catalogue by locale."""
    catalogue = get_catalogue("zu-ZA")
    assert catalogue.locale == "zu-ZA"
    assert len(catalogue.voices) > 0


def test_get_catalogue_unsupported_locale():
    """Test error for unsupported locale."""
    with pytest.raises(ValueError, match="Unsupported locale"):
        get_catalogue("en-US")


def test_resolve_voice_for_family():
    """Test resolving voice for family."""
    voice, status = resolve_voice_for_family("zu-ZA", "masculine")
    assert voice == "zu-ZA-ThembaNeural"
    assert status == "resolved_by_voice_family"
    
    voice, status = resolve_voice_for_family("zu-ZA", "feminine")
    assert voice == "zu-ZA-ThandoNeural"
    assert status == "resolved_by_voice_family"


def test_resolve_voice_for_family_fallback():
    """Test fallback voice resolution."""
    voice, status = resolve_voice_for_family("zu-ZA", "neutral", fallback_allowed=True)
    assert voice is not None
    assert "fallback" in status


def test_resolve_voice_for_family_no_fallback():
    """Test error when fallback not allowed."""
    with pytest.raises(ValueError, match="No voice available"):
        resolve_voice_for_family("zu-ZA", "neutral", fallback_allowed=False)


def test_validate_voice_in_catalogue():
    """Test voice validation."""
    assert validate_voice_in_catalogue("zu-ZA", "zu-ZA-ThembaNeural") is True
    assert validate_voice_in_catalogue("zu-ZA", "invalid-voice") is False
    assert validate_voice_in_catalogue("en-US", "zu-ZA-ThembaNeural") is False
