"""Tests for build_group_bookmark_ssml, the bookmark-tagged SSML builder used by
the Speech SDK grouped-synthesis path (see speech_islands.slice_wav_by_bookmarks).
"""
import pytest

from mathula_tv.tts_ssml import (
    BreakPart,
    CharacterPart,
    SSMLBounds,
    SSMLValidationError,
    TextPart,
    build_group_bookmark_ssml,
    validate_ssml,
)

_VOICE = "zu-ZA-ThandoNeural"
_ALLOWED_VOICES = {_VOICE}


def test_builds_valid_ssml_with_one_bookmark_per_island():
    islands = [
        ("native_phrase_0001__isl00", "Sawubona."),
        ("native_phrase_0001__isl01", "Ninjani?"),
    ]
    xml, digest = build_group_bookmark_ssml(islands, voice=_VOICE, allowed_voices=_ALLOWED_VOICES)
    assert xml.count("<bookmark") == 2
    assert 'mark="native_phrase_0001__isl00"' in xml
    assert 'mark="native_phrase_0001__isl01"' in xml
    assert "Sawubona." in xml
    assert "Ninjani?" in xml
    import hashlib
    assert digest == hashlib.sha256(xml.encode("utf-8")).hexdigest()


def test_output_passes_validate_ssml():
    islands = [
        ("g1__isl00", "Sawubona."),
        ("g1__isl01", "Ninjani?"),
        ("g1__isl02", "Ngiyabonga kakhulu."),
    ]
    xml, _ = build_group_bookmark_ssml(islands, voice=_VOICE, allowed_voices=_ALLOWED_VOICES)
    metadata = validate_ssml(xml, bounds=SSMLBounds(), allowed_voices=_ALLOWED_VOICES)
    assert metadata["voice"] == _VOICE


def test_rejects_voice_not_in_allowlist():
    with pytest.raises(SSMLValidationError, match="not allowlisted"):
        build_group_bookmark_ssml(
            [("g1__isl00", "Sawubona.")], voice="zu-ZA-OtherNeural", allowed_voices=_ALLOWED_VOICES
        )


def test_rejects_empty_islands():
    with pytest.raises(SSMLValidationError, match="at least one island"):
        build_group_bookmark_ssml([], voice=_VOICE, allowed_voices=_ALLOWED_VOICES)


def test_rejects_empty_island_text():
    with pytest.raises(SSMLValidationError, match="cannot be empty"):
        build_group_bookmark_ssml(
            [("g1__isl00", "   ")], voice=_VOICE, allowed_voices=_ALLOWED_VOICES
        )


def test_rejects_duplicate_bookmark_marks():
    with pytest.raises(SSMLValidationError, match="Duplicate"):
        build_group_bookmark_ssml(
            [("g1__isl00", "Sawubona."), ("g1__isl00", "Ninjani?")],
            voice=_VOICE,
            allowed_voices=_ALLOWED_VOICES,
        )


def test_rejects_unsafe_bookmark_mark():
    # island ids are always f"{group_id}__isl{index:02d}" in production, but the
    # builder must not trust that blindly -- validate defensively.
    with pytest.raises(SSMLValidationError, match="Invalid SSML bookmark mark"):
        build_group_bookmark_ssml(
            [("g1__isl00<script>", "Sawubona.")], voice=_VOICE, allowed_voices=_ALLOWED_VOICES
        )


def test_rejects_bookmark_mark_containing_quote_or_space():
    with pytest.raises(SSMLValidationError, match="Invalid SSML bookmark mark"):
        build_group_bookmark_ssml(
            [('g1" isl00', "Sawubona.")], voice=_VOICE, allowed_voices=_ALLOWED_VOICES
        )


def test_escapes_special_characters_in_island_text():
    xml, _ = build_group_bookmark_ssml(
        [("g1__isl00", "Tom & Jerry <said> \"hi\"")], voice=_VOICE, allowed_voices=_ALLOWED_VOICES
    )
    assert "<said>" not in xml
    assert "&amp;" in xml
    metadata = validate_ssml(xml, bounds=SSMLBounds(), allowed_voices=_ALLOWED_VOICES)
    assert metadata["voice"] == _VOICE


def test_rejects_text_exceeding_character_bound():
    tight_bounds = SSMLBounds(max_text_characters=10)
    with pytest.raises(SSMLValidationError, match="exceeds the configured character limit"):
        build_group_bookmark_ssml(
            [("g1__isl00", "This island text is far longer than ten characters.")],
            voice=_VOICE,
            allowed_voices=_ALLOWED_VOICES,
            bounds=tight_bounds,
        )


def test_rejects_out_of_bounds_rate():
    bounds = SSMLBounds()
    with pytest.raises(SSMLValidationError, match="rate"):
        build_group_bookmark_ssml(
            [("g1__isl00", "Sawubona.")],
            voice=_VOICE,
            allowed_voices=_ALLOWED_VOICES,
            rate_percent=bounds.rate_max_percent + 1,
        )


def test_preserves_island_order_in_xml():
    islands = [
        ("g1__isl00", "First."),
        ("g1__isl01", "Second."),
        ("g1__isl02", "Third."),
    ]
    xml, _ = build_group_bookmark_ssml(islands, voice=_VOICE, allowed_voices=_ALLOWED_VOICES)
    assert xml.index("isl00") < xml.index("isl01") < xml.index("isl02")
    assert xml.index("First.") < xml.index("Second.") < xml.index("Third.")


def test_island_content_may_be_typed_parts_with_a_character_mode_acronym():
    # Real motivating case: "ANC" must be spelled letter-by-letter, not read
    # as a whole word, without disturbing the plain text around it.
    islands = [
        ("g1__isl00", (TextPart("Lawo mawele e-"), CharacterPart("ANC"), TextPart(" kumele azi."))),
    ]
    xml, _ = build_group_bookmark_ssml(islands, voice=_VOICE, allowed_voices=_ALLOWED_VOICES)
    assert '<say-as interpret-as="characters">ANC</say-as>' in xml
    assert "Lawo mawele e-" in xml
    assert " kumele azi." in xml
    metadata = validate_ssml(xml, bounds=SSMLBounds(), allowed_voices=_ALLOWED_VOICES)
    assert metadata["voice"] == _VOICE


def test_typed_parts_island_still_gets_its_own_bookmark():
    islands = [
        ("g1__isl00", "Sawubona."),
        ("g1__isl01", (TextPart("e-"), CharacterPart("EFF"))),
    ]
    xml, _ = build_group_bookmark_ssml(islands, voice=_VOICE, allowed_voices=_ALLOWED_VOICES)
    assert xml.index('mark="g1__isl00"') < xml.index('mark="g1__isl01"') < xml.index("EFF")


def test_typed_parts_string_and_object_render_identically_for_plain_text():
    # A single TextPart must render byte-identically to the plain-string form
    # -- proof this is a real generalization, not a separate code path.
    string_xml, _ = build_group_bookmark_ssml(
        [("g1__isl00", "Ngiyabonga.")], voice=_VOICE, allowed_voices=_ALLOWED_VOICES
    )
    typed_xml, _ = build_group_bookmark_ssml(
        [("g1__isl00", (TextPart("Ngiyabonga."),))], voice=_VOICE, allowed_voices=_ALLOWED_VOICES
    )
    assert string_xml == typed_xml


def test_character_part_must_be_letters_or_digits_only():
    with pytest.raises(SSMLValidationError, match="letters or digits"):
        build_group_bookmark_ssml(
            [("g1__isl00", (CharacterPart("AN C"),))], voice=_VOICE, allowed_voices=_ALLOWED_VOICES,
        )


def test_rejects_unsupported_part_type_in_an_island():
    with pytest.raises(SSMLValidationError, match="TextPart/CharacterPart"):
        build_group_bookmark_ssml(
            [("g1__isl00", (BreakPart(200),))], voice=_VOICE, allowed_voices=_ALLOWED_VOICES,
        )


def test_rejects_typed_parts_island_with_no_real_spoken_text():
    with pytest.raises(SSMLValidationError, match="cannot be empty"):
        build_group_bookmark_ssml(
            [("g1__isl00", (TextPart("   "),))], voice=_VOICE, allowed_voices=_ALLOWED_VOICES,
        )
