from __future__ import annotations

from mathula_tv.tiktok_editor import (
    _choose_accessible_hook_text,
    _obscure_title_acronyms,
    _title_acronyms,
)


def test_r14_is_a_public_reference_not_an_unexplained_acronym() -> None:
    title = "Ubufakazi be-R14 bushiya imibuzo emisha"

    selected, policy = _choose_accessible_hook_text(
        {
            "hook_text": title,
            "accessible_hook_text": title,
            "title_lead": {"type": "topic", "text": "R14"},
        },
        target_language="zu-ZA",
    )

    assert selected == title
    assert _title_acronyms(title) == []
    assert _obscure_title_acronyms(title) == []
    assert policy["policy_version"] == "south-african-public-title-acronyms-v6"
    assert policy["fallback_applied"] is False


def test_single_letter_number_references_are_not_acronyms() -> None:
    assert _title_acronyms("R14 R21 R350 N1 G20") == []


def test_multi_letter_alphanumeric_label_still_uses_acronym_guard() -> None:
    assert _obscure_title_acronyms("XYZQ2: kuvela ubufakazi") == ["XYZQ2"]
