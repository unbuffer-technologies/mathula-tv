from __future__ import annotations

from mathula_tv.tiktok_editor import (
    _choose_accessible_hook_text,
    _obscure_title_acronyms,
)


def test_cctv_is_a_reviewed_public_title_acronym() -> None:
    title, policy = _choose_accessible_hook_text(
        {
            "hook_text": "CCTV: kuvela ubufakazi obusha",
            "accessible_hook_text": "CCTV: kuvela ubufakazi obusha",
            "title_lead": {"type": "topic", "text": "CCTV evidence"},
        },
        target_language="zu-ZA",
    )

    assert title == "CCTV: kuvela ubufakazi obusha"
    assert _obscure_title_acronyms(title) == []
    assert policy["policy_version"] == "south-african-public-title-acronyms-v6"
    assert policy["fallback_applied"] is False
    assert policy["allowlisted_acronyms"] == ["CCTV"]


def test_cctv_is_recognized_inside_an_isizulu_title() -> None:
    assert _obscure_title_acronyms(
        "Ubufakazi be-CCTV buphikisa ubufakazi bakhe"
    ) == []


def test_unreviewed_acronym_is_audited_without_terminating_publication() -> None:
    title, policy = _choose_accessible_hook_text(
        {
            "hook_text": "XYZQ: kuvela ubufakazi obusha",
            "accessible_hook_text": "XYZQ: kuvela ubufakazi obusha",
            "title_lead": {"type": "organisation", "text": "XYZQ"},
        },
        target_language="zu-ZA",
    )

    assert title == "XYZQ: kuvela ubufakazi obusha"
    assert policy["unresolved_acronyms"] == ["XYZQ"]
