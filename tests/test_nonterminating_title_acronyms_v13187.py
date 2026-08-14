from __future__ import annotations

from mathula_tv.tiktok_editor import (
    _choose_accessible_hook_text,
    _obscure_title_acronyms,
)


def test_ipid_is_a_reviewed_south_african_public_title_acronym() -> None:
    title, policy = _choose_accessible_hook_text(
        {
            "hook_text": "IPID: kuvela ubufakazi obusha",
            "accessible_hook_text": "IPID: kuvela ubufakazi obusha",
            "title_lead": {"type": "organisation", "text": "IPID"},
        },
        target_language="zu-ZA",
    )

    assert title == "IPID: kuvela ubufakazi obusha"
    assert _obscure_title_acronyms(title) == []
    assert policy["unresolved_acronyms"] == []
    assert policy["allowlisted_acronyms"] == ["IPID"]


def test_unknown_title_acronym_never_terminates_completed_dub() -> None:
    title, policy = _choose_accessible_hook_text(
        {
            "hook_text": "XYZQ: kuvela ubufakazi obusha",
            "accessible_hook_text": "XYZQ: kuvela ubufakazi obusha",
            "title_lead": {"type": "organisation", "text": "XYZQ"},
        },
        target_language="zu-ZA",
    )

    assert title
    assert policy["unresolved_acronyms"] == ["XYZQ"]
    assert policy["deterministic_rewrite"]["unresolved_acronyms"] == ["XYZQ"]
