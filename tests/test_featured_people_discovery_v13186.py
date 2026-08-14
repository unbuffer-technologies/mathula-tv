from __future__ import annotations

import json

from mathula_tv.output_naming import _tiktok_caption_text
from mathula_tv.tiktok_editor import _normalize_editorial_package_wire


def _two_subject_editorial_wire() -> dict:
    return {
        "caption": (
            "Ramsamy explains why the testimony raises new questions about the case."
        ),
        "featured_person_names": [
            "Advocate Andrea Johnson",
            "General Dumisani Khumalo",
        ],
        "primary_story_angle": "Ramsamy discusses Johnson and Khumalo",
        "search_keywords": [
            "Andrea Johnson",
            "Dumisani Khumalo",
            "Ramsamy",
            "Madlanga Commission",
        ],
        "topic_hashtags": [
            "#AdvocateJohnson",
            "#GeneralKhumalo",
            "#Ramsamy",
            "#MadlangaCommission",
        ],
        "human_review_flags": [],
    }


def test_two_discussed_people_outrank_witness_and_use_full_name_tags() -> None:
    normalized = _normalize_editorial_package_wire(_two_subject_editorial_wire())

    assert normalized["featured_person_names"] == [
        "Andrea Johnson",
        "Dumisani Khumalo",
    ]
    assert normalized["topic_hashtags"] == [
        "#AndreaJohnson",
        "#DumisaniKhumalo",
        "#Ramsamy",
        "#MadlangaCommission",
    ]
    assert normalized["caption"].startswith(
        "Andrea Johnson and Dumisani Khumalo — Ramsamy"
    )
    assert "Ramsamy" in normalized["caption"]


def test_final_caption_keeps_two_people_with_zulu_community_tag(
    tmp_path, monkeypatch
) -> None:
    registry = tmp_path / "tiktok_accounts.json"
    registry.write_text(
        json.dumps(
            {
                "accounts": {
                    "zu": {
                        "account_name": "Mathula TV",
                        "community_hashtag": "#ZuluTikTok",
                        "value_hashtag": "#NgesiZulu",
                    }
                },
                "shared_hashtags": ["#Mzansi", "#SouthAfrica"],
                "max_hashtags": 5,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MATHULA_TIKTOK_ACCOUNTS_PATH", str(registry))
    normalized = _normalize_editorial_package_wire(_two_subject_editorial_wire())
    seo = {
        "tiktok_caption": normalized["caption"],
        "tiktok_hashtags": normalized["topic_hashtags"],
    }

    published = _tiktok_caption_text(seo, "zu")

    assert published.endswith(
        "#ZuluTikTok #AndreaJohnson #DumisaniKhumalo "
        "#Ramsamy #MadlangaCommission"
    )
    assert "#AdvocateJohnson" not in published
    assert "#GeneralKhumalo" not in published
