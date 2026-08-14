from __future__ import annotations

import json

from mathula_tv.output_naming import _tiktok_caption_text, _tiktok_hashtag_values
from mathula_tv.tiktok_editor import (
    EDITORIAL_PACKAGE_SCHEMA,
    TIKTOK_HOOK_PROMPT_VERSION,
    _normalize_editorial_package_wire,
)


def test_editorial_contract_uses_english_caption_and_four_story_tags() -> None:
    assert TIKTOK_HOOK_PROMPT_VERSION.endswith("featured-people-discovery")
    hashtag_schema = EDITORIAL_PACKAGE_SCHEMA["properties"]["topic_hashtags"]
    assert hashtag_schema["minItems"] == 4
    assert hashtag_schema["maxItems"] == 4
    assert hashtag_schema["uniqueItems"] is True


def test_pktt_hashtag_keeps_full_searchable_name_in_caption() -> None:
    normalized = _normalize_editorial_package_wire(
        {
            "caption": "A former minister faces questions about the task team.",
            "primary_story_angle": "Questions about the task team",
            "search_keywords": [
                "Political Killings Task Team",
                "Nathi Mthethwa",
                "Madlanga Commission",
                "South African Police Service",
            ],
            "topic_hashtags": [
                "#PoliticalKillingsTaskTeam",
                "#NathiMthethwa",
                "#MadlangaCommission",
                "#SAPS",
            ],
            "human_review_flags": [],
        }
    )

    assert normalized["topic_hashtags"] == [
        "#NathiMthethwa",
        "#PKTT",
        "#MadlangaCommission",
        "#SAPS",
    ]
    assert "Political Killings Task Team (PKTT)" in normalized["caption"]


def test_publication_uses_one_community_and_four_dynamic_tags(
    tmp_path, monkeypatch
) -> None:
    registry = tmp_path / "tiktok_accounts.json"
    registry.write_text(
        json.dumps(
            {
                "accounts": {
                    "zu": {
                        "account_name": "Mathula TV",
                        "brand_hashtag": "#MathulaTV",
                        "community_hashtag": "#ZuluTikTok",
                        "value_hashtag": "#NgesiZulu",
                        "discovery_hashtags": [],
                    }
                },
                "shared_hashtags": ["#Mzansi", "#SouthAfrica"],
                "max_hashtags": 5,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MATHULA_TIKTOK_ACCOUNTS_PATH", str(registry))
    seo = {
        "tiktok_caption": (
            "Nathi Mthethwa faces questions about the Political Killings "
            "Task Team (PKTT)."
        ),
        "tiktok_hashtags": [
            "#PoliticalKillingsTaskTeam",
            "#NathiMthethwa",
            "#MadlangaCommission",
            "#SAPS",
        ],
    }

    hashtags = _tiktok_hashtag_values(seo, "zu")

    assert hashtags == [
        "#ZuluTikTok",
        "#PKTT",
        "#NathiMthethwa",
        "#MadlangaCommission",
        "#SAPS",
    ]
    assert "#MathulaTV" not in hashtags
    assert "#NgesiZulu" not in hashtags
    assert "#Mzansi" not in hashtags
    assert "#SouthAfrica" not in hashtags
    assert _tiktok_caption_text(seo, "zu").endswith(" ".join(hashtags))
