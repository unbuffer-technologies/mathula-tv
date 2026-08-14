"""Generate publication SEO from an approved target-language transcript."""
from __future__ import annotations

from typing import Any, Mapping

import jsonschema

from .ai_provider import AIProvider


LOCALIZED_SEO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "caption",
        "cover_hook",
        "search_keywords",
        "topic_hashtags",
        "human_review_flags",
    ],
    "properties": {
        "caption": {"type": "string", "minLength": 1},
        "cover_hook": {"type": "string", "minLength": 1, "maxLength": 100},
        "search_keywords": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {"type": "string", "minLength": 1},
        },
        "topic_hashtags": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^#[A-Za-z0-9]+$"},
        },
        "human_review_flags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


def build_deferred_localized_seo(
    *,
    job_id: str,
    target_language: str,
) -> dict[str, Any]:
    """Create a non-AI placeholder until rendered blocks are editorially reviewed.

    The single high-thinking editorial call runs after dubbing, when accurate
    rendered block IDs and timing boundaries exist. This placeholder exists only
    so the clean dubbed master and provisional sidecars can be published safely.
    """
    provisional_title = f"Mathula TV {job_id[:8]}"
    return {
        "schema_version": "mathula-editorial-deferred-v1",
        "platform": "tiktok",
        "job_id": job_id,
        "language": target_language,
        "status": "pending_editorial_call",
        "title": provisional_title,
        "cover_hook": provisional_title,
        "caption": "",
        "tiktok_caption": "",
        "search_keywords": [],
        "topic_hashtags": [],
        "tiktok_hashtags": [],
        "hashtags": [],
        "human_review_flags": ["single_editorial_call_pending"],
        "human_review_required": True,
        "source": "deferred-until-rendered-blocks-exist",
        "approved_translation_immutable": True,
        "ai_generation": None,
    }


def generate_localized_seo(
    *,
    provider: AIProvider,
    job_id: str,
    target_language: str,
    english_transcript: Mapping[str, Any],
    approved_translation: Mapping[str, Any],
    classification: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate SEO only; the approved translated transcript is immutable."""
    payload = {
        "job_id": job_id,
        "target_language": target_language,
        "english_transcript": english_transcript,
        "approved_target_transcript": approved_translation,
        "classification": classification,
        "grounded_context": context,
        "publication_requirements": {
            "write_cover_hook_in_target_language": True,
            "write_search_caption_in_english": True,
            "write_search_keywords_in_english": True,
            "approved_transcript_is_immutable": True,
            "preserve_names_brands_official_titles_and_risky_english_spans": True,
            "do_not_invent_or_strengthen_claims": True,
            "retain_attribution_and_uncertainty": True,
            "tiktok_hashtags_are_topic_only": True,
            "configured_community_hashtag_is_added_later": True,
            "do_not_generate_brand_language_or_geographic_padding_hashtags": True,
            "exact_story_hashtags": 4,
            "human_review_required_before_publication": True,
        },
    }
    response = provider.generate_seo(
        payload,
        output_schema=LOCALIZED_SEO_SCHEMA,
        response_schema_version="tiktok-seo-v1",
    )
    result = dict(response.data)
    jsonschema.validate(result, LOCALIZED_SEO_SCHEMA)
    keywords = [
        str(value).strip()
        for value in result["search_keywords"]
        if str(value).strip()
    ]
    topic_hashtags = list(dict.fromkeys(result["topic_hashtags"]))
    caption = result["caption"].strip()
    cover_hook = result["cover_hook"].strip()
    return {
        "schema_version": "mathula-tiktok-seo-v1",
        "platform": "tiktok",
        "job_id": job_id,
        "language": target_language,
        # title is an internal renderer label, not a YouTube deliverable.
        "title": cover_hook,
        "caption": caption,
        "tiktok_caption": caption,
        "caption_language": "en",
        "cover_hook": cover_hook,
        "search_keywords": keywords,
        "topic_hashtags": topic_hashtags,
        "tiktok_hashtags": topic_hashtags,
        "hashtags": topic_hashtags,
        "hashtag_policy": {
            "policy_version": "one-community-plus-four-story-tags-v1",
            "configured_community_hashtag_added_at_publication": True,
            "story_hashtag_count": 4,
            "brand_hashtag_added": False,
            "geographic_padding_hashtags_added": False,
        },
        "human_review_flags": list(result["human_review_flags"]),
        "human_review_required": True,
        "source": "ai-localized-seo-from-approved-translation",
        "approved_translation_immutable": True,
        "ai_generation": response.metadata.to_dict(),
    }
