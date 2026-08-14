from dataclasses import dataclass, field

from mathula_tv.localized_seo import generate_localized_seo


@dataclass
class _Metadata:
    def to_dict(self):
        return {"provider": "test", "model_requested": "test-model"}


@dataclass
class _Response:
    data: dict
    metadata: _Metadata = field(default_factory=_Metadata)


class _Provider:
    def __init__(self):
        self.payload = None

    def generate_seo(self, payload, *, output_schema, response_schema_version):
        self.payload = payload
        assert response_schema_version == "tiktok-seo-v1"
        assert output_schema["properties"]["topic_hashtags"]["maxItems"] == 4
        return _Response(
            {
                "caption": "President Ramaphosa faces renewed Section 89 questions.",
                "cover_hook": "BUKA OKWENZEKE",
                "search_keywords": ["izindaba ngesiZulu", "Section 89"],
                "topic_hashtags": [
                    "#Section89",
                    "#CyrilRamaphosa",
                    "#Parliament",
                    "#ConstitutionalCourt",
                ],
                "human_review_flags": [],
            }
        )


def test_generates_localized_seo_without_mutating_translation():
    provider = _Provider()
    approved = {"segments": [{"translated_text": "Umbhalo ogunyaziwe"}]}

    seo = generate_localized_seo(
        provider=provider,
        job_id="jobid12345678",
        target_language="zu-ZA",
        english_transcript={"segments": [{"source_text": "Approved source"}]},
        approved_translation=approved,
        classification={"primary_domain": "politics"},
        context={"providers": []},
    )

    assert seo["language"] == "zu-ZA"
    assert seo["platform"] == "tiktok"
    assert seo["title"] == seo["cover_hook"] == "BUKA OKWENZEKE"
    assert seo["tiktok_caption"].startswith("President Ramaphosa")
    assert seo["caption_language"] == "en"
    assert seo["tiktok_hashtags"] == [
        "#Section89",
        "#CyrilRamaphosa",
        "#Parliament",
        "#ConstitutionalCourt",
    ]
    assert seo["source"] == "ai-localized-seo-from-approved-translation"
    assert seo["approved_translation_immutable"] is True
    assert approved == {"segments": [{"translated_text": "Umbhalo ogunyaziwe"}]}
    assert provider.payload["approved_target_transcript"] is approved
    assert provider.payload["publication_requirements"][
        "write_search_caption_in_english"
    ] is True


def test_deferred_seo_does_not_call_provider():
    from mathula_tv.localized_seo import build_deferred_localized_seo

    seo = build_deferred_localized_seo(
        job_id="jobid12345678",
        target_language="zu-ZA",
    )

    assert seo["status"] == "pending_editorial_call"
    assert seo["source"] == "deferred-until-rendered-blocks-exist"
    assert seo["ai_generation"] is None
    assert seo["approved_translation_immutable"] is True
