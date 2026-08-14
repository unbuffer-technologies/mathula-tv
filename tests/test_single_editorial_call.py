from __future__ import annotations

from types import SimpleNamespace

from mathula_tv import tiktok_editor
from mathula_tv.ai_provider import AZURE_FOUNDRY_CLAUDE_PROVIDER


class _Provider:
    provider = AZURE_FOUNDRY_CLAUDE_PROVIDER
    config = SimpleNamespace(
        editorial_effort="high",
        editorial_thinking_type="adaptive",
        editorial_max_output_tokens=32_000,
    )

    def __init__(self) -> None:
        self.requests = []

    def complete_structured(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            data={
                "primary_story_angle": "Johnson resignation and accountability",
                "caption": "U-Andrea Johnson usesulile ngesikhathi kusalindwe izimpendulo.",
                "search_keywords": ["Andrea Johnson", "IDAC", "NPA"],
                "topic_hashtags": ["#AndreaJohnson", "#IDAC"],
                "human_review_flags": [],
                "opening_boundary_block_id": "block_0001",
                "opening_payoff_block_id": "block_0002",
                "opening_relationship": "question_answer",
                "opening_boundary_rationale": "The question establishes the payoff.",
                "candidates": [
                    {
                        "candidate_id": f"candidate_{index}",
                        "selected_block_id": "block_0002",
                        "hook_text": text,
                        "accessible_hook_text": (
                            "UJohnson ne-NPA phakathi kwengxabano"
                            if "IDAC" in text
                            else text
                        ),
                        "title_lead": (
                            {"type": "person", "text": "Johnson"}
                            if text.startswith("UJohnson:")
                            else {"type": "topic", "text": ""}
                        ),
                        "rationale": "Grounded in the response.",
                        "confidence": 0.95 - index * 0.05,
                        "grounded": True,
                        "preserves_attribution": True,
                        "no_sensationalism": True,
                        "scores": {
                            "visual_impact": 8 - index,
                            "curiosity": 9 - index,
                            "specificity": 9 - index,
                            "stakes": 8 - index,
                            "immediacy": 8 - index,
                            "audience_relevance": 9 - index,
                        },
                        "human_review_flags": [],
                    }
                    for index, text in enumerate(
                        (
                            "UJohnson: Kungani esakhuluma ngokuhlaselwa?",
                            "Ukusula kukaJohnson kuvusa imibuzo",
                            "I-IDAC ne-NPA phakathi kwengxabano",
                        )
                    )
                ],
            },
            metadata=SimpleNamespace(to_dict=lambda: {"provider": "fake"}),
        )


def _blocks():
    return [
        {
            "block_id": "block_0001",
            "speaker_id": "HOST",
            "start_ms": 0,
            "end_ms": 8000,
            "source_text": "Are these the consequences of serious allegations?",
            "translated_text": "Ingabe lena yimiphumela yezinsolo ezinzima?",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "GUEST",
            "start_ms": 8500,
            "end_ms": 22000,
            "source_text": "Thank you for that question. We welcome the resignation.",
            "translated_text": "Ngiyabonga ngalowo mbuzo. Siyakwamukela ukusula.",
        },
    ]


def test_one_high_thinking_editorial_call_generates_seo_and_hook() -> None:
    provider = _Provider()
    result = tiktok_editor.select_tiktok_hook(
        provider=provider,
        job_id="editorial-job",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={"status": "pending_editorial_call"},
        english_transcript={"segments": [{"source_text": "Full source"}]},
        classification={"primary_domain": "politics"},
        context={"entities": ["IDAC", "NPA"]},
        source_duration_seconds=90.0,
    )

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.operation == "editorial_package"
    assert request.effort == "high"
    assert request.thinking_type == "adaptive"
    assert request.max_output_tokens == 32_000
    assert request.max_repairs == 0
    assert request.stream_response is True
    assert request.payload["complete_english_transcript"]["segments"]
    assert request.payload["requirements"]["do_not_translate_or_rewrite_approved_speech"] is True
    assert request.payload["requirements"]["truth_constrained_tabloid_style"] is True
    assert request.payload["requirements"]["allow_aggressive_truthful_baiting"] is True
    assert request.payload["requirements"]["no_deceptive_clickbait"] is True
    assert request.payload["requirements"]["prefer_named_actor_claim_but_reveal_formula"] is True
    assert request.payload["requirements"]["reveal_subject_withhold_explanation"] is True
    assert request.payload["requirements"]["reject_whole_subject_withheld_clickbait"] is True
    assert request.payload["requirements"]["prefer_title_evidence_near_opening"] is True
    assert request.payload["requirements"]["target_title_payoff_within_first_35_seconds"] is True
    assert request.payload["requirements"]["write_search_caption_in_english"] is True
    assert request.payload["requirements"]["exact_story_hashtags"] == 4
    assert request.payload["requirements"]["configured_community_hashtag_added_later"] is True
    assert request.payload["requirements"]["do_not_generate_brand_language_or_geographic_padding_tags"] is True

    assert result["editorial_call_count"] == 1
    assert result["automatic_model_repairs"] == 0
    assert result["translation_operation_performed"] is False
    assert result["editorial_seo"]["source"] == "single-high-thinking-editorial-call"
    assert result["editorial_seo"]["cover_hook"] == result["hook_text"]
    assert result["opening_anchor_block_id"] == "block_0001"
    assert result["opening_payoff_block_id"] == "block_0002"


def test_paid_editorial_response_checkpoint_is_reused_without_provider_call(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "editorial_response_checkpoint.json"
    provider = _Provider()
    first = tiktok_editor.select_tiktok_hook(
        provider=provider,
        job_id="editorial-checkpoint-job",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={"status": "pending_editorial_call"},
        english_transcript={"segments": [{"source_text": "Full source"}]},
        classification={"primary_domain": "politics"},
        context={"entities": ["IDAC", "NPA"]},
        source_duration_seconds=90.0,
        response_checkpoint_path=checkpoint_path,
    )

    class ProviderMustNotRun(_Provider):
        def complete_structured(self, request):
            raise AssertionError("saved paid response should have been reused")

    second = tiktok_editor.select_tiktok_hook(
        provider=ProviderMustNotRun(),
        job_id="editorial-checkpoint-job",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={"status": "pending_editorial_call"},
        english_transcript={"segments": [{"source_text": "Full source"}]},
        classification={"primary_domain": "politics"},
        context={"entities": ["IDAC", "NPA"]},
        source_duration_seconds=90.0,
        response_checkpoint_path=checkpoint_path,
    )

    assert len(provider.requests) == 1
    assert checkpoint_path.is_file()
    assert first["hook_text"] == second["hook_text"]
    assert first["editorial_response_checkpoint"]["reused"] is False
    assert second["editorial_response_checkpoint"]["reused"] is True


def test_edit_job_writes_combined_seo_without_touching_translation(
    tmp_path, monkeypatch
) -> None:
    import json
    from pathlib import Path

    job_id = "job-single-editorial"
    root = tmp_path / "jobs" / job_id
    translation_path = root / "translation" / "transcript_zu.json"
    english_path = root / "analysis" / "transcript_en.json"
    classification_path = root / "analysis" / "domain_classification.json"
    context_path = root / "analysis" / "context.json"
    report_path = root / "direct_dub" / "report.json"
    master_path = root / "direct_dub" / "dubbed_master.mp4"
    for path in (
        translation_path,
        english_path,
        classification_path,
        context_path,
        report_path,
        master_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    translation_bytes = json.dumps(
        {"target_language": "zu-ZA", "units": []}, ensure_ascii=False
    ).encode("utf-8")
    translation_path.write_bytes(translation_bytes)
    english_path.write_text(
        json.dumps({"segments": [{"source_text": "Complete English context"}]}),
        encoding="utf-8",
    )
    classification_path.write_text(
        json.dumps({"primary_domain": "politics"}), encoding="utf-8"
    )
    context_path.write_text(
        json.dumps({"entities": ["IDAC", "NPA"]}), encoding="utf-8"
    )
    master_path.write_bytes(b"clean-master")
    report_path.write_text(
        json.dumps(
            {
                "blocks": _blocks(),
                "outputs": {"canonical_dubbed_master": str(master_path)},
            }
        ),
        encoding="utf-8",
    )

    provider = _Provider()
    monkeypatch.setattr(tiktok_editor, "probe", lambda _path: {"duration": 90.0})

    def fake_render(**kwargs):
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(b"publication")
        result = {
            **dict(kwargs["selection"]),
            "output": {"path": str(kwargs["output_path"])},
            "translation": {"immutable": True},
            "title_panel": {"text": kwargs["title_text"]},
            "hook_edit_enabled": kwargs["apply_opening_cut"],
            "hook_card_enabled": True,
        }
        kwargs["manifest_path"].write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(tiktok_editor, "render_tiktok_hook_edit", fake_render)
    monkeypatch.setattr(
        tiktok_editor, "_record_publication_artifacts", lambda **_kwargs: None
    )

    job = SimpleNamespace(job_id=job_id, target_language="zu-ZA", media={})
    result = tiktok_editor.edit_tiktok_job(
        work_dir=tmp_path,
        job=job,
        provider=provider,
    )

    assert len(provider.requests) == 1
    assert translation_path.read_bytes() == translation_bytes
    seo = json.loads(
        (root / "translation" / "tiktok_zu.json").read_text(encoding="utf-8")
    )
    assert seo["source"] == "single-high-thinking-editorial-call"
    assert seo["primary_story_angle"] == "Johnson resignation and accountability"
    assert seo["cover_hook"] == result["hook_text"]
    assert result["translation_operation_performed"] is False



def test_obscure_title_acronym_uses_accessible_named_speaker_fallback() -> None:
    class PisaProvider(_Provider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["candidates"][0]["hook_text"] = (
                "PISA: uJohnson uzenza isisulu esikhundleni sokuziphendulela"
            )
            response.data["candidates"][0]["accessible_hook_text"] = (
                "UKass: UJohnson uzenza isisulu esikhundleni sokuziphendulela"
            )
            response.data["candidates"][0]["scores"] = {
                "visual_impact": 10,
                "curiosity": 10,
                "specificity": 10,
                "stakes": 10,
                "immediacy": 10,
                "audience_relevance": 10,
            }
            return response

    result = tiktok_editor.select_tiktok_hook(
        provider=PisaProvider(),
        job_id="editorial-job",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={"status": "pending_editorial_call"},
        english_transcript={"segments": [{"source_text": "Full source"}]},
        classification={"primary_domain": "politics"},
        context={"entities": ["Public Interest South Africa"]},
        source_duration_seconds=90.0,
    )

    assert result["hook_text"].startswith("UKass:")
    assert result["editorial_seo"]["cover_hook"] == result["hook_text"]
    assert result["title_acronym_policy"]["fallback_applied"] is True
    assert result["title_acronym_policy"]["obscure_acronyms"] == ["PISA"]
    assert "obscure_title_acronym_replaced:PISA" in result["editorial_seo"][
        "human_review_flags"
    ]


def test_obscure_title_candidate_is_skipped_when_safe_candidates_exist() -> None:
    class UnsafePisaProvider(_Provider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["candidates"][0]["hook_text"] = "XYZQ: UJohnson uyaziphendulela"
            response.data["candidates"][0]["accessible_hook_text"] = (
                "XYZQ: UJohnson uyaziphendulela"
            )
            response.data["candidates"][0]["title_lead"] = {
                "type": "organisation",
                "text": "XYZQ",
            }
            return response

    result = tiktok_editor.select_tiktok_hook(
        provider=UnsafePisaProvider(),
        job_id="editorial-job",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={"status": "pending_editorial_call"},
        english_transcript={"segments": [{"source_text": "Full source"}]},
        classification={"primary_domain": "politics"},
        context={"entities": ["Public Interest South Africa"]},
        source_duration_seconds=90.0,
    )

    assert "XYZQ" not in result["hook_text"]
    assert any(
        item.get("reason")
        == "accessible_candidate_without_unexplained_acronyms_available"
        for item in result["engagement_ranking"]["rejected_candidates"]
    )


def test_bare_person_attribution_is_prefixed_for_isizulu_title() -> None:
    class BareKassProvider(_Provider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            candidate = response.data["candidates"][0]
            candidate["hook_text"] = (
                "Kass: ukwesula akumkhululi uJohnson eKhomishini kaMadlanga nasenkantolo"
            )
            candidate["accessible_hook_text"] = candidate["hook_text"]
            candidate["title_lead"] = {"type": "person", "text": "Kass"}
            candidate["scores"] = {
                "visual_impact": 10,
                "curiosity": 10,
                "specificity": 10,
                "stakes": 10,
                "immediacy": 10,
                "audience_relevance": 10,
            }
            return response

    result = tiktok_editor.select_tiktok_hook(
        provider=BareKassProvider(),
        job_id="kass-title-job",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={"status": "pending_editorial_call"},
        english_transcript={"segments": [{"source_text": "Full source"}]},
        classification={"primary_domain": "politics"},
        context={"entities": ["Deboho Kass"]},
        source_duration_seconds=90.0,
    )

    assert result["hook_text"] == (
        "UKass: ukwesula akumkhululi uJohnson eKhomishini kaMadlanga nasenkantolo"
    )
    assert result["editorial_seo"]["cover_hook"] == result["hook_text"]
    assert result["title_person_prefix_policy"]["prefix_applied"] is True
    assert "zulu_person_attribution_prefix_applied" in result["editorial_seo"][
        "human_review_flags"
    ]


def test_organisation_title_lead_is_not_given_person_prefix() -> None:
    title, policy = tiktok_editor._normalize_zulu_person_attribution_title(
        "Eskom: kubuyela ukucinywa kukagesi",
        title_lead={"type": "organisation", "text": "Eskom"},
        target_language="zu-ZA",
    )

    assert title == "Eskom: kubuyela ukucinywa kukagesi"
    assert policy["prefix_applied"] is False


def test_provider_schema_defers_hook_length_to_local_card_gate() -> None:
    hook_schema = tiktok_editor.EDITORIAL_PACKAGE_SCHEMA["properties"]["candidates"]["items"]["properties"]
    assert "maxLength" not in hook_schema["hook_text"]
    assert "maxLength" not in hook_schema["accessible_hook_text"]
    assert hook_schema["evidence_block_ids"]["maxItems"] == 4
    assert "contradiction" in hook_schema["editorial_angle"]["enum"]
    assert "named_actor_but_reveal" in hook_schema["hook_strategy"]["enum"]
    assert "maxLength" not in hook_schema["withheld_answer"]
    assert "maxLength" not in hook_schema["title_lead"]["properties"]["text"]
    assert tiktok_editor.TIKTOK_HOOK_PROMPT_VERSION == (
        "mathula-editorial-package-v12-featured-people-discovery"
    )


def test_overlong_accessible_person_hook_is_compacted_locally_without_second_call() -> None:
    class LongAccessibleProvider(_Provider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            candidate = response.data["candidates"][0]
            candidate["hook_text"] = "IDAC: UJohnson usamele aphendule izinsolo"
            candidate["accessible_hook_text"] = (
                "U-Advocate Andrea Johnson uyahamba kodwa usamele aphendule "
                "izinsolo zophenyo lweCrime Intelligence"
            )
            candidate["title_lead"] = {
                "type": "person",
                "text": "Andrea Johnson",
            }
            candidate["scores"] = {
                "visual_impact": 10,
                "curiosity": 10,
                "specificity": 10,
                "stakes": 10,
                "immediacy": 10,
                "audience_relevance": 10,
            }
            return response

    provider = LongAccessibleProvider()
    result = tiktok_editor.select_tiktok_hook(
        provider=provider,
        job_id="long-accessible-title-job",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={"status": "pending_editorial_call"},
        english_transcript={"segments": [{"source_text": "Full source"}]},
        classification={"primary_domain": "politics"},
        context={"entities": ["Andrea Johnson", "IDAC", "Crime Intelligence"]},
        source_duration_seconds=90.0,
    )

    assert len(provider.requests) == 1
    assert result["hook_text"] == (
        "UJohnson: uyahamba kodwa usamele aphendule izinsolo zophenyo lweCrime Intelligence"
    )
    assert len(result["hook_text"]) <= 90
    assert result["title_length_policy"]["adjusted"] is True
    assert result["title_length_policy"]["strategy"] == "compact_person_lead"
    assert result["editorial_seo"]["cover_hook"] == result["hook_text"]
    assert "title_length_adjusted_locally:compact_person_lead" in result["editorial_seo"][
        "human_review_flags"
    ]


def test_irreducibly_long_hook_is_clipped_locally_at_word_boundary() -> None:
    text = (
        "Uphenyo lweCrime Intelligence lusaqhubeka futhi kusalindwe izimpendulo "
        "eziningi ezisemqoka emphakathini waseNingizimu Afrika"
    )
    fitted, policy = tiktok_editor._fit_title_card_hook_text(
        text,
        title_lead={"type": "topic", "text": ""},
        target_language="zu-ZA",
    )

    assert len(fitted) <= 90
    assert policy["adjusted"] is True
    assert policy["strategy"] == "word_boundary_clip"
    assert not fitted.endswith((" ", ":", ";", ",", "-"))


def test_idac_in_both_hooks_is_rewritten_to_grounded_person_without_second_call() -> None:
    class RepeatedIdacProvider(_Provider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            candidate = response.data["candidates"][0]
            candidate["hook_text"] = (
                "IDAC: UJohnson usamele aphendule izinsolo zophenyo lweCrime Intelligence"
            )
            candidate["accessible_hook_text"] = (
                "I-IDAC: UJohnson usamele aphendule izinsolo zophenyo lweCrime Intelligence"
            )
            candidate["title_lead"] = {
                "type": "person",
                "text": "Andrea Johnson",
            }
            candidate["scores"] = {
                "visual_impact": 10,
                "curiosity": 10,
                "specificity": 10,
                "stakes": 10,
                "immediacy": 10,
                "audience_relevance": 10,
            }
            return response

    provider = RepeatedIdacProvider()
    result = tiktok_editor.select_tiktok_hook(
        provider=provider,
        job_id="idac-double-fallback-job",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={"status": "pending_editorial_call"},
        english_transcript={"segments": [{"source_text": "Full source"}]},
        classification={"primary_domain": "politics"},
        context={"entities": ["Andrea Johnson", "IDAC", "Crime Intelligence"]},
        source_duration_seconds=90.0,
    )

    assert len(provider.requests) == 1
    assert result["hook_text"] == (
        "UJohnson: usamele aphendule izinsolo zophenyo lweCrime Intelligence"
    )
    policy = result["title_acronym_policy"]
    assert policy["fallback_applied"] is True
    assert policy["deterministic_rewrite"]["applied"] is True
    assert policy["deterministic_rewrite"]["strategy"] == "grounded_person_lead"
    assert policy["deterministic_rewrite"]["rewritten_acronyms"] == ["IDAC"]


def test_idac_is_expanded_when_no_grounded_person_lead_exists() -> None:
    title, policy = tiktok_editor._choose_accessible_hook_text(
        {
            "hook_text": "IDAC: kuphakama imibuzo ngobuqotho",
            "accessible_hook_text": "I-IDAC: kuphakama imibuzo ngobuqotho",
            "title_lead": {"type": "organisation", "text": "IDAC"},
        },
        target_language="zu-ZA",
    )

    assert title.startswith("Investigating Directorate Against Corruption:")
    assert not tiktok_editor._obscure_title_acronyms(title)
    assert policy["deterministic_rewrite"]["strategy"] == (
        "reviewed_full_name_expansion"
    )
