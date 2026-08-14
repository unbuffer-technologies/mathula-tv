from __future__ import annotations

import jsonschema

from mathula_tv import tiktok_editor


def _candidate(index: int) -> dict[str, object]:
    return {
        "candidate_id": f"candidate_{index}",
        "selected_block_id": "block_0001",
        "evidence_block_ids": ["block_0001"],
        "editorial_angle": "contradiction",
        "hook_strategy": "named_actor_but_reveal",
        "withheld_answer": (
            "UBaloyi uthi lokhu kukhomba ukushushiswa okukhethayo—icala "
            "lezimali ze-secret service elithinta uBheki Cele yilo kuphela "
            "eliwela emsebenzini we-IDAC kodwa alaphenywa. " * 3
        ),
        "hook_text": "UBaloyi: kungani icala likaCele lingaphenywanga?",
        "accessible_hook_text": "UBaloyi ubuza ukuthi kungani icala likaCele lingaphenywanga",
        "title_lead": {
            "type": "person",
            "text": "Commissioner Baloyi who raised the central contradiction " * 4,
        },
        "rationale": "The candidate states the named actor and verified contradiction.",
        "confidence": 0.9,
        "grounded": True,
        "preserves_attribution": True,
        "no_sensationalism": True,
        "scores": {
            "visual_impact": 8,
            "curiosity": 8,
            "specificity": 8,
            "stakes": 8,
            "immediacy": 8,
            "audience_relevance": 8,
        },
        "human_review_flags": [],
    }


def test_provider_schema_accepts_verbose_internal_editorial_text() -> None:
    payload = {
        "opening_boundary_block_id": "block_0001",
        "opening_payoff_block_id": "block_0001",
        "opening_relationship": "standalone",
        "opening_boundary_rationale": "The first block states the verified issue.",
        "candidates": [_candidate(1), _candidate(2), _candidate(3)],
        "primary_story_angle": "Selective investigation allegation",
        "caption": "UBaloyi raises questions about which allegations were investigated.",
        "featured_person_names": ["Bheki Cele", "Andrea Johnson"],
        "search_keywords": ["Baloyi", "Bheki Cele", "IDAC"],
        "topic_hashtags": [
            "#BhekiCele",
            "#IDAC",
            "#MadlangaCommission",
            "#SAPS",
        ],
        "human_review_flags": [],
    }

    jsonschema.validate(payload, tiktok_editor.EDITORIAL_PACKAGE_SCHEMA)


def test_internal_editorial_text_is_normalized_locally() -> None:
    value = "  UBaloyi   uthi lokhu   kukhomba  " + ("ukushushiswa " * 80)
    normalized = tiktok_editor._normalize_editorial_metadata_text(
        value, max_characters=160
    )

    assert len(normalized) <= 160
    assert "  " not in normalized
    assert normalized.endswith("…")
