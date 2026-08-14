from __future__ import annotations

from types import SimpleNamespace

import pytest

from mathula_tv import tiktok_editor


class _Provider:
    def complete_structured(self, request):
        scores = {
            "visual_impact": 8,
            "curiosity": 8,
            "specificity": 8,
            "stakes": 8,
            "immediacy": 8,
            "audience_relevance": 8,
        }
        return SimpleNamespace(
            data={
                "opening_boundary_block_id": "block_0002",
                "opening_payoff_block_id": "block_0002",
                "opening_relationship": "standalone",
                "opening_boundary_rationale": "The answer contains the strongest hook.",
                "candidates": [
                    {
                        "candidate_id": f"candidate_{index}",
                        "selected_block_id": "block_0002",
                        "hook_text": f"UJohnson uphendula ngokuhlaselwa {index}",
                        "rationale": "Grounded in the response.",
                        "confidence": 0.9,
                        "grounded": True,
                        "preserves_attribution": True,
                        "no_sensationalism": True,
                        "scores": scores,
                        "human_review_flags": [],
                    }
                    for index in range(1, 4)
                ],
            },
            metadata=SimpleNamespace(to_dict=lambda: {"provider": "fake"}),
        )


def test_real_johnson_response_is_moved_back_to_question() -> None:
    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "INTERVIEWER",
            "start_ms": 960,
            "end_ms": 27900,
            "source_text": (
                "Johnson says the attacks against her have taken a personal and "
                "professional toll. Are these the inevitable consequences of "
                "serious allegations against a public office bearer?"
            ),
            "translated_text": "Ingabe lena imiphumela engenakugwenywa?",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "GUEST",
            "start_ms": 29520,
            "end_ms": 140000,
            "source_text": (
                "Thank you for that question, Lopam. Before I address it directly, "
                "let me say we welcome the decision by Advocate Johnson."
            ),
            "translated_text": (
                "Ngiyabonga ngalowo mbuzo, Lopam. Ngaphambi kokuwuphendula ngqo, "
                "ake ngithi siyasamukela isinqumo sika-Advocate Johnson."
            ),
        },
    ]

    result = tiktok_editor.select_tiktok_hook(
        provider=_Provider(),
        job_id="17d8715b86ce4eb7977f91af455c18d5",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=396.0,
    )

    assert result["ai_requested_opening_block_id"] == "block_0002"
    assert result["opening_anchor_block_id"] == "block_0001"
    assert result["opening_payoff_block_id"] == "block_0002"
    assert result["opening_relationship"] == "question_answer"
    assert result["cut_start_seconds"] == pytest.approx(0.96)
    assert result["question_anchor_guard"]["guard_applied"] is True
    assert result["question_anchor_guard"]["response_cue_detected"] is True


def test_legacy_manifest_is_not_reused() -> None:
    assert not tiktok_editor._is_current_question_anchor_selection(
        {
            "selection_prompt_version": tiktok_editor.TIKTOK_HOOK_PROMPT_VERSION,
            "hook_text": "Legacy hook",
            "selected_candidate": {"block_id": "block_0002"},
        }
    )
