from __future__ import annotations

import json

from mathula_tv.ai_consumption import (
    consumption_record,
    update_ai_consumption_report,
)
from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    AnthropicConfig,
)
from mathula_tv.config import load_settings
from mathula_tv.multivariant_translation import PROMPT_VERSION
from mathula_tv.one_call_translation import (
    _prior_context,
    _prior_context_legacy,
    _prior_terminology,
    _prior_terminology_legacy,
    _resumable_session_manifest,
)
from mathula_tv.semantic_translation_chunks import (
    adaptive_context_spine_tokens,
)


def _metadata(*, status: str = "completed") -> dict:
    return {
        "provider": "azure-foundry-claude",
        "model_returned": "claude-opus-4-8",
        "prompt_hash": "prompt-hash",
        "request_sha256": "request-hash",
        "request_timestamp": "2026-07-30T00:00:00+00:00",
        "token_usage": {
            "input_tokens": 1_000,
            "output_tokens": 200 if status == "completed" else 100,
            "cache_read_input_tokens": 300,
            "cache_creation_input_tokens": 50,
        },
        "request_summary": {
            "wire_requests": [
                {
                    "request_body_bytes": 8_000,
                    "estimated_input_tokens": 2_000,
                    "output_schema_bytes": 1_000,
                    "max_output_tokens": 8_000,
                }
            ]
        },
        # A report must not accidentally copy arbitrary provider fields.
        "raw_output": "private transcript and model output",
    }


def test_consumption_report_is_content_free_and_idempotently_replaces_retry(tmp_path):
    job_root = tmp_path / "job-123"
    failed = consumption_record(
        operation="multivariant_translation",
        metadata=_metadata(status="failed"),
        scope="chunk_0001",
        status="failed",
    )
    update_ai_consumption_report(
        job_root=job_root,
        records=[failed],
        video_duration_seconds=120,
    )

    completed = consumption_record(
        operation="multivariant_translation",
        metadata=_metadata(),
        scope="chunk_0001",
    )
    report = update_ai_consumption_report(
        job_root=job_root,
        records=[completed],
        video_duration_seconds=120,
    )

    assert len(report["records"]) == 1
    assert report["records"][0]["status"] == "completed"
    assert report["summary"]["input_tokens"] == 1_000
    assert report["summary"]["output_tokens"] == 200
    assert report["summary"]["tokens_per_video_minute"] == 600.0
    encoded = json.dumps(report)
    assert "private transcript" not in encoded
    assert "raw_output" not in encoded


def test_context_spine_budget_scales_with_transcript_complexity():
    short = {
        "units": [
            {"unit_id": f"unit_{index}", "source_text": "Short statement."}
            for index in range(12)
        ]
    }
    long = {
        "global_protected_spans": ["IDAC", "NPA", "Andrea Johnson"],
        "units": [
            {
                "unit_id": f"unit_{index}",
                "source_text": "Detailed commission testimony.",
                "protected_spans": ["IDAC"],
            }
            for index in range(500)
        ],
    }

    assert adaptive_context_spine_tokens(short) == 1_600
    assert adaptive_context_spine_tokens(short) < adaptive_context_spine_tokens(long)
    assert adaptive_context_spine_tokens(long) <= 4_500


def test_prior_continuity_omits_duplicate_variants_and_irrelevant_terms():
    responses = [
        {
            "units": [
                {
                    "unit_id": "unit_0001",
                    "speaker_id": "SPEAKER_00",
                    "source_text": "The source is already in adjacent context.",
                    "variants": [
                        {"variant_id": "natural", "spoken_text": "Natural version"},
                        {"variant_id": "concise", "spoken_text": "Concise version"},
                        {"variant_id": "compact", "spoken_text": "Compact version"},
                    ],
                }
            ],
            "global_terminology": [
                {
                    "source_term": "IDAC",
                    "target_rendering": "i-IDAC",
                    "preserve_english": True,
                    "reason": "acronym",
                },
                {
                    "source_term": "Unrelated Entity",
                    "target_rendering": "Unrelated Entity",
                    "preserve_english": True,
                    "reason": "name",
                },
            ],
        }
    ]
    current = {
        "units": [
            {
                "source_text": "IDAC continued its investigation.",
                "protected_spans": ["IDAC"],
            }
        ],
        "translation_batch": {"context_units": []},
    }

    assert _prior_context(responses) == [
        {
            "unit_id": "unit_0001",
            "speaker_id": "SPEAKER_00",
            "natural_translation": "Natural version",
        }
    ]
    assert [item["source_term"] for item in _prior_terminology(responses, current)] == [
        "IDAC"
    ]
    assert _prior_context_legacy(responses)[0]["concise_translation"] == "Concise version"
    assert len(_prior_terminology_legacy(responses)) == 2


def test_incomplete_v13165_session_is_discovered_for_checkpoint_compatible_resume(
    tmp_path,
):
    session_root = tmp_path / "translation" / "cloud_chunks" / "old-session"
    session_root.mkdir(parents=True)
    manifest = {
        "full_request_sha256": "full-request",
        "prompt_version": PROMPT_VERSION,
        "context_spine_tokens": 4_500,
        # v13.16.5 intentionally has no continuity_payload_policy field.
    }
    (session_root / "session_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    discovered = _resumable_session_manifest(
        job_root=tmp_path,
        request_sha256="full-request",
    )

    assert discovered == manifest


def test_editorial_default_is_reduced_but_explicit_override_still_wins(
    monkeypatch, tmp_path
):
    environment = {
        "MATHULA_TV_AI_PROVIDER": AZURE_FOUNDRY_CLAUDE_PROVIDER,
        "MATHULA_TV_FOUNDRY_CLAUDE_API_KEY": "test-key",
        "MATHULA_TV_FOUNDRY_CLAUDE_ENDPOINT": (
            "https://mathula-test.services.ai.azure.com/anthropic"
        ),
        "MATHULA_TV_FOUNDRY_CLAUDE_DEPLOYMENT": "claude-opus-4-8",
    }
    assert AnthropicConfig.from_environment(environment).editorial_max_output_tokens == 12_000

    environment["MATHULA_TV_EDITORIAL_MAX_OUTPUT_TOKENS"] = "24000"
    assert AnthropicConfig.from_environment(environment).editorial_max_output_tokens == 24_000

    monkeypatch.setenv("MATHULA_TV_WORK_DIR", str(tmp_path / "working"))
    monkeypatch.delenv("MATHULA_TV_EDITORIAL_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv(
        "MATHULA_TV_EDITORIAL_CLAUDE_MAX_OUTPUT_TOKENS", raising=False
    )
    assert load_settings(tmp_path).editorial_max_output_tokens == 12_000
