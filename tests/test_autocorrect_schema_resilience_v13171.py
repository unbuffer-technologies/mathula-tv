from __future__ import annotations

from types import SimpleNamespace

import jsonschema

from mathula_tv.autocorrect_research import _OUTPUT_SCHEMA
from mathula_tv.azure_web_research import (
    _normalise_research_output,
    _structured_text_format,
)
from mathula_tv.cli import _reset_for_autocorrect_research, parser


def test_schema_repair_requires_every_property_at_every_object_depth() -> None:
    schema = _structured_text_format(_OUTPUT_SCHEMA)["format"]["schema"]

    assert schema["required"] == list(schema["properties"])
    correction = schema["properties"]["corrections"]["items"]
    assert correction["required"] == list(correction["properties"])
    assert "canonical_text" in correction["required"]
    unresolved = schema["properties"]["unresolved_candidates"]["items"]
    assert unresolved["required"] == list(unresolved["properties"])
    assert schema["additionalProperties"] is False
    assert correction["additionalProperties"] is False
    assert unresolved["additionalProperties"] is False


def test_bare_entity_id_correction_is_preserved_as_unresolved() -> None:
    entity_id = "person_reporter_ayanda_nyatee"
    normalized, _dropped, canonicalized = _normalise_research_output(
        {
            "schema_version": "wrong-version",
            "corrections": [entity_id],
            "unresolved_candidates": [],
        },
        _OUTPUT_SCHEMA,
        {
            "candidate_entities": [
                {
                    "entity_id": entity_id,
                    "representative_text": "Ayanda Nyatee",
                    "aliases": ["Ayanda Nyatee"],
                }
            ]
        },
    )

    jsonschema.Draft202012Validator(_OUTPUT_SCHEMA).validate(normalized)
    assert normalized["corrections"] == []
    assert normalized["unresolved_candidates"] == [
        {
            "entity_id": entity_id,
            "reason": (
                "research_model_returned_entity_id_without_supported_correction"
            ),
        }
    ]
    assert "scalar_correction_moved_to_unresolved" in canonicalized
    assert "schema_version" in canonicalized


def test_resume_command_resets_only_post_autocorrect_stages() -> None:
    job = SimpleNamespace(
        state="translation_running",
        completed_stages=[
            "azure_transcription",
            "autocorrection",
            "classification",
            "entity_binding",
            "translation",
        ],
        review_readiness="translation_running",
        last_error={"stage": "translation"},
        attempt_history=[],
    )

    previous = _reset_for_autocorrect_research(job)

    assert previous == "translation_running"
    assert job.state == "analysis_running"
    assert job.completed_stages == [
        "azure_transcription",
        "autocorrection",
    ]
    assert job.review_readiness == "autocorrection_research_rerun"
    assert job.last_error is None
    assert job.attempt_history[-1]["from_state"] == "translation_running"


def test_resume_autocorrect_research_is_an_explicit_live_command() -> None:
    args = parser().parse_args(
        [
            "resume-autocorrect-research",
            "job-123",
            "--live-operation",
        ]
    )

    assert args.command == "resume-autocorrect-research"
    assert args.job_id == "job-123"
    assert args.live_operation is True
