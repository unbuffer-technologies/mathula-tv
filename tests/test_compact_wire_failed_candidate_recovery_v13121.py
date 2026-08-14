from pathlib import Path

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.multivariant_translation import build_translation_request
from mathula_tv.one_call_translation import _recover_failed_candidate


def test_failed_provider_schema_candidate_with_nested_warning_recovers_locally(tmp_path: Path) -> None:
    request = build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "unit_0310",
                    "speaker": "SPEAKER_01",
                    "start": 0,
                    "end": 5,
                    "source_text": "The witness answered the question.",
                    "protected_spans": [],
                }
            ],
        },
        target_locale="zu-ZA",
        job_id="job-nested-warning-recovery",
    )
    compact = {
        "units": [
            {
                "id": "unit_0310",
                "f": [],
                "o": [],
                "n": {
                    "t": "Ufakazi uwuphendulile umbuzo.",
                    "ms": 2600,
                    "p": [],
                    "o": [],
                    "ok": True,
                    "warnings": ["Natural wording retained for fluency"],
                },
                "c": {"t": "Ufakazi uphendulile.", "ms": 2100, "p": [], "o": [], "ok": True},
                "k": {"t": "Uphendulile.", "ms": 1600, "p": [], "o": [], "ok": True},
            }
        ],
        "terms": [],
        "warnings": [],
    }
    chunk_dir = tmp_path / "chunk_0017"
    chunk_dir.mkdir()
    atomic_write_json(
        chunk_dir / "failure.json",
        {
            "schema_version": "mathula.translation.semantic-chunk-failure.v1",
            "request_sha256": request["request_sha256"],
            "error": {
                "details": {
                    "provider": "azure-foundry-claude",
                    "candidate_output": compact,
                    "provider_attempts": 1,
                }
            },
        },
    )

    recovered = _recover_failed_candidate(chunk_dir=chunk_dir, request=request)

    assert recovered is not None
    response, metadata, _validation = recovered
    assert metadata["recovered_from_failed_candidate"] is True
    assert response["warnings"] == [
        "unit_0310/natural: Natural wording retained for fluency"
    ]
