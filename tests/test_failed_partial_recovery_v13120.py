from __future__ import annotations

import json
from pathlib import Path

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.multivariant_translation import build_translation_request
from mathula_tv.one_call_translation import _recover_failed_candidate


def test_complete_compact_sse_text_is_recovered_without_provider_call(tmp_path: Path) -> None:
    request = build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "unit_0197",
                    "speaker": "SPEAKER_01",
                    "start": 0,
                    "end": 5,
                    "source_text": "The Madlanga Commission of Inquiry heard evidence.",
                    "protected_spans": ["Madlanga Commission of Inquiry"],
                }
            ],
        },
        target_locale="zu-ZA",
        job_id="job-partial-recovery",
    )
    chunk_dir = tmp_path / "chunk_0011"
    chunk_dir.mkdir()
    atomic_write_json(
        chunk_dir / "failure.json",
        {
            "schema_version": "mathula.translation.semantic-chunk-failure.v1",
            "request_sha256": request["request_sha256"],
            "error": {
                "stage": "claude_multivariant_translation_semantic_chunk",
                "error_type": "MultivariantTranslationError",
                "message": "legacy contextual identity rejection",
            },
        },
    )
    compact = {
        "units": [
            {
                "id": "unit_0197",
                "f": [],
                "o": [],
                "n": {"t": "Ubufakazi buzwakele.", "ms": 3000, "p": [], "o": [], "ok": True},
                "c": {"t": "Ubufakazi buzwakele.", "ms": 2500, "p": [], "o": [], "ok": True},
                "k": {"t": "Buzwakele.", "ms": 1900, "p": [], "o": [], "ok": True},
            }
        ],
        "terms": [],
        "warnings": [],
    }
    (chunk_dir / "translation_response.partial.txt").write_text(
        json.dumps(compact, ensure_ascii=False),
        encoding="utf-8",
    )

    recovered = _recover_failed_candidate(chunk_dir=chunk_dir, request=request)

    assert recovered is not None
    response, metadata, validation = recovered
    assert response["units"][0]["unit_id"] == "unit_0197"
    assert metadata["recovered_from_failed_candidate"] is True
    assert metadata["recovery_source"].endswith("translation_response.partial.txt")
    assert validation["contextual_identity_review_count"] >= 1
