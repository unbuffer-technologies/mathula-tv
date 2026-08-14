from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathula_tv.contextual_speaker_identity import (
    CONTEXTUAL_SPEAKER_POLICY_VERSION,
    resolve_contextual_speaker_identities,
)
from mathula_tv.direct_azure_dub import (
    DirectDubOptions,
    SpeechBlock,
    build_source_word_speaker_anchors,
    enforce_source_word_speaker_alignment,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _segment(
    segment_id: str,
    speaker_id: str,
    start: float,
    end: float,
    text: str,
    word_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "speaker_id": speaker_id,
        "speaker": speaker_id,
        "start": start,
        "end": end,
        "source_text": text,
        "source_word_ids": word_ids or [],
    }


def _registry(*names: str) -> dict[str, object]:
    return {
        "schema_version": "mathula-contextual-speaker-identity-registry-v1",
        "identities": {
            name.casefold(): {
                "canonical_name": name,
                "gender": "male",
                "confidence": 0.99,
                "evidence_count": 1,
                "evidence": [{"job_id": "audited-earlier-job"}],
            }
            for name in names
        },
    }


def _resolve_reporter_attribution(
    tmp_path: Path,
    attribution_text: str,
    *registry_names: str,
) -> tuple[dict[str, object], dict[str, object]]:
    registry_path = tmp_path / "contextual_speaker_identities" / "registry.json"
    output_path = tmp_path / "contextual_speaker_identities.json"
    _write_json(registry_path, _registry(*registry_names))
    result = resolve_contextual_speaker_identities(
        job_id="7f4d9024e2444cc385b913048494a3c9",
        transcript={
            "segments": [
                _segment("seg-14", "SPEAKER_00", 99.47, 118.91, attribution_text),
                _segment(
                    "seg-15",
                    "SPEAKER_01",
                    120.07,
                    149.55,
                    (
                        "Our work is not done. Our efforts will continue. I meet "
                        "with General Dimpane tomorrow, as well as General Mkhwanazi."
                    ),
                ),
                _segment(
                    "seg-21",
                    "SPEAKER_00",
                    150.51,
                    167.55,
                    "Investigators continue to work to establish what happened.",
                ),
            ]
        },
        name_research={"accepted_corrections": [], "rejected_corrections": []},
        required_speaker_ids=["SPEAKER_01"],
        output_path=output_path,
        registry_path=registry_path,
    )
    return result, json.loads(output_path.read_text(encoding="utf-8"))


def test_exact_katalia_says_attribution_rebinds_audited_firoz_cachalia(
    tmp_path: Path,
) -> None:
    result, audit = _resolve_reporter_attribution(
        tmp_path,
        (
            "Katalia says police efforts in gang-infested areas will be bolstered "
            "by the national task team led by Lieutenant-General Mkhwanazi."
        ),
        "Firoz Cachalia",
    )

    speaker = result["SPEAKER_01"]
    assert speaker["identified_name"] == "Firoz Cachalia"
    assert speaker["identity_source"] == (
        "audited_registry_reporter_quote_attribution"
    )
    assert speaker["voice_family"] == "masculine"
    assert speaker["voice_family_confidence"] == 0.99
    assert speaker["identity_confidence"] == pytest.approx(2 / 3, abs=1e-6)
    proposal = speaker["proposals"][0]
    assert proposal["binding_mode"] == (
        "audited_registry_reporter_attribution_before"
    )
    assert proposal["reporter_attribution"]["observed_transcript_surname"] == (
        "Katalia"
    )
    assert proposal["reporter_attribution"]["canonical_surname"] == "Cachalia"
    assert audit["policy_version"] == CONTEXTUAL_SPEAKER_POLICY_VERSION
    assert audit["safety_contract"]["acoustic_threshold_lowered"] is False


def test_surname_mention_without_attribution_verb_does_not_bind(tmp_path: Path) -> None:
    result, _audit = _resolve_reporter_attribution(
        tmp_path,
        "Katalia discussed the national task team led by General Mkhwanazi.",
        "Firoz Cachalia",
    )
    assert result == {}


def test_ambiguous_audited_surname_attribution_does_not_bind(tmp_path: Path) -> None:
    result, _audit = _resolve_reporter_attribution(
        tmp_path,
        "Katalia says the work will continue.",
        "Firoz Cachalia",
        "Themba Kachalia",
    )
    assert result == {}


def _block(
    block_id: str,
    speaker_id: str,
    start_ms: int,
    end_ms: int,
) -> SpeechBlock:
    return SpeechBlock(
        block_id=block_id,
        speaker_id=speaker_id,
        start_ms=start_ms,
        end_ms=end_ms,
        segment_ids=(block_id,),
        source_text=block_id,
        translated_text=block_id,
        tts_text=block_id,
    )


def test_cross_speaker_timeline_is_clamped_to_canonical_source_words() -> None:
    source_blocks = [
        _block("block_0001", "SPEAKER_00", 0, 2_000),
        _block("block_0002", "SPEAKER_01", 2_100, 4_000),
        _block("block_0003", "SPEAKER_01", 4_100, 5_000),
    ]
    transcript = {
        "segments": [
            _segment("seg-1", "SPEAKER_00", 0, 2, "left", ["word-1"]),
            _segment("seg-2", "SPEAKER_01", 2.1, 5, "right", ["word-2", "word-3"]),
        ],
        "words": [
            {
                "word_id": "word-1",
                "speaker": "AZURE_1",
                "text": "left",
                "start": 1.5,
                "end": 1.9,
            },
            {
                "word_id": "word-2",
                "speaker": "AZURE_2",
                "text": "right",
                "start": 2.1,
                "end": 2.4,
            },
            {
                "word_id": "word-3",
                "speaker": "AZURE_2",
                "text": "continues",
                "start": 4.3,
                "end": 4.7,
            },
        ],
    }
    anchors = build_source_word_speaker_anchors(
        source_blocks,
        transcript,
        tolerance_ms=250,
    )
    assert len(anchors) == 1
    assert anchors[0]["left_word"]["word_id"] == "word-1"
    assert anchors[0]["right_word"]["word_id"] == "word-2"

    geometrically_shifted = [
        _block("block_0001", "SPEAKER_00", 0, 2_800),
        _block("block_0002", "SPEAKER_01", 2_800, 4_000),
        _block("block_0003", "SPEAKER_01", 4_100, 5_000),
    ]
    adjusted, corrections = enforce_source_word_speaker_alignment(
        geometrically_shifted,
        source_blocks,
        anchors,
    )

    assert adjusted[0].end_ms == adjusted[1].start_ms == 2_150
    assert abs(adjusted[0].end_ms - 1_900) <= 250
    assert abs(adjusted[1].start_ms - 2_100) <= 250
    assert adjusted[1].end_ms == 4_000
    assert adjusted[2] == geometrically_shifted[2]
    assert corrections[0]["human_review_required"] is False
    assert corrections[0]["text_immutable"] is True


def test_word_alignment_option_defaults_to_quarter_second_envelope() -> None:
    assert DirectDubOptions().source_word_alignment_tolerance_ms == 250

