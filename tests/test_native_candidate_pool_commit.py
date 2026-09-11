"""Phase 6 of the --candidate-pool architecture: commit_candidate_pool writes
pass2_candidate_pool.json's per-sentence winners into
pass2_natural_translation.json, in the SAME temporal_mask_groups shape the
existing --temporal-mask path writes, so every downstream consumer
(referent_audit, native-qa, render_native_dub) works unchanged. The most
important test here proves that compatibility directly, by feeding a real
commit into _temporal_mask_groups_for_render (the actual render-time
consumer) rather than just asserting on the artifact's own shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.media import checksum
from mathula_tv.native_dub import (
    DEFAULT_MAX_NATURAL_SPEED_PERCENT,
    _temporal_mask_groups_for_render,
    commit_candidate_pool,
    native_dub_paths,
)


def _seed_job(tmp_path: Path, *, sentences: list[str]) -> Path:
    job_root = tmp_path / "job"
    paths = native_dub_paths(job_root)
    paths.pass1.parent.mkdir(parents=True, exist_ok=True)
    segments = [
        {"segment_id": f"seg{i:02d}", "speaker_id": "SPEAKER_00", "restored_text": text,
         "start_ms": i * 4000, "end_ms": i * 4000 + 3000}
        for i, text in enumerate(sentences)
    ]
    atomic_write_json(paths.pass1, {
        "corrections": [], "segments": segments,
        "context_ledger": {"summary": "", "entities": [], "speaker_roles": [], "story_beats": []},
    })
    return job_root


def _candidate(candidate_id: str, spoken_text: str, *, measured_ms: int = 3000,
               required_speed_percent: float = 5.0, translator: str = "grok",
               variant_id: str = "en_natural") -> dict:
    return {
        "candidate_id": candidate_id, "spoken_text": spoken_text, "measured_ms": measured_ms,
        "required_speed_percent": required_speed_percent, "speed_fit_mode": "sentence_end",
        "direction": None, "translator": translator, "variant_id": variant_id,
        "semantic_guard_requirements": [], "path": f"/audio/{candidate_id}.wav",
    }


def _seed_pool(job_root: Path, *, records: list[dict]) -> None:
    paths = native_dub_paths(job_root)
    atomic_write_json(paths.candidate_pool, {
        "candidate_pool_mode": True,
        "candidate_pool": records,
        "usage": {"input_tokens": 10, "output_tokens": 5, "attempts": 1},
    })


def _record(
    group_id: str, *, index: int, selected_candidate_id: str | None, candidates: list[dict],
    requires_review: bool = False, source_text: str = "", speaker_id: str = "SPEAKER_00",
    **extra: object,
) -> dict:
    """Build one candidate-pool record with the fields Phase 16 requires
    (segment_ids/start_ms/source_end_ms) -- `index` matches _seed_job's own
    deterministic spacing (`i * 4000` to `i * 4000 + 3000`, raw segment_id
    `f"seg{i:02d}"`). The REAL sentence-level segment_id _build_pass1_
    sentence_records produces is `f"{raw_segment_id}__s00"` (one sentence per
    raw segment here, via _split_pass1_segment_into_sentences's whole-segment
    fallback -- confirmed by direct read), not the bare raw id.
    """
    start_ms = index * 4000
    source_end_ms = start_ms + 3000
    return {
        "group_id": group_id, "speaker_id": speaker_id,
        "segment_ids": [f"seg{index:02d}__s00"], "start_ms": start_ms, "source_end_ms": source_end_ms,
        "source_text": source_text, "selected_candidate_id": selected_candidate_id,
        "candidates": candidates, "requires_review": requires_review,
        **extra,
    }


def test_commit_requires_the_pool_artifact_first(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["Hello there."])
    with pytest.raises(ValueError, match="candidate-pool"):
        commit_candidate_pool(job_root=job_root)


def test_commit_rejects_an_incomplete_pool(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["Hello there."])
    _seed_pool(job_root, records=[{
        "group_id": "native_phrase_0001", "selected_candidate_id": None,
        "candidates": [], "requires_review": True,
    }])
    with pytest.raises(ValueError, match="no selected candidate-pool winner"):
        commit_candidate_pool(job_root=job_root)


def test_commit_rejects_a_pool_whose_sentences_no_longer_match_pass1(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["Hello there.", "Goodbye."])
    _seed_pool(job_root, records=[
        _record(
            "native_phrase_0001", index=0, selected_candidate_id="grok_en_natural",
            candidates=[_candidate("grok_en_natural", "Sawubona.")],
        ),
    ])  # missing native_phrase_0002 entirely
    with pytest.raises(ValueError, match="no longer reconstruct"):
        commit_candidate_pool(job_root=job_root)


def test_commit_writes_temporal_mask_shape_the_renderer_can_consume(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["Hello there.", "Goodbye."])
    _seed_pool(job_root, records=[
        _record(
            "native_phrase_0001", index=0, selected_candidate_id="grok_en_natural",
            candidates=[_candidate("grok_en_natural", "Sawubona.")],
        ),
        _record(
            "native_phrase_0002", index=1, selected_candidate_id="grok_short1",
            candidates=[_candidate("grok_short1", "Sala kahle.", translator="grok")],
        ),
    ])

    artifact = commit_candidate_pool(job_root=job_root)

    assert artifact["temporal_mask_mode"] is True
    assert artifact["candidate_pool_committed"] is True
    groups = {g["group_id"]: g for g in artifact["temporal_mask_groups"]}
    assert groups["native_phrase_0001"]["spoken_text"] == "Sawubona."
    assert groups["native_phrase_0002"]["spoken_text"] == "Sala kahle."
    assert groups["native_phrase_0002"]["selected_candidate_translator"] == "grok"

    # The real proof: the actual render-time consumer can process this artifact,
    # fed the SAME pass1_sentences.json commit_candidate_pool itself wrote (this
    # is exactly what render_native_dub reads at render time -- see its own
    # temporal_mask_mode branch).
    paths = native_dub_paths(job_root)
    sentence_records = read_json(paths.pass1_sentences)["records"]
    phrase_groups, translations = _temporal_mask_groups_for_render(artifact, sentence_records)
    assert len(phrase_groups) == 2
    committed_texts = sorted(translations.values())
    assert committed_texts == ["Sala kahle.", "Sawubona."]

    # A real pass2_natural_translation.json was actually written to disk.
    on_disk = read_json(paths.pass2)
    assert on_disk["input_sha256"] == artifact["input_sha256"]


def test_commit_sets_rush_warning_when_required_speed_exceeds_threshold(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["Hello there."])
    _seed_pool(job_root, records=[
        _record(
            "native_phrase_0001", index=0, selected_candidate_id="grok_en_natural",
            candidates=[_candidate(
                "grok_en_natural", "Sawubona.",
                required_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT + 5,
            )],
        ),
    ])
    artifact = commit_candidate_pool(job_root=job_root)
    group = artifact["temporal_mask_groups"][0]
    assert group["rush_warning"] is not None
    assert group["rush_warning"]["type"] == "measured_rush_warning"


def test_commit_reuses_checkpoint_without_rewriting(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["Hello there."])
    _seed_pool(job_root, records=[
        _record(
            "native_phrase_0001", index=0, selected_candidate_id="grok_en_natural",
            candidates=[_candidate("grok_en_natural", "Sawubona.")],
        ),
    ])
    first = commit_candidate_pool(job_root=job_root)
    second = commit_candidate_pool(job_root=job_root)
    assert first["completed_at"] == second["completed_at"]


def test_commit_force_regenerates_even_with_a_matching_checkpoint(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["Hello there."])
    _seed_pool(job_root, records=[
        _record(
            "native_phrase_0001", index=0, selected_candidate_id="grok_en_natural",
            candidates=[_candidate("grok_en_natural", "Sawubona.")],
        ),
    ])
    first = commit_candidate_pool(job_root=job_root)
    second = commit_candidate_pool(job_root=job_root, force=True)
    assert first["completed_at"] != second["completed_at"]


def test_commit_writes_pass1_sentences_json_when_missing(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["Hello there."])
    paths = native_dub_paths(job_root)
    assert not paths.pass1_sentences.is_file()
    _seed_pool(job_root, records=[
        _record(
            "native_phrase_0001", index=0, selected_candidate_id="grok_en_natural",
            candidates=[_candidate("grok_en_natural", "Sawubona.")],
        ),
    ])
    commit_candidate_pool(job_root=job_root)
    assert paths.pass1_sentences.is_file()


def test_commit_marks_requires_review_groups_in_the_committed_artifact(tmp_path):
    job_root = _seed_job(tmp_path, sentences=["Hello there."])
    _seed_pool(job_root, records=[
        _record(
            "native_phrase_0001", index=0, selected_candidate_id="grok_en_natural",
            candidates=[_candidate("grok_en_natural", "Sawubona.")], requires_review=True,
        ),
    ])
    artifact = commit_candidate_pool(job_root=job_root)
    assert artifact["temporal_mask_groups"][0]["requires_review"] is True
    assert artifact["summary"]["requires_review_count"] == 1


def test_commit_uses_the_effective_window_override_when_a_turn_was_rebalanced(tmp_path):
    """Phase 13: a sentence whose speaker turn was rebalanced via window
    reallocation carries effective_start_ms/effective_source_end_ms on its
    pool record -- commit must use those instead of the raw Pass-1 window,
    since every downstream consumer (render's per-sentence synthesis/speed-fit)
    reads start_ms/source_end_ms/source_span_ms straight off the committed group.
    """
    job_root = _seed_job(tmp_path, sentences=["Hello there.", "Goodbye now friend."])
    _seed_pool(job_root, records=[
        _record(
            "native_phrase_0001", index=0, selected_candidate_id="grok_en_natural",
            candidates=[_candidate("grok_en_natural", "Sawubona kakhulu.")],
            effective_start_ms=0, effective_source_end_ms=3800,
        ),
        _record(
            "native_phrase_0002", index=1, selected_candidate_id="grok_en_natural",
            candidates=[_candidate("grok_en_natural", "Sala kahle.")],
            effective_start_ms=4800, effective_source_end_ms=7000,
        ),
    ])
    artifact = commit_candidate_pool(job_root=job_root)
    groups = {g["group_id"]: g for g in artifact["temporal_mask_groups"]}
    assert groups["native_phrase_0001"]["start_ms"] == 0
    assert groups["native_phrase_0001"]["source_end_ms"] == 3800
    assert groups["native_phrase_0001"]["source_span_ms"] == 3800
    assert groups["native_phrase_0002"]["start_ms"] == 4800
    assert groups["native_phrase_0002"]["source_end_ms"] == 7000
    assert groups["native_phrase_0002"]["source_span_ms"] == 2200


def test_commit_falls_back_to_the_raw_pass1_window_when_no_override_is_present(tmp_path):
    """The overwhelming majority of sentences are never touched by a boundary
    shift -- their pool record simply has no effective_* fields, and commit
    must fall back to exactly the raw Pass-1-derived window, unchanged from
    before Phase 10.
    """
    job_root = _seed_job(tmp_path, sentences=["Hello there."])
    _seed_pool(job_root, records=[
        _record(
            "native_phrase_0001", index=0, selected_candidate_id="grok_en_natural",
            candidates=[_candidate("grok_en_natural", "Sawubona.")],
        ),
    ])
    artifact = commit_candidate_pool(job_root=job_root)
    group = artifact["temporal_mask_groups"][0]
    assert group["start_ms"] == 0
    assert group["source_end_ms"] == 3000
    assert group["source_span_ms"] == 3000
