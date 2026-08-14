from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mathula_tv.viral_moments as viral
from mathula_tv.viral_moments import ViralMomentsOptions
from mathula_tv.viral_moments_cli import _provider_calls_required


def _transcript(segment_count: int = 16) -> dict:
    segments = []
    for index in range(segment_count):
        start = float(index * 5)
        speaker = "AZURE_1" if index % 2 == 0 else "AZURE_2"
        if index % 4 == 0:
            text = (
                "Commissioner Khumalo, are you saying the document contradicts the earlier "
                "statement and that nobody informed the investigators about the payment?"
            )
        elif index % 4 == 1:
            text = (
                "Yes, that is exactly what happened, and the evidence shows R2 million was "
                "approved even though the written record says the request was rejected."
            )
        elif index % 4 == 2:
            text = (
                "I deny that conclusion because Major General Feroz Khan never gave me the "
                "instruction, but I accept that my signature appears on the final memorandum."
            )
        else:
            text = (
                "The surprising point is that the official timeline changes after the meeting, "
                "which raises a serious question about who altered the record and why."
            )
        segments.append(
            {
                "segment_id": f"seg-{index + 1:05d}",
                "speaker": speaker,
                "start": start,
                "end": start + 4.5,
                "source_text": text,
                "confidence": 0.94,
            }
        )
    return {
        "schema_version": "transcript-v1",
        "language": "en-ZA",
        "segments": segments,
    }


def test_mathula_transcript_adapter_preserves_timing_and_speakers() -> None:
    utterances = viral.utterances_from_mathula_transcript(_transcript(3))
    assert len(utterances) == 3
    assert utterances[0].start_ms == 0
    assert utterances[0].end_ms == 4500
    assert utterances[1].speaker == "AZURE_2"
    assert "R2 million" in utterances[1].text


def test_candidate_builder_returns_ranked_nonduplicate_moments() -> None:
    utterances = viral.utterances_from_mathula_transcript(_transcript())
    candidates = viral.build_candidates(
        utterances,
        min_ms=20_000,
        target_ms=35_000,
        max_ms=55_000,
        candidate_limit=12,
    )
    assert candidates
    assert candidates == sorted(candidates, key=lambda item: item.heuristic_score, reverse=True)
    assert all(candidate.duration_ms <= 55_000 for candidate in candidates)
    assert any(candidate.heuristic_features["hook_pattern_hits"] > 0 for candidate in candidates)


def test_job_run_reuses_authoritative_transcript_and_writes_tiktok_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = "job123"
    job_root = tmp_path / "jobs" / job_id
    source = job_root / "input" / "hearing.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    (job_root / "analysis").mkdir(parents=True)
    transcript_path = job_root / "analysis" / "transcript_en.json"
    transcript_path.write_text(json.dumps(_transcript()), encoding="utf-8")
    (job_root / "status.json").write_text(
        json.dumps({"job_id": job_id, "local_source_path": str(source)}),
        encoding="utf-8",
    )

    monkeypatch.setattr(viral, "require_binary", lambda name: name)
    monkeypatch.setattr(viral, "ffprobe_duration_ms", lambda path: 90_000)

    manifest = viral.run_viral_moments_for_job(
        job_id=job_id,
        work_dir=tmp_path,
        options=ViralMomentsOptions(
            clip_count=3,
            min_clip_seconds=20,
            target_clip_seconds=35,
            max_clip_seconds=55,
            candidate_limit=12,
            no_ai=True,
            dry_run=True,
        ),
    )

    output = job_root / "viral_moments"
    assert manifest["source_job_id"] == job_id
    assert manifest["clip_count"] >= 1
    assert manifest["transcript_source"] == str(transcript_path.resolve())
    queue = [json.loads(line) for line in (output / "tiktok_queue.jsonl").read_text().splitlines()]
    assert len(queue) == manifest["clip_count"]
    assert all(item["pipeline_stage"] == "clean_source_clip" for item in queue)
    assert all(item["tiktok_input"].endswith(".mp4") for item in queue)
    assert all(item["downstream_target_language"] == "zu-ZA" for item in queue)
    assert all(item["submit_argv"][-2:] == ["--target-language", "zu-ZA"] for item in queue)
    submit_script = output / "submit_clips_to_mathula_tv.sh"
    assert submit_script.is_file()
    assert "python -m mathula_tv.cli submit" in submit_script.read_text(encoding="utf-8")
    assert (output / "ranking_debug.json").is_file()

    reused = viral.run_viral_moments_for_job(
        job_id=job_id,
        work_dir=tmp_path,
        options=ViralMomentsOptions(
            clip_count=3,
            min_clip_seconds=20,
            target_clip_seconds=35,
            max_clip_seconds=55,
            candidate_limit=12,
            no_ai=True,
            dry_run=True,
        ),
    )
    assert reused["idempotent_reuse"] is True


def test_force_transcribe_uses_injected_mathula_speech_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hearing.mp4"
    source.write_bytes(b"source")

    class Backend:
        provider = "azure-speech-fast-transcription"

        def __init__(self) -> None:
            self.calls = []

        def transcribe(self, audio: Path, locale: str) -> dict:
            self.calls.append((audio, locale))
            return _transcript()

    backend = Backend()
    monkeypatch.setattr(viral, "require_binary", lambda name: name)
    monkeypatch.setattr(viral, "ffprobe_duration_ms", lambda path: 90_000)

    def fake_extract(_video: Path, audio: Path) -> None:
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"wav")

    monkeypatch.setattr(viral, "_extract_mathula_analysis_audio", fake_extract)

    manifest = viral.run_viral_moments(
        video_path=source,
        output_dir=tmp_path / "out",
        options=ViralMomentsOptions(
            clip_count=2,
            min_clip_seconds=20,
            target_clip_seconds=35,
            max_clip_seconds=55,
            candidate_limit=10,
            no_ai=True,
            dry_run=True,
        ),
        speech_backend=backend,
    )
    assert len(backend.calls) == 1
    assert backend.calls[0][1] == "en-ZA"
    assert manifest["azure_stt_provider"] == backend.provider
    assert (tmp_path / "out" / "azure_stt_raw.json").is_file()


def test_cli_requires_live_operation_only_for_actual_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "jobs" / "job1" / "analysis"
    job.mkdir(parents=True)
    (job / "transcript_en.json").write_text(json.dumps(_transcript()), encoding="utf-8")
    settings = SimpleNamespace(work_dir=tmp_path)
    args = SimpleNamespace(job_id="job1", force_transcribe=False, no_ai=True, dry_run=False)
    assert _provider_calls_required(args, settings) == []

    args.force_transcribe = True
    assert _provider_calls_required(args, settings) == ["Azure Speech transcription"]
