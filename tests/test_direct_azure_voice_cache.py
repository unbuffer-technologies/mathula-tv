from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mathula_tv.direct_azure_dub as direct
from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.direct_azure_dub import (
    DIRECT_AZURE_PERSONA_VERSION,
    LEGACY_VOICE_RESOLUTION_SCHEMA_VERSION,
    DirectAzureDubRenderer,
    DirectDubArtifacts,
    DirectDubOptions,
    SpeechBlock,
)
from mathula_tv.models import JobManifest
from mathula_tv.pronunciation import PronunciationDictionary


class _Backend:
    ssml_bounds = SimpleNamespace(rate_min_percent=-50, rate_max_percent=50)


def _renderer(tmp_path: Path) -> DirectAzureDubRenderer:
    settings = SimpleNamespace(
        work_dir=tmp_path,
        max_source_duration_minutes=240,
    )
    return DirectAzureDubRenderer(settings, _Backend())


def _job(tmp_path: Path, renderer: DirectAzureDubRenderer) -> JobManifest:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    job = JobManifest(
        job_id="job-1",
        source_filename=source.name,
        local_source_path=str(source),
        source_checksum="source-sha",
        source_duration=2.0,
    )
    renderer.jobs.save(job)
    return job


def _assignment() -> dict[str, object]:
    return {
        "speaker_id": "SPEAKER_00",
        "selected_voice": "zu-ZA-ThembaNeural",
        "voice_family": "masculine",
        "voice_family_confidence": 0.98,
        "voice_family_source": "speechbrain_ecapa_known_speaker",
        "raw_speaker_ids": ["Guest-1"],
        "base_prosody": {
            "strategy": "source_relative_single_speaker",
            "persona_version": DIRECT_AZURE_PERSONA_VERSION,
            "rate_percent": 0,
            "pitch_percent": 0,
            "volume_percent": 0,
        },
        "source_features": {},
        "resolution_status": "known_speaker_signature",
        "known_speaker": {
            "person_id": "guest",
            "azure_voice": "zu-ZA-ThembaNeural",
        },
    }


def _legacy_cache(
    renderer: DirectAzureDubRenderer,
    job: JobManifest,
    *,
    mapping: dict[str, object],
) -> Path:
    path = DirectDubArtifacts(renderer.jobs.job_dir(job.job_id)).voice_resolution
    assignment = _assignment()
    atomic_write_json(
        path,
        {
            "schema_version": LEGACY_VOICE_RESOLUTION_SCHEMA_VERSION,
            "persona_version": DIRECT_AZURE_PERSONA_VERSION,
            "job_id": job.job_id,
            "speaker_id_mapping": mapping,
            "minimum_voice_family_confidence": 0.70,
            "sources": {"explicit_voice_map": None},
            "speakers": [
                {
                    "status": "resolved",
                    **assignment,
                    "mapped_acoustic_evidence": {},
                    "known_speaker_match": assignment["known_speaker"],
                    "known_speaker_error": None,
                }
            ],
            "unresolved_speakers": [],
        },
    )
    return path


def test_loads_legacy_job_local_voice_resolution_without_speechbrain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = _renderer(tmp_path)
    job = _job(tmp_path, renderer)
    mapping = {
        "raw_to_canonical": {"Guest-1": "SPEAKER_00"},
        "canonical_to_raw": {"SPEAKER_00": ["Guest-1"]},
    }
    _legacy_cache(renderer, job, mapping=mapping)
    monkeypatch.setattr(direct, "build_speaker_id_mapping", lambda *_: mapping)

    assignments = renderer._load_cached_voice_assignments(
        job,
        None,
        DirectDubOptions(),
        required_speaker_ids=["SPEAKER_00"],
        output_path=DirectDubArtifacts(renderer.jobs.job_dir(job.job_id)).voice_resolution,
    )

    assert assignments is not None
    assert assignments["SPEAKER_00"] == _assignment()


def _prepare_render_inputs(
    tmp_path: Path,
    renderer: DirectAzureDubRenderer,
    job: JobManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_root = renderer.jobs.job_dir(job.job_id)
    translation = job_root / "translation" / "transcript_zu.json"
    atomic_write_json(
        translation,
        {
            "segments": [
                {
                    "segment_id": "seg-1",
                    "speaker_id": "SPEAKER_00",
                    "start": 0.0,
                    "end": 2.0,
                    "source_text": "Hello",
                    "translated_text": "Sawubona",
                }
            ]
        },
    )
    seo_path = job_root / "translation" / "tiktok_zu.json"
    atomic_write_json(seo_path, {"title": "Test"})
    monkeypatch.setattr(direct, "probe", lambda *_: {"duration": 2.0})
    monkeypatch.setattr(
        direct,
        "load_seo_output_context",
        lambda *_: SimpleNamespace(seo_path=seo_path),
    )


def test_unchanged_job_returns_before_voice_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = _renderer(tmp_path)
    job = _job(tmp_path, renderer)
    _prepare_render_inputs(tmp_path, renderer, job, monkeypatch)
    monkeypatch.setattr(
        renderer,
        "_load_cached_voice_assignments",
        lambda *args, **kwargs: {"SPEAKER_00": _assignment()},
    )
    monkeypatch.setattr(renderer, "_request_fingerprint", lambda *args: "fingerprint")
    monkeypatch.setattr(
        renderer,
        "_reuse_completed",
        lambda *args: {"idempotent_reuse": True, "fast_path": True},
    )
    monkeypatch.setattr(
        renderer,
        "_voice_assignments",
        lambda *args, **kwargs: pytest.fail("SpeechBrain path must not run"),
    )

    events: list[dict[str, object]] = []
    result = renderer.render(
        job,
        options=DirectDubOptions(),
        force=False,
        progress_callback=events.append,
    )

    assert result["idempotent_reuse"] is True
    assert result["fast_path"] is True
    assert [event["stage"] for event in events] == [
        "prepare",
        "prepare",
        "voice_resolution",
        "voice_resolution",
        "complete",
    ]
    assert events[-1]["idempotent_reuse"] is True


def test_force_rebuild_reuses_voice_resolution_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = _renderer(tmp_path)
    job = _job(tmp_path, renderer)
    _prepare_render_inputs(tmp_path, renderer, job, monkeypatch)
    monkeypatch.setattr(
        renderer,
        "_load_cached_voice_assignments",
        lambda *args, **kwargs: {"SPEAKER_00": _assignment()},
    )
    monkeypatch.setattr(renderer, "_request_fingerprint", lambda *args: "fingerprint")
    monkeypatch.setattr(
        renderer,
        "_voice_assignments",
        lambda *args, **kwargs: pytest.fail("--force must not rerun SpeechBrain"),
    )

    def stop_after_voice_cache(*args, **kwargs):
        raise RuntimeError("reached Azure synthesis")

    monkeypatch.setattr(renderer, "_synthesize_block", stop_after_voice_cache)
    with pytest.raises(RuntimeError, match="reached Azure synthesis"):
        renderer.render(job, options=DirectDubOptions(), force=True)


def test_explicit_refresh_bypasses_voice_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = _renderer(tmp_path)
    job = _job(tmp_path, renderer)
    _prepare_render_inputs(tmp_path, renderer, job, monkeypatch)
    monkeypatch.setattr(
        renderer,
        "_load_cached_voice_assignments",
        lambda *args, **kwargs: pytest.fail("refresh must bypass cache"),
    )
    observed: dict[str, object] = {}

    def resolve(*args, **kwargs):
        observed["refresh"] = kwargs["refresh_voice_analysis"]
        return {"SPEAKER_00": _assignment()}

    monkeypatch.setattr(renderer, "_voice_assignments", resolve)
    monkeypatch.setattr(renderer, "_request_fingerprint", lambda *args: "fingerprint")
    monkeypatch.setattr(
        renderer,
        "_synthesize_block",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("resolved")),
    )

    with pytest.raises(RuntimeError, match="resolved"):
        renderer.render(
            job,
            options=DirectDubOptions(),
            force=True,
            refresh_voice_analysis=True,
        )

    assert observed["refresh"] is True


def test_render_fingerprint_changes_with_pronunciation_dictionary(
    tmp_path: Path,
) -> None:
    renderer = _renderer(tmp_path)
    job = _job(tmp_path, renderer)
    translation = tmp_path / "translation.json"
    translation.write_text("{}", encoding="utf-8")
    seo = tmp_path / "seo.json"
    seo.write_text("{}", encoding="utf-8")
    block = SpeechBlock(
        "block-1",
        "SPEAKER_00",
        0,
        2_000,
        ("seg-1",),
        "KZN",
        "KZN",
        "KZN",
    )

    def fingerprint(dictionary_version: str) -> str:
        return renderer._request_fingerprint(
            job,
            translation,
            [block],
            {"SPEAKER_00": _assignment()},
            DirectDubOptions(),
            PronunciationDictionary(
                dictionary_version,
                language="zu-ZA",
                job_id=job.job_id,
            ),
            None,
            seo,
            {},
            {},
        )

    assert fingerprint("v1") != fingerprint("v2")
