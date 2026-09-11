from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.foundry_grok import FoundryGrokConfig, FoundryGrokProvider
from mathula_tv.models import JobManifest
from mathula_tv.native_dub import (
    MANUAL_WEB_OVERRIDE_SCHEMA_VERSION,
    build_phrase_groups,
    install_manual_web_overrides,
    load_native_voice_assignments,
    render_native_dub,
    render_native_hook_publication,
    run_native_pass1,
    run_native_pass2,
)


class _HTTPResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _HTTPResponse(self.payload)


def test_foundry_grok_uses_openai_v1_chat_without_tools():
    session = _Session({
        "model": "grok-deployment",
        "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    })
    provider = FoundryGrokProvider(
        FoundryGrokConfig(
            api_key="secret",
            endpoint="https://example.services.ai.azure.com",
            deployment="grok-deployment",
            max_retries=0,
            max_schema_repairs=0,
        ),
        session=session,
        sleep=lambda _: None,
    )
    result = provider.complete_json(
        operation="test",
        system_prompt="Return JSON.",
        payload={"x": 1},
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
    )
    assert result.data == {"ok": True}
    url, kwargs = session.calls[0]
    assert url == "https://example.services.ai.azure.com/openai/v1/chat/completions"
    assert kwargs["headers"]["api-key"] == "secret"
    body = kwargs["json"]
    assert body["model"] == "grok-deployment"
    assert "tools" not in body
    assert "tool_choice" not in body
    assert "response_format" not in body


class _PassProvider:
    provider = "azure-foundry-grok"

    def __init__(self, responses):
        self.responses = list(responses)
        self.config = SimpleNamespace(deployment="grok-test")
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=self.responses.pop(0),
            model="grok-test",
            input_tokens=10,
            output_tokens=5,
            attempts=1,
        )


def _write_transcript(job_root: Path):
    atomic_write_json(
        job_root / "analysis" / "transcript_en.json",
        {
            "schema_version": "test",
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "seg-1",
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 2.0,
                    "source_text": "The political killings tossed him spoke.",
                },
                {
                    "segment_id": "seg-2",
                    "speaker": "SPEAKER_00",
                    "start": 2.4,
                    "end": 4.0,
                    "source_text": "He continued naturally.",
                },
            ],
        },
    )


def test_native_two_pass_keeps_production_transcript_immutable(tmp_path: Path):
    job_root = tmp_path / "jobs" / "job1"
    _write_transcript(job_root)
    original = (job_root / "analysis" / "transcript_en.json").read_bytes()
    provider = _PassProvider([
        {
            "schema_version": "mathula-native-stt-restoration-v1",
            "corrections": [
                {
                    "segment_id": "seg-1",
                    "original_text": "The political killings tossed him spoke.",
                    "corrected_text": "The Political Killings Task Team spoke.",
                    "reason_code": "asr_substitution",
                }
            ],
        },
        {
            "schema_version": "mathula-native-natural-translation-v1",
            "translations": [
                {
                    "segment_id": "seg-1",
                    "source_text": "The Political Killings Task Team spoke.",
                    "spoken_text": "I-Political Killings Task Team yakhuluma.",
                },
                {
                    "segment_id": "seg-2",
                    "source_text": "He continued naturally.",
                    "spoken_text": "Waqhubeka ekhuluma ngokwemvelo.",
                },
            ],
        },
    ])
    pass1 = run_native_pass1(job_root=job_root, provider=provider)
    pass2 = run_native_pass2(job_root=job_root, provider=provider)
    assert (job_root / "analysis" / "transcript_en.json").read_bytes() == original
    assert pass1["segments"][0]["restored_text"] == "The Political Killings Task Team spoke."
    assert pass1["segments"][1]["changed"] is False
    assert pass2["variant_count"] == 1
    assert pass2["timing_optimized"] is True
    assert pass2["preferred_raw_speed_percent"] == 6
    assert pass2["word_count_optimized"] is False
    assert all("duration" not in item for item in pass2["translations"])
    assert all("tools" not in call for call in provider.calls)


def test_manual_web_override_is_explicit_artifact_only(tmp_path: Path):
    source = tmp_path / "override.json"
    source.write_text(json.dumps({
        "schema_version": MANUAL_WEB_OVERRIDE_SCHEMA_VERSION,
        "entries": [
            {
                "heard_as": "Fanima Semula",
                "canonical": "Fannie Masemola",
                "reason": "human-reviewed web research",
                "evidence_url": "https://example.test/evidence",
            }
        ],
    }), encoding="utf-8")
    result = install_manual_web_overrides(tmp_path / "job", source)
    assert result["manual_only"] is True
    assert result["automatic_web_search"] is False


def test_phrase_groups_do_not_borrow_positive_same_speaker_source_gap():
    segments = [
        {"segment_id": "a", "speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 1000, "restored_text": "A"},
        {"segment_id": "b", "speaker_id": "SPEAKER_00", "start_ms": 1400, "end_ms": 2400, "restored_text": "B"},
        {"segment_id": "c", "speaker_id": "SPEAKER_01", "start_ms": 2500, "end_ms": 3000, "restored_text": "C"},
    ]
    groups = build_phrase_groups(
        segments,
        {"a": "Zulu A", "b": "Zulu B", "c": "Zulu C"},
        max_join_gap_ms=900,
        max_phrase_source_ms=45000,
    )
    assert len(groups) == 3
    assert groups[0]["segment_ids"] == ["a"]
    assert groups[1]["segment_ids"] == ["b"]
    assert all(part["type"] != "break" for group in groups for part in group["parts"])


def test_voice_mapping_fails_closed_for_missing_speaker(tmp_path: Path):
    root = tmp_path / "job"
    atomic_write_json(
        root / "direct_dub" / "voice_resolution.json",
        {
            "speakers": [
                {
                    "speaker_id": "SPEAKER_00",
                    "status": "resolved",
                    "selected_voice": "zu-ZA-ThembaNeural",
                    "base_prosody": {},
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="missing mappings"):
        load_native_voice_assignments(root, required_speakers={"SPEAKER_00", "SPEAKER_01"})


class _FakeTTS:
    sample_rate = 24000

    def __init__(self):
        self.requests = []

    def synthesize(self, request, output_path, *, force=False):
        self.requests.append(request)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-wav")
        # Short enough to avoid timing repair and final tail failure.
        duration = min(request.preferred_duration_ms, 1000)
        return SimpleNamespace(
            output_path=str(output_path),
            voice=request.voice,
            duration_ms=duration,
        )


def test_native_render_uses_rate_zero_and_keeps_video_unmodified(tmp_path: Path, monkeypatch):
    job_root = tmp_path / "jobs" / "job1"
    source_video = job_root / "input" / "source.mp4"
    source_video.parent.mkdir(parents=True, exist_ok=True)
    source_video.write_bytes(b"video-source-bytes")
    source_hash_before = source_video.read_bytes()
    _write_transcript(job_root)
    atomic_write_json(
        job_root / "native_dub" / "pass1_restored_transcript.json",
        {
            "segments": [
                {"segment_id": "seg-1", "speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 2000, "source_text": "A", "restored_text": "A"},
                {"segment_id": "seg-2", "speaker_id": "SPEAKER_00", "start_ms": 2400, "end_ms": 4000, "source_text": "B", "restored_text": "B"},
            ]
        },
    )
    atomic_write_json(
        job_root / "native_dub" / "pass2_natural_translation.json",
        {
            "schema_version": "mathula-native-natural-translation-v1",
            "translations": [
                {"segment_id": "seg-1", "source_text": "A", "spoken_text": "Zulu A"},
                {"segment_id": "seg-2", "source_text": "B", "spoken_text": "Zulu B"},
            ],
        },
    )
    atomic_write_json(
        job_root / "direct_dub" / "voice_resolution.json",
        {
            "speakers": [
                {
                    "speaker_id": "SPEAKER_00",
                    "status": "resolved",
                    "selected_voice": "zu-ZA-ThembaNeural",
                    "base_prosody": {"rate_percent": 8, "pitch_percent": 3, "volume_percent": 0},
                }
            ]
        },
    )

    def fake_dialogue(clips, output_dir, dialogue_output, **kwargs):
        dialogue_output.parent.mkdir(parents=True, exist_ok=True)
        dialogue_output.write_bytes(b"dialogue")
        return {"schema_version": "fake", "clips": clips}

    def fake_render(source, audio, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"rendered-video")

    monkeypatch.setattr("mathula_tv.native_dub.render_dialogue_tracks", fake_dialogue)
    monkeypatch.setattr("mathula_tv.native_dub.render_video", fake_render)
    tts = _FakeTTS()
    job = JobManifest(
        job_id="job1",
        source_filename="source.mp4",
        local_source_path=str(source_video),
        source_checksum="x",
        source_duration=5.0,
    )
    result = render_native_dub(
        job=job,
        job_root=job_root,
        tts=tts,
        timing_repair_provider=None,
        render_hook_overlay=False,
    )
    assert tts.requests
    assert all(request.rate_percent == 0 for request in tts.requests)
    assert all(request.voice == "zu-ZA-ThembaNeural" for request in tts.requests)
    assert source_video.read_bytes() == source_hash_before
    assert result["video_retimed"] is False
    assert result["tts_rate_percent"] == 0



def test_foundry_grok_structured_adapter_still_sends_no_tools():
    from mathula_tv.ai_provider import StructuredAIRequest

    session = _Session({
        "model": "grok-deployment",
        "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    })
    provider = FoundryGrokProvider(
        FoundryGrokConfig(
            api_key="secret",
            endpoint="https://example.services.ai.azure.com",
            deployment="grok-deployment",
            max_retries=0,
            max_schema_repairs=0,
        ),
        session=session,
        sleep=lambda _: None,
    )
    request = StructuredAIRequest(
        operation="editorial-test",
        payload={"x": 1},
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        prompt_version="test-v1",
        system_prompt="Return JSON.",
        response_schema_version="test-v1",
    )
    response = provider.complete_structured(request)
    assert response.data == {"ok": True}
    assert response.metadata.provider == "azure-foundry-grok"
    body = session.calls[0][1]["json"]
    assert "tools" not in body
    assert "tool_choice" not in body


def test_native_hook_publication_reuses_production_footer_renderer(tmp_path: Path, monkeypatch):
    job_root = tmp_path / "jobs" / "job1"
    master = job_root / "native_dub" / "native_dubbed.mp4"
    master.parent.mkdir(parents=True, exist_ok=True)
    master.write_bytes(b"native-master")
    atomic_write_json(
        job_root / "native_dub" / "pass2_natural_translation.json",
        {
            "schema_version": "mathula-native-natural-translation-v1",
            "translations": [
                {"segment_id": "seg-1", "source_text": "Hello", "spoken_text": "Sawubona"}
            ],
        },
    )
    captured = {}

    def fake_select(**kwargs):
        captured["select"] = kwargs
        return {
            "hook_text": "I-HOOK YESIZULU",
            "cut_start_seconds": 0.0,
            "selected_block_id": "seg-1",
            "editorial_seo": {
                "schema_version": "mathula-tiktok-seo-v2-single-editorial-call",
                "title": "I-HOOK YESIZULU",
                "cover_hook": "I-HOOK YESIZULU",
                "caption": "Searchable English caption",
                "tiktok_caption": "Searchable English caption",
                "search_keywords": ["keyword one", "keyword two"],
                "topic_hashtags": ["#One", "#Two", "#Three", "#Four"],
            },
        }

    def fake_render(**kwargs):
        captured["render"] = kwargs
        kwargs["output_path"].write_bytes(b"hook-video")
        return {
            "output": {"path": str(kwargs["output_path"]), "sha256": "x"},
            "title_panel": {"text": kwargs["title_text"]},
            "hook_edit_enabled": kwargs["apply_opening_cut"],
            "hook_card_enabled": True,
        }

    monkeypatch.setattr("mathula_tv.tiktok_editor.select_tiktok_hook", fake_select)
    monkeypatch.setattr("mathula_tv.tiktok_editor.render_tiktok_hook_edit", fake_render)
    monkeypatch.setattr("mathula_tv.tiktok_editor.load_publication_title_authority", lambda root: {})
    provider = SimpleNamespace(
        provider="azure-foundry-grok",
        config=SimpleNamespace(deployment="grok-test"),
    )
    job = JobManifest(
        job_id="job1",
        source_filename="source.mp4",
        local_source_path=str(master),
        source_checksum="x",
        source_duration=2.0,
        target_language="zu-ZA",
    )
    result = render_native_hook_publication(
        job=job,
        job_root=job_root,
        master_video=master,
        segments=[
            {
                "segment_id": "seg-1",
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 2000,
                "restored_text": "Hello",
            }
        ],
        translations={"seg-1": "Sawubona"},
        provider=provider,
        force=False,
        apply_opening_cut=False,
    )
    assert captured["select"]["blocks"][0]["translated_text"] == "Sawubona"
    assert captured["render"]["apply_opening_cut"] is False
    assert captured["render"]["title_text"] == "I-HOOK YESIZULU"
    assert captured["render"]["translation_path"].name == "pass2_natural_translation.json"
    assert result["hook_card_enabled"] is True
    assert result["native_publication"]["hook_overlay_preserved_from_production_renderer"] is True
    assert result["native_publication"]["seo_preserved_from_production_editorial_call"] is True
    seo = json.loads((job_root / "native_dub" / "tiktok_seo.json").read_text(encoding="utf-8"))
    assert seo["caption"] == "Searchable English caption"
    assert seo["topic_hashtags"] == ["#One", "#Two", "#Three", "#Four"]
    assert (job_root / "native_dub" / "tiktok_caption.txt").read_text(encoding="utf-8").strip() == "Searchable English caption"
    assert (job_root / "native_dub" / "tiktok_hashtags.txt").read_text(encoding="utf-8").strip() == "#One #Two #Three #Four"
    assert (job_root / "native_dub" / "tiktok_search_keywords.txt").read_text(encoding="utf-8").splitlines() == ["keyword one", "keyword two"]


def test_native_pass1_builds_raw_transcript_directly_from_azure(tmp_path: Path):
    job_root = tmp_path / "jobs" / "job1"
    atomic_write_json(
        job_root / "analysis" / "azure_diarization.json",
        {
            "schema_version": "azure-stt-normalized-v1",
            "provider": "azure-speech-fast-transcription",
            "duration": 2.0,
            "words": [
                {
                    "word_id": "word_000001",
                    "source_phrase_id": "phrase_00001",
                    "text": "Hello",
                    "start": 0.0,
                    "end": 0.4,
                    "duration": 0.4,
                    "confidence": 0.99,
                    "speaker": "AZURE_1",
                },
                {
                    "word_id": "word_000002",
                    "source_phrase_id": "phrase_00001",
                    "text": "there",
                    "start": 0.45,
                    "end": 0.8,
                    "duration": 0.35,
                    "confidence": 0.99,
                    "speaker": "AZURE_1",
                },
            ],
            "turns": [
                {
                    "phrase_id": "phrase_00001",
                    "speaker": "AZURE_1",
                    "start": 0.0,
                    "end": 0.8,
                    "duration": 0.8,
                    "overlap": False,
                    "confidence": 0.99,
                    "source": "azure",
                }
            ],
            "warnings": ["Azure diarization is authoritative; speaker identifiers are job-local"],
        },
    )
    provider = _PassProvider([
        {
            "schema_version": "mathula-native-stt-restoration-v1",
            "corrections": [],
        }
    ])
    result = run_native_pass1(job_root=job_root, provider=provider)
    raw_path = job_root / "analysis" / "transcript_en_raw.json"
    assert raw_path.is_file()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw["segments"][0]["source_text"] == "Hello there"
    assert raw["segments"][0]["speaker_id"] == "SPEAKER_00"
    assert raw["native_source_reconstruction"]["automatic_web_search"] is False
    assert raw["native_source_reconstruction"]["legacy_autocorrect_used"] is False
    assert result["source_transcript_kind"] == "raw_stt"
    assert result["segments"][0]["source_text"] == "Hello there"


def test_native_voice_resolution_is_compact_persisted_and_rate_zero(tmp_path: Path):
    from mathula_tv.native_dub import ensure_native_voice_assignments

    job_root = tmp_path / "jobs" / "job1"
    provider = _PassProvider([
        {
            "voices": [
                {"speaker_id": "SPEAKER_00", "voice_family": "masculine", "confidence": 0.95},
                {"speaker_id": "SPEAKER_01", "voice_family": "feminine", "confidence": 0.91},
            ]
        }
    ])
    segments = [
        {"segment_id": "a", "speaker_id": "SPEAKER_00", "source_text": "Good morning commissioner."},
        {"segment_id": "b", "speaker_id": "SPEAKER_01", "source_text": "Thank you, Mr Adams."},
    ]
    path = ensure_native_voice_assignments(
        job=JobManifest(job_id="job1", source_filename="x.mp4", local_source_path="x.mp4", source_checksum="x", source_duration=1.0),
        job_root=job_root,
        segments=segments,
        context_ledger={"speaker_roles": [], "entities": []},
        provider=provider,
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["automatic_web_search"] is False
    assert value["tts_rate_percent"] == 0
    assert value["assignments"]["SPEAKER_00"]["selected_voice"] == "zu-ZA-ThembaNeural"
    assert value["assignments"]["SPEAKER_01"]["selected_voice"] == "zu-ZA-ThandoNeural"
    call = provider.calls[0]
    assert call["operation"] == "native_voice_resolution"
    assert "segments" not in call["payload"]
    assert len(call["payload"]["speakers"]) == 2


def test_native_voice_resolution_fails_closed_when_uncertain(tmp_path: Path):
    from mathula_tv.native_dub import ensure_native_voice_assignments

    provider = _PassProvider([{"voices": [{"speaker_id": "SPEAKER_00", "voice_family": "unknown", "confidence": 0.2}]}])
    with pytest.raises(ValueError, match="will not default unknown voices"):
        ensure_native_voice_assignments(
            job=JobManifest(job_id="job1", source_filename="x.mp4", local_source_path="x.mp4", source_checksum="x", source_duration=1.0),
            job_root=tmp_path / "job1",
            segments=[{"segment_id": "a", "speaker_id": "SPEAKER_00", "source_text": "Hello"}],
            context_ledger={"speaker_roles": [], "entities": []},
            provider=provider,
        )


def test_native_voice_resolution_requires_legacy_classifier_before_grok_when_model_missing(tmp_path: Path, monkeypatch):
    import mathula_tv.native_dub as native_dub

    provider = _PassProvider([{"voices": [{"speaker_id": "SPEAKER_00", "voice_family": "masculine", "confidence": 0.99}]}])

    monkeypatch.setattr(
        native_dub,
        "_native_legacy_acoustic_evidence",
        lambda **kwargs: ({}, {}, "The pinned voice-family classifier is not installed locally."),
    )

    with pytest.raises(ValueError, match="install_voice_gender_classifier.py"):
        native_dub.ensure_native_voice_assignments(
            job=JobManifest(job_id="job1", source_filename="x.mp4", local_source_path="x.mp4", source_checksum="x", source_duration=1.0),
            job_root=tmp_path / "job1",
            segments=[{"segment_id": "a", "speaker_id": "SPEAKER_00", "source_text": "Hello"}],
            context_ledger={"speaker_roles": [], "entities": []},
            provider=provider,
        )
    assert provider.calls == []
