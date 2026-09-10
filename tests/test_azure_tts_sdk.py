from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from mathula_tv.azure_tts import AzureTTSError, AzureTTSIdempotencyError
from mathula_tv import azure_tts_sdk
from mathula_tv.azure_tts_sdk import (
    AzureSpeechSDKLiveBoundary,
    GroupBookmarkSynthesisResult,
    SDKBookmarkEvent,
    SDKGroupSynthesisResponse,
    _extract_bookmark_offsets_ms,
    _import_speech_sdk,
    synthesize_group_with_bookmarks,
    write_group_slices,
)

VOICE = "zu-ZA-ThandoNeural"
ALLOWED_VOICES = {VOICE}


def pcm_wav(*, seconds: float, leading_silence_ms: int = 0, rate: int = 24_000, value: int = 1200) -> bytes:
    frames = (
        b"\x00\x00" * int(rate * leading_silence_ms / 1000)
        + int(value).to_bytes(2, "little", signed=True) * int(rate * seconds)
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        target.writeframes(frames)
    return output.getvalue()


class FakeBoundary:
    def __init__(self, *, responses=()):
        self.responses = list(responses)
        self.calls: list[tuple[str, float]] = []

    def synthesize_group(self, ssml, *, timeout_seconds):
        self.calls.append((ssml, timeout_seconds))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _islands():
    return [
        ("g1__isl00", "Sawubona."),
        ("g1__isl01", "Ninjani?"),
        ("g1__isl02", "Ngiyabonga kakhulu."),
    ]


def _completed_response(*, bookmarks_ms: dict[str, int], leading_silence_ms: int = 0, seconds: float = 3.0):
    marks = sorted(bookmarks_ms, key=lambda mark: bookmarks_ms[mark])
    return SDKGroupSynthesisResponse(
        audio_data=pcm_wav(seconds=seconds, leading_silence_ms=leading_silence_ms),
        bookmarks=tuple(
            SDKBookmarkEvent(mark=mark, audio_offset_ticks=bookmarks_ms[mark] * 10_000) for mark in marks
        ),
    )


def test_synthesizes_group_writes_artifacts_and_reports_bookmark_offsets(tmp_path):
    boundary = FakeBoundary(
        responses=[_completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800})]
    )
    output_path = tmp_path / "g1.sdk-group.wav"

    result = synthesize_group_with_bookmarks(
        boundary,
        _islands(),
        parent_group_id="g1",
        voice=VOICE,
        allowed_voices=ALLOWED_VOICES,
        output_path=output_path,
    )

    assert result.island_ids == ("g1__isl00", "g1__isl01", "g1__isl02")
    assert result.bookmark_offsets_ms == {"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800}
    assert result.group_duration_ms == 3000
    assert output_path.exists()
    assert Path(result.ssml_path).exists()
    assert Path(result.manifest_path).exists()
    assert len(boundary.calls) == 1


def test_reuses_idempotently_without_calling_boundary_again(tmp_path):
    boundary = FakeBoundary(
        responses=[_completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800})]
    )
    output_path = tmp_path / "g1.sdk-group.wav"
    kwargs = dict(
        parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES, output_path=output_path
    )
    first = synthesize_group_with_bookmarks(boundary, _islands(), **kwargs)
    second = synthesize_group_with_bookmarks(boundary, _islands(), **kwargs)

    assert len(boundary.calls) == 1
    assert second.idempotent_reuse is True
    assert second.bookmark_offsets_ms == first.bookmark_offsets_ms


def test_force_bypasses_reuse_and_resynthesizes(tmp_path):
    boundary = FakeBoundary(
        responses=[
            _completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800}),
            _completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800}),
        ]
    )
    output_path = tmp_path / "g1.sdk-group.wav"
    kwargs = dict(
        parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES, output_path=output_path
    )
    synthesize_group_with_bookmarks(boundary, _islands(), **kwargs)
    synthesize_group_with_bookmarks(boundary, _islands(), force=True, **kwargs)

    assert len(boundary.calls) == 2


def test_different_text_at_same_path_conflicts_without_force(tmp_path):
    # Mirrors AzureTTSBackend._reuse_existing: a request-hash mismatch at an
    # existing artifact path is a real conflict (different content trying to reuse
    # someone else's synthesis), not something to silently regenerate over --
    # forced regeneration must be explicit.
    boundary = FakeBoundary(
        responses=[_completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800})]
    )
    output_path = tmp_path / "g1.sdk-group.wav"
    kwargs = dict(
        parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES, output_path=output_path
    )
    synthesize_group_with_bookmarks(boundary, _islands(), **kwargs)
    changed_islands = [("g1__isl00", "Sawubona."), ("g1__isl01", "Kunjani?"), ("g1__isl02", "Ngiyabonga kakhulu.")]

    with pytest.raises(AzureTTSIdempotencyError, match="different synthesis request"):
        synthesize_group_with_bookmarks(boundary, changed_islands, **kwargs)
    assert len(boundary.calls) == 1

    boundary.responses.append(
        _completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1200, "g1__isl02": 2000})
    )
    forced = synthesize_group_with_bookmarks(boundary, changed_islands, force=True, **kwargs)
    assert len(boundary.calls) == 2
    assert forced.bookmark_offsets_ms == {"g1__isl00": 0, "g1__isl01": 1200, "g1__isl02": 2000}


def test_leading_silence_trim_shifts_bookmark_offsets(tmp_path):
    # Bookmarks are reported relative to the RAW audio; canonicalization trims
    # leading digital silence off the front of the WAV that gets written to disk
    # (and that slice_wav_by_bookmarks will later slice), so offsets must shift
    # left by the trimmed amount to stay aligned with what's actually on disk.
    boundary = FakeBoundary(
        responses=[
            _completed_response(
                bookmarks_ms={"g1__isl00": 100, "g1__isl01": 1100, "g1__isl02": 1900},
                leading_silence_ms=100,
                seconds=3.0,
            )
        ]
    )
    output_path = tmp_path / "g1.sdk-group.wav"
    result = synthesize_group_with_bookmarks(
        boundary, _islands(), parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES,
        output_path=output_path, trim_digital_silence_ms=200,
    )
    assert result.bookmark_offsets_ms == {"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800}


def test_bookmark_order_mismatch_raises_malformed_response(tmp_path):
    boundary = FakeBoundary(
        responses=[_completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1800, "g1__isl02": 1000})]
    )
    with pytest.raises(AzureTTSError) as excinfo:
        synthesize_group_with_bookmarks(
            boundary, _islands(), parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES,
            output_path=tmp_path / "g1.sdk-group.wav",
        )
    assert excinfo.value.category == "malformed_response"


def test_missing_bookmark_raises_malformed_response(tmp_path):
    boundary = FakeBoundary(responses=[_completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000})])
    with pytest.raises(AzureTTSError, match="do not match the requested islands"):
        synthesize_group_with_bookmarks(
            boundary, _islands(), parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES,
            output_path=tmp_path / "g1.sdk-group.wav",
        )


def test_transient_cancellation_is_retried_then_succeeds(tmp_path):
    boundary = FakeBoundary(
        responses=[
            SDKGroupSynthesisResponse(
                audio_data=b"", bookmarks=(), cancelled=True,
                cancellation_reason="Error", cancellation_error_code="ServiceTimeout",
            ),
            _completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800}),
        ]
    )
    sleeps: list[float] = []
    result = synthesize_group_with_bookmarks(
        boundary, _islands(), parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES,
        output_path=tmp_path / "g1.sdk-group.wav", max_retries=1, sleep=sleeps.append,
    )
    assert result.attempt == 2
    assert len(boundary.calls) == 2
    assert sleeps


def test_non_transient_cancellation_raises_immediately(tmp_path):
    boundary = FakeBoundary(
        responses=[
            SDKGroupSynthesisResponse(
                audio_data=b"", bookmarks=(), cancelled=True,
                cancellation_reason="Error", cancellation_error_code="AuthenticationFailure",
            ),
        ]
    )
    with pytest.raises(AzureTTSError) as excinfo:
        synthesize_group_with_bookmarks(
            boundary, _islands(), parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES,
            output_path=tmp_path / "g1.sdk-group.wav", max_retries=3,
        )
    assert excinfo.value.retryable is False
    assert len(boundary.calls) == 1


def test_extract_bookmark_offsets_rejects_duplicate_offsets():
    response = SDKGroupSynthesisResponse(
        audio_data=b"",
        bookmarks=(
            SDKBookmarkEvent(mark="a", audio_offset_ticks=1000 * 10_000),
            SDKBookmarkEvent(mark="b", audio_offset_ticks=1000 * 10_000),
        ),
    )
    with pytest.raises(AzureTTSError, match="out of order or duplicated"):
        _extract_bookmark_offsets_ms(response, ["a", "b"])


class _FakeSpeechSDKModule:
    """Minimal stand-in for azure.cognitiveservices.speech, shaped to match the
    REAL installed SDK (1.51.2) rather than the (wrong) API the production code
    used to assume -- there is no CancellationDetails.from_result classmethod
    for a synthesis result in this version; the sole get()-result attribute
    (`.cancellation_details`) is deliberately the only source of cancellation
    info this fake exposes, so a regression back to the old, broken API would
    fail this test with an AttributeError exactly like the real bug did.
    """

    class ResultReason:
        SynthesizingAudioCompleted = "SynthesizingAudioCompleted"
        Canceled = "Canceled"

    class SpeechSynthesisOutputFormat:
        Riff24Khz16BitMonoPcm = "Riff24Khz16BitMonoPcm"

    class SpeechConfig:
        def __init__(self, *, subscription, region):
            self.subscription = subscription
            self.region = region

        def set_speech_synthesis_output_format(self, fmt):
            self.output_format = fmt

    class _CancellationDetails:
        def __init__(self, *, reason, error_code, error_details):
            self.reason = reason
            self.error_code = error_code
            self.error_details = error_details

    class _BookmarkSignal:
        def connect(self, callback):
            pass

    class _Result:
        def __init__(self, *, reason, audio_data=b"", cancellation_details=None):
            self.reason = reason
            self.audio_data = audio_data
            self.cancellation_details = cancellation_details

    class _ResultFuture:
        def __init__(self, result):
            self._result = result

        def get(self):
            return self._result

    def __init__(self, result):
        self._result = result

    def SpeechSynthesizer(self, *, speech_config, audio_config):
        module = self

        class _Synthesizer:
            bookmark_reached = module._BookmarkSignal()

            def speak_ssml_async(self, ssml):
                return module._ResultFuture(module._result)

        return _Synthesizer()


def test_live_boundary_reads_cancellation_details_from_the_result_property(monkeypatch):
    # Real production crash, confirmed against the installed SDK (1.51.2):
    # speechsdk.CancellationDetails.from_result(result) raised AttributeError
    # (that classmethod belongs to speech RECOGNITION results, not synthesis,
    # and doesn't exist on either class in this SDK version) -- masking
    # whatever the real cancellation reason was. Fixed to read
    # result.cancellation_details directly, matching Microsoft's own
    # synthesis quickstart samples.
    fake_details = _FakeSpeechSDKModule._CancellationDetails(
        reason="Error", error_code="AuthenticationFailure", error_details="401",
    )
    fake_sdk = _FakeSpeechSDKModule(
        _FakeSpeechSDKModule._Result(reason="Canceled", cancellation_details=fake_details)
    )
    monkeypatch.setattr(azure_tts_sdk, "_import_speech_sdk", lambda: fake_sdk)
    boundary = AzureSpeechSDKLiveBoundary(region="eastus", subscription_key="fake-key")
    response = boundary.synthesize_group("<speak/>", timeout_seconds=30)
    assert response.cancelled is True
    assert response.cancellation_reason == "Error"
    assert response.cancellation_error_code == "AuthenticationFailure"
    assert response.cancellation_details == "401"


def test_live_boundary_returns_audio_on_a_completed_synthesis(monkeypatch):
    fake_sdk = _FakeSpeechSDKModule(
        _FakeSpeechSDKModule._Result(reason="SynthesizingAudioCompleted", audio_data=b"pcm-bytes")
    )
    monkeypatch.setattr(azure_tts_sdk, "_import_speech_sdk", lambda: fake_sdk)
    boundary = AzureSpeechSDKLiveBoundary(region="eastus", subscription_key="fake-key")
    response = boundary.synthesize_group("<speak/>", timeout_seconds=30)
    assert response.cancelled is False
    assert response.audio_data == b"pcm-bytes"


def test_import_speech_sdk_raises_typed_error_when_not_installed(monkeypatch):
    # Force the ImportError regardless of whether the optional extra happens to be
    # installed in the environment running this test -- the behavior under test is
    # the typed-error wrapping, not this machine's package set.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "azure.cognitiveservices.speech" or name.startswith("azure.cognitiveservices.speech"):
            raise ImportError("no module named azure.cognitiveservices.speech")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(AzureTTSError) as excinfo:
        _import_speech_sdk()
    assert excinfo.value.category == "sdk_unavailable"
    assert excinfo.value.retryable is False


def test_write_group_slices_produces_per_island_wavs_with_correct_boundaries(tmp_path):
    boundary = FakeBoundary(
        responses=[_completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800})]
    )
    group_result = synthesize_group_with_bookmarks(
        boundary, _islands(), parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES,
        output_path=tmp_path / "g1.sdk-group.wav",
    )
    slices = write_group_slices(group_result, tmp_path)

    assert slices["g1__isl00"].start_ms == 0
    assert slices["g1__isl00"].end_ms == 1000
    assert slices["g1__isl01"].start_ms == 1000
    assert slices["g1__isl01"].end_ms == 1800
    assert slices["g1__isl02"].start_ms == 1800
    assert slices["g1__isl02"].end_ms == 3000
    for slice_result in slices.values():
        assert Path(slice_result.path).exists()
        assert Path(slice_result.manifest_path).exists()
        assert slice_result.idempotent_reuse is False


def test_write_group_slices_reuses_idempotently(tmp_path):
    boundary = FakeBoundary(
        responses=[_completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800})]
    )
    group_result = synthesize_group_with_bookmarks(
        boundary, _islands(), parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES,
        output_path=tmp_path / "g1.sdk-group.wav",
    )
    first = write_group_slices(group_result, tmp_path)
    second = write_group_slices(group_result, tmp_path)

    assert all(result.idempotent_reuse is False for result in first.values())
    assert all(result.idempotent_reuse is True for result in second.values())
    assert {k: (v.start_ms, v.end_ms) for k, v in first.items()} == {
        k: (v.start_ms, v.end_ms) for k, v in second.items()
    }


def test_write_group_slices_force_regenerates(tmp_path):
    boundary = FakeBoundary(
        responses=[_completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800})]
    )
    group_result = synthesize_group_with_bookmarks(
        boundary, _islands(), parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES,
        output_path=tmp_path / "g1.sdk-group.wav",
    )
    write_group_slices(group_result, tmp_path)
    forced = write_group_slices(group_result, tmp_path, force=True)
    assert all(result.idempotent_reuse is False for result in forced.values())


def test_write_group_slices_raises_when_partially_missing(tmp_path):
    boundary = FakeBoundary(
        responses=[_completed_response(bookmarks_ms={"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800})]
    )
    group_result = synthesize_group_with_bookmarks(
        boundary, _islands(), parent_group_id="g1", voice=VOICE, allowed_voices=ALLOWED_VOICES,
        output_path=tmp_path / "g1.sdk-group.wav",
    )
    write_group_slices(group_result, tmp_path)
    (tmp_path / "g1__isl01.sdk-slice.wav").unlink()

    with pytest.raises(AzureTTSIdempotencyError, match="incomplete"):
        write_group_slices(group_result, tmp_path)
