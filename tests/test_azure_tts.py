from __future__ import annotations

import hashlib
import io
import json
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from mathula_tv.azure_tts import (
    AzureCancellationDetails,
    AzureTTSBackend,
    AzureTTSHTTPBoundary,
    AzureTTSError,
    AzureTTSIdempotencyError,
    AzureTTSRequest,
    AzureTTSResponse,
    AzureTTSResult,
    AzureTTSTransportError,
)
from mathula_tv.tts_ssml import (
    BreakPart,
    EmphasisPart,
    SSMLBounds,
    SSMLDocument,
    SSMLValidationError,
    SafeSSMLBuilder,
    SubstitutionPart,
    TextPart,
    validate_document,
    validate_ssml,
)
from mathula_tv.tts_timing import AzureTTSTimingSearch


VOICES = [
    {"ShortName": "zu-ZA-ThandoNeural", "Locale": "zu-ZA", "Gender": "Female"},
    {"ShortName": "zu-ZA-ThembaNeural", "Locale": "zu-ZA", "Gender": "Male"},
    {"ShortName": "en-ZA-LeahNeural", "Locale": "en-ZA", "Gender": "Female"},
]


class Boundary:
    def __init__(self, *, voices=(), audio=()):
        self.voice_responses = list(voices)
        self.audio_responses = list(audio)
        self.voice_calls: list[tuple[str, float]] = []
        self.audio_calls: list[tuple[str, str, float]] = []

    def get_voices(self, locale, *, timeout_seconds):
        self.voice_calls.append((locale, timeout_seconds))
        response = self.voice_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def synthesize_ssml(self, ssml, *, output_format, timeout_seconds):
        self.audio_calls.append((ssml, output_format, timeout_seconds))
        response = self.audio_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def json_response(value, status=200, headers=None):
    return AzureTTSResponse(status, json.dumps(value).encode(), headers or {})


def pcm_wav(*, tone_ms=500, leading_ms=0, trailing_ms=0, value=1200, rate=24_000):
    frames = (
        b"\x00\x00" * int(rate * leading_ms / 1000)
        + int(value).to_bytes(2, "little", signed=True) * int(rate * tone_ms / 1000)
        + b"\x00\x00" * int(rate * trailing_ms / 1000)
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        target.writeframes(frames)
    return output.getvalue()


def backend(boundary, *, sleeps=None, configured_voices=("zu-ZA-ThandoNeural",), max_retries=0):
    return AzureTTSBackend(
        boundary,
        configured_voices=configured_voices,
        default_voice=configured_voices[0],
        max_retries=max_retries,
        sleep=(sleeps if sleeps is not None else []).append,
    )


def request(**updates):
    values = {
        "turn_id": "unit_0001",
        "speaker_id": "SPEAKER_00",
        "text": "Sawubona Mzansi.",
        "preferred_duration_ms": 5700,
        "maximum_duration_ms": 6040,
    }
    values.update(updates)
    return AzureTTSRequest(**values)


class HTTPResponse:
    def __init__(self, status_code=200, content=b"ok", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class HTTPSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)


def test_http_boundary_uses_region_endpoints_required_headers_timeouts_and_redacts_echoed_key():
    secret = "private-subscription-key"
    session = HTTPSession(
        [
            HTTPResponse(403, f"bad {secret}".encode(), {"x-requestid": "safe"}),
            HTTPResponse(200, b"wav"),
        ]
    )
    boundary = AzureTTSHTTPBoundary("SouthAfricaNorth", secret, session=session)
    voices = boundary.get_voices("zu-ZA", timeout_seconds=120)
    audio = boundary.synthesize_ssml(
        "<speak />", output_format="riff-24khz-16bit-mono-pcm", timeout_seconds=120
    )
    assert secret not in repr(boundary) and secret.encode() not in voices.body
    assert session.calls[0][1] == (
        "https://southafricanorth.tts.speech.microsoft.com/cognitiveservices/voices/list"
    )
    assert session.calls[1][1].endswith("/cognitiveservices/v1")
    assert session.calls[0][2]["timeout"] == (15, 120)
    assert session.calls[1][2]["headers"]["X-Microsoft-OutputFormat"] == (
        "riff-24khz-16bit-mono-pcm"
    )
    assert audio.body == b"wav"


def test_safe_ssml_builder_escapes_untrusted_text_and_uses_only_allowlisted_elements():
    builder = SafeSSMLBuilder(
        allowed_voices=["zu-ZA-ThandoNeural"],
        verified_emphasis_voices=["zu-ZA-ThandoNeural"],
    )
    document = builder.build(
        [
            TextPart('Sawubona <voice name="evil"> & '),
            BreakPart(250),
            EmphasisPart("iNingizimu Afrika", "strong"),
            SubstitutionPart("Feroz Khan", 'Feh-rohz "Kaan"'),
        ],
        voice="zu-ZA-ThandoNeural",
        rate_percent=4,
        pitch_percent=-2,
        volume_percent=1,
    )
    assert "&lt;voice name=\"evil\"&gt;" in document.xml
    assert "&amp;" in document.xml and "&quot;Kaan&quot;" in document.xml
    assert document.sha256 == hashlib.sha256(document.xml.encode()).hexdigest()
    assert "breaks=1" in document.summary and "substitutions=1" in document.summary
    validate_document(
        document,
        bounds=builder.bounds,
        allowed_voices=builder.allowed_voices,
        verified_emphasis_voices=builder.verified_emphasis_voices,
    )


def test_ssml_rejects_bounds_unknown_xml_namespaces_external_audio_and_forged_documents():
    bounds = SSMLBounds()
    builder = SafeSSMLBuilder(allowed_voices=["zu-ZA-ThandoNeural"], bounds=bounds)
    with pytest.raises(SSMLValidationError, match="rate"):
        builder.build_plain_text("Sawubona", voice="zu-ZA-ThandoNeural", rate_percent=16)
    with pytest.raises(SSMLValidationError, match="pause"):
        builder.build([TextPart("A"), BreakPart(2001)], voice="zu-ZA-ThandoNeural")
    with pytest.raises(SSMLValidationError, match="not been verified"):
        builder.build([EmphasisPart("A")], voice="zu-ZA-ThandoNeural")
    malicious = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zu-ZA">'
        '<voice name="zu-ZA-ThandoNeural"><prosody rate="+0%" pitch="+0%" volume="+0dB">'
        '<audio src="https://attacker.invalid/a.wav" /></prosody></voice></speak>'
    )
    with pytest.raises(SSMLValidationError, match="unknown element|External URLs"):
        validate_ssml(malicious, bounds=bounds, allowed_voices=["zu-ZA-ThandoNeural"])
    unknown_namespace = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:untrusted="https://attacker.invalid/ns" xml:lang="zu-ZA">'
        '<voice name="zu-ZA-ThandoNeural"><prosody rate="+0%" pitch="+0%" volume="+0%">'
        "Sawubona</prosody></voice></speak>"
    )
    with pytest.raises(SSMLValidationError, match="namespace"):
        validate_ssml(unknown_namespace, bounds=bounds, allowed_voices=["zu-ZA-ThandoNeural"])
    valid = builder.build_plain_text("Sawubona", voice="zu-ZA-ThandoNeural")
    forged = SSMLDocument(**{**valid.__dict__, "_provenance": object()})
    with pytest.raises(SSMLValidationError, match="application-built"):
        validate_document(forged, bounds=bounds, allowed_voices=["zu-ZA-ThandoNeural"])
    with pytest.raises(SSMLValidationError, match="summary"):
        validate_document(
            replace(valid, summary="forged"),
            bounds=bounds,
            allowed_voices=["zu-ZA-ThandoNeural"],
        )


def test_voice_discovery_is_explicit_cached_and_validates_every_configured_voice():
    boundary = Boundary(voices=[json_response(VOICES)])
    service = backend(
        boundary,
        configured_voices=("zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"),
    )
    found = service.validate_configured_voices()
    assert [voice.name for voice in found] == ["zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"]
    assert service.validate_voice("zu-ZA-ThandoNeural").gender == "Female"
    assert len(boundary.voice_calls) == 1

    unavailable = backend(
        Boundary(voices=[json_response(VOICES[:1])]),
        configured_voices=("zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"),
    )
    with pytest.raises(AzureTTSError, match="unavailable") as error:
        unavailable.validate_configured_voices()
    assert error.value.category == "voice_unavailable" and not error.value.retryable


def test_synthesis_writes_canonical_atomic_wav_ssml_manifest_and_reuses_idempotently(tmp_path):
    boundary = Boundary(
        voices=[json_response(VOICES)],
        audio=[AzureTTSResponse(200, pcm_wav(tone_ms=500, leading_ms=100, trailing_ms=80))],
    )
    service = backend(boundary)
    output = tmp_path / "tts" / "unit_0001.wav"
    result = service.synthesize(request(rate_percent=4), output)

    assert result.backend == "azure_tts" and result.duration_ms == 500
    assert result.sample_rate == 24_000 and result.channels == 1 and result.sample_width == 2
    assert result.trimmed_leading_ms == 100 and result.trimmed_trailing_ms == 80
    assert result.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert Path(result.ssml_path).read_text().rstrip("\n") == boundary.audio_calls[0][0]
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["request_hash"] == result.request_hash
    assert manifest["result"]["input_text_hash"] == hashlib.sha256(b"Sawubona Mzansi.").hexdigest()
    assert boundary.audio_calls[0][1] == "riff-24khz-16bit-mono-pcm"
    assert not list(output.parent.glob(".*unit_0001*"))

    reused = service.synthesize(request(rate_percent=4), output)
    assert reused.idempotent_reuse and reused.sha256 == result.sha256
    assert len(boundary.audio_calls) == 1 and len(boundary.voice_calls) == 1


def test_idempotency_detects_tampering_and_force_regenerates(tmp_path):
    boundary = Boundary(
        voices=[json_response(VOICES)],
        audio=[
            AzureTTSResponse(200, pcm_wav(value=1000)),
            AzureTTSResponse(200, pcm_wav(value=2000)),
        ],
    )
    service = backend(boundary)
    output = tmp_path / "unit.wav"
    first = service.synthesize(request(), output)
    output.write_bytes(output.read_bytes() + b"tampered")
    with pytest.raises(AzureTTSIdempotencyError, match="SHA-256"):
        service.synthesize(request(), output)
    second = service.synthesize(request(), output, force=True)
    assert second.sha256 != first.sha256 and len(boundary.audio_calls) == 2


def test_failed_forced_regeneration_does_not_overwrite_valid_wav(tmp_path):
    boundary = Boundary(
        voices=[json_response(VOICES)],
        audio=[AzureTTSResponse(200, pcm_wav()), AzureTTSResponse(200, b"not a wav")],
    )
    service = backend(boundary)
    output = tmp_path / "unit.wav"
    valid = service.synthesize(request(), output)
    before = output.read_bytes()
    with pytest.raises(AzureTTSError, match="invalid WAV") as error:
        service.synthesize(request(), output, force=True)
    assert error.value.category == "invalid_output"
    assert output.read_bytes() == before and hashlib.sha256(before).hexdigest() == valid.sha256


def test_retry_classification_retry_after_attempt_and_safe_authentication_error(tmp_path):
    sleeps: list[float] = []
    boundary = Boundary(
        voices=[
            AzureTTSResponse(429, b'{"error":{"message":"slow"}}', {"Retry-After": "2.5"}),
            json_response(VOICES),
        ],
        audio=[
            AzureTTSResponse(503, b"temporary"),
            AzureTTSResponse(200, pcm_wav()),
        ],
    )
    service = backend(boundary, sleeps=sleeps, max_retries=2)
    result = service.synthesize(request(), tmp_path / "unit.wav")
    assert result.attempt == 2 and sleeps == [2.5, 1]

    secret = "not-a-real-subscription-key"
    denied = backend(Boundary(voices=[AzureTTSResponse(401, secret.encode())]), max_retries=3)
    with pytest.raises(AzureTTSError, match="authentication/authorization") as error:
        denied.discover_voices()
    assert secret not in str(error.value) and not error.value.retryable


def test_transport_and_cancellation_errors_are_bounded_and_structured():
    sleeps: list[float] = []
    boundary = Boundary(
        voices=[
            AzureTTSTransportError("Azure TTS timed out", category="timeout"),
            json_response(VOICES),
        ]
    )
    assert backend(boundary, sleeps=sleeps, max_retries=1).discover_voices()
    assert sleeps == [1]

    cancelled = Boundary(
        voices=[
            AzureTTSResponse(
                200,
                cancellation=AzureCancellationDetails(
                    "Canceled", "AuthenticationFailure", "Bearer abc.def.ghi"
                ),
            )
        ]
    )
    with pytest.raises(AzureTTSError, match="cancelled") as error:
        backend(cancelled, max_retries=3).discover_voices()
    assert not error.value.retryable
    assert error.value.cancellation_details["details"] == "Bearer [REDACTED]"


def timing_result(rate: int, duration_ms: int) -> AzureTTSResult:
    digest = hashlib.sha256(f"{rate}:{duration_ms}".encode()).hexdigest()
    return AzureTTSResult(
        backend="azure_tts",
        turn_id="unit_0001",
        speaker_id="SPEAKER_00",
        voice="zu-ZA-ThandoNeural",
        language="zu-ZA",
        input_text_hash="a" * 64,
        ssml_hash="b" * 64,
        output_path=f"candidate_{rate}.wav",
        sample_rate=24_000,
        channels=1,
        sample_width=2,
        duration_ms=duration_ms,
        preferred_duration_ms=5700,
        maximum_duration_ms=6040,
        rate_percent=rate,
        attempt=1,
        sha256=digest,
        warnings=(),
        request_hash="c" * 64,
        ssml_path=f"candidate_{rate}.ssml",
        manifest_path=f"candidate_{rate}.json",
        ssml_summary="safe",
    )


def test_timing_search_accepts_naturally_short_audio_without_slowing():
    rates: list[int] = []

    def synthesize(rate):
        rates.append(rate)
        return timing_result(rate, 4500)

    fitted = AzureTTSTimingSearch().fit(
        synthesize, preferred_duration_ms=5700, maximum_duration_ms=6040
    )
    assert fitted.accepted and fitted.accepted.rate_percent == 0
    assert rates == [0] and not fitted.requires_translation_repair


def test_timing_search_interpolates_within_bounds_and_selects_least_accelerated_fit():
    rates: list[int] = []

    def synthesize(rate):
        rates.append(rate)
        durations = {0: 6500, 4: 6200, 8: 5900}
        return timing_result(rate, durations[rate])

    fitted = AzureTTSTimingSearch(search_attempts=3).fit(
        synthesize, preferred_duration_ms=5700, maximum_duration_ms=6040
    )
    assert rates == [0, 8, 4]
    assert fitted.accepted and fitted.accepted.rate_percent == 8
    assert fitted.hard_window_fit and not fitted.requires_translation_repair


def test_timing_search_rejects_overlong_audio_and_emits_targeted_repair_signal():
    rates: list[int] = []

    def synthesize(rate):
        rates.append(rate)
        return timing_result(rate, 7200 - rate * 20)

    fitted = AzureTTSTimingSearch(search_attempts=3).fit(
        synthesize, preferred_duration_ms=5700, maximum_duration_ms=6040
    )
    assert fitted.accepted is None and fitted.requires_translation_repair
    assert len(rates) <= 3 and max(rates) <= 15 and min(rates) >= 0
    assert fitted.repair_signal is not None
    assert fitted.repair_signal.reason == "azure_tts_exceeds_hard_timing_window"
    assert fitted.repair_signal.attempted_rates_percent == tuple(rates)
    assert fitted.repair_signal.required_duration_reduction_ratio < 1
