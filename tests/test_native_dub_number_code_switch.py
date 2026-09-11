from __future__ import annotations

from pathlib import Path
from typing import Any

from mathula_tv.azure_tts import AzureTTSRequest, AzureTTSResult
from mathula_tv.native_dub import _synthesize_group
from mathula_tv.tts_ssml import TextPart


class _FakeTTS:
    """Stand-in for AzureTTSBackend that just captures the built request."""

    configured_voices = ("zu-ZA-ThembaNeural",)
    sample_rate = 24_000
    channels = 1
    sample_width = 2

    def __init__(self) -> None:
        self.captured: AzureTTSRequest | None = None

    def synthesize(self, request: AzureTTSRequest, output_path: Path, *, force: bool) -> AzureTTSResult:
        self.captured = request
        return AzureTTSResult(
            backend="azure_tts",
            turn_id=request.turn_id,
            speaker_id=request.speaker_id,
            voice=str(request.voice),
            language=request.language,
            input_text_hash="deadbeef",
            ssml_hash="deadbeef",
            output_path=output_path,
            sample_rate=24_000,
            channels=1,
            sample_width=2,
            duration_ms=1000,
            preferred_duration_ms=request.preferred_duration_ms,
            maximum_duration_ms=request.maximum_duration_ms,
            rate_percent=0,
            attempt=1,
            sha256="deadbeef",
            ssml_path=output_path.with_suffix(".ssml"),
            manifest_path=output_path.with_suffix(".json"),
            idempotent_reuse=False,
            warnings=(),
            request_hash="deadbeef",
            ssml_summary="",
        )


def _group(text: str) -> dict[str, Any]:
    return {
        "group_id": "native_phrase_0044",
        "speaker_id": "SPEAKER_00",
        "parts": [{"type": "text", "text": text}],
    }


def test_synthesize_group_code_switches_a_bare_reference_number(tmp_path):
    # Confirmed real defect this exists to fix: "3978." synthesized at 4930ms
    # against a 1120ms source window because Azure's zu-ZA voice reads bare
    # digits as a full isiZulu cardinal-number expansion.
    tts = _FakeTTS()
    _synthesize_group(
        tts=tts,
        group=_group("3978. 3978."),
        voice_info={"selected_voice": "zu-ZA-ThembaNeural"},
        output_path=tmp_path / "native_phrase_0044.wav",
        preferred_ms=1000,
        maximum_ms=2000,
        force=False,
    )
    assert tts.captured is not None
    text_parts = [part for part in tts.captured.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    assert text_parts[0].text == "triiy nayn seven eyt. triiy nayn seven eyt."


def test_synthesize_group_leaves_ordinary_text_untouched(tmp_path):
    tts = _FakeTTS()
    _synthesize_group(
        tts=tts,
        group=_group("Sawubona Ayanda."),
        voice_info={"selected_voice": "zu-ZA-ThembaNeural"},
        output_path=tmp_path / "native_phrase_0001.wav",
        preferred_ms=1000,
        maximum_ms=2000,
        force=False,
    )
    assert tts.captured is not None
    text_parts = [part for part in tts.captured.parts if isinstance(part, TextPart)]
    assert text_parts[0].text == "Sawubona Ayanda."


def test_synthesize_group_request_text_keeps_the_original_committed_text(tmp_path):
    # request.text is a faithful record of what was actually committed by Pass 2,
    # kept separate from the TTS-only substitution applied to the SSML parts --
    # idempotency correctness relies on the SSML hash, not request.text, so this
    # is safe (see azure_tts.py's _request_hash, which hashes document.sha256).
    tts = _FakeTTS()
    _synthesize_group(
        tts=tts,
        group=_group("3978."),
        voice_info={"selected_voice": "zu-ZA-ThembaNeural"},
        output_path=tmp_path / "native_phrase_0044.wav",
        preferred_ms=1000,
        maximum_ms=2000,
        force=False,
    )
    assert tts.captured is not None
    assert tts.captured.text == "3978."
