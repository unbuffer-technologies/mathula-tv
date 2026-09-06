from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from mathula_tv import mixing, rendering


def test_source_frame_rate_is_preserved_for_smooth_cfr() -> None:
    assert rendering.select_playback_frame_rate(
        {"avg_frame_rate": "30000/1001", "r_frame_rate": "30000/1001"}
    ) == Fraction(30000, 1001)
    assert rendering.select_playback_frame_rate(
        {"avg_frame_rate": "25/1", "r_frame_rate": "25/1"}
    ) == Fraction(25, 1)
    assert rendering.frame_rates_match("60000/2002", Fraction(30000, 1001))


def test_final_master_regenerates_cfr_and_uses_high_quality_encoding(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "mix.wav"
    output = tmp_path / "review.mp4"
    manifest = tmp_path / "manifest.json"
    source.write_bytes(b"source")
    audio.write_bytes(b"audio")
    source_info = {
        "duration": 10.0,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
            }
        ],
    }
    audio_info = {"duration": 9.9, "streams": [{"codec_type": "audio"}]}
    rendered_info = {
        "duration": 10.0,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
            },
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
        ],
    }

    def fake_probe(path: Path):
        if path == source:
            return source_info
        if path == audio:
            return audio_info
        return rendered_info

    commands: list[list[str]] = []

    def fake_runner(command, check=True, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"rendered")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rendering, "probe", fake_probe)
    monkeypatch.delenv("MATHULA_TV_FINAL_VIDEO_MODE", raising=False)
    result = rendering.render_review_mp4(source, audio, output, manifest, runner=fake_runner)

    command = commands[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    # TikTok re-encodes every upload regardless, so the default favors a much
    # faster local encode over one optimized for standalone fidelity.
    assert command[command.index("-preset") + 1] == "veryfast"
    assert command[command.index("-crf") + 1] == "20"
    assert command[command.index("-b:a") + 1] == "256k"
    assert "fps=fps=30000/1001:round=near" in command[command.index("-vf") + 1]
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert command[command.index("-video_track_timescale") + 1] == "90000"
    assert result["render_mode"] == "smooth_cfr"
    assert result["playback_stability"]["timestamps_regenerated"] is True


def test_stream_copy_remains_an_explicit_emergency_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "mix.wav"
    output = tmp_path / "review.mp4"
    manifest = tmp_path / "manifest.json"
    source.write_bytes(b"source")
    audio.write_bytes(b"audio")
    info = {
        "duration": 2.0,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 640,
                "height": 360,
                "avg_frame_rate": "25/1",
            }
        ],
    }
    rendered = {
        "duration": 2.0,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 640,
                "height": 360,
                "avg_frame_rate": "25/1",
            },
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
        ],
    }

    def fake_probe(path: Path):
        if path == source:
            return info
        if path == audio:
            return {"duration": 2.0, "streams": [{"codec_type": "audio"}]}
        return rendered

    commands = []

    def fake_runner(command, check=True, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"rendered")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rendering, "probe", fake_probe)
    monkeypatch.setattr(rendering, "_video_packet_payload_hash", lambda path: "same")
    monkeypatch.setenv("MATHULA_TV_FINAL_VIDEO_MODE", "stream_copy")
    result = rendering.render_review_mp4(source, audio, output, manifest, runner=fake_runner)
    assert commands[0][commands[0].index("-c:v") + 1] == "copy"
    assert result["video_stream_copy"] is True


def test_dialogue_boundaries_use_short_equal_power_fades() -> None:
    command = mixing._speaker_track_command(
        [
            {
                "unit_id": "unit_0001",
                "speaker_id": "SPEAKER_01",
                "start_ms": 0,
                "duration_ms": 1000,
                "path": "/tmp/unit.wav",
            }
        ],
        Path("/tmp/out.wav"),
        2000,
        24000,
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "curve=qsin" in graph
    assert "d=0.0100" in graph
    assert "d=0.02" not in graph
    assert "aresample=24000:async=1:first_pts=0" in graph
    assert "acompressor=" in graph
