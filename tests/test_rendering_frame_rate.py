from pathlib import Path
from types import SimpleNamespace

from mathula_tv import rendering


def test_rendering_defaults_to_smooth_cfr(tmp_path, monkeypatch):
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
    audio_info = {"duration": 9.8, "streams": [{"codec_type": "audio"}]}
    rendered_info = {
        "duration": 10.0,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "60000/2002",
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

    commands = []

    def fake_runner(command, check=True, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"rendered")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rendering, "probe", fake_probe)
    monkeypatch.delenv("MATHULA_TV_FINAL_VIDEO_MODE", raising=False)

    result = rendering.render_review_mp4(source, audio, output, manifest, runner=fake_runner)

    command = commands[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert "fps=fps=30000/1001:round=near" in command[command.index("-vf") + 1]
    assert result["video_stream_copy"] is False
    assert result["playback_stability"]["timestamps_regenerated"] is True
    assert output.is_file()

