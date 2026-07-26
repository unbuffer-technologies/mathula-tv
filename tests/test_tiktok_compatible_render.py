from pathlib import Path
from types import SimpleNamespace

from mathula_tv import rendering


def test_tiktok_derivative_uses_standard_constant_frame_rate(
    tmp_path, monkeypatch
):
    source = tmp_path / "master.mp4"
    output = tmp_path / "tiktok.mp4"
    manifest = tmp_path / "tiktok.json"
    source.write_bytes(b"master")
    source_info = {
        "duration": 10.0,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "25339/1000",
            },
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
        ],
    }
    rendered_info = {
        "duration": 10.0,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "profile": "Main",
                "width": 1920,
                "height": 1080,
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "25/1",
            },
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
        ],
    }
    monkeypatch.setattr(
        rendering,
        "probe",
        lambda path: source_info if path == source else rendered_info,
    )
    commands = []

    def fake_runner(command, check=True, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"compatible")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = rendering.render_tiktok_compatible_mp4(
        source,
        output,
        manifest,
        runner=fake_runner,
    )

    command = commands[0]
    assert command[command.index("-vf") + 1] == "fps=25,format=yuv420p"
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert command[command.index("-profile:v") + 1] == "main"
    assert result["frame_rate_mode"] == "constant"
    assert result["master_preserved"] is True
    assert output.read_bytes() == b"compatible"
