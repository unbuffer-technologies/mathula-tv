from pathlib import Path
from types import SimpleNamespace

from mathula_tv import rendering


def test_rendering_stream_copies_video_and_uses_packet_hash_not_frame_rate_string(tmp_path, monkeypatch):
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
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
            }
        ],
    }
    audio_info = {"duration": 9.8, "streams": [{"codec_type": "audio"}]}
    rendered_info = {
        "duration": 10.0,
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                # Container-level representation may differ after remuxing.
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
    monkeypatch.setattr(rendering, "_video_packet_payload_hash", lambda path: "same-packet-hash")

    result = rendering.render_review_mp4(
        source,
        audio,
        output,
        manifest,
        runner=fake_runner,
    )

    assert commands
    command = commands[0]
    assert command[command.index("-c:v") + 1] == "copy"
    assert result["video_stream_copy"] is True
    assert result["source_video_packet_hash"] == result["output_video_packet_hash"]
    assert output.is_file()
