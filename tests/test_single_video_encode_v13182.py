from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mathula_tv import rendering, tiktok_editor


def _video_probe(duration: float = 10.0) -> dict:
    return {
        "duration": duration,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "profile": "High",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "25/1",
                "r_frame_rate": "25/1",
            },
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
        ],
    }


def test_clean_master_mode_override_uses_video_copy_even_when_env_requests_cfr(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "final_mix.wav"
    output = tmp_path / "clean_master.mp4"
    manifest = tmp_path / "render_manifest.json"
    source.write_bytes(b"source")
    audio.write_bytes(b"lossless-audio")
    source_info = _video_probe()
    audio_info = {"duration": 9.9, "streams": [{"codec_type": "audio"}]}

    def fake_probe(path: Path):
        if path == source:
            return source_info
        if path == audio:
            return audio_info
        return _video_probe()

    commands: list[list[str]] = []

    def fake_runner(command, check=True, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"remuxed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rendering, "probe", fake_probe)
    monkeypatch.setattr(rendering, "_video_packet_payload_hash", lambda _path: "same")
    monkeypatch.setenv("MATHULA_TV_FINAL_VIDEO_MODE", "smooth_cfr")

    result = rendering.render_review_mp4(
        source,
        audio,
        output,
        manifest,
        runner=fake_runner,
        video_mode="stream_copy",
        progress_stage="video_remux",
    )

    command = commands[0]
    assert command[command.index("-c:v") + 1] == "copy"
    assert "libx264" not in command
    assert result["render_mode"] == "stream_copy"
    assert result["video_stream_copy"] is True


def test_publication_encode_reads_lossless_final_mix_directly(
    tmp_path: Path, monkeypatch
) -> None:
    master = tmp_path / "clean_master.mp4"
    final_mix = tmp_path / "final_mix.wav"
    output = tmp_path / "publication.mp4"
    manifest = tmp_path / "manifest.json"
    hook = tmp_path / "hook.txt"
    translation = tmp_path / "translation.json"
    master.write_bytes(b"video-copy-master")
    final_mix.write_bytes(b"lossless-final-mix")
    translation.write_text('{"target_language":"zu-ZA"}', encoding="utf-8")

    monkeypatch.setattr(tiktok_editor, "probe", lambda _path: _video_probe())
    monkeypatch.setattr(tiktok_editor, "_FONT_PATH", tmp_path / "font.ttf")
    tiktok_editor._FONT_PATH.write_bytes(b"font")

    def fake_title_panel(*, output_path, title, width, height):
        output_path.write_bytes(b"panel")
        return {
            "rendered_text": title,
            "line_count": 1,
            "font_size": 48,
            "minimum_font_size": 35,
            "maximum_font_size": 84,
            "fit_action": "unchanged",
            "truncated": False,
            "panel_bounds": [88, 852, 1832, 1068],
        }

    monkeypatch.setattr(tiktok_editor, "_render_title_panel", fake_title_panel)
    commands: list[list[str]] = []

    def fake_runner(command, check=True, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"publication")
        return SimpleNamespace(returncode=0)

    result = tiktok_editor.render_tiktok_hook_edit(
        master_video=master,
        publication_audio=final_mix,
        output_path=output,
        manifest_path=manifest,
        hook_text_path=hook,
        selection={
            "selected_block_id": "block_0001",
            "cut_start_seconds": 0.0,
            "hook_text": "Isihloko",
        },
        translation_path=translation,
        title_text="Isihloko",
        runner=fake_runner,
    )

    command = commands[0]
    assert command[command.index("-filter_complex") + 1].find("[2:a]atrim") >= 0
    assert command[command.index("-filter_complex") + 1].count("apad") == 1
    assert str(final_mix) in command
    assert command[command.index("-c:v") + 1] == "libx264"
    assert result["publication_audio"]["source"] == "lossless_final_mix"
    assert result["publication_audio"]["reencoded_generations_before_publication"] == 0

