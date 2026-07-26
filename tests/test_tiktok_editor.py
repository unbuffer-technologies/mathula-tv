from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv import tiktok_editor


class FakeProvider:
    def __init__(self, selected_block_id: str = "block_0002"):
        self.selected_block_id = selected_block_id
        self.requests = []

    def complete_structured(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            data={
                "selected_block_id": self.selected_block_id,
                "hook_text": "I-EFF ayihlehli!",
                "rationale": "This is the earliest self-contained central claim.",
                "confidence": 0.91,
                "human_review_flags": [],
            },
            metadata=SimpleNamespace(to_dict=lambda: {"provider": "fake"}),
        )


def _blocks():
    return [
        {
            "block_id": "block_0001",
            "speaker_id": "SPEAKER_00",
            "start_ms": 2000,
            "source_text": "Welcome to the programme.",
            "translated_text": "Siyanamukela ohlelweni.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "SPEAKER_01",
            "start_ms": 19840,
            "source_text": "The EFF is not deterred.",
            "translated_text": "I-EFF ayihlehli.",
        },
    ]


def test_ai_selects_only_server_supplied_block_boundaries() -> None:
    result = tiktok_editor.select_tiktok_hook(
        provider=FakeProvider(),
        job_id="job1",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={},
        source_duration_seconds=120.0,
    )

    assert result["selected_block_id"] == "block_0002"
    assert result["cut_start_seconds"] == 19.84
    assert result["approved_speech_immutable"] is True


def test_ai_cannot_select_a_non_candidate_boundary() -> None:
    with pytest.raises(ValueError, match="non-candidate"):
        tiktok_editor.select_tiktok_hook(
            provider=FakeProvider("block_9999"),
            job_id="job1",
            target_language="zu-ZA",
            blocks=_blocks(),
            seo={},
            source_duration_seconds=120.0,
        )


def test_hook_edit_preserves_translation_and_keeps_remainder(
    tmp_path, monkeypatch
) -> None:
    master = tmp_path / "master.mp4"
    translation = tmp_path / "translation.json"
    output = tmp_path / "edited.mp4"
    manifest = tmp_path / "edit.json"
    hook = tmp_path / "hook.txt"
    master.write_bytes(b"master")
    translation.write_text('{"segments": [{"translated_text": "akuguquki"}]}')
    source_info = {
        "duration": 120.0,
        "streams": [
            {"codec_type": "video"},
            {"codec_type": "audio"},
        ],
    }
    rendered_info = {
        "duration": 100.16,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "profile": "Main",
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "25/1",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
            },
        ],
    }
    monkeypatch.setattr(
        tiktok_editor,
        "probe",
        lambda path: source_info if path == master else rendered_info,
    )
    monkeypatch.setattr(tiktok_editor, "_FONT_PATH", tmp_path / "font.ttf")
    tiktok_editor._FONT_PATH.write_bytes(b"font")
    commands = []

    def fake_runner(command, check=True, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"edited")
        return SimpleNamespace(returncode=0)

    before = translation.read_bytes()
    result = tiktok_editor.render_tiktok_hook_edit(
        master_video=master,
        output_path=output,
        manifest_path=manifest,
        hook_text_path=hook,
        selection={
            "selected_block_id": "block_0002",
            "cut_start_seconds": 19.84,
            "hook_text": "I-EFF ayihlehli!",
        },
        translation_path=translation,
        runner=fake_runner,
    )

    assert translation.read_bytes() == before
    assert result["all_content_after_boundary_preserved"] is True
    assert result["translation"]["immutable"] is True
    command = commands[0]
    assert command[command.index("-ss") + 1] == "19.840000"
    assert command.index("-ss") < command.index("-i")
    video_filter = command[command.index("-vf") + 1]
    assert "setpts=PTS-STARTPTS" in video_filter
    assert "drawtext=" in video_filter
    assert command[command.index("-af") + 1] == "asetpts=PTS-STARTPTS"
    assert result["render_version"] == tiktok_editor.TIKTOK_EDIT_RENDER_VERSION
    assert output.read_bytes() == b"edited"
