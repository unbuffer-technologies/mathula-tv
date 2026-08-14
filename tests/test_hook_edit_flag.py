from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mathula_tv import tiktok_editor
from mathula_tv.cli import _print_dub_azure_result, parser


def test_dub_azure_parser_supports_no_hook_edit() -> None:
    parsed = parser().parse_args(["dub-azure", "job-123"])
    assert parsed.hook_edit is True

    parsed = parser().parse_args(["dub-azure", "job-123", "--no-hook-edit"])
    assert parsed.hook_edit is False

    parsed = parser().parse_args(["dub-azure", "job-123", "--hook-edit"])
    assert parsed.hook_edit is True


def test_readable_summary_reports_card_only_mode(capsys) -> None:
    _print_dub_azure_result(
        job_id="job-123",
        result={
            "output": {"path": "/tmp/final.mp4", "duration_seconds": 120},
            "title_panel": {"text": "Ikhadi le-hook"},
            "translation": {"immutable": True},
            "hook_edit_enabled": False,
            "hook_card_enabled": True,
            "idempotent_reuse": False,
        },
        json_output=False,
    )

    output = capsys.readouterr().out
    assert "Hook edit: disabled (hook card retained)" in output
    assert "Title panel: Ikhadi le-hook" in output


def test_render_without_hook_edit_preserves_full_video_and_keeps_card(
    tmp_path, monkeypatch
) -> None:
    master = tmp_path / "master.mp4"
    output = tmp_path / "final.mp4"
    manifest = tmp_path / "manifest.json"
    hook = tmp_path / "hook.txt"
    translation = tmp_path / "translation.json"
    master.write_bytes(b"master")
    translation.write_text('{"units": []}', encoding="utf-8")

    source_info = {
        "duration": 120.0,
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
            }
        ],
    }
    rendered_info = {
        "duration": 120.0,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "25/1",
                "profile": "Main",
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
        lambda path: source_info if Path(path) == master else rendered_info,
    )
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

    def fake_runner(command, check=True, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"rendered")
        return SimpleNamespace(returncode=0)

    result = tiktok_editor.render_tiktok_hook_edit(
        master_video=master,
        output_path=output,
        manifest_path=manifest,
        hook_text_path=hook,
        selection={
            "selected_block_id": "block_0002",
            "cut_start_seconds": 19.84,
            "hook_text": "Isihloko",
        },
        translation_path=translation,
        title_text="Isihloko sekhadi",
        apply_opening_cut=False,
        runner=fake_runner,
    )

    command = commands[0]
    assert "-ss" not in command
    assert "overlay=0:0" in command[command.index("-filter_complex") + 1]
    assert result["hook_edit_enabled"] is False
    assert result["hook_card_enabled"] is True
    assert result["selected_cut_start_seconds"] == 19.84
    assert result["applied_cut_start_seconds"] == 0.0
    assert result["all_source_content_preserved"] is True
    assert result["output"]["duration_seconds"] == 120.0
    assert result["title_panel"]["text"] == "Isihloko sekhadi"
    assert result["cut_policy"] == "opening_cut_disabled_full_master_with_hook_card"


def test_edit_job_passes_card_only_mode_to_renderer(tmp_path, monkeypatch) -> None:
    job_id = "job-card-only"
    root = tmp_path / "jobs" / job_id
    master = root / "direct_dub" / "dubbed_master.mp4"
    translation = root / "translation" / "transcript_zu.json"
    seo = root / "translation" / "tiktok_zu.json"
    report = root / "direct_dub" / "report.json"
    for path in (master, translation, seo, report):
        path.parent.mkdir(parents=True, exist_ok=True)
    master.write_bytes(b"master")
    translation.write_text('{"target_language":"zu-ZA"}', encoding="utf-8")
    seo.write_text('{"cover_hook":"Ikhadi"}', encoding="utf-8")
    report.write_text(
        json.dumps(
            {
                "blocks": [
                    {
                        "block_id": "block_0001",
                        "speaker_id": "speaker-1",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "source_text": "Question?",
                        "translated_text": "Umbuzo?",
                    }
                ],
                "outputs": {"canonical_dubbed_master": str(master)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(tiktok_editor, "probe", lambda path: {"duration": 60.0})
    monkeypatch.setattr(
        tiktok_editor,
        "select_tiktok_hook",
        lambda **kwargs: {
            "selected_block_id": "block_0001",
            "opening_anchor_block_id": "block_0001",
            "opening_payoff_block_id": "block_0001",
            "opening_relationship": "standalone",
            "question_anchor_guard": {"guard_applied": False},
            "selected_candidate": {"block_id": "block_0001"},
            "opening_payoff_candidate": {"block_id": "block_0001"},
            "selection_prompt_version": tiktok_editor.TIKTOK_HOOK_PROMPT_VERSION,
            "cut_start_seconds": 12.0,
            "hook_text": "Ikhadi",
        },
    )
    captured: dict = {}

    def fake_render(**kwargs):
        captured.update(kwargs)
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(b"publication")
        return {
            "output": {"path": str(kwargs["output_path"])},
            "hook_edit_enabled": kwargs["apply_opening_cut"],
            "hook_card_enabled": True,
        }

    monkeypatch.setattr(tiktok_editor, "render_tiktok_hook_edit", fake_render)
    monkeypatch.setattr(tiktok_editor, "_record_publication_artifacts", lambda **kwargs: None)

    job = SimpleNamespace(job_id=job_id, target_language="zu-ZA", media={})
    result = tiktok_editor.edit_tiktok_job(
        work_dir=tmp_path,
        job=job,
        provider=object(),
        apply_opening_cut=False,
    )

    assert captured["apply_opening_cut"] is False
    assert result["hook_edit_enabled"] is False
    assert result["hook_card_enabled"] is True
