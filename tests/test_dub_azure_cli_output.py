from __future__ import annotations

import json

from mathula_tv.cli import _print_dub_azure_result, parser


def _result() -> dict:
    return {
        "output": {
            "path": "/tmp/final_dubbed_job-123.mp4",
            "sha256": "abc123",
            "duration_seconds": 395.072,
            "frame_rate": "25/1",
            "video_codec": "h264",
            "video_profile": "Main",
            "pixel_format": "yuv420p",
            "audio_codec": "aac",
            "audio_sample_rate": 48000,
        },
        "title_panel": {
            "text": "UKhaas: uJohnson usakhuluma ngokuhlaselwa",
        },
        "translation": {
            "immutable": True,
        },
        "idempotent_reuse": False,
        "tiktok_edit": {"technical": "details"},
    }


def test_dub_azure_prints_readable_summary_by_default(capsys) -> None:
    _print_dub_azure_result(
        job_id="job-123",
        result=_result(),
        json_output=False,
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "Dubbing complete.\n"
        "Job: job-123\n"
        "Video: /tmp/final_dubbed_job-123.mp4\n"
        "Duration: 06:35\n"
        "Media: H.264 Main · 25 fps · AAC 48 kHz\n"
        "Title panel: UKhaas: uJohnson usakhuluma ngokuhlaselwa\n"
        "Translation: unchanged\n"
        "Publication render: created\n"
    )
    assert "sha256" not in captured.out
    assert "technical" not in captured.out


def test_dub_azure_json_output_preserves_complete_report(capsys) -> None:
    result = _result()

    _print_dub_azure_result(
        job_id="job-123",
        result=result,
        json_output=True,
    )

    assert json.loads(capsys.readouterr().out) == result


def test_dub_azure_parser_accepts_json_aliases() -> None:
    parsed = parser().parse_args(["dub-azure", "job-123"])
    assert parsed.json_output is False

    parsed = parser().parse_args(["dub-azure", "job-123", "--json"])
    assert parsed.json_output is True

    parsed = parser().parse_args(["dub-azure", "job-123", "--json-output"])
    assert parsed.json_output is True


def test_dub_azure_summary_handles_reused_and_missing_optional_fields(capsys) -> None:
    _print_dub_azure_result(
        job_id="job-456",
        result={
            "output": {"path": "/tmp/reused.mp4", "duration_seconds": 65},
            "idempotent_reuse": True,
        },
        json_output=False,
    )

    output = capsys.readouterr().out
    assert "Duration: 01:05" in output
    assert "Media: not reported" in output
    assert "Title panel: none" in output
    assert "Translation: not reported" in output
    assert "Publication render: reused" in output


def test_dub_azure_summary_reads_nested_publication_result(capsys) -> None:
    result = {
        "outputs": {"canonical_dubbed_master": "/tmp/clean-master.mp4"},
        "tiktok_edit": {
            "output": {
                "path": "/tmp/final-publication.mp4",
                "duration_seconds": 395.072,
                "frame_rate": "25/1",
                "video_codec": "h264",
                "video_profile": "Main",
                "audio_codec": "aac",
                "audio_sample_rate": 48000,
            },
            "title_panel": {"text": "Isihloko sephaneli"},
            "translation": {"immutable": True},
            "idempotent_reuse": True,
        },
    }

    _print_dub_azure_result(
        job_id="job-nested",
        result=result,
        json_output=False,
    )

    output = capsys.readouterr().out
    assert "Video: /tmp/final-publication.mp4" in output
    assert "Duration: 06:35" in output
    assert "Media: H.264 Main · 25 fps · AAC 48 kHz" in output
    assert "Title panel: Isihloko sephaneli" in output
    assert "Translation: unchanged" in output
    assert "Publication render: reused" in output
    assert "unknown" not in output


def test_dub_azure_summary_uses_readable_not_reported_fallback(capsys) -> None:
    _print_dub_azure_result(
        job_id="job-empty",
        result={},
        json_output=False,
    )

    output = capsys.readouterr().out
    assert "Video: not reported" in output
    assert "Duration: not reported" in output
    assert "Media: not reported" in output
    assert "unknown · unknown" not in output
