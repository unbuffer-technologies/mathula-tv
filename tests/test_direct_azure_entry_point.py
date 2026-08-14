from __future__ import annotations

from mathula_tv.cli import parser


def test_dub_azure_cli_exposes_direct_renderer_options_only() -> None:
    args = parser().parse_args(["dub-azure", "job-1"])
    assert args.min_confidence == 0.70
    assert args.voice_map is None
    assert args.refresh_voice_analysis is False
    assert not hasattr(args, "skip_to")
    assert not hasattr(args, "fallback_allowed")


def test_dub_azure_cli_exposes_explicit_voice_refresh_flag() -> None:
    args = parser().parse_args([
        "dub-azure",
        "job-1",
        "--force",
        "--refresh-voice-analysis",
    ])
    assert args.force is True
    assert args.refresh_voice_analysis is True


def test_dub_azure_exposes_no_cloud_timing_repair_option() -> None:
    args = parser().parse_args(["dub-azure", "job-1"])
    assert not hasattr(args, "timing_repair_backend")


def test_translation_override_commands_are_exposed() -> None:
    export_args = parser().parse_args(["export-translation-package", "job-1"])
    assert export_args.target_locale is None
    import_args = parser().parse_args(
        [
            "import-translation",
            "job-1",
            "translated.json",
            "--reviewed-by",
            "Reviewer",
        ]
    )
    assert import_args.reviewed_by == "Reviewer"
