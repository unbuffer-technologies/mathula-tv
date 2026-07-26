from __future__ import annotations

from mathula_tv.cli import parser


def test_dub_azure_cli_exposes_direct_renderer_options_only() -> None:
    args = parser().parse_args(["dub-azure", "job-1"])
    assert args.min_confidence == 0.70
    assert args.voice_map is None
    assert not hasattr(args, "skip_to")
    assert not hasattr(args, "fallback_allowed")
