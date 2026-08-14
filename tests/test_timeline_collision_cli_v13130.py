from mathula_tv.cli import parser


def test_timeline_collision_panel_parser() -> None:
    args = parser().parse_args(
        [
            "timeline-collision-panel",
            "job-1",
            "--resolve",
            "block_0055",
            "--strategy",
            "allow_overlap",
            "--overlap-before-ms",
            "200",
            "--overlap-after-ms",
            "300",
            "--reviewed-by",
            "Reviewer",
        ]
    )
    assert args.command == "timeline-collision-panel"
    assert args.resolve == "block_0055"
    assert args.overlap_after_ms == 300
