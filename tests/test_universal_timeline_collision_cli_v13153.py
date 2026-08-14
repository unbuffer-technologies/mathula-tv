from mathula_tv.cli import parser


def test_universal_panel_no_longer_requires_job_id() -> None:
    args = parser().parse_args(["timeline-collision-panel", "--serve"])
    assert args.command == "timeline-collision-panel"
    assert args.legacy_job_id is None
    assert args.job_id is None
    assert args.serve is True


def test_legacy_job_id_remains_optional_for_terminal_actions() -> None:
    args = parser().parse_args(
        [
            "timeline-collision-panel",
            "job-1",
            "--resolve",
            "block_0005",
            "--strategy",
            "allow_overlap",
        ]
    )
    assert args.legacy_job_id == "job-1"
    assert args.resolve == "block_0005"


def test_dub_azure_exposes_universal_panel_port() -> None:
    args = parser().parse_args(
        [
            "dub-azure",
            "job-1",
            "--live-operation",
            "--timeline-panel-port",
            "9876",
        ]
    )
    assert args.timeline_panel_port == 9876
