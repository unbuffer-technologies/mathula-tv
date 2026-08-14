from __future__ import annotations

import json
from types import SimpleNamespace

from mathula_tv.cli import _print_job_status, parser


class FakeJob:
    state = "azure_tts_queued"

    def to_dict(self):
        return {
            "job_id": "job-123",
            "state": self.state,
            "secret": "not-a-real-secret",
        }


def test_status_defaults_to_current_state_only(capsys):
    _print_job_status(FakeJob(), json_output=False)

    captured = capsys.readouterr()
    assert captured.out == "azure_tts_queued\n"
    assert captured.err == ""


def test_status_json_preserves_full_manifest_option(capsys):
    _print_job_status(FakeJob(), json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "job-123"
    assert payload["state"] == "azure_tts_queued"


def test_status_parser_accepts_json_aliases():
    parsed = parser().parse_args(["status", "job-123"])
    assert parsed.json_output is False

    parsed = parser().parse_args(["status", "job-123", "--json"])
    assert parsed.json_output is True

    parsed = parser().parse_args(["status", "job-123", "--json-output"])
    assert parsed.json_output is True
