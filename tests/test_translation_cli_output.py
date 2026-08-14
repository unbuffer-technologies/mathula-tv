from __future__ import annotations

import json

from mathula_tv.cli import _print_translation_result, parser


def test_translation_success_is_concise_by_default(capsys) -> None:
    result = {
        "schema_version": "mathula.translation.multivariant.v1",
        "units": [{"unit_id": "unit_0001", "variants": []}],
    }

    _print_translation_result(
        job_id="job-123",
        result=result,
        json_output=False,
    )

    captured = capsys.readouterr()
    assert captured.out == "Translation complete for job job-123.\n"
    assert "unit_0001" not in captured.out


def test_translation_json_output_is_explicit(capsys) -> None:
    result = {
        "schema_version": "mathula.translation.multivariant.v1",
        "units": [{"unit_id": "unit_0001"}],
    }

    _print_translation_result(
        job_id="job-123",
        result=result,
        json_output=True,
    )

    captured = capsys.readouterr()
    assert json.loads(captured.out) == result


def test_translate_parser_accepts_json_output() -> None:
    args = parser().parse_args(["translate", "job-123", "--json-output"])

    assert args.command == "translate"
    assert args.json_output is True
