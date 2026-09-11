from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.native_dub import (
    _apply_qa_group_correction,
    native_dub_paths,
    run_native_qa_back_translation,
)


class _FakeProvider:
    def __init__(self, responses: list[tuple[str, dict]]):
        self._responses = list(responses)
        self.config = SimpleNamespace(max_output_tokens=8192, deployment="grok-test")
        self.provider = "foundry"
        self.calls: list[dict] = []

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        self.calls.append({"operation": operation, "payload": payload})
        assert self._responses, f"no scripted response left for operation={operation}"
        expected_operation, data = self._responses.pop(0)
        assert operation == expected_operation, f"expected {expected_operation}, got {operation}"
        return SimpleNamespace(data=data, model="grok-test", input_tokens=10, output_tokens=5, attempts=1)


def _write_pass2(job_root: Path, *, groups: list[dict], translations: list[dict], input_sha256: str = "abc123") -> None:
    paths = native_dub_paths(job_root)
    atomic_write_json(
        paths.pass2,
        {
            "input_sha256": input_sha256,
            "temporal_mask_mode": True,
            "temporal_mask_groups": groups,
            "translations": translations,
        },
    )


def test_flagged_sentence_is_repaired_and_reverified_clean(tmp_path):
    _write_pass2(
        tmp_path,
        groups=[
            {
                "group_id": "g1",
                "segment_ids": ["seg-1"],
                "source_text": "He is not here.",
                "spoken_text": "Ukhona lapha.",
            }
        ],
        translations=[{"segment_id": "seg-1", "spoken_text": "Ukhona lapha."}],
    )
    provider = _FakeProvider(
        [
            (
                "native_qa_back_translation_batch",
                {
                    "verdicts": [
                        {
                            "ref": "g1",
                            "back_translation": "He is here.",
                            "matches_original": False,
                            "discrepancy_summary": "dropped the negation",
                        }
                    ]
                },
            ),
            (
                "native_qa_repair_batch",
                {"repairs": [{"group_id": "g1", "corrected_zulu": "Akekho lapha."}]},
            ),
            (
                "native_qa_back_translation_batch",
                {
                    "verdicts": [
                        {
                            "ref": "g1",
                            "back_translation": "He is not here.",
                            "matches_original": True,
                            "discrepancy_summary": "faithful, no meaningful difference",
                        }
                    ]
                },
            ),
        ]
    )

    result = run_native_qa_back_translation(job_root=tmp_path, provider=provider)

    assert result["flagged"] == []
    assert result["unresolved_group_ids"] == []
    assert len(result["repairs"]) == 1
    assert result["repairs"][0]["corrected_zulu"] == "Akekho lapha."
    assert len(provider.calls) == 3

    pass2 = read_json(native_dub_paths(tmp_path).pass2)
    assert pass2["temporal_mask_groups"][0]["spoken_text"] == "Akekho lapha."
    assert pass2["translations"][0]["spoken_text"] == "Akekho lapha."


def test_sentence_still_flagged_after_round_cap_is_reported_not_dropped(tmp_path):
    _write_pass2(
        tmp_path,
        groups=[
            {
                "group_id": "g1",
                "segment_ids": ["seg-1"],
                "source_text": "He is not here.",
                "spoken_text": "Ukhona lapha.",
            }
        ],
        translations=[{"segment_id": "seg-1", "spoken_text": "Ukhona lapha."}],
    )
    verdict_drift = {
        "ref": "g1",
        "back_translation": "He is here.",
        "matches_original": False,
        "discrepancy_summary": "dropped the negation",
    }
    provider = _FakeProvider(
        [
            ("native_qa_back_translation_batch", {"verdicts": [verdict_drift]}),
            (
                "native_qa_repair_batch",
                {"repairs": [{"group_id": "g1", "corrected_zulu": "Usekhona lapha."}]},
            ),
            ("native_qa_back_translation_batch", {"verdicts": [verdict_drift]}),
        ]
    )

    result = run_native_qa_back_translation(job_root=tmp_path, provider=provider, max_repair_rounds=1)

    assert result["unresolved_group_ids"] == ["g1"]
    assert len(result["flagged"]) == 1
    assert result["flagged"][0]["ref"] == "g1"
    # The attempted correction is still logged and applied, even though it didn't fix it --
    # never silently dropped, and the canonical text reflects the best attempt made.
    assert len(result["repairs"]) == 1
    pass2 = read_json(native_dub_paths(tmp_path).pass2)
    assert pass2["temporal_mask_groups"][0]["spoken_text"] == "Usekhona lapha."


def test_repair_disabled_leaves_flagged_untouched_and_skips_pass2_rewrite(tmp_path):
    paths = native_dub_paths(tmp_path)
    _write_pass2(
        tmp_path,
        groups=[
            {
                "group_id": "g1",
                "segment_ids": ["seg-1"],
                "source_text": "He is not here.",
                "spoken_text": "Ukhona lapha.",
            }
        ],
        translations=[{"segment_id": "seg-1", "spoken_text": "Ukhona lapha."}],
    )
    original_pass2_bytes = paths.pass2.read_bytes()
    provider = _FakeProvider(
        [
            (
                "native_qa_back_translation_batch",
                {
                    "verdicts": [
                        {
                            "ref": "g1",
                            "back_translation": "He is here.",
                            "matches_original": False,
                            "discrepancy_summary": "dropped the negation",
                        }
                    ]
                },
            ),
        ]
    )

    result = run_native_qa_back_translation(job_root=tmp_path, provider=provider, repair=False)

    assert len(result["flagged"]) == 1
    assert result["repairs"] == []
    assert len(provider.calls) == 1  # only the audit call, no repair call
    assert paths.pass2.read_bytes() == original_pass2_bytes


def test_apply_qa_group_correction_mutates_translations_and_temporal_group():
    pass2 = {
        "translations": [
            {"segment_id": "seg-1", "spoken_text": "Ukhona lapha."},
            {"segment_id": "seg-2", "spoken_text": "leftover"},
        ],
        "temporal_mask_groups": [
            {"group_id": "g1", "segment_ids": ["seg-1", "seg-2"], "spoken_text": "Ukhona lapha."},
        ],
    }
    group = pass2["temporal_mask_groups"][0]

    _apply_qa_group_correction(pass2, group, "Akekho lapha.")

    assert pass2["translations"][0]["spoken_text"] == "Akekho lapha."
    assert pass2["translations"][1]["spoken_text"] == ""
    assert pass2["temporal_mask_groups"][0]["spoken_text"] == "Akekho lapha."


def test_checkpoint_is_reused_without_any_provider_call(tmp_path):
    paths = native_dub_paths(tmp_path)
    _write_pass2(
        tmp_path,
        groups=[{"group_id": "g1", "segment_ids": ["seg-1"], "source_text": "Yes.", "spoken_text": "Yebo."}],
        translations=[{"segment_id": "seg-1", "spoken_text": "Yebo."}],
        input_sha256="matching-hash",
    )
    from mathula_tv.native_dub import QA_BACK_TRANSLATION_SCHEMA_VERSION, _hash_payload

    existing_hash = _hash_payload({
        "pass2_input_sha256": "matching-hash",
        "schema_version": QA_BACK_TRANSLATION_SCHEMA_VERSION,
        "repair_enabled": True,
        "max_repair_rounds": 1,
        "direct_adequacy_enabled": False,
    })
    atomic_write_json(
        paths.qa_back_translation,
        {"input_sha256": existing_hash, "flagged": [], "requires_review": False},
    )
    provider = _FakeProvider([])

    result = run_native_qa_back_translation(job_root=tmp_path, provider=provider, force=False)

    assert result["requires_review"] is False
    assert provider.calls == []
