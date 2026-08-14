from pathlib import Path

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.production_pipeline import (
    _clear_azure_timing_repair_markers,
    _timing_repair_generation_sha256,
    _timing_repair_stage,
)


def test_timing_repair_budget_is_scoped_to_full_translation_revision(tmp_path):
    path = tmp_path / "translation.json"
    atomic_write_json(path, {"units": [{"unit_id": "unit_0001", "tts_text": "old"}]})
    old_generation = _timing_repair_generation_sha256({"units": []}, path)

    atomic_write_json(path, {"units": [{"unit_id": "unit_0001", "tts_text": "new"}]})
    new_generation = _timing_repair_generation_sha256({"units": []}, path)

    assert old_generation != new_generation
    assert _timing_repair_stage("unit_0001", old_generation) != _timing_repair_stage(
        "unit_0001", new_generation
    )


def test_successful_azure_tts_clears_one_shot_timing_repair_markers(tmp_path):
    required = tmp_path / "translation_repair_required.json"
    applied = tmp_path / "translation_repair_applied.json"
    required.write_text("{}", encoding="utf-8")
    applied.write_text("{}", encoding="utf-8")

    _clear_azure_timing_repair_markers(required, applied)

    assert not required.exists()
    assert not applied.exists()
