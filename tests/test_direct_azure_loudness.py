from __future__ import annotations

import shutil
import wave
from types import SimpleNamespace

from mathula_tv.direct_azure_dub import (
    _extract_loudnorm_json,
    normalize_dialogue_loudness,
)


MEASURED = """
[Parsed_loudnorm_0 @ 0x1] {
    "input_i" : "-22.10",
    "input_tp" : "-4.50",
    "input_lra" : "4.20",
    "input_thresh" : "-32.40",
    "output_i" : "-16.02",
    "output_tp" : "-1.50",
    "output_lra" : "4.10",
    "output_thresh" : "-26.20",
    "normalization_type" : "linear",
    "target_offset" : "0.02"
}
"""


class FakeRunner:
    def __init__(self):
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command[-2:] == ["null", "-"]:
            return SimpleNamespace(stderr=MEASURED)
        source = command[command.index("-i") + 1]
        output = command[-1]
        shutil.copy2(source, output)
        return SimpleNamespace(stderr=MEASURED)


def test_extracts_loudnorm_json_from_ffmpeg_diagnostics():
    parsed = _extract_loudnorm_json("prefix\n" + MEASURED + "\nsuffix")
    assert parsed["input_i"] == "-22.10"
    assert parsed["target_offset"] == "0.02"


def test_normalizes_completed_dialogue_bus_with_two_pass_loudnorm(
    tmp_path,
    monkeypatch,
):
    dialogue = tmp_path / "dialogue.wav"
    with wave.open(str(dialogue), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x00\x00" * 24000)
    original = dialogue.read_bytes()
    runner = FakeRunner()

    manifest = normalize_dialogue_loudness(
        dialogue,
        target_lufs=-16.0,
        true_peak_dbfs=-1.5,
        loudness_range_lu=11.0,
        sample_rate=24000,
        runner=runner,
    )

    assert dialogue.read_bytes() == original
    assert len(runner.commands) == 2
    first_command = runner.commands[0][0]
    assert "print_format=json" in first_command[first_command.index("-af") + 1]
    second_command = runner.commands[1][0]
    applied_filter = second_command[second_command.index("-af") + 1]
    assert "measured_I=-22.10" in applied_filter
    assert "linear=true" in applied_filter
    assert "asetpts=N/SR/TB" in applied_filter
    assert "atrim=end_sample=24000" in applied_filter
    assert manifest["target"]["integrated_lufs"] == -16.0
    assert manifest["per_speaker_volume_percent"] == 0
    assert manifest["input_sample_count"] == 24000
    assert manifest["output_sample_count"] == 24000
    assert manifest["timeline_preserved"] is True
