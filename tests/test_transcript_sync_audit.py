from __future__ import annotations

import copy
import json
import math
import re
import wave
from array import array
from pathlib import Path

import pytest

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.dubbing_units import build_dubbing_units
from mathula_tv.transcript_sync_audit import (
    TranscriptSyncAuditFailure,
    run_transcript_sync_audit,
    validate_pre_translation,
    validate_translation_plan,
)


FIXTURE = Path(__file__).with_name("fixtures") / "transcript_en_raw_mighty_jamie.json"
REPO_ROOT = Path(__file__).resolve().parents[1]


_CORRECTIONS = (
    ("malanga ward", "Madlanga ward"),
    ("malanga supermarket", "Madlanga supermarket"),
    ("horse curry", "hoshkhari"),
    ("Tabo Pesta", "Thabo Bester"),
    ("Suleiman Heart Attack Karim", 'Suleiman "heart attack" Carrim'),
    ("Suleiman heart attack, Karim", 'Suleiman "heart attack" Carrim'),
    ("Woolwitz", "Woolies"),
    ("Malanga Commission", "Madlanga Commission"),
    ("advocate Charles Carlson", "Advocate Chaskalson"),
)


def _tokens(value: str) -> list[str]:
    return [token.casefold() for token in re.findall(r"[^\W_]+(?:['’][^\W_]+)*", value)]


def _semantic_match(text: str, phrase: str) -> tuple[int, int] | None:
    text_matches = list(re.finditer(r"[^\W_]+(?:['’][^\W_]+)*", text))
    phrase_tokens = _tokens(phrase)
    text_tokens = [match.group(0).casefold() for match in text_matches]
    for index in range(len(text_tokens) - len(phrase_tokens) + 1):
        if text_tokens[index : index + len(phrase_tokens)] == phrase_tokens:
            return text_matches[index].start(), text_matches[index + len(phrase_tokens) - 1].end()
    return None


def _corrected_transcript() -> dict:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    corrected = copy.deepcopy(raw)
    overlays = []
    for segment in corrected["segments"]:
        original = segment["source_text"]
        revised = original
        for source, replacement in _CORRECTIONS:
            revised = re.sub(re.escape(source), replacement, revised, flags=re.IGNORECASE)
        segment["source_text"] = revised

        raw_segment = next(item for item in raw["segments"] if item["segment_id"] == segment["segment_id"])
        segment_words = [
            word
            for word in raw["words"]
            if word["speaker"] == raw_segment["speaker"]
            and word["start"] >= raw_segment["start"] - 0.001
            and word["end"] <= raw_segment["end"] + 0.001
        ]
        word_tokens = [re.sub(r"(^[^\w]+|[^\w]+$)", "", word["text"].casefold()) for word in segment_words]
        for source, replacement in _CORRECTIONS:
            needle = _tokens(source)
            for index in range(len(word_tokens) - len(needle) + 1):
                if word_tokens[index : index + len(needle)] != needle:
                    continue
                overlays.append(
                    {
                        "item_id": segment["segment_id"],
                        "azure_text": " ".join(word["text"] for word in segment_words[index : index + len(needle)]),
                        "corrected_text": replacement,
                        "start_ms": round(segment_words[index]["start"] * 1000),
                        "end_ms": round(segment_words[index + len(needle) - 1]["end"] * 1000),
                        "status": "confirmed",
                        "raw_azure_artifacts_unchanged": True,
                    }
                )
    corrected["complete_text"] = " ".join(segment["source_text"] for segment in corrected["segments"])
    corrected["autocorrect"] = {
        "schema_version": "autocorrect-render-v1",
        "overlays": overlays,
        "active_correction_count": len(overlays),
    }
    return corrected


def _write_wav(path: Path, rate: int, samples: array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(samples.tobytes())


def _tone(rate: int, duration_ms: int, frequency: float = 220.0, amplitude: int = 4000) -> array:
    frames = max(1, round(duration_ms * rate / 1000))
    return array(
        "h",
        (
            round(amplitude * math.sin(2 * math.pi * frequency * index / rate))
            for index in range(frames)
        ),
    )


def _build_full_job(tmp_path: Path, *, omit_late_dialogue: bool = False) -> Path:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    corrected = _corrected_transcript()
    units_artifact = build_dubbing_units(corrected)
    job_root = tmp_path / "working/jobs/test-job"
    atomic_write_json(job_root / "analysis/transcript_en_raw.json", raw)
    atomic_write_json(job_root / "analysis/transcript_en.json", corrected)
    atomic_write_json(job_root / "dubbing/dubbing_units.json", units_artifact)

    plan_units = []
    for source in units_artifact["units"]:
        spoken = source["source_text"]
        tts = re.sub(r"\bBig\s+W\b", "Big double-you", spoken, flags=re.IGNORECASE)
        tts = re.sub(r"\bW\s+bag\b", "double-you bag", tts, flags=re.IGNORECASE)
        plan_units.append({**source, "faithful_translation": spoken, "spoken_text": spoken, "tts_text": tts})
    atomic_write_json(job_root / "dubbing/dubbing_plan.json", {"schema_version": "dubbing-plan-v1", "units": plan_units})

    rate = 8000
    total_ms = round(max(segment["end"] for segment in raw["segments"]) * 1000)
    dialogue = array("h", [0]) * round(total_ms * rate / 1000)
    alignments = []
    for index, unit in enumerate(plan_units):
        unit_id = unit["unit_id"]
        duration_ms = unit["preferred_duration_ms"]
        clip = _tone(rate, duration_ms, frequency=180 + index * 7)
        aligned_path = job_root / "dubbing/aligned/turns" / f"{unit_id}.wav"
        _write_wav(aligned_path, rate, clip)
        if not omit_late_dialogue or index == 0:
            offset = round(unit["start_ms"] * rate / 1000)
            for sample_index, value in enumerate(clip):
                destination = offset + sample_index
                if destination < len(dialogue):
                    dialogue[destination] = max(-32768, min(32767, dialogue[destination] + value))
        alignments.append(
            {
                "unit_id": unit_id,
                "speaker_id": unit["speaker_id"],
                "output_path": str(aligned_path),
                "source_duration_ms": duration_ms,
                "corrected_duration_ms": duration_ms,
                "final_duration_ms": duration_ms,
                "preferred_duration_ms": duration_ms,
                "maximum_duration_ms": unit["maximum_duration_ms"],
                "silence_padding_ms": 0,
            }
        )
    _write_wav(job_root / "audio/dub_dialogue.wav", rate, dialogue)
    atomic_write_json(job_root / "dubbing/aligned/manifest.json", {"schema_version": "alignment-manifest-v1", "units": alignments})
    atomic_write_json(
        job_root / "subtitles/spoken_zu.json",
        {
            "schema_version": "spoken-subtitles-v1",
            "entries": [
                {
                    "unit_id": unit["unit_id"],
                    "speaker_id": unit["speaker_id"],
                    "start_ms": unit["start_ms"],
                    "end_ms": unit["start_ms"] + alignment["final_duration_ms"],
                    "spoken_text": unit["spoken_text"],
                }
                for unit, alignment in zip(plan_units, alignments)
            ],
        },
    )
    return job_root


def test_mighty_jamie_full_transcript_and_timing_contract(tmp_path: Path) -> None:
    job_root = _build_full_job(tmp_path)
    preflight = validate_pre_translation(repo_root=REPO_ROOT, job_root=job_root)
    assert preflight["overall_passed"] is True
    assert preflight["metrics"] == {
        "raw_transcript_sha256": "0afed99afe4bc6ffa77095da444c7a1605b70058873aee14bf6825f802672b5f",
        "segment_count": 4,
        "word_count": 345,
        "unit_count": 24,
    }

    units = json.loads((job_root / "dubbing/dubbing_units.json").read_text())["units"]
    joined = " ".join(unit["source_text"] for unit in units)
    assert "Madlanga ward" in joined
    assert "Madlanga supermarket" in joined
    assert "hoshkhari" in joined
    assert "Thabo Bester" in joined
    assert 'Suleiman "heart attack" Carrim' in joined
    assert "non-descript" in joined
    assert "e-mail" in joined
    assert "horse curry" not in joined.casefold()
    assert "Woolwitz" not in joined

    report = run_transcript_sync_audit(
        repo_root=REPO_ROOT,
        job_root=job_root,
        require_final_media=False,
    )
    assert report["overall_passed"] is True
    assert report["error_count"] == 0
    assert report["warning_count"] == 2


def test_pre_translation_uses_rendered_overlays_not_invented_static_corrections(tmp_path: Path) -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    authoritative = copy.deepcopy(raw)
    first = authoritative["segments"][0]
    first["source_text"] = first["source_text"].replace("horse curry", "hoshkhari")
    authoritative["complete_text"] = " ".join(segment["source_text"] for segment in authoritative["segments"])
    authoritative["autocorrect"] = {
        "schema_version": "autocorrect-render-v1",
        "active_correction_count": 1,
        "overlays": [
            {
                "item_id": "seg-00001",
                "azure_text": "horse curry",
                "corrected_text": "hoshkhari",
                "start_ms": 10480,
                "end_ms": 11440,
                "status": "auto_applied",
                "raw_azure_artifacts_unchanged": True,
            }
        ],
    }
    units = build_dubbing_units(authoritative)
    job_root = tmp_path / "working/jobs/partial-corrections"
    atomic_write_json(job_root / "analysis/transcript_en_raw.json", raw)
    atomic_write_json(job_root / "analysis/transcript_en.json", authoritative)
    atomic_write_json(job_root / "dubbing/dubbing_units.json", units)

    report = validate_pre_translation(repo_root=REPO_ROOT, job_root=job_root)
    assert report["overall_passed"] is True
    assert not any(issue["code"] == "raw_misrecognition_survived" for issue in report["issues"])


def test_pre_translation_rejects_overlay_that_was_not_rendered(tmp_path: Path) -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    authoritative = copy.deepcopy(raw)
    authoritative["complete_text"] = " ".join(segment["source_text"] for segment in authoritative["segments"])
    authoritative["autocorrect"] = {
        "schema_version": "autocorrect-render-v1",
        "active_correction_count": 1,
        "overlays": [
            {
                "item_id": "seg-00001",
                "azure_text": "horse curry",
                "corrected_text": "hoshkhari",
                "start_ms": 10480,
                "end_ms": 11440,
                "status": "auto_applied",
                "raw_azure_artifacts_unchanged": True,
            }
        ],
    }
    units = build_dubbing_units(authoritative)
    job_root = tmp_path / "working/jobs/bad-overlay"
    atomic_write_json(job_root / "analysis/transcript_en_raw.json", raw)
    atomic_write_json(job_root / "analysis/transcript_en.json", authoritative)
    atomic_write_json(job_root / "dubbing/dubbing_units.json", units)

    with pytest.raises(TranscriptSyncAuditFailure) as error:
        validate_pre_translation(repo_root=REPO_ROOT, job_root=job_root)
    codes = {issue["code"] for issue in error.value.report["issues"]}
    assert "autocorrect_overlay_replacement_missing" in codes
    assert "applied_correction_source_survived" in codes
    assert "horse curry" in str(error.value)


def test_timeline_audit_rejects_dub_that_stops_after_first_unit(tmp_path: Path) -> None:
    job_root = _build_full_job(tmp_path, omit_late_dialogue=True)
    with pytest.raises(TranscriptSyncAuditFailure) as error:
        run_transcript_sync_audit(
            repo_root=REPO_ROOT,
            job_root=job_root,
            require_final_media=False,
        )
    missing = [
        issue
        for issue in error.value.report["issues"]
        if issue["code"] == "dialogue_bus_missing_unit_audio"
    ]
    assert len(missing) >= 20
    assert any(issue["details"]["unit_id"] == "unit_0002" for issue in missing)
    assert any(issue["details"]["unit_id"] == "unit_0024" for issue in missing)


def test_pre_tts_contract_rejects_translated_w_bag(tmp_path: Path) -> None:
    job_root = _build_full_job(tmp_path)
    plan_path = job_root / "dubbing/dubbing_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    target = next(unit for unit in plan["units"] if "W bag" in unit["source_text"])
    target["spoken_text"] = target["spoken_text"].replace("W bag", "isikhwama saseWoolworths")
    target["tts_text"] = target["tts_text"].replace("double-you bag", "isikhwama saseWoolworths")
    atomic_write_json(plan_path, plan)

    with pytest.raises(TranscriptSyncAuditFailure) as error:
        validate_translation_plan(repo_root=REPO_ROOT, job_root=job_root)
    codes = {issue["code"] for issue in error.value.report["issues"]}
    assert "protected_spoken_span_missing" in codes
    assert "protected_tts_span_missing" in codes
