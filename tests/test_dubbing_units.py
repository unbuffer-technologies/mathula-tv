import copy

import pytest

from mathula_tv.dubbing_units import DubbingUnitConfig, build_dubbing_units


def phrase(phrase_id, speaker, start_ms, end_ms, text, words, **updates):
    value = {
        "phrase_id": phrase_id,
        "speaker": speaker,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": text,
        "words": [
            {"word_id": word_id, "text": token, "start_ms": start, "end_ms": end}
            for word_id, token, start, end in words
        ],
    }
    value.update(updates)
    return value


def test_short_same_speaker_phrases_merge_with_word_and_phrase_provenance():
    transcript = {
        "schema_version": "azure-stt-normalized-v1",
        "duration_ms": 2_000,
        "phrases": [
            phrase("phrase_001", "SPEAKER_00", 100, 500, "Hello", [("word_001", "Hello", 100, 500)]),
            phrase("phrase_002", "SPEAKER_00", 550, 1_100, "world.", [("word_002", "world.", 550, 1_100)]),
        ],
    }
    artifact = build_dubbing_units(transcript)
    assert len(artifact["units"]) == 1
    unit = artifact["units"][0]
    assert unit["source_phrase_ids"] == ["phrase_001", "phrase_002"]
    assert unit["source_word_ids"] == ["word_001", "word_002"]
    assert unit["source_text"] == "Hello world."
    assert unit["start_ms"] == 100
    assert unit["preferred_end_ms"] == 1_100
    assert unit["hard_end_ms"] == 1_600
    assert unit["preceding_silence_ms"] == 100
    assert unit["following_silence_ms"] == 900
    assert unit["preferred_duration_ms"] == 1_000
    assert unit["maximum_duration_ms"] == 1_500


def test_speaker_change_and_long_pause_are_hard_boundaries():
    transcript = {
        "phrases": [
            phrase("p1", "A", 0, 400, "First", [("w1", "First", 0, 400)]),
            phrase("p2", "B", 450, 800, "Second", [("w2", "Second", 450, 800)]),
            phrase("p3", "B", 2_100, 2_500, "Third", [("w3", "Third", 2_100, 2_500)]),
        ]
    }
    units = build_dubbing_units(transcript)["units"]
    assert [unit["speaker_id"] for unit in units] == ["A", "B", "B"]
    assert [unit["source_phrase_ids"] for unit in units] == [["p1"], ["p2"], ["p3"]]


def test_overlap_group_boundaries_are_preserved_and_never_cross_merged():
    transcript = {
        "phrases": [
            phrase("p1", "A", 0, 800, "Before", [("w1", "Before", 0, 800)]),
            phrase(
                "p2",
                "A",
                810,
                1_300,
                "overlap one",
                [("w2", "overlap", 810, 1_000), ("w3", "one", 1_010, 1_300)],
                overlap=True,
                overlap_group_id="overlap_01",
                overlap_metadata={"other_speaker": "B"},
            ),
            phrase(
                "p3",
                "A",
                1_310,
                1_800,
                "overlap two",
                [("w4", "overlap", 1_310, 1_500), ("w5", "two", 1_510, 1_800)],
                overlap=True,
                overlap_group_id="overlap_02",
            ),
        ]
    }
    units = build_dubbing_units(transcript)["units"]
    assert len(units) == 3
    assert units[0]["has_overlap"] is False
    assert [unit["overlap_group_id"] for unit in units[1:]] == ["overlap_01", "overlap_02"]
    assert units[1]["overlap_metadata"] == [{"other_speaker": "B"}]


def test_long_phrase_splits_at_punctuation_or_word_boundaries_and_preserves_every_word_once():
    words = [
        ("w1", "One", 0, 500),
        ("w2", "two.", 510, 1_000),
        ("w3", "Three", 1_010, 1_500),
        ("w4", "four", 1_510, 2_000),
        ("w5", "five;", 2_010, 2_500),
        ("w6", "six", 2_510, 3_000),
        ("w7", "seven", 3_010, 3_500),
        ("w8", "eight.", 3_510, 4_000),
    ]
    transcript = {"phrases": [phrase("p1", "A", 0, 4_000, "One two. Three four five; six seven eight.", words)]}
    config = DubbingUnitConfig(minimum_duration_ms=700, maximum_duration_ms=1_600)
    units = build_dubbing_units(transcript, config)["units"]
    assert len(units) >= 3
    provenance = [word_id for unit in units for word_id in unit["source_word_ids"]]
    assert provenance == [f"w{index}" for index in range(1, 9)]
    assert len(provenance) == len(set(provenance))
    assert all(unit["preferred_duration_ms"] <= 1_600 for unit in units)
    assert units[0]["source_text"].endswith(".")


def test_manual_boundaries_and_timing_overrides_are_versioned_metadata():
    transcript = {
        "phrases": [
            phrase("p1", "A", 0, 400, "One", [("w1", "One", 0, 400)]),
            phrase("p2", "A", 410, 800, "Two", [("w2", "Two", 410, 800)]),
        ]
    }
    artifact = build_dubbing_units(
        transcript,
        phrase_overrides={"p1": {"force_boundary_after": True, "reviewer": "editor"}},
        unit_overrides={"unit_0001": {"preferred_end_ms": 450, "hard_end_ms": 500}},
    )
    assert len(artifact["units"]) == 2
    first = artifact["units"][0]
    assert first["preferred_end_ms"] == 450 and first["hard_end_ms"] == 500
    assert any(item.get("reviewer") == "editor" for item in first["manual_override_metadata"])
    assert artifact["schema_version"] == "dubbing-units-v1"
    assert len(artifact["artifact_sha256"]) == 64


def test_builder_is_deterministic_and_does_not_mutate_transcript():
    transcript = {
        "phrases": [
            phrase("p2", "A", 500, 900, "World.", [("w2", "World.", 500, 900)]),
            phrase("p1", "A", 0, 450, "Hello", [("w1", "Hello", 0, 450)]),
        ]
    }
    original = copy.deepcopy(transcript)
    first = build_dubbing_units(transcript)
    second = build_dubbing_units(transcript)
    assert first == second
    assert transcript == original
    assert first["units"][0]["source_phrase_ids"] == ["p1", "p2"]


def test_invalid_manual_timing_window_fails():
    transcript = {"phrases": [phrase("p1", "A", 100, 500, "One", [("w1", "One", 100, 500)])]}
    with pytest.raises(ValueError, match="Invalid manual timing"):
        build_dubbing_units(
            transcript,
            unit_overrides={"unit_0001": {"preferred_end_ms": 50, "hard_end_ms": 60}},
        )
